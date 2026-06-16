#
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import os
import random
import weakref
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from transformers import PretrainedConfig

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..attention_backend import AttentionForwardArgs, AttentionInputType, AttentionMetadata
from ..attention_backend.interface import AttentionBackend, PositionalEmbeddingParams, RopeParams
from ..attention_backend.sparse.dsa import transform_local_topk_and_prepare_pool_view
from ..attention_backend.utils import create_attention
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.attention import _helix_post_process, fp8_block_scaling_bmm_out
from ..modules.linear import Linear, TensorParallelMode, WeightsLoadingConfig
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..peft.lora.layer import LoraLayer
from .modeling_deepseekv3_mla_pytorch import dsv3_mla_context_pytorch, dsv3_mla_decode_pytorch

_FUSED_MLA_MODE_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_MODE"
_FUSED_MLA_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DEBUG_OUTPUT_DIR"
_FUSED_MLA_DUMP_Q_B_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DUMP_Q_B"
_FUSED_MLA_DUMP_Q_B_DECODE_ITER_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DUMP_Q_B_DECODE_ITER"
_FUSED_MLA_DUMP_Q_B_LAYERS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DUMP_Q_B_LAYERS"
_FUSED_MLA_DUMP_Q_B_NUM_LAYERS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DUMP_Q_B_NUM_LAYERS"
_FUSED_MLA_DUMP_Q_B_LAYER_SEED_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DUMP_Q_B_LAYER_SEED"
_FUSED_MLA_DUMP_Q_B_RANKS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DUMP_Q_B_RANKS"
FUSED_MLA_MODE_BASELINE = "baseline"
FUSED_MLA_MODE_PYTORCH = "pytorch"
FUSED_MLA_MODE_WIP = "wip"


def get_fused_mla_mode() -> str:
    mode = os.environ.get(_FUSED_MLA_MODE_ENV, "").strip().lower()
    if not mode:
        return FUSED_MLA_MODE_BASELINE

    mode_aliases = {
        "0": FUSED_MLA_MODE_BASELINE,
        "false": FUSED_MLA_MODE_BASELINE,
        "off": FUSED_MLA_MODE_BASELINE,
        "original": FUSED_MLA_MODE_BASELINE,
        "baseline": FUSED_MLA_MODE_BASELINE,
        "mla": FUSED_MLA_MODE_BASELINE,
        "pytorch": FUSED_MLA_MODE_PYTORCH,
        "torch": FUSED_MLA_MODE_PYTORCH,
        "pytorch_mla": FUSED_MLA_MODE_PYTORCH,
        "1": FUSED_MLA_MODE_WIP,
        "true": FUSED_MLA_MODE_WIP,
        "on": FUSED_MLA_MODE_WIP,
        "wip": FUSED_MLA_MODE_WIP,
        "fused_mla": FUSED_MLA_MODE_WIP,
        "wip_fused_mla": FUSED_MLA_MODE_WIP,
    }
    if mode not in mode_aliases:
        allowed_modes = ", ".join(
            (FUSED_MLA_MODE_BASELINE, FUSED_MLA_MODE_PYTORCH, FUSED_MLA_MODE_WIP)
        )
        raise ValueError(
            f"Unsupported {_FUSED_MLA_MODE_ENV}={mode!r}; expected one of: {allowed_modes}"
        )
    return mode_aliases[mode]


def _is_env_enabled(env_name: str, default: bool = True) -> bool:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no"}


def _env_int(env_name: str, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None:
        return default
    return int(value.strip())


def _parse_int_set(value: str) -> set[int]:
    result: set[int] = set()
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.add(int(item))
    return result


def _selected_q_b_dump_layers(num_hidden_layers: int) -> set[int]:
    explicit_layers = os.environ.get(_FUSED_MLA_DUMP_Q_B_LAYERS_ENV)
    if explicit_layers:
        return _parse_int_set(explicit_layers)
    if num_hidden_layers <= 0:
        return set()

    num_layers = min(
        num_hidden_layers,
        max(0, _env_int(_FUSED_MLA_DUMP_Q_B_NUM_LAYERS_ENV, 10)),
    )
    seed = _env_int(_FUSED_MLA_DUMP_Q_B_LAYER_SEED_ENV, 6108841)
    return set(random.Random(seed).sample(range(num_hidden_layers), num_layers))


def _rank_is_selected_for_q_b_dump(rank: int) -> bool:
    explicit_ranks = os.environ.get(_FUSED_MLA_DUMP_Q_B_RANKS_ENV)
    return explicit_ranks is None or rank in _parse_int_set(explicit_ranks)


class DeepseekV3Linear(Linear):
    """
    A wrapper around Linear because we may optionally use min-latency kernels
    depending on input shapes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = None,
        mapping: Optional[Mapping] = None,
        tensor_parallel_mode: Optional[TensorParallelMode] = None,
        gather_output: bool = False,  # COLUMN parallel only
        quant_config: Optional[QuantConfig] = None,
        weights_loading_config: Optional[WeightsLoadingConfig] = None,
        reduce_output: bool = True,  # ROW parallel only
        skip_create_weights_in_init: bool = False,
        use_custom_cublas_mm: bool = False,
        use_cute_dsl_blockscaling_mm: bool = False,
        lora: Optional[LoraLayer] = None,
    ):
        super().__init__(
            in_features,
            out_features,
            bias,
            dtype,
            mapping,
            tensor_parallel_mode,
            gather_output,
            quant_config,
            weights_loading_config,
            reduce_output,
            skip_create_weights_in_init,
            use_custom_cublas_mm,
            lora,
            use_cute_dsl_blockscaling_mm=use_cute_dsl_blockscaling_mm,
        )

    def apply_linear(
        self,
        input,
        bias,
        lora_params: dict | None = None,
        layer_idx: int | None = None,
    ):
        num_tokens = input.shape[0]
        if not self.has_any_quant and 1 <= num_tokens <= 16 and get_sm_version() not in [120, 121]:
            output = torch.ops.trtllm.dsv3_fused_a_gemm_op(input, self.weight.t(), bias, None)
        else:
            output = super().apply_linear(input, bias, lora_params, layer_idx)
        return output


class FusedMLA(nn.Module):
    """
            Multi-head Latent Attention module.

                                      ┌──────────────────────────────┐
                                      │   x ∈ ℝ⁶¹⁴⁴   (1 token in)   │
                                      └───────────────┬──────────────┘
                                                      │
                                      x̃ = γ_in ⊙ x / √(⟨x²⟩ + ε)         RMSNorm
                                                      │
                                                      ▼
                                      ┌──────────────────────────────┐
                                      │   x̃ ∈ ℝ⁶¹⁴⁴                  │
                                      └───────────────┬──────────────┘
                                                      │
                    W_qkvia : ℝ⁶¹⁴⁴ ─► ℝ²⁶²⁴ "Combine q_a_proj and kv_a_proj: kv_a_proj_with_mqa, QKV down-projection"
                                                      │
                                                      ▼
                            ┌──────────────────────────┬─────────────────────┐
                            ▼                          ▼                     ▼
                     q_a ∈ ℝ²⁰⁴⁸                c_kv ∈ ℝ⁵¹²             k_pe ∈ ℝ⁶⁴
                            │                          │                     │
                      γ_q ⊙·/‖·‖                 γ_kv ⊙·/‖·‖             R_t (RoPE)
                            │                          │                     │
                            ▼                          ▼                     ▼
                      q̇_a ∈ ℝ²⁰⁴⁸                ċ_kv ∈ ℝ⁵¹²           k̂_pe ∈ ℝ⁶⁴
                            │                          │                     │
                W_qib:ℝ²⁰⁴⁸─►ℝ²⁰⁴⁸⁽¹⁶³⁸⁴⁾              └──────────┬──────────┘
               "q_b_proj, Q up-projection"                        │
                            │                                     ▼
                            ▼                         ┌─────────────────────────┐
                     q ∈ ℝ⁸⁽⁶⁴⁾ˣ²⁵⁶                   │   c_t = [ċ_kv ‖ k̂_pe]   │
                            │                         │   ∈ ℝ⁵⁷⁶                │
                    split [192 | 64]                  │                         │
                            │                         │   write to cache:       │
                    ┌───────┴────────┐                │   cache[seq, pos] := c_t│
                    ▼                ▼                └───────────┬─────────────┘
             q_nope ∈ ℝ⁸⁽⁶⁴⁾ˣ¹⁹² q_rope ∈ ℝ⁸⁽⁶⁴⁾ˣ⁶⁴               │
                    │                │                            │
                    │         per-h R_t (RoPE)                    │
                    │                │                            │
                    │                ▼                            │
                    │         q̂_rope ∈ ℝ⁸⁽⁶⁴⁾ˣ⁶⁴                  │
                    │                │                            │
        per head h ∈ {0..7(63)}:     │                            │
    q̃_h^lat = W_uk,h^T · q_nope_h    │                            │
        ∈ ℝ⁵¹²                       │                            │
        (absorption: k_b_proj,       │                            │
        K-up folded into Q)          │                            │
                    │                │                            │
                    └────────┬───────┘                            │
                             ▼                                    │
                      q̃ ∈ ℝ⁸⁽⁶⁴⁾ˣ⁵⁷⁶                              │
                      (per-h: [q̃^lat ‖ q̂_rope])                   │
                             │                                    │
                             └──────────────────┬─────────────────┘
                                                ▼
                               for each past token j ∈ {0..T_kv}:
                                 s_h(j) = ⟨ q̃_h , cache[seq, j] ⟩ / √256
                                                │
                                                ▼
                                     p_h(·) = softmax( s_h(·) )         ∈ ℝ^(T_kv+1)
                                                │
                                                ▼
                                   o_h^lat = Σ_j  p_h(j) · cache[seq, j, :512]   ∈ ℝ⁵¹²
                                                │            └──── V = first 512 of K
                                                │
                                                ▼
                                    per head h:   o_h = W_uv,h · o_h^lat  ∈ ℝ²⁵⁶
                                                          (v_b_proj, V-up)
                                                │
                                                ▼
                                           o ∈ ℝ⁸⁽⁶⁴⁾ˣ²⁵⁶
                                                │
                                             flatten
                                                │
                                                ▼
                                        o ∈ ℝ²⁰⁴⁸⁽¹⁶³⁸⁴⁾
                                                │
                                     W_o : ℝ²⁰⁴⁸⁽¹⁶³⁸⁴⁾ ─► ℝ⁶¹⁴⁴
                                                │
                                                ▼
                                             y ∈ ℝ⁶¹⁴⁴
                                                │
                                    Σ_rank  (AllReduce, TP=8)
                                                │
                                                ▼
                                 ┌──────────────────────────────┐
                                 │   y_out ∈ ℝ⁶¹⁴⁴  (1 token)   │
                                 └──────────────────────────────┘
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        predicted_tokens_per_seq: int,
        max_position_embeddings: int,
        bias: bool,
        aux_stream: Optional[torch.cuda.Stream] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        layer_idx: Optional[int] = None,
        dtype: torch.dtype = None,
        dense_bias: Optional[bool] = None,
        config: Optional[ModelConfig] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
        num_groups: int = 1,
        o_lora_rank: int = 1024,
    ):
        """
        Initialize the MLA module.

        Args:
            hidden_size (int): The size of the hidden dimension.
            num_attention_heads (int): The number of attention heads.
            num_key_value_heads (int): The number of key value heads.
            qk_nope_head_dim (int): The dimension of the query and key without Rope.
            qk_rope_head_dim (int): The dimension of the Rope of query and key.
            v_head_dim (int): The dimension of the value.
            q_lora_rank (int): The dimension of the compressed query.
            kv_lora_rank (int): The dimension of the compressed key and value.
            predicted_tokens_per_seq (int): The number of predicted tokens per sequence.
            max_position_embeddings (int): The maximum position embeddings.
            bias (bool): Whether to use bias in the linear layers.
            aux_stream (Optional[torch.cuda.Stream]): The auxiliary CUDA stream for running
                operations in two parallel streams.
            pos_embd_params (PositionalEmbeddingParams): The positional embedding parameters.
            layer_idx (int): The layer index.
            dtype (torch.dtype): The data type.
            dense_bias (bool): Whether to use bias in the output projection layer.
            config (ModelConfig): The model configuration.
            num_groups (int): The number of groups.
            o_lora_rank (int): The dimension of the compressed output.

        It loads following weights for inference (shapes are for TP=8 case):
        - indexer_k_norm_bias: shape=(128,), dtype=torch.float32
        - indexer_k_norm_weight: shape=(128,), dtype=torch.float32
        - indexer_softmax_scale: scalar, dtype=float
        - indexer_weight_scale_factor: scalar, dtype=float
        - indexer_weights_proj_weight: shape=(32, 6144), dtype=torch.float32
        - indexer_wk_weight: shape=(128, 6144), dtype=torch.float32
        - indexer_wq_b_weight: shape=(4096, 2048), dtype=torch.float8_e4m3fn
        - indexer_wq_b_weight_scale: shape=(32, 16), dtype=torch.float32
        - k_b_proj_trans: shape=(8, 512, 192), dtype=torch.bfloat16
            # made by [num_local_heads(8), kv_lora_rank(512), qk_nope_dim(192)]
            converted from kv_b_proj, which is dtype float8_e4m3fn on disk
        - kv_a_layernorm_variance_epsilon: scalar, dtype=float
        - kv_a_layernorm_weight: shape=(512,), dtype=torch.bfloat16
        - kv_a_proj_with_mqa_weight: shape=(2624, 6144), dtype=torch.float8_e4m3fn
        - kv_a_proj_with_mqa_weight_scale: shape=(21, 48), dtype=torch.float32
        - kv_b_proj_weight: shape=(3584, 512), dtype=torch.bfloat16
            # made by [num_local_heads(8) * qk_nope_dim(192) + num_local_heads*v_dim(256), kv_lora_rank(512)]
            # dequantized from dtype float8_e4m3fn on disk
        - o_proj_weight: shape=(6144, 2048), dtype=torch.float8_e4m3fn
            # 2048 is num_local_heads(8) * v_dim(256)
        - o_proj_weight_scale: shape=(48, 16), dtype=torch.float32
        - q_a_layernorm_variance_epsilon: scalar, dtype=float
        - q_a_layernorm_weight: shape=(2048,), dtype=torch.bfloat16
        - q_b_proj_weight: shape=(2048, 2048), dtype=torch.float8_e4m3fn
        - q_b_proj_weight_scale: shape=(2048, 4), dtype=torch.int32 after post-load UE8M0 packing
        - softmax_scale: scalar, dtype=float

        Input activation:
        - hidden_states: shape=(1, 6144), dtype=torch.bfloat16
        """
        super().__init__()
        print(f"FusedMLA::__init__: layer_idx={layer_idx}")
        self.layer_idx = layer_idx
        self.layer_idx_str = str(layer_idx)
        self.dtype = dtype

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        # this is 4 for our MTP=3 case
        self.predicted_tokens_per_seq = predicted_tokens_per_seq
        self.max_position_embeddings = max_position_embeddings
        self.pos_embd_params = pos_embd_params
        self.dense_bias = dense_bias
        self.num_groups = num_groups
        self.num_hidden_layers = getattr(config.pretrained_config, "num_hidden_layers", 0)
        self._q_b_proj_generation_call_count = 0
        self._q_b_proj_debug_dumped = False
        self.o_lora_rank = o_lora_rank
        if dense_bias is None:
            self.dense_bias = bias

        if self.q_lora_rank is None:
            self.q_lora_rank = hidden_size
            self.is_lite = True  # DeepSeek-V2-Lite-style
        else:
            self.is_lite = False

        assert pos_embd_params is not None, "pos_embd_params must be provided in MLA"

        self.register_to_config = False
        if config is not None:
            if "mla_layers" not in config.extra_attrs:
                config.extra_attrs["mla_layers"] = {}
            config.extra_attrs["mla_layers"][self.layer_idx_str] = weakref.ref(self)
            self.register_to_config = True

        # Currently only support DSA sparse attention
        self.is_dsa = True
        assert config is not None and config.sparse_attention_config is not None
        assert config.sparse_attention_config.algorithm == "dsa"

        # tensor parallel
        config = config or ModelConfig()
        if mapping_with_cp is not None:
            logger.warning_once(
                "[MLA::__init__] Overriding mapping with CP detected.",
                key="mla_init_mapping_with_cp",
            )
            self.mapping = mapping_with_cp
        else:
            self.mapping = config.mapping
        tp_size = self.mapping.tp_size
        pp_size = self.mapping.pp_size
        cp_size = self.mapping.cp_size
        dp_size = 1
        if self.mapping.enable_attention_dp:
            dp_size = tp_size
            tp_size = 1
        if self.mapping.has_cp_ulysses():
            raise NotImplementedError("MLA doesn't support CP Ulysses yet")
        if self.mapping.cp_size > 1:
            assert self.mapping.has_cp_helix(), (
                f"CP type must be HELIX for MLA, but got {self.mapping.cp_config['cp_type']}."
            )

        mapping = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size,
            pp_size=pp_size * dp_size,
            cp_size=cp_size,
            cp_config=self.mapping.cp_config,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )

        assert self.num_heads % (tp_size * cp_size) == 0
        self.num_heads_tp = self.num_heads // tp_size
        self.num_heads_tp_cp = self.num_heads_tp // cp_size
        self.num_key_value_heads_tp = (self.num_key_value_heads + tp_size - 1) // tp_size
        self.n_local_groups = self.num_groups // tp_size

        rms_norm_eps = getattr(config.pretrained_config, "rms_norm_eps", 1e-6)
        quant_config = config.get_quant_config()
        self.quant_config = quant_config

        self.use_cute_dsl_blockscaling_mm = config.use_cute_dsl_blockscaling_mm
        self.use_cute_dsl_blockscaling_bmm = config.use_cute_dsl_blockscaling_bmm
        self.use_cute_dsl_bf16_bmm = config.use_cute_dsl_bf16_bmm
        self.use_cute_dsl_bf16_gemm = config.use_cute_dsl_bf16_gemm

        if not self.is_lite:
            self.kv_a_proj_with_mqa = DeepseekV3Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            )

            self.q_a_layernorm = RMSNorm(
                hidden_size=self.q_lora_rank, eps=rms_norm_eps, dtype=dtype
            )

            self.q_b_proj = Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
                maintain_original_weight=True,
            )
        else:
            self.kv_a_proj_with_mqa = DeepseekV3Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            )

            self.q_b_proj = Linear(
                self.q_lora_rank,
                self.num_heads * self.qk_head_dim,
                bias=bias,
                dtype=dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                allreduce_strategy=config.allreduce_strategy,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
                maintain_original_weight=True,
            )

        kv_a_layernorm_hidden_size = kv_lora_rank
        self.kv_a_layernorm = RMSNorm(
            hidden_size=kv_a_layernorm_hidden_size, dtype=dtype, eps=rms_norm_eps
        )

        self.kv_b_proj = Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=bias,
            dtype=dtype,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
            use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
        )
        # This parameter will view into self.kv_b_proj.weight after loading weights.
        # For dummy weight initialization, this parameter is initialized with empty tensor.
        # Used in forward_absorption only
        self.v_b_proj = nn.Parameter(
            torch.empty(
                (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                dtype=dtype,
            ),
            requires_grad=False,
        )

        mapping_o = Mapping(
            world_size=pp_size * dp_size * tp_size * cp_size,
            tp_size=tp_size * cp_size,
            pp_size=pp_size * dp_size,
            cp_size=1,
            rank=self.mapping.rank,
            gpus_per_node=self.mapping.gpus_per_node,
            enable_attention_dp=self.mapping.enable_attention_dp,
        )
        self.mapping_o = mapping_o

        self.o_proj = Linear(
            self.num_key_value_heads * self.v_head_dim,
            self.hidden_size,
            bias=self.dense_bias,
            dtype=dtype,
            mapping=mapping_o,
            tensor_parallel_mode=TensorParallelMode.ROW,
            quant_config=quant_config,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            reduce_output=reduce_output,
            allreduce_strategy=config.allreduce_strategy,
            force_dynamic_quantization=config.force_dynamic_quantization,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
        )

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        mscale_all_dim = pos_embd_params.rope.mscale_all_dim
        scaling_factor = pos_embd_params.rope.scale
        mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
        q_scaling = 1.0 / (mscale * mscale)

        self.mqa = create_attention(
            config.attn_backend,
            self.layer_idx,
            self.num_heads_tp,
            head_dim=self.kv_lora_rank + self.qk_rope_head_dim,
            num_kv_heads=1,
            pos_embd_params=pos_embd_params,
            quant_config=quant_config,
            q_scaling=q_scaling,
            is_mla_enable=True,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.kv_lora_rank,
            hidden_size=self.hidden_size,
            predicted_tokens_per_seq=self.predicted_tokens_per_seq,
            skip_create_weights_in_init=config.skip_create_weights_in_init,
            sparse_attention_config=config.sparse_attention_config,
            dtype=dtype,
            aux_stream=aux_stream,
            rope_append=True,
        )

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.rotary_emb = None
        # self.apply_rotary_emb should be False in our case
        # when it is True in base MLA class, it will set
        # self.rotary_emb = RotaryEmbedding()
        self.apply_rotary_emb = False

        self.llama_4_scaling = False
        if hasattr(config.pretrained_config, "llama_4_scaling"):
            self.llama_4_scaling = True
            self.floor_scale = getattr(
                config.pretrained_config.llama_4_scaling, "original_max_position_embeddings", 8192
            )
            self.attn_scale = getattr(config.pretrained_config.llama_4_scaling, "beta", 0.1)

        if not config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self):
        # self.mqa has no weights but has states that are related to
        # quant_config, which could be modified after __init__.
        self.mqa.update_quant_config(self.quant_config)

        # Although we use FP8 MLA for context/generation phase, the output is still in BF16
        self.out_scale = None

        # k_b_proj_trans's dtype must be consistent with self.kv_b_proj,
        # which can be modified after __init__
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales()
        )

        mla_weight_dtype = torch.float8_e4m3fn if has_fp8_block_scales else self.dtype
        self.k_b_proj_trans = nn.Parameter(
            torch.empty(
                (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                dtype=mla_weight_dtype,
            ),
            requires_grad=False,
        )

        self.k_b_proj_trans_dequant = None
        self.v_b_proj_dequant = None
        self.o_a_proj_dequant = None
        if has_fp8_block_scales:
            self.k_b_proj_trans_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp,
                        self.kv_lora_rank // 128,
                        self.qk_nope_head_dim // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            # This parameter will view into self.kv_b_proj.weight_scale after loading weights.
            # For dummy weight initialization, this parameter is initialized with empty tensor.
            self.v_b_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.num_heads_tp_cp,
                        self.v_head_dim // 128,
                        self.kv_lora_rank // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if is_sm_100f() and not self.use_cute_dsl_blockscaling_bmm:
                assert self.dtype == torch.bfloat16
                self.k_b_proj_trans_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp, self.kv_lora_rank, self.qk_nope_head_dim),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
                self.v_b_proj_dequant = nn.Parameter(
                    torch.empty(
                        (self.num_heads_tp_cp, self.v_head_dim, self.kv_lora_rank),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        elif has_fp8_block_scales:
            self.o_a_proj_scale = nn.Parameter(
                torch.empty(
                    (
                        self.n_local_groups,
                        self.o_lora_rank // 128,
                        self.num_heads * self.qk_head_dim // self.num_groups // 128,
                    ),
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            if is_sm_100f():
                self.o_a_proj_dequant = nn.Parameter(
                    torch.empty(
                        (
                            self.n_local_groups,
                            self.o_lora_rank,
                            self.num_heads * self.qk_head_dim // self.num_groups,
                        ),
                        dtype=self.dtype,
                    ),
                    requires_grad=False,
                )
        else:
            self.k_b_proj_trans_scale = None
            self.v_b_proj_scale = None
            self.o_a_proj_scale = None

    def _is_fused_q_fp8_quant_enabled(self, num_generations: int = 0) -> bool:
        # Context-only batches: the fused path leaves a placeholder bf16 q_buf
        # that generation would read uninitialized, so mixed/gen batches must
        # take the legacy unfused path.
        # `TRTLLM_DISABLE_FUSED_Q_FP8_QUANT=1` is an opt-out kill switch for
        # debugging numerical drift between fused (skips bf16 round-trip) and
        # legacy (bf16 store -> reload -> FP8) Q-quant paths. Keep it disabled
        # by default until DSv4 accuracy recovers on the fused path.
        if os.environ.get("TRTLLM_DISABLE_FUSED_Q_FP8_QUANT", "1") == "1":
            return False
        return False

    def _attn_forward_gen(
        self,
        attn_backend: AttentionBackend,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        **kwargs,
    ):
        if self.mapping.has_cp_helix():
            # partial_o: [num_tokens, num_heads_tp * kv_lora_rank]
            # softmax_stats: [num_tokens, num_heads_tp, 2]
            softmax_stats = torch.empty(
                (q.shape[0], self.num_heads_tp, 2), device=q.device, dtype=torch.float32
            )
            kwargs["softmax_stats_tensor"] = softmax_stats
            partial_o = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            kv_lora_rank = partial_o.shape[-1] // self.num_heads_tp
            assert self.kv_lora_rank == kv_lora_rank

            return _helix_post_process(
                partial_o,
                softmax_stats,
                self.mapping,
                self.num_heads_tp_cp,
                kv_lora_rank,
                self.aux_stream,
                self.ln_events,
            )
        else:
            attn_output = attn_backend.forward(
                q,
                k,
                v,
                attn_metadata,
                forward_args=AttentionForwardArgs(**kwargs),
            )
            return attn_output

    def create_output(self, hidden_states: torch.Tensor, num_contexts: int):
        num_tokens = hidden_states.shape[0]
        attention_output_size = self.o_proj.in_features
        return hidden_states.new_empty(
            [num_tokens, attention_output_size], dtype=hidden_states.dtype
        )

    def _attention_scaling(self, q, position_ids):
        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def _bmm_bf16_out(self, a, b_no_transpose, b_transposed, output):
        """BMM with optional CuTe DSL bf16 acceleration on Blackwell."""
        if self.use_cute_dsl_bf16_bmm and is_sm_100f():
            torch.ops.trtllm.cute_dsl_bf16_bmm_blackwell(a, b_no_transpose, output)
        else:
            torch.ops.trtllm.bmm_out(a, b_transposed, output)

    def _save_q_b_debug_dump_tensor(
        self,
        output_dir: Path,
        rank: int,
        layer_idx: int,
        tensor_name: str,
        tensor: torch.Tensor,
    ) -> None:
        path = output_dir / f"r{rank}_l{layer_idx}_{tensor_name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor.detach().cpu(), path)

    def _save_q_b_debug_dump_value(
        self,
        output_dir: Path,
        rank: int,
        layer_idx: int,
        tensor_name: str,
        value: int | float,
    ) -> None:
        path = output_dir / f"r{rank}_l{layer_idx}_{tensor_name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(value, path)

    def _maybe_dump_q_b_projection(
        self,
        q_b_proj_input: torch.Tensor,
        generation_call_count: int,
    ) -> None:
        """Dump one live decode q_b projection case for WIP-kernel debugging."""
        if not _is_env_enabled(_FUSED_MLA_DUMP_Q_B_ENV, default=False):
            return
        if self._q_b_proj_debug_dumped:
            return
        debug_output_dir = os.environ.get(_FUSED_MLA_DEBUG_OUTPUT_DIR_ENV)
        if not debug_output_dir:
            return
        if torch.cuda.is_current_stream_capturing():
            return

        target_generation_call = _env_int(_FUSED_MLA_DUMP_Q_B_DECODE_ITER_ENV, 10)
        if generation_call_count != target_generation_call:
            return

        layer_idx = self.layer_idx
        if layer_idx is None or layer_idx not in _selected_q_b_dump_layers(self.num_hidden_layers):
            return
        rank = self.mapping.tp_rank
        if not _rank_is_selected_for_q_b_dump(rank):
            return

        # [num_tokens, num_heads_tp * qk_head_dim]
        q_b_proj_output = self.q_b_proj(q_b_proj_input)
        output_dir = Path(debug_output_dir).expanduser()
        self._save_q_b_debug_dump_tensor(
            output_dir, rank, layer_idx, "q_b_proj_input", q_b_proj_input
        )
        self._save_q_b_debug_dump_tensor(
            output_dir, rank, layer_idx, "q_b_proj_output", q_b_proj_output
        )
        self._save_q_b_debug_dump_tensor(
            output_dir, rank, layer_idx, "q_b_proj_weight", self.q_b_proj.weight
        )
        self._save_q_b_debug_dump_tensor(
            output_dir,
            rank,
            layer_idx,
            "q_b_proj_weight_scale",
            self.q_b_proj.weight_scale,
        )
        self._save_q_b_debug_dump_tensor(
            output_dir, rank, layer_idx, "k_b_proj_trans", self.k_b_proj_trans
        )
        self._save_q_b_debug_dump_value(
            output_dir, rank, layer_idx, "num_heads_tp", self.num_heads_tp
        )
        self._save_q_b_debug_dump_value(
            output_dir, rank, layer_idx, "q_lora_rank", self.q_lora_rank
        )
        self._save_q_b_debug_dump_value(
            output_dir, rank, layer_idx, "qk_head_dim", self.qk_head_dim
        )
        self._save_q_b_debug_dump_value(
            output_dir, rank, layer_idx, "qk_nope_head_dim", self.qk_nope_head_dim
        )
        self._save_q_b_debug_dump_value(
            output_dir, rank, layer_idx, "qk_rope_head_dim", self.qk_rope_head_dim
        )
        self._save_q_b_debug_dump_value(
            output_dir, rank, layer_idx, "kv_lora_rank", self.kv_lora_rank
        )
        self._q_b_proj_debug_dumped = True

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        # latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run fused MLA for a DeepSeekV3/GLM DSA attention layer.

        Performs token-wise Q/KV projections, sparse DSA indexer routing, context
        and generation attention dispatch, latent-to-value projection, and final
        output projection. `input_num_tokens` is `hidden_states.shape[0]`; it
        equals `attn_metadata.padded_num_tokens` when piecewise CUDA graph
        padding is active, otherwise it equals `attn_metadata.num_tokens`.
        Runtime-sized attention tensors are sliced using `attn_metadata.num_tokens`.
        The baseline context path may consume and clear stashed fused FP8-Q
        buffers, and the WIP generation path may emit q_b projection debug dumps
        when the debug environment is enabled.

        KV-cache interface used by the DSA/TRTLLM kernels:

        - `attn_metadata.kv_cache_manager.kv_cache_pool_pointers` is a CPU
          tensor containing raw primary/secondary KV-cache pool addresses. It
          is normally `torch.int64` with shape `[num_pools, 2]`. When the
          KV-cache format has separate block-scale pools, it is stacked as
          `[num_pools, 2, 2]` where the last dimension is data vs scale
          pointer. For the GLM-5 FP8 MLA path, the cache data pool stores the
          compressed latent KV vector of size `kv_lora_rank + qk_rope_head_dim`
          with `num_kv_heads = 1` and `CacheType.SELFKONLY`.

        - `attn_metadata.kv_cache_manager.kv_cache_pool_mapping` is a CPU
          `torch.int32` tensor with shape `[num_local_layers, 2]`. For each
          local layer it stores `[pool_index, layer_index_within_pool]`; the
          THOP paged-cache helper uses this row with `self.layer_idx` to pick
          the base pool pointer and layer offset for the current layer.

        - `attn_metadata.kv_cache_block_offsets` is a CUDA `torch.int32`
          tensor with shape
          `[num_attention_op_pools, max_num_sequences, 2, max_blocks_per_seq]`.
          `TrtllmAttentionMetadata.prepare()` fills it from the C++ KV cache
          manager for the active request IDs every forward step. For the
          batch-size-1, MTP=3 GLM-5 serving case, `max_num_sequences` is the
          engine capacity; the runtime slice actually consumed by kernels is
          determined by `attn_metadata.num_seqs` and the context/generation
          counts for the current step.

        Args
        - position_ids: Optional[torch.Tensor], token position IDs consumed by
          RoPE/indexer paths and sliced to runtime `num_tokens` when provided.
        - hidden_states: torch.Tensor, [input_num_tokens, hidden_size], input
          activations for this attention layer.
        - attn_metadata: AttentionMetadata, batch metadata containing runtime
          token counts, context/generation split, and KV-cache views.
        - all_reduce_params: Optional[AllReduceParams], tensor-parallel
          all-reduce parameters forwarded to `o_proj`.

        Returns
        - attn_output: torch.Tensor, [input_num_tokens, hidden_size], same dtype
          as `hidden_states`, final attention output after `o_proj`.
        """
        input_num_tokens = hidden_states.shape[0]

        # [input_num_tokens, num_heads_tp_cp * v_head_dim]
        attn_output = hidden_states.new_empty([input_num_tokens, self.o_proj.in_features])
        assert self.register_to_config
        assert self.mqa is not None, "DSA is only supported in MQA mode"
        assert self.k_b_proj_trans.dtype == torch.bfloat16
        assert self.num_heads_tp == 8
        assert self.num_heads_tp_cp == self.num_heads_tp
        assert self.kv_lora_rank == 512
        assert self.qk_rope_head_dim == 64
        assert hasattr(attn_metadata, "ctx_cached_token_indptr")
        fused_mla_mode = get_fused_mla_mode()
        assert (
            fused_mla_mode != FUSED_MLA_MODE_BASELINE
        )  # baseline impl is in modeling_deepseekv3.py

        # num of prefill requests
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens
        num_generation_tokens = num_tokens - num_ctx_tokens
        # num active requests
        num_seqs = attn_metadata.kv_lens_cuda_runtime.size(0)

        # we currently only work on a single request
        assert attn_metadata.num_contexts <= 1

        # The input may include padded rows for piecewise CUDA graph capture.
        # Batch-dependent attention only consumes runtime token rows, but
        # attn_output keeps input_num_tokens rows so downstream layers see the
        # same padded shape. Rows at [num_tokens:input_num_tokens] are ignored
        # by the final model-level slice.

        # [num_tokens, hidden_size]
        hidden_states = hidden_states[:num_tokens]
        if position_ids is not None:
            # [..., num_tokens]
            position_ids = position_ids[..., :num_tokens]

        # q_lor: [actual_num_tokens, q_lora_rank]
        # compressed_kv: [actual_num_tokens, kv_lora_rank]
        # k_pe: [actual_num_tokens, qk_rope_head_dim]
        q_lor, compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
        )

        # q_lor: [actual_num_tokens, q_lora_rank]
        # compressed_kv: [actual_num_tokens, kv_lora_rank]
        q_lor, compressed_kv = maybe_execute_in_parallel(
            lambda: self.q_a_layernorm(q_lor),
            lambda: self.kv_a_layernorm(compressed_kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        # [actual_num_tokens, kv_lora_rank + qk_rope_head_dim]
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

        # q_fp8: [actual_num_tokens, indexer.n_heads, indexer.head_dim]
        # k_fp8: [actual_num_tokens, indexer.head_dim]
        # k_scale: [actual_num_tokens, 1]
        # weights: [actual_num_tokens, indexer.n_heads]
        # q_scale: [actual_num_tokens, indexer.n_heads, 1]
        q_fp8, k_fp8, k_scale, weights, q_scale = self.mqa.indexer.pre_indexer_proj(
            q_lor, hidden_states, position_ids
        )
        # [actual_num_tokens, top_k], int32
        topk_indices = self.mqa.indexer.sparse_attn_indexer(
            attn_metadata,
            hidden_states,  # only used for shape[0]/device in buffer allocation
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            q_scale=q_scale,
        )

        # fused_q contains 1) q_nope @ k_b_proj with shape
        # [num_tokens, num_heads, kv_lora_rank], and 2) rope(q_pe) with shape
        # [num_tokens, num_heads, qk_rope_head_dim]. The second part is filled
        # by the attention-specific RoPE path.
        fused_q = torch.empty(
            [num_tokens, self.num_heads_tp, (self.kv_lora_rank + self.qk_rope_head_dim)],
            dtype=q_lor.dtype,
            device=q_lor.device,
        )

        context_slice = slice(0, num_ctx_tokens)
        generation_slice = slice(num_ctx_tokens, num_tokens)
        # [num_tokens, num_heads_tp * kv_lora_rank]
        attn_out_latent = hidden_states.new_empty(
            [num_tokens, self.num_heads_tp * self.kv_lora_rank]
        )
        assert hidden_states.dtype == q_lor.dtype

        self.mqa._ensure_rope_table_size(attn_metadata.max_seq_len)

        # [num_seqs + 1], int32
        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q_lor.device)
        # [num_seqs + 1], int32
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q_lor.device)
        # [1], uint32
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q_lor.device)
        has_fp8_kv_cache = (
            self.mqa.has_fp8_kv_cache if hasattr(self.mqa, "has_fp8_kv_cache") else False
        )
        assert has_fp8_kv_cache
        # [2], float32
        mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q_lor.device)
        # [1], float32
        mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q_lor.device)
        # [num_generation_tokens, num_heads_tp, kv_lora_rank + qk_rope_head_dim], uint8
        quant_q_buffer = torch.empty(
            num_generation_tokens,
            self.num_heads_tp,
            self.kv_lora_rank + self.qk_rope_head_dim,
            dtype=torch.uint8,
            device=q_lor.device,
        )

        def baseline_context() -> None:
            if num_ctx_tokens == 0:
                return
            # [num_ctx_tokens, num_heads_tp * (kv_lora_rank + qk_rope_head_dim)]
            context_fused_q = fused_q[context_slice].view(
                [
                    num_ctx_tokens,
                    self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                ]
            )
            # [num_ctx_tokens, num_heads_tp_cp * kv_lora_rank]
            attn_out_latent[:num_ctx_tokens] = self.mqa.forward(
                context_fused_q,
                None,
                None,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    out_scale=self.out_scale,
                    output=None,
                    latent_cache=latent_cache[context_slice],
                    q_pe=q_pe[context_slice],
                    quant_q_buffer=None,  # fused-FP8 path only
                    quant_scale_qkv=None,  # fused-FP8 path only
                    topk_indices=topk_indices[context_slice],
                    is_generation=False,  # used by DSA attention
                ),
            )

        def baseline_generation() -> None:
            if num_generation_tokens == 0:
                return
            # [num_generation_tokens, top_k], int32
            self.mqa.mla_rope_generation(
                fused_q[generation_slice],
                q_pe[generation_slice],
                latent_cache[generation_slice],
                attn_metadata,
                cu_q_seqlens,
                cu_kv_seqlens,
                fmha_scheduler_counter,
                mla_bmm1_scale,
                mla_bmm2_scale,
                quant_q_buffer,
            )
            # [num_generation_tokens, num_heads_tp_cp * kv_lora_rank]
            attn_out_latent[num_ctx_tokens:num_tokens] = self.mqa.forward(
                generation_fused_q.view(
                    [
                        num_generation_tokens,
                        self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim),
                    ]
                ),
                None,
                None,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.generation_only,
                    out_scale=self.out_scale,
                    output=None,
                    latent_cache=latent_cache[generation_slice],  # kvcache and k_pe
                    q_pe=q_pe[generation_slice],  # used by `invokeMLARopeGeneration`
                    topk_indices=topk_indices[generation_slice],  # used by DSA attention
                    is_generation=True,  # used by DSA attention
                    cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
                    cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
                    fmha_scheduler_counter=fmha_scheduler_counter,  # used by `mlaGeneration`
                    mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
                    mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
                    quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
                ),
            )

        # if num_tokens > self.predicted_tokens_per_seq:
        #     # too many tokens, use baseline attention
        #     # This could happen in a long prefill or warmup with long context or generation tokens
        #     baseline_context()
        #     baseline_generation()
        if fused_mla_mode == FUSED_MLA_MODE_WIP:
            q_b_proj_weight_for_kernel = self.q_b_proj.weight
            q_b_proj_weight_scale_for_kernel = self.q_b_proj.weight_scale
            assert q_b_proj_weight_for_kernel.dtype == torch.float8_e4m3fn
            assert q_b_proj_weight_scale_for_kernel.dtype in (torch.int32, torch.float32)

            # [num_tokens, num_heads_tp, qk_nope_head_dim]
            q_nope = q_lor.new_empty([num_tokens, self.num_heads_tp, self.qk_nope_head_dim])
            # [num_tokens, num_heads_tp, qk_rope_head_dim]
            q_pe = q_lor.new_empty([num_tokens, self.num_heads_tp, self.qk_rope_head_dim])

            if num_ctx_tokens > 0:
                # topk_indices_pool: [num_ctx_tokens, top_k], int32
                # kv_cache_pool: pooled KV cache view used by sparse attention.
                topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
                    topk_indices[context_slice],
                    attn_metadata,
                    layer_idx=self.mqa.get_local_layer_idx(attn_metadata),
                    is_generation=False,
                )
                layer_idx = self.mqa.get_local_layer_idx(attn_metadata)

                cached_context_tokens = int(getattr(attn_metadata, "num_ctx_cached_tokens", 0))
                sequence_length = getattr(attn_metadata, "kv_lens_cuda_runtime")

                chunk_size = self.predicted_tokens_per_seq
                for chunk_start in range(0, num_ctx_tokens, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, num_ctx_tokens)
                    chunk_len = chunk_end - chunk_start
                    chunk_slice = slice(chunk_start, chunk_end)
                    # Absolute local KV position for the first token in this
                    # chunk. The CUDA op writes chunk token `t` to
                    # `context_chunk_kv_start + t`, so later prefill chunks
                    # append after earlier chunks instead of reusing slot 0.
                    context_chunk_kv_start = cached_context_tokens + chunk_start
                    # [chunk_len, num_heads_tp, kv_lora_rank + qk_rope_head_dim]
                    fused_q_chunk = torch.empty(
                        [
                            chunk_len,
                            self.num_heads_tp,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                        ],
                        dtype=q_lor.dtype,
                        device=q_lor.device,
                    )
                    # [chunk_len, num_heads_tp, kv_lora_rank + qk_rope_head_dim], uint8
                    quant_q_buffer_chunk = torch.empty(
                        [
                            chunk_len,
                            self.num_heads_tp,
                            self.kv_lora_rank + self.qk_rope_head_dim,
                        ],
                        dtype=torch.uint8,
                        device=q_lor.device,
                    )
                    # [chunk_len, num_heads_tp * kv_lora_rank]
                    attn_out_latent[chunk_slice] = torch.ops.trtllm.dsv3_fused_mla_generation(
                        fused_q_chunk,
                        q_nope[chunk_slice],
                        self.k_b_proj_trans,
                        q_pe[chunk_slice],
                        latent_cache[chunk_slice],
                        self.mqa.rotary_cos_sin,
                        sequence_length,
                        attn_metadata.kv_cache_block_offsets,
                        attn_metadata.kv_cache_manager.kv_cache_pool_pointers,
                        attn_metadata.kv_cache_manager.kv_cache_pool_mapping,
                        topk_indices[chunk_slice],
                        topk_indices_pool[chunk_slice],
                        kv_cache_pool,
                        self.mqa.kv_scale_orig_quant,
                        self.mqa.kv_scale_quant_orig,
                        quant_q_buffer_chunk,
                        mla_bmm1_scale,
                        mla_bmm2_scale,
                        None,
                        layer_idx,
                        attn_metadata.kv_cache_manager.tokens_per_block,
                        int(self.mqa.quant_mode),
                        float(self.mqa.q_scaling),
                        True,
                        context_chunk_kv_start,
                        q_lor[chunk_slice],
                        q_b_proj_weight_for_kernel,
                        q_b_proj_weight_scale_for_kernel,
                    )

            if num_generation_tokens > 0:  # WIP
                self._q_b_proj_generation_call_count += 1
                self._maybe_dump_q_b_projection(
                    q_lor[generation_slice],
                    self._q_b_proj_generation_call_count,
                )

                # topk_indices_pool: [num_generation_tokens, top_k], int32
                # kv_cache_pool: pooled KV cache view used by sparse attention.
                topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
                    topk_indices[generation_slice],
                    attn_metadata,
                    layer_idx=self.mqa.get_local_layer_idx(attn_metadata),
                    is_generation=True,
                )
                # [max_num_requests + 1], int64 CPU
                host_gen_cached_token_indptr = getattr(
                    attn_metadata, "host_gen_cached_token_indptr", None
                )
                assert host_gen_cached_token_indptr is not None
                layer_idx = self.mqa.get_local_layer_idx(attn_metadata)

                num_generation_seqs = attn_metadata.num_generations
                assert num_generation_seqs > 0
                assert num_generation_tokens % num_generation_seqs == 0
                tokens_per_generation_seq = num_generation_tokens // num_generation_seqs
                chunk_size = min(self.predicted_tokens_per_seq, tokens_per_generation_seq)
                for generation_seq_idx in range(num_generation_seqs):
                    generation_global_seq_idx = attn_metadata.num_contexts + generation_seq_idx
                    generation_token_start = generation_seq_idx * tokens_per_generation_seq
                    cached_generation_tokens = int(
                        (
                            host_gen_cached_token_indptr[generation_seq_idx + 1]
                            - host_gen_cached_token_indptr[generation_seq_idx]
                        ).item()
                    )
                    # [1], int32
                    generation_sequence_length = attn_metadata.kv_lens_cuda_runtime[
                        generation_global_seq_idx : generation_global_seq_idx + 1
                    ]
                    # [num_attention_op_pools, 1, 2, max_blocks_per_seq], int32
                    generation_kv_cache_block_offsets = attn_metadata.kv_cache_block_offsets[
                        :, generation_global_seq_idx : generation_global_seq_idx + 1, :, :
                    ]
                    for chunk_start in range(0, tokens_per_generation_seq, chunk_size):
                        chunk_end = min(chunk_start + chunk_size, tokens_per_generation_seq)
                        chunk_len = chunk_end - chunk_start
                        generation_chunk_start = generation_token_start + chunk_start
                        generation_chunk_end = generation_token_start + chunk_end
                        generation_chunk_slice = slice(
                            num_ctx_tokens + generation_chunk_start,
                            num_ctx_tokens + generation_chunk_end,
                        )
                        generation_pool_chunk_slice = slice(
                            generation_chunk_start, generation_chunk_end
                        )
                        use_explicit_generation_start = chunk_len != tokens_per_generation_seq
                        # Full decode/MTP groups use the device-corrected KV
                        # length. Only partial chunks need an explicit start
                        # because `sequence_length - chunk_len` would point at
                        # the wrong position inside the group.
                        generation_chunk_kv_start = (
                            cached_generation_tokens + chunk_start
                            if use_explicit_generation_start
                            else 0
                        )
                        # [chunk_len, num_heads_tp, kv_lora_rank + qk_rope_head_dim]
                        fused_q_chunk = torch.empty(
                            [
                                chunk_len,
                                self.num_heads_tp,
                                self.kv_lora_rank + self.qk_rope_head_dim,
                            ],
                            dtype=q_lor.dtype,
                            device=q_lor.device,
                        )
                        # [chunk_len, num_heads_tp, kv_lora_rank + qk_rope_head_dim], uint8
                        quant_q_buffer_chunk = torch.empty(
                            [
                                chunk_len,
                                self.num_heads_tp,
                                self.kv_lora_rank + self.qk_rope_head_dim,
                            ],
                            dtype=torch.uint8,
                            device=q_lor.device,
                        )
                        # [chunk_len, num_heads_tp * kv_lora_rank]
                        attn_out_latent[generation_chunk_slice] = (
                            torch.ops.trtllm.dsv3_fused_mla_generation(
                                fused_q_chunk,
                                q_nope[generation_chunk_slice],
                                self.k_b_proj_trans,
                                q_pe[generation_chunk_slice],
                                latent_cache[generation_chunk_slice],
                                self.mqa.rotary_cos_sin,
                                generation_sequence_length,
                                generation_kv_cache_block_offsets,
                                attn_metadata.kv_cache_manager.kv_cache_pool_pointers,
                                attn_metadata.kv_cache_manager.kv_cache_pool_mapping,
                                topk_indices[generation_chunk_slice],
                                topk_indices_pool[generation_pool_chunk_slice],
                                kv_cache_pool,
                                self.mqa.kv_scale_orig_quant,
                                self.mqa.kv_scale_quant_orig,
                                quant_q_buffer_chunk,
                                mla_bmm1_scale,
                                mla_bmm2_scale,
                                None,
                                layer_idx,
                                attn_metadata.kv_cache_manager.tokens_per_block,
                                int(self.mqa.quant_mode),
                                float(self.mqa.q_scaling),
                                use_explicit_generation_start,
                                generation_chunk_kv_start,
                                q_lor[generation_chunk_slice],
                                q_b_proj_weight_for_kernel,
                                q_b_proj_weight_scale_for_kernel,
                            )
                        )
        else:  # pytorch path
            # q: [actual_num_tokens, num_heads_tp * qk_head_dim]
            q = self.q_b_proj(q_lor)

            # q_nope: [num_tokens, num_heads_tp, qk_nope_head_dim]
            # q_pe: [num_tokens, num_heads_tp, qk_rope_head_dim]
            q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )

            if self.k_b_proj_trans.dtype == torch.bfloat16:
                # [num_heads, num_tokens, self.qk_nope_head_dim]
                q_nope_t = q_nope.transpose(0, 1)
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                # [num_heads, num_tokens, self.qk_nope_head_dim]
                # x [num_heads, kv_lora_rank, qk_nope_head_dim]
                # -> [num_heads, num_tokens, kv_lora_rank]
                self._bmm_bf16_out(
                    q_nope_t,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans.transpose(1, 2),
                    q_nope_out,
                )
            elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
                # [num_heads, num_tokens, self.kv_lora_rank]
                q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

                if is_sm_100f() and not self.use_cute_dsl_blockscaling_bmm:
                    torch.bmm(
                        q_nope.transpose(0, 1),
                        self.k_b_proj_trans_dequant.transpose(1, 2),
                        out=q_nope_out,
                    )
                else:
                    fp8_block_scaling_bmm_out(
                        q_nope,
                        self.k_b_proj_trans,
                        self.k_b_proj_trans_scale,
                        q_nope_out,
                        self.k_b_proj_trans_dequant,
                        self.use_cute_dsl_blockscaling_bmm,
                    )
            else:
                raise NotImplementedError(
                    f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}."
                )

            if num_ctx_tokens > 0:  # pytorch context
                # topk_indices_pool: [num_ctx_tokens, top_k], int32
                # kv_cache_pool: pooled KV cache view used by sparse attention.
                # [num_ctx_tokens, top_k], int32
                context_topk_indices = topk_indices[context_slice]
                topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
                    context_topk_indices,
                    attn_metadata,
                    layer_idx=self.mqa.get_local_layer_idx(attn_metadata),
                    is_generation=False,
                )
                # [num_ctx_tokens, num_heads_tp, kv_lora_rank + qk_rope_head_dim]
                context_fused_q = fused_q[context_slice]
                # [num_ctx_tokens, num_heads_tp, qk_rope_head_dim]
                context_q_pe = q_pe[context_slice]
                # [num_ctx_tokens, kv_lora_rank + qk_rope_head_dim]
                context_latent_cache = latent_cache[context_slice]
                attn_out_latent[:num_ctx_tokens] = dsv3_mla_context_pytorch(
                    context_fused_q,
                    context_q_pe,
                    context_latent_cache,
                    context_topk_indices,
                    topk_indices_pool,
                    kv_cache_pool,
                    attn_metadata,
                    self.mqa,
                )
            if num_generation_tokens > 0:  # pytorch generation
                # [num_generation_tokens, num_heads_tp, kv_lora_rank + qk_rope_head_dim]
                generation_fused_q = fused_q[generation_slice]
                # [num_generation_tokens, num_heads_tp, qk_rope_head_dim]
                generation_q_pe = q_pe[generation_slice]
                # [num_generation_tokens, kv_lora_rank + qk_rope_head_dim]
                generation_latent_cache = latent_cache[generation_slice]
                # [num_generation_tokens, top_k], int32
                generation_topk_indices = topk_indices[generation_slice]
                self.mqa.mla_rope_generation(
                    generation_fused_q,
                    generation_q_pe,
                    generation_latent_cache,
                    attn_metadata,
                    cu_q_seqlens,
                    cu_kv_seqlens,
                    fmha_scheduler_counter,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    quant_q_buffer,
                )
                # topk_indices_pool: [num_generation_tokens, top_k], int32
                # kv_cache_pool: pooled KV cache view used by sparse attention.
                topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
                    generation_topk_indices,
                    attn_metadata,
                    layer_idx=self.mqa.get_local_layer_idx(attn_metadata),
                    is_generation=True,
                )
                # TRTLLM-Gen MLA generation ignores spec_decoding_packed_mask and relies on
                # causal current-group visibility, so keep the Python reference on the same
                # semantics.
                attn_out_latent[num_ctx_tokens:num_tokens] = dsv3_mla_decode_pytorch(
                    quant_q_buffer,
                    generation_topk_indices,
                    topk_indices_pool,
                    kv_cache_pool,
                    attn_metadata.kv_lens_cuda_runtime,
                    mla_bmm1_scale,
                    mla_bmm2_scale,
                    None,
                )

        # [num_tokens, num_heads_tp_cp, kv_lora_rank]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])
        # [num_tokens, num_heads_tp_cp, v_head_dim]
        attn_output_view = attn_output[:num_tokens].view(
            [-1, self.num_heads_tp_cp, self.v_head_dim]
        )
        # [num_heads_tp_cp, num_tokens, kv_lora_rank]
        attn_out_latent_t = attn_out_latent.transpose(0, 1)
        # [num_heads_tp_cp, num_tokens, v_head_dim]
        attn_output_t = attn_output_view.transpose(0, 1)

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads_tp_cp, kv_lora_rank, v_head_dim]
            v_b_proj_t = self.v_b_proj.transpose(1, 2)
            # [num_heads_tp_cp, num_tokens, kv_lora_rank]
            # x [num_heads_tp_cp, kv_lora_rank, v_head_dim]
            # -> [num_heads_tp_cp, num_tokens, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent_t,
                self.v_b_proj,
                v_b_proj_t,
                attn_output_t,
            )
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            if is_sm_100f() and not self.use_cute_dsl_blockscaling_bmm:
                # [num_heads_tp_cp, kv_lora_rank, v_head_dim]
                v_b_proj_dequant_t = self.v_b_proj_dequant.transpose(1, 2)
                torch.bmm(
                    attn_out_latent_t,
                    v_b_proj_dequant_t,
                    out=attn_output_t,
                )
            else:
                fp8_block_scaling_bmm_out(
                    attn_out_latent,
                    self.v_b_proj,
                    self.v_b_proj_scale,
                    attn_output_t,
                    self.v_b_proj_dequant,
                    self.use_cute_dsl_blockscaling_bmm,
                )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        # [input_num_tokens, hidden_size]
        attn_output = self.o_proj(
            attn_output,
            all_reduce_params=all_reduce_params,
            lora_params=None,
            layer_idx=self.layer_idx,
        )
        return attn_output

    def resmooth_parameters(self, module_weight, module_weight_scale, recipe=(1, 128, 128)):
        weight, weight_scale = fp8_utils.resmooth_to_fp8_e8m0(module_weight, module_weight_scale)

        transfromed_scale = fp8_utils.transform_sf_into_required_layout(
            weight_scale,
            mn=weight.shape[1],
            k=weight.shape[2],
            recipe=recipe,
            num_groups=weight.shape[0],
            is_sfa=False,
        )

        weight_param = torch.nn.Parameter(weight, requires_grad=False)
        scale_param = torch.nn.Parameter(transfromed_scale, requires_grad=False)

        return weight_param, scale_param

    def post_load_weights(self):
        has_fp8_block_scales = (
            self.kv_b_proj.quant_config
            and self.kv_b_proj.quant_config.quant_mode.has_fp8_block_scales()
        )
        is_sm120 = get_sm_version() == 120
        if is_sm120 and has_fp8_block_scales:
            self.k_b_proj_trans, self.k_b_proj_trans_scale = self.resmooth_parameters(
                self.k_b_proj_trans, self.k_b_proj_trans_scale, recipe=(1, 128, 128)
            )

            self.v_b_proj, self.v_b_proj_scale = self.resmooth_parameters(
                self.v_b_proj, self.v_b_proj_scale, recipe=(1, 128, 128)
            )


class DeepseekV3FusedMLA(FusedMLA):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ):
        mapping = mapping_with_cp if mapping_with_cp is not None else model_config.mapping
        if (
            model_config.mapping.enable_attention_dp
            or mapping.cp_size > 1
            or model_config.mapping.pp_size > 1
        ):
            raise RuntimeError(
                "TRTLLM_DEEPSEEKV3_FUSED_MLA_MODE=wip currently supports TP-only "
                "DeepSeekV3/DeepSeekV32 execution."
            )

        config = model_config.pretrained_config
        predicted_tokens_per_seq = (
            model_config.spec_config.tokens_per_gen_step
            if model_config.spec_config is not None
            else 1
        )
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            qk_rope_head_dim=config.qk_rope_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            q_lora_rank=config.q_lora_rank,
            kv_lora_rank=config.kv_lora_rank,
            v_head_dim=config.v_head_dim,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
            max_position_embeddings=config.max_position_embeddings,
            bias=False,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.yarn,
                rope=RopeParams.from_config(config),
                is_neox=False,
            ),
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            aux_stream=aux_stream,
            mapping_with_cp=mapping_with_cp,
            reduce_output=reduce_output,
        )


class DeepseekV32FusedMLA(DeepseekV3FusedMLA):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        aux_stream: Optional[torch.cuda.Stream] = None,
        mapping_with_cp: Optional[Mapping] = None,
        reduce_output: bool = True,
    ):
        super().__init__(
            model_config,
            layer_idx=layer_idx,
            aux_stream=aux_stream,
            mapping_with_cp=mapping_with_cp,
            reduce_output=reduce_output,
        )

        config = model_config.pretrained_config
        self.indexer = self.mqa.indexer
        self.kv_a_proj_with_mqa = DeepseekV3Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim + self.q_lora_rank,
            bias=False,
            dtype=config.torch_dtype,
            quant_config=model_config.get_quant_config(),
            skip_create_weights_in_init=model_config.skip_create_weights_in_init,
            use_custom_cublas_mm=True,
        )
