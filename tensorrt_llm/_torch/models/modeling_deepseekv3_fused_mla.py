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
import weakref
from typing import List, Optional

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
from .modeling_deepseekv3_mla_pytorch import dsv3_mla_context_pytorch

_FUSED_MLA_MODE_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_MODE"
_FUSED_MLA_CONTEXT_KERNEL_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_CONTEXT_KERNEL"
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


def _dsv3_fused_mla_generation_torch(
    quant_q_buffer: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_indices_pool: torch.Tensor,
    kv_cache_pool: torch.Tensor,
    sequence_length: torch.Tensor,
    mla_bmm1_scale: torch.Tensor,
    mla_bmm2_scale: torch.Tensor,
    spec_decoding_packed_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """GLM-5 BS=1 MTP=3 sparse MLA generation reference in PyTorch.

    Shapes are fixed by the target benchmark:
    - quant_q_buffer: uint8-backed FP8 E4M3 [4, 8, 576].
    - topk_indices: int32 [4, topK], local KV positions before pool conversion.
    - topk_indices_pool: int32 [4, topK], with negative padding entries.
    - kv_cache_pool: FP8 E4M3 [pool_tokens, 1, 576].
    - sequence_length: int32 [1], total KV length after appending the four decode rows.
    - spec_decoding_packed_mask: optional int32 [max_requests, 4, 1]. For the
      current four-row decode group, bit k says whether query row t may attend
      to current-group row k. Historical KV rows before the group are always
      valid if selected by topK.
    - return: BF16 [4, 8 * 512].
    """
    num_tokens, num_heads, head_dim = quant_q_buffer.shape
    assert num_tokens == 4
    assert num_heads == 8
    assert head_dim == 576
    assert kv_cache_pool.shape[1:] == (1, 576)

    current_group_start = sequence_length[0] - num_tokens
    current_group_offset = topk_indices - current_group_start
    historical_kv = topk_indices < current_group_start
    current_group_kv = (current_group_offset >= 0) & (current_group_offset < num_tokens)
    if spec_decoding_packed_mask is None:
        current_group_valid = (
            current_group_offset
            <= torch.arange(num_tokens, dtype=topk_indices.dtype, device=topk_indices.device)[
                :, None
            ]
        )
    else:
        packed_mask = spec_decoding_packed_mask[0, :num_tokens, 0].to(topk_indices.dtype)
        current_group_bit = torch.bitwise_right_shift(
            packed_mask[:, None], current_group_offset.clamp(0, 31)
        )
        current_group_valid = torch.bitwise_and(current_group_bit, 1).to(torch.bool)
    valid = (topk_indices_pool >= 0) & (historical_kv | (current_group_kv & current_group_valid))
    pool_indices = topk_indices_pool.clamp_min(0).to(torch.long)

    q = quant_q_buffer.view(torch.float8_e4m3fn)
    kv_fp8 = kv_cache_pool[:, 0, :][pool_indices]
    mm_scale = torch.ones((), dtype=torch.float32, device=quant_q_buffer.device)
    scores = torch.stack(
        [
            torch._scaled_mm(
                q[token_idx],
                kv_fp8[token_idx].transpose(0, 1).contiguous(),
                scale_a=mm_scale,
                scale_b=mm_scale,
                out_dtype=torch.float32,
            )
            for token_idx in range(num_tokens)
        ],
        dim=0,
    )
    scores = scores * mla_bmm1_scale[1]
    scores = scores.masked_fill(~valid[:, None, :], -float("inf"))

    max_scores = scores.max(dim=-1, keepdim=True).values
    weights = torch.exp2(scores - max_scores)
    weights = weights.masked_fill(~valid[:, None, :], 0.0)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)

    kv = kv_fp8.to(torch.float32)
    output = torch.einsum("thk,tkd->thd", weights, kv[..., :512])
    output = output * mla_bmm2_scale[0]
    return output.to(torch.bfloat16).reshape(num_tokens, num_heads * 512)


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
        - q_b_proj_weight_scale: shape=(16, 16), dtype=torch.float32
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

            self.q_proj = Linear(
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
            )
            self.q_b_proj = self.q_proj

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
        hidden_size = self.o_proj.in_features
        return hidden_states.new_empty([num_tokens, hidden_size], dtype=hidden_states.dtype)

    def _attention_scaling(self, q, position_ids):
        def _get_attn_scale(position_ids: torch.Tensor) -> torch.Tensor:
            positions = position_ids.view(-1)
            floor = torch.floor((positions + 1.0) / self.floor_scale)
            attn_scale = torch.log(floor + 1.0) * self.attn_scale + 1.0
            return attn_scale.unsqueeze(-1)

        attn_scale = _get_attn_scale(position_ids)
        q = (q * attn_scale).to(q.dtype)
        return q

    def forward_dsa_attn(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        latent_cache: torch.Tensor,
        indexer_intermediates: List[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """Batch-structure-dependent attention for DSA MLA (Op 2, not graph-captured).

        indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale].

        All num_tokens slicing happens here (not in Op 1) because
        num_tokens comes from batch-specific metadata and must not be
        baked into CUDA graph capture.
        """
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        # Slice Op 1 outputs to actual num_tokens (Op 1 operates on the
        # full padded tensor for CUDA graph compatibility).
        q = q[:num_tokens, ...]
        compressed_kv = compressed_kv[:num_tokens, ...]
        k_pe = k_pe[:num_tokens, ...]
        latent_cache = latent_cache[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        q_fp8, k_fp8, k_scale, weights, q_scale = indexer_intermediates
        # Slice indexer intermediates to actual num_tokens (they were
        # computed on the full padded tensor in Op 1).
        q_fp8 = q_fp8[:num_tokens, ...]
        k_fp8 = k_fp8[:num_tokens, ...]
        k_scale = k_scale[:num_tokens, ...]
        weights = weights[:num_tokens, ...]
        q_scale = q_scale[:num_tokens, ...]
        topk_indices = self.mqa.indexer.sparse_attn_indexer(
            attn_metadata,
            q,  # only used for shape/device in buffer allocation
            q_fp8,
            k_fp8,
            k_scale,
            weights,
            q_scale=q_scale,
        )

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]

            context_output = output[:num_ctx_tokens, :]
            topk_indices_ctx = (
                topk_indices[:num_ctx_tokens, :] if topk_indices is not None else None
            )
            self.forward_absorption_context(
                q_ctx,
                attn_metadata,
                context_output,
                position_ids=position_ids,
                latent_cache=latent_cache_ctx,
                topk_indices=topk_indices_ctx,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]

            generation_output = output[num_ctx_tokens:num_tokens, :]
            topk_indices_gen = topk_indices[num_ctx_tokens:num_tokens, :]
            self.forward_absorption_generation(
                q_gen,
                attn_metadata,
                generation_output,
                position_ids=position_ids,
                latent_cache=latent_cache_gen,
                topk_indices=topk_indices_gen,
            )

    def _bmm_bf16_out(self, a, b_no_transpose, b_transposed, output):
        """BMM with optional CuTe DSL bf16 acceleration on Blackwell."""
        if self.use_cute_dsl_bf16_bmm and is_sm_100f():
            torch.ops.trtllm.cute_dsl_bf16_bmm_blackwell(a, b_no_transpose, output)
        else:
            torch.ops.trtllm.bmm_out(a, b_transposed, output)

    def forward_absorption_generation(
        self,
        q: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        num_seqs = attn_metadata.kv_lens_cuda_runtime.size(0)

        cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=q.device)
        fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=q.device)
        has_fp8_kv_cache = (
            self.mqa.has_fp8_kv_cache if hasattr(self.mqa, "has_fp8_kv_cache") else False
        )

        mla_bmm1_scale = None
        mla_bmm2_scale = None
        quant_q_buffer = None
        if has_fp8_kv_cache:
            mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=q.device)
            mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=q.device)
            quant_q_buffer = torch.empty(
                num_tokens,
                self.num_heads_tp,
                (self.kv_lora_rank + self.qk_rope_head_dim),
                dtype=torch.uint8,
                device=q.device,
            )

        fused_q = torch.empty(
            [num_tokens, self.num_heads_tp, (self.kv_lora_rank + self.qk_rope_head_dim)],
            dtype=q.dtype,
            device=q.device,
        )

        rope_stream = self.aux_stream if not has_fp8_kv_cache else None
        assert self.k_b_proj_trans.dtype == torch.bfloat16
        # [num_heads, num_tokens, self.qk_nope_head_dim]
        q_nope_t = q_nope.transpose(0, 1)
        # [num_heads, num_tokens, self.kv_lora_rank]
        q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

        # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
        # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
        # The output of bmm is written directly into fused_q
        maybe_execute_in_parallel(
            lambda: self._bmm_bf16_out(
                q_nope_t, self.k_b_proj_trans, self.k_b_proj_trans.transpose(1, 2), q_nope_out
            ),
            lambda: self.mqa.mla_rope_generation(
                fused_q,
                q_pe,
                latent_cache,
                attn_metadata,
                cu_q_seqlens,
                cu_kv_seqlens,
                fmha_scheduler_counter,
                mla_bmm1_scale,
                mla_bmm2_scale,
                quant_q_buffer,
            ),
            self.ln_events[0],
            self.ln_events[1],
            rope_stream,
        )

        # Use generation_only for generation phase and context_only for context phase in DSA attention
        attention_input_type = AttentionInputType.generation_only

        position_ids_lifetime = position_ids

        # use_generation_kernel = (
        #     _is_env_enabled(_FUSED_MLA_GENERATION_KERNEL_ENV)
        #     and num_tokens == 4
        #     and num_seqs == 1
        #     and topk_indices is not None
        #     and quant_q_buffer is not None
        #     and mla_bmm1_scale is not None
        #     and mla_bmm2_scale is not None
        # )

        use_generation_kernel = False

        if use_generation_kernel:
            self.mqa._ensure_rope_table_size(attn_metadata.max_seq_len)
            topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
                topk_indices,
                attn_metadata,
                layer_idx=self.mqa.get_local_layer_idx(attn_metadata),
                is_generation=True,
            )
            attn_out_latent = _dsv3_fused_mla_generation_torch(
                quant_q_buffer,
                topk_indices,
                topk_indices_pool,
                kv_cache_pool,
                attn_metadata.kv_lens_cuda_runtime,
                mla_bmm1_scale,
                mla_bmm2_scale,
                attn_metadata.spec_decoding_packed_mask,
            )
        else:
            fused_q = fused_q.view(
                [num_tokens, self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)]
            )
            attn_out_latent = self.mqa.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=attention_input_type,
                    out_scale=self.out_scale,
                    output=None,
                    latent_cache=latent_cache,  # kvcache and k_pe
                    q_pe=q_pe,  # used by `invokeMLARopeGeneration`
                    topk_indices=topk_indices,  # used by DSA attention
                    is_generation=True,  # used by DSA attention
                    cu_q_seqlens=cu_q_seqlens,  # used by `mlaGeneration`
                    cu_kv_seqlens=cu_kv_seqlens,  # used by `mlaGeneration`
                    fmha_scheduler_counter=fmha_scheduler_counter,  # used by `mlaGeneration`
                    mla_bmm1_scale=mla_bmm1_scale,  # used by `mlaGeneration`
                    mla_bmm2_scale=mla_bmm2_scale,  # used by `mlaGeneration`
                    quant_q_buffer=quant_q_buffer,  # used by `mlaGeneration`
                ),
            )
        _ = position_ids_lifetime
        fused_q = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (
            attn_out_latent.shape[0] == q.shape[0]
            and attn_out_latent.shape[1] == self.num_heads_tp_cp * self.kv_lora_rank
        )

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        assert self.v_b_proj.dtype == torch.bfloat16
        # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
        # -> [num_heads, seq, v_head_dim]
        self._bmm_bf16_out(
            attn_out_latent.transpose(0, 1),
            self.v_b_proj,
            self.v_b_proj.transpose(1, 2),
            attn_output.transpose(0, 1),
        )

        return output

    def forward_absorption_context(
        self,
        q: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Prefill forward

        - q: Query tensor, shape [num_ctx_tokens, num_heads * qk_head_dim].
        - attn_metadata: Attention metadata for the current batch.
        - output: Pre-allocated output tensor, written in-place.
        - latent_cache: Concatenated [compressed_kv, k_pe] for KV cache.
            - compressed_kv: Latent KV, shape [num_ctx_tokens, kv_lora_rank].
            - k_pe: RoPE key portion, shape [num_ctx_tokens, qk_rope_head_dim].
        - topk_indices: Sparse routing indices from the indexer.
        - position_ids: Token position IDs.
        """
        num_tokens = q.shape[0]

        q_nope, q_pe = q.view([-1, self.num_heads_tp, self.qk_head_dim]).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # fused_q contains 1) the result of the following bmm with shape [num_tokens, num_heads, kv_lora_rank]
        # 2) rope(q_pe) with shape [num_tokens, num_heads, qk_rope_head_dim]. rope is applied inside AttentionOp
        fused_q = torch.empty(
            [num_tokens, self.num_heads_tp, (self.kv_lora_rank + self.qk_rope_head_dim)],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            self._bmm_bf16_out(
                q_nope_t, self.k_b_proj_trans, self.k_b_proj_trans.transpose(1, 2), q_nope_out
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
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        # Use generation_only for generation phase and context_only for context phase in DSA attention
        attention_input_type = AttentionInputType.context_only

        # Fused FP8-Q path: forward the pre-quantized buffers stashed in
        # `_q_branch`; the C++ op enables fusion when both are non-None.
        quant_q_buffer = getattr(self, "_fused_quant_q_buffer", None)
        fused_q_pe = getattr(self, "_fused_q_pe", None)
        quant_scale_qkv = getattr(self, "_quant_scale_qkv", None)
        use_fused_q_fp8 = (
            quant_q_buffer is not None and fused_q_pe is not None and quant_scale_qkv is not None
        )

        if use_fused_q_fp8:
            # Defensive prefix slicing — context-only batches today, mixed-batch later.
            q_pe = fused_q_pe[:num_tokens]
            quant_q_buffer = quant_q_buffer[:num_tokens].view(
                num_tokens, self.num_heads_tp, self.kv_lora_rank + self.qk_rope_head_dim
            )
        else:
            quant_q_buffer = None
            quant_scale_qkv = None

        position_ids_lifetime = position_ids

        fused_mla_mode = get_fused_mla_mode()

        assert self.num_heads_tp == 8
        assert self.num_heads_tp_cp == self.num_heads_tp
        assert self.kv_lora_rank == 512
        assert self.qk_rope_head_dim == 64
        assert attn_metadata.num_contexts == 1
        assert getattr(attn_metadata, "num_ctx_cached_tokens", 0) == 0
        assert topk_indices is not None
        assert hasattr(attn_metadata, "ctx_cached_token_indptr")

        if fused_mla_mode == FUSED_MLA_MODE_PYTORCH or fused_mla_mode == FUSED_MLA_MODE_WIP:
            self.mqa._ensure_rope_table_size(attn_metadata.max_seq_len)
            topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
                topk_indices,
                attn_metadata,
                layer_idx=self.mqa.get_local_layer_idx(attn_metadata),
                is_generation=False,
            )
            layer_idx = self.mqa.get_local_layer_idx(attn_metadata)

            if fused_mla_mode == FUSED_MLA_MODE_PYTORCH:
                attn_out_latent = dsv3_mla_context_pytorch(
                    fused_q,
                    q_pe,
                    latent_cache,
                    topk_indices,
                    topk_indices_pool,
                    kv_cache_pool,
                    attn_metadata,
                    self.mqa,
                )
            else:
                attn_out_latent = torch.ops.trtllm.dsv3_fused_mla_context(
                    fused_q,
                    q_pe,
                    latent_cache,
                    topk_indices_pool,
                    topk_indices,
                    kv_cache_pool,
                    self.mqa.rotary_cos_sin,
                    attn_metadata.ctx_cached_token_indptr,
                    attn_metadata.kv_cache_block_offsets,
                    attn_metadata.kv_cache_manager.kv_cache_pool_pointers,
                    attn_metadata.kv_cache_manager.kv_cache_pool_mapping,
                    self.mqa.kv_scale_orig_quant,
                    self.mqa.kv_scale_quant_orig,
                    layer_idx,
                    attn_metadata.kv_cache_manager.tokens_per_block,
                    int(self.mqa.quant_mode),
                    float(self.mqa.q_scaling),
                )
        else:  # baseline mode
            fused_q = fused_q.view(
                [num_tokens, self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)]
            )
            attn_out_latent = self.mqa.forward(
                fused_q,
                None,
                None,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=attention_input_type,
                    out_scale=self.out_scale,
                    output=None,
                    latent_cache=latent_cache,  # kvcache and k_pe
                    q_pe=q_pe,  # used by applyMLARopeAndAssignQKVKernelOptContext
                    quant_q_buffer=quant_q_buffer,  # fused-FP8 path only
                    quant_scale_qkv=quant_scale_qkv,  # fused-FP8 path only
                    topk_indices=topk_indices,  # used by DSA attention
                    is_generation=False,  # used by DSA attention
                ),
            )
        _ = position_ids_lifetime
        fused_q = None
        self._fused_quant_q_buffer = None
        self._fused_q_pe = None

        # note: if we do not have CP, then num_heads_tp_cp == num_heads_tp
        assert (
            attn_out_latent.shape[0] == q.shape[0]
            and attn_out_latent.shape[1] == self.num_heads_tp_cp * self.kv_lora_rank
        )

        # [seq, num_heads, kv_lora_rank]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp_cp, self.kv_lora_rank])

        attn_output = output.view([num_tokens, self.num_heads_tp_cp, self.v_head_dim])

        if self.v_b_proj.dtype == torch.bfloat16:
            # [num_heads, seq, kv_lora_rank] x [num_heads, kv_lora_rank, v_head_dim]
            # -> [num_heads, seq, v_head_dim]
            self._bmm_bf16_out(
                attn_out_latent.transpose(0, 1),
                self.v_b_proj,
                self.v_b_proj.transpose(1, 2),
                attn_output.transpose(0, 1),
            )
        elif self.v_b_proj.dtype == torch.float8_e4m3fn:
            attn_output_t = attn_output.transpose(0, 1)
            if is_sm_100f() and not self.use_cute_dsl_blockscaling_bmm:
                torch.bmm(
                    attn_out_latent.transpose(0, 1),
                    self.v_b_proj_dequant.transpose(1, 2),
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

        return output

    def forward(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        all_reduce_params: Optional[AllReduceParams] = None,
        # latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run fused MLA for a DeepSeekV3/GLM DSA attention layer.

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
        """
        attn_output = hidden_states.new_empty(
            [hidden_states.shape[0], self.o_proj.in_features], dtype=hidden_states.dtype
        )
        assert self.register_to_config
        assert self.mqa is not None, "DSA is only supported in MQA mode"

        q, compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
            [self.q_lora_rank, self.kv_lora_rank, self.qk_rope_head_dim], -1
        )

        q, compressed_kv = maybe_execute_in_parallel(
            lambda: self.q_a_layernorm(q),
            lambda: self.kv_a_layernorm(compressed_kv),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )
        qr = q
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)

        q = self.q_b_proj(q)

        q_fp8, k_fp8, k_scale, weights, q_scale = self.mqa.indexer.pre_indexer_proj(
            qr, hidden_states, position_ids
        )
        indexer_intermediates = [q_fp8, k_fp8, k_scale, weights, q_scale]

        self.forward_dsa_attn(
            q,
            compressed_kv,
            k_pe,
            latent_cache,
            indexer_intermediates,
            position_ids,
            attn_metadata,
            attn_output,
        )
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
