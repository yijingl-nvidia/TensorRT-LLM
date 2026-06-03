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

import functools
import math
import os
import weakref
from typing import List, Optional, cast

import torch
from torch import nn

import tensorrt_llm.quantization.utils.fp8_utils as fp8_utils
from tensorrt_llm._utils import get_sm_version, is_sm_100f, nvtx_range, nvtx_range_debug
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

from ..attention_backend import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    TrtllmAttention,
    TrtllmAttentionMetadata,
)
from ..attention_backend.interface import (
    AttentionBackend,
    PositionalEmbeddingParams,
    PredefinedAttentionMask,
)
from ..attention_backend.sparse.dsa import (
    DSAtrtllmAttentionMetadata,
    transform_local_topk_and_prepare_pool_view,
)
from ..attention_backend.utils import create_attention
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.attention import _helix_post_process, extract_extra_attrs, fp8_block_scaling_bmm_out
from ..modules.linear import Linear, TensorParallelMode
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..modules.rms_norm import RMSNorm
from ..modules.rotary_embedding import RotaryEmbedding
from ..utils import is_torch_compiling, maybe_compiled_cat, maybe_compiled_copy_

# Import FlashMLA sparse attention kernel
try:
    from tensorrt_llm.flash_mla import flash_mla_sparse_fwd
except ImportError:
    flash_mla_sparse_fwd = None


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
                W_qkvia : ℝ⁶¹⁴⁴ ─► ℝ²⁶²⁴ "Combine q_a_proj and kv_a_proj_with_mqa, QKV down-projection"
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
              "Q up-projection"                               │
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
    q̃_h^lat = W_uk,h · q_nope_h  │                            │
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
                                                      (v_b_proj, V-up, also absorbed)
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
        """
        super().__init__()
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
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.q_lora_rank + self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
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
            self.kv_a_proj_with_mqa = Linear(
                hidden_size,
                self.kv_lora_rank + self.qk_rope_head_dim,
                bias=bias,
                dtype=dtype,
                quant_config=quant_config,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                use_custom_cublas_mm=True,
                force_dynamic_quantization=config.force_dynamic_quantization,
                use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
                use_cute_dsl_bf16_gemm=self.use_cute_dsl_bf16_gemm,
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

        self.softmax_scale = 1.0 / (math.sqrt(self.qk_head_dim) * q_scaling)

        self.aux_stream = aux_stream
        self.ln_events = [torch.cuda.Event(), torch.cuda.Event()]

        self.rope_fusion = self.mqa.support_fused_rope()
        self.rotary_emb = None
        self.apply_rotary_emb = not self.rope_fusion
        if self.apply_rotary_emb:
            self.rotary_emb = RotaryEmbedding(
                pos_embd_params.rope,
                head_dim=self.qk_rope_head_dim,
                is_neox=pos_embd_params.is_neox,
            )

        # Short-sequence MHA optimization for DSA models:
        # For short prefill sequences, use MHA (kv_b_proj expansion + standard
        # attention) instead of the absorption path, which has overhead from
        # extra BMMs and larger head_dim (kv_lora_rank + qk_rope_head_dim).
        # Only active when rope_fusion is True (DSA with TrtllmAttention).
        _threshold_str = os.environ.get("TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD", "0")
        try:
            self.short_seq_mha_threshold = int(_threshold_str)
        except ValueError as err:
            raise ValueError(
                f"TRTLLM_MLA_SHORT_SEQ_MHA_THRESHOLD must be an integer, got '{_threshold_str}'"
            ) from err

        # MHA attention backend: used by non-DSA (standard MLA) and optionally
        # by DSA for the short-seq path (dense attention, no sparse config).
        _short_seq_mha = (
            self.is_dsa and self.short_seq_mha_threshold > 0 and not self.apply_rotary_emb
        )
        if _short_seq_mha:
            self.mha = create_attention(
                config.attn_backend,
                self.layer_idx,
                self.num_heads_tp,
                head_dim=self.qk_head_dim,
                num_kv_heads=self.num_key_value_heads_tp,
                pos_embd_params=pos_embd_params,
                quant_config=quant_config,
                q_scaling=q_scaling,
                is_mla_enable=True,
                q_lora_rank=self.q_lora_rank,
                kv_lora_rank=self.kv_lora_rank,
                qk_nope_head_dim=self.qk_nope_head_dim,
                qk_rope_head_dim=self.qk_rope_head_dim,
                v_head_dim=self.v_head_dim,
                predicted_tokens_per_seq=self.predicted_tokens_per_seq,
                skip_create_weights_in_init=config.skip_create_weights_in_init,
                sparse_attention_config=(
                    None if _short_seq_mha else config.sparse_attention_config
                ),
            )
        else:
            self.mha = None

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
        # self.mha/mqa has no weights but has states that are related to
        # quant_config, which could be modified after __init__.
        # self.mha is non-None for non-DSA models (standard MHA) and for DSA
        # models when the short-seq MHA optimization is active.
        if self.mha is not None:
            self.mha.update_quant_config(self.quant_config)
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

    def apply_rope(
        self,
        q: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        q = q.view(-1, self.num_heads_tp, self.qk_head_dim)
        q_pe = q[..., self.qk_nope_head_dim :].reshape(
            -1, self.num_heads_tp * self.qk_rope_head_dim
        )
        q_pe, k_pe = self.rotary_emb(position_ids, [q_pe, k_pe])
        q[..., self.qk_nope_head_dim :] = q_pe.view(-1, self.num_heads_tp, self.qk_rope_head_dim)
        return k_pe

    def _is_fused_q_fp8_quant_enabled(self, num_generations: int = 0) -> bool:
        # Context-only batches: the fused path leaves a placeholder bf16 q_buf
        # that forward_generation_sparse_mla would read uninitialized, so
        # mixed/gen batches must take the legacy unfused path.
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

    def forward_impl(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache_gen: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Forward pass for the MLA module. Writes result into output tensor in-place.

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            output (torch.Tensor): The output tensor to write results into.
            latent_cache_gen (Optional[torch.Tensor]): The latent cache used in generation.
        """
        # split q, k, v into context and gen batches
        num_contexts = attn_metadata.num_contexts
        num_generations = attn_metadata.num_generations
        num_ctx_tokens = attn_metadata.num_ctx_tokens
        num_tokens = attn_metadata.num_tokens

        hidden_states = hidden_states[:num_tokens, ...]
        if position_ids is not None:
            position_ids = position_ids[..., :num_tokens]

        if self.is_lite:
            compressed_kv, k_pe = self.kv_a_proj_with_mqa(hidden_states).split(
                [self.kv_lora_rank, self.qk_rope_head_dim], -1
            )
            compressed_kv = self.kv_a_layernorm(compressed_kv)
            q = hidden_states
        else:
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

        q, latent_cache = maybe_execute_in_parallel(
            lambda: self.q_b_proj(q),
            lambda: torch.concat([compressed_kv, k_pe], dim=-1),
            self.ln_events[0],
            self.ln_events[1],
            self.aux_stream,
        )

        assert q.shape[0] == num_tokens, (
            f"Expect q.shape[0] to be {num_tokens}, but got {q.shape[0]}"
        )

        assert output is not None, "output must be provided"

        if num_contexts > 0:
            q_ctx = q[:num_ctx_tokens, ...]
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            if self.llama_4_scaling:
                q_ctx = self._attention_scaling(q_ctx, position_ids[..., :num_ctx_tokens])

            self.forward_context(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                position_ids,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            if latent_cache_gen is None:
                latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            if self.llama_4_scaling:
                q_gen = self._attention_scaling(q_gen, position_ids[..., num_ctx_tokens:])

            self.forward_absorption_generation(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                position_ids=position_ids,
                latent_cache=latent_cache_gen,
            )

    def forward_impl_with_dsa(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> None:
        """
        Forward pass for the MLA module with DSA (always in MQA mode).
        Writes result into output tensor in-place.

        Delegates to forward_dsa_proj (token-wise projections) followed by
        forward_dsa_attn (batch-dependent attention dispatch).

        Args:
            position_ids (Optional[torch.IntTensor]): The position IDs.
            hidden_states (torch.Tensor): The hidden states.
            attn_metadata (AttentionMetadata): The attention metadata.
            output (torch.Tensor): The output tensor to write results into.
        """
        proj_outputs = self.forward_dsa_proj(position_ids, hidden_states, attn_metadata)
        q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
        indexer_intermediates = proj_outputs[4:]
        self.forward_dsa_attn(
            q,
            compressed_kv,
            k_pe,
            latent_cache,
            indexer_intermediates,
            position_ids,
            attn_metadata,
            output,
        )

    def forward_dsa_proj(
        self,
        position_ids: Optional[torch.Tensor],
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> List[torch.Tensor]:
        """Token-wise projections for DSA MLA (CUDA-graph-capturable Op 1).

        Runs kv_a_proj, layernorms, q_b_proj, and conditionally
        indexer.pre_indexer_proj().

        IMPORTANT: This method must NOT slice tensors by num_tokens or
        access batch-specific metadata, so that all operations are
        unconditionally straight-line for CUDA graph capture.  Slicing
        to num_tokens happens in forward_dsa_attn (Op 2, outside graph).

        Returns [q, compressed_kv, k_pe, latent_cache] when short-MHA
        handles all tokens (eager only), or
        [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale,
        weights, q_scale] when the indexer runs.  q_scale is unused on the
        FP8 path but always present so CUDA graph capture sees a stable
        9-tensor shape regardless of indexer dtype.
        """
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

        use_short_mha_for_ctx = self._should_use_short_mha(attn_metadata, position_ids)

        # Skip the indexer when the short MHA path handles all context
        # tokens and there are no generation tokens.
        if use_short_mha_for_ctx and attn_metadata.num_generations == 0:
            return [q, compressed_kv, k_pe, latent_cache]

        # pre_indexer_proj is the CUDA-graph-safe portion: pure token-wise
        # compute (cublas_mm, rope, FP4/FP8 quantize, weight scaling) with no
        # access to batch-specific metadata or the k cache. Returns q_scale
        # as a 5th element so the FP4 dispatch can forward it to the kernel;
        # the FP8 path ignores it in forward_dsa_attn.
        q_fp8, k_fp8, k_scale, weights, q_scale = self.mqa.indexer.pre_indexer_proj(
            qr, hidden_states, position_ids
        )

        return [q, compressed_kv, k_pe, latent_cache, q_fp8, k_fp8, k_scale, weights, q_scale]

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

        indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale]
        when the indexer ran in Op 1, or [] when short-MHA handled all tokens.

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

        use_short_mha_for_ctx = num_contexts > 0 and self._should_use_short_mha(
            attn_metadata, position_ids
        )

        if use_short_mha_for_ctx and num_generations == 0:
            topk_indices = None
        else:
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
            compressed_kv_ctx = compressed_kv[:num_ctx_tokens, ...]
            k_pe_ctx = k_pe[:num_ctx_tokens, ...]
            latent_cache_ctx = latent_cache[:num_ctx_tokens, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_ctx = self.apply_rope(q_ctx, k_pe_ctx, position_ids)

            self.forward_context_sparse_mla(
                q_ctx,
                compressed_kv_ctx,
                k_pe_ctx,
                attn_metadata,
                output[:num_ctx_tokens, :],
                latent_cache_ctx,
                topk_indices=topk_indices[:num_ctx_tokens, :] if topk_indices is not None else None,
                position_ids=position_ids,
            )

        if num_generations > 0:
            q_gen = q[num_ctx_tokens:, ...]
            compressed_kv_gen = compressed_kv[num_ctx_tokens:, ...]
            k_pe_gen = k_pe[num_ctx_tokens:, ...]
            latent_cache_gen = latent_cache[num_ctx_tokens:, ...]
            if self.apply_rotary_emb:
                assert position_ids is not None
                k_pe_gen = self.apply_rope(q_gen, k_pe_gen, position_ids)

            self.forward_generation_sparse_mla(
                q_gen,
                compressed_kv_gen,
                k_pe_gen,
                attn_metadata,
                output[num_ctx_tokens:num_tokens, :],
                latent_cache=latent_cache_gen,
                topk_indices=topk_indices[num_ctx_tokens:num_tokens, :],
            )

    def forward_context_default(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Dense MHA context path: expand KV via kv_b_proj and run attention.

        Used by non-DSA models and as the short-seq MHA fallback for DSA models.
        """
        kv = self.kv_b_proj(compressed_kv)
        k_nope, v = kv.split(
            [self.num_heads_tp * self.qk_nope_head_dim, self.num_heads_tp * self.v_head_dim],
            -1,
        )

        k = torch.empty_like(q).view(-1, self.num_heads_tp, self.qk_head_dim)
        maybe_compiled_copy_(
            k[..., : self.qk_nope_head_dim],
            k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim),
        )
        # When rope_fusion=True (apply_rotary_emb=False), the rope portion
        # of k is left uninitialized here; the fused attention kernel
        # handles k_pe RoPE via latent_cache instead.
        if self.apply_rotary_emb:
            k[..., self.qk_nope_head_dim :] = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=latent_cache,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def _should_use_short_mha(
        self, attn_metadata: AttentionMetadata, position_ids: Optional[torch.Tensor]
    ) -> bool:
        """Check if the short-seq MHA optimization should be used for context.

        Uses max_ctx_kv_len (max total KV length per context sequence,
        including cached tokens) when available, to correctly account for
        chunked context where the full attention span exceeds the threshold
        even if the new token count is small.  Falls back to num_ctx_tokens
        (total new context tokens) when max_ctx_kv_len is not set.

        Disabled under torch compile so that the split DSA custom ops
        (mla_dsa_proj / mla_dsa_attn_inplace) have unconditionally
        straight-line control flow for CUDA graph capture.
        """
        if is_torch_compiling():
            return False
        if not (
            self.short_seq_mha_threshold > 0
            and not self.apply_rotary_emb
            and self.mapping.cp_size == 1
            and position_ids is not None
        ):
            return False
        effective_len = getattr(attn_metadata, "max_ctx_kv_len", attn_metadata.num_ctx_tokens)
        return effective_len <= self.short_seq_mha_threshold

    def forward_context_sparse_mla(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run context-phase attention for DSA models.

        Dispatches to the short-seq MHA path (forward_context) when the max
        per-sequence KV length (including cached tokens) is within the
        threshold, or falls through to the absorption/sparse MLA path
        otherwise.  forward_context() further dispatches to the appropriate
        handler (forward_context_default, forward_context_with_cached_kv, or
        forward_context_with_chunked_prefill) based on cached-KV state.

        Args:
            q: Query tensor, shape [num_ctx_tokens, num_heads * qk_head_dim].
            compressed_kv: Latent KV, shape [num_ctx_tokens, kv_lora_rank].
            k_pe: RoPE key portion, shape [num_ctx_tokens, qk_rope_head_dim].
            attn_metadata: Attention metadata for the current batch.
            output: Pre-allocated output tensor, written in-place.
            latent_cache: Concatenated [compressed_kv, k_pe] for KV cache.
            topk_indices: Sparse routing indices from the indexer (None when
                the short-seq MHA path is used).
            position_ids: Token position IDs (required for short-seq MHA).
        """
        # Short-sequence MHA: bypass absorption path for short prefills,
        # using kv_b_proj expansion + standard attention instead.
        # See __init__ comment for rationale. topk_indices is not used
        # because dense attention is faster than sparse routing at this scale.
        # forward_context() handles cached tokens by dispatching to
        # forward_context_with_cached_kv or forward_context_with_chunked_prefill.
        if self._should_use_short_mha(attn_metadata, position_ids):
            return self.forward_context(
                q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
            )

        if get_sm_version() >= 100:
            return self.forward_absorption_context(
                q,
                compressed_kv,
                k_pe,
                attn_metadata,
                output,
                position_ids=position_ids,
                latent_cache=latent_cache,
                topk_indices=topk_indices,
            )
        else:
            return self.forward_sparse_mla_kvcache_bf16(
                q, latent_cache, attn_metadata, output, topk_indices, is_generation=False
            )

    def forward_generation_sparse_mla(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        latent_cache: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if get_sm_version() >= 100:
            return self.forward_absorption_generation(
                q,
                compressed_kv,
                k_pe,
                attn_metadata,
                output,
                position_ids=position_ids,
                latent_cache=latent_cache,
                topk_indices=topk_indices,
            )
        else:
            return self.forward_sparse_mla_kvcache_bf16(
                q, latent_cache, attn_metadata, output, topk_indices, is_generation=True
            )

    def forward_context_with_cached_kv(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        assert latent_cache is not None
        trtllm_attention = cast(TrtllmAttention, self.mha)

        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

        # copy full_compressed_kv and full_k_pe from paged kv cache
        full_compressed_kv, full_k_pe = trtllm_attention.load_paged_kv_cache_for_mla(
            attn_metadata, q.dtype
        )
        assert (
            full_compressed_kv.shape[0]
            == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        )
        assert full_compressed_kv.shape[1] == self.kv_lora_rank
        assert (
            full_k_pe.shape[0] == attn_metadata.num_ctx_cached_tokens + attn_metadata.num_ctx_tokens
        )
        assert full_k_pe.shape[1] == self.qk_rope_head_dim
        assert full_compressed_kv.is_contiguous()
        assert full_k_pe.is_contiguous()

        # compute full_k_nope and full_v from full_compressed_kv
        full_kv = self.kv_b_proj(full_compressed_kv)
        full_k_nope, full_v = full_kv.split(
            [self.num_heads_tp * self.qk_nope_head_dim, self.num_heads_tp * self.v_head_dim],
            -1,
        )

        full_k_nope = full_k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        full_k_pe = full_k_pe.view(-1, 1, self.qk_rope_head_dim)
        full_k = maybe_compiled_cat(
            (full_k_nope, full_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1
        )
        full_k = full_k.view(-1, self.num_heads_tp * self.qk_head_dim)

        # release pytorch activation memory
        full_compressed_kv = None
        full_k_pe = None
        full_kv = None
        full_k_nope = None

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        attn_output = self.mha.forward(
            q,
            full_k,
            full_v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                output=output,
            ),
        )

        return attn_output

    def forward_context_with_chunked_prefill(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        latent_cache: torch.Tensor,  # compressed_kv + k_pe [context_tokens, 1, lora_size + rope_size]
        attn_metadata: TrtllmAttentionMetadata,
        output: torch.Tensor,
    ) -> torch.Tensor:
        trtllm_attention = cast(TrtllmAttention, self.mha)
        # apply RoPE, append compressed_kv + k_pe to paged kv cache and assign q_pe to q
        trtllm_attention.mla_rope_append_paged_kv_assign_q(q, latent_cache, attn_metadata)

        # determine the number of loop
        # currently we assume that the chunk size is the same as the max_num_tokens
        chunked_loop_num = attn_metadata.chunked_loop_num

        # [total_token_q, num_heads, 2] -> [total_token_q, num_heads] float2
        self.softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device="cuda",
        )
        self.temp_softmax_stats_tensor = torch.empty(
            (attn_metadata.num_ctx_tokens, self.num_heads_tp, 2),
            dtype=torch.float,
            device="cuda",
        )

        attn_output = output
        temp_attn_output = q.new_empty(
            (q.size(0), self.num_heads_tp * self.v_head_dim), dtype=q.dtype
        )

        # use fake cached_cu_seq_len for chunked loop
        origin_kv_lens_cuda_runtime = attn_metadata.kv_lens_cuda_runtime
        origin_kv_lens_runtime = attn_metadata.kv_lens_runtime
        origin_ctx_total_kv_len = attn_metadata.host_total_kv_lens[0]

        for loop_idx in range(chunked_loop_num):
            # {b, chunked_unit_size, h, kv_lora_rank + qk_rope_head_dim} zero padded
            # fetch `loop_idx` chunk from kv cache
            temp_cu_chunked_seq_len = attn_metadata.cu_chunked_seq_len[loop_idx]
            total_ctx_chunked_tokens = attn_metadata.host_cu_chunked_seq_len[
                loop_idx, attn_metadata.num_contexts
            ]
            chunked_global_offset = attn_metadata.chunked_global_offset[loop_idx]
            chunked_max_seq_len = attn_metadata.max_chunk_len_per_loop[loop_idx]
            chunked_compressed_kv, chunked_k_pe = trtllm_attention.load_chunked_kv_cache_for_mla(
                metadata=attn_metadata,
                num_ctx_cached_tokens=total_ctx_chunked_tokens,
                cu_chunked_seq_len=temp_cu_chunked_seq_len,
                chunked_global_offset=chunked_global_offset,
                chunked_max_seq_len=chunked_max_seq_len,
                out_dtype=q.dtype,
            )

            # up proj to uncompressed kv
            # [tokens, 2, h, kv_dim], without rope_dim
            chunked_kv = self.kv_b_proj(chunked_compressed_kv)
            chunked_k_nope, chunked_v = chunked_kv.split(
                [self.num_heads_tp * self.qk_nope_head_dim, self.num_heads_tp * self.v_head_dim],
                -1,
            )

            chunked_k_nope = chunked_k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
            chunked_k_pe = chunked_k_pe.view(-1, 1, self.qk_rope_head_dim)
            chunked_k = maybe_compiled_cat(
                (chunked_k_nope, chunked_k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1
            )
            chunked_k = chunked_k.view(-1, self.num_heads_tp * self.qk_head_dim)

            # release pytorch activation memory
            chunked_compressed_kv = None
            chunked_k_pe = None
            chunked_kv = None
            chunked_k_nope = None

            # copy chunked_seq_len to replace kv_lens_runtime
            attn_metadata.kv_lens_runtime = attn_metadata.host_chunked_seq_len[loop_idx]
            attn_metadata.kv_lens_cuda_runtime = attn_metadata.chunked_seq_len[loop_idx]
            attn_metadata.host_total_kv_lens[0] = total_ctx_chunked_tokens

            # do not apply mask for attention within loop
            # latent_cache must be None to differentiate from normal context phase,
            # so that we can skip applying RoPE and appending KV cache inside attention op
            temp_attn_output = self.mha.forward(
                q,
                chunked_k,
                chunked_v,
                attn_metadata,
                forward_args=AttentionForwardArgs(
                    attention_input_type=AttentionInputType.context_only,
                    latent_cache=None,
                    out_scale=self.out_scale,
                    attention_mask=PredefinedAttentionMask.FULL,
                    softmax_stats_tensor=self.temp_softmax_stats_tensor,
                    chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.chunked_prefill_buffer_batch_size,
                    output=temp_attn_output,
                ),
            )
            # merge attn result
            temp_merge_op = attn_metadata.merge_op_tensor[loop_idx]
            trtllm_attention.merge_attention_for_mla(
                attn_output,
                temp_attn_output,
                self.softmax_stats_tensor,
                self.temp_softmax_stats_tensor,
                temp_merge_op,
                attn_metadata,
            )

        # deal with the uncached kv
        kv = self.kv_b_proj(compressed_kv)
        _, k_pe = latent_cache.view([-1, self.kv_lora_rank + self.qk_rope_head_dim]).split(
            [self.kv_lora_rank, self.qk_rope_head_dim], -1
        )
        # final round of attention

        k_nope, v = kv.split(
            [self.num_heads_tp * self.qk_nope_head_dim, self.num_heads_tp * self.v_head_dim],
            -1,
        )

        k_nope = k_nope.view(-1, self.num_heads_tp, self.qk_nope_head_dim)
        k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)
        k = maybe_compiled_cat((k_nope, k_pe.expand(-1, self.num_heads_tp, -1)), dim=-1)
        k = k.view(-1, self.num_heads_tp * self.qk_head_dim)

        # copy q_lens to replace kv_lens_runtime
        attn_metadata.kv_lens_runtime = attn_metadata.prompt_lens_cpu_runtime
        attn_metadata.kv_lens_cuda_runtime = attn_metadata.prompt_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = (
            attn_metadata.prompt_lens_cpu_runtime[: attn_metadata.num_contexts].sum().item()
        )

        # latent_cache must be None to differentiate from normal context phase,
        # so that we can skip applying RoPE and appending KV cache inside attention op
        temp_attn_output = self.mha.forward(
            q,
            k,
            v,
            attn_metadata,
            forward_args=AttentionForwardArgs(
                attention_input_type=AttentionInputType.context_only,
                latent_cache=None,
                out_scale=self.out_scale,
                softmax_stats_tensor=self.temp_softmax_stats_tensor,
                chunked_prefill_buffer_batch_size=attn_metadata.runtime_features.chunked_prefill_buffer_batch_size,
                output=temp_attn_output,
            ),
        )
        temp_merge_op = attn_metadata.merge_op_tensor[chunked_loop_num]
        trtllm_attention.merge_attention_for_mla(
            attn_output,
            temp_attn_output,
            self.softmax_stats_tensor,
            self.temp_softmax_stats_tensor,
            temp_merge_op,
            attn_metadata,
        )
        # copy back kv_lens_runtime and kv_lens_cuda_runtime
        attn_metadata.kv_lens_runtime = origin_kv_lens_runtime
        attn_metadata.kv_lens_cuda_runtime = origin_kv_lens_cuda_runtime
        attn_metadata.host_total_kv_lens[0] = origin_ctx_total_kv_len

        return attn_output

    @staticmethod
    @functools.cache
    def cached_warmup_forward_context_with_chunked_prefill(
        num_heads_tp, qk_nope_head_dim, qk_rope_head_dim, kv_lora_rank, v_head_dim, dtype, device
    ):
        """Warmup torch.compile for cat operations with different tensor layouts.

        Tensors are marked with torch._dynamo.maybe_mark_dynamic(..., 0) on the
        num_tokens dimension, so for num_tokens != 1 a single warmup run is
        enough and the compiled kernel generalizes across varying num_tokens at
        runtime. num_tokens=1 still triggers recompile (torch.compile specializes
        for it), so it is warmed up separately. Do not use torch.compile with
        dynamic=True here because it completely ignores tensor layout/stride
        information, resulting in significantly degraded performance.
        """

        def warmup(num_tokens):
            chunked_k_nope = k_nope = torch.empty(
                num_tokens,
                num_heads_tp * (qk_nope_head_dim + v_head_dim),
                dtype=dtype,
                device=device,
            )[:, : num_heads_tp * qk_nope_head_dim].view(num_tokens, num_heads_tp, qk_nope_head_dim)
            chunked_k_pe = torch.empty(
                num_tokens, 1, qk_rope_head_dim, dtype=dtype, device=device
            ).expand(-1, num_heads_tp, -1)
            k_pe = torch.empty(
                num_tokens, 1, kv_lora_rank + qk_rope_head_dim, dtype=dtype, device=device
            )[:, :, -qk_rope_head_dim:].expand(-1, num_heads_tp, -1)
            torch._dynamo.maybe_mark_dynamic(chunked_k_nope, 0)
            torch._dynamo.maybe_mark_dynamic(chunked_k_pe, 0)
            torch._dynamo.maybe_mark_dynamic(k_pe, 0)
            maybe_compiled_cat((chunked_k_nope, chunked_k_pe), dim=-1)
            maybe_compiled_cat((k_nope, k_pe), dim=-1)

        # With dim 0 (num_tokens) marked dynamic, one warmup suffices for all
        # num_tokens != 1 at runtime.
        warmup(2)

        # num_tokens=1 still triggers recompile; warm it separately.
        warmup(1)

    def forward_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        output: torch.Tensor,
        latent_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(self.mha, TrtllmAttention):
            assert isinstance(attn_metadata, TrtllmAttentionMetadata)
            trtllm_attention = cast(TrtllmAttention, self.mha)
            if trtllm_attention.is_chunked_prefill_mla_context_for_warmup(attn_metadata):
                self.cached_warmup_forward_context_with_chunked_prefill(
                    self.num_heads_tp,
                    self.qk_nope_head_dim,
                    self.qk_rope_head_dim,
                    self.kv_lora_rank,
                    self.v_head_dim,
                    q.dtype,
                    q.device,
                )
            if (
                trtllm_attention.is_chunked_prefill_for_mla_context(attn_metadata)
                and get_sm_version() >= 100
            ):
                return self.forward_context_with_chunked_prefill(
                    q, compressed_kv, latent_cache, attn_metadata, output
                )
            elif trtllm_attention.has_cached_kv_for_mla_context(
                attn_metadata
            ) or trtllm_attention.is_chunked_prefill_for_mla_context(attn_metadata):
                return self.forward_context_with_cached_kv(q, latent_cache, attn_metadata, output)
        return self.forward_context_default(
            q, compressed_kv, k_pe, position_ids, attn_metadata, output, latent_cache
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
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
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
        if self.k_b_proj_trans.dtype == torch.bfloat16:
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

        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = fused_q[..., : self.kv_lora_rank].transpose(0, 1)

            maybe_execute_in_parallel(
                lambda: fp8_block_scaling_bmm_out(
                    q_nope,
                    self.k_b_proj_trans,
                    self.k_b_proj_trans_scale,
                    q_nope_out,
                    self.k_b_proj_trans_dequant,
                    self.use_cute_dsl_blockscaling_bmm,
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
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.k_b_proj_trans.dtype}.")

        fused_q = fused_q.view(
            [num_tokens, self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)]
        )

        # Use generation_only for generation phase and context_only for context phase in DSA attention
        attention_input_type = AttentionInputType.generation_only

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
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
        )
        fused_q = None

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
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    def forward_absorption_context(
        self,
        q: torch.Tensor,
        compressed_kv: torch.Tensor,
        k_pe: torch.Tensor,
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

        if self.apply_rotary_emb:
            fused_q[..., self.kv_lora_rank :] = q_pe
        fused_q = fused_q.view(
            [num_tokens, self.num_heads_tp * (self.kv_lora_rank + self.qk_rope_head_dim)]
        )

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

        attn_out_latent = self._attn_forward_gen(
            self.mqa,
            fused_q,
            None,
            None,
            position_ids,
            attn_metadata,
            attention_input_type=attention_input_type,
            out_scale=self.out_scale,
            output=None,
            latent_cache=latent_cache,  # kvcache and k_pe
            q_pe=q_pe,  # used by applyMLARopeAndAssignQKVKernelOptContext
            quant_q_buffer=quant_q_buffer,  # fused-FP8 path only
            quant_scale_qkv=quant_scale_qkv,  # fused-FP8 path only
            topk_indices=topk_indices,  # used by DSA attention
            is_generation=False,  # used by DSA attention
        )
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
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
                self.v_b_proj_dequant,
                self.use_cute_dsl_blockscaling_bmm,
            )
        else:
            raise NotImplementedError(f"Missing bmm impl for dtype: {self.v_b_proj.dtype}.")

        return output

    @nvtx_range("forward_sparse_mla_kvcache_bf16")
    def forward_sparse_mla_kvcache_bf16(
        self,
        q: torch.Tensor,
        latent_cache: torch.Tensor,
        attn_metadata: DSAtrtllmAttentionMetadata,
        output: torch.Tensor,
        topk_indices: torch.Tensor,
        is_generation: bool = False,
    ) -> torch.Tensor:
        """
        Forward sparse MLA (DSA) for BF16 KV cache for both context and generation phases using FlashMLA kernels

        To form the input for FlashMLA kernel and adapt our KV cache manager, we need to:
        1. Append current tokens to paged cache and apply rope to q/k via mla_rope_append_paged_kv_assign_q
        2. Load full kv cache from paged memory (with k rope applied)
        3. Call FlashMLA sparse attention kernel for sparse prefill/decode
        """
        assert isinstance(attn_metadata, DSAtrtllmAttentionMetadata), (
            "DSA requires DSAtrtllmAttentionMetadata"
        )
        # Append current tokens to paged cache and apply RoPE to q
        # This writes latent_cache to paged KV and modifies q in-place
        trtllm_attention = self.mqa
        with nvtx_range_debug(f"mla_rope_append_paged_kv_assign_q_is_generation={is_generation}"):
            trtllm_attention.mla_rope_append_paged_kv_assign_q(
                q, latent_cache, attn_metadata, is_generation=is_generation
            )

        num_tokens = q.shape[0]
        q_nope, q_rope = q.view(-1, self.num_heads_tp, self.qk_head_dim).split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_nope_out = torch.empty(
            [num_tokens, self.num_heads_tp, (self.kv_lora_rank)],
            dtype=q.dtype,
            device=q.device,
        )

        if self.k_b_proj_trans.dtype == torch.bfloat16:
            # [num_heads, num_tokens, self.qk_nope_head_dim]
            q_nope_t = q_nope.transpose(0, 1)
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

            # [num_heads, num_tokens, self.qk_nope_head_dim] x [num_heads, kv_lora_rank, qk_nope_head_dim]
            # -> [num_heads, num_tokens, kv_lora_rank] -> [num_tokens, num_heads, kv_lora_rank]
            # The output of bmm is written directly into fused_q
            self._bmm_bf16_out(
                q_nope_t, self.k_b_proj_trans, self.k_b_proj_trans.transpose(1, 2), q_nope_out
            )
        elif self.k_b_proj_trans.dtype == torch.float8_e4m3fn:
            # [num_heads, num_tokens, self.kv_lora_rank]
            q_nope_out = q_nope_out.transpose(0, 1)

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

        q_nope_out = q_nope_out.transpose(0, 1)
        q_concat = torch.cat([q_nope_out, q_rope], dim=-1)

        sm_version = get_sm_version()
        # FlashMLA sparse kernel (bf16) requires num_heads=128 on sm100 or multiple of 64 on sm90
        if sm_version >= 100:
            padding = 128
            assert self.num_heads_tp <= padding, (
                f"SM100 FlashMLA sparse kernel requires exactly {padding} heads, "
                f"got {self.num_heads_tp}. Padding from values > {padding} is not supported."
            )
        else:  # SM90
            padding = ((self.num_heads_tp + 63) // 64) * 64  # multiple of 64

        if self.num_heads_tp != padding:
            logger.warning_once(
                f"Padding num_heads from {self.num_heads_tp} to {padding} "
                f"due to FlashMLA sparse attention kernel requirement",
                key="sparse_mla_padding_warning",
            )

            # Create padded tensor with zeros for extra heads
            q_padded = q_concat.new_empty((num_tokens, padding, q_concat.shape[2]))
            q_padded[:, : self.num_heads_tp, :] = q_concat
            q_concat = q_padded

        # Convert indices and return all-layer KV pool
        # Note: underlying pool is layer-interleaved:
        # [num_blocks, num_layers, kv_factor, tokens_per_block, num_kv_heads, head_dim]
        # to avoid reshape(copy) per-layer KV cache, we return all-layer KV pool w/ topk indices adjusted
        # by stride_factor=num_layers*tokens_per_block
        topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
            topk_indices,
            attn_metadata,
            layer_idx=self.layer_idx,
            is_generation=is_generation,
        )
        topk_indices_pool = topk_indices_pool.view(num_tokens, 1, -1)
        if flash_mla_sparse_fwd is not None:
            attn_out_latent = flash_mla_sparse_fwd(
                q_concat, kv_cache_pool, topk_indices_pool, self.softmax_scale
            )[0]
        else:
            raise RuntimeError(
                "flash_mla_sparse_fwd not available. Please ensure FlashMLA module is built."
            )

        # [seq, num_heads, kv_lora_rank], account for padding
        attn_out_latent = attn_out_latent[:, : self.num_heads_tp, :]
        attn_out_latent = attn_out_latent.view([-1, self.num_heads_tp, self.kv_lora_rank])
        if self.num_heads_tp != padding:
            attn_out_latent = attn_out_latent.contiguous()

        assert (
            attn_out_latent.shape[0] == q.shape[0] and attn_out_latent.shape[1] == self.num_heads_tp
        )

        attn_output = output.view([num_tokens, self.num_heads_tp, self.v_head_dim])

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
            fp8_block_scaling_bmm_out(
                attn_out_latent,
                self.v_b_proj,
                self.v_b_proj_scale,
                attn_output.transpose(0, 1),
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
        attn_output = self.create_output(hidden_states, attn_metadata.num_contexts)
        assert self.register_to_config
        proj_outputs = torch.ops.trtllm.fused_mla_dsa_proj(
            hidden_states, position_ids, self.layer_idx_str
        )
        q, compressed_kv, k_pe, latent_cache = proj_outputs[:4]
        indexer_intermediates = proj_outputs[4:]
        torch.ops.trtllm.fused_mla_dsa_attn_inplace(
            q,
            compressed_kv,
            k_pe,
            latent_cache,
            indexer_intermediates,
            position_ids,
            self.layer_idx_str,
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


@torch.library.custom_op("trtllm::fused_mla_dsa_proj", mutates_args=())
def mla_dsa_proj(
    hidden_states: torch.Tensor,
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
) -> List[torch.Tensor]:
    """Token-wise projections for DSA MLA (CUDA-graph-capturable).

    Runs kv_a_proj, layernorms, q_b_proj, and conditionally
    indexer.pre_indexer_proj (FP8/FP4 quantize, weight scaling).  Does NOT
    update the indexer k cache — that happens in Op 2 (mla_dsa_attn_inplace)
    because the scatter kernel accesses batch-specific metadata.

    Returns [q, compressed_kv, k_pe, latent_cache] when the short-MHA path
    handles all tokens, or [q, compressed_kv, k_pe, latent_cache, q_fp8,
    k_fp8, k_scale, weights, q_scale] when the indexer runs.  Under torch
    compile, _should_use_short_mha returns False so the result is always
    length 9, keeping control flow straight-line for CUDA graph capture.
    The trailing q_scale is only consumed by the FP4 dispatch; the FP8
    path ignores it in forward_dsa_attn.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    return mla_layer.forward_dsa_proj(position_ids, hidden_states, metadata)


@torch.library.custom_op("trtllm::fused_mla_dsa_attn_inplace", mutates_args=("output",))
def mla_dsa_attn_inplace(
    q: torch.Tensor,
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    indexer_intermediates: List[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    layer_idx: str,
    output: torch.Tensor,
) -> None:
    """Batch-structure-dependent attention dispatch for DSA MLA.

    indexer_intermediates is [q_fp8, k_fp8, k_scale, weights, q_scale] when
    the indexer ran in Op 1, or [] when short-MHA handled all tokens. The
    trailing q_scale is only consumed by the FP4 dispatch; the FP8 path
    ignores it. Runs sparse_attn_indexer then dispatches context/generation
    attention. This op is excluded from CUDA graph capture.
    """
    metadata, mla_layer = extract_extra_attrs(layer_idx, "mla")
    mla_layer.forward_dsa_attn(
        q, compressed_kv, k_pe, latent_cache, indexer_intermediates, position_ids, metadata, output
    )
