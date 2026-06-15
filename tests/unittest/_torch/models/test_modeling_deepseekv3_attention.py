# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionBackend,
    AttentionInputType,
    AttentionMetadata,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    DSACacheManager,
    transform_local_topk_and_prepare_pool_view,
)
from tensorrt_llm._torch.attention_backend.utils import get_attention_backend
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3_mla_pytorch import (
    dsv3_mla_context_pytorch,
    dsv3_mla_decode_pytorch,
)
from tensorrt_llm._utils import get_sm_version, str_dtype_to_binding, torch_dtype_to_str
from tensorrt_llm.bindings.executor import KvCacheConfig
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_LOCAL_NUM_HEADS = 8
_NUM_KV_HEADS = 1
_Q_LORA_RANK = 2048
_KV_LORA_RANK = 512
_QK_NOPE_HEAD_DIM = 192
_QK_ROPE_HEAD_DIM = 64
_V_HEAD_DIM = _KV_LORA_RANK
_HEAD_DIM = _KV_LORA_RANK + _QK_ROPE_HEAD_DIM
_QK_HEAD_DIM = _QK_NOPE_HEAD_DIM + _QK_ROPE_HEAD_DIM
_HIDDEN_SIZE = 6144
_PREDICTED_TOKENS_PER_SEQ = 4
_CONTEXT_SEQUENCE_LENGTH = 64
_BENCH_CONTEXT_SEQUENCE_LENGTH = 1070
_DECODE_NUM_TOKENS = _PREDICTED_TOKENS_PER_SEQ
_INDEX_TOPK = 2048
_TOKENS_PER_BLOCK = 32
_MAX_POSITION_EMBEDDINGS = 4096
_CONTEXT_REFERENCE_ATOL = 6.9e-3


class _Config(SimpleNamespace):
    def update(self, values: dict[str, object]) -> None:
        self.__dict__.update(values)


def _require_glm5_context_attention_runtime() -> None:
    if not torch.cuda.is_available():
        pytest.skip("GLM-5 context attention test requires CUDA")
    if get_sm_version() not in (90, 100, 103, 120):
        pytest.skip("FP8 MLA context attention is only enabled on SM90/SM100/SM103/SM120")


def _make_glm5_config() -> _Config:
    return _Config(
        hidden_size=_HIDDEN_SIZE,
        num_attention_heads=64,
        num_key_value_heads=1,
        q_lora_rank=_Q_LORA_RANK,
        kv_lora_rank=_KV_LORA_RANK,
        qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
        qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
        v_head_dim=256,
        max_position_embeddings=_MAX_POSITION_EMBEDDINGS,
        model_type="deepseek_v3",
        rope_theta=10000.0,
        rope_scaling={
            "beta_fast": 32,
            "beta_slow": 1,
            "factor": 40.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "original_max_position_embeddings": 4096,
            "type": "yarn",
        },
        rms_norm_eps=1e-6,
        torch_dtype=torch.bfloat16,
    )


def _compute_q_scaling(config: _Config) -> float:
    scale = config.rope_scaling["factor"]
    mscale_all_dim = config.rope_scaling["mscale_all_dim"]
    mscale = 1.0 if scale <= 1 else 0.1 * mscale_all_dim * math.log(scale) + 1.0
    return 1.0 / (mscale * mscale)


def _get_scalar(scale: torch.Tensor | None, device: torch.device) -> torch.Tensor:
    if scale is None:
        return torch.ones((), dtype=torch.float32, device=device)
    return scale.reshape(-1)[0].to(dtype=torch.float32, device=device)


def _set_kv_cache_scale(attention: AttentionBackend, dequant_scale: float) -> None:
    device = attention.kv_scale_quant_orig.device
    attention.kv_scale_quant_orig = torch.tensor(
        [dequant_scale], dtype=torch.float32, device=device
    )
    attention.kv_scale_orig_quant = torch.tensor(
        [1.0 / dequant_scale], dtype=torch.float32, device=device
    )


def _build_causal_topk(
    device: torch.device,
    context_sequence_length: int = _CONTEXT_SEQUENCE_LENGTH,
    cached_sequence_length: int = 0,
) -> torch.Tensor:
    topk_indices = torch.full(
        (context_sequence_length, _INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for token_idx in range(context_sequence_length):
        max_visible_position = cached_sequence_length + token_idx
        valid_len = min(max_visible_position + 1, _INDEX_TOPK)
        topk_indices[token_idx, :valid_len] = torch.arange(
            max_visible_position + 1 - valid_len,
            max_visible_position + 1,
            dtype=torch.int32,
            device=device,
        )
    return topk_indices


def _build_decode_topk(
    device: torch.device,
    context_sequence_length: int = _CONTEXT_SEQUENCE_LENGTH,
) -> torch.Tensor:
    topk_indices = torch.full(
        (_DECODE_NUM_TOKENS, _INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for token_idx in range(_DECODE_NUM_TOKENS):
        valid_len = min(context_sequence_length + token_idx + 1, _INDEX_TOPK)
        topk_indices[token_idx, :valid_len] = torch.arange(
            context_sequence_length + token_idx + 1 - valid_len,
            context_sequence_length + token_idx + 1,
            dtype=torch.int32,
            device=device,
        )
    return topk_indices


def _torch_context_attention_reference(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    attention: AttentionBackend,
    metadata: AttentionMetadata,
) -> torch.Tensor:
    """PyTorch mirror of TRTLLM sparse FP8 MLA context attention."""
    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        metadata,
        layer_idx=attention.get_local_layer_idx(metadata),
        is_generation=False,
    )
    kv_cache_pool_ref = kv_cache_pool.clone()
    return dsv3_mla_context_pytorch(
        fused_q,
        q_pe,
        latent_cache,
        topk_indices,
        topk_indices_pool,
        kv_cache_pool_ref,
        metadata,
        attention,
    )


def _assert_context_attention_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    diff = (actual - expected).abs().float()
    message = (
        f"max_abs={diff.max().item():.8f}, mean_abs={diff.mean().item():.8f}, "
        f"p99_abs={diff.flatten().quantile(0.99).item():.8f}"
    )
    try:
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=_CONTEXT_REFERENCE_ATOL,
            msg=message,
        )
    except AssertionError as err:
        raise AssertionError(message) from err


def _build_decode_projected_query(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_for_split = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _LOCAL_NUM_HEADS,
            _QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    q_nope, _ = q_for_split.split([_QK_NOPE_HEAD_DIM, _QK_ROPE_HEAD_DIM], dim=-1)
    k_b_proj_trans = torch.zeros(
        _LOCAL_NUM_HEADS,
        _KV_LORA_RANK,
        _QK_NOPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    dim_indices = torch.arange(_KV_LORA_RANK, device=device)
    reduction_indices = dim_indices % _QK_NOPE_HEAD_DIM
    k_b_proj_trans[:, dim_indices, reduction_indices] = 1

    fused_q = torch.zeros(
        _DECODE_NUM_TOKENS,
        _LOCAL_NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    fused_q[..., :_KV_LORA_RANK] = q_nope.index_select(dim=-1, index=reduction_indices)
    return fused_q, q_nope, k_b_proj_trans


def _custom_context_attention(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    attention: AttentionBackend,
    metadata: AttentionMetadata,
) -> torch.Tensor:
    attention._ensure_rope_table_size(metadata.max_seq_len)
    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        metadata,
        layer_idx=attention.get_local_layer_idx(metadata),
        is_generation=False,
    )
    return torch.ops.trtllm.dsv3_fused_mla_context(
        fused_q,
        q_pe,
        latent_cache,
        topk_indices_pool,
        topk_indices,
        kv_cache_pool,
        attention.rotary_cos_sin,
        metadata.ctx_cached_token_indptr,
        metadata.kv_cache_block_offsets,
        metadata.kv_cache_manager.kv_cache_pool_pointers,
        metadata.kv_cache_manager.kv_cache_pool_mapping,
        attention.kv_scale_orig_quant,
        attention.kv_scale_quant_orig,
        attention.get_local_layer_idx(metadata),
        metadata.kv_cache_manager.tokens_per_block,
        int(attention.quant_mode),
        float(attention.q_scaling),
    )


def _build_attention_and_metadata(
    layer_idx: int = 0,
    num_layers: int = 1,
    context_sequence_length: int = _CONTEXT_SEQUENCE_LENGTH,
    cached_sequence_length: int = 0,
) -> tuple[AttentionBackend, AttentionMetadata, _Config]:
    config = _make_glm5_config()
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=_INDEX_TOPK,
        skip_indexer_for_short_seqs=False,
    )
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=quant_config,
        sparse_attention_config=sparse_config,
    )
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(config),
        is_neox=False,
    )
    attn_cls = get_attention_backend("TRTLLM", sparse_config)
    attention = attn_cls(
        layer_idx=layer_idx,
        num_heads=_LOCAL_NUM_HEADS,
        head_dim=_HEAD_DIM,
        num_kv_heads=_NUM_KV_HEADS,
        quant_config=quant_config,
        q_scaling=_compute_q_scaling(config),
        pos_embd_params=pos_embd_params,
        mla_params=MLAParams(
            q_lora_rank=_Q_LORA_RANK,
            kv_lora_rank=_KV_LORA_RANK,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
            v_head_dim=_V_HEAD_DIM,
            rope_append=True,
            predicted_tokens_per_seq=_PREDICTED_TOKENS_PER_SEQ,
            hidden_size=_HIDDEN_SIZE,
        ),
        sparse_attention_config=sparse_config,
        dtype=torch.bfloat16,
    )
    max_seq_len = cached_sequence_length + context_sequence_length
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(
            max_tokens=max(
                _TOKENS_PER_BLOCK * 4,
                math.ceil(max_seq_len / _TOKENS_PER_BLOCK) * _TOKENS_PER_BLOCK,
            ),
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    kv_cache_manager.add_dummy_requests([0], [max_seq_len])

    metadata = attn_cls.Metadata(
        seq_lens=torch.tensor([context_sequence_length], dtype=torch.int),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=1,
        prompt_lens=[context_sequence_length],
        max_num_tokens=context_sequence_length,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[cached_sequence_length],
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    metadata.prepare()
    return attention, metadata, config


def _get_context_kv_pool_rows(
    attention: AttentionBackend,
    metadata: AttentionMetadata,
    token_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    metadata._ensure_pool_view_cached()
    layer_idx = attention.get_local_layer_idx(metadata)
    tokens_per_block = metadata._cached_tokens_per_block
    stride_factor = metadata._cached_stride_factor
    token_positions = token_positions.to(
        device=metadata._cached_block_table_ctx.device,
        dtype=torch.long,
    )
    block_indices = token_positions // tokens_per_block
    inblock_offsets = token_positions % tokens_per_block + layer_idx * tokens_per_block
    block_bases = metadata._cached_block_table_ctx[0, block_indices].to(torch.long)
    pool_rows = block_bases * stride_factor + inblock_offsets
    return pool_rows, metadata._cached_pool_view


def _seed_context_cached_kv_cache(
    attention: AttentionBackend,
    metadata: AttentionMetadata,
    device: torch.device,
    cached_sequence_length: int,
) -> torch.Tensor:
    token_positions = torch.arange(
        cached_sequence_length,
        dtype=torch.long,
        device=device,
    )
    pool_rows, kv_cache_pool = _get_context_kv_pool_rows(
        attention,
        metadata,
        token_positions,
    )
    kv_scale_orig_quant = _get_scalar(attention.kv_scale_orig_quant, device)
    history_cache = (
        torch.randn(
            cached_sequence_length,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    history_cache_fp8 = (history_cache.float() * kv_scale_orig_quant).to(torch.float8_e4m3fn)
    kv_cache_pool.view(torch.uint8).index_copy_(
        0,
        pool_rows,
        history_cache_fp8.view(torch.uint8).unsqueeze(1),
    )
    return kv_cache_pool


def _build_decode_attention_and_metadata(
    layer_idx: int = 0,
    num_layers: int = 1,
    context_sequence_length: int = _CONTEXT_SEQUENCE_LENGTH,
) -> tuple[AttentionBackend, AttentionMetadata, _Config]:
    config = _make_glm5_config()
    sparse_config = DeepSeekSparseAttentionConfig(
        index_n_heads=64,
        index_head_dim=128,
        index_topk=_INDEX_TOPK,
        skip_indexer_for_short_seqs=False,
    )
    mapping = Mapping(world_size=1, tp_size=1, rank=0)
    quant_config = QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8)
    model_config = ModelConfig(
        pretrained_config=config,
        mapping=mapping,
        quant_config=quant_config,
        sparse_attention_config=sparse_config,
    )
    pos_embd_params = PositionalEmbeddingParams(
        type=PositionEmbeddingType.yarn,
        rope=RopeParams.from_config(config),
        is_neox=False,
    )
    attn_cls = get_attention_backend("TRTLLM", sparse_config)
    attention = attn_cls(
        layer_idx=layer_idx,
        num_heads=_LOCAL_NUM_HEADS,
        head_dim=_HEAD_DIM,
        num_kv_heads=_NUM_KV_HEADS,
        quant_config=quant_config,
        q_scaling=_compute_q_scaling(config),
        pos_embd_params=pos_embd_params,
        mla_params=MLAParams(
            q_lora_rank=_Q_LORA_RANK,
            kv_lora_rank=_KV_LORA_RANK,
            qk_rope_head_dim=_QK_ROPE_HEAD_DIM,
            qk_nope_head_dim=_QK_NOPE_HEAD_DIM,
            v_head_dim=_V_HEAD_DIM,
            rope_append=True,
            predicted_tokens_per_seq=_PREDICTED_TOKENS_PER_SEQ,
            hidden_size=_HIDDEN_SIZE,
        ),
        sparse_attention_config=sparse_config,
        dtype=torch.bfloat16,
    )
    max_seq_len = context_sequence_length + _DECODE_NUM_TOKENS
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(
            max_tokens=max(
                _TOKENS_PER_BLOCK * 4,
                math.ceil(max_seq_len / _TOKENS_PER_BLOCK) * _TOKENS_PER_BLOCK,
            ),
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    kv_cache_manager.add_dummy_requests([0], [max_seq_len])

    metadata = attn_cls.Metadata(
        seq_lens=torch.tensor([_DECODE_NUM_TOKENS], dtype=torch.int),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=0,
        prompt_lens=[context_sequence_length],
        max_num_tokens=_DECODE_NUM_TOKENS,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(
            use_cache=True,
            num_cached_tokens_per_seq=[context_sequence_length],
        ),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    metadata.prepare()
    return attention, metadata, config


def _seed_decode_history_kv_cache(
    attention: AttentionBackend,
    metadata: AttentionMetadata,
    device: torch.device,
    context_sequence_length: int = _CONTEXT_SEQUENCE_LENGTH,
) -> torch.Tensor:
    history_indices = torch.arange(
        context_sequence_length,
        dtype=torch.int32,
        device=device,
    ).expand(_DECODE_NUM_TOKENS, -1)
    history_pool_indices, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        history_indices,
        metadata,
        layer_idx=attention.get_local_layer_idx(metadata),
        is_generation=True,
    )
    kv_scale_orig_quant = _get_scalar(attention.kv_scale_orig_quant, device)
    history_cache = (
        torch.randn(
            context_sequence_length,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    history_cache_fp8 = (history_cache.float() * kv_scale_orig_quant).to(torch.float8_e4m3fn)
    kv_cache_pool.view(torch.uint8).index_copy_(
        0,
        history_pool_indices[0].to(torch.long),
        history_cache_fp8.view(torch.uint8).unsqueeze(1),
    )
    return kv_cache_pool


def _custom_decode_attention(
    fused_q: torch.Tensor,
    q_nope: torch.Tensor,
    k_b_proj_trans: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_indices_pool: torch.Tensor,
    kv_cache_pool: torch.Tensor,
    attention: AttentionBackend,
    metadata: AttentionMetadata,
    cu_q_seqlens: torch.Tensor,
    cu_kv_seqlens: torch.Tensor,
    fmha_scheduler_counter: torch.Tensor,
    mla_bmm1_scale: torch.Tensor,
    mla_bmm2_scale: torch.Tensor,
    quant_q_buffer: torch.Tensor,
    spec_decoding_packed_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    _ = cu_q_seqlens, cu_kv_seqlens, fmha_scheduler_counter
    attention._ensure_rope_table_size(metadata.max_seq_len)
    return torch.ops.trtllm.dsv3_fused_mla_generation(
        fused_q,
        q_nope,
        k_b_proj_trans,
        q_pe,
        latent_cache,
        attention.rotary_cos_sin,
        metadata.kv_lens_cuda_runtime,
        metadata.kv_cache_block_offsets,
        metadata.kv_cache_manager.kv_cache_pool_pointers,
        metadata.kv_cache_manager.kv_cache_pool_mapping,
        topk_indices,
        topk_indices_pool,
        kv_cache_pool,
        attention.kv_scale_orig_quant,
        attention.kv_scale_quant_orig,
        quant_q_buffer,
        mla_bmm1_scale,
        mla_bmm2_scale,
        spec_decoding_packed_mask,
        attention.get_local_layer_idx(metadata),
        metadata.kv_cache_manager.tokens_per_block,
        int(attention.quant_mode),
        float(attention.q_scaling),
    )


def test_glm5_fp8_context_attention_matches_pytorch_reference() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    attention, metadata, _ = _build_attention_and_metadata()
    fused_q = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _LOCAL_NUM_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    q_pe = fused_q[..., _KV_LORA_RANK:].contiguous()
    fused_q = fused_q.reshape(_CONTEXT_SEQUENCE_LENGTH, _LOCAL_NUM_HEADS * _HEAD_DIM)
    latent_cache = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    topk_indices = _build_causal_topk(device)

    with torch.inference_mode():
        expected = _torch_context_attention_reference(
            fused_q,
            q_pe,
            latent_cache,
            topk_indices,
            attention,
            metadata,
        )
        actual = attention.forward(
            fused_q.clone(),
            None,
            None,
            metadata,
            attention_input_type=AttentionInputType.context_only,
            latent_cache=latent_cache.clone(),
            q_pe=q_pe.clone(),
            topk_indices=topk_indices,
            is_generation=False,
        )

    _assert_context_attention_close(actual, expected)


def test_glm5_fp8_decode_attention_matches_pytorch_reference() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(234)
    torch.cuda.manual_seed(234)

    attention, metadata, _ = _build_decode_attention_and_metadata()
    kv_cache_pool = _seed_decode_history_kv_cache(attention, metadata, device)
    fused_q = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _LOCAL_NUM_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    q_pe = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _LOCAL_NUM_HEADS,
            _QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    latent_cache = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    topk_indices = _build_decode_topk(device)

    num_seqs = metadata.kv_lens_cuda_runtime.size(0)
    cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
    mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
    quant_q_buffer = torch.empty(
        _DECODE_NUM_TOKENS,
        _LOCAL_NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.uint8,
        device=device,
    )

    with torch.inference_mode():
        attention.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )
        topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
            topk_indices,
            metadata,
            layer_idx=attention.get_local_layer_idx(metadata),
            is_generation=True,
        )
        expected = attention.forward(
            fused_q.reshape(_DECODE_NUM_TOKENS, _LOCAL_NUM_HEADS * _HEAD_DIM),
            None,
            None,
            metadata,
            attention_input_type=AttentionInputType.generation_only,
            latent_cache=latent_cache,
            q_pe=q_pe,
            topk_indices=topk_indices,
            is_generation=True,
            cu_q_seqlens=cu_q_seqlens,
            cu_kv_seqlens=cu_kv_seqlens,
            fmha_scheduler_counter=fmha_scheduler_counter,
            mla_bmm1_scale=mla_bmm1_scale,
            mla_bmm2_scale=mla_bmm2_scale,
            quant_q_buffer=quant_q_buffer,
        )
        actual = dsv3_mla_decode_pytorch(
            quant_q_buffer,
            topk_indices,
            topk_indices_pool,
            kv_cache_pool,
            metadata.kv_lens_cuda_runtime,
            mla_bmm1_scale,
            mla_bmm2_scale,
            metadata.spec_decoding_packed_mask,
        )

    _assert_context_attention_close(actual, expected)


def test_glm5_fp8_decode_custom_op_matches_pytorch_reference_with_spec_mask() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(345)
    torch.cuda.manual_seed(345)

    attention, metadata, _ = _build_decode_attention_and_metadata(
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    _seed_decode_history_kv_cache(
        attention,
        metadata,
        device,
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    fused_q, q_nope, k_b_proj_trans = _build_decode_projected_query(device)
    q_pe = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _LOCAL_NUM_HEADS,
            _QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    latent_cache = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    topk_indices = _build_decode_topk(
        device,
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    spec_decoding_packed_mask = torch.tensor(
        [[[1], [1], [5], [13]]],
        dtype=torch.int32,
        device=device,
    )

    num_seqs = metadata.kv_lens_cuda_runtime.size(0)
    cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
    mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
    quant_q_buffer = torch.empty(
        _DECODE_NUM_TOKENS,
        _LOCAL_NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.uint8,
        device=device,
    )

    with torch.inference_mode():
        attention.mla_rope_generation(
            fused_q,
            q_pe,
            latent_cache,
            metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )
        topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
            topk_indices,
            metadata,
            layer_idx=attention.get_local_layer_idx(metadata),
            is_generation=True,
        )
        expected = dsv3_mla_decode_pytorch(
            quant_q_buffer,
            topk_indices,
            topk_indices_pool,
            kv_cache_pool,
            metadata.kv_lens_cuda_runtime,
            mla_bmm1_scale,
            mla_bmm2_scale,
            spec_decoding_packed_mask,
        )
        actual = _custom_decode_attention(
            fused_q,
            q_nope,
            k_b_proj_trans,
            q_pe,
            latent_cache,
            topk_indices,
            topk_indices_pool,
            kv_cache_pool,
            attention,
            metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
            spec_decoding_packed_mask,
        )

    _assert_context_attention_close(actual, expected)


def test_glm5_fp8_decode_custom_op_cuda_graph_replay_infers_stale_sequence_length() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(567)
    torch.cuda.manual_seed(567)

    attention, metadata, _ = _build_decode_attention_and_metadata(
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    _seed_decode_history_kv_cache(
        attention,
        metadata,
        device,
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    fused_q, q_nope, k_b_proj_trans = _build_decode_projected_query(device)
    q_pe = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _LOCAL_NUM_HEADS,
            _QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    latent_cache = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    topk_indices = _build_decode_topk(
        device,
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )

    num_seqs = metadata.kv_lens_cuda_runtime.size(0)
    cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
    mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
    quant_q_buffer = torch.empty(
        _DECODE_NUM_TOKENS,
        _LOCAL_NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.uint8,
        device=device,
    )

    attention.mla_rope_generation(
        fused_q,
        q_pe,
        latent_cache,
        metadata,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
    )
    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        metadata,
        layer_idx=attention.get_local_layer_idx(metadata),
        is_generation=True,
    )
    _custom_decode_attention(
        fused_q,
        q_nope,
        k_b_proj_trans,
        q_pe,
        latent_cache,
        topk_indices,
        topk_indices_pool,
        kv_cache_pool,
        attention,
        metadata,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
    )
    torch.cuda.synchronize()

    metadata.kv_lens_cuda_runtime.fill_(_DECODE_NUM_TOKENS)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _custom_decode_attention(
            fused_q,
            q_nope,
            k_b_proj_trans,
            q_pe,
            latent_cache,
            topk_indices,
            topk_indices_pool,
            kv_cache_pool,
            attention,
            metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )

    graph.replay()
    expected = dsv3_mla_decode_pytorch(
        quant_q_buffer,
        topk_indices,
        topk_indices_pool,
        kv_cache_pool,
        metadata.kv_lens_cuda_runtime,
        mla_bmm1_scale,
        mla_bmm2_scale,
        metadata.spec_decoding_packed_mask,
    )

    _assert_context_attention_close(actual, expected)


def test_glm5_fp8_decode_custom_op_handles_large_pool_indices() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(678)
    torch.cuda.manual_seed(678)

    overflow_pool_row = 7_680_000
    pool_rows = overflow_pool_row + torch.arange(
        _DECODE_NUM_TOKENS,
        dtype=torch.int32,
        device=device,
    )
    required_bytes = (overflow_pool_row + _DECODE_NUM_TOKENS) * _HEAD_DIM
    free_bytes, _ = torch.cuda.mem_get_info(device)
    if free_bytes < required_bytes + (1 << 30):
        pytest.skip("large-pool-index regression test needs about 5.5 GiB of free GPU memory")

    attention, metadata, _ = _build_decode_attention_and_metadata()
    topk_indices = torch.arange(
        _DECODE_NUM_TOKENS,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(1)
    topk_indices_pool = pool_rows.unsqueeze(1)
    kv_cache_pool = torch.zeros(
        (overflow_pool_row + _DECODE_NUM_TOKENS, 1, _HEAD_DIM),
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    kv_rows = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    ).to(torch.float8_e4m3fn)
    kv_cache_pool[pool_rows.to(torch.long), 0, :] = kv_rows

    fused_q = torch.zeros(
        _DECODE_NUM_TOKENS,
        _LOCAL_NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    q_nope = torch.zeros(
        _DECODE_NUM_TOKENS,
        _LOCAL_NUM_HEADS,
        _QK_NOPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    k_b_proj_trans = torch.zeros(
        _LOCAL_NUM_HEADS,
        _KV_LORA_RANK,
        _QK_NOPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    q_pe = torch.zeros(
        _DECODE_NUM_TOKENS,
        _LOCAL_NUM_HEADS,
        _QK_ROPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    latent_cache = torch.zeros(
        _DECODE_NUM_TOKENS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    quant_q_buffer = torch.zeros_like(fused_q, dtype=torch.uint8)
    cu_q_seqlens = torch.empty(2, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(2, dtype=torch.int32, device=device)
    fmha_scheduler_counter = torch.empty(1, dtype=torch.uint32, device=device)
    mla_bmm1_scale = torch.tensor([1.0, 1.0], dtype=torch.float32, device=device)
    mla_bmm2_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    metadata.kv_lens_cuda_runtime.fill_(_DECODE_NUM_TOKENS)

    with torch.inference_mode():
        actual = _custom_decode_attention(
            fused_q,
            q_nope,
            k_b_proj_trans,
            q_pe,
            latent_cache,
            topk_indices,
            topk_indices_pool,
            kv_cache_pool,
            attention,
            metadata,
            cu_q_seqlens,
            cu_kv_seqlens,
            fmha_scheduler_counter,
            mla_bmm1_scale,
            mla_bmm2_scale,
            quant_q_buffer,
        )

    output_scale = _get_scalar(attention.kv_scale_quant_orig, device)
    expected = (
        (kv_rows[:, :_KV_LORA_RANK].to(torch.float32) * output_scale)
        .unsqueeze(1)
        .expand(_DECODE_NUM_TOKENS, _LOCAL_NUM_HEADS, _KV_LORA_RANK)
        .reshape(_DECODE_NUM_TOKENS, _LOCAL_NUM_HEADS * _KV_LORA_RANK)
        .to(torch.bfloat16)
    )
    _assert_context_attention_close(actual, expected)


def test_glm5_fp8_context_custom_op_matches_pytorch_reference_with_strided_q_pe() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(321)
    torch.cuda.manual_seed(321)

    attention, metadata, _ = _build_attention_and_metadata()
    _set_kv_cache_scale(attention, dequant_scale=0.5)
    fused_q = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _LOCAL_NUM_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    q_for_split = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _LOCAL_NUM_HEADS,
            _QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    _, q_pe = q_for_split.split([_QK_NOPE_HEAD_DIM, _QK_ROPE_HEAD_DIM], dim=-1)
    assert q_pe.stride() == (
        _LOCAL_NUM_HEADS * _QK_HEAD_DIM,
        _QK_HEAD_DIM,
        1,
    )
    assert not q_pe.is_contiguous()

    latent_cache = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    topk_indices = _build_causal_topk(device)

    with torch.inference_mode():
        expected = _torch_context_attention_reference(
            fused_q,
            q_pe,
            latent_cache,
            topk_indices,
            attention,
            metadata,
        )
        actual_latent_cache = latent_cache.clone()
        latent_cache_before = actual_latent_cache.clone()
        actual = _custom_context_attention(
            fused_q.clone(),
            q_pe,
            actual_latent_cache,
            topk_indices,
            attention,
            metadata,
        )

    _assert_context_attention_close(actual, expected)
    torch.testing.assert_close(actual_latent_cache, latent_cache_before, rtol=0.0, atol=0.0)


def test_glm5_fp8_context_custom_op_matches_pytorch_reference_with_cached_prefix() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(654)
    torch.cuda.manual_seed(654)

    cached_sequence_length = 48
    context_sequence_length = 16
    attention, metadata, _ = _build_attention_and_metadata(
        context_sequence_length=context_sequence_length,
        cached_sequence_length=cached_sequence_length,
    )
    _set_kv_cache_scale(attention, dequant_scale=0.5)
    _seed_context_cached_kv_cache(
        attention,
        metadata,
        device,
        cached_sequence_length,
    )
    fused_q = (
        torch.randn(
            context_sequence_length,
            _LOCAL_NUM_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    q_pe = (
        torch.randn(
            context_sequence_length,
            _LOCAL_NUM_HEADS,
            _QK_ROPE_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    latent_cache = (
        torch.randn(
            context_sequence_length,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    topk_indices = _build_causal_topk(
        device,
        context_sequence_length=context_sequence_length,
        cached_sequence_length=cached_sequence_length,
    )

    with torch.inference_mode():
        expected = _torch_context_attention_reference(
            fused_q,
            q_pe,
            latent_cache,
            topk_indices,
            attention,
            metadata,
        )
        actual_latent_cache = latent_cache.clone()
        latent_cache_before = actual_latent_cache.clone()
        actual = _custom_context_attention(
            fused_q.clone(),
            q_pe,
            actual_latent_cache,
            topk_indices,
            attention,
            metadata,
        )

    _assert_context_attention_close(actual, expected)
    torch.testing.assert_close(actual_latent_cache, latent_cache_before, rtol=0.0, atol=0.0)


def test_glm5_fp8_context_custom_op_matches_pytorch_reference_for_nonzero_layer() -> None:
    _require_glm5_context_attention_runtime()
    device = torch.device("cuda")
    torch.manual_seed(456)
    torch.cuda.manual_seed(456)

    attention, metadata, _ = _build_attention_and_metadata(layer_idx=1, num_layers=2)
    fused_q = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _LOCAL_NUM_HEADS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    q_for_split = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _LOCAL_NUM_HEADS,
            _QK_HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    _, q_pe = q_for_split.split([_QK_NOPE_HEAD_DIM, _QK_ROPE_HEAD_DIM], dim=-1)
    latent_cache = (
        torch.randn(
            _CONTEXT_SEQUENCE_LENGTH,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    topk_indices = _build_causal_topk(device)

    with torch.inference_mode():
        expected = _torch_context_attention_reference(
            fused_q,
            q_pe,
            latent_cache,
            topk_indices,
            attention,
            metadata,
        )
        actual_latent_cache = latent_cache.clone()
        latent_cache_before = actual_latent_cache.clone()
        actual = _custom_context_attention(
            fused_q.clone(),
            q_pe,
            actual_latent_cache,
            topk_indices,
            attention,
            metadata,
        )

    _assert_context_attention_close(actual, expected)
    torch.testing.assert_close(actual_latent_cache, latent_cache_before, rtol=0.0, atol=0.0)
