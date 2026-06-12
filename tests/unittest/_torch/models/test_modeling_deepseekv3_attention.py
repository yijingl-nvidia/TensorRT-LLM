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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


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


def _rope_cos_sin_for_rotation(
    rope_cos_sin: torch.Tensor,
    max_positions: int,
) -> torch.Tensor:
    if rope_cos_sin.dim() == 2 and rope_cos_sin.shape[0] == 1:
        rope_cos_sin = rope_cos_sin.reshape(max_positions, -1)
    if rope_cos_sin.dim() == 2:
        rope_cos_sin = rope_cos_sin.reshape(rope_cos_sin.shape[0], -1, 2)
        return rope_cos_sin.transpose(-2, -1)
    return rope_cos_sin


def _rotate_context_tensors(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    cached_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = fused_q.shape[0]
    fused_q = fused_q.clone().view(num_tokens, _LOCAL_NUM_HEADS, _HEAD_DIM)
    latent_cache = latent_cache.clone()

    rope_cos_sin = _rope_cos_sin_for_rotation(rope_cos_sin, _MAX_POSITION_EMBEDDINGS)
    cos, sin = rope_cos_sin[cached_len : cached_len + num_tokens].chunk(2, dim=-2)

    q_pe = q_pe.clone()
    q_pe = q_pe.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
    q_pe = ((q_pe * cos) + (_rotate_half(q_pe) * sin)).to(dtype=fused_q.dtype)
    q_pe = q_pe.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
    fused_q[..., _KV_LORA_RANK:] = q_pe

    k_pe = latent_cache[:, _KV_LORA_RANK:].unsqueeze(-2)
    k_pe = k_pe.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
    k_pe = ((k_pe * cos) + (_rotate_half(k_pe) * sin)).to(dtype=latent_cache.dtype)
    k_pe = k_pe.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
    latent_cache[:, _KV_LORA_RANK:] = k_pe.squeeze(-2)

    return fused_q, latent_cache


def _build_causal_topk(device: torch.device) -> torch.Tensor:
    topk_indices = torch.full(
        (_CONTEXT_SEQUENCE_LENGTH, _INDEX_TOPK),
        -1,
        dtype=torch.int32,
        device=device,
    )
    for token_idx in range(_CONTEXT_SEQUENCE_LENGTH):
        valid_len = min(token_idx + 1, _INDEX_TOPK)
        topk_indices[token_idx, :valid_len] = torch.arange(
            token_idx + 1 - valid_len,
            token_idx + 1,
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
    """PyTorch mirror of TRTLLM sparse FP8 MLA context attention.

    This mirrors the trusted TRTLLM DSA path, not the WIP fused context kernel:
    1. `mla_rope_context` applies GPT-J/interleaved RoPE to BF16 Q/K.
    2. The rotated latent K/V rows are written to the paged KV cache as FP8.
    3. The rotated Q rows are quantized to FP8 for FP8 context MLA.
    4. Sparse FMHA uses DSA's global pool indices, FP8 Q/K/V dequant scales,
       and `1 / (q_scaling * sqrt(qk_nope + qk_rope))` score scaling.
    """
    num_tokens = fused_q.shape[0]
    device = fused_q.device
    cached_len = 0
    if hasattr(metadata, "host_ctx_cached_token_indptr"):
        host_indptr = metadata.host_ctx_cached_token_indptr
        cached_len = int((host_indptr[1] - host_indptr[0]).item())

    fused_q, latent_cache = _rotate_context_tensors(
        fused_q,
        q_pe,
        latent_cache,
        attention.rotary_cos_sin,
        cached_len,
    )

    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        metadata,
        layer_idx=attention.get_local_layer_idx(metadata),
        is_generation=False,
    )
    token_positions = torch.arange(num_tokens, dtype=torch.int32, device=device)[:, None]
    token_pool_rows, _ = transform_local_topk_and_prepare_pool_view(
        token_positions,
        metadata,
        layer_idx=attention.get_local_layer_idx(metadata),
        is_generation=False,
    )

    kv_scale_orig_quant = _get_scalar(attention.kv_scale_orig_quant, device)
    kv_scale_quant_orig = _get_scalar(attention.kv_scale_quant_orig, device)
    quant_q_scale = torch.ones((), dtype=torch.float32, device=device)
    dequant_q_scale = kv_scale_quant_orig
    dequant_kv_scale = kv_scale_quant_orig
    bmm2_scale = kv_scale_quant_orig
    bmm1_host_scale = torch.tensor(
        1.0 / (float(attention.q_scaling) * math.sqrt(_QK_HEAD_DIM)),
        dtype=torch.float32,
        device=device,
    )

    kv_cache_pool_ref = torch.empty_like(kv_cache_pool)
    kv_cache_pool_ref[token_pool_rows.squeeze(-1).long(), 0, :] = (
        latent_cache.float() * kv_scale_orig_quant
    ).to(torch.float8_e4m3fn)
    q_fp8 = (fused_q.float() * quant_q_scale).to(torch.float8_e4m3fn)

    rows: list[torch.Tensor] = []
    for token_idx in range(num_tokens):
        pool_indices = topk_indices_pool[token_idx]
        valid = pool_indices >= 0
        gather_indices = pool_indices.clamp_min(0).to(torch.long)
        kv_fp8 = kv_cache_pool_ref[gather_indices, 0, :]
        scores = torch._scaled_mm(
            q_fp8[token_idx],
            kv_fp8.transpose(0, 1).contiguous(),
            scale_a=dequant_q_scale,
            scale_b=dequant_kv_scale,
            out_dtype=torch.float32,
        )
        scores = scores * bmm1_host_scale
        scores = scores.masked_fill(~valid.unsqueeze(0), -float("inf"))
        scores_log2 = scores * math.log2(math.e)
        max_scores = scores_log2.max(dim=-1, keepdim=True).values
        weights = torch.exp2(scores_log2 - max_scores)
        weights = weights.masked_fill(~valid.unsqueeze(0), 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        # TRTLLM-Gen sparse MLA context uses FP8 Q/K/V storage, but BMM2's closest
        # PyTorch-visible numerics are BF16-rounded probabilities and BF16-rounded V.
        row = (
            torch.matmul(
                weights.to(torch.bfloat16),
                kv_fp8[:, :_KV_LORA_RANK].to(torch.bfloat16),
            ).float()
            * bmm2_scale
        )
        rows.append(row.reshape(1, _LOCAL_NUM_HEADS * _KV_LORA_RANK))

    return torch.cat(rows, dim=0).to(torch.bfloat16)


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


def _build_attention_and_metadata() -> tuple[AttentionBackend, AttentionMetadata, _Config]:
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
        layer_idx=0,
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
    kv_cache_manager = DSACacheManager(
        KvCacheConfig(
            max_tokens=_TOKENS_PER_BLOCK * 4,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=_HEAD_DIM,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_CONTEXT_SEQUENCE_LENGTH,
        max_batch_size=1,
        mapping=mapping,
        dtype=str_dtype_to_binding(torch_dtype_to_str(torch.float8_e4m3fn)),
        sparse_attn_config=sparse_config,
        model_config=model_config,
    )
    kv_cache_manager.add_dummy_requests([0], [_CONTEXT_SEQUENCE_LENGTH])

    metadata = attn_cls.Metadata(
        seq_lens=torch.tensor([_CONTEXT_SEQUENCE_LENGTH], dtype=torch.int),
        request_ids=[0],
        max_num_requests=1,
        num_contexts=1,
        prompt_lens=[_CONTEXT_SEQUENCE_LENGTH],
        max_num_tokens=_CONTEXT_SEQUENCE_LENGTH,
        kv_cache_manager=kv_cache_manager,
        kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=[0]),
        mapping=mapping,
        sparse_attention_config=sparse_config,
    )
    metadata.prepare()
    return attention, metadata, config


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
