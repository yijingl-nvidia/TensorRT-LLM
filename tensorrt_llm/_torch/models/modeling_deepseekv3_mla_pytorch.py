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
from typing import Optional

import torch

from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import AttentionBackend


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last tensor dimension by half.

    This is the RoPE helper that moves the first half of the final dimension to the
    second half, and negates the second half and moves it to the first half.

    Args
    - x: torch.Tensor, shape [..., rope_dim], tensor with an even final dimension.

    Returns
    - torch.Tensor, shape [..., rope_dim], tensor with the last dimension rotated by half.
    """
    # [*, rope_dim // 2]
    x1 = x[..., : x.shape[-1] // 2]
    # [*, rope_dim // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin_for_rotation(rope_cos_sin: torch.Tensor, max_positions: int) -> torch.Tensor:
    """Normalize the cached RoPE table layout for the reference rotation.

    The TRTLLM attention backend may store the RoPE table as a flat two-dimensional
    tensor. This helper reshapes that table into the layout used by
    '_rotate_context_tensors' so the cosine and sine halves can be chunked from
    dimension -2.

    Args
    - rope_cos_sin: torch.Tensor, shape [1, max_positions * 2 * rope_dim] or
      [max_positions, 2 * rope_dim] or [max_positions, 2, rope_dim], RoPE table.
    - max_positions: int, maximum number of RoPE positions in the table.

    Returns
    - torch.Tensor, shape [max_positions, 2, rope_dim] when the input is flat, otherwise
      the original tensor layout.
    """
    if rope_cos_sin.dim() == 2 and rope_cos_sin.shape[0] == 1:
        # [max_positions, 2 * rope_dim]
        rope_cos_sin = rope_cos_sin.reshape(max_positions, -1)
    if rope_cos_sin.dim() == 2:
        # [max_positions, 2, rope_dim]
        rope_cos_sin = rope_cos_sin.reshape(rope_cos_sin.shape[0], -1, 2)
        return rope_cos_sin.transpose(-2, -1)
    return rope_cos_sin


def _get_scalar(scale: Optional[torch.Tensor], device: torch.device) -> torch.Tensor:
    if scale is None:
        return torch.ones((), dtype=torch.float32, device=device)
    return scale.reshape(-1)[0].to(dtype=torch.float32, device=device)


def _rotate_context_tensors(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    cached_len: int,
    max_positions: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply context-stage RoPE to query and latent-cache tensors.

    The fused MLA context path builds the latent query as '[q_nope_absorbed, q_pe]'
    and the latent cache as '[compressed_kv, k_pe]'. This helper mirrors the
    backend context preprocessing by rotating the query RoPE suffix and key RoPE
    suffix at the current context positions.

    Args
    - fused_q: torch.Tensor, shape [num_tokens, 8, 576] or [num_tokens, 4608],
      bfloat16, latent query before context RoPE is applied.
    - q_pe: torch.Tensor, shape [num_tokens, 8, 64], bfloat16, query RoPE suffix.
    - latent_cache: torch.Tensor, shape [num_tokens, 576], bfloat16, latent cache
      rows before key RoPE is applied.
    - rope_cos_sin: torch.Tensor, RoPE table accepted by '_rope_cos_sin_for_rotation'.
    - cached_len: int, number of already-cached context tokens before these tokens.
    - max_positions: int, maximum number of RoPE positions in the table.

    Returns
    - torch.Tensor, shape [num_tokens, 8, 576], bfloat16, latent query with rotated
      query RoPE suffix.
    - torch.Tensor, shape [num_tokens, 576], bfloat16, latent cache with rotated key
      RoPE suffix.
    """
    num_tokens = fused_q.shape[0]
    # [num_tokens, 8, 576]
    fused_q = fused_q.clone().view(num_tokens, 8, 576)
    # [num_tokens, 576]
    latent_cache = latent_cache.clone()

    # [max_positions, 2, 64]
    rope_cos_sin = _rope_cos_sin_for_rotation(rope_cos_sin, max_positions)
    # cos: [num_tokens, 1, 64]
    # sin: [num_tokens, 1, 64]
    cos, sin = rope_cos_sin[cached_len : cached_len + num_tokens].chunk(2, dim=-2)

    # [num_tokens, 8, 64]
    q_pe = q_pe.clone()
    # [num_tokens, 8, 64]
    q_pe = q_pe.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
    # [num_tokens, 8, 64]
    q_pe = ((q_pe * cos) + (_rotate_half(q_pe) * sin)).to(dtype=fused_q.dtype)
    # [num_tokens, 8, 64]
    q_pe = q_pe.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
    fused_q[..., 512:] = q_pe

    # [num_tokens, 1, 64]
    k_pe = latent_cache[:, 512:].unsqueeze(-2)
    # [num_tokens, 1, 64]
    k_pe = k_pe.unflatten(-1, [-1, 2]).transpose(-2, -1).flatten(start_dim=-2)
    # [num_tokens, 1, 64]
    k_pe = ((k_pe * cos) + (_rotate_half(k_pe) * sin)).to(dtype=latent_cache.dtype)
    # [num_tokens, 1, 64]
    k_pe = k_pe.unflatten(-1, [2, -1]).transpose(-2, -1).flatten(start_dim=-2)
    latent_cache[:, 512:] = k_pe.squeeze(-2)

    return fused_q, latent_cache


def _get_context_kv_pool_indices(
    num_tokens: int,
    cached_len: int,
    attn_metadata: AttentionMetadata,
    attention: AttentionBackend,
) -> torch.Tensor:
    """Compute paged KV-cache pool rows for context tokens.

    The DSA metadata exposes the primary KV-cache pool as a flattened writable
    tensor. This helper mirrors 'convert_req_index_to_global' for the dense
    sequence of context token positions so the PyTorch reference can populate
    the paged cache without invoking baseline attention.

    Args
    - num_tokens: int, number of context tokens in the current reference call.
    - cached_len: int, number of context tokens already present before the
      current context chunk.
    - attn_metadata: AttentionMetadata, DSA/TRTLLM metadata with cached
      block-table and request-index tensors.
    - attention: AttentionBackend, attention module used to resolve the local
      layer index in the KV-cache pool.

    Returns
    - torch.Tensor, shape [num_tokens], int64, flattened writable row indices
      into 'kv_cache_pool'.
    """
    if getattr(attn_metadata, "num_contexts", 1) != 1:
        raise NotImplementedError("PyTorch MLA context reference currently supports one context")

    attn_metadata._ensure_pool_view_cached()
    layer_idx = attention.get_local_layer_idx(attn_metadata)
    tokens_per_block = attn_metadata._cached_tokens_per_block
    stride_factor = attn_metadata._cached_stride_factor

    # [num_tokens], int64
    token_positions = torch.arange(
        cached_len,
        cached_len + num_tokens,
        dtype=torch.long,
        device=attn_metadata._cached_block_table_ctx.device,
    )
    # [num_tokens], int64
    block_indices = token_positions // tokens_per_block
    # [num_tokens], int64
    inblock_offsets = token_positions % tokens_per_block + layer_idx * tokens_per_block
    # [num_tokens], int64
    req_indices = attn_metadata._cached_req_idx_ctx[:num_tokens].to(torch.long)
    # [num_tokens], int64
    block_bases = attn_metadata._cached_block_table_ctx[req_indices, block_indices].to(torch.long)

    # [num_tokens], int64
    return block_bases * stride_factor + inblock_offsets


def _write_context_kv_cache(
    latent_cache: torch.Tensor,
    kv_cache_pool: torch.Tensor,
    pool_indices: torch.Tensor,
    kv_scale_orig_quant: torch.Tensor,
) -> None:
    """Write rotated latent context rows into the paged KV cache.

    The context reference attention reads the rotated local latent cache
    directly for numerical comparison, while subsequent decode steps read from
    the paged KV cache. This helper writes the same rotated rows to the exposed
    Python KV-cache pool view.

    Args
    - latent_cache: torch.Tensor, shape [num_tokens, 576], bfloat16, rotated
      latent cache rows to store.
    - kv_cache_pool: torch.Tensor, shape [pool_tokens, 1, 576], float8_e4m3fn
      or bfloat16, writable flattened KV-cache pool view.
    - pool_indices: torch.Tensor, shape [num_tokens], int64, destination rows
      in 'kv_cache_pool'.
    - kv_scale_orig_quant: torch.Tensor, scalar float32, BF16-to-FP8 cache
      quantization scale.
    """
    if kv_cache_pool.dtype == torch.float8_e4m3fn:
        # [num_tokens, 576], float8_e4m3fn
        cache_rows = (latent_cache.float() * kv_scale_orig_quant).to(torch.float8_e4m3fn)
        # [pool_tokens, 1, 576], uint8
        kv_cache_pool_uint8 = kv_cache_pool.view(torch.uint8)
        # [num_tokens, 1, 576], uint8
        cache_rows_uint8 = cache_rows.view(torch.uint8).unsqueeze(1)
        kv_cache_pool_uint8.index_copy_(0, pool_indices, cache_rows_uint8)
    else:
        # [num_tokens, 576], same dtype as kv_cache_pool
        cache_rows = latent_cache.to(kv_cache_pool.dtype)
        # [num_tokens, 1, 576]
        kv_cache_pool.index_copy_(0, pool_indices, cache_rows.unsqueeze(1))


def dsv3_mla_context_pytorch(
    fused_q: torch.Tensor,
    q_pe: torch.Tensor,
    latent_cache: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_indices_pool: torch.Tensor,
    kv_cache_pool: torch.Tensor,
    attn_metadata: AttentionMetadata,
    attention: AttentionBackend,
    write_kv_cache: bool = True,
) -> torch.Tensor:
    """Run the GLM-5 FP8 sparse MLA context attention in PyTorch.

    This is the high-accuracy reference path for the context-stage fused MLA
    branch. It mirrors the TRTLLM sparse context attention path closely enough
    for debugging and benching: apply context RoPE, quantize query and gathered
    latent cache rows to FP8 E4M3, compute FP8 scaled matrix multiply scores,
    apply the top-k sparse mask, run log2-domain softmax, and project weighted
    latent values back to BF16.

    Args
    - fused_q: torch.Tensor, shape [num_tokens, 8, 576] or [num_tokens, 4608],
      bfloat16, latent query before context RoPE and FP8 quantization.
    - q_pe: torch.Tensor, shape [num_tokens, 8, 64], bfloat16, query RoPE suffix.
    - latent_cache: torch.Tensor, shape [num_tokens, 576], bfloat16, local context
      latent cache rows before key RoPE and FP8 quantization.
    - topk_indices: torch.Tensor, shape [num_tokens, top_k], int32, local sparse
      attention indices. Negative entries are padding.
    - topk_indices_pool: torch.Tensor, shape [num_tokens, top_k], int32, sparse
      attention indices converted to KV-pool rows. Kept for signature parity.
    - kv_cache_pool: torch.Tensor, shape [pool_tokens, 1, 576], float8_e4m3fn,
      writable paged KV-cache pool view. The function stores the rotated latent
      context rows into this pool for subsequent decode.
    - attn_metadata: AttentionMetadata, context metadata carrying cached-token
      offsets and KV-cache layout.
    - attention: AttentionBackend, TRTLLM attention backend object carrying RoPE
      table, quantization scales, and attention score scale.
    - write_kv_cache: bool, whether to write rotated latent context rows into
      'kv_cache_pool' for subsequent decode.

    Returns
    - torch.Tensor, shape [num_tokens, 4096], bfloat16, latent attention output
      flattened across 8 local heads and 512 latent value dimensions.
    """
    num_tokens = fused_q.shape[0]
    device = fused_q.device
    cached_len = 0
    if hasattr(attn_metadata, "host_ctx_cached_token_indptr"):
        # [num_contexts + 1]
        host_indptr = attn_metadata.host_ctx_cached_token_indptr
        cached_len = int((host_indptr[1] - host_indptr[0]).item())

    max_positions = attention.rope_params.max_positions
    # fused_q: [num_tokens, 8, 576]
    # latent_cache: [num_tokens, 576]
    fused_q, latent_cache = _rotate_context_tensors(
        fused_q,
        q_pe,
        latent_cache,
        attention.rotary_cos_sin,
        cached_len,
        max_positions,
    )

    # scalar float32
    kv_scale_orig_quant = _get_scalar(attention.kv_scale_orig_quant, device)
    # scalar float32
    kv_scale_quant_orig = _get_scalar(attention.kv_scale_quant_orig, device)
    # scalar float32
    bmm1_host_scale = torch.tensor(
        1.0 / (float(attention.q_scaling) * math.sqrt(256.0)),
        dtype=torch.float32,
        device=device,
    )

    # [num_tokens, 8, 576], float8_e4m3fn
    q_fp8 = (fused_q.float() * kv_scale_orig_quant).to(torch.float8_e4m3fn)

    if write_kv_cache:
        # [num_tokens], int64
        pool_indices = _get_context_kv_pool_indices(
            num_tokens, cached_len, attn_metadata, attention
        )
        _write_context_kv_cache(latent_cache, kv_cache_pool, pool_indices, kv_scale_orig_quant)

    rows: list[torch.Tensor] = []
    for token_idx in range(num_tokens):
        # [top_k], int32
        local_indices = topk_indices[token_idx]
        # [top_k], bool
        valid = local_indices >= 0
        # [top_k], int64
        gather_indices = local_indices.clamp(0, num_tokens - 1).to(torch.long)
        # [top_k, 576], float8_e4m3fn
        kv_fp8 = (latent_cache[gather_indices].float() * kv_scale_orig_quant).to(
            torch.float8_e4m3fn
        )
        # [8, top_k], float32
        scores = torch._scaled_mm(
            q_fp8[token_idx],
            kv_fp8.transpose(0, 1).contiguous(),
            scale_a=kv_scale_quant_orig,
            scale_b=kv_scale_quant_orig,
            out_dtype=torch.float32,
        )
        # [8, top_k], float32
        scores = scores * bmm1_host_scale
        # [8, top_k], float32
        scores = scores.masked_fill(~valid.unsqueeze(0), -float("inf"))
        # [8, top_k], float32
        scores_log2 = scores * math.log2(math.e)
        # [8, 1], float32
        max_scores = scores_log2.max(dim=-1, keepdim=True).values
        # [8, top_k], float32
        weights = torch.exp2(scores_log2 - max_scores)
        # [8, top_k], float32
        weights = weights.masked_fill(~valid.unsqueeze(0), 0.0)
        # [8, top_k], float32
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)
        # [8, 512], float32
        row = (
            torch.matmul(
                weights.to(torch.bfloat16),
                kv_fp8[:, :512].to(torch.bfloat16),
            ).float()
            * kv_scale_quant_orig
        )
        # [1, 4096]
        rows.append(row.reshape(1, 8 * 512))

    _ = topk_indices_pool
    # [num_tokens, 4096], bfloat16
    return torch.cat(rows, dim=0).to(torch.bfloat16)


def dsv3_mla_decode_pytorch(
    quant_q_buffer: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_indices_pool: torch.Tensor,
    kv_cache_pool: torch.Tensor,
    sequence_length: torch.Tensor,
    mla_bmm1_scale: torch.Tensor,
    mla_bmm2_scale: torch.Tensor,
    spec_decoding_packed_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run the GLM-5 FP8 sparse MLA decode attention in PyTorch.

    This is the reference path for the BS=1, MTP=3 generation/decode fused
    MLA branch. It consumes the buffers filled by 'mla_rope_generation':
    rotated and FP8-quantized Q, current-step latent KV rows already appended
    into the paged KV cache, top-k local indices, and top-k pool rows. The
    computation mirrors the TRTLLM generation path by doing FP8 QK scaled
    matrix multiplies, applying historical/current-group decode validity, using
    the log2-domain softmax scale produced by 'mla_rope_generation', and
    projecting FP8 latent V rows back to BF16.

    Args
    - quant_q_buffer: torch.Tensor, shape [num_tokens, 8, 576], uint8 storage
      reinterpreted as float8_e4m3fn, rotated and quantized query rows.
    - topk_indices: torch.Tensor, shape [num_tokens, top_k], int32, local KV
      positions before pool conversion.
    - topk_indices_pool: torch.Tensor, shape [num_tokens, top_k], int32, global
      rows into 'kv_cache_pool'. Negative entries are padding.
    - kv_cache_pool: torch.Tensor, shape [pool_tokens, 1, 576], float8_e4m3fn,
      flattened view of the primary paged KV-cache pool.
    - sequence_length: torch.Tensor, shape [1], int32, post-append KV length
      for the single generation sequence.
    - mla_bmm1_scale: torch.Tensor, shape [2], float32, score scale where index
      1 is already multiplied by log2(e).
    - mla_bmm2_scale: torch.Tensor, shape [1], float32, FP8 V dequantization
      scale for the latent output.
    - spec_decoding_packed_mask: Optional[torch.Tensor], shape
      [max_requests, num_tokens, packed_words], int32, current-group attention
      mask for speculative/MTP decode. Historical KV rows are always valid.

    Returns
    - torch.Tensor, shape [num_tokens, 4096], bfloat16, latent attention output
      flattened across 8 local heads and 512 latent value dimensions.
    """
    num_tokens, num_heads, head_dim = quant_q_buffer.shape
    assert num_heads == 8
    assert head_dim == 576
    assert kv_cache_pool.shape[1:] == (1, 576)

    # scalar int32
    current_group_start = sequence_length[0] - num_tokens
    # scalar int32
    max_local_index = topk_indices.clamp_min(0).max()
    # CUDA graph warmup can capture generation with a dummy post-append length
    # of 'num_tokens'. For the BS=1 1k prompt path, topK covers the full KV
    # sequence, so the largest local top-k index recovers the real group end.
    # scalar int32
    inferred_group_start = max_local_index - (num_tokens - 1)
    # scalar bool
    use_inferred_group_start = (sequence_length[0] <= num_tokens) & (max_local_index >= num_tokens)
    # scalar int32
    current_group_start = torch.where(
        use_inferred_group_start,
        inferred_group_start,
        current_group_start,
    )
    # [num_tokens, top_k], int32
    current_group_offset = topk_indices - current_group_start
    # [num_tokens, top_k], bool
    historical_kv = topk_indices < current_group_start
    # [num_tokens, top_k], bool
    current_group_kv = (current_group_offset >= 0) & (current_group_offset < num_tokens)
    if spec_decoding_packed_mask is None:
        # [num_tokens], int32
        token_positions = torch.arange(
            num_tokens,
            dtype=topk_indices.dtype,
            device=topk_indices.device,
        )
        # [num_tokens, top_k], bool
        current_group_valid = current_group_offset <= token_positions[:, None]
    else:
        # [num_tokens, packed_words], int32
        packed_mask = spec_decoding_packed_mask[0, :num_tokens, :].to(topk_indices.dtype)
        # [num_tokens, top_k], int32
        packed_word_idx = torch.div(
            current_group_offset.clamp_min(0),
            32,
            rounding_mode="floor",
        )
        # [num_tokens, top_k], int32
        packed_bit_idx = current_group_offset.clamp(0, 31)
        # [num_tokens, top_k], int32
        packed_words = packed_mask.gather(
            1,
            packed_word_idx.clamp_max(packed_mask.shape[1] - 1).to(torch.long),
        )
        # [num_tokens, top_k], bool
        current_group_valid = torch.bitwise_and(
            torch.bitwise_right_shift(packed_words, packed_bit_idx),
            1,
        ).to(torch.bool)

    # [num_tokens, top_k], bool
    valid = (topk_indices_pool >= 0) & (historical_kv | (current_group_kv & current_group_valid))
    # [num_tokens, top_k], int64
    pool_indices = topk_indices_pool.clamp_min(0).to(torch.long)

    # [num_tokens, 8, 576], float8_e4m3fn
    q = quant_q_buffer.view(torch.float8_e4m3fn)
    # [num_tokens, top_k, 576], float8_e4m3fn
    kv_fp8 = kv_cache_pool[:, 0, :][pool_indices]
    # scalar float32
    mm_scale = torch.ones((), dtype=torch.float32, device=quant_q_buffer.device)
    # [num_tokens, 8, top_k], float32
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
    # [num_tokens, 8, top_k], float32
    scores = scores * mla_bmm1_scale[1]
    # [num_tokens, 8, top_k], float32
    scores = scores.masked_fill(~valid[:, None, :], -float("inf"))

    # [num_tokens, 8, 1], float32
    max_scores = scores.max(dim=-1, keepdim=True).values
    # [num_tokens, 8, top_k], float32
    weights = torch.exp2(scores - max_scores)
    # [num_tokens, 8, top_k], float32
    weights = weights.masked_fill(~valid[:, None, :], 0.0)
    # [num_tokens, 8, top_k], float32
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-20)

    # [num_tokens, top_k, 576], float32
    kv = kv_fp8.to(torch.float32)
    # [num_tokens, 8, 512], float32
    output = torch.einsum("thk,tkd->thd", weights, kv[..., :512])
    # [num_tokens, 8, 512], float32
    output = output * mla_bmm2_scale[0]
    # [num_tokens, 4096], bfloat16
    return output.to(torch.bfloat16).reshape(num_tokens, num_heads * 512)
