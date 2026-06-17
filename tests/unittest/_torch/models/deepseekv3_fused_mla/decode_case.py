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

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    transform_local_topk_and_prepare_pool_view,
)
from tensorrt_llm._utils import get_sm_version
from tests.unittest._torch.models.deepseekv3_fused_mla.dump_utils import (
    FusedMlaDumpGroup,
    _live_q_b_fusion_dump_group,
    _load_int,
    _load_tensor,
    _q_b_fusion_dump_group,
    _require_cuda_and_ops,
)
from tests.unittest._torch.models.deepseekv3_fused_mla.reference_ops import (
    _build_dump_q_b_projection_inputs,
    _build_fp8_block_scale_linear,
)
from tests.unittest._torch.models.test_modeling_deepseekv3_attention import (
    _BENCH_CONTEXT_SEQUENCE_LENGTH,
    _DECODE_NUM_TOKENS,
    _HEAD_DIM,
    _KV_LORA_RANK,
    _LOCAL_NUM_HEADS,
    _QK_HEAD_DIM,
    _QK_NOPE_HEAD_DIM,
    _QK_ROPE_HEAD_DIM,
    _build_decode_attention_and_metadata,
    _build_decode_topk,
    _custom_decode_attention,
    _require_glm5_context_attention_runtime,
    _seed_decode_history_kv_cache,
)


@dataclass(frozen=True)
class FusedMlaDumpDecodeCase:
    group: FusedMlaDumpGroup
    attention: Any
    metadata: Any
    q_b_proj_input: torch.Tensor
    q_b_proj_output: torch.Tensor
    q_b_proj_weight: torch.Tensor
    q_b_proj_weight_scale: torch.Tensor
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    k_b_proj_trans: torch.Tensor
    latent_cache: torch.Tensor
    topk_indices: torch.Tensor
    topk_indices_pool: torch.Tensor
    kv_cache_pool: torch.Tensor


def _build_dump_decode_q_b_case(
    rank: int,
    require_live_q_b: bool = False,
    group: FusedMlaDumpGroup | None = None,
) -> FusedMlaDumpDecodeCase:
    """
    Build a dump-backed GLM-5 decode case for q_b fusion tests.

    The metadata, KV cache layout, top-k indices, and MTP token count follow the
    synthetic GLM-5 tests. q_b/k_b tensors come from the debug dump directory,
    and q_b input/output are produced by running loaded hidden_states through the
    existing TRTLLM module path. Side effects include disk I/O, CUDA allocations,
    and seeding the test KV cache.

    Args
    - rank: int, tensor-parallel rank id selected by pytest parametrization.
    - require_live_q_b: bool, whether q_b input/output must come from a live
        bench-path dump instead of hidden-state reconstruction.
    - group: FusedMlaDumpGroup | None, explicit rank/layer dump group to use.

    Returns
    - case: FusedMlaDumpDecodeCase, all tensors and metadata needed to run
        preprojected and fused-q_b decode attention paths.
    """
    _require_cuda_and_ops()
    _require_glm5_context_attention_runtime()
    if get_sm_version() not in (100, 103):
        pytest.skip("q_b_proj fusion uses SM100 packed UE8M0 scale layout")

    device = torch.device("cuda")
    torch.manual_seed(8_000 + rank)
    torch.cuda.manual_seed(8_000 + rank)
    if group is None:
        group = (
            _live_q_b_fusion_dump_group(rank) if require_live_q_b else _q_b_fusion_dump_group(rank)
        )
    assert group.rank == rank
    kv_lora_rank = _load_int(group, "kv_lora_rank")
    assert kv_lora_rank == _KV_LORA_RANK

    attention, metadata, _ = _build_decode_attention_and_metadata(
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    _seed_decode_history_kv_cache(
        attention,
        metadata,
        device,
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    (
        q_b_proj_input,
        q_b_proj_output,
        q_b_proj_weight,
        q_b_proj_weight_scale,
        q_nope,
        q_pe,
    ) = _build_dump_q_b_projection_inputs(group, _DECODE_NUM_TOKENS, device)

    # [_LOCAL_NUM_HEADS, _KV_LORA_RANK, _QK_NOPE_HEAD_DIM]
    k_b_proj_trans = _load_tensor(group, "k_b_proj_trans").to(torch.bfloat16)
    assert k_b_proj_trans.shape == (
        _LOCAL_NUM_HEADS,
        _KV_LORA_RANK,
        _QK_NOPE_HEAD_DIM,
    )

    # [_DECODE_NUM_TOKENS, _HEAD_DIM]
    latent_cache = (
        torch.randn(
            _DECODE_NUM_TOKENS,
            _HEAD_DIM,
            dtype=torch.bfloat16,
            device=device,
        )
        * 0.125
    )
    # [_DECODE_NUM_TOKENS, _INDEX_TOPK]
    topk_indices = _build_decode_topk(
        device,
        context_sequence_length=_BENCH_CONTEXT_SEQUENCE_LENGTH,
    )
    topk_indices_pool, kv_cache_pool = transform_local_topk_and_prepare_pool_view(
        topk_indices,
        metadata,
        layer_idx=attention.get_local_layer_idx(metadata),
        is_generation=True,
    )

    return FusedMlaDumpDecodeCase(
        group=group,
        attention=attention,
        metadata=metadata,
        q_b_proj_input=q_b_proj_input,
        q_b_proj_output=q_b_proj_output,
        q_b_proj_weight=q_b_proj_weight,
        q_b_proj_weight_scale=q_b_proj_weight_scale,
        q_nope=q_nope,
        q_pe=q_pe,
        k_b_proj_trans=k_b_proj_trans,
        latent_cache=latent_cache,
        topk_indices=topk_indices,
        topk_indices_pool=topk_indices_pool,
        kv_cache_pool=kv_cache_pool,
    )


def _decode_runtime_buffers(
    case: FusedMlaDumpDecodeCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Allocate mutable decode runtime buffers for one custom op invocation.

    The fused decode op writes fused_q, quant_q_buffer, and the two MLA BMM
    scales. cu_q_seqlens/cu_kv_seqlens/fmha_scheduler_counter are kept for
    signature compatibility with the baseline attention path.

    Args
    - case: FusedMlaDumpDecodeCase, decode metadata and device source.

    Returns
    - fused_q: torch.Tensor, shape [4, 8, 576], bf16, mutable projected query
        and RoPE buffer.
    - quant_q_buffer: torch.Tensor, shape [4, 8, 576], uint8, mutable FP8 query
        buffer.
    - mla_bmm1_scale: torch.Tensor, shape [2], fp32, mutable BMM1 scales.
    - mla_bmm2_scale: torch.Tensor, shape [1], fp32, mutable BMM2 scale.
    - cu_q_seqlens: torch.Tensor, shape [num_seqs + 1], int32, compatibility
        buffer.
    - cu_kv_seqlens: torch.Tensor, shape [num_seqs + 1], int32, compatibility
        buffer.
    """
    device = case.q_b_proj_input.device
    num_tokens = case.q_b_proj_input.shape[0]
    num_seqs = case.metadata.kv_lens_cuda_runtime.size(0)
    fused_q = torch.empty(
        num_tokens,
        _LOCAL_NUM_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    quant_q_buffer = torch.empty_like(fused_q, dtype=torch.uint8)
    mla_bmm1_scale = torch.empty(2, dtype=torch.float32, device=device)
    mla_bmm2_scale = torch.empty(1, dtype=torch.float32, device=device)
    cu_q_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    cu_kv_seqlens = torch.empty(num_seqs + 1, dtype=torch.int32, device=device)
    return (
        fused_q,
        quant_q_buffer,
        mla_bmm1_scale,
        mla_bmm2_scale,
        cu_q_seqlens,
        cu_kv_seqlens,
    )


def _run_dump_decode_preprojected(
    case: FusedMlaDumpDecodeCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the WIP decode op with preprojected q_nope/q_pe reference inputs.

    This path mirrors the model code before q_b projection was fused into the
    decode op. It is the closest apples-to-apples reference for isolating q_b
    projection differences inside dsv3_fused_mla_generation.

    Args
    - case: FusedMlaDumpDecodeCase, dump-backed decode setup.
    - q_b_proj_output: torch.Tensor | None, optional BF16 [4, 2048] tensor
        filled by the op with raw q_b projection output for direct validation.

    Returns
    - output: torch.Tensor, shape [4, 4096], bf16, latent attention output.
    - fused_q: torch.Tensor, shape [4, 8, 576], bf16, projected query buffer
        written by the op.
    - quant_q_buffer: torch.Tensor, shape [4, 8, 576], uint8, FP8 query buffer
        written by the op.
    - mla_bmm1_scale: torch.Tensor, shape [2], fp32, BMM1 scales written by the
        op.
    - mla_bmm2_scale: torch.Tensor, shape [1], fp32, BMM2 scale written by the
        op.
    """
    (
        fused_q,
        quant_q_buffer,
        mla_bmm1_scale,
        mla_bmm2_scale,
        cu_q_seqlens,
        cu_kv_seqlens,
    ) = _decode_runtime_buffers(case)
    fmha_scheduler_counter = torch.empty(
        1,
        dtype=torch.uint32,
        device=case.q_b_proj_input.device,
    )
    output = _custom_decode_attention(
        fused_q,
        case.q_nope,
        case.k_b_proj_trans,
        case.q_pe.clone(),
        case.latent_cache,
        case.topk_indices,
        case.topk_indices_pool,
        case.kv_cache_pool,
        case.attention,
        case.metadata,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
    )
    return output, fused_q, quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale


def _run_dump_decode_fused_q_b(
    case: FusedMlaDumpDecodeCase,
    q_b_proj_output: torch.Tensor | None = None,
    q_b_proj_use_mma: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the WIP decode op with q_b projection fused into the op.

    Dummy q_nope/q_pe inputs are passed because the fused path must derive them
    from q_b_proj_input, q_b_proj_weight, and q_b_proj_weight_scale. The q_b
    weight scale is the packed int32 post-load tensor used by the model.

    Args
    - case: FusedMlaDumpDecodeCase, dump-backed decode setup.

    Returns
    - output: torch.Tensor, shape [4, 4096], bf16, latent attention output.
    - fused_q: torch.Tensor, shape [4, 8, 576], bf16, projected query buffer
        written by the op.
    - quant_q_buffer: torch.Tensor, shape [4, 8, 576], uint8, FP8 query buffer
        written by the op.
    - mla_bmm1_scale: torch.Tensor, shape [2], fp32, BMM1 scales written by the
        op.
    - mla_bmm2_scale: torch.Tensor, shape [1], fp32, BMM2 scale written by the
        op.
    """
    (
        fused_q,
        quant_q_buffer,
        mla_bmm1_scale,
        mla_bmm2_scale,
        cu_q_seqlens,
        cu_kv_seqlens,
    ) = _decode_runtime_buffers(case)
    fmha_scheduler_counter = torch.empty(
        1,
        dtype=torch.uint32,
        device=case.q_b_proj_input.device,
    )
    if q_b_proj_output is None:
        q_b_proj_output = torch.empty(
            case.q_b_proj_input.shape[0],
            _LOCAL_NUM_HEADS * _QK_HEAD_DIM,
            dtype=case.q_b_proj_input.dtype,
            device=case.q_b_proj_input.device,
        )
    output = _custom_decode_attention(
        fused_q,
        torch.empty_like(case.q_nope),
        case.k_b_proj_trans,
        torch.empty_like(case.q_pe),
        case.latent_cache,
        case.topk_indices,
        case.topk_indices_pool,
        case.kv_cache_pool,
        case.attention,
        case.metadata,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
        q_b_proj_input=case.q_b_proj_input,
        q_b_proj_weight=case.q_b_proj_weight,
        q_b_proj_weight_scale=case.q_b_proj_weight_scale,
        q_b_proj_output=q_b_proj_output,
        q_b_proj_use_mma=q_b_proj_use_mma,
    )
    return output, fused_q, quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale


def _with_original_q_b_weight_and_scale(
    case: FusedMlaDumpDecodeCase,
) -> FusedMlaDumpDecodeCase:
    """
    Replace a dump decode case with the raw original q_b weight-scale path.

    The normal dump case uses the post-load resmoothed q_b tensors. This helper
    reloads the raw FP8 E4M3 q_b weight and FP32 128x128 block scales, builds a
    Linear with DeepGEMM resmoothing disabled, and computes the matching
    preprojected q_nope/q_pe reference. The returned case can be passed to the
    fused op to exercise q_b_proj_weight_scale as FP32 [16, 16].

    Args
    - case: FusedMlaDumpDecodeCase, dump-backed decode setup using the regular
        post-load q_b tensors.

    Returns
    - original_case: FusedMlaDumpDecodeCase, same attention metadata and KV
        cache as `case`, but with original q_b weight, original q_b FP32 scales,
        and q_nope/q_pe derived from that original-scale Linear output.
    """
    # [num_heads_tp * qk_head_dim, q_lora_rank]
    q_b_proj_weight = _load_tensor(case.group, "q_b_proj_weight")
    # [ceil(out_features / 128), ceil(in_features / 128)]
    q_b_proj_weight_scale = _load_tensor(case.group, "q_b_proj_weight_scale")
    if q_b_proj_weight_scale.dtype != torch.float32:
        pytest.skip("original q_b FP32 scale dump is not available")

    q_b_linear = _build_fp8_block_scale_linear(
        q_b_proj_weight,
        q_b_proj_weight_scale,
        disable_deep_gemm=True,
    )

    # [num_tokens, num_heads_tp * qk_head_dim]
    q_b_proj_output = q_b_linear(case.q_b_proj_input.contiguous()).contiguous()
    # [num_tokens, num_heads_tp, qk_head_dim]
    q_heads = q_b_proj_output.view(
        q_b_proj_output.shape[0],
        _LOCAL_NUM_HEADS,
        _QK_HEAD_DIM,
    )
    # q_nope: [num_tokens, num_heads_tp, qk_nope_head_dim]
    # q_pe: [num_tokens, num_heads_tp, qk_rope_head_dim]
    q_nope, q_pe = q_heads.split([_QK_NOPE_HEAD_DIM, _QK_ROPE_HEAD_DIM], dim=-1)
    return replace(
        case,
        q_b_proj_output=q_b_proj_output,
        q_b_proj_weight=q_b_linear.weight.detach(),
        q_b_proj_weight_scale=q_b_linear.weight_scale.detach(),
        q_nope=q_nope.contiguous(),
        q_pe=q_pe.contiguous(),
    )


def _selector_k_b_proj_trans(device: torch.device) -> torch.Tensor:
    """
    Build a k_b projection tensor that exposes raw q_b output in fused_q.

    The production k_b projection mixes the 192 q_nope dimensions into the
    512-dim latent prefix, which can hide a q_b-only mismatch. This selector
    maps q_nope dim d to fused_q prefix dim d and leaves dims [192, 512) zero.

    Args
    - device: torch.device, CUDA device where the selector tensor should live.

    Returns
    - k_b_proj_trans: torch.Tensor, shape [8, 512, 192], bf16, deterministic
        selector projection for q_b debugging.
    """
    k_b_proj_trans = torch.zeros(
        _LOCAL_NUM_HEADS,
        _KV_LORA_RANK,
        _QK_NOPE_HEAD_DIM,
        dtype=torch.bfloat16,
        device=device,
    )
    dim = torch.arange(_QK_NOPE_HEAD_DIM, dtype=torch.long, device=device)
    k_b_proj_trans[:, dim, dim] = 1.0
    return k_b_proj_trans.contiguous()


def _run_dump_decode_backend(
    case: FusedMlaDumpDecodeCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Run the baseline TRTLLM attention backend with dumped q_b/k_b tensors.

    This path constructs fused_q with the same standalone bmm_out sequence used
    by modeling_deepseekv3_fused_mla.py before the q_b fusion and then calls
    mla_rope_generation plus attention.forward.

    Args
    - case: FusedMlaDumpDecodeCase, dump-backed decode setup.

    Returns
    - output: torch.Tensor, shape [4, 4096], bf16, baseline latent attention
        output.
    - fused_q: torch.Tensor, shape [4, 8, 576], bf16, projected query buffer
        consumed by the backend.
    - quant_q_buffer: torch.Tensor, shape [4, 8, 576], uint8, FP8 query buffer
        written by mla_rope_generation.
    - mla_bmm1_scale: torch.Tensor, shape [2], fp32, BMM1 scales written by
        mla_rope_generation.
    - mla_bmm2_scale: torch.Tensor, shape [1], fp32, BMM2 scale written by
        mla_rope_generation.
    """
    (
        fused_q,
        quant_q_buffer,
        mla_bmm1_scale,
        mla_bmm2_scale,
        cu_q_seqlens,
        cu_kv_seqlens,
    ) = _decode_runtime_buffers(case)
    fmha_scheduler_counter = torch.empty(
        1,
        dtype=torch.uint32,
        device=case.q_b_proj_input.device,
    )

    # [num_heads, 4, qk_nope_head_dim]
    q_nope_t = case.q_nope.transpose(0, 1)
    # [num_heads, 4, kv_lora_rank]
    q_nope_out = fused_q[..., :_KV_LORA_RANK].transpose(0, 1)
    torch.ops.trtllm.bmm_out(
        q_nope_t,
        case.k_b_proj_trans.transpose(1, 2),
        q_nope_out,
    )

    q_pe = case.q_pe.clone()
    case.attention.mla_rope_generation(
        fused_q,
        q_pe,
        case.latent_cache,
        case.metadata,
        cu_q_seqlens,
        cu_kv_seqlens,
        fmha_scheduler_counter,
        mla_bmm1_scale,
        mla_bmm2_scale,
        quant_q_buffer,
    )
    output = case.attention.forward(
        fused_q.reshape(_DECODE_NUM_TOKENS, _LOCAL_NUM_HEADS * _HEAD_DIM),
        None,
        None,
        case.metadata,
        attention_input_type=AttentionInputType.generation_only,
        latent_cache=case.latent_cache,
        q_pe=q_pe,
        topk_indices=case.topk_indices,
        is_generation=True,
        cu_q_seqlens=cu_q_seqlens,
        cu_kv_seqlens=cu_kv_seqlens,
        fmha_scheduler_counter=fmha_scheduler_counter,
        mla_bmm1_scale=mla_bmm1_scale,
        mla_bmm2_scale=mla_bmm2_scale,
        quant_q_buffer=quant_q_buffer,
    )
    return output, fused_q, quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale
