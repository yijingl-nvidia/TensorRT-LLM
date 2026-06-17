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

from dataclasses import replace

import pytest
import torch
from torch.nn import Parameter

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tests.unittest._torch.models.deepseekv3_fused_mla.decode_case import (
    _build_dump_decode_q_b_case,
    _run_dump_decode_fused_q_b,
)
from tests.unittest._torch.models.deepseekv3_fused_mla.dump_utils import _NUM_RANKS
from tests.unittest._torch.models.test_modeling_deepseekv3_attention import (
    _LOCAL_NUM_HEADS,
    _Q_LORA_RANK,
    _QK_HEAD_DIM,
)


def _slice_case_tokens(case, token_count: int):
    assert 0 < token_count <= case.q_b_proj_input.shape[0]
    return replace(
        case,
        q_b_proj_input=case.q_b_proj_input[:token_count].contiguous(),
        q_b_proj_output=case.q_b_proj_output[:token_count].contiguous(),
        q_nope=case.q_nope[:token_count].contiguous(),
        q_pe=case.q_pe[:token_count].contiguous(),
        latent_cache=case.latent_cache[:token_count].contiguous(),
        topk_indices=case.topk_indices[:token_count].contiguous(),
        topk_indices_pool=case.topk_indices_pool[:token_count].contiguous(),
    )


def _ensure_tma_col_major_int_scale(scale: torch.Tensor) -> torch.Tensor:
    """
    Recreate DeepGEMM's TMA column-major int scale view after dump loading.

    The shared dump loader intentionally returns contiguous CUDA tensors. That
    preserves the logical packed scale values but drops the post-load view
    strides expected by fp8_swap_ab_gemm. Rebuild that view locally for the
    Linear reference path without changing the dump fallback behavior.
    """
    if scale.dtype != torch.int32:
        return scale
    if scale.stride(-2) == 1 and scale.stride(-1) == ((scale.size(-2) + 3) // 4) * 4:
        return scale

    remove_dim = False
    if scale.dim() == 2:
        scale = scale.unsqueeze(0)
        remove_dim = True

    batches, rows, packed_cols = scale.shape
    aligned_rows = ((rows + 3) // 4) * 4
    col_major = torch.transpose(
        torch.empty(
            (batches, packed_cols, aligned_rows),
            device=scale.device,
            dtype=scale.dtype,
        ),
        1,
        2,
    )
    col_major[:, :rows, :] = scale
    result = col_major[:, :rows, :]
    return result.squeeze(0) if remove_dim else result


def _build_local_q_b_proj(
    q_b_proj_weight: torch.Tensor,
    q_b_proj_weight_scale: torch.Tensor,
) -> Linear:
    """
    Build the q_b_proj module shape used by FusedMLA for one local TP rank.

    This mirrors the self.q_b_proj definition in
    tensorrt_llm/_torch/models/modeling_deepseekv3_fused_mla.py, but uses the
    already-local output width from the dump-backed test case. Post-load int32
    scales are attached directly because they are already in the runtime layout
    consumed by Linear.forward and the fused MLA kernel.

    Args
    - q_b_proj_weight: torch.Tensor, shape [2048, 2048], fp8_e4m3, local q_b
        projection weight.
    - q_b_proj_weight_scale: torch.Tensor, shape [2048, 4], int32 post-load
        packed UE8M0 scale, or raw FP32 block scales before post_load_weights.

    Returns
    - q_b_proj: Linear, eval-mode module matching the local q_b projection path.
    """
    q_b_proj = Linear(
        _Q_LORA_RANK,
        _LOCAL_NUM_HEADS * _QK_HEAD_DIM,
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
        maintain_original_weight=True,
    )
    q_b_proj.cuda()
    if q_b_proj_weight_scale.dtype == torch.int32:
        q_b_proj.weight = Parameter(q_b_proj_weight.contiguous(), requires_grad=False)
        q_b_proj.weight_scale = Parameter(
            _ensure_tma_col_major_int_scale(q_b_proj_weight_scale),
            requires_grad=False,
        )
    else:
        q_b_proj.load_weights(
            [
                {
                    "weight": q_b_proj_weight,
                    "weight_scale": q_b_proj_weight_scale,
                }
            ]
        )
        q_b_proj.post_load_weights()
    q_b_proj.eval()
    return q_b_proj


def _matrix_min_max(tensor: torch.Tensor) -> tuple[float, float]:
    tensor_float = tensor.float()
    return tensor_float.amin().item(), tensor_float.amax().item()


def _assert_tensors_identical(
    actual: torch.Tensor,
    expected: torch.Tensor,
    message_prefix: str,
) -> None:
    if torch.equal(actual, expected):
        return

    mismatch = actual != expected
    mismatch_indices = mismatch.nonzero()
    first_index = tuple(mismatch_indices[0].tolist())
    actual_float = actual.float()
    expected_float = expected.float()
    abs_diff = (actual_float - expected_float).abs()
    raise AssertionError(
        f"{message_prefix}: tensors differ exactly; "
        f"num_mismatches={mismatch_indices.shape[0]} "
        f"max_abs_diff={abs_diff.amax().item():.6g} "
        f"first_mismatch_index={first_index} "
        f"actual={actual[first_index].item()} "
        f"expected={expected[first_index].item()}"
    )


@pytest.mark.parametrize("token_count", [1, 2, 3, 4])
@pytest.mark.parametrize("q_b_proj_use_mma", [True, False])
@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_mla_q_b_proj(
    rank: int,
    q_b_proj_use_mma: bool,
    token_count: int,
) -> None:
    """
    Compare q_b projection inside fused MLA kernel directly against
    existing TRTLLM Linear module.
    """
    case = _slice_case_tokens(_build_dump_decode_q_b_case(rank), token_count)
    q_b_proj = _build_local_q_b_proj(case.q_b_proj_weight, case.q_b_proj_weight_scale)

    with torch.inference_mode():
        q_b_proj_expected = q_b_proj(case.q_b_proj_input.contiguous()).contiguous()
        q_b_proj_actual = torch.empty_like(q_b_proj_expected)
        q_b_proj_actual.fill_(float("nan"))
        _run_dump_decode_fused_q_b(
            case,
            q_b_proj_output=q_b_proj_actual,
            q_b_proj_use_mma=q_b_proj_use_mma,
        )

    assert not torch.isnan(q_b_proj_actual).any().item()
    ref_min, ref_max = _matrix_min_max(q_b_proj_expected)
    wip_min, wip_max = _matrix_min_max(q_b_proj_actual)
    print(
        f"rank={rank} token_count={token_count} "
        f"q_b_proj_use_mma={q_b_proj_use_mma} "
        f"q_b_proj_ref_min={ref_min:.6g} "
        f"q_b_proj_ref_max={ref_max:.6g} "
        f"q_b_proj_wip_min={wip_min:.6g} "
        f"q_b_proj_wip_max={wip_max:.6g}"
    )
    torch.testing.assert_close(
        q_b_proj_actual,
        q_b_proj_expected,
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("token_count", [1, 2, 3, 4])
@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_mla_q_b_proj_mma_matches_scalar(
    rank: int,
    token_count: int,
) -> None:
    """
    Compare MMA and scalar q_b projection on the same q_b input tensors.

    This test intentionally compares the two fused-kernel q_b implementations
    directly instead of comparing either one against Linear. The input tensors
    are the dump-backed q_b projection inputs prepared by the shared decode
    case builder.
    """
    case = _slice_case_tokens(_build_dump_decode_q_b_case(rank), token_count)

    with torch.inference_mode():
        q_b_proj_scalar = torch.empty_like(case.q_b_proj_output)
        q_b_proj_mma = torch.empty_like(case.q_b_proj_output)
        q_b_proj_scalar.fill_(float("nan"))
        q_b_proj_mma.fill_(float("nan"))
        _run_dump_decode_fused_q_b(
            case,
            q_b_proj_output=q_b_proj_scalar,
            q_b_proj_use_mma=False,
        )
        _run_dump_decode_fused_q_b(
            case,
            q_b_proj_output=q_b_proj_mma,
            q_b_proj_use_mma=True,
        )

    assert not torch.isnan(q_b_proj_scalar).any().item()
    assert not torch.isnan(q_b_proj_mma).any().item()
    _assert_tensors_identical(
        q_b_proj_mma,
        q_b_proj_scalar,
        f"rank={rank} layer={case.group.layer_idx} token_count={token_count}",
    )
