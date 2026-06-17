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

from tests.unittest._torch.models.deepseekv3_fused_mla.assertions import (
    _assert_q_b_selector_prefix_close,
)
from tests.unittest._torch.models.deepseekv3_fused_mla.decode_case import (
    _build_dump_decode_q_b_case,
    _run_dump_decode_fused_q_b,
    _run_dump_decode_preprojected,
    _selector_k_b_proj_trans,
)
from tests.unittest._torch.models.deepseekv3_fused_mla.dump_utils import (
    _NUM_RANKS,
    _live_q_b_fusion_dump_groups,
)


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_mla_live_decode_q_b_proj_raw_output_matches_linear(
    rank: int,
) -> None:
    """
    Compare fused q_b projection against q_b output dumped from live benching.

    Unlike the hidden-state-derived test, this requires q_b_proj_input and
    q_b_proj_output saved by modeling_deepseekv3_fused_mla.py from a real WIP
    decode iteration. It therefore validates the exact activation distribution
    and post-load q_b weight/scale layout used by the bench path.

    Args
    - rank: int, tensor-parallel rank id selected by pytest parametrization.

    Returns
    - None: successful return means the WIP q_b projection reproduces the live
        TRTLLM q_b Linear output exactly for the selector-visible q_nope values.
    """
    failures = []
    for group in _live_q_b_fusion_dump_groups(rank):
        case = _build_dump_decode_q_b_case(
            rank,
            require_live_q_b=True,
            group=group,
        )
        selector_case = replace(
            case,
            k_b_proj_trans=_selector_k_b_proj_trans(case.q_b_proj_input.device),
        )

        with torch.inference_mode():
            (
                _,
                fused_q_expected,
                quant_q_buffer_expected,
                mla_bmm1_scale_expected,
                mla_bmm2_scale_expected,
            ) = _run_dump_decode_preprojected(selector_case)
            (
                _,
                fused_q_actual,
                quant_q_buffer_actual,
                mla_bmm1_scale_actual,
                mla_bmm2_scale_actual,
            ) = _run_dump_decode_fused_q_b(selector_case)

        try:
            _assert_q_b_selector_prefix_close(fused_q_actual, case.q_b_proj_output)
            torch.testing.assert_close(
                fused_q_actual,
                fused_q_expected,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                mla_bmm1_scale_actual,
                mla_bmm1_scale_expected,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                mla_bmm2_scale_actual,
                mla_bmm2_scale_expected,
                rtol=0.0,
                atol=0.0,
            )
            torch.testing.assert_close(
                quant_q_buffer_actual,
                quant_q_buffer_expected,
                rtol=0.0,
                atol=0.0,
            )
        except AssertionError as exc:
            failures.append(f"rank={group.rank} layer={group.layer_idx}: {exc}")

    if failures:
        raise AssertionError("\n\n".join(failures))
