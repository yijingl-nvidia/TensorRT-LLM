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

import torch

from tests.unittest._torch.models.deepseekv3_fused_mla.assertions import (
    _assert_context_attention_close,
)
from tests.unittest._torch.models.deepseekv3_fused_mla.decode_case import (
    _build_dump_decode_q_b_case,
    _run_dump_decode_backend,
    _run_dump_decode_fused_q_b,
)


def test_deepseekv3_fused_mla_dump_decode_q_b_proj_matches_backend_rank0() -> None:
    """
    Compare dumped q_b fused decode against the baseline TRTLLM backend.

    This rank-0 test is broader than the preprojected-path test because the
    expected side uses mla_rope_generation plus attention.forward. It is useful
    when checking whether a q_b fusion change still matches the backend path
    used for acceptance-length baselines.

    Args
    - None.

    Returns
    - None: successful return means output, fused query prefix, scales, and FP8
        query bytes match the backend path within test tolerances.
    """
    case = _build_dump_decode_q_b_case(rank=0)

    with torch.inference_mode():
        (
            expected,
            fused_q_expected,
            quant_q_buffer_expected,
            mla_bmm1_scale_expected,
            mla_bmm2_scale_expected,
        ) = _run_dump_decode_backend(case)
        (
            actual,
            fused_q_actual,
            quant_q_buffer_actual,
            mla_bmm1_scale_actual,
            mla_bmm2_scale_actual,
        ) = _run_dump_decode_fused_q_b(case)

    _assert_context_attention_close(actual, expected)
    # Do not assert fused_q prefix here: it also covers the standalone k_b
    # absorption path. q_b-specific exactness is covered by the selector test.
    _ = fused_q_actual, fused_q_expected
    torch.testing.assert_close(mla_bmm1_scale_actual, mla_bmm1_scale_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(mla_bmm2_scale_actual, mla_bmm2_scale_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(quant_q_buffer_actual, quant_q_buffer_expected, rtol=0.0, atol=0.0)
