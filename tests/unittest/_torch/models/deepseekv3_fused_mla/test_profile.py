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

from typing import Callable

import pytest
import torch

from tests.unittest._torch.models.deepseekv3_fused_mla.decode_case import (
    _build_dump_decode_q_b_case,
    _run_dump_decode_backend,
    _run_dump_decode_fused_q_b,
    _run_dump_decode_preprojected,
)
from tests.unittest._torch.models.deepseekv3_fused_mla.dump_utils import (
    _PROFILE_ENV,
    _profile_enabled,
    _profile_iterations,
)


def _measure_cuda_ms(
    func: Callable[[], object],
    iterations: int,
    warmup_iterations: int = 10,
) -> float:
    """
    Measure average CUDA elapsed time for a callable.

    The callable may launch one or more kernels. CUDA events bracket only the
    measured loop, after a warmup loop. The result is intended for local
    profiling through the opt-in pytest profile test, not for CI assertions.

    Args
    - func: Callable[[], object], function that launches the work to time.
    - iterations: int, number of measured loop iterations.
    - warmup_iterations: int, number of unmeasured warmup iterations.

    Returns
    - average_ms: float, average elapsed CUDA event time per callable invocation.
    """
    for _ in range(warmup_iterations):
        func()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        func()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)) / iterations


def test_deepseekv3_fused_mla_dump_decode_q_b_proj_profile_rank0() -> None:
    """
    Profile dumped q_b preprojected, fused, and backend decode paths.

    The test is skipped unless TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_PROFILE=1.
    It prints CUDA-event average milliseconds for the same rank-0 dump-backed
    decode setup used by the accuracy tests.

    Args
    - None.

    Returns
    - None: successful return prints timing data and performs no accuracy
        assertion.
    """
    if not _profile_enabled():
        pytest.skip(f"set {_PROFILE_ENV}=1 to run dump-backed fused MLA profiling")

    iterations = _profile_iterations()
    case = _build_dump_decode_q_b_case(rank=0)

    with torch.inference_mode():
        preprojected_ms = _measure_cuda_ms(
            lambda: _run_dump_decode_preprojected(case),
            iterations,
        )
        fused_q_b_ms = _measure_cuda_ms(
            lambda: _run_dump_decode_fused_q_b(case),
            iterations,
        )
        backend_ms = _measure_cuda_ms(
            lambda: _run_dump_decode_backend(case),
            iterations,
        )

    print(
        "dump_q_b_decode_profile "
        f"rank={case.group.rank} layer={case.group.layer_idx} "
        f"iters={iterations} "
        f"preprojected_ms={preprojected_ms:.6f} "
        f"fused_q_b_ms={fused_q_b_ms:.6f} "
        f"backend_ms={backend_ms:.6f}"
    )
