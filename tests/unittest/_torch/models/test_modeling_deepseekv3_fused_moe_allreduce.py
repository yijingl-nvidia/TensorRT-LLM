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

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams
from tensorrt_llm._torch.models.modeling_deepseekv3_fused_moe import Deepseekv3FusedMoE
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.mapping import Mapping

_NUM_RANKS = 8
_HELPER_MODULE_NAME = "_deepseekv3_fused_moe_test_helpers"
_HELPER_FILE = Path(__file__).with_name("test_modeling_deepseekv3_fused_moe.py")
_BASELINE_ABS_THRESHOLD_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_BASELINE_ABS_THRESHOLD"
_FUSED_ABS_THRESHOLD_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_FUSED_ABS_THRESHOLD"
_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD_ENV = (
    "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD"
)
_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD_ENV = (
    "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD"
)
_RESIDUAL_SEED_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_RESIDUAL_SEED"
_RMS_NORM_EPS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_RMS_NORM_EPS"
_DEFAULT_BASELINE_ABS_THRESHOLD = 0.02
_DEFAULT_FUSED_ABS_THRESHOLD = 0.02
_DEFAULT_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD = 3.90625e-03
_DEFAULT_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD = 1.5625e-02
_DEFAULT_RESIDUAL_SEED = 20260602
_DEFAULT_RMS_NORM_EPS = 1e-6


def _helpers() -> ModuleType:
    module = sys.modules.get(_HELPER_MODULE_NAME)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(_HELPER_MODULE_NAME, _HELPER_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to import fused MoE test helpers from {_HELPER_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HELPER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _rms_norm_eps() -> float:
    return _float_env(_RMS_NORM_EPS_ENV, _DEFAULT_RMS_NORM_EPS)


def _distributed_world_size() -> int:
    return MPI.COMM_WORLD.Get_size()


def _distributed_rank() -> int:
    return MPI.COMM_WORLD.Get_rank()


def _make_residual_and_norm_weight(shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_int_env(_RESIDUAL_SEED_ENV, _DEFAULT_RESIDUAL_SEED))
    residual = (0.25 * torch.randn(shape, dtype=torch.float32, generator=generator)).to(
        torch.bfloat16
    )
    norm_weight = (0.5 + torch.rand((shape[-1],), dtype=torch.float32, generator=generator)).to(
        torch.bfloat16
    )
    return residual.cuda(), norm_weight.cuda()


def _rms_norm(tensor: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    tensor_fp32 = tensor.to(torch.float32)
    output = tensor_fp32 * torch.rsqrt(tensor_fp32.pow(2).mean(-1, keepdim=True) + eps)
    output *= norm_weight.to(torch.float32)
    return output.to(torch.bfloat16)


def _post_moe_allreduce_residual_rms_norm_reference(
    local_outputs: list[torch.Tensor],
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This mirrors DeepseekV3DecoderLayer's post-MoE production call:
    # AllReduceFusionOp.RESIDUAL_RMS_NORM first sums the TP-local MoE outputs,
    # then writes the residual-add output and RMS-normalized hidden states.
    fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
    assert fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    allreduced_output = torch.stack(
        [local_output.to(torch.float32) for local_output in local_outputs],
        dim=0,
    ).sum(dim=0)
    allreduced_output = allreduced_output.to(torch.bfloat16)
    residual_out = (allreduced_output + residual).to(torch.bfloat16)
    hidden_states = _rms_norm(residual_out, norm_weight, _rms_norm_eps())
    return hidden_states, residual_out


def _run_fused_expert_down_local(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    helpers = _helpers()
    helpers._ensure_fused_expert_up_prepacked_tensors(tensors)
    expert_indices, expert_weights, slot_swiglu_output = (
        Deepseekv3FusedMoE._run_dsv3_fused_expert_up(
            tensors["hidden_states"],
            tensors["router_weight"],
            tensors["routing_bias"],
            tensors["shared_gate_up_weight_packed_fused_expert_up"],
            tensors["shared_gate_up_weight_scale_org"],
            tensors["routed_w3_w1_weight_packed_fused_expert_up"],
            tensors["routed_w3_w1_weight_scale"],
            helpers._top_k(),
            helpers._n_group(),
            helpers._topk_group(),
            helpers._routed_scaling_factor(),
        )
    )
    output = torch.empty_like(tensors["hidden_states"])
    return Deepseekv3FusedMoE._run_dsv3_fused_expert_down_chunked(
        slot_swiglu_output,
        expert_indices,
        expert_weights,
        tensors["routed_w2_weight"],
        tensors["routed_w2_weight_scale"],
        tensors["shared_down_weight_org"],
        tensors["shared_down_weight_scale_org"],
        output,
    )


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    helpers = _helpers()
    rel_error, abs_error, _, _ = helpers._max_errors(actual, expected)
    return rel_error, abs_error


def _write_summary(
    baseline_rel: float,
    baseline_abs: float,
    baseline_residual_rel: float,
    baseline_residual_abs: float,
    fused_rel: float,
    fused_abs: float,
    fused_residual_rel: float,
    fused_residual_abs: float,
) -> None:
    helpers = _helpers()
    path = helpers._debug_output_dir() / "fused_moe_allreduce_reference_errors.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        " ".join(
            [
                "Test allreduce:",
                f"baseline_rel {baseline_rel:.6e}",
                f"baseline_abs {baseline_abs:.6e}",
                f"baseline_residual_rel {baseline_residual_rel:.6e}",
                f"baseline_residual_abs {baseline_residual_abs:.6e}",
                f"fused_rel {fused_rel:.6e}",
                f"fused_abs {fused_abs:.6e}",
                f"fused_residual_rel {fused_residual_rel:.6e}",
                f"fused_residual_abs {fused_residual_abs:.6e}",
            ]
        )
        + "\n"
    )


def test_deepseekv3_fused_moe_post_moe_allreduce_reference() -> None:
    if _distributed_world_size() > 1:
        pytest.skip("single-process reference test is disabled under distributed launch")

    helpers = _helpers()
    helpers._require_cuda_and_ops(require_baseline_ops=True, require_fused_ops=True)

    baseline_local_outputs = []
    pytorch_local_outputs = []
    fused_local_outputs = []
    residual = None
    norm_weight = None

    with torch.inference_mode():
        for rank in range(_NUM_RANKS):
            group = helpers._dump_group(rank)
            tensors = helpers._load_inputs(
                group,
                max_num_tokens=helpers._max_fused_kernel_num_tokens(),
                include_fused_kernel_tensors=True,
            )
            if residual is None or norm_weight is None:
                residual, norm_weight = _make_residual_and_norm_weight(
                    tensors["hidden_states"].shape
                )
            else:
                assert tuple(tensors["hidden_states"].shape) == tuple(residual.shape)

            old_moe = helpers._build_old_moe_baseline(tensors, group)
            baseline_local_outputs.append(helpers._run_multi_stream_baseline(old_moe, tensors))
            pytorch_local_outputs.append(helpers._run_pytorch_reference(tensors))
            fused_local_outputs.append(_run_fused_expert_down_local(tensors))
            torch.cuda.synchronize()
            del old_moe, tensors
            torch.cuda.empty_cache()

        assert residual is not None
        assert norm_weight is not None
        baseline_hidden_states, baseline_residual = _post_moe_allreduce_residual_rms_norm_reference(
            baseline_local_outputs,
            residual,
            norm_weight,
        )
        pytorch_hidden_states, pytorch_residual = _post_moe_allreduce_residual_rms_norm_reference(
            pytorch_local_outputs,
            residual,
            norm_weight,
        )
        fused_hidden_states, fused_residual = _post_moe_allreduce_residual_rms_norm_reference(
            fused_local_outputs,
            residual,
            norm_weight,
        )
        torch.cuda.synchronize()

    baseline_rel, baseline_abs = _max_errors(baseline_hidden_states, pytorch_hidden_states)
    baseline_residual_rel, baseline_residual_abs = _max_errors(
        baseline_residual,
        pytorch_residual,
    )
    fused_rel, fused_abs = _max_errors(fused_hidden_states, pytorch_hidden_states)
    fused_residual_rel, fused_residual_abs = _max_errors(fused_residual, pytorch_residual)
    _write_summary(
        baseline_rel,
        baseline_abs,
        baseline_residual_rel,
        baseline_residual_abs,
        fused_rel,
        fused_abs,
        fused_residual_rel,
        fused_residual_abs,
    )

    baseline_abs_threshold = _float_env(
        _BASELINE_ABS_THRESHOLD_ENV,
        _DEFAULT_BASELINE_ABS_THRESHOLD,
    )
    fused_abs_threshold = _float_env(
        _FUSED_ABS_THRESHOLD_ENV,
        _DEFAULT_FUSED_ABS_THRESHOLD,
    )
    print(
        "DeepSeekV3 fused MoE post-MoE AllReduce reference: "
        f"baseline_abs {baseline_abs:.6e}, "
        f"baseline_residual_abs {baseline_residual_abs:.6e}, "
        f"fused_abs {fused_abs:.6e}, "
        f"fused_residual_abs {fused_residual_abs:.6e}, "
        f"baseline_abs_threshold {baseline_abs_threshold:.6e}, "
        f"fused_abs_threshold {fused_abs_threshold:.6e}"
    )

    assert baseline_abs <= baseline_abs_threshold
    assert baseline_residual_abs <= baseline_abs_threshold
    assert fused_abs <= fused_abs_threshold
    assert fused_residual_abs <= fused_abs_threshold


def _init_mpi_for_trtllm_allreduce() -> tuple[int, int, MPI.Comm]:
    world_size = _distributed_world_size()
    if world_size == 1:
        pytest.skip(
            "run this test with: mpirun --allow-run-as-root --oversubscribe -n 8 "
            "python3 -m pytest -q "
            "tests/unittest/_torch/models/test_modeling_deepseekv3_fused_moe_allreduce.py"
        )
    if world_size != _NUM_RANKS:
        pytest.skip(f"expected WORLD_SIZE={_NUM_RANKS}, got {world_size}")
    if os.environ.get("TLLM_DISABLE_MPI") == "1":
        pytest.skip("MPI-backed TRT-LLM AllReduce test requires TLLM_DISABLE_MPI to be unset")

    rank = _distributed_rank()
    torch.cuda.set_device(rank)
    return rank, world_size, MPI.COMM_WORLD


def _run_trtllm_allreduce_residual_rms_norm(
    allreduce: AllReduce,
    local_output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return allreduce(
        local_output,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=_rms_norm_eps(),
        ),
    )


def _write_wip_summary(rank_results: list[dict[str, float]]) -> None:
    helpers = _helpers()
    path = helpers._debug_output_dir() / "fused_moe_allreduce_wip_vs_baseline_errors.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for result in sorted(rank_results, key=lambda item: item["rank"]):
        lines.append(
            " ".join(
                [
                    f"Test {int(result['rank'])}:",
                    f"wip_vs_baseline_residual_rel {result['residual_rel']:.6e}",
                    f"wip_vs_baseline_residual_abs {result['residual_abs']:.6e}",
                    f"wip_vs_baseline_hidden_rel {result['hidden_rel']:.6e}",
                    f"wip_vs_baseline_hidden_abs {result['hidden_abs']:.6e}",
                ]
            )
        )
    max_residual_abs = max(result["residual_abs"] for result in rank_results)
    max_hidden_abs = max(result["hidden_abs"] for result in rank_results)
    lines.append(
        " ".join(
            [
                "Average over 8 ranks:",
                f"max_wip_vs_baseline_residual_abs {max_residual_abs:.6e}",
                f"max_wip_vs_baseline_hidden_abs {max_hidden_abs:.6e}",
            ]
        )
    )
    path.write_text("\n".join(lines) + "\n")


def test_deepseekv3_fused_moe_post_moe_allreduce_wip_path() -> None:
    rank, world_size, comm = _init_mpi_for_trtllm_allreduce()
    helpers = _helpers()
    helpers._require_cuda_and_ops(require_baseline_ops=True, require_fused_ops=True)

    with torch.inference_mode():
        group = helpers._dump_group(rank)
        tensors = helpers._load_inputs(
            group,
            max_num_tokens=helpers._max_fused_kernel_num_tokens(),
            include_fused_kernel_tensors=True,
        )
        residual, norm_weight = _make_residual_and_norm_weight(tensors["hidden_states"].shape)
        old_moe = helpers._build_old_moe_baseline(tensors, group)
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=world_size,
            tp_size=world_size,
            pp_size=1,
        )
        allreduce = AllReduce(
            mapping=mapping,
            strategy=AllReduceStrategy.NCCL,
            dtype=torch.bfloat16,
        )

        baseline_local_output = helpers._run_multi_stream_baseline(old_moe, tensors)
        wip_local_output = _run_fused_expert_down_local(tensors)
        baseline_hidden_states, baseline_residual = _run_trtllm_allreduce_residual_rms_norm(
            allreduce,
            baseline_local_output,
            residual,
            norm_weight,
        )
        wip_hidden_states, wip_residual = _run_trtllm_allreduce_residual_rms_norm(
            allreduce,
            wip_local_output,
            residual,
            norm_weight,
        )
        torch.cuda.synchronize()

    residual_rel, residual_abs = _max_errors(wip_residual, baseline_residual)
    hidden_rel, hidden_abs = _max_errors(wip_hidden_states, baseline_hidden_states)
    result = {
        "rank": float(rank),
        "residual_rel": residual_rel,
        "residual_abs": residual_abs,
        "hidden_rel": hidden_rel,
        "hidden_abs": hidden_abs,
    }
    gathered_results = [item for item in comm.allgather(result) if item is not None]
    if rank == 0:
        _write_wip_summary(gathered_results)
        max_residual_abs = max(item["residual_abs"] for item in gathered_results)
        max_hidden_abs = max(item["hidden_abs"] for item in gathered_results)
        print(
            "DeepSeekV3 fused MoE post-MoE AllReduce WIP path: "
            f"max_wip_vs_baseline_residual_abs {max_residual_abs:.6e}, "
            f"max_wip_vs_baseline_hidden_abs {max_hidden_abs:.6e}"
        )
    comm.Barrier()

    residual_abs_threshold = _float_env(
        _WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD_ENV,
        _DEFAULT_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD,
    )
    hidden_abs_threshold = _float_env(
        _WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD_ENV,
        _DEFAULT_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD,
    )
    assert residual_abs <= residual_abs_threshold
    assert hidden_abs <= hidden_abs_threshold
