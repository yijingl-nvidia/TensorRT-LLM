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

import importlib
import os
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pytest
import torch

from tensorrt_llm._torch.models.modeling_deepseekv3_mega_moe import Deepseekv3MegaMoE

_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_DEBUG_OUTPUT_DIR"
_RUNTIME_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_MEGAMOE_DEBUG_OUTPUT_DIR"
_DEFAULT_DEBUG_OUTPUT_DIR = "~/dev/debug_output"
_PHASE_ENV = "TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_PHASE"
_REF_USE_ORG_ENV = "TRTLLM_DEEPSEEKV3_MEGAMOE_REF_USE_SHARED_GATE_UP_WEIGHT_ORG"

_NUM_RANKS = 8
_DEFAULT_TOP_K = 8
_DEFAULT_N_GROUP = 8
_DEFAULT_TOPK_GROUP = 4
_DEFAULT_ROUTED_SCALING_FACTOR = 2.5
_DEFAULT_MAX_WIP_NUM_TOKENS = 4
_DEFAULT_PROFILE_WARMUP_ITERS = 20
_DEFAULT_PROFILE_ITERS = 100
_ERROR_EPS = 1e-12

# Tensor shape symbols used by the comments below:
# T: number of test tokens after the optional max-token slice.
# H: model hidden size.
# E: number of routed experts.
# I: expert intermediate size per TP partition.
# K: top-k routed experts per token.
# BH: ceil(H / 128), BI: ceil(I / 128), B2I: ceil((2 * I) / 128).
# PH: ceil(BH / 4), PI: ceil(BI / 4).


@dataclass(frozen=True)
class MegaMoeDumpGroup:
    rank: int
    weight_layer_idx: int
    activation_layer_idx: int


def _debug_output_dir() -> Path:
    return Path(
        os.environ.get(
            _DEBUG_OUTPUT_DIR_ENV,
            os.environ.get(_RUNTIME_DEBUG_OUTPUT_DIR_ENV, _DEFAULT_DEBUG_OUTPUT_DIR),
        )
    ).expanduser()


def _phase_enabled(phase: str) -> bool:
    selected_phase = os.environ.get(_PHASE_ENV, "both").strip().lower()
    phase_aliases = {
        "benchmark": "profile",
        "bench": "profile",
        "timing": "profile",
    }
    selected_phase = phase_aliases.get(selected_phase, selected_phase)
    if selected_phase == "both":
        return phase in ("reference", "test")
    return selected_phase in ("all", phase)


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _optional_float_env(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip().lower() in ("", "none", "inf", "infinity"):
        return None
    return float(value)


def _top_k() -> int:
    return _int_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_TOP_K", _DEFAULT_TOP_K)


def _n_group() -> int:
    return _int_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_N_GROUP", _DEFAULT_N_GROUP)


def _topk_group() -> int:
    return _int_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_TOPK_GROUP", _DEFAULT_TOPK_GROUP)


def _routed_scaling_factor() -> float:
    return _float_env(
        "TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_ROUTED_SCALING_FACTOR",
        _DEFAULT_ROUTED_SCALING_FACTOR,
    )


def _shared_swiglu_limit() -> float | None:
    return _optional_float_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_SHARED_SWIGLU_LIMIT")


def _routed_swiglu_limit() -> float | None:
    return _optional_float_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_ROUTED_SWIGLU_LIMIT")


def _max_wip_num_tokens() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_MAX_WIP_NUM_TOKENS",
        _DEFAULT_MAX_WIP_NUM_TOKENS,
    )


def _profile_warmup_iters() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_PROFILE_WARMUP_ITERS",
        _DEFAULT_PROFILE_WARMUP_ITERS,
    )


def _profile_iters() -> int:
    return _int_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_PROFILE_ITERS", _DEFAULT_PROFILE_ITERS)


def _baseline_tune_max_num_tokens(num_tokens: int) -> int:
    return _int_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_BASELINE_TUNE_MAX_NUM_TOKENS", num_tokens)


def _baseline_local_expert_offset() -> int:
    return _int_env("TRTLLM_DEEPSEEKV3_MEGAMOE_TEST_BASELINE_LOCAL_EXPERT_OFFSET", 0)


def _register_trtllm_baseline_custom_ops() -> None:
    importlib.import_module("tensorrt_llm._torch.custom_ops.torch_custom_ops")
    importlib.import_module("tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops")


def _load_tensor(group: MegaMoeDumpGroup, layer_idx: int, tensor_name: str) -> torch.Tensor:
    path = _debug_output_dir() / f"r{group.rank}_l{layer_idx}_{tensor_name}.pt"
    if not path.exists():
        pytest.skip(f"missing tensor dump: {path}")
    return torch.load(path, map_location="cpu").cuda()


def _load_saved_tensor(group: MegaMoeDumpGroup, tensor_name: str) -> torch.Tensor:
    path = _debug_output_dir() / f"r{group.rank}_l{group.weight_layer_idx}_{tensor_name}.pt"
    if not path.exists():
        pytest.skip(f"missing {path}; run the reference phase first")
    return torch.load(path, map_location="cpu").cuda()


def _save_tensor(group: MegaMoeDumpGroup, tensor_name: str, tensor: torch.Tensor) -> None:
    path = _debug_output_dir() / f"r{group.rank}_l{group.weight_layer_idx}_{tensor_name}.pt"
    torch.save(tensor.detach().cpu(), path)


def _save_summary(summary_name: str, rank: int, rel_error: float, abs_error: float) -> None:
    path = _debug_output_dir() / summary_name
    new_line = f"Test {rank}: rel {rel_error:.6e} abs {abs_error:.6e}"

    lines_by_rank: dict[int, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("Test "):
                continue
            test_idx = int(line.split(":", 1)[0].split()[1])
            lines_by_rank[test_idx] = line
    lines_by_rank[rank] = new_line
    path.write_text("\n".join(lines_by_rank[idx] for idx in sorted(lines_by_rank)) + "\n")


def _save_profile_summary(
    rank: int, baseline_stats: dict[str, float], wip_stats: dict[str, float]
) -> None:
    path = _debug_output_dir() / "mega_moe_profile_times.txt"
    speedup = baseline_stats["mean_ms"] / wip_stats["mean_ms"]
    new_line = (
        f"Test {rank}: baseline_mean_ms {baseline_stats['mean_ms']:.6f} "
        f"baseline_median_ms {baseline_stats['median_ms']:.6f} "
        f"wip_mean_ms {wip_stats['mean_ms']:.6f} "
        f"wip_median_ms {wip_stats['median_ms']:.6f} speedup {speedup:.3f}x"
    )

    lines_by_rank: dict[int, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if not line.startswith("Test "):
                continue
            test_idx = int(line.split(":", 1)[0].split()[1])
            lines_by_rank[test_idx] = line
    lines_by_rank[rank] = new_line
    path.write_text("\n".join(lines_by_rank[idx] for idx in sorted(lines_by_rank)) + "\n")


@contextmanager
def _ref_use_org_weight(enabled: bool):
    old_value = os.environ.get(_REF_USE_ORG_ENV)
    os.environ[_REF_USE_ORG_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(_REF_USE_ORG_ENV, None)
        else:
            os.environ[_REF_USE_ORG_ENV] = old_value


def _single_match(pattern: str) -> Path:
    matches = sorted(_debug_output_dir().glob(pattern))
    if not matches:
        pytest.skip(f"missing tensor dump matching {pattern!r} under {_debug_output_dir()}")
    if len(matches) > 1:
        raise AssertionError(f"expected one tensor dump for {pattern!r}, got {matches}")
    return matches[0]


def _dump_group(rank: int) -> MegaMoeDumpGroup:
    router_weight_path = _single_match(f"r{rank}_l*_router_weight.pt")
    hidden_states_path = _single_match(f"r{rank}_l*_hidden_states.pt")
    return MegaMoeDumpGroup(
        rank=rank,
        weight_layer_idx=int(router_weight_path.name.split("_", 2)[1][1:]),
        activation_layer_idx=int(hidden_states_path.name.split("_", 2)[1][1:]),
    )


def _require_cuda_and_ops(
    require_baseline_ops: bool = False, require_wip_ops: bool = False
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Deepseekv3MegaMoE dump tests require CUDA")
    required_ops = ["dsv3_router_gemm_op", "noaux_tc_op"]
    if require_baseline_ops:
        _register_trtllm_baseline_custom_ops()
        from tensorrt_llm._utils import is_sm_100f

        if not is_sm_100f():
            pytest.skip("Deepseekv3MegaMoE baseline dump tests require SM100-family GPUs")
        required_ops.extend(
            [
                "fp8_quantize_1x128_packed_ue8m0",
                "silu_and_mul",
                "fp8_quantize_1x128",
                "fp8_block_scale_moe_runner",
                "fp8_swap_ab_gemm",
            ]
        )
    if require_wip_ops:
        required_ops.extend(["glm5_expert_select_up_gate_silu", "glm5_expert_down_project"])
    missing_ops = [name for name in required_ops if not hasattr(torch.ops.trtllm, name)]
    if missing_ops:
        pytest.skip(f"missing torch.ops.trtllm ops: {missing_ops}")


def _load_inputs(
    group: MegaMoeDumpGroup, max_num_tokens: int | None = None
) -> dict[str, torch.Tensor]:
    weight_layer_idx = group.weight_layer_idx
    activation_layer_idx = group.activation_layer_idx
    hidden_states = _load_tensor(group, activation_layer_idx, "hidden_states")
    if max_num_tokens is not None:
        hidden_states = hidden_states[:max_num_tokens].contiguous()

    return {
        # torch.bfloat16, [T, H].
        "hidden_states": hidden_states,
        # torch.bfloat16, [E, H].
        "router_weight": _load_tensor(group, weight_layer_idx, "router_weight"),
        # torch.bfloat16, [E].
        "routing_bias": _load_tensor(group, weight_layer_idx, "routing_bias"),
        # torch.float8_e4m3fn, [2 * I, H].
        "shared_gate_up_weight": _load_tensor(group, weight_layer_idx, "shared_gate_up_weight"),
        # torch.int32, [2 * I, PH], four packed UE8M0 scale bytes per int32.
        "shared_gate_up_weight_scale": _load_tensor(
            group, weight_layer_idx, "shared_gate_up_weight_scale"
        ),
        # torch.float8_e4m3fn, [2 * I, H].
        "shared_gate_up_weight_org": _load_tensor(
            group, weight_layer_idx, "shared_gate_up_weight_org"
        ),
        # torch.float32, [B2I, BH].
        "shared_gate_up_weight_scale_org": _load_tensor(
            group, weight_layer_idx, "shared_gate_up_weight_scale_org"
        ),
        # torch.float8_e4m3fn, [E, 2 * I, H].
        "routed_w3_w1_weight": _load_tensor(group, weight_layer_idx, "routed_w3_w1_weight"),
        # torch.float32, [E, B2I, BH].
        "routed_w3_w1_weight_scale": _load_tensor(
            group, weight_layer_idx, "routed_w3_w1_weight_scaling_factor"
        ),
        # torch.float8_e4m3fn, [E, H, I].
        "routed_w2_weight": _load_tensor(group, weight_layer_idx, "routed_w2_weight"),
        # torch.float32, [E, BH, BI].
        "routed_w2_weight_scale": _load_tensor(
            group, weight_layer_idx, "routed_w2_weight_scaling_factor"
        ),
        # torch.float8_e4m3fn, [H, I].
        "shared_down_weight": _load_tensor(group, weight_layer_idx, "shared_down_weight"),
        # torch.int32, [H, PI], four packed UE8M0 scale bytes per int32.
        "shared_down_weight_scale": _load_tensor(
            group, weight_layer_idx, "shared_down_weight_scale"
        ),
        # torch.float8_e4m3fn, [H, I].
        "shared_down_weight_org": _load_tensor(group, weight_layer_idx, "shared_down_weight_org"),
        # torch.float32, [BH, BI].
        "shared_down_weight_scale_org": _load_tensor(
            group, weight_layer_idx, "shared_down_weight_scale_org"
        ),
    }


def _shared_down_project(
    shared_swiglu_output: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # shared_swiglu_output: torch.bfloat16, [T, I].
    # shared_down_weight: torch.float8_e4m3fn, [H, I].
    # shared_down_weight_scale: torch.int32 [H, PI] or torch.float32 [BH, BI].
    if shared_down_weight_scale.dtype == torch.int32:
        shared_weight = Deepseekv3MegaMoE._dequantize_fp8_1x128_packed_ue8m0_weight(
            shared_down_weight, shared_down_weight_scale
        )
    else:
        shared_weight = Deepseekv3MegaMoE._dequantize_fp8_128x128_block_weight(
            shared_down_weight, shared_down_weight_scale
        )
    shared_output = Deepseekv3MegaMoE._matmul_fp32_no_tf32(
        shared_swiglu_output.to(torch.float32), shared_weight.t()
    )
    # shared_output: output_dtype, [T, H].
    return shared_output.to(output_dtype)


def _run_trtllm_baseline(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    _register_trtllm_baseline_custom_ops()

    from tensorrt_llm import deep_gemm
    from tensorrt_llm._torch.modules.fused_moe.routing import RoutingMethodType

    hidden_states = tensors["hidden_states"]
    num_tokens, hidden_size = hidden_states.shape
    num_router_experts = tensors["router_weight"].shape[0]
    expert_intermediate_size = tensors["routed_w2_weight"].shape[-1]
    gate_up_output_size = 2 * expert_intermediate_size

    # shared_gate_up_input_fp8: torch.float8_e4m3fn, [T, H].
    # shared_gate_up_input_scale: torch.int32, [T, PH].
    shared_gate_up_input_fp8, shared_gate_up_input_scale = (
        torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(hidden_states)
    )
    # shared_gate_up_output: torch.bfloat16, [T, 2 * I].
    shared_gate_up_output = torch.empty(
        (num_tokens, gate_up_output_size),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    deep_gemm.fp8_gemm_nt(
        (shared_gate_up_input_fp8, shared_gate_up_input_scale),
        (tensors["shared_gate_up_weight"], tensors["shared_gate_up_weight_scale"]),
        shared_gate_up_output,
    )
    # shared_swiglu_output: torch.bfloat16, [T, I].
    shared_swiglu_output = torch.ops.trtllm.silu_and_mul(
        shared_gate_up_output, swiglu_limit=_shared_swiglu_limit()
    )

    # router_logits: torch.float32, [T, E].
    router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
        hidden_states,
        tensors["router_weight"].t(),
        bias=None,
        out_dtype=torch.float32,
    )

    # routed_input_fp8: torch.float8_e4m3fn, [T, H].
    # routed_input_scale: torch.float32, [BH, T].
    routed_input_fp8, routed_input_scale = torch.ops.trtllm.fp8_quantize_1x128(hidden_states)
    # routed_output: torch.bfloat16, [T, H].
    routed_output = torch.ops.trtllm.fp8_block_scale_moe_runner(
        router_logits,
        tensors["routing_bias"],
        routed_input_fp8,
        routed_input_scale,
        tensors["routed_w3_w1_weight"],
        tensors["routed_w3_w1_weight_scale"],
        tensors["routed_w2_weight"],
        tensors["routed_w2_weight_scale"],
        num_router_experts,
        _top_k(),
        _n_group(),
        _topk_group(),
        expert_intermediate_size,
        _baseline_local_expert_offset(),
        num_router_experts,
        _routed_scaling_factor(),
        int(RoutingMethodType.DeepSeekV3),
        None,
        None,
        0,
        _routed_swiglu_limit(),
        None,
        _baseline_tune_max_num_tokens(num_tokens),
        False,
    )

    # shared_output: torch.bfloat16, [T, H].
    shared_output = torch.ops.trtllm.fp8_swap_ab_gemm(
        shared_swiglu_output,
        tensors["shared_down_weight"],
        tensors["shared_down_weight_scale"],
    )
    assert shared_output.shape == (num_tokens, hidden_size)
    assert routed_output.shape == (num_tokens, hidden_size)
    return shared_output.add_(routed_output)


def _run_pytorch_decomposed(
    tensors: dict[str, torch.Tensor],
    *,
    use_org_shared_gate_up: bool,
    use_org_shared_down: bool,
) -> torch.Tensor:
    shared_gate_up_weight_org = tensors["shared_gate_up_weight_org"]
    shared_gate_up_weight_scale_org = tensors["shared_gate_up_weight_scale_org"]
    with _ref_use_org_weight(use_org_shared_gate_up):
        expert_indices, expert_weights, shared_swiglu_output, routed_swiglu_output = (
            Deepseekv3MegaMoE._run_pytorch_ref_mega_kernel(
                tensors["hidden_states"],
                tensors["router_weight"],
                tensors["routing_bias"],
                tensors["shared_gate_up_weight"],
                tensors["shared_gate_up_weight_scale"],
                shared_gate_up_weight_org,
                shared_gate_up_weight_scale_org,
                tensors["routed_w3_w1_weight"],
                tensors["routed_w3_w1_weight_scale"],
                _top_k(),
                _n_group(),
                _topk_group(),
                _routed_scaling_factor(),
                _shared_swiglu_limit(),
                _routed_swiglu_limit(),
            )
        )
    # expert_indices: torch.int32, [T, K].
    # expert_weights: torch.bfloat16, [T, K].
    # shared_swiglu_output: torch.bfloat16, [T, I].
    # routed_swiglu_output: torch.float32, [T, K, I].

    routed_output = Deepseekv3MegaMoE._routed_down_project_pytorch(
        routed_swiglu_output,
        expert_indices,
        expert_weights,
        tensors["routed_w2_weight"],
        tensors["routed_w2_weight_scale"],
        tensors["hidden_states"].dtype,
    )

    if use_org_shared_down:
        shared_down_weight = tensors["shared_down_weight_org"]
        shared_down_weight_scale = tensors["shared_down_weight_scale_org"]
    else:
        shared_down_weight = tensors["shared_down_weight"]
        shared_down_weight_scale = tensors["shared_down_weight_scale"]
    shared_output = _shared_down_project(
        shared_swiglu_output,
        shared_down_weight,
        shared_down_weight_scale,
        tensors["hidden_states"].dtype,
    )
    # shared_output, routed_output, and the returned hidden states:
    # torch.bfloat16, [T, H].
    return shared_output + routed_output


def _run_wip(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    expert_indices, expert_weights, slot_swiglu_output = Deepseekv3MegaMoE._run_wip_mega_kernel(
        tensors["hidden_states"],
        tensors["router_weight"],
        tensors["routing_bias"],
        tensors["shared_gate_up_weight_org"],
        tensors["shared_gate_up_weight_scale_org"],
        tensors["routed_w3_w1_weight"],
        tensors["routed_w3_w1_weight_scale"],
        _top_k(),
        _n_group(),
        _topk_group(),
        _routed_scaling_factor(),
        _shared_swiglu_limit(),
        _routed_swiglu_limit(),
    )
    # expert_indices: torch.int32, [T, K].
    # expert_weights: torch.float32, [T, K].
    # slot_swiglu_output: torch.float16, [T, K + 1, I].
    output = torch.empty_like(tensors["hidden_states"])
    # output: torch.bfloat16, [T, H].
    return torch.ops.trtllm.glm5_expert_down_project(
        slot_swiglu_output.contiguous(),
        expert_indices.contiguous(),
        expert_weights.contiguous(),
        tensors["routed_w2_weight"],
        tensors["routed_w2_weight_scale"],
        tensors["shared_down_weight_org"],
        tensors["shared_down_weight_scale_org"],
        output,
    )


def _profile_cuda_events(
    fn: Callable[[], torch.Tensor], warmup_iters: int, profile_iters: int
) -> tuple[torch.Tensor, dict[str, float]]:
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")
    if profile_iters <= 0:
        raise ValueError(f"profile_iters must be positive, got {profile_iters}")

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]
    for idx in range(profile_iters):
        starts[idx].record()
        fn()
        ends[idx].record()

    torch.cuda.synchronize()
    times_ms = torch.tensor(
        [starts[idx].elapsed_time(ends[idx]) for idx in range(profile_iters)],
        dtype=torch.float32,
    )
    times = [float(time_ms) for time_ms in times_ms.tolist()]
    stats = {
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "std_ms": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_ms": min(times),
        "max_ms": max(times),
    }
    return times_ms, stats


def _max_errors(
    actual: torch.Tensor, expected: torch.Tensor
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    # actual and expected: torch.bfloat16, [T, H].
    actual_fp32 = actual.to(torch.float32)
    expected_fp32 = expected.to(torch.float32)
    # abs_error_by_dim and rel_error_by_dim: torch.float32, [H].
    abs_error_by_dim = (actual_fp32 - expected_fp32).abs().amax(dim=0)
    rel_error_by_dim = (
        (actual_fp32 - expected_fp32).abs() / expected_fp32.abs().clamp_min(_ERROR_EPS)
    ).amax(dim=0)
    return (
        float(rel_error_by_dim.max().item()),
        float(abs_error_by_dim.max().item()),
        rel_error_by_dim,
        abs_error_by_dim,
    )


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_mega_moe_reference_phase(rank: int) -> None:
    if not _phase_enabled("reference"):
        pytest.skip(f"{_PHASE_ENV} disables reference phase")
    _require_cuda_and_ops(require_baseline_ops=True)
    group = _dump_group(rank)
    tensors = _load_inputs(group, max_num_tokens=_max_wip_num_tokens())

    with torch.inference_mode():
        # baseline_output: torch.bfloat16, [T, H].
        baseline_output = _run_trtllm_baseline(tensors)
        # pytorch_ref_output: torch.bfloat16, [T, H].
        pytorch_ref_output = _run_pytorch_decomposed(
            tensors,
            use_org_shared_gate_up=True,
            use_org_shared_down=True,
        )
        torch.cuda.synchronize()

    _save_tensor(group, "baseline_output", baseline_output)
    _save_tensor(group, "pytorch_ref_output", pytorch_ref_output)

    rel_error, abs_error, rel_error_by_dim, abs_error_by_dim = _max_errors(
        baseline_output, pytorch_ref_output
    )
    # rel_error_by_dim and abs_error_by_dim: torch.float32, [H].
    _save_tensor(group, "baseline_vs_pytorch_ref_rel_error_by_dim", rel_error_by_dim)
    _save_tensor(group, "baseline_vs_pytorch_ref_abs_error_by_dim", abs_error_by_dim)
    _save_summary("mega_moe_reference_errors.txt", rank, rel_error, abs_error)


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_mega_moe_wip_phase(rank: int) -> None:
    if not _phase_enabled("test"):
        pytest.skip(f"{_PHASE_ENV} disables test phase")
    _require_cuda_and_ops(require_wip_ops=True)
    group = _dump_group(rank)
    # pytorch_ref_output: torch.bfloat16, [T, H].
    pytorch_ref_output = _load_saved_tensor(group, "pytorch_ref_output")
    # reference_abs_error_by_dim: torch.float32, [H].
    reference_abs_error_by_dim = _load_saved_tensor(
        group, "baseline_vs_pytorch_ref_abs_error_by_dim"
    )
    reference_abs_threshold = float(reference_abs_error_by_dim.max().item())

    tensors = _load_inputs(group, max_num_tokens=_max_wip_num_tokens())
    with torch.inference_mode():
        # wip_output: torch.bfloat16, [T, H].
        wip_output = _run_wip(tensors)
        torch.cuda.synchronize()

    _save_tensor(group, "wip_output", wip_output)

    rel_error, abs_error, rel_error_by_dim, abs_error_by_dim = _max_errors(
        wip_output, pytorch_ref_output
    )
    # rel_error_by_dim and abs_error_by_dim: torch.float32, [H].
    _save_tensor(group, "wip_vs_pytorch_ref_rel_error_by_dim", rel_error_by_dim)
    _save_tensor(group, "wip_vs_pytorch_ref_abs_error_by_dim", abs_error_by_dim)
    _save_summary("mega_moe_wip_errors.txt", rank, rel_error, abs_error)

    assert abs_error <= reference_abs_threshold, (
        f"rank {rank} WIP output differs from PyTorch reference: rel={rel_error} "
        f"abs={abs_error} reference_abs_threshold={reference_abs_threshold}"
    )


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_mega_moe_profile_phase(rank: int) -> None:
    if not _phase_enabled("profile"):
        pytest.skip(f"{_PHASE_ENV} disables profile phase")
    _require_cuda_and_ops(require_baseline_ops=True, require_wip_ops=True)
    group = _dump_group(rank)
    tensors = _load_inputs(group, max_num_tokens=_max_wip_num_tokens())
    warmup_iters = _profile_warmup_iters()
    profile_iters = _profile_iters()

    with torch.inference_mode():
        # baseline_times_ms and wip_times_ms: torch.float32, [profile_iters].
        baseline_times_ms, baseline_stats = _profile_cuda_events(
            lambda: _run_trtllm_baseline(tensors), warmup_iters, profile_iters
        )
        wip_times_ms, wip_stats = _profile_cuda_events(
            lambda: _run_wip(tensors), warmup_iters, profile_iters
        )

    _save_tensor(group, "baseline_profile_times_ms", baseline_times_ms)
    _save_tensor(group, "wip_profile_times_ms", wip_times_ms)
    _save_profile_summary(rank, baseline_stats, wip_stats)

    speedup = baseline_stats["mean_ms"] / wip_stats["mean_ms"]
    print(
        f"rank {rank}: baseline {baseline_stats['mean_ms']:.6f} ms, "
        f"wip {wip_stats['mean_ms']:.6f} ms, speedup {speedup:.3f}x"
    )
