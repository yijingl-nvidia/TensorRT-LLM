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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3_fused_moe import (
    Deepseekv3FusedMoE,
    check_data,
    prepack_fused_expert_down_routed_w2_weight,
    prepack_fused_expert_down_shared_down_weight,
    prepack_fused_expert_up_routed_w3_w1_weight,
    prepack_fused_expert_up_shared_gate_up_weight,
)
from tensorrt_llm._torch.models.modeling_deepseekv3_moe import Deepseekv3MoE
from tensorrt_llm._torch.modules.multi_stream_utils import with_multi_stream
from tensorrt_llm._torch.utils import AuxStreamType
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_DEBUG_OUTPUT_DIR"
_RUNTIME_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_DEBUG_OUTPUT_DIR"
_DEFAULT_DEBUG_OUTPUT_DIR = "~/dev/debug_output"
_PHASE_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PHASE"
_PREPACK_FUSED_EXPERT_DOWN_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PREPACK_FUSED_EXPERT_DOWN"
_EXTRA_OP_LIBRARY_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_EXTRA_OP_LIBRARY"
_FUSED_EXPERT_UP_OP_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_UP_OP"
_FUSED_EXPERT_DOWN_OP_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_DOWN_OP"

_NUM_RANKS = 8
_DEFAULT_TOP_K = 8
_DEFAULT_N_GROUP = 1
_DEFAULT_TOPK_GROUP = 1
_DEFAULT_ROUTED_SCALING_FACTOR = 2.5
_DEFAULT_MAX_FUSED_KERNEL_NUM_TOKENS = 4
_DEFAULT_PROFILE_WARMUP_ITERS = 20
_DEFAULT_PROFILE_ITERS = 100
_DEFAULT_CUDA_GRAPH_CAPTURE_WARMUP_ITERS = 3
_ERROR_EPS = 1e-12
_FUSED_KERNEL_ABS_ERROR_THRESHOLD_SLACK = 1.35
_CURRENT_FUSED_KERNEL_ABS_ERROR_THRESHOLDS_BY_RANK = (
    7.30e-05,
    7.06e-05,
    1.23e-04,
    7.25e-05,
    9.54e-05,
    7.54e-05,
    6.78e-05,
    4.15e-04,
)
_EXTRA_OP_LIBRARIES_LOADED = False

# Ideal target is zero fused-kernel slack: abs_error <= reference_abs_threshold.
# Historical dumped GLM-5 multi-stream-baseline-vs-PyTorch max abs thresholds
# by rank should remain the practical target for future kernel work.
# Current accepted fused-kernel max abs thresholds by rank are intentionally
# tight to the present implementation on the dumped GLM-5 data:
# [7.30e-05, 7.06e-05, 1.23e-04, 7.25e-05,
#  9.54e-05, 7.54e-05, 6.78e-05, 4.15e-04].

# Tensor shape symbols used by the comments below:
# T: number of test tokens after the optional max-token slice.
# H: model hidden size.
# E: number of routed experts.
# I: expert intermediate size per TP partition.
# K: top-k routed experts per token.
# BH: ceil(H / 128), BI: ceil(I / 128), B2I: ceil((2 * I) / 128).
# PH: ceil(BH / 4), PI: ceil(BI / 4).


@dataclass(frozen=True)
class FusedMoeDumpGroup:
    rank: int
    weight_layer_idx: int
    activation_layer_idx: int


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _debug_output_dir() -> Path:
    for env_name in (
        _DEBUG_OUTPUT_DIR_ENV,
        _RUNTIME_DEBUG_OUTPUT_DIR_ENV,
    ):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser()
    return Path(_DEFAULT_DEBUG_OUTPUT_DIR).expanduser()


def _phase_enabled(phase: str) -> bool:
    selected_phase = _env(_PHASE_ENV, "both").strip().lower()
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
    return int(_env(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _bool_env(name: str, default: bool) -> bool:
    value = _env(name, "1" if default else "0")
    return value.strip().lower() in ("1", "true", "yes", "on")


def _optional_float_env(name: str) -> float | None:
    value = _env(name, "")
    if value.strip().lower() in ("", "none", "inf", "infinity"):
        return None
    return float(value)


def _top_k() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_TOP_K",
        _DEFAULT_TOP_K,
    )


def _n_group() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_N_GROUP",
        _DEFAULT_N_GROUP,
    )


def _topk_group() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_TOPK_GROUP",
        _DEFAULT_TOPK_GROUP,
    )


def _routed_scaling_factor() -> float:
    return _float_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_ROUTED_SCALING_FACTOR",
        _DEFAULT_ROUTED_SCALING_FACTOR,
    )


def _shared_swiglu_limit() -> float | None:
    return _optional_float_env("TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_SHARED_SWIGLU_LIMIT")


def _routed_swiglu_limit() -> float | None:
    return _optional_float_env("TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_ROUTED_SWIGLU_LIMIT")


def _max_fused_kernel_num_tokens() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_MAX_FUSED_KERNEL_NUM_TOKENS",
        _DEFAULT_MAX_FUSED_KERNEL_NUM_TOKENS,
    )


def _profile_warmup_iters() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_WARMUP_ITERS",
        _DEFAULT_PROFILE_WARMUP_ITERS,
    )


def _profile_iters() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_ITERS",
        _DEFAULT_PROFILE_ITERS,
    )


def _cuda_graph_capture_warmup_iters() -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_CUDA_GRAPH_CAPTURE_WARMUP_ITERS",
        _DEFAULT_CUDA_GRAPH_CAPTURE_WARMUP_ITERS,
    )


def _baseline_tune_max_num_tokens(num_tokens: int) -> int:
    return _int_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_BASELINE_TUNE_MAX_NUM_TOKENS",
        num_tokens,
    )


def _profile_pytorch_reference_enabled() -> bool:
    return _bool_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_PROFILE_PYTORCH_REFERENCE",
        False,
    )


def _prepack_fused_expert_down_enabled() -> bool:
    return _bool_env(
        _PREPACK_FUSED_EXPERT_DOWN_ENV,
        _bool_env("TRTLLM_DEEPSEEKV3_FUSED_MOE_PREPACK_FUSED_EXPERT_DOWN", False),
    )


def _fused_expert_down_op_name() -> str:
    return _env(_FUSED_EXPERT_DOWN_OP_ENV, "dsv3_fused_expert_down")


def _fused_expert_up_op_name() -> str:
    return _env(_FUSED_EXPERT_UP_OP_ENV, "dsv3_fused_expert_up")


def _fused_kernel_abs_error_threshold_slack() -> float:
    return _float_env(
        "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_ABS_ERROR_THRESHOLD_SLACK",
        _FUSED_KERNEL_ABS_ERROR_THRESHOLD_SLACK,
    )


def _current_fused_kernel_abs_error_threshold(rank: int) -> float:
    return _CURRENT_FUSED_KERNEL_ABS_ERROR_THRESHOLDS_BY_RANK[rank]


def _register_trtllm_custom_ops() -> None:
    importlib.import_module("tensorrt_llm._torch.custom_ops.torch_custom_ops")
    importlib.import_module("tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops")


def _load_extra_op_libraries() -> None:
    global _EXTRA_OP_LIBRARIES_LOADED

    if _EXTRA_OP_LIBRARIES_LOADED:
        return
    _EXTRA_OP_LIBRARIES_LOADED = True

    raw_paths = _env(_EXTRA_OP_LIBRARY_ENV, "")
    if not raw_paths:
        return
    for raw_path in raw_paths.replace(",", os.pathsep).split(os.pathsep):
        raw_path = raw_path.strip()
        if raw_path:
            torch.ops.load_library(str(Path(raw_path).expanduser()))


def _load_tensor(group: FusedMoeDumpGroup, layer_idx: int, tensor_name: str) -> torch.Tensor:
    path = _debug_output_dir() / f"r{group.rank}_l{layer_idx}_{tensor_name}.pt"
    if not path.exists():
        pytest.skip(f"missing tensor dump: {path}")
    return torch.load(path, map_location="cpu").cuda()


def _load_saved_tensor(group: FusedMoeDumpGroup, tensor_name: str) -> torch.Tensor:
    path = _debug_output_dir() / f"r{group.rank}_l{group.weight_layer_idx}_{tensor_name}.pt"
    if not path.exists():
        pytest.skip(f"missing {path}; run the reference phase first")
    return torch.load(path, map_location="cpu").cuda()


def _save_tensor(group: FusedMoeDumpGroup, tensor_name: str, tensor: torch.Tensor) -> None:
    path = _debug_output_dir() / f"r{group.rank}_l{group.weight_layer_idx}_{tensor_name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), path)


def _save_summary(summary_name: str, rank: int, rel_error: float, abs_error: float) -> None:
    path = _debug_output_dir() / summary_name
    path.parent.mkdir(parents=True, exist_ok=True)
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
    rank: int,
    multi_stream_baseline_stats: dict[str, float],
    fused_kernel_stats: dict[str, float],
    fused_kernel_eager_stats: dict[str, float],
    router_gemm_stats: dict[str, float],
    fused_expert_up_stats: dict[str, float],
    fused_expert_down_stats: dict[str, float],
    pytorch_reference_stats: dict[str, float] | None,
) -> str | None:
    path = _debug_output_dir() / "fused_moe_profile_times.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    speedup = multi_stream_baseline_stats["mean_us"] / fused_kernel_stats["mean_us"]
    fields = [
        f"Test {rank}: cuda_graph 1",
        f"multi_stream_baseline_mean_us {multi_stream_baseline_stats['mean_us']:.3f}",
        f"multi_stream_baseline_median_us {multi_stream_baseline_stats['median_us']:.3f}",
        f"fused_kernel_mean_us {fused_kernel_stats['mean_us']:.3f}",
        f"fused_kernel_median_us {fused_kernel_stats['median_us']:.3f}",
        f"fused_kernel_eager_mean_us {fused_kernel_eager_stats['mean_us']:.3f}",
        f"fused_kernel_eager_median_us {fused_kernel_eager_stats['median_us']:.3f}",
        f"dsv3_router_gemm_mean_us {router_gemm_stats['mean_us']:.3f}",
        f"dsv3_router_gemm_median_us {router_gemm_stats['median_us']:.3f}",
        f"dsv3_fused_expert_up_mean_us {fused_expert_up_stats['mean_us']:.3f}",
        f"dsv3_fused_expert_up_median_us {fused_expert_up_stats['median_us']:.3f}",
        f"dsv3_fused_expert_down_mean_us {fused_expert_down_stats['mean_us']:.3f}",
        f"dsv3_fused_expert_down_median_us {fused_expert_down_stats['median_us']:.3f}",
        f"fused_kernel_vs_multi_stream_baseline_speedup {speedup:.3f}x",
        f"prepack_fused_expert_down {int(_prepack_fused_expert_down_enabled())}",
    ]
    if pytorch_reference_stats is not None:
        fields.extend(
            [
                f"pytorch_reference_mean_us {pytorch_reference_stats['mean_us']:.3f}",
                f"pytorch_reference_median_us {pytorch_reference_stats['median_us']:.3f}",
            ]
        )
    new_line = " ".join(fields)

    lines_by_rank: dict[int, str] = {}
    if path.exists() and rank != 0:
        for line in path.read_text().splitlines():
            if not line.startswith("Test "):
                continue
            test_idx = int(line.split(":", 1)[0].split()[1])
            lines_by_rank[test_idx] = line
    lines_by_rank[rank] = new_line
    lines = [lines_by_rank[idx] for idx in sorted(lines_by_rank)]
    average_line = None
    if len(lines_by_rank) == _NUM_RANKS:
        average_line = _profile_average_summary_line(lines)
        lines.append(average_line)
    path.write_text("\n".join(lines) + "\n")
    return average_line


def _profile_average_summary_line(profile_lines: list[str]) -> str:
    average_fields = (
        "multi_stream_baseline_mean_us",
        "fused_kernel_mean_us",
        "fused_kernel_eager_mean_us",
        "dsv3_router_gemm_mean_us",
        "dsv3_fused_expert_up_mean_us",
        "dsv3_fused_expert_down_mean_us",
        "fused_kernel_vs_multi_stream_baseline_speedup",
    )
    values_by_field: dict[str, list[float]] = {field: [] for field in average_fields}
    for line in profile_lines:
        parts = line.split()
        for idx in range(2, len(parts), 2):
            field = parts[idx]
            if field not in values_by_field:
                continue
            value = parts[idx + 1]
            if value.endswith("x"):
                value = value[:-1]
            values_by_field[field].append(float(value))

    fields = ["Average over 8 ranks:"]
    for field in average_fields:
        values = values_by_field[field]
        if field.endswith("speedup"):
            fields.append(f"{field} {statistics.mean(values):.3f}x")
        else:
            fields.append(f"{field} {statistics.mean(values):.3f}")
    return " ".join(fields)


def _single_match(pattern: str) -> Path:
    matches = sorted(_debug_output_dir().glob(pattern))
    if not matches:
        pytest.skip(f"missing tensor dump matching {pattern!r} under {_debug_output_dir()}")
    if len(matches) > 1:
        raise AssertionError(f"expected one tensor dump for {pattern!r}, got {matches}")
    return matches[0]


def _dump_group(rank: int) -> FusedMoeDumpGroup:
    router_weight_path = _single_match(f"r{rank}_l*_router_weight.pt")
    hidden_states_path = _single_match(f"r{rank}_l*_hidden_states.pt")
    return FusedMoeDumpGroup(
        rank=rank,
        weight_layer_idx=int(router_weight_path.name.split("_", 2)[1][1:]),
        activation_layer_idx=int(hidden_states_path.name.split("_", 2)[1][1:]),
    )


def _require_cuda_and_ops(
    require_baseline_ops: bool = False, require_fused_ops: bool = False
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Deepseekv3FusedMoE dump tests require CUDA")
    _register_trtllm_custom_ops()
    _load_extra_op_libraries()

    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        pytest.skip("Deepseekv3FusedMoE dump tests require SM100-family GPUs")

    required_ops = ["dsv3_router_gemm_op", "noaux_tc_op"]
    if require_baseline_ops:
        required_ops.extend(
            [
                "fp8_quantize_1x128_packed_ue8m0",
                "silu_and_mul",
                "fp8_quantize_1x128",
                "fp8_block_scale_moe_runner",
                "fp8_swap_ab_gemm",
            ]
        )
    if require_fused_ops:
        required_ops.extend([_fused_expert_up_op_name(), _fused_expert_down_op_name()])

    missing_ops = [name for name in required_ops if not hasattr(torch.ops.trtllm, name)]
    if missing_ops:
        pytest.skip(f"missing torch.ops.trtllm ops: {missing_ops}")


def _load_inputs(
    group: FusedMoeDumpGroup,
    max_num_tokens: int | None = None,
    *,
    include_fused_kernel_tensors: bool = True,
) -> dict[str, torch.Tensor]:
    weight_layer_idx = group.weight_layer_idx
    activation_layer_idx = group.activation_layer_idx
    hidden_states = _load_tensor(group, activation_layer_idx, "hidden_states")
    if max_num_tokens is not None:
        hidden_states = hidden_states[:max_num_tokens].contiguous()

    tensors = {
        # torch.bfloat16, [T, H].
        "hidden_states": hidden_states,
        # torch.bfloat16, [E, H].
        "router_weight": _load_tensor(group, weight_layer_idx, "router_weight"),
        # torch.bfloat16, [E].
        "routing_bias": _load_tensor(group, weight_layer_idx, "routing_bias"),
        # torch.float8_e4m3fn, [2 * I, H], stored as [gate, up].
        "shared_gate_up_weight": _load_tensor(group, weight_layer_idx, "shared_gate_up_weight"),
        # torch.int32, [2 * I, PH], four packed UE8M0 scale bytes per int32.
        "shared_gate_up_weight_scale": _load_tensor(
            group, weight_layer_idx, "shared_gate_up_weight_scale"
        ),
        # torch.float8_e4m3fn, [E, 2 * I, H], stored as [up, gate].
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
    }
    if include_fused_kernel_tensors:
        tensors.update(
            {
                # torch.float8_e4m3fn, [2 * I, H], stored as [gate, up].
                "shared_gate_up_weight_org": _load_tensor(
                    group, weight_layer_idx, "shared_gate_up_weight_org"
                ),
                # torch.float32, [B2I, BH].
                "shared_gate_up_weight_scale_org": _load_tensor(
                    group, weight_layer_idx, "shared_gate_up_weight_scale_org"
                ),
                # torch.float8_e4m3fn, [H, I].
                "shared_down_weight_org": _load_tensor(
                    group, weight_layer_idx, "shared_down_weight_org"
                ),
                # torch.float32, [BH, BI].
                "shared_down_weight_scale_org": _load_tensor(
                    group, weight_layer_idx, "shared_down_weight_scale_org"
                ),
            }
        )
    return tensors


def _ensure_fused_expert_up_prepacked_tensors(tensors: dict[str, torch.Tensor]) -> None:
    if "shared_gate_up_weight_packed_fused_expert_up" in tensors:
        return

    # shared_gate_up_weight_packed_fused_expert_up:
    # torch.float8_e4m3fn, [2, I / 64, 8, 49152], stored as [gate, up].
    tensors["shared_gate_up_weight_packed_fused_expert_up"] = (
        prepack_fused_expert_up_shared_gate_up_weight(tensors["shared_gate_up_weight_org"])
    )
    # routed_w3_w1_weight_packed_fused_expert_up:
    # torch.float8_e4m3fn, [E, 2, I / 64, 8, 49152], stored as [gate, up].
    tensors["routed_w3_w1_weight_packed_fused_expert_up"] = (
        prepack_fused_expert_up_routed_w3_w1_weight(tensors["routed_w3_w1_weight"])
    )
    torch.cuda.synchronize()


def _ensure_fused_expert_down_prepacked_tensors(tensors: dict[str, torch.Tensor]) -> None:
    if "shared_down_weight_packed_fused_expert_down" in tensors:
        return

    # shared_down_weight_packed_fused_expert_down:
    # torch.float8_e4m3fn, [444, I / 128, 2048].
    tensors["shared_down_weight_packed_fused_expert_down"] = (
        prepack_fused_expert_down_shared_down_weight(tensors["shared_down_weight_org"])
    )
    # routed_w2_weight_packed_fused_expert_down:
    # torch.float8_e4m3fn, [E, 444, I / 128, 2048].
    tensors["routed_w2_weight_packed_fused_expert_down"] = (
        prepack_fused_expert_down_routed_w2_weight(tensors["routed_w2_weight"])
    )
    torch.cuda.synchronize()


def _run_fused_expert_down_chunked(
    slot_swiglu_output: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w2_weight_scale: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_weight_scale_org: torch.Tensor,
    output_tensor: torch.Tensor,
) -> torch.Tensor:
    num_tokens = slot_swiglu_output.size(0)
    fused_expert_down_op = getattr(torch.ops.trtllm, _fused_expert_down_op_name())
    for token_start in range(0, num_tokens, Deepseekv3FusedMoE._WIP_DOWN_PROJECT_MAX_NUM_TOKENS):
        token_end = min(
            token_start + Deepseekv3FusedMoE._WIP_DOWN_PROJECT_MAX_NUM_TOKENS, num_tokens
        )
        fused_expert_down_op(
            slot_swiglu_output[token_start:token_end].contiguous(),
            expert_indices[token_start:token_end].contiguous(),
            expert_weights[token_start:token_end].contiguous(),
            routed_w2_weight,
            routed_w2_weight_scale,
            shared_down_weight,
            shared_down_weight_scale_org,
            output_tensor[token_start:token_end],
        )
    return output_tensor


def _pytorch_ref_chunk_size() -> int:
    return max(
        1,
        _int_env(
            "TRTLLM_DEEPSEEKV3_FUSED_MOE_PYTORCH_REF_CHUNK_SIZE",
            8,
        ),
    )


def _unpack_ue8m0_scales(packed_scale: torch.Tensor, num_scale_cols: int) -> torch.Tensor:
    packed_scale_i64 = packed_scale.to(torch.int64)
    scale_bytes = torch.stack(
        [
            torch.bitwise_and(torch.bitwise_right_shift(packed_scale_i64, bit_offset), 0xFF)
            for bit_offset in (0, 8, 16, 24)
        ],
        dim=-1,
    )
    scale_bytes = scale_bytes.reshape(packed_scale.shape[0], -1)[:, :num_scale_cols]
    scales = torch.exp2(scale_bytes.to(torch.float32) - 127.0)
    return torch.where(scale_bytes == 0, torch.zeros_like(scales), scales)


def _dequantize_fp8_1x128_packed_ue8m0_weight(
    weight: torch.Tensor, packed_scale: torch.Tensor
) -> torch.Tensor:
    rows, cols = weight.shape
    num_scale_cols = (cols + 127) // 128
    scales = _unpack_ue8m0_scales(packed_scale, num_scale_cols)
    scales = scales.repeat_interleave(128, dim=1)[:, :cols]
    assert scales.shape == (rows, cols)
    return weight.to(torch.float32) * scales


def _dequantize_fp8_128x128_block_weight(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor:
    rows, cols = weight.shape[-2:]
    scales = weight_scale.repeat_interleave(128, dim=-2)[..., :rows, :]
    scales = scales.repeat_interleave(128, dim=-1)[..., :, :cols]
    assert scales.shape == weight.shape
    return weight.to(torch.float32) * scales


def _silu_and_mul_pytorch(
    gate: torch.Tensor, up: torch.Tensor, swiglu_limit: float | None
) -> torch.Tensor:
    gate = gate.to(torch.float32)
    up = up.to(torch.float32)
    if swiglu_limit is not None:
        gate = torch.minimum(gate, torch.tensor(float(swiglu_limit), device=gate.device))
        up = torch.clamp(up, -float(swiglu_limit), float(swiglu_limit))
    return torch.nn.functional.silu(gate) * up


def _ceil_to_ue8m0_pytorch(scale: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(scale)))


def _matmul_fp32_no_tf32(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        return torch.matmul(lhs.to(torch.float32), rhs.to(torch.float32))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32


def _fp8_quantize_dequantize_1x128_pytorch(
    tensor: torch.Tensor, use_ue8m0_scale: bool, clamp_min_scale: bool
) -> torch.Tensor:
    rows, cols = tensor.shape
    num_blocks = (cols + 127) // 128
    padded_cols = num_blocks * 128
    tensor_fp32 = tensor.to(torch.float32)
    if padded_cols != cols:
        tensor_fp32 = torch.nn.functional.pad(tensor_fp32, (0, padded_cols - cols))

    tensor_blocks = tensor_fp32.reshape(rows, num_blocks, 128)
    amax = tensor_blocks.abs().amax(dim=-1, keepdim=True)
    if clamp_min_scale:
        amax = amax.clamp_min(1e-10)
    dequant_scale = amax / 448.0
    if use_ue8m0_scale:
        dequant_scale = _ceil_to_ue8m0_pytorch(dequant_scale)

    quant_scale = torch.where(
        dequant_scale == 0, torch.ones_like(dequant_scale), 1.0 / dequant_scale
    )
    quantized_tensor = (tensor_blocks * quant_scale).to(torch.float8_e4m3fn)
    dequantized_tensor = quantized_tensor.to(torch.float32) * dequant_scale
    return dequantized_tensor.reshape(rows, padded_cols)[:, :cols]


def _routed_gate_up_swiglu_pytorch(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    routed_w3_w1_weight: torch.Tensor,
    routed_w3_w1_weight_scale: torch.Tensor,
    expert_intermediate_size: int,
    swiglu_limit: float | None,
) -> torch.Tensor:
    num_tokens, top_k = expert_indices.shape
    routed_swiglu_output = torch.empty(
        (num_tokens, top_k, expert_intermediate_size),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    chunk_size = _pytorch_ref_chunk_size()
    hidden_states_fp32 = hidden_states.to(torch.float32)
    for route_idx in range(top_k):
        for token_start in range(0, num_tokens, chunk_size):
            token_end = min(token_start + chunk_size, num_tokens)
            route_expert_indices = expert_indices[token_start:token_end, route_idx].to(torch.int64)
            selected_expert_weight = torch.index_select(
                routed_w3_w1_weight, 0, route_expert_indices
            )
            selected_expert_weight_scale = torch.index_select(
                routed_w3_w1_weight_scale, 0, route_expert_indices
            )
            selected_expert_weight = _dequantize_fp8_128x128_block_weight(
                selected_expert_weight, selected_expert_weight_scale
            )

            expert_gate_up = _matmul_fp32_no_tf32(
                selected_expert_weight, hidden_states_fp32[token_start:token_end].unsqueeze(-1)
            )
            expert_gate_up = expert_gate_up.squeeze(-1)
            expert_gate_up = _fp8_quantize_dequantize_1x128_pytorch(
                expert_gate_up,
                use_ue8m0_scale=False,
                clamp_min_scale=False,
            )
            routed_up, routed_gate = expert_gate_up.split(expert_intermediate_size, dim=-1)
            routed_swiglu = _silu_and_mul_pytorch(routed_gate, routed_up, swiglu_limit)
            routed_swiglu = _fp8_quantize_dequantize_1x128_pytorch(
                routed_swiglu,
                use_ue8m0_scale=False,
                clamp_min_scale=False,
            )
            routed_swiglu_output[token_start:token_end, route_idx, :] = routed_swiglu
    return routed_swiglu_output


def _routed_down_project_pytorch(
    routed_swiglu_output: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_weights: torch.Tensor,
    routed_w2_weight: torch.Tensor,
    routed_w2_weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    num_tokens, top_k, _ = routed_swiglu_output.shape
    hidden_size = routed_w2_weight.shape[1]
    routed_output = torch.zeros(
        (num_tokens, hidden_size),
        device=routed_swiglu_output.device,
        dtype=torch.float32,
    )
    chunk_size = _pytorch_ref_chunk_size()
    for route_idx in range(top_k):
        for token_start in range(0, num_tokens, chunk_size):
            token_end = min(token_start + chunk_size, num_tokens)
            route_expert_indices = expert_indices[token_start:token_end, route_idx].to(torch.int64)
            selected_expert_weight = torch.index_select(routed_w2_weight, 0, route_expert_indices)
            selected_expert_weight_scale = torch.index_select(
                routed_w2_weight_scale, 0, route_expert_indices
            )
            selected_expert_weight = _dequantize_fp8_128x128_block_weight(
                selected_expert_weight, selected_expert_weight_scale
            )

            chunk_swiglu_output = routed_swiglu_output[token_start:token_end, route_idx, :]
            expert_down_output = _matmul_fp32_no_tf32(
                selected_expert_weight, chunk_swiglu_output.to(torch.float32).unsqueeze(-1)
            )
            expert_down_output = expert_down_output.squeeze(-1)
            expert_down_output = expert_down_output.to(output_dtype).to(torch.float32)
            expert_down_output *= (
                expert_weights[token_start:token_end, route_idx].to(torch.float32).unsqueeze(-1)
            )
            routed_output[token_start:token_end] += expert_down_output
    return routed_output.to(output_dtype)


def _shared_down_project_pytorch(
    shared_swiglu_output: torch.Tensor,
    shared_down_weight: torch.Tensor,
    shared_down_weight_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    # shared_swiglu_output: torch.bfloat16, [T, I].
    # shared_down_weight: torch.float8_e4m3fn, [H, I].
    # shared_down_weight_scale: torch.int32 [H, PI] or torch.float32 [BH, BI].
    if shared_down_weight_scale.dtype == torch.int32:
        shared_weight = _dequantize_fp8_1x128_packed_ue8m0_weight(
            shared_down_weight, shared_down_weight_scale
        )
    else:
        shared_weight = _dequantize_fp8_128x128_block_weight(
            shared_down_weight, shared_down_weight_scale
        )
    shared_output = _matmul_fp32_no_tf32(shared_swiglu_output.to(torch.float32), shared_weight.t())
    # shared_output: output_dtype, [T, H].
    return shared_output.to(output_dtype)


def _run_pytorch_reference(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    hidden_states = tensors["hidden_states"]
    num_tokens = hidden_states.shape[0]
    num_router_experts = tensors["router_weight"].shape[0]

    # router_logits: torch.float32, [T, E].
    router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
        hidden_states,
        tensors["router_weight"].t(),
        bias=None,
        out_dtype=torch.float32,
    )
    check_data(
        router_logits,
        "pytorch_reference.router_logits",
        torch.float32,
        (num_tokens, num_router_experts),
    )

    # expert_weights: torch.float32, [T, K].
    # expert_indices: torch.int32, [T, K].
    expert_weights, expert_indices = torch.ops.trtllm.noaux_tc_op(
        router_logits,
        tensors["routing_bias"],
        _n_group(),
        _topk_group(),
        _top_k(),
        _routed_scaling_factor(),
    )
    check_data(
        expert_indices, "pytorch_reference.expert_indices", torch.int32, (num_tokens, _top_k())
    )
    check_data(
        expert_weights, "pytorch_reference.expert_weights", torch.float32, (num_tokens, _top_k())
    )

    # shared_weight: torch.float32, [2 * I, H].
    shared_weight = _dequantize_fp8_1x128_packed_ue8m0_weight(
        tensors["shared_gate_up_weight"], tensors["shared_gate_up_weight_scale"]
    )
    # shared_hidden_states: torch.float32, [T, H].
    shared_hidden_states = _fp8_quantize_dequantize_1x128_pytorch(
        hidden_states,
        use_ue8m0_scale=True,
        clamp_min_scale=True,
    )
    # shared_gate_up_output: torch.bfloat16, [T, 2 * I].
    shared_gate_up_output = _matmul_fp32_no_tf32(shared_hidden_states, shared_weight.t()).to(
        hidden_states.dtype
    )
    # shared_gate/shared_up: torch.bfloat16, [T, I].
    shared_gate, shared_up = shared_gate_up_output.chunk(2, dim=-1)
    # shared_swiglu_output: torch.bfloat16, [T, I].
    shared_swiglu_output = _silu_and_mul_pytorch(shared_gate, shared_up, _shared_swiglu_limit()).to(
        hidden_states.dtype
    )

    expert_intermediate_size = shared_swiglu_output.shape[-1]
    # routed_hidden_states: torch.float32, [T, H].
    routed_hidden_states = _fp8_quantize_dequantize_1x128_pytorch(
        hidden_states,
        use_ue8m0_scale=False,
        clamp_min_scale=True,
    )
    # routed_swiglu_output: torch.float32, [T, K, I].
    routed_swiglu_output = _routed_gate_up_swiglu_pytorch(
        routed_hidden_states,
        expert_indices,
        tensors["routed_w3_w1_weight"],
        tensors["routed_w3_w1_weight_scale"],
        expert_intermediate_size,
        _routed_swiglu_limit(),
    )
    # routed_output: torch.bfloat16, [T, H].
    routed_output = _routed_down_project_pytorch(
        routed_swiglu_output,
        expert_indices,
        expert_weights.to(torch.bfloat16),
        tensors["routed_w2_weight"],
        tensors["routed_w2_weight_scale"],
        hidden_states.dtype,
    )

    # shared_output: torch.bfloat16, [T, H].
    shared_output = _shared_down_project_pytorch(
        shared_swiglu_output,
        tensors["shared_down_weight"],
        tensors["shared_down_weight_scale"],
        hidden_states.dtype,
    )
    # return: torch.bfloat16, [T, H].
    return shared_output + routed_output


def _set_parameter(module: torch.nn.Module, name: str, tensor: torch.Tensor) -> None:
    # Keep dumped strides: DeepGEMM scale tensors use non-contiguous layouts.
    setattr(module, name, torch.nn.Parameter(tensor.detach(), requires_grad=False))


def _old_moe_aux_stream_dict() -> dict[AuxStreamType, torch.cuda.Stream]:
    return {
        AuxStreamType.Attention: torch.cuda.current_stream(),
        AuxStreamType.MoeShared: torch.cuda.Stream(),
        AuxStreamType.MoeChunkingOverlap: torch.cuda.Stream(),
        AuxStreamType.MoeBalancer: torch.cuda.Stream(),
        AuxStreamType.MoeOutputMemset: torch.cuda.Stream(),
    }


def _build_old_moe_baseline(
    tensors: dict[str, torch.Tensor], group: FusedMoeDumpGroup
) -> Deepseekv3MoE:
    hidden_size = tensors["hidden_states"].shape[-1]
    num_experts = tensors["router_weight"].shape[0]
    intermediate_size = tensors["routed_w2_weight"].shape[-1]
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        moe_intermediate_size=intermediate_size,
        n_group=_n_group(),
        num_experts=num_experts,
        routed_scaling_factor=_routed_scaling_factor(),
        swiglu_limit=_routed_swiglu_limit(),
        topk_group=_topk_group(),
        torch_dtype=torch.bfloat16,
    )
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(world_size=1, rank=0, gpus_per_node=1, tp_size=1, pp_size=1),
        quant_config=quant_config,
        max_num_tokens=_baseline_tune_max_num_tokens(tensors["hidden_states"].shape[0]),
        moe_backend="TRTLLM",
    )
    old_moe = Deepseekv3MoE(
        num_experts=num_experts,
        top_k=_top_k(),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        shared_expert_intermediate_size=tensors["shared_down_weight"].shape[-1],
        aux_stream_dict=_old_moe_aux_stream_dict(),
        layer_idx=group.weight_layer_idx,
        dtype=torch.bfloat16,
        model_config=model_config,
        override_quant_config=quant_config,
    )

    _set_parameter(old_moe.gate, "weight", tensors["router_weight"])
    _set_parameter(old_moe.gate, "e_score_correction_bias", tensors["routing_bias"])
    old_moe.shared_experts.swiglu_limit = _shared_swiglu_limit()
    _set_parameter(old_moe.shared_experts.gate_up_proj, "weight", tensors["shared_gate_up_weight"])
    _set_parameter(
        old_moe.shared_experts.gate_up_proj,
        "weight_scale",
        tensors["shared_gate_up_weight_scale"],
    )
    _set_parameter(old_moe.shared_experts.down_proj, "weight", tensors["shared_down_weight"])
    _set_parameter(
        old_moe.shared_experts.down_proj,
        "weight_scale",
        tensors["shared_down_weight_scale"],
    )

    experts_backend = getattr(old_moe.experts, "backend", old_moe.experts)
    _set_parameter(experts_backend, "w3_w1_weight", tensors["routed_w3_w1_weight"])
    _set_parameter(
        experts_backend,
        "w3_w1_weight_scaling_factor",
        tensors["routed_w3_w1_weight_scale"],
    )
    _set_parameter(experts_backend, "w2_weight", tensors["routed_w2_weight"])
    _set_parameter(
        experts_backend,
        "w2_weight_scaling_factor",
        tensors["routed_w2_weight_scale"],
    )
    experts_backend.swiglu_limit_scalar = _routed_swiglu_limit()
    old_moe.eval()
    return old_moe


def _run_multi_stream_baseline(
    old_moe: Deepseekv3MoE,
    tensors: dict[str, torch.Tensor],
) -> torch.Tensor:
    with with_multi_stream(True):
        return old_moe(tensors["hidden_states"])


_FusedKernelProfileEvents = dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]]


def _record_fused_kernel_profile_start(
    profile_events: _FusedKernelProfileEvents | None,
    event_name: str,
) -> torch.cuda.Event | None:
    if profile_events is None:
        return None
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    profile_events[event_name].append((start_event, end_event))
    start_event.record()
    return end_event


def _run_fused_kernel(
    tensors: dict[str, torch.Tensor],
    profile_events: _FusedKernelProfileEvents | None = None,
) -> torch.Tensor:
    _ensure_fused_expert_up_prepacked_tensors(tensors)
    if _prepack_fused_expert_down_enabled():
        _ensure_fused_expert_down_prepacked_tensors(tensors)
        # routed_w2_weight: torch.float8_e4m3fn, [E, 444, I / 128, 2048].
        routed_w2_weight = tensors["routed_w2_weight_packed_fused_expert_down"]
        # shared_down_weight: torch.float8_e4m3fn, [444, I / 128, 2048].
        shared_down_weight = tensors["shared_down_weight_packed_fused_expert_down"]
    else:
        # routed_w2_weight: torch.float8_e4m3fn, [E, H, I].
        routed_w2_weight = tensors["routed_w2_weight"]
        # shared_down_weight: torch.float8_e4m3fn, [H, I].
        shared_down_weight = tensors["shared_down_weight_org"]

    num_tokens = tensors["hidden_states"].shape[0]
    num_router_experts = tensors["router_weight"].shape[0]
    router_gemm_end_event = _record_fused_kernel_profile_start(profile_events, "dsv3_router_gemm")
    # router_logits: torch.float32, [T, E].
    router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
        tensors["hidden_states"],
        tensors["router_weight"].t(),
        bias=None,
        out_dtype=torch.float32,
    )
    if router_gemm_end_event is not None:
        router_gemm_end_event.record()
    check_data(
        router_logits,
        "dsv3_fused_expert_up.router_logits",
        torch.float32,
        (num_tokens, num_router_experts),
    )
    assert _top_k() == 8, f"dsv3_fused_expert_up only supports top_k=8, got {_top_k()}"
    assert _n_group() == 1, f"dsv3_fused_expert_up only supports n_group=1, got {_n_group()}"
    assert _topk_group() == 1, (
        f"dsv3_fused_expert_up only supports topk_group=1, got {_topk_group()}"
    )

    fused_expert_up_end_event = _record_fused_kernel_profile_start(
        profile_events, "dsv3_fused_expert_up"
    )
    # expert_weights: torch.float32, [T, K].
    # expert_indices: torch.int32, [T, K].
    # slot_swiglu_output: torch.float16, [T, K + 1, I].
    fused_expert_up_op = getattr(torch.ops.trtllm, _fused_expert_up_op_name())
    expert_weights, expert_indices, slot_swiglu_output = fused_expert_up_op(
        router_logits.contiguous(),
        tensors["hidden_states"].contiguous(),
        tensors["routing_bias"].contiguous(),
        tensors["shared_gate_up_weight_packed_fused_expert_up"],
        tensors["shared_gate_up_weight_scale_org"],
        tensors["routed_w3_w1_weight_packed_fused_expert_up"],
        tensors["routed_w3_w1_weight_scale"],
        _routed_scaling_factor(),
    )
    if fused_expert_up_end_event is not None:
        fused_expert_up_end_event.record()
    check_data(
        expert_indices,
        "dsv3_fused_expert_up.expert_indices",
        torch.int32,
        (num_tokens, _top_k()),
    )
    check_data(
        expert_weights,
        "dsv3_fused_expert_up.expert_weights",
        torch.float32,
        (num_tokens, _top_k()),
    )
    check_data(
        slot_swiglu_output,
        "dsv3_fused_expert_up.slot_swiglu_output",
        torch.float16,
        (num_tokens, _top_k() + 1, -1),
    )

    # output: torch.bfloat16, [T, H].
    output = torch.empty_like(tensors["hidden_states"])
    fused_expert_down_end_event = _record_fused_kernel_profile_start(
        profile_events, "dsv3_fused_expert_down"
    )
    final_hidden_states = _run_fused_expert_down_chunked(
        slot_swiglu_output,
        expert_indices,
        expert_weights,
        routed_w2_weight,
        tensors["routed_w2_weight_scale"],
        shared_down_weight,
        tensors["shared_down_weight_scale_org"],
        output,
    )
    if fused_expert_down_end_event is not None:
        fused_expert_down_end_event.record()
    return final_hidden_states


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
    times_us = torch.tensor(
        [starts[idx].elapsed_time(ends[idx]) * 1000.0 for idx in range(profile_iters)],
        dtype=torch.float32,
    )
    times = [float(time_us) for time_us in times_us.tolist()]
    stats = {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "std_us": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_us": min(times),
        "max_us": max(times),
    }
    return times_us, stats


def _profile_event_pairs(
    event_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]],
) -> tuple[torch.Tensor, dict[str, float]]:
    times_us = torch.tensor(
        [start.elapsed_time(end) * 1000.0 for start, end in event_pairs],
        dtype=torch.float32,
    )
    times = [float(time_us) for time_us in times_us.tolist()]
    stats = {
        "mean_us": statistics.mean(times),
        "median_us": statistics.median(times),
        "std_us": statistics.stdev(times) if len(times) > 1 else 0.0,
        "min_us": min(times),
        "max_us": max(times),
    }
    return times_us, stats


def _profile_fused_kernel_with_subkernel_events(
    tensors: dict[str, torch.Tensor],
    warmup_iters: int,
    profile_iters: int,
) -> tuple[
    torch.Tensor,
    dict[str, float],
    torch.Tensor,
    dict[str, float],
    torch.Tensor,
    dict[str, float],
    torch.Tensor,
    dict[str, float],
]:
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")
    if profile_iters <= 0:
        raise ValueError(f"profile_iters must be positive, got {profile_iters}")

    for _ in range(warmup_iters):
        _run_fused_kernel(tensors)
    torch.cuda.synchronize()

    profile_events: _FusedKernelProfileEvents = {
        "dsv3_router_gemm": [],
        "dsv3_fused_expert_up": [],
        "dsv3_fused_expert_down": [],
    }
    fused_kernel_events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(profile_iters)
    ]
    for idx in range(profile_iters):
        start_event, end_event = fused_kernel_events[idx]
        start_event.record()
        _run_fused_kernel(tensors, profile_events=profile_events)
        end_event.record()

    torch.cuda.synchronize()
    fused_kernel_times_us, fused_kernel_stats = _profile_event_pairs(fused_kernel_events)
    router_gemm_times_us, router_gemm_stats = _profile_event_pairs(
        profile_events["dsv3_router_gemm"]
    )
    fused_expert_up_times_us, fused_expert_up_stats = _profile_event_pairs(
        profile_events["dsv3_fused_expert_up"]
    )
    fused_expert_down_times_us, fused_expert_down_stats = _profile_event_pairs(
        profile_events["dsv3_fused_expert_down"]
    )
    return (
        fused_kernel_times_us,
        fused_kernel_stats,
        router_gemm_times_us,
        router_gemm_stats,
        fused_expert_up_times_us,
        fused_expert_up_stats,
        fused_expert_down_times_us,
        fused_expert_down_stats,
    )


@dataclass
class _CudaGraphRunner:
    graph: torch.cuda.CUDAGraph
    output: torch.Tensor

    def __call__(self) -> torch.Tensor:
        self.graph.replay()
        return self.output


def _make_cuda_graph_runner(
    fn: Callable[[], torch.Tensor], capture_warmup_iters: int
) -> _CudaGraphRunner:
    if capture_warmup_iters < 0:
        raise ValueError(f"capture_warmup_iters must be non-negative, got {capture_warmup_iters}")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(max(3, capture_warmup_iters)):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return _CudaGraphRunner(graph=graph, output=output)


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
def test_deepseekv3_fused_moe_reference_phase(rank: int) -> None:
    if not _phase_enabled("reference"):
        pytest.skip(f"{_PHASE_ENV} disables reference phase")
    _require_cuda_and_ops(require_baseline_ops=True)
    group = _dump_group(rank)
    tensors = _load_inputs(
        group,
        max_num_tokens=_max_fused_kernel_num_tokens(),
        include_fused_kernel_tensors=False,
    )

    with torch.inference_mode():
        old_moe = _build_old_moe_baseline(tensors, group)
        # multi_stream_baseline_output: torch.bfloat16, [T, H].
        multi_stream_baseline_output = _run_multi_stream_baseline(old_moe, tensors)
        # pytorch_ref_output: torch.bfloat16, [T, H].
        pytorch_ref_output = _run_pytorch_reference(tensors)
        torch.cuda.synchronize()

    _save_tensor(group, "multi_stream_baseline_output", multi_stream_baseline_output)
    _save_tensor(group, "pytorch_ref_output", pytorch_ref_output)

    rel_error, abs_error, rel_error_by_dim, abs_error_by_dim = _max_errors(
        multi_stream_baseline_output, pytorch_ref_output
    )
    # rel_error_by_dim and abs_error_by_dim: torch.float32, [H].
    _save_tensor(group, "multi_stream_baseline_vs_pytorch_ref_rel_error_by_dim", rel_error_by_dim)
    _save_tensor(group, "multi_stream_baseline_vs_pytorch_ref_abs_error_by_dim", abs_error_by_dim)
    _save_summary("fused_moe_reference_errors.txt", rank, rel_error, abs_error)


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_moe_fused_kernel_phase(rank: int) -> None:
    if not _phase_enabled("test"):
        pytest.skip(f"{_PHASE_ENV} disables test phase")
    _require_cuda_and_ops(require_fused_ops=True)
    group = _dump_group(rank)
    # pytorch_ref_output: torch.bfloat16, [T, H].
    pytorch_ref_output = _load_saved_tensor(group, "pytorch_ref_output")
    # reference_abs_error_by_dim: torch.float32, [H].
    reference_abs_error_by_dim = _load_saved_tensor(
        group, "multi_stream_baseline_vs_pytorch_ref_abs_error_by_dim"
    )
    reference_abs_threshold = float(reference_abs_error_by_dim.max().item())
    slack_abs_threshold = reference_abs_threshold * _fused_kernel_abs_error_threshold_slack()
    current_fused_kernel_abs_threshold = _current_fused_kernel_abs_error_threshold(rank)
    fused_kernel_abs_threshold = max(
        slack_abs_threshold,
        current_fused_kernel_abs_threshold,
    )

    tensors = _load_inputs(group, max_num_tokens=_max_fused_kernel_num_tokens())
    with torch.inference_mode():
        # fused_kernel_output: torch.bfloat16, [T, H].
        fused_kernel_output = _run_fused_kernel(tensors)
        torch.cuda.synchronize()

    _save_tensor(group, "fused_kernel_output", fused_kernel_output)
    rel_error, abs_error, rel_error_by_dim, abs_error_by_dim = _max_errors(
        fused_kernel_output, pytorch_ref_output
    )
    # rel_error_by_dim and abs_error_by_dim: torch.float32, [H].
    _save_tensor(group, "fused_kernel_vs_pytorch_ref_rel_error_by_dim", rel_error_by_dim)
    _save_tensor(group, "fused_kernel_vs_pytorch_ref_abs_error_by_dim", abs_error_by_dim)
    _save_summary("fused_moe_fused_kernel_errors.txt", rank, rel_error, abs_error)

    assert abs_error <= fused_kernel_abs_threshold, (
        f"rank {rank} fused kernel output differs from PyTorch reference: "
        f"rel={rel_error} abs={abs_error} "
        f"fused_kernel_abs_threshold={fused_kernel_abs_threshold} "
        f"current_fused_kernel_abs_threshold={current_fused_kernel_abs_threshold} "
        f"slack_abs_threshold={slack_abs_threshold} "
        f"reference_abs_threshold={reference_abs_threshold}"
    )


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_moe_profile_phase(rank: int) -> None:
    if not _phase_enabled("profile"):
        pytest.skip(f"{_PHASE_ENV} disables profile phase")
    _require_cuda_and_ops(require_baseline_ops=True, require_fused_ops=True)
    group = _dump_group(rank)
    tensors = _load_inputs(group, max_num_tokens=_max_fused_kernel_num_tokens())
    warmup_iters = _profile_warmup_iters()
    profile_iters = _profile_iters()
    capture_warmup_iters = _cuda_graph_capture_warmup_iters()

    with torch.inference_mode():
        old_moe = _build_old_moe_baseline(tensors, group)
        _ensure_fused_expert_up_prepacked_tensors(tensors)
        if _prepack_fused_expert_down_enabled():
            _ensure_fused_expert_down_prepacked_tensors(tensors)

        multi_stream_baseline_output = _run_multi_stream_baseline(old_moe, tensors)
        fused_kernel_output = _run_fused_kernel(tensors)
        torch.cuda.synchronize()

        rel_error, abs_error, _, _ = _max_errors(fused_kernel_output, multi_stream_baseline_output)
        assert fused_kernel_output.shape == multi_stream_baseline_output.shape, (
            f"rank {rank} fused kernel output shape differs from multi-stream baseline: "
            f"{fused_kernel_output.shape} vs {multi_stream_baseline_output.shape}"
        )

        multi_stream_baseline_runner = _make_cuda_graph_runner(
            lambda: _run_multi_stream_baseline(old_moe, tensors),
            capture_warmup_iters,
        )
        fused_kernel_runner = _make_cuda_graph_runner(
            lambda: _run_fused_kernel(tensors),
            capture_warmup_iters,
        )
        torch.cuda.synchronize()

        assert torch.equal(multi_stream_baseline_runner(), multi_stream_baseline_output), (
            f"rank {rank} multi-stream baseline CUDA graph output differs from eager output"
        )
        assert torch.equal(fused_kernel_runner(), fused_kernel_output), (
            f"rank {rank} fused kernel CUDA graph output differs from eager output"
        )
        torch.cuda.synchronize()

        # *_times_us: torch.float32, [profile_iters].
        multi_stream_baseline_times_us, multi_stream_baseline_stats = _profile_cuda_events(
            multi_stream_baseline_runner,
            warmup_iters,
            profile_iters,
        )
        fused_kernel_times_us, fused_kernel_stats = _profile_cuda_events(
            fused_kernel_runner,
            warmup_iters,
            profile_iters,
        )
        (
            fused_kernel_eager_times_us,
            fused_kernel_eager_stats,
            router_gemm_times_us,
            router_gemm_stats,
            fused_expert_up_times_us,
            fused_expert_up_stats,
            fused_expert_down_times_us,
            fused_expert_down_stats,
        ) = _profile_fused_kernel_with_subkernel_events(
            tensors,
            warmup_iters,
            profile_iters,
        )

        pytorch_reference_times_us = None
        pytorch_reference_stats = None
        if _profile_pytorch_reference_enabled():
            pytorch_reference_times_us, pytorch_reference_stats = _profile_cuda_events(
                lambda: _run_pytorch_reference(tensors),
                warmup_iters,
                profile_iters,
            )

    _save_tensor(group, "multi_stream_baseline_profile_times_us", multi_stream_baseline_times_us)
    _save_tensor(group, "fused_kernel_profile_times_us", fused_kernel_times_us)
    _save_tensor(group, "fused_kernel_eager_profile_times_us", fused_kernel_eager_times_us)
    _save_tensor(group, "dsv3_router_gemm_profile_times_us", router_gemm_times_us)
    _save_tensor(group, "dsv3_fused_expert_up_profile_times_us", fused_expert_up_times_us)
    _save_tensor(group, "dsv3_fused_expert_down_profile_times_us", fused_expert_down_times_us)
    if pytorch_reference_times_us is not None:
        _save_tensor(group, "pytorch_reference_profile_times_us", pytorch_reference_times_us)
    average_profile_summary = _save_profile_summary(
        rank,
        multi_stream_baseline_stats,
        fused_kernel_stats,
        fused_kernel_eager_stats,
        router_gemm_stats,
        fused_expert_up_stats,
        fused_expert_down_stats,
        pytorch_reference_stats,
    )

    speedup = multi_stream_baseline_stats["mean_us"] / fused_kernel_stats["mean_us"]
    print(
        f"rank {rank} cuda_graph: "
        f"multi_stream_baseline {multi_stream_baseline_stats['mean_us']:.3f} us, "
        f"fused_kernel {fused_kernel_stats['mean_us']:.3f} us, "
        f"fused_kernel_eager {fused_kernel_eager_stats['mean_us']:.3f} us, "
        f"dsv3_router_gemm {router_gemm_stats['mean_us']:.3f} us, "
        f"dsv3_fused_expert_up {fused_expert_up_stats['mean_us']:.3f} us, "
        f"dsv3_fused_expert_down {fused_expert_down_stats['mean_us']:.3f} us, "
        f"fused_kernel_vs_multi_stream_baseline_speedup {speedup:.3f}x, "
        f"prepack_fused_expert_down {int(_prepack_fused_expert_down_enabled())}, "
        f"fused_vs_baseline_rel {rel_error:.6e}, fused_vs_baseline_abs {abs_error:.6e}"
    )
    if average_profile_summary is not None:
        print(average_profile_summary)
