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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch

_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_DEBUG_OUTPUT_DIR"
_RUNTIME_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DEBUG_OUTPUT_DIR"
_DEFAULT_DEBUG_OUTPUT_DIR = "~/dev/mla-debug-output"
_EXTRA_OP_LIBRARY_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_EXTRA_OP_LIBRARY"
_MAX_NUM_TOKENS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_MAX_NUM_TOKENS"
_PROFILE_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_PROFILE"
_PROFILE_ITERS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_PROFILE_ITERS"

_NUM_RANKS = 8
_DEFAULT_MAX_NUM_TOKENS = 128
_DEFAULT_PROFILE_ITERS = 100
_EXTRA_OP_LIBRARIES_LOADED = False

_REQUIRED_PROJECTION_DUMPS = (
    "kv_a_proj_with_mqa_weight",
    "kv_a_proj_with_mqa_weight_scale",
    "q_a_layernorm_weight",
    "q_a_layernorm_variance_epsilon",
    "q_b_proj_weight",
    "q_b_proj_weight_scale",
    "kv_a_layernorm_weight",
    "kv_a_layernorm_variance_epsilon",
    "k_b_proj_trans",
    "num_heads_tp",
    "num_heads_tp_cp",
    "o_proj_weight",
    "o_proj_weight_scale",
    "q_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "softmax_scale",
    "v_b_proj",
    "v_head_dim",
)
_REQUIRED_Q_B_FUSION_DUMPS = (
    "q_b_proj_weight",
    "q_b_proj_weight_scale",
    "k_b_proj_trans",
    "num_heads_tp",
    "q_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "kv_lora_rank",
)
_REQUIRED_LIVE_Q_B_FUSION_DUMPS = _REQUIRED_Q_B_FUSION_DUMPS + (
    "q_b_proj_input",
    "q_b_proj_output",
)


@dataclass(frozen=True)
class FusedMlaDumpGroup:
    rank: int
    layer_idx: int


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _has_dump_files(path: Path) -> bool:
    return path.exists() and any(path.glob("r*_l*_*.pt"))


def _debug_output_dir() -> Path:
    """
    Resolve the debug output directory used by the fused MLA dump test.

    Explicit test/runtime environment variables take priority. Without them,
    the helper picks a fallback directory that actually contains rank/layer
    dumps, preferring the home debug path and then repo-relative paths. It has
    no side effects.

    Args
    - None.

    Returns
    - debug_output_dir: Path, directory where r<rank>_l<layer>_<name>.pt files
        are expected.
    """
    for env_name in (
        _DEBUG_OUTPUT_DIR_ENV,
        _RUNTIME_DEBUG_OUTPUT_DIR_ENV,
    ):
        value = os.environ.get(env_name)
        if value:
            return Path(value).expanduser()

    home_default = Path(_DEFAULT_DEBUG_OUTPUT_DIR).expanduser()
    fallback_dirs = [home_default]

    src_dir = os.environ.get("TRTLLM_SRC_DIR")
    if src_dir:
        fallback_dirs.append(Path(src_dir).expanduser().parent / "mla-debug-output")

    fallback_dirs.append(Path(__file__).resolve().parents[5].parent / "mla-debug-output")

    for fallback_dir in fallback_dirs:
        if _has_dump_files(fallback_dir):
            return fallback_dir

    return home_default


def _max_num_tokens() -> int | None:
    """
    Resolve the optional token slice length for dumped hidden states.

    The environment value may be an integer or one of the unlimited sentinels.
    This controls how many dumped tokens the smoke test runs through the local
    MLA flow. It has no side effects.

    Args
    - None.

    Returns
    - max_num_tokens: int | None, maximum M to test, or None to use all dumped
        tokens.
    """
    value = _env(_MAX_NUM_TOKENS_ENV, str(_DEFAULT_MAX_NUM_TOKENS)).strip()
    if value.lower() in ("", "none", "all"):
        return None
    return int(value)


def _profile_enabled() -> bool:
    value = _env(_PROFILE_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def _profile_iterations() -> int:
    return int(_env(_PROFILE_ITERS_ENV, str(_DEFAULT_PROFILE_ITERS)))


def _register_trtllm_custom_ops() -> None:
    """
    Import TensorRT-LLM custom op registration modules.

    Importing these modules registers Python-visible torch custom ops used by
    the smoke test. The side effect is registration in torch.ops.

    Args
    - None.

    Returns
    - None: this function is used for its torch custom op registration side
        effects.
    """
    importlib.import_module("tensorrt_llm._torch.custom_ops.torch_custom_ops")
    importlib.import_module("tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops")


def _load_extra_op_libraries() -> None:
    """
    Load optional extra torch op shared libraries once.

    The test can be pointed at development .so files with
    TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_EXTRA_OP_LIBRARY. Libraries are colon
    separated and are loaded through torch.ops.load_library. The side effect is
    dynamic library loading and torch custom op registration.

    Args
    - None.

    Returns
    - None: this function is used for dynamic library loading side effects.
    """
    global _EXTRA_OP_LIBRARIES_LOADED

    if _EXTRA_OP_LIBRARIES_LOADED:
        return

    raw_paths = _env(_EXTRA_OP_LIBRARY_ENV, "")
    for raw_path in raw_paths.split(":"):
        raw_path = raw_path.strip()
        if raw_path:
            torch.ops.load_library(str(Path(raw_path).expanduser()))
    _EXTRA_OP_LIBRARIES_LOADED = True


def _require_cuda_and_ops() -> None:
    """
    Skip the test unless CUDA, SM100, and required ops are available.

    This helper performs the runtime capability checks for the dump smoke test.
    It registers custom ops, optionally loads extra op libraries, and calls
    pytest.skip when a requirement is missing.

    Args
    - None.

    Returns
    - None: successful return means the current process can run the CUDA test.
    """
    if not torch.cuda.is_available():
        pytest.skip("Deepseekv3 fused MLA dump tests require CUDA")

    _register_trtllm_custom_ops()
    _load_extra_op_libraries()

    from tensorrt_llm._utils import is_sm_100f

    if not is_sm_100f():
        pytest.skip("Deepseekv3 fused MLA dump tests require SM100-family GPUs")

    required_ops = (
        "bmm_out",
        "dsv3_fused_mla_generation",
        "fp8_swap_ab_gemm",
        "mla_rope_inplace",
    )
    missing_ops = [name for name in required_ops if not hasattr(torch.ops.trtllm, name)]
    if missing_ops:
        pytest.skip(f"missing torch.ops.trtllm ops: {missing_ops}")


def _layer_tensor_names(rank: int) -> dict[int, set[str]]:
    """
    Collect dumped tensor names grouped by layer for one rank.

    The dump directory can contain multiple layers per rank. This helper scans
    only file names and extracts the layer id and tensor-name suffix so later
    code can choose a layer with all required tensors. It has no side effects.

    Args
    - rank: int, tensor-parallel rank id in dump file names.

    Returns
    - layers: dict[int, set[str]], mapping from layer id to tensor names present
        for that rank/layer.
    """
    layers: dict[int, set[str]] = {}
    for path in _debug_output_dir().glob(f"r{rank}_l*_*.pt"):
        parts = path.name.split("_", 2)
        if len(parts) != 3:
            continue
        layer_part = parts[1]
        if not layer_part.startswith("l"):
            continue
        tensor_name = parts[2].removesuffix(".pt")
        layers.setdefault(int(layer_part[1:]), set()).add(tensor_name)
    return layers


def _find_dump_group_for_required_tensors(
    rank: int,
    required_tensors: tuple[str, ...],
) -> FusedMlaDumpGroup | None:
    """
    Find a rank/layer group containing every requested tensor.

    This is the non-skipping version used by selectors that want to prefer a
    richer dump group, such as live q_b activations, before falling back to
    older parameter-only dumps.

    Args
    - rank: int, tensor-parallel rank id to inspect.
    - required_tensors: tuple[str, ...], tensor suffixes that must all exist in
        the same rank/layer group.

    Returns
    - group: FusedMlaDumpGroup | None, matching group if one exists.
    """
    layers = _layer_tensor_names(rank)
    matching_layers = [
        layer_idx
        for layer_idx, names in layers.items()
        if all(name in names for name in required_tensors)
    ]
    if not matching_layers:
        return None
    return FusedMlaDumpGroup(rank=rank, layer_idx=min(matching_layers))


def _find_dump_groups_for_required_tensors(
    rank: int,
    required_tensors: tuple[str, ...],
) -> list[FusedMlaDumpGroup]:
    """
    Find all rank/layer groups containing every requested tensor.

    Args
    - rank: int, tensor-parallel rank id to inspect.
    - required_tensors: tuple[str, ...], tensor suffixes that must all exist in
        the same rank/layer group.

    Returns
    - groups: list[FusedMlaDumpGroup], sorted matching groups.
    """
    layers = _layer_tensor_names(rank)
    return [
        FusedMlaDumpGroup(rank=rank, layer_idx=layer_idx)
        for layer_idx, names in sorted(layers.items())
        if all(name in names for name in required_tensors)
    ]


def _dump_group_for_required_tensors(
    rank: int,
    required_tensors: tuple[str, ...],
    description: str,
) -> FusedMlaDumpGroup:
    """
    Select a rank/layer dump group with all requested tensors.

    The first matching layer for the requested rank is used. If no matching
    layer exists, the pytest case is skipped with a message describing the
    missing dump group.

    Args
    - rank: int, tensor-parallel rank id to test.
    - required_tensors: tuple[str, ...], tensor suffixes that must be present in
        one rank/layer group.
    - description: str, human-readable dump group description used in skip
        messages.

    Returns
    - group: FusedMlaDumpGroup, rank and layer id used for all tensor loads.
    """
    group = _find_dump_group_for_required_tensors(rank, required_tensors)
    if group is None:
        pytest.skip(f"missing {description} dump for rank {rank} under {_debug_output_dir()}")
    return group


def _dump_group(rank: int) -> FusedMlaDumpGroup:
    """
    Select a rank/layer dump group with projection parameters.

    The current development dumps store full projection parameters and
    hidden-state activations under different rank/layer keys. This helper
    therefore selects the parameter group only; callers that need activations
    should use _load_hidden_states.

    Args
    - rank: int, tensor-parallel rank id to test.

    Returns
    - group: FusedMlaDumpGroup, rank and layer id used for projection tensor
        loads.
    """
    return _dump_group_for_required_tensors(
        rank,
        _REQUIRED_PROJECTION_DUMPS,
        "complete fused MLA projection-parameter",
    )


def _q_b_fusion_dump_group(rank: int) -> FusedMlaDumpGroup:
    """
    Select a rank/layer dump group with tensors needed by q_b fusion tests.

    The q_b fusion tests only need q_b projection weights/scales, k_b projection
    weights, and scalar shape metadata. This allows the tests to run even when
    the dump directory does not contain full hidden-state activations for the
    same layer.

    Args
    - rank: int, tensor-parallel rank id to test.

    Returns
    - group: FusedMlaDumpGroup, rank and layer id used for q_b fusion tensor
        loads.
    """
    return _dump_group_for_required_tensors(
        rank,
        _REQUIRED_Q_B_FUSION_DUMPS,
        "q_b fusion",
    )


def _live_q_b_fusion_dump_group(rank: int) -> FusedMlaDumpGroup:
    """
    Select a rank/layer dump group with live bench-path q_b activations.

    The live q_b tests should not fall back to reconstructed hidden states:
    they are meant to validate the exact q_b input and output saved from the
    model's WIP decode path during benching.

    Args
    - rank: int, tensor-parallel rank id in dump file names.

    Returns
    - group: FusedMlaDumpGroup, rank/layer group with live q_b input/output and
        post-load projection parameters.
    """
    return _dump_group_for_required_tensors(
        rank,
        _REQUIRED_LIVE_Q_B_FUSION_DUMPS,
        "live q_b fusion",
    )


def _live_q_b_fusion_dump_groups(rank: int) -> list[FusedMlaDumpGroup]:
    """
    Select all rank/layer dump groups with live bench-path q_b activations.

    Args
    - rank: int, tensor-parallel rank id in dump file names.

    Returns
    - groups: list[FusedMlaDumpGroup], all matching live q_b groups for rank.
    """
    groups = _find_dump_groups_for_required_tensors(
        rank,
        _REQUIRED_LIVE_Q_B_FUSION_DUMPS,
    )
    if not groups:
        pytest.skip(f"missing live q_b fusion dump for rank {rank} under {_debug_output_dir()}")
    return groups


def _dump_path(group: FusedMlaDumpGroup, tensor_name: str) -> Path:
    """
    Build the path to one dumped tensor file.

    The file naming convention is r<rank>_l<layer>_<tensor_name>.pt. This
    helper only constructs the path and has no file-system side effects.

    Args
    - group: FusedMlaDumpGroup, rank and layer id for the dump.
    - tensor_name: str, tensor suffix in the dump file name.

    Returns
    - path: Path, expected .pt file path for the requested tensor.
    """
    return _debug_output_dir() / f"r{group.rank}_l{group.layer_idx}_{tensor_name}.pt"


def _find_dump_path(
    group: FusedMlaDumpGroup,
    tensor_name: str,
    allow_any_rank: bool = True,
) -> Path | None:
    """
    Find a dumped tensor path, preferring the selected rank/layer group.

    This is used for optional activation dumps. It first checks the exact group,
    then any layer for the same rank, and finally any rank when allow_any_rank is
    true. It only inspects file names.

    Args
    - group: FusedMlaDumpGroup, preferred rank/layer id.
    - tensor_name: str, tensor suffix in the dump file name.
    - allow_any_rank: bool, whether to fall back to activation dumps from other
        ranks.

    Returns
    - path: Path | None, found dump path, or None when no matching file exists.
    """
    exact_path = _dump_path(group, tensor_name)
    if exact_path.exists():
        return exact_path

    same_rank_paths = sorted(_debug_output_dir().glob(f"r{group.rank}_l*_{tensor_name}.pt"))
    if same_rank_paths:
        return same_rank_paths[0]

    if not allow_any_rank:
        return None

    any_rank_paths = sorted(_debug_output_dir().glob(f"r*_l*_{tensor_name}.pt"))
    if any_rank_paths:
        return any_rank_paths[0]

    return None


def _load_tensor_from_path(path: Path) -> torch.Tensor:
    """
    Load one dumped tensor from an explicit path and move it to CUDA.

    The returned tensor is contiguous and lives on the current CUDA device. This
    helper raises AssertionError if the dump exists but is not a tensor. The
    side effects are disk I/O and a host-to-device copy.

    Args
    - path: Path, .pt file to load.

    Returns
    - tensor: torch.Tensor, CUDA contiguous tensor with the dtype and shape saved
        in the dump.
    """
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"{path} did not contain a tensor")
    return tensor.cuda().contiguous()


def _load_dump(group: FusedMlaDumpGroup, tensor_name: str) -> Any:
    """
    Load one dumped object from disk onto CPU.

    The helper uses torch.load with weights_only=False because the dump may
    contain scalars, lists, or tensors. It skips the pytest case if the file is
    missing. The side effect is reading the dump file from disk.

    Args
    - group: FusedMlaDumpGroup, rank and layer id for the dump.
    - tensor_name: str, tensor or scalar suffix in the dump file name.

    Returns
    - value: Any, object loaded from the requested .pt file on CPU.
    """
    path = _dump_path(group, tensor_name)
    if not path.exists():
        pytest.skip(f"missing tensor dump: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_tensor(group: FusedMlaDumpGroup, tensor_name: str) -> torch.Tensor:
    """
    Load one dumped tensor and move it to CUDA.

    The returned tensor is contiguous and lives on the current CUDA device. This
    helper raises AssertionError if the dump exists but is not a tensor. The
    side effects are disk I/O and a host-to-device copy.

    Args
    - group: FusedMlaDumpGroup, rank and layer id for the dump.
    - tensor_name: str, tensor suffix in the dump file name.

    Returns
    - tensor: torch.Tensor, CUDA contiguous tensor with the dtype and shape saved
        in the dump.
    """
    return _load_tensor_from_path(_dump_path(group, tensor_name))


def _load_optional_tensor(
    group: FusedMlaDumpGroup,
    tensor_name: str,
    allow_any_rank: bool = False,
) -> torch.Tensor | None:
    """
    Load an optional dumped tensor when it exists.

    Optional dumps are used for future richer activation dumps such as
    q_b_proj_input. Missing files return None instead of skipping the test.
    Existing files are loaded to CUDA.

    Args
    - group: FusedMlaDumpGroup, preferred rank/layer id.
    - tensor_name: str, tensor suffix in the dump file name.
    - allow_any_rank: bool, whether to fall back to matching files from other
        ranks.

    Returns
    - tensor: torch.Tensor | None, CUDA tensor when the dump exists, otherwise
        None.
    """
    path = _find_dump_path(group, tensor_name, allow_any_rank=allow_any_rank)
    if path is None:
        return None
    return _load_tensor_from_path(path)


def _load_hidden_states(group: FusedMlaDumpGroup) -> torch.Tensor:
    """
    Load hidden-state activations for projection smoke tests.

    The current debug dumps often save hidden_states under a different layer than
    the parameter dump. This helper first tries the selected group, then any
    hidden_states dump for the same rank, then any rank. It skips the pytest
    case if no activation dump is available.

    Args
    - group: FusedMlaDumpGroup, preferred rank/layer id.

    Returns
    - hidden_states: torch.Tensor, shape [M, hidden], CUDA tensor with dumped
        dtype, usually bf16.
    """
    tensor = _load_optional_tensor(group, "hidden_states", allow_any_rank=True)
    if tensor is None:
        pytest.skip(f"missing hidden_states dump under {_debug_output_dir()}")
    return tensor


def _load_scalar(group: FusedMlaDumpGroup, tensor_name: str) -> Any:
    """
    Load one dumped scalar-like value.

    Tensor scalar dumps are converted to Python values with item(); non-tensor
    objects are returned as loaded. The side effect is reading the dump file.

    Args
    - group: FusedMlaDumpGroup, rank and layer id for the dump.
    - tensor_name: str, scalar suffix in the dump file name.

    Returns
    - value: Any, Python scalar or object loaded from the requested dump.
    """
    value = _load_dump(group, tensor_name)
    if isinstance(value, torch.Tensor):
        return value.item()
    return value


def _load_int(group: FusedMlaDumpGroup, tensor_name: str) -> int:
    """
    Load one dumped scalar and cast it to int.

    This is used for MLA shape parameters such as head counts and LoRA ranks.
    The side effect is reading the dump file.

    Args
    - group: FusedMlaDumpGroup, rank and layer id for the dump.
    - tensor_name: str, scalar suffix in the dump file name.

    Returns
    - value: int, scalar value cast to int.
    """
    return int(_load_scalar(group, tensor_name))


def _load_float(group: FusedMlaDumpGroup, tensor_name: str) -> float:
    """
    Load one dumped scalar and cast it to float.

    This is used for floating-point MLA parameters such as softmax scale and
    RMSNorm epsilon. The side effect is reading the dump file.

    Args
    - group: FusedMlaDumpGroup, rank and layer id for the dump.
    - tensor_name: str, scalar suffix in the dump file name.

    Returns
    - value: float, scalar value cast to float.
    """
    return float(_load_scalar(group, tensor_name))
