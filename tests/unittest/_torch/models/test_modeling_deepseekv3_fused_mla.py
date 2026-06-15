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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from torch.nn import Parameter

from tensorrt_llm._torch.attention_backend.interface import AttentionInputType
from tensorrt_llm._torch.attention_backend.sparse.dsa import (
    transform_local_topk_and_prepare_pool_view,
)
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tests.unittest._torch.models.test_modeling_deepseekv3_attention import (
    _BENCH_CONTEXT_SEQUENCE_LENGTH,
    _DECODE_NUM_TOKENS,
    _HEAD_DIM,
    _KV_LORA_RANK,
    _LOCAL_NUM_HEADS,
    _Q_LORA_RANK,
    _QK_HEAD_DIM,
    _QK_NOPE_HEAD_DIM,
    _QK_ROPE_HEAD_DIM,
    _assert_context_attention_close,
    _assert_fused_q_close,
    _build_decode_attention_and_metadata,
    _build_decode_topk,
    _custom_decode_attention,
    _require_glm5_context_attention_runtime,
    _seed_decode_history_kv_cache,
)

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

    fallback_dirs.append(Path(__file__).resolve().parents[4].parent / "mla-debug-output")

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


def _run_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> torch.Tensor:
    """
    Run TensorRT-LLM RMSNorm with dumped weights.

    The helper builds a temporary RMSNorm module on the same device as the
    dumped weight and applies it to the input tensor. It is used to mirror the
    q_a_layernorm and kv_a_layernorm stages in the MLA projection flow. The
    side effect is temporary module allocation.

    Args
    - hidden_states: torch.Tensor, shape [M, hidden], bf16, tensor to normalize.
    - weight: torch.Tensor, shape [hidden], bf16, RMSNorm scale parameter.
    - variance_epsilon: float, epsilon used by the dumped RMSNorm module.

    Returns
    - output: torch.Tensor, shape [M, hidden], bf16, normalized tensor.
    """
    rms_norm = RMSNorm(
        hidden_size=weight.numel(),
        eps=variance_epsilon,
        dtype=weight.dtype,
        device=weight.device,
    )
    rms_norm.weight = Parameter(weight.contiguous(), requires_grad=False)
    rms_norm.eval()
    return rms_norm(hidden_states)


def _run_fp8_block_scale_linear(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Run TensorRT-LLM FP8 block-scale Linear with dumped weights.

    The helper constructs a temporary quantized Linear module, loads the dumped
    raw FP8 block-scale weight tensors, runs post_load_weights to transform
    scales into the runtime layout, then executes the module. The side effects
    are temporary module allocation, CUDA parameter copies, and autotuner/kernel
    execution.

    Args
    - hidden_states: torch.Tensor, shape [M, in_features], bf16, activation input.
    - weight: torch.Tensor, shape [out_features, in_features], fp8_e4m3, dumped
        quantized linear weight.
    - weight_scale: torch.Tensor, shape [ceil(out_features / 128),
        ceil(in_features / 128)], fp32, dumped block scale tensor.

    Returns
    - output: torch.Tensor, shape [M, out_features], bf16, linear output.
    """
    assert hidden_states.dtype == torch.bfloat16
    linear = _build_fp8_block_scale_linear(weight, weight_scale)
    return linear(hidden_states.contiguous())


def _build_fp8_block_scale_linear(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> Linear:
    """
    Build a TensorRT-LLM FP8 block-scale Linear module from dumped weights.

    The helper loads raw dumped FP8 block-scale weights, runs post_load_weights
    to transform scale tensors into the runtime layout, and returns the module
    without running it. q_b fusion tests use this to feed the WIP kernel the
    exact post-load weight and scale tensors that the model path uses.

    Args
    - weight: torch.Tensor, shape [out_features, in_features], fp8_e4m3, dumped
        quantized linear weight.
    - weight_scale: torch.Tensor, shape [ceil(out_features / 128),
        ceil(in_features / 128)], fp32, dumped block scale tensor.

    Returns
    - linear: Linear, CUDA eval-mode module with post-load FP8 block-scale
        weights.
    """
    assert weight.dtype == torch.float8_e4m3fn
    linear = Linear(
        weight.shape[1],
        weight.shape[0],
        bias=False,
        dtype=torch.bfloat16,
        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES),
    )
    linear.cuda()
    linear.load_weights([{"weight": weight, "weight_scale": weight_scale}])
    linear.post_load_weights()
    linear.eval()
    return linear


def _load_q_b_kernel_weight_and_scale(
    group: FusedMlaDumpGroup,
) -> tuple[torch.Tensor, torch.Tensor, Linear | None]:
    """
    Load q_b weight and scale in the layout expected by the WIP kernel.

    Live bench-path dumps save q_b_proj_weight_scale after Linear.post_load_weights,
    so the int32 scale tensor can be passed directly to the WIP op. Older dumps
    save raw floating-point block scales, which must be post-loaded through a
    temporary TensorRT-LLM Linear module.

    Args
    - group: FusedMlaDumpGroup, rank/layer q_b dump group.

    Returns
    - q_b_proj_weight: torch.Tensor, shape [2048, 2048], fp8_e4m3, post-load
        q_b projection weight.
    - q_b_proj_weight_scale: torch.Tensor, shape [2048, 4], int32, post-load
        packed UE8M0 q_b projection scale.
    - q_b_linear: Linear | None, temporary Linear module when raw scales needed
        post-load conversion, otherwise None for live post-load dumps.
    """
    q_b_proj_weight = _load_tensor(group, "q_b_proj_weight")
    q_b_proj_weight_scale = _load_tensor(group, "q_b_proj_weight_scale")
    if q_b_proj_weight_scale.dtype == torch.int32:
        return q_b_proj_weight, q_b_proj_weight_scale, None

    q_b_linear = _build_fp8_block_scale_linear(q_b_proj_weight, q_b_proj_weight_scale)
    return q_b_linear.weight.detach(), q_b_linear.weight_scale.detach(), q_b_linear


def _run_bf16_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Run TensorRT-LLM bmm_out for a BF16 batched matrix multiply.

    This wraps torch.ops.trtllm.bmm_out so the test follows the same BF16 BMM path
    used by the fused MLA code. That op works around a torch.bmm graph break when
    the out tensor is non-contiguous.
    The side effect is custom op execution into a newly allocated CUDA output tensor.

    Args
    - a: torch.Tensor, shape [B, M, K], bf16, left batched matrix operand.
    - b: torch.Tensor, shape [B, K, N], bf16, right batched matrix operand.

    Returns
    - out: torch.Tensor, shape [B, M, N], bf16, batched matrix product.
    """
    assert a.dtype == torch.bfloat16
    assert b.dtype == torch.bfloat16
    assert a.dim() == 3
    assert b.dim() == 3
    assert a.shape[0] == b.shape[0]
    assert a.shape[2] == b.shape[1]
    out = torch.empty(
        (a.shape[0], a.shape[1], b.shape[2]),
        dtype=a.dtype,
        device=a.device,
    )
    torch.ops.trtllm.bmm_out(a.contiguous(), b.contiguous(), out)
    return out


def _dummy_position_ids(num_tokens: int, device: torch.device) -> torch.Tensor:
    """
    Create deterministic dummy position ids for local RoPE smoke testing.

    The dumped test path does not reconstruct the full production RoPE metadata,
    so this helper supplies a simple sequence starting at position 4.

    Args
    - num_tokens: int, number of token positions M to generate.
    - device: torch.device, CUDA device where the returned tensor should live.

    Returns
    - position_ids: torch.Tensor, shape [M], int32, values [4, 5, 6, ...].
    """
    return torch.arange(4, 4 + num_tokens, dtype=torch.int32, device=device)


def _dummy_rope_cos_sin_cache(
    num_positions: int,
    rope_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a dummy cos/sin cache for the TRTLLM MLA RoPE kernel.

    The cache uses the same base-10000 RoPE formula as _apply_dummy_rope and is
    laid out as [max_position, 2, rope_dim / 2], matching
    torch.ops.trtllm.mla_rope_inplace.

    Args
    - num_positions: int, number of position rows to materialize.
    - rope_dim: int, rotary dimension per head.
    - device: torch.device, CUDA device where the returned cache should live.

    Returns
    - cos_sin_cache: torch.Tensor, shape [num_positions, 2, rope_dim / 2],
        fp32, cosine and sine factors consumed by the TRTLLM RoPE kernel.
    """
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
    )
    positions = torch.arange(num_positions, dtype=torch.float32, device=device)
    freqs = torch.outer(positions, inv_freq)
    return torch.stack((freqs.cos(), freqs.sin()), dim=1).contiguous()


def _apply_neox_rope(
    tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply NeoX-style RoPE to a tensor using precomputed cos and sin values.

    The helper rotates the leading rotary dimensions and preserves any trailing
    pass-through dimensions. It accepts either per-token tensors or per-token
    per-head tensors.

    Args
    - tensor: torch.Tensor, shape [M, D] or [M, n_heads, D], bf16, tensor to
        rotate.
    - cos: torch.Tensor, shape [M, D / 2], bf16, cosine factors.
    - sin: torch.Tensor, shape [M, D / 2], bf16, sine factors.

    Returns
    - output: torch.Tensor, same shape and dtype as tensor, RoPE-rotated tensor.
    """
    if tensor.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

    rot_dim = cos.shape[-1] * 2
    tensor_rot, tensor_pass = tensor[..., :rot_dim], tensor[..., rot_dim:]
    tensor_1, tensor_2 = tensor_rot.chunk(2, dim=-1)
    tensor_rot = torch.cat(
        (tensor_1 * cos - tensor_2 * sin, tensor_2 * cos + tensor_1 * sin),
        dim=-1,
    )
    return torch.cat((tensor_rot, tensor_pass), dim=-1)


def _apply_dummy_rope(
    q_pe: torch.Tensor,
    k_pe: torch.Tensor,
    position_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply dummy NeoX-style RoPE to q_pe and k_pe.

    This uses the standard base-10000 RoPE formula with caller-provided dummy
    position ids. The rotation is computed in fp32 and cast back to the input
    dtype to mirror the TRTLLM MLA RoPE kernel. It is only a smoke-test
    approximation and does not claim to match the production GLM-5/DeepSeek
    RoPE configuration. It has no side effects.

    Args
    - q_pe: torch.Tensor, shape [M, n_heads, rope_dim], bf16, query RoPE slice.
    - k_pe: torch.Tensor, shape [M, rope_dim], bf16, key RoPE slice.
    - position_ids: torch.Tensor, shape [M], int32, dummy token positions.

    Returns
    - q_pe: torch.Tensor, shape [M, n_heads, rope_dim], bf16, rotated query RoPE
        slice.
    - k_pe: torch.Tensor, shape [M, rope_dim], bf16, rotated key RoPE slice.
    """
    rope_dim = q_pe.shape[-1]
    inv_freq = 1.0 / (
        10000.0
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=q_pe.device) / rope_dim)
    )
    freqs = torch.outer(position_ids.to(torch.float32), inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return (
        _apply_neox_rope(q_pe.float(), cos, sin).to(q_pe.dtype),
        _apply_neox_rope(k_pe.float(), cos, sin).to(k_pe.dtype),
    )


def _apply_trtllm_mla_rope(
    q_heads: torch.Tensor,
    k_pe: torch.Tensor,
    position_ids: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply TRTLLM MLA RoPE to q heads and k_pe with mla_rope_inplace.

    The TRTLLM kernel rotates only the last rope_dim values of each head. For
    q_heads this preserves the leading no-RoPE query slice; for k_pe the helper
    temporarily views the tensor as a single-head tensor with zero no-RoPE
    dimensions. The side effect is CUDA custom op execution on cloned tensors.

    Args
    - q_heads: torch.Tensor, shape [M, n_heads, nope_dim + rope_dim], bf16,
        unrotated query heads.
    - k_pe: torch.Tensor, shape [M, rope_dim], bf16, unrotated key RoPE slice.
    - position_ids: torch.Tensor, shape [M], int32, dummy token positions.
    - nope_dim: int, number of leading query dimensions that are not rotated.
    - rope_dim: int, number of rotary dimensions per query/key head.

    Returns
    - q_nope: torch.Tensor, shape [M, n_heads, nope_dim], bf16, query no-RoPE
        slice preserved by the kernel.
    - q_pe: torch.Tensor, shape [M, n_heads, rope_dim], bf16, query RoPE slice
        rotated by torch.ops.trtllm.mla_rope_inplace.
    - k_pe: torch.Tensor, shape [M, rope_dim], bf16, key RoPE slice rotated by
        torch.ops.trtllm.mla_rope_inplace.
    """
    assert q_heads.dtype == torch.bfloat16
    assert k_pe.dtype == torch.bfloat16
    assert position_ids.dtype == torch.int32
    assert q_heads.shape[-1] == nope_dim + rope_dim
    assert k_pe.shape[-1] == rope_dim

    cos_sin_cache = _dummy_rope_cos_sin_cache(
        int(position_ids.max().item()) + 1,
        rope_dim,
        q_heads.device,
    )
    q_heads = q_heads.clone().contiguous()
    k_pe_heads = k_pe.view(k_pe.shape[0], 1, rope_dim).clone().contiguous()

    torch.ops.trtllm.mla_rope_inplace(
        q_heads,
        position_ids,
        cos_sin_cache,
        q_heads.shape[1],
        nope_dim,
        rope_dim,
        False,
        True,
    )
    torch.ops.trtllm.mla_rope_inplace(
        k_pe_heads,
        position_ids,
        cos_sin_cache,
        1,
        0,
        rope_dim,
        False,
        True,
    )
    q_nope, q_pe = q_heads.split([nope_dim, rope_dim], dim=-1)
    return q_nope, q_pe, k_pe_heads.view(k_pe.shape[0], rope_dim)


def _assert_bf16_cuda_finite(name: str, tensor: torch.Tensor, num_tokens: int) -> None:
    """
    Assert that a tensor is finite BF16 CUDA data with the expected token count.

    The smoke test uses this for every major output tensor. Assertion failures
    identify the named tensor in the failure message. It has no side effects
    beyond test assertions.

    Args
    - name: str, human-readable tensor name used in assertion messages.
    - tensor: torch.Tensor, shape [M, ...], bf16, CUDA tensor to validate.
    - num_tokens: int, expected leading dimension M.

    Returns
    - None: successful return means all validation assertions passed.
    """
    assert tensor.is_cuda, f"{name} must be a CUDA tensor"
    assert tensor.dtype == torch.bfloat16, f"{name} has dtype {tensor.dtype}"
    assert tensor.shape[0] == num_tokens, f"{name} has shape {tuple(tensor.shape)}"
    assert torch.isfinite(tensor.float()).all().item(), f"{name} contains NaN or Inf"


def _build_q_b_projection_input_from_hidden_states(
    group: FusedMlaDumpGroup,
    num_tokens: int,
) -> torch.Tensor:
    """
    Build q_b projection input by running dumped hidden states through TRTLLM modules.

    This mirrors the existing non-WIP path up to q_b_proj: hidden_states are
    passed through kv_a_proj_with_mqa, the q slice is normalized by
    q_a_layernorm, and that normalized q becomes q_b_proj_input. The dump
    directory can hold hidden_states under a different rank/layer than the
    projection parameters, so _load_hidden_states applies the same fallback
    search used by the projection smoke test.

    Args
    - group: FusedMlaDumpGroup, parameter group for q_b and kv_a tensors.
    - num_tokens: int, number of decode tokens to prepare.

    Returns
    - q_b_proj_input: torch.Tensor, shape [num_tokens, q_lora_rank], bf16, input
        activation produced by the existing TRTLLM module path for q_b_proj.
    """
    # [M_dump, hidden_size]
    hidden_states = _load_hidden_states(group).to(torch.bfloat16)
    assert hidden_states.dim() == 2
    assert hidden_states.shape[0] > 0
    repeat_count = (num_tokens + hidden_states.shape[0] - 1) // hidden_states.shape[0]
    # [num_tokens, hidden_size]
    hidden_states = hidden_states.repeat(repeat_count, 1)[:num_tokens].contiguous()
    q_lora_rank = _load_int(group, "q_lora_rank")
    kv_lora_rank = _load_int(group, "kv_lora_rank")
    qk_rope_head_dim = _load_int(group, "qk_rope_head_dim")

    # [num_tokens, q_lora_rank + kv_lora_rank + qk_rope_head_dim]
    kv_a_output = _run_fp8_block_scale_linear(
        hidden_states,
        _load_tensor(group, "kv_a_proj_with_mqa_weight"),
        _load_tensor(group, "kv_a_proj_with_mqa_weight_scale"),
    )
    # q: [num_tokens, q_lora_rank]
    q, _, _ = kv_a_output.split(
        [q_lora_rank, kv_lora_rank, qk_rope_head_dim],
        dim=-1,
    )
    # [num_tokens, q_lora_rank]
    return _run_rms_norm(
        q,
        _load_tensor(group, "q_a_layernorm_weight"),
        _load_float(group, "q_a_layernorm_variance_epsilon"),
    ).contiguous()


def _build_dump_q_b_projection_inputs(
    group: FusedMlaDumpGroup,
    num_tokens: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build q_b fused-kernel inputs from dumped FP8 projection parameters.

    The q_b input and output come from the existing non-WIP TRTLLM module path:
    dumped hidden_states are run through kv_a_proj_with_mqa, q_a_layernorm, and
    q_b_proj. The returned q_b weight and scale are post-load tensors, matching
    the dtype and layout that modeling_deepseekv3_fused_mla.py passes to
    torch.ops.trtllm.dsv3_fused_mla_generation.

    Args
    - group: FusedMlaDumpGroup, rank/layer parameter dump to load.
    - num_tokens: int, number of decode tokens to prepare.
    - device: torch.device, CUDA device associated with the decode case; kept in
        the signature for symmetry with other dump builders.

    Returns
    - q_b_proj_input: torch.Tensor, shape [num_tokens, 2048], bf16, q_b input.
    - q_b_proj_output: torch.Tensor, shape [num_tokens, 2048], bf16, existing
        TRTLLM q_b projection output before splitting into heads.
    - q_b_proj_weight: torch.Tensor, shape [2048, 2048], fp8_e4m3, post-load
        q_b projection weight.
    - q_b_proj_weight_scale: torch.Tensor, shape [2048, 4], int32, post-load
        packed UE8M0 q_b projection scale.
    - q_nope: torch.Tensor, shape [num_tokens, 8, 192], bf16, preprojected
        non-RoPE query slice.
    - q_pe: torch.Tensor, shape [num_tokens, 8, 64], bf16, preprojected query
        RoPE slice before rotary embedding.
    """
    _ = device
    (
        q_b_proj_weight,
        q_b_proj_weight_scale,
        q_b_linear,
    ) = _load_q_b_kernel_weight_and_scale(group)
    dumped_q_b_input = _load_optional_tensor(group, "q_b_proj_input")
    dumped_q_b_output = _load_optional_tensor(group, "q_b_proj_output")
    if (
        dumped_q_b_input is not None
        and dumped_q_b_output is not None
        and dumped_q_b_input.shape[0] >= num_tokens
        and dumped_q_b_output.shape[0] >= num_tokens
    ):
        # [num_tokens, q_lora_rank]
        q_b_proj_input = dumped_q_b_input[:num_tokens].to(torch.bfloat16).contiguous()
        # [num_tokens, num_heads_tp * qk_head_dim]
        q_output = dumped_q_b_output[:num_tokens].to(torch.bfloat16).contiguous()
    else:
        q_b_proj_input = _build_q_b_projection_input_from_hidden_states(group, num_tokens)
        if q_b_linear is None:
            q_b_linear = _build_fp8_block_scale_linear(
                q_b_proj_weight,
                q_b_proj_weight_scale,
            )

        # [num_tokens, num_heads_tp * qk_head_dim]
        q_output = q_b_linear(q_b_proj_input.contiguous())

    num_heads_tp = _load_int(group, "num_heads_tp")
    qk_head_dim = _load_int(group, "qk_head_dim")
    qk_nope_head_dim = _load_int(group, "qk_nope_head_dim")
    qk_rope_head_dim = _load_int(group, "qk_rope_head_dim")
    assert num_heads_tp == _LOCAL_NUM_HEADS
    assert q_b_proj_input.shape[-1] == _Q_LORA_RANK
    assert qk_head_dim == _QK_HEAD_DIM
    assert qk_nope_head_dim == _QK_NOPE_HEAD_DIM
    assert qk_rope_head_dim == _QK_ROPE_HEAD_DIM

    # [num_tokens, num_heads_tp, qk_head_dim]
    q_heads = q_output.view(num_tokens, num_heads_tp, qk_head_dim)
    # q_nope: [num_tokens, num_heads_tp, qk_nope_head_dim]
    # q_pe: [num_tokens, num_heads_tp, qk_rope_head_dim]
    q_nope, q_pe = q_heads.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
    return (
        q_b_proj_input,
        q_output.contiguous(),
        q_b_proj_weight,
        q_b_proj_weight_scale,
        q_nope.contiguous(),
        q_pe.contiguous(),
    )


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
    num_seqs = case.metadata.kv_lens_cuda_runtime.size(0)
    fused_q = torch.empty(
        _DECODE_NUM_TOKENS,
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
    )
    return output, fused_q, quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale


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


def _assert_q_b_selector_prefix_close(
    fused_q: torch.Tensor,
    q_b_proj_output: torch.Tensor,
) -> None:
    """
    Assert that selector k_b exposes q_b output exactly in fused_q.

    With _selector_k_b_proj_trans, the first 192 latent-prefix dimensions should
    be exactly the q_nope slice from q_b_proj_output and the remaining 320
    dimensions should be exactly zero. This is intentionally stricter than the
    full attention output tolerance so a one-BF16-step q_b projection mismatch
    fails the test.

    Args
    - fused_q: torch.Tensor, shape [4, 8, 576], bf16, output buffer from the
        fused decode kernel.
    - q_b_proj_output: torch.Tensor, shape [4, 2048], bf16, existing TRTLLM q_b
        projection output before splitting into heads.

    Returns
    - None: successful return means the q_b projection path is bit-close enough
        to the Linear reference for all exposed q_nope values.
    """
    # [num_tokens, 8, 256]
    q_heads = q_b_proj_output.view(q_b_proj_output.shape[0], _LOCAL_NUM_HEADS, _QK_HEAD_DIM)
    # [num_tokens, 8, 192]
    q_nope = q_heads[..., :_QK_NOPE_HEAD_DIM]
    # [num_tokens, 8, 512]
    expected_prefix = torch.zeros_like(fused_q[..., :_KV_LORA_RANK])
    # [num_tokens, 8, 192]
    expected_prefix[..., :_QK_NOPE_HEAD_DIM] = q_nope
    torch.testing.assert_close(
        fused_q[..., :_KV_LORA_RANK],
        expected_prefix,
        rtol=0.0,
        atol=0.0,
    )


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


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_mla_dump_projection_smoke(rank: int) -> None:
    """
    Smoke test dumped fused MLA projection and downstream local attention flow.

    For each rank, this test loads a realistic dumped MLA activation/weight
    group, runs kv_a_proj_with_mqa, q/kv RMSNorm, q_b_proj, TRTLLM MLA RoPE
    checked against a dummy PyTorch RoPE reference, absorption-style local latent
    attention, v_b_proj, and o_proj. The attention section is a local smoke path
    rather than a reconstruction of production paged-KV AttentionMetadata. Side
    effects include disk reads, CUDA tensor allocations, custom op execution,
    and possible pytest skips when dumps or runtime support are missing.

    Args
    - rank: int, tensor-parallel rank id selected by pytest parametrization.

    Returns
    - None: successful return means the dumped rank flow produced finite tensors
        with expected shapes.
    """
    _require_cuda_and_ops()
    group = _dump_group(rank)

    # Shape symbols used below:
    # M: token count after optional slicing.
    # H: hidden size.
    # Qrank: q LoRA rank.
    # KVrank: compressed KV LoRA rank.
    # Nh: local TP attention heads.
    # Nhcp: local TP/CP attention heads.
    # Dq: q/k head dim.
    # Dnope: q/k non-RoPE head dim.
    # Drope: q/k RoPE head dim.
    # Dv: value head dim.
    hidden_states = _load_hidden_states(group).to(torch.bfloat16)
    # [M_dump, H]
    max_num_tokens = _max_num_tokens()
    if max_num_tokens is not None:
        hidden_states = hidden_states[:max_num_tokens].contiguous()
        # [M, H]
    num_tokens = hidden_states.shape[0]
    assert hidden_states.dim() == 2
    assert num_tokens > 0

    q_lora_rank = _load_int(group, "q_lora_rank")
    kv_lora_rank = _load_int(group, "kv_lora_rank")
    num_heads_tp = _load_int(group, "num_heads_tp")
    num_heads_tp_cp = _load_int(group, "num_heads_tp_cp")
    qk_head_dim = _load_int(group, "qk_head_dim")
    qk_nope_head_dim = _load_int(group, "qk_nope_head_dim")
    qk_rope_head_dim = _load_int(group, "qk_rope_head_dim")
    v_head_dim = _load_int(group, "v_head_dim")
    softmax_scale = _load_float(group, "softmax_scale")
    kv_a_output_width = q_lora_rank + kv_lora_rank + qk_rope_head_dim

    with torch.inference_mode(), autotune():
        # [M, Q_lora_r + kv_lora_r + rope_dim]
        kv_a_output = _run_fp8_block_scale_linear(
            hidden_states,
            _load_tensor(group, "kv_a_proj_with_mqa_weight"),
            _load_tensor(group, "kv_a_proj_with_mqa_weight_scale"),
        )
        assert kv_a_output.shape[-1] == kv_a_output_width
        # q: [M, Q_lora_r]
        # compressed_kv: [M, kv_lora_r]
        # k_pe: [M, rope_dim]
        q, compressed_kv, k_pe = kv_a_output.split(
            [q_lora_rank, kv_lora_rank, qk_rope_head_dim], dim=-1
        )
        # q: [M, Q_lora_r]
        q = _run_rms_norm(
            q,
            _load_tensor(group, "q_a_layernorm_weight"),
            _load_float(group, "q_a_layernorm_variance_epsilon"),
        )
        # [M, kv_lora_r]
        compressed_kv = _run_rms_norm(
            compressed_kv,
            _load_tensor(group, "kv_a_layernorm_weight"),
            _load_float(group, "kv_a_layernorm_variance_epsilon"),
        )
        # [M, n_heads * q_dim]
        q_output = _run_fp8_block_scale_linear(
            q,
            _load_tensor(group, "q_b_proj_weight"),
            _load_tensor(group, "q_b_proj_weight_scale"),
        )
        assert q_output.shape[-1] == num_heads_tp * qk_head_dim
        assert num_heads_tp == num_heads_tp_cp

        # [M, n_heads, q_dim]
        q_heads = q_output.view(num_tokens, num_heads_tp, qk_head_dim)
        # [M], starts at [4, 5, 6, 7]
        dummy_position_ids = _dummy_position_ids(num_tokens, q_heads.device)
        # q_nope: [M, n_heads, q_nope_dim]
        # q_pe: [M, n_heads, rope_dim]
        # k_pe: [M, rope_dim]
        q_nope, q_pe, k_pe = _apply_trtllm_mla_rope(
            q_heads,
            k_pe,
            dummy_position_ids,
            qk_nope_head_dim,
            qk_rope_head_dim,
        )
        # [M, kv_lora_r + rope_dim]
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
        # [n_heads, kv_lora_r, q_nope_dim]
        k_b_proj_trans = _load_tensor(group, "k_b_proj_trans")
        # absorp into query vector
        # [M, n_heads, kv_lora_r]
        q_nope_latent = _run_bf16_bmm(
            q_nope.transpose(0, 1),  # [n_heads, M, q_nope_dim]
            k_b_proj_trans.transpose(1, 2),  # [n_heads, q_nope_dim, kv_lora_r]
        ).transpose(0, 1)
        # [M, n_heads, kv_lora_r + rope_dim]
        fused_q = torch.concat([q_nope_latent, q_pe], dim=-1)

        # ===== TRTLLM C++ MLA Kernel =====
        # dot product of each token's query head to each token's kv latent cache
        # [M, n_heads, M]
        scores = torch.matmul(fused_q.float(), latent_cache.float().transpose(0, 1))
        # [M, n_heads, M]
        scores = scores * softmax_scale
        # [M, n_heads, M]
        weights = torch.softmax(scores, dim=-1)
        # compressed_kv: [M, kv_lora_r]
        # compute weighted latent output for each head: [M, n_heads, kv_lora_r]
        attn_out_latent = torch.matmul(
            weights.float(), compressed_kv[..., :kv_lora_rank].float()
        ).to(torch.bfloat16)
        # ===== TRTLLM C++ MLA Kernel Ends =====

        # [n_heads, v_dim, kv_lora_r]
        v_b_proj = _load_tensor(group, "v_b_proj")
        # [M, n_heads, v_dim]
        attn_output = _run_bf16_bmm(
            attn_out_latent.transpose(0, 1),  # [n_heads, M, kv_lora_r]
            v_b_proj.transpose(1, 2),  # [n_heads, kv_lora_r, v_dim]
        ).transpose(0, 1)
        # [M, n_heads*v_dim]
        attn_output = attn_output.contiguous().view(num_tokens, num_heads_tp_cp * v_head_dim)
        # [M, hidden_size]
        final_output = _run_fp8_block_scale_linear(
            attn_output,
            _load_tensor(group, "o_proj_weight"),
            _load_tensor(group, "o_proj_weight_scale"),
        )

    torch.cuda.synchronize()

    _assert_bf16_cuda_finite("kv_a_output", kv_a_output, num_tokens)
    _assert_bf16_cuda_finite("q_output", q_output, num_tokens)
    _assert_bf16_cuda_finite("latent_cache", latent_cache, num_tokens)
    _assert_bf16_cuda_finite("attn_out_latent", attn_out_latent, num_tokens)
    _assert_bf16_cuda_finite("attn_output", attn_output, num_tokens)
    _assert_bf16_cuda_finite("final_output", final_output, num_tokens)
    assert latent_cache.shape[-1] == kv_lora_rank + qk_rope_head_dim
    assert attn_output.shape[-1] == num_heads_tp_cp * v_head_dim
    assert final_output.shape[-1] == hidden_states.shape[-1]


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_mla_dump_decode_q_b_proj_matches_preprojected_path(
    rank: int,
) -> None:
    """
    Compare dumped q_b fused decode against the preprojected WIP decode path.

    This is the main q_b fusion accuracy test. Both sides call
    dsv3_fused_mla_generation; the expected side receives q_nope/q_pe from the
    dumped q_b Linear reference, while the actual side receives q_b input,
    post-load q_b FP8 weight, and packed q_b scale. It isolates differences
    introduced by fusing q_b projection into the custom op.

    Args
    - rank: int, tensor-parallel rank id selected by pytest parametrization.

    Returns
    - None: successful return means output, fused query prefix, scales, and FP8
        query bytes match the preprojected path within test tolerances.
    """
    case = _build_dump_decode_q_b_case(rank)

    with torch.inference_mode():
        (
            expected,
            fused_q_expected,
            quant_q_buffer_expected,
            mla_bmm1_scale_expected,
            mla_bmm2_scale_expected,
        ) = _run_dump_decode_preprojected(case)
        (
            actual,
            fused_q_actual,
            quant_q_buffer_actual,
            mla_bmm1_scale_actual,
            mla_bmm2_scale_actual,
        ) = _run_dump_decode_fused_q_b(case)

    _assert_context_attention_close(actual, expected)
    _assert_fused_q_close(
        fused_q_actual[..., :_KV_LORA_RANK],
        fused_q_expected[..., :_KV_LORA_RANK],
    )
    torch.testing.assert_close(mla_bmm1_scale_actual, mla_bmm1_scale_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(mla_bmm2_scale_actual, mla_bmm2_scale_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(quant_q_buffer_actual, quant_q_buffer_expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_mla_dump_decode_q_b_proj_raw_output_matches_linear(
    rank: int,
) -> None:
    """
    Compare fused q_b projection directly against dumped Linear reference output.

    This test replaces the real k_b projection with a selector projection so the
    fused kernel writes q_b's q_nope output directly into fused_q. It catches
    q_b projection errors that can be hidden by the subsequent real k_b
    projection and attention tolerance. The reference q_b input/output are
    produced by running dumped hidden_states through the existing TRTLLM module
    path.

    Args
    - rank: int, tensor-parallel rank id selected by pytest parametrization.

    Returns
    - None: successful return means fused q_b projection reproduces the existing
        TensorRT-LLM Linear q_b output exactly for the selector-visible q_nope
        values, and produces identical downstream fused_q/FP8 query buffers.
    """
    case = _build_dump_decode_q_b_case(rank)
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

    _assert_q_b_selector_prefix_close(fused_q_actual, case.q_b_proj_output)
    torch.testing.assert_close(
        fused_q_actual,
        fused_q_expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(mla_bmm1_scale_actual, mla_bmm1_scale_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(mla_bmm2_scale_actual, mla_bmm2_scale_expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(quant_q_buffer_actual, quant_q_buffer_expected, rtol=0.0, atol=0.0)


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
