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
from torch.nn import Parameter

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_DEBUG_OUTPUT_DIR"
_RUNTIME_DEBUG_OUTPUT_DIR_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_DEBUG_OUTPUT_DIR"
_DEFAULT_DEBUG_OUTPUT_DIR = "~/dev/mla-debug-output"
_EXTRA_OP_LIBRARY_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_EXTRA_OP_LIBRARY"
_MAX_NUM_TOKENS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MLA_TEST_MAX_NUM_TOKENS"

_NUM_RANKS = 8
_DEFAULT_MAX_NUM_TOKENS = 128
_EXTRA_OP_LIBRARIES_LOADED = False

_REQUIRED_PROJECTION_DUMPS = (
    "hidden_states",
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

    required_ops = ("bmm_out", "fp8_swap_ab_gemm", "mla_rope_inplace")
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


def _dump_group(rank: int) -> FusedMlaDumpGroup:
    """
    Select a rank/layer dump group with all tensors required by the smoke test.

    The first matching layer for the requested rank is used. If no matching
    layer exists, the pytest case is skipped with a message describing the
    missing dump group.

    Args
    - rank: int, tensor-parallel rank id to test.

    Returns
    - group: FusedMlaDumpGroup, rank and layer id used for all tensor loads.
    """
    layers = _layer_tensor_names(rank)
    matching_layers = [
        layer_idx
        for layer_idx, names in layers.items()
        if all(name in names for name in _REQUIRED_PROJECTION_DUMPS)
    ]
    if not matching_layers:
        pytest.skip(
            f"missing complete fused MLA projection dump for rank {rank} "
            f"under {_debug_output_dir()}"
        )
    return FusedMlaDumpGroup(rank=rank, layer_idx=min(matching_layers))


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
    tensor = _load_dump(group, tensor_name)
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"{_dump_path(group, tensor_name)} did not contain a tensor")
    return tensor.cuda().contiguous()


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
    return linear(hidden_states.contiguous())


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
    hidden_states = _load_tensor(group, "hidden_states").to(torch.bfloat16)
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
