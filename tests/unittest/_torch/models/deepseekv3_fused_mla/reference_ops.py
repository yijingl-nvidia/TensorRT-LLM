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
from torch.nn import Parameter

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo
from tests.unittest._torch.models.deepseekv3_fused_mla.dump_utils import (
    FusedMlaDumpGroup,
    _load_float,
    _load_hidden_states,
    _load_int,
    _load_optional_tensor,
    _load_tensor,
)
from tests.unittest._torch.models.test_modeling_deepseekv3_attention import (
    _LOCAL_NUM_HEADS,
    _Q_LORA_RANK,
    _QK_HEAD_DIM,
    _QK_NOPE_HEAD_DIM,
    _QK_ROPE_HEAD_DIM,
)


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
    *,
    disable_deep_gemm: bool = False,
    maintain_original_weight: bool = False,
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
    - disable_deep_gemm: bool, whether to keep the raw FP32 block-scale layout
        instead of post-load resmoothing for DeepGEMM.
    - maintain_original_weight: bool, whether Linear should preserve the
        pre-resmooth FP8 weight and FP32 scales as `weight_orig` and
        `weight_scale_orig`.

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
        disable_deep_gemm=disable_deep_gemm,
        maintain_original_weight=maintain_original_weight,
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
