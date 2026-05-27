import os
from typing import Dict, Optional

import torch
import triton
from torch import nn

from tensorrt_llm import deep_gemm
from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.fused_moe import MoE
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_deepseekv3_moe import Deepseekv3MoE


def check_data(tensor: torch.Tensor, name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    """
    Debug function call to assert input tensor shape and dtype
    Tensor shape value can have -1, meaning no assert on that dimension
    """
    assert tensor.dtype == dtype, f"{name} must have dtype {dtype}, got {tensor.dtype}"
    assert tensor.ndim == len(shape), f"{name} must have {len(shape)} dims, got {tensor.shape}"
    for dim, expected_dim in enumerate(shape):
        if expected_dim == -1:
            continue
        assert tensor.shape[dim] == expected_dim, (
            f"{name} dim {dim} must be {expected_dim}, got "
            f"{tensor.shape[dim]} for shape {tensor.shape}"
        )


class Deepseekv3MegaMoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
        model_config: ModelConfig = ModelConfig(),
        override_quant_config: Optional[QuantConfig] = None,
    ):
        super().__init__()
        self._old_moe = Deepseekv3MoE(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
            dtype=dtype,
            model_config=model_config,
            override_quant_config=override_quant_config,
        )
        self._old_moe.shared_experts.gate_up_proj.retain_pre_deep_gemm_weight = True
        self.layer_idx = layer_idx

    @property
    def experts(self) -> MoE:
        return self._old_moe.experts

    _MEGA_MOE_MODE_BASELINE = "baseline"
    _MEGA_MOE_MODE_PYTORCH_REF = "pytorch_ref"
    _MEGA_MOE_MODE_WIP = "wip_mega_kernel"

    @classmethod
    def _mega_moe_mode(cls) -> str:
        mode = os.environ.get("TRTLLM_DEEPSEEKV3_MEGAMOE_MODE", "").strip().lower()
        if not mode:
            return cls._MEGA_MOE_MODE_BASELINE

        mode_aliases = {
            "baseline": cls._MEGA_MOE_MODE_BASELINE,
            "trtllm": cls._MEGA_MOE_MODE_BASELINE,
            "trtllm_baseline": cls._MEGA_MOE_MODE_BASELINE,
            "ref": cls._MEGA_MOE_MODE_PYTORCH_REF,
            "reference": cls._MEGA_MOE_MODE_PYTORCH_REF,
            "torch": cls._MEGA_MOE_MODE_PYTORCH_REF,
            "pytorch": cls._MEGA_MOE_MODE_PYTORCH_REF,
            "pytorch_ref": cls._MEGA_MOE_MODE_PYTORCH_REF,
            "mega": cls._MEGA_MOE_MODE_WIP,
            "mega_kernel": cls._MEGA_MOE_MODE_WIP,
            "wip": cls._MEGA_MOE_MODE_WIP,
            "wip_mega_kernel": cls._MEGA_MOE_MODE_WIP,
        }
        if mode not in mode_aliases:
            allowed_modes = ", ".join(
                (
                    cls._MEGA_MOE_MODE_BASELINE,
                    cls._MEGA_MOE_MODE_PYTORCH_REF,
                    cls._MEGA_MOE_MODE_WIP,
                )
            )
            raise ValueError(
                f"Unsupported TRTLLM_DEEPSEEKV3_MEGAMOE_MODE={mode!r}; "
                f"expected one of: {allowed_modes}"
            )
        return mode_aliases[mode]

    @staticmethod
    def _pytorch_ref_chunk_size() -> int:
        chunk_size = os.environ.get("TRTLLM_DEEPSEEKV3_MEGAMOE_PYTORCH_REF_CHUNK_SIZE", "8")
        return max(1, int(chunk_size))

    @staticmethod
    def _pytorch_ref_use_shared_gate_up_weight_org() -> bool:
        value = os.environ.get("TRTLLM_DEEPSEEKV3_MEGAMOE_REF_USE_SHARED_GATE_UP_WEIGHT_ORG", "0")
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
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

    @classmethod
    def _dequantize_fp8_1x128_packed_ue8m0_weight(
        cls, weight: torch.Tensor, packed_scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize FP8 weights that use 1x128 block scales in packed UE8M0 format.

        Each row has one UE8M0 scale byte for every 128 contiguous columns of
        ``weight``. The scale tensor stores four scale bytes in each int32
        element, unpacked from least significant byte to most significant byte.
        A UE8M0 byte of zero is treated as an exact zero scale; otherwise the
        scale value is ``2 ** (byte - 127)``.

        Args:
            weight: FP8 E4M3 weight tensor with shape ``[rows, cols]``.
            packed_scale: Int32 packed scale tensor with shape
                ``[rows, ceil(ceil(cols / 128) / 4)]``.

        Returns:
            Dequantized FP32 weight tensor with shape ``[rows, cols]``.
        """
        rows, cols = weight.shape
        num_scale_cols = (cols + 127) // 128
        scales = cls._unpack_ue8m0_scales(packed_scale, num_scale_cols)
        scales = scales.repeat_interleave(128, dim=1)[:, :cols]
        assert scales.shape == (rows, cols)
        return weight.to(torch.float32) * scales

    @staticmethod
    def _dequantize_fp8_128x128_block_weight(
        weight: torch.Tensor, weight_scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantize FP8 weights that use 128x128 block scales.

        Args:
            weight: FP8 E4M3 weight tensor with shape `[..., rows, cols]`.
            weight_scale: FP32 weight scale tensor with shape
                `[..., rows / 128, cols / 128]`.

        Returns:
            Dequantized FP32 weight tensor with shape `[...,rows, cols]`.
        """
        rows, cols = weight.shape[-2:]
        scales = weight_scale.repeat_interleave(128, dim=-2)[..., :rows, :]
        scales = scales.repeat_interleave(128, dim=-1)[..., :, :cols]
        assert scales.shape == weight.shape
        return weight.to(torch.float32) * scales

    @staticmethod
    def _silu_and_mul_pytorch(
        gate: torch.Tensor, up: torch.Tensor, swiglu_limit: Optional[float]
    ) -> torch.Tensor:
        gate = gate.to(torch.float32)
        up = up.to(torch.float32)
        if swiglu_limit is not None:
            gate = torch.minimum(gate, torch.tensor(float(swiglu_limit), device=gate.device))
            up = torch.clamp(up, -float(swiglu_limit), float(swiglu_limit))
        return torch.nn.functional.silu(gate) * up

    @staticmethod
    def _ceil_to_ue8m0_pytorch(scale: torch.Tensor) -> torch.Tensor:
        return torch.exp2(torch.ceil(torch.log2(scale)))

    @staticmethod
    def _matmul_fp32_no_tf32(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return torch.matmul(lhs.to(torch.float32), rhs.to(torch.float32))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    @classmethod
    def _fp8_quantize_dequantize_1x128_pytorch(
        cls, tensor: torch.Tensor, use_ue8m0_scale: bool, clamp_min_scale: bool
    ) -> torch.Tensor:
        """Reference quantization-dequantiziation for 1x128 activation quantization kernels.

        Args:
            tensor: Input activation tensor with shape [num_tokens, hidden_size].
                It is converted to FP32 before quantization.
            use_ue8m0_scale: Whether to round each 1x128 block scale up to the
                UE8M0 power-of-two scale used by the shared-expert input path.
            clamp_min_scale: Whether to clamp the block amax to a small positive
                value before computing the dequantization scale.

        Returns:
            FP32 tensor with the same shape as ``tensor``. The returned values
            are quantized to FP8 E4M3 per 1x128 block and immediately
            dequantized, so PyTorch matmul can reproduce fused-kernel numerics.
        """
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
            dequant_scale = cls._ceil_to_ue8m0_pytorch(dequant_scale)

        quant_scale = torch.where(
            dequant_scale == 0, torch.ones_like(dequant_scale), 1.0 / dequant_scale
        )
        quantized_tensor = (tensor_blocks * quant_scale).to(torch.float8_e4m3fn)
        dequantized_tensor = quantized_tensor.to(torch.float32) * dequant_scale
        return dequantized_tensor.reshape(rows, padded_cols)[:, :cols]

    @classmethod
    def _routed_gate_up_swiglu_pytorch(
        cls,
        hidden_states: torch.Tensor,
        expert_indices: torch.Tensor,
        routed_w3_w1_weight: torch.Tensor,
        routed_w3_w1_weight_scale: torch.Tensor,
        expert_intermediate_size: int,
        swiglu_limit: Optional[float],
    ) -> torch.Tensor:
        num_tokens, top_k = expert_indices.shape
        routed_swiglu_output = torch.empty(
            (num_tokens, top_k, expert_intermediate_size),
            device=hidden_states.device,
            dtype=torch.float32,
        )
        chunk_size = cls._pytorch_ref_chunk_size()
        hidden_states_fp32 = hidden_states.to(torch.float32)
        for route_idx in range(top_k):
            for token_start in range(0, num_tokens, chunk_size):
                token_end = min(token_start + chunk_size, num_tokens)
                route_expert_indices = expert_indices[token_start:token_end, route_idx].to(
                    torch.int64
                )
                selected_expert_weight = torch.index_select(
                    routed_w3_w1_weight, 0, route_expert_indices
                )
                selected_expert_weight_scale = torch.index_select(
                    routed_w3_w1_weight_scale, 0, route_expert_indices
                )
                selected_expert_weight = cls._dequantize_fp8_128x128_block_weight(
                    selected_expert_weight, selected_expert_weight_scale
                )

                expert_gate_up = cls._matmul_fp32_no_tf32(
                    selected_expert_weight, hidden_states_fp32[token_start:token_end].unsqueeze(-1)
                )
                expert_gate_up = expert_gate_up.squeeze(-1)
                expert_gate_up = cls._fp8_quantize_dequantize_1x128_pytorch(
                    expert_gate_up,
                    use_ue8m0_scale=False,
                    clamp_min_scale=False,
                )
                routed_up, routed_gate = expert_gate_up.split(expert_intermediate_size, dim=-1)
                routed_swiglu = cls._silu_and_mul_pytorch(routed_gate, routed_up, swiglu_limit)
                routed_swiglu = cls._fp8_quantize_dequantize_1x128_pytorch(
                    routed_swiglu,
                    use_ue8m0_scale=False,
                    clamp_min_scale=False,
                )
                routed_swiglu_output[token_start:token_end, route_idx, :] = routed_swiglu
        return routed_swiglu_output

    @classmethod
    def _run_pytorch_ref_mega_kernel(
        cls,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        routing_bias: torch.Tensor,
        shared_gate_up_weight: torch.Tensor,
        shared_gate_up_weight_scale: torch.Tensor,
        shared_gate_up_weight_org: torch.Tensor | None,
        shared_gate_up_weight_scale_org: torch.Tensor | None,
        routed_w3_w1_weight: torch.Tensor,
        routed_w3_w1_weight_scale: torch.Tensor,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        shared_swiglu_limit: Optional[float],
        routed_swiglu_limit: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Reference PyTorch implementation of the mega expert select up gate silu kernel.

        hidden_states:
            - bf16
            - [num_tokens, 6144]
            - input activation
        router_weight:
            - bf16
            - [num_router_experts, hidden_size]
            - from Deepseekv3MoE.gate.weight
        routing_bias:
            - bf16
            - [num_router_experts,]
            - from Deepseekv3MoE.experts.backend._extract_routing_params().routing_bias
        shared_gate_up_weight: gate & up proj (in this order) weight
            - fp8_e4m3
            - [2*expert_immidiate_size, hidden_size]
            - Deepseekv3MoE.shared_experts.gate_up_proj.weight
        shared_gate_up_weight_scale:
            - 4 ue8m0 packed as int32
            - [2*expert_immidiate_size, hidden_size / 128 / 4]
            - from Deepseekv3MoE.shared_experts.gate_up_proj.weight_scale
        shared_gate_up_weight_org:
            - fp8_e4m3
            - [2*expert_immidiate_size, hidden_size]
            - Deepseekv3MoE.shared_experts.gate_up_proj.weight before DeepGEMM post-load resmoothing
        shared_gate_up_weight_scale_org:
            - fp32
            - [ceil(2*expert_immidiate_size / 128), hidden_size / 128]
            - Deepseekv3MoE.shared_experts.gate_up_proj.weight_scale before DeepGEMM post-load resmoothing
        routed_w3_w1_weight: up proj & gate (in this order) weight
            - fp8_e4m3
            - [num_router_experts, 2*expert_immidiate_size, hidden_size]
            - from Deepseekv3MoE.experts.backend.w3_w1_weight
        routed_w3_w1_weight_scale:
            - fp32
            - [num_router_experts, 2*expert_immidiate_size / 128, hidden_size / 128]
            - from Deepseekv3MoE.experts.backend.w3_w1_weight_scaling_factor

        Returns:
        expert_indices
            - int32
            - [num_tokens, top_k]
        expert_weights
            - bf16
            - [num_tokens, top_k]
        shared_swiglu_output
            - bf16
            - [num_tokens, expert_intermediate_size]
        routed_swiglu_output
            - fp32
            - [num_tokens, top_k, expert_intermediate_size]


        Computation Accuracy:
        - Router GEMM: dsv3_router_gemm_op, computes in fp32 and returns as fp32
        - Routing: noaux_tc_op, computes in fp32 and returns as int32 selected expert indices and fp32 expert weights
        - Shared expert gate & up proj: input quantized to FP8 E4M3 with packed UE8M0 scales.
          DeepGEMM uses FP8 operands + block scales, accumulates in fp32, then returns as bf16
        - Shared expert SwiGLU: silu_and_mul_pytorch, computes in fp32, returns as bf16
        - Routed experts:
           - GEMM1: up & gate proj: FP8 activation/weights, FP32 block scales, accumulates in fp32, returns FP8 plus
             FP32 output scale
           - SwiGLU: dequantizes FP8 GEMM1 output to FP32, does SwiGLU in FP32, requantizes to FP8 plus FP32 scale
           - [not in this mega kernel] GEMM2: down proj: FP8 activation/weight, FP32 accumulation, output BF16.
           - [not in this mega kernel] run_finalize: weighted sum of top-k expert outputs, computes in fp32, returns
             as bf16
           - Source: cpp/tensorrt_llm/thop/fp8BlockScaleMoe.cpp:302,
             cpp/tensorrt_llm/kernels/trtllmGenKernels/blockScaleMoe/DevKernel.cu:313
        """
        # fp32, [num_tokens, num_router_experts]
        router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states, router_weight.t(), bias=None, out_dtype=torch.float32
        )
        num_tokens = hidden_states.shape[0]
        num_router_experts = 256
        check_data(
            router_logits,
            "ref_mega_kernels.router_logits",
            torch.float32,
            (num_tokens, num_router_experts),
        )

        expert_weights, expert_indices = torch.ops.trtllm.noaux_tc_op(
            router_logits,
            routing_bias,
            n_group,
            topk_group,
            top_k,
            routed_scaling_factor,
        )
        check_data(
            expert_indices, "ref_mega_kernels.expert_indices", torch.int32, (num_tokens, top_k)
        )
        check_data(
            expert_weights, "ref_mega_kernels.expert_weights", torch.float32, (num_tokens, top_k)
        )
        if cls._pytorch_ref_use_shared_gate_up_weight_org():
            if shared_gate_up_weight_org is None or shared_gate_up_weight_scale_org is None:
                raise RuntimeError(
                    "TRTLLM_DEEPSEEKV3_MEGAMOE_REF_USE_SHARED_GATE_UP_WEIGHT_ORG requires "
                    "shared_experts.gate_up_proj.weight_org and weight_scale_org to be retained"
                )
            # shared_weight: fp32, [2 * expert_intermediate_size, hidden_size]
            shared_weight = cls._dequantize_fp8_128x128_block_weight(
                shared_gate_up_weight_org, shared_gate_up_weight_scale_org
            )
            # Match the routed GEMM1 activation quantization: FP8 E4M3 with FP32 1x128 scales.
            shared_hidden_states = cls._fp8_quantize_dequantize_1x128_pytorch(
                hidden_states,
                use_ue8m0_scale=False,
                clamp_min_scale=True,
            )
        else:
            # shared_weight: fp32, [2 * expert_intermediate_size, hidden_size]
            shared_weight = cls._dequantize_fp8_1x128_packed_ue8m0_weight(
                shared_gate_up_weight, shared_gate_up_weight_scale
            )
            # quantize into float8_e4m3fn with UE8M0 scale, then dequantize back to fp32
            # hidden_states: bf16, [num_tokens, hidden_size]
            # shared_hidden_states: fp32, [num_tokens, hidden_size]
            shared_hidden_states = cls._fp8_quantize_dequantize_1x128_pytorch(
                hidden_states,
                use_ue8m0_scale=True,
                clamp_min_scale=True,
            )
        # shared_gate_up_output: bf16, [num_tokens, 2 * expert_intermediate_size]
        shared_gate_up_output = cls._matmul_fp32_no_tf32(
            shared_hidden_states, shared_weight.t()
        ).to(hidden_states.dtype)
        # shared_gate: bf16, [num_tokens, expert_intermediate_size]
        # shared_up: bf16, [num_tokens, expert_intermediate_size]
        shared_gate, shared_up = shared_gate_up_output.chunk(2, dim=-1)
        # fp32 casted to fb16, [num_tokens, expert_intermediate_size]
        shared_swiglu_output = cls._silu_and_mul_pytorch(
            shared_gate, shared_up, shared_swiglu_limit
        ).to(hidden_states.dtype)

        expert_intermediate_size = shared_swiglu_output.shape[-1]
        routed_hidden_states = cls._fp8_quantize_dequantize_1x128_pytorch(
            hidden_states,
            use_ue8m0_scale=False,
            clamp_min_scale=True,
        )
        routed_swiglu_output = cls._routed_gate_up_swiglu_pytorch(
            routed_hidden_states,
            expert_indices,
            routed_w3_w1_weight,
            routed_w3_w1_weight_scale,
            expert_intermediate_size,
            routed_swiglu_limit,
        )

        return (
            expert_indices,
            expert_weights.to(torch.bfloat16),
            shared_swiglu_output,
            routed_swiglu_output,
        )

    @classmethod
    def _run_wip_mega_kernel(
        cls,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        routing_bias: torch.Tensor,
        shared_gate_up_weight_org: torch.Tensor,
        shared_gate_up_weight_scale_org: torch.Tensor,
        routed_w3_w1_weight: torch.Tensor,
        routed_w3_w1_weight_scale: torch.Tensor,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        shared_swiglu_limit: Optional[float],
        routed_swiglu_limit: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # fp32, [num_tokens, num_router_experts]
        router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states, router_weight.t(), bias=None, out_dtype=torch.float32
        )
        num_tokens = hidden_states.shape[0]
        num_router_experts = 256
        check_data(
            router_logits,
            "wip_mega_kernels.router_logits",
            torch.float32,
            (num_tokens, num_router_experts),
        )

        assert top_k == 8, f"v68 WIP mega kernel only supports top_k=8, got {top_k}"
        assert n_group > 0, f"expected positive n_group, got {n_group}"
        assert topk_group > 0, f"expected positive topk_group, got {topk_group}"

        expert_weights, expert_indices, slot_swiglu_output = (
            torch.ops.trtllm.glm5_expert_select_up_gate_silu(
                router_logits.contiguous(),
                hidden_states.contiguous(),
                routing_bias.contiguous(),
                shared_gate_up_weight_org,
                shared_gate_up_weight_scale_org,
                routed_w3_w1_weight,
                routed_w3_w1_weight_scale,
                routed_scaling_factor,
            )
        )
        check_data(
            expert_indices,
            "wip_mega_kernels.expert_indices",
            torch.int32,
            (num_tokens, top_k),
        )
        check_data(
            expert_weights,
            "wip_mega_kernels.expert_weights",
            torch.float32,
            (num_tokens, top_k),
        )
        check_data(
            slot_swiglu_output,
            "wip_mega_kernels.slot_swiglu_output",
            torch.float16,
            (num_tokens, top_k + 1, -1),
        )

        shared_swiglu_output = slot_swiglu_output[:, 0, :].to(hidden_states.dtype).contiguous()
        routed_swiglu_output = slot_swiglu_output[:, 1:, :].to(torch.float32).contiguous()

        return (
            expert_indices,
            expert_weights.to(torch.bfloat16),
            shared_swiglu_output,
            routed_swiglu_output,
        )

    @classmethod
    def _routed_down_project_pytorch(
        cls,
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
        chunk_size = cls._pytorch_ref_chunk_size()
        for route_idx in range(top_k):
            for token_start in range(0, num_tokens, chunk_size):
                token_end = min(token_start + chunk_size, num_tokens)
                route_expert_indices = expert_indices[token_start:token_end, route_idx].to(
                    torch.int64
                )
                selected_expert_weight = torch.index_select(
                    routed_w2_weight, 0, route_expert_indices
                )
                selected_expert_weight_scale = torch.index_select(
                    routed_w2_weight_scale, 0, route_expert_indices
                )
                selected_expert_weight = cls._dequantize_fp8_128x128_block_weight(
                    selected_expert_weight, selected_expert_weight_scale
                )

                chunk_swiglu_output = routed_swiglu_output[token_start:token_end, route_idx, :]
                expert_down_output = cls._matmul_fp32_no_tf32(
                    selected_expert_weight, chunk_swiglu_output.to(torch.float32).unsqueeze(-1)
                )
                expert_down_output = expert_down_output.squeeze(-1)
                expert_down_output = expert_down_output.to(output_dtype).to(torch.float32)
                expert_down_output *= (
                    expert_weights[token_start:token_end, route_idx].to(torch.float32).unsqueeze(-1)
                )
                routed_output[token_start:token_end] += expert_down_output
        return routed_output.to(output_dtype)

    def _forward_pytorch_ref(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens: list[int] | None,
        num_tokens: int,
        hidden_size: int,
        num_router_experts: int,
        expert_intermediate_size: int,
        block_scale_fp32_hidden_size: int,
        block_scale_int32_hidden_size: int,
        block_scale_fp32_expert_intermediate_size: int,
        gate_up_output_size: int,
    ) -> torch.Tensor:
        """
        Reference PyTorch implementation of forward()
        """
        old_moe = self._old_moe
        shared_experts = old_moe.shared_experts
        shared_gate_up_proj = shared_experts.gate_up_proj
        routed_experts = old_moe.experts

        assert routed_experts.__class__.__name__ == "ConfigurableMoE", (
            f"expected ConfigurableMoE, got {type(routed_experts)}"
        )
        assert routed_experts.comm is None, (
            f"expected no routed-experts communication, got {routed_experts.comm}"
        )
        assert not routed_experts.enable_alltoall
        assert not routed_experts.apply_router_weight_on_input
        assert not routed_experts._using_load_balancer()
        assert not routed_experts.backend._supports_load_balancer()
        assert not routed_experts.routing_method.requires_separated_routing

        routed_all_rank_num_tokens = (
            all_rank_num_tokens if all_rank_num_tokens is not None else [num_tokens]
        )
        assert routed_experts.calculate_num_chunks(routed_all_rank_num_tokens) == 1

        moe_backend = routed_experts.backend
        assert moe_backend.__class__.__name__ == "TRTLLMGenFusedMoE", (
            f"expected TRTLLMGenFusedMoE, got {type(moe_backend)}"
        )
        assert not moe_backend.use_flashinfer
        assert moe_backend.has_deepseek_fp8_block_scales
        assert moe_backend.num_experts == num_router_experts
        assert moe_backend.num_slots == num_router_experts
        assert moe_backend.hidden_size == hidden_size
        assert moe_backend.intermediate_size_per_partition == expert_intermediate_size
        assert num_router_experts == moe_backend.w3_w1_weight.size(0)

        routing_params = moe_backend._extract_routing_params()
        routing_bias = routing_params.routing_bias
        assert routing_bias is not None
        assert routing_params.n_group is not None
        assert routing_params.topk_group is not None
        assert routing_params.routed_scaling_factor is not None

        check_data(
            old_moe.gate.weight,
            "pytorch_ref_mega_router_weight",
            torch.bfloat16,
            (num_router_experts, hidden_size),
        )
        check_data(
            routing_bias,
            "pytorch_ref_mega_router_logit_offset",
            torch.bfloat16,
            (num_router_experts,),
        )
        check_data(
            shared_gate_up_proj.weight,
            "pytorch_ref_mega_shared_gate_up_weight",
            torch.float8_e4m3fn,
            (gate_up_output_size, hidden_size),
        )
        check_data(
            shared_gate_up_proj.weight_scale,
            "pytorch_ref_mega_shared_gate_up_weight_scale",
            torch.int32,
            (gate_up_output_size, block_scale_int32_hidden_size),
        )
        shared_gate_up_weight_org = getattr(shared_gate_up_proj, "weight_org", None)
        shared_gate_up_weight_scale_org = getattr(shared_gate_up_proj, "weight_scale_org", None)
        if self._pytorch_ref_use_shared_gate_up_weight_org():
            if shared_gate_up_weight_org is None or shared_gate_up_weight_scale_org is None:
                raise RuntimeError(
                    "TRTLLM_DEEPSEEKV3_MEGAMOE_REF_USE_SHARED_GATE_UP_WEIGHT_ORG requires "
                    "shared_experts.gate_up_proj.weight_org and weight_scale_org to be retained"
                )
            check_data(
                shared_gate_up_weight_org,
                "pytorch_ref_mega_shared_gate_up_weight_org",
                torch.float8_e4m3fn,
                (gate_up_output_size, hidden_size),
            )
            check_data(
                shared_gate_up_weight_scale_org,
                "pytorch_ref_mega_shared_gate_up_weight_scale_org",
                torch.float32,
                (-1, block_scale_fp32_hidden_size),
            )
        check_data(
            moe_backend.w3_w1_weight,
            "pytorch_ref_mega_routed_w3_w1_weight",
            torch.float8_e4m3fn,
            (num_router_experts, 2 * expert_intermediate_size, hidden_size),
        )
        check_data(
            moe_backend.w3_w1_weight_scaling_factor,
            "pytorch_ref_mega_routed_w3_w1_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                2 * block_scale_fp32_expert_intermediate_size,
                block_scale_fp32_hidden_size,
            ),
        )
        check_data(
            moe_backend.w2_weight,
            "pytorch_ref_mega_routed_w2_weight",
            torch.float8_e4m3fn,
            (num_router_experts, hidden_size, expert_intermediate_size),
        )
        check_data(
            moe_backend.w2_weight_scaling_factor,
            "pytorch_ref_mega_routed_w2_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                block_scale_fp32_hidden_size,
                block_scale_fp32_expert_intermediate_size,
            ),
        )

        (
            expert_indices,
            expert_weights,
            shared_swiglu_output,
            routed_swiglu_output,
        ) = self._run_pytorch_ref_mega_kernel(
            hidden_states,
            old_moe.gate.weight,
            routing_bias,
            shared_gate_up_proj.weight,
            shared_gate_up_proj.weight_scale,
            shared_gate_up_weight_org,
            shared_gate_up_weight_scale_org,
            moe_backend.w3_w1_weight,
            moe_backend.w3_w1_weight_scaling_factor,
            routing_params.top_k,
            routing_params.n_group,
            routing_params.topk_group,
            routing_params.routed_scaling_factor,
            shared_experts.swiglu_limit,
            moe_backend.swiglu_limit_scalar,
        )
        check_data(
            expert_indices,
            "pytorch_ref_mega_expert_indices",
            torch.int32,
            (num_tokens, routing_params.top_k),
        )
        check_data(
            expert_weights,
            "pytorch_ref_mega_expert_weights",
            torch.bfloat16,
            (num_tokens, routing_params.top_k),
        )
        check_data(
            shared_swiglu_output,
            "pytorch_ref_mega_shared_swiglu",
            torch.bfloat16,
            (num_tokens, expert_intermediate_size),
        )
        check_data(
            routed_swiglu_output,
            "pytorch_ref_mega_routed_swiglu",
            torch.float32,
            (num_tokens, routing_params.top_k, expert_intermediate_size),
        )

        routed_output = self._routed_down_project_pytorch(
            routed_swiglu_output,
            expert_indices,
            expert_weights,
            moe_backend.w2_weight,
            moe_backend.w2_weight_scaling_factor,
            hidden_states.dtype,
        )
        check_data(
            routed_output,
            "pytorch_ref_mega_routed_output",
            torch.bfloat16,
            tuple(hidden_states.shape),
        )

        if routed_experts.enable_dwdp:
            routed_experts.dwdp_manager.record_compute_and_prefetch_next(routed_experts.layer_idx)
        routed_experts.repeat_idx = (routed_experts.repeat_idx + 1) % routed_experts.repeat_count

        shared_output = shared_experts.down_proj(
            shared_swiglu_output, all_reduce_params=None, layer_idx=shared_experts.layer_idx
        )
        check_data(
            shared_output,
            "pytorch_ref_mega_shared_output",
            torch.bfloat16,
            tuple(hidden_states.shape),
        )

        assert old_moe.shared_output_scale is None, (
            f"shared_output_scale must be None, got {old_moe.shared_output_scale}"
        )

        output_tensor = None
        if not old_moe.use_dp and old_moe.mapping.tp_size > 1:
            allocate_output_input = shared_output
            allocate_output_buffer_kind = old_moe.allreduce.output_buffer_kind
            allocate_output_tp_group = old_moe.mapping.tp_group
            w, actual_kind = torch.ops.trtllm.allocate_output(
                allocate_output_input, allocate_output_buffer_kind, allocate_output_tp_group
            )
            if actual_kind == int(BufferKind.NCCL_WINDOW):
                output_tensor = w

        assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
        if output_tensor is not None:
            final_hidden_states = torch.add(shared_output, routed_output, out=output_tensor)
        else:
            final_hidden_states = shared_output.add_(routed_output)

        return final_hidden_states

    def _forward_wip_mega_kernel(
        self,
        hidden_states: torch.Tensor,
        all_rank_num_tokens: list[int] | None,
        num_tokens: int,
        hidden_size: int,
        num_router_experts: int,
        expert_intermediate_size: int,
        block_scale_fp32_hidden_size: int,
        block_scale_int32_hidden_size: int,
        block_scale_fp32_expert_intermediate_size: int,
        gate_up_output_size: int,
    ) -> torch.Tensor:
        """
        WIP CUDA mega-kernel implementation of forward().
        """
        old_moe = self._old_moe
        shared_experts = old_moe.shared_experts
        shared_gate_up_proj = shared_experts.gate_up_proj
        routed_experts = old_moe.experts

        assert routed_experts.__class__.__name__ == "ConfigurableMoE", (
            f"expected ConfigurableMoE, got {type(routed_experts)}"
        )
        assert routed_experts.comm is None, (
            f"expected no routed-experts communication, got {routed_experts.comm}"
        )
        assert not routed_experts.enable_alltoall
        assert not routed_experts.apply_router_weight_on_input
        assert not routed_experts._using_load_balancer()
        assert not routed_experts.backend._supports_load_balancer()
        assert not routed_experts.routing_method.requires_separated_routing

        routed_all_rank_num_tokens = (
            all_rank_num_tokens if all_rank_num_tokens is not None else [num_tokens]
        )
        assert routed_experts.calculate_num_chunks(routed_all_rank_num_tokens) == 1

        moe_backend = routed_experts.backend
        assert moe_backend.__class__.__name__ == "TRTLLMGenFusedMoE", (
            f"expected TRTLLMGenFusedMoE, got {type(moe_backend)}"
        )
        assert not moe_backend.use_flashinfer
        assert moe_backend.has_deepseek_fp8_block_scales
        assert moe_backend.num_experts == num_router_experts
        assert moe_backend.num_slots == num_router_experts
        assert moe_backend.hidden_size == hidden_size
        assert moe_backend.intermediate_size_per_partition == expert_intermediate_size
        assert num_router_experts == moe_backend.w3_w1_weight.size(0)

        routing_params = moe_backend._extract_routing_params()
        routing_bias = routing_params.routing_bias
        assert routing_bias is not None
        assert routing_params.n_group is not None
        assert routing_params.topk_group is not None
        assert routing_params.routed_scaling_factor is not None

        check_data(
            old_moe.gate.weight,
            "wip_mega_router_weight",
            torch.bfloat16,
            (num_router_experts, hidden_size),
        )
        check_data(
            routing_bias,
            "wip_mega_router_logit_offset",
            torch.bfloat16,
            (num_router_experts,),
        )
        check_data(
            shared_gate_up_proj.weight,
            "wip_mega_shared_gate_up_weight",
            torch.float8_e4m3fn,
            (gate_up_output_size, hidden_size),
        )
        check_data(
            shared_gate_up_proj.weight_scale,
            "wip_mega_shared_gate_up_weight_scale",
            torch.int32,
            (gate_up_output_size, block_scale_int32_hidden_size),
        )
        shared_gate_up_weight_org = getattr(shared_gate_up_proj, "weight_org", None)
        shared_gate_up_weight_scale_org = getattr(shared_gate_up_proj, "weight_scale_org", None)
        if shared_gate_up_weight_org is None or shared_gate_up_weight_scale_org is None:
            raise RuntimeError(
                "TRTLLM_DEEPSEEKV3_MEGAMOE_MODE=wip_mega_kernel requires "
                "shared_experts.gate_up_proj.weight_org and weight_scale_org to be retained"
            )
        check_data(
            shared_gate_up_weight_org,
            "wip_mega_shared_gate_up_weight_org",
            torch.float8_e4m3fn,
            (gate_up_output_size, hidden_size),
        )
        check_data(
            shared_gate_up_weight_scale_org,
            "wip_mega_shared_gate_up_weight_scale_org",
            torch.float32,
            (2 * block_scale_fp32_expert_intermediate_size, block_scale_fp32_hidden_size),
        )
        check_data(
            moe_backend.w3_w1_weight,
            "wip_mega_routed_w3_w1_weight",
            torch.float8_e4m3fn,
            (num_router_experts, 2 * expert_intermediate_size, hidden_size),
        )
        check_data(
            moe_backend.w3_w1_weight_scaling_factor,
            "wip_mega_routed_w3_w1_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                2 * block_scale_fp32_expert_intermediate_size,
                block_scale_fp32_hidden_size,
            ),
        )
        check_data(
            moe_backend.w2_weight,
            "wip_mega_routed_w2_weight",
            torch.float8_e4m3fn,
            (num_router_experts, hidden_size, expert_intermediate_size),
        )
        check_data(
            moe_backend.w2_weight_scaling_factor,
            "wip_mega_routed_w2_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                block_scale_fp32_hidden_size,
                block_scale_fp32_expert_intermediate_size,
            ),
        )

        (
            expert_indices,
            expert_weights,
            shared_swiglu_output,
            routed_swiglu_output,
        ) = self._run_wip_mega_kernel(
            hidden_states,
            old_moe.gate.weight,
            routing_bias,
            shared_gate_up_weight_org,
            shared_gate_up_weight_scale_org,
            moe_backend.w3_w1_weight,
            moe_backend.w3_w1_weight_scaling_factor,
            routing_params.top_k,
            routing_params.n_group,
            routing_params.topk_group,
            routing_params.routed_scaling_factor,
            shared_experts.swiglu_limit,
            moe_backend.swiglu_limit_scalar,
        )
        check_data(
            expert_indices,
            "wip_mega_expert_indices",
            torch.int32,
            (num_tokens, routing_params.top_k),
        )
        check_data(
            expert_weights,
            "wip_mega_expert_weights",
            torch.bfloat16,
            (num_tokens, routing_params.top_k),
        )
        check_data(
            shared_swiglu_output,
            "wip_mega_shared_swiglu",
            torch.bfloat16,
            (num_tokens, expert_intermediate_size),
        )
        check_data(
            routed_swiglu_output,
            "wip_mega_routed_swiglu",
            torch.float32,
            (num_tokens, routing_params.top_k, expert_intermediate_size),
        )

        routed_output = self._routed_down_project_pytorch(
            routed_swiglu_output,
            expert_indices,
            expert_weights,
            moe_backend.w2_weight,
            moe_backend.w2_weight_scaling_factor,
            hidden_states.dtype,
        )
        check_data(
            routed_output,
            "wip_mega_routed_output",
            torch.bfloat16,
            tuple(hidden_states.shape),
        )

        if routed_experts.enable_dwdp:
            routed_experts.dwdp_manager.record_compute_and_prefetch_next(routed_experts.layer_idx)
        routed_experts.repeat_idx = (routed_experts.repeat_idx + 1) % routed_experts.repeat_count

        shared_output = shared_experts.down_proj(
            shared_swiglu_output, all_reduce_params=None, layer_idx=shared_experts.layer_idx
        )
        check_data(
            shared_output,
            "wip_mega_shared_output",
            torch.bfloat16,
            tuple(hidden_states.shape),
        )

        assert old_moe.shared_output_scale is None, (
            f"shared_output_scale must be None, got {old_moe.shared_output_scale}"
        )

        output_tensor = None
        if not old_moe.use_dp and old_moe.mapping.tp_size > 1:
            allocate_output_input = shared_output
            allocate_output_buffer_kind = old_moe.allreduce.output_buffer_kind
            allocate_output_tp_group = old_moe.mapping.tp_group
            w, actual_kind = torch.ops.trtllm.allocate_output(
                allocate_output_input, allocate_output_buffer_kind, allocate_output_tp_group
            )
            if actual_kind == int(BufferKind.NCCL_WINDOW):
                output_tensor = w

        assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
        if output_tensor is not None:
            final_hidden_states = torch.add(shared_output, routed_output, out=output_tensor)
        else:
            final_hidden_states = shared_output.add_(routed_output)

        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Fp4QuantizedTensor | None = None,
        all_rank_num_tokens: list[int] | None = None,
        final_all_reduce_params: AllReduceParams | None = None,
        do_finalize: bool | None = True,
    ) -> torch.Tensor:
        # if self._old_moe.mapping.rank == 0:
        #     print("Using old MoE forward!")
        old_moe = self._old_moe
        assert do_finalize is True, "Deepseekv3MegaMoE only supports do_finalize=True"

        # Direct Blackwell DeepGEMM replacement for shared_experts.gate_up_proj.
        # ========= Shared experts gate_up_proj  =========
        shared_experts = old_moe.shared_experts
        shared_gate_up_proj = shared_experts.gate_up_proj

        check_data(hidden_states, "hidden_states", torch.bfloat16, (-1, 6144))
        assert shared_gate_up_proj.has_fp8_block_scales
        assert shared_gate_up_proj.bias is None, (
            f"shared_gate_up_proj.bias must be None, got {shared_gate_up_proj.bias}"
        )
        assert is_sm_100f(), "fp8_quantize_1x128_packed_ue8m0 requires SM100-family Blackwell"

        assert isinstance(shared_gate_up_proj.weight, torch.Tensor), (
            f"shared_gate_up_proj.weight must be a tensor, got {type(shared_gate_up_proj.weight)}"
        )
        assert isinstance(shared_gate_up_proj.weight_scale, torch.Tensor), (
            f"shared_gate_up_proj.weight_scale must be a tensor, got {type(shared_gate_up_proj.weight_scale)}"
        )

        num_tokens: int = hidden_states.size(0)
        hidden_size: int = hidden_states.size(1)
        block_scale_fp32_hidden_Size: int = int(triton.cdiv(hidden_size, 128))
        block_scale_int32_hidden_size: int = int(triton.cdiv(hidden_size, 128 * 4))
        num_router_experts: int = 256
        # The weight combines gate and up projection matrices.
        expert_intermediate_size: int = shared_gate_up_proj.weight.size(0) // 2
        block_scale_fp32_expert_intermediate_size: int = int(
            triton.cdiv(expert_intermediate_size, 128)
        )
        gate_up_output_size: int = 2 * expert_intermediate_size

        # Debug reference for the first mega-kernel candidate:
        # router logits + shared/routed gate-up projections + SwiGLU.
        mega_moe_mode = self._mega_moe_mode()
        if mega_moe_mode == self._MEGA_MOE_MODE_PYTORCH_REF:
            return self._forward_pytorch_ref(
                hidden_states=hidden_states,
                all_rank_num_tokens=all_rank_num_tokens,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                num_router_experts=num_router_experts,
                expert_intermediate_size=expert_intermediate_size,
                block_scale_fp32_hidden_size=block_scale_fp32_hidden_Size,
                block_scale_int32_hidden_size=block_scale_int32_hidden_size,
                block_scale_fp32_expert_intermediate_size=(
                    block_scale_fp32_expert_intermediate_size
                ),
                gate_up_output_size=gate_up_output_size,
            )
        elif mega_moe_mode == self._MEGA_MOE_MODE_WIP:
            return self._forward_wip_mega_kernel(
                hidden_states=hidden_states,
                all_rank_num_tokens=all_rank_num_tokens,
                num_tokens=num_tokens,
                hidden_size=hidden_size,
                num_router_experts=num_router_experts,
                expert_intermediate_size=expert_intermediate_size,
                block_scale_fp32_hidden_size=block_scale_fp32_hidden_Size,
                block_scale_int32_hidden_size=block_scale_int32_hidden_size,
                block_scale_fp32_expert_intermediate_size=(
                    block_scale_fp32_expert_intermediate_size
                ),
                gate_up_output_size=gate_up_output_size,
            )

        shared_gate_up_output = torch.empty(
            (num_tokens, gate_up_output_size),
            device=hidden_states.device,
            dtype=torch.bfloat16,
        )

        shared_gate_up_input_fp8, shared_gate_up_input_scale = (
            torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(hidden_states)
        )
        check_data(
            shared_gate_up_input_fp8,
            "shared_gate_up_input_fp8",
            torch.float8_e4m3fn,
            tuple(hidden_states.shape),
        )
        check_data(
            shared_gate_up_input_scale,
            "shared_gate_up_input_scale",
            torch.int32,
            (num_tokens, block_scale_int32_hidden_size),
        )
        check_data(
            shared_gate_up_proj.weight,
            "shared_gate_up_weight",
            torch.float8_e4m3fn,
            (gate_up_output_size, hidden_size),
        )
        check_data(
            shared_gate_up_proj.weight_scale,
            "shared_gate_up_weight_scale",
            torch.int32,
            (gate_up_output_size, block_scale_int32_hidden_size),
        )

        # ========== Shared experts gate_up_proj  =========
        # JIT CUDA template: cpp/include/tensorrt_llm/deep_gemm/fp8_gemm_impl.cuh
        # Launch helper: cpp/include/tensorrt_llm/deep_gemm/fp8_gemm.cuh
        deep_gemm.fp8_gemm_nt(
            # Op wrapper at cpp/tensorrt_llm/thop/fp8Quantize.cpp
            # cpp/tensorrt_llm/kernels/cutlass_kernels/fp8_blockscale_gemm/fp8_blockscale_quant_packed.cu
            (shared_gate_up_input_fp8, shared_gate_up_input_scale),
            # weight: [2 * expert_intermediate_size, hidden_size] fp8_e4m3
            # weight_scale: [2 * expert_intermediate_size, hidden_size / 128 / 4] packed UE8M0 int32
            (shared_gate_up_proj.weight, shared_gate_up_proj.weight_scale),
            shared_gate_up_output,
        )
        # ========= End of Shared experts gate_up_proj  =========

        # ========= Shared experts SwiGLU activation  =========
        assert shared_experts.activation == torch.nn.functional.silu
        assert not shared_experts.down_proj.has_fp8_qdq
        assert not shared_experts.down_proj.has_w4a8_nvfp4_fp8

        # tensorrt_llm/_torch/modules/swiglu.py
        shared_swiglu_output = torch.ops.trtllm.silu_and_mul(
            shared_gate_up_output, swiglu_limit=shared_experts.swiglu_limit
        )
        # ========= End of Shared experts SwiGLU activation  =========

        # ========= Experts router logits  =========
        # compute routed output
        # Torch op wrapper: cpp/tensorrt_llm/thop/dsv3RouterGemmOp.cpp
        # cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
        # Launcher: cpp/tensorrt_llm/kernels/dsv3MinLatencyKernels/dsv3RouterGemm.cu
        check_data(
            old_moe.gate.weight,
            "shared_experts.gate.weight",
            torch.bfloat16,
            (num_router_experts, hidden_size),
        )
        router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states, old_moe.gate.weight.t(), bias=None, out_dtype=torch.float32
        )
        check_data(router_logits, "router_logits", torch.float32, (num_tokens, num_router_experts))
        # ========= End of Experts router logits  =========

        # ========= Routed experts TRTLLMGen FP8 block-scale MoE  =========
        routed_experts = old_moe.experts
        assert routed_experts.__class__.__name__ == "ConfigurableMoE", (
            f"expected ConfigurableMoE, got {type(routed_experts)}"
        )
        assert routed_experts.comm is None, (
            f"expected no routed-experts communication, got {routed_experts.comm}"
        )
        assert not routed_experts.enable_alltoall
        assert not routed_experts.apply_router_weight_on_input
        assert not routed_experts._using_load_balancer()
        assert not routed_experts.backend._supports_load_balancer()
        assert not routed_experts.routing_method.requires_separated_routing

        routed_all_rank_num_tokens = (
            all_rank_num_tokens if all_rank_num_tokens is not None else [num_tokens]
        )
        assert routed_experts.calculate_num_chunks(routed_all_rank_num_tokens) == 1

        moe_backend = routed_experts.backend
        assert moe_backend.__class__.__name__ == "TRTLLMGenFusedMoE", (
            f"expected TRTLLMGenFusedMoE, got {type(moe_backend)}"
        )
        assert not moe_backend.use_flashinfer
        assert moe_backend.has_deepseek_fp8_block_scales
        assert moe_backend.num_experts == num_router_experts
        assert moe_backend.num_slots == num_router_experts
        assert moe_backend.hidden_size == hidden_size
        assert moe_backend.intermediate_size_per_partition == expert_intermediate_size
        assert num_router_experts == moe_backend.w3_w1_weight.size(0)

        routed_input_fp8, routed_input_scale = torch.ops.trtllm.fp8_quantize_1x128(hidden_states)
        check_data(
            routed_input_fp8,
            "routed_moe_input_fp8",
            torch.float8_e4m3fn,
            tuple(hidden_states.shape),
        )
        check_data(
            routed_input_scale,
            "routed_moe_input_scale",
            torch.float32,
            (block_scale_fp32_hidden_Size, num_tokens),
        )

        routing_params = moe_backend._extract_routing_params()
        routing_bias = routing_params.routing_bias
        check_data(routing_bias, "routed_moe_routing_bias", torch.bfloat16, (num_router_experts,))

        # w1: gate proj
        # w2: down proj
        # w3: up proj
        check_data(
            moe_backend.w3_w1_weight,
            "routed_moe_w3_w1_weight",
            torch.float8_e4m3fn,
            (num_router_experts, 2 * expert_intermediate_size, hidden_size),
        )
        check_data(
            moe_backend.w3_w1_weight_scaling_factor,
            "routed_moe_w3_w1_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                2 * block_scale_fp32_expert_intermediate_size,
                block_scale_fp32_hidden_Size,
            ),
        )
        check_data(
            moe_backend.w2_weight,
            "routed_moe_w2_weight",
            torch.float8_e4m3fn,
            (num_router_experts, hidden_size, expert_intermediate_size),
        )
        check_data(
            moe_backend.w2_weight_scaling_factor,
            "routed_moe_w2_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                block_scale_fp32_hidden_Size,
                block_scale_fp32_expert_intermediate_size,
            ),
        )

        from tensorrt_llm._torch.custom_ops.trtllm_gen_custom_ops import (
            AutoTuner,
            FP8BlockScaleMoERunner,
            prepare_dummy_topk_and_hook,
        )

        fp8_moe_runner = FP8BlockScaleMoERunner(
            moe_backend.num_slots,
            routing_params.top_k,
            routing_params.n_group,
            routing_params.topk_group,
            moe_backend.intermediate_size_per_partition,
            moe_backend.slot_start,
            num_router_experts,
            routing_params.routed_scaling_factor,
            moe_backend.routing_method.routing_method_type,
            act_type=0,
            tune_max_num_tokens=moe_backend.max_num_tokens,
            use_dp=moe_backend.use_dp,
            gemm1_clamp_limit_value=moe_backend.swiglu_limit_scalar,
        )
        fp8_moe_tuner = AutoTuner.get()
        routing_method_type = moe_backend.routing_method.routing_method_type
        (
            routing_logits_for_tuner,
            topk_weights_for_tuner,
            topk_ids_for_tuner,
            tuning_config_with_hook,
        ) = prepare_dummy_topk_and_hook(
            routing_method_type=routing_method_type,
            topk_weights=None,
            topk_ids=None,
            hidden_states=routed_input_fp8,
            routing_logits=router_logits,
            base_tuning_config=fp8_moe_runner.tuning_config,
            top_k=routing_params.top_k,
            num_experts=moe_backend.num_slots,
            local_num_experts=num_router_experts,
            n_group=routing_params.n_group,
            topk_group=routing_params.topk_group,
            routed_scaling_factor=routing_params.routed_scaling_factor,
            hidden_states_index=2,
            local_expert_offset=moe_backend.slot_start,
            use_dp=moe_backend.use_dp,
        )
        input_tensors_for_tuner = [
            routing_logits_for_tuner,
            routing_bias,
            routed_input_fp8,
            routed_input_scale,
            moe_backend.w3_w1_weight,
            moe_backend.w3_w1_weight_scaling_factor,
            moe_backend.w2_weight,
            moe_backend.w2_weight_scaling_factor,
            topk_weights_for_tuner,
            topk_ids_for_tuner,
        ]
        fp8_moe_runner, best_tactic = fp8_moe_tuner.choose_one(
            "trtllm::fp8_block_scale_moe_runner",
            [fp8_moe_runner],
            tuning_config_with_hook,
            input_tensors_for_tuner,
        )
        tactic = [-1, -1] if best_tactic == -1 else best_tactic
        fp8_moe_cpp_runner = fp8_moe_runner.get_runner()
        resolved_tactic = fp8_moe_cpp_runner.resolve_tactic(
            routing_params.top_k,
            hidden_size,
            moe_backend.intermediate_size_per_partition,
            num_router_experts,
            num_tokens,
            tactic,
        )

        def ceil_div(x: int, y: int) -> int:
            return (x + y - 1) // y

        def get_max_num_ctas_in_batch_dim(
            num_tokens: int, top_k: int, num_experts: int, tile_tokens_dim: int
        ) -> int:
            num_remaining_tokens = num_tokens * top_k
            max_num_ctas = min(num_experts, num_remaining_tokens)
            num_remaining_tokens -= max_num_ctas
            if num_remaining_tokens > 0:
                max_num_ctas += num_remaining_tokens // tile_tokens_dim
            return max_num_ctas

        def maybe_get_min_token_count(
            num_padded_tokens: int, hidden_dim: int, dtype_size_bits: int
        ) -> int:
            min_num_tokens_required = ceil_div(128 * 1024 * 8, hidden_dim * dtype_size_bits)
            return max(num_padded_tokens, min_num_tokens_required)

        tile_tokens_dim = int(resolved_tactic[0])
        max_num_ctas = get_max_num_ctas_in_batch_dim(
            num_tokens, routing_params.top_k, moe_backend.num_slots, tile_tokens_dim
        )
        max_num_padded_tokens = max_num_ctas * tile_tokens_dim
        max_num_padded_tokens_gemm1 = maybe_get_min_token_count(
            max_num_padded_tokens, 2 * moe_backend.intermediate_size_per_partition, 8
        )
        max_num_padded_tokens_gemm2 = maybe_get_min_token_count(
            max_num_padded_tokens, hidden_size, 16
        )

        # Routed experts routing.
        routing_outputs = fp8_moe_cpp_runner.run_routing(
            router_logits,
            routing_bias,
            routed_input_fp8,
            moe_backend.num_slots,
            routing_params.top_k,
            routing_params.n_group,
            routing_params.topk_group,
            moe_backend.intermediate_size_per_partition,
            moe_backend.slot_start,
            num_router_experts,
            routing_params.routed_scaling_factor,
            routing_method_type,
            resolved_tactic,
            None,
            None,
        )
        (
            routing_expert_indexes,
            expert_count_histogram,
            total_num_padded_tokens,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            expert_weights,
            num_tokens_per_expert,
            cta_idx_xy_to_batch_idx,
            cta_idx_xy_to_mn_limit,
            num_non_exiting_ctas,
        ) = routing_outputs
        check_data(
            routing_expert_indexes,
            "routed_moe_routing_expert_indexes",
            torch.int32,
            (num_tokens, routing_params.top_k),
        )
        check_data(
            expert_count_histogram,
            "routed_moe_expert_count_histogram",
            torch.int32,
            (max(num_router_experts * 2, 256 * 2),),
        )
        check_data(total_num_padded_tokens, "routed_moe_total_num_padded_tokens", torch.int32, ())
        check_data(
            expanded_idx_to_permuted_idx,
            "routed_moe_expanded_idx_to_permuted_idx",
            torch.int32,
            (num_tokens * routing_params.top_k,),
        )
        check_data(
            permuted_idx_to_token_idx,
            "routed_moe_permuted_idx_to_token_idx",
            torch.int32,
            (max_num_padded_tokens,),
        )
        check_data(
            expert_weights,
            "routed_moe_expert_weights",
            torch.bfloat16,
            (num_tokens, routing_params.top_k),
        )
        check_data(
            num_tokens_per_expert,
            "routed_moe_num_tokens_per_expert",
            torch.int32,
            (num_router_experts,),
        )
        check_data(
            cta_idx_xy_to_batch_idx,
            "routed_moe_cta_idx_xy_to_batch_idx",
            torch.int32,
            (max_num_ctas,),
        )
        check_data(
            cta_idx_xy_to_mn_limit,
            "routed_moe_cta_idx_xy_to_mn_limit",
            torch.int32,
            (max_num_ctas,),
        )
        check_data(num_non_exiting_ctas, "routed_moe_num_non_exiting_ctas", torch.int32, ())

        # Routed experts permute + GEMM1.
        gemm1_output, gemm1_output_scale = fp8_moe_cpp_runner.run_permute_gemm1(
            routed_input_fp8,
            routed_input_scale,
            moe_backend.w3_w1_weight,
            moe_backend.w3_w1_weight_scaling_factor,
            expert_weights,
            permuted_idx_to_token_idx,
            total_num_padded_tokens,
            cta_idx_xy_to_batch_idx,
            cta_idx_xy_to_mn_limit,
            num_non_exiting_ctas,
            moe_backend.num_slots,
            routing_params.top_k,
            moe_backend.intermediate_size_per_partition,
            num_router_experts,
            resolved_tactic,
        )
        check_data(
            gemm1_output,
            "routed_moe_gemm1_output",
            torch.float8_e4m3fn,
            (max_num_padded_tokens_gemm1, 2 * moe_backend.intermediate_size_per_partition),
        )
        check_data(
            gemm1_output_scale,
            "routed_moe_gemm1_output_scale",
            torch.float32,
            (2 * block_scale_fp32_expert_intermediate_size, max_num_padded_tokens_gemm1),
        )

        # Routed experts SwiGLU activation.
        activation_output, activation_output_scale = fp8_moe_cpp_runner.run_activation(
            gemm1_output,
            gemm1_output_scale,
            expanded_idx_to_permuted_idx,
            total_num_padded_tokens,
            routing_params.top_k,
            num_tokens,
            moe_backend.intermediate_size_per_partition,
            resolved_tactic,
            moe_backend.swiglu_limit_scalar,
        )
        check_data(
            activation_output,
            "routed_moe_activation_output",
            torch.float8_e4m3fn,
            (max_num_padded_tokens_gemm1, moe_backend.intermediate_size_per_partition),
        )
        check_data(
            activation_output_scale,
            "routed_moe_activation_output_scale",
            torch.float32,
            (block_scale_fp32_expert_intermediate_size, max_num_padded_tokens_gemm1),
        )

        # Routed experts GEMM2.
        gemm2_output = fp8_moe_cpp_runner.run_gemm2(
            activation_output,
            activation_output_scale,
            moe_backend.w2_weight,
            moe_backend.w2_weight_scaling_factor,
            total_num_padded_tokens,
            cta_idx_xy_to_batch_idx,
            cta_idx_xy_to_mn_limit,
            num_non_exiting_ctas,
            moe_backend.num_slots,
            routing_params.top_k,
            num_tokens,
            hidden_size,
            moe_backend.intermediate_size_per_partition,
            num_router_experts,
            resolved_tactic,
        )
        check_data(
            gemm2_output,
            "routed_moe_gemm2_output",
            torch.bfloat16,
            (max_num_padded_tokens_gemm2, hidden_size),
        )

        # Routed experts finalize.
        routed_output = fp8_moe_cpp_runner.run_finalize(
            gemm2_output,
            expert_weights,
            expanded_idx_to_permuted_idx,
            total_num_padded_tokens,
            moe_backend.num_slots,
            routing_params.top_k,
            num_tokens,
            hidden_size,
            resolved_tactic,
            None,
        )
        check_data(routed_output, "routed_moe_output", torch.bfloat16, tuple(hidden_states.shape))

        if routed_experts.enable_dwdp:
            routed_experts.dwdp_manager.record_compute_and_prefetch_next(routed_experts.layer_idx)
        routed_experts.repeat_idx = (routed_experts.repeat_idx + 1) % routed_experts.repeat_count
        # ========= End of Routed experts TRTLLMGen FP8 block-scale MoE  =========

        # ========== Shared experts down_proj  =========
        # Kernel input to shared_experts.down_proj:
        # FP8-block scale path uses Linear -> FP8BlockScalesLinearMethod.apply.
        shared_output = shared_experts.down_proj(
            shared_swiglu_output, all_reduce_params=None, layer_idx=shared_experts.layer_idx
        )

        assert old_moe.shared_output_scale is None, (
            f"shared_output_scale must be None, got {old_moe.shared_output_scale}"
        )
        # if old_moe.shared_output_scale is not None:
        #     shared_output *= old_moe.shared_output_scale
        # ========= End of Shared experts down_proj  =========

        output_tensor = None
        if not old_moe.use_dp and old_moe.mapping.tp_size > 1:
            allocate_output_input = shared_output
            allocate_output_buffer_kind = old_moe.allreduce.output_buffer_kind
            allocate_output_tp_group = old_moe.mapping.tp_group
            w, actual_kind = torch.ops.trtllm.allocate_output(
                allocate_output_input, allocate_output_buffer_kind, allocate_output_tp_group
            )
            if actual_kind == int(BufferKind.NCCL_WINDOW):
                output_tensor = w

        assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
        if output_tensor is not None:
            final_hidden_states = torch.add(shared_output, routed_output, out=output_tensor)
        else:
            final_hidden_states = shared_output.add_(routed_output)

        return final_hidden_states
