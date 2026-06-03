import os
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import triton
from torch import nn

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..distributed import AllReduceFusionOp, AllReduceParams
from ..distributed.ops import get_allreduce_workspace
from ..model_config import ModelConfig
from ..modules.fused_moe import MoE
from ..modules.linear import TensorParallelMode, load_weight_shard
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_deepseekv3_moe import Deepseekv3MoE

_FUSED_MOE_PREPACK_FUSED_EXPERT_DOWN_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_PREPACK_FUSED_EXPERT_DOWN"
_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV = "TRTLLM_DEEPSEEKV3_FUSED_EXPERT_DOWN_FINALIZE_MODE"
FUSED_MOE_MODE_BASELINE = "baseline"
FUSED_MOE_MODE_WIP = "wip"
FUSED_EXPERT_DOWN_FINALIZE_MODE_LOCAL = "local"
FUSED_EXPERT_DOWN_FINALIZE_MODE_ALLREDUCE_RESIDUAL_RMS_NORM = "allreduce_residual_rms_norm"
_FUSED_EXPERT_UP_HIDDEN_SIZE = 6144
_FUSED_EXPERT_UP_CTA_OUT_ROWS = 64
_FUSED_EXPERT_UP_M_TILES_PER_CTA = 4
_FUSED_EXPERT_UP_ROW_HALVES_PER_M_TILE = 2
_FUSED_EXPERT_UP_ROWS_PER_HALF = 8
_FUSED_EXPERT_UP_NUM_K_ITER = 8
_FUSED_EXPERT_UP_K_THIRDS_PER_ITER = 6
_FUSED_EXPERT_UP_K_SUBS_PER_THIRD = 4
_FUSED_EXPERT_UP_COL_HALVES_PER_K_SUB = 2
_FUSED_EXPERT_UP_COL_QUADS_PER_HALF = 4
_FUSED_EXPERT_UP_BYTES_PER_COL_QUAD = 4
_FUSED_EXPERT_UP_TILE_BYTES = 49152
_FUSED_EXPERT_DOWN_HIDDEN_SIZE = 6144
_FUSED_EXPERT_DOWN_NUM_CTAS = 148
_FUSED_EXPERT_DOWN_ROWS_PER_CTA = 42
_FUSED_EXPERT_DOWN_MMA_M = 16
_FUSED_EXPERT_DOWN_ROW_TILES_PER_CTA = 3
_FUSED_EXPERT_DOWN_PACKED_ROW_TILES = (
    _FUSED_EXPERT_DOWN_NUM_CTAS * _FUSED_EXPERT_DOWN_ROW_TILES_PER_CTA
)
_FUSED_EXPERT_DOWN_BLOCK_K = 128
_FUSED_EXPERT_DOWN_TILE_BYTES = 2048


def _fused_moe_prepack_fused_expert_down_enabled() -> bool:
    value = os.environ.get(_FUSED_MOE_PREPACK_FUSED_EXPERT_DOWN_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


def get_fused_moe_mode() -> str:
    mode = os.environ.get("TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE", "").strip().lower()
    if not mode:
        return FUSED_MOE_MODE_BASELINE

    mode_aliases = {
        "baseline": FUSED_MOE_MODE_BASELINE,
        "trtllm": FUSED_MOE_MODE_BASELINE,
        "trtllm_baseline": FUSED_MOE_MODE_BASELINE,
        "wip": FUSED_MOE_MODE_WIP,
        "fused_moe": FUSED_MOE_MODE_WIP,
        "fused_moe_kernel": FUSED_MOE_MODE_WIP,
        "wip_fused_moe": FUSED_MOE_MODE_WIP,
        "wip_fused_moe_kernel": FUSED_MOE_MODE_WIP,
    }
    if mode not in mode_aliases:
        allowed_modes = ", ".join((FUSED_MOE_MODE_BASELINE, FUSED_MOE_MODE_WIP))
        raise ValueError(
            f"Unsupported TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE={mode!r}; "
            f"expected one of: {allowed_modes}"
        )
    return mode_aliases[mode]


def get_fused_expert_down_finalize_mode() -> str:
    mode = (
        os.environ.get(
            _FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV,
            FUSED_EXPERT_DOWN_FINALIZE_MODE_ALLREDUCE_RESIDUAL_RMS_NORM,
        )
        .strip()
        .lower()
    )
    mode_aliases = {
        "local": FUSED_EXPERT_DOWN_FINALIZE_MODE_LOCAL,
        "none": FUSED_EXPERT_DOWN_FINALIZE_MODE_LOCAL,
        "debug": FUSED_EXPERT_DOWN_FINALIZE_MODE_LOCAL,
        "allreduce_residual_rms_norm": (
            FUSED_EXPERT_DOWN_FINALIZE_MODE_ALLREDUCE_RESIDUAL_RMS_NORM
        ),
        "trtllm_allreduce_residual_rms_norm": (
            FUSED_EXPERT_DOWN_FINALIZE_MODE_ALLREDUCE_RESIDUAL_RMS_NORM
        ),
    }
    if mode not in mode_aliases:
        allowed_modes = ", ".join(
            (
                FUSED_EXPERT_DOWN_FINALIZE_MODE_LOCAL,
                FUSED_EXPERT_DOWN_FINALIZE_MODE_ALLREDUCE_RESIDUAL_RMS_NORM,
            )
        )
        raise ValueError(
            f"Unsupported {_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV}={mode!r}; "
            f"expected one of: {allowed_modes}"
        )
    return mode_aliases[mode]


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


def _prepack_fused_expert_up_weight_side_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [..., I, H].
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"fused expert up prepack expects float8_e4m3fn weights, got {weight.dtype}"
        )
    if weight.shape[-1] != _FUSED_EXPERT_UP_HIDDEN_SIZE:
        raise ValueError(
            f"fused expert up prepack expects hidden size {_FUSED_EXPERT_UP_HIDDEN_SIZE}, got {weight.shape[-1]}"
        )
    inter_per_tp = weight.shape[-2]
    if inter_per_tp % _FUSED_EXPERT_UP_CTA_OUT_ROWS != 0:
        raise ValueError(
            f"fused expert up prepack expects I to be divisible by {_FUSED_EXPERT_UP_CTA_OUT_ROWS}, got {inter_per_tp}"
        )

    prefix_shape = tuple(weight.shape[:-2])
    num_prefix_dims = len(prefix_shape)
    sub_rows = inter_per_tp // _FUSED_EXPERT_UP_CTA_OUT_ROWS

    reshaped = weight.contiguous().reshape(
        *prefix_shape,
        sub_rows,
        _FUSED_EXPERT_UP_M_TILES_PER_CTA,
        _FUSED_EXPERT_UP_ROW_HALVES_PER_M_TILE,
        _FUSED_EXPERT_UP_ROWS_PER_HALF,
        _FUSED_EXPERT_UP_NUM_K_ITER,
        _FUSED_EXPERT_UP_K_THIRDS_PER_ITER,
        _FUSED_EXPERT_UP_K_SUBS_PER_THIRD,
        _FUSED_EXPERT_UP_COL_HALVES_PER_K_SUB,
        _FUSED_EXPERT_UP_COL_QUADS_PER_HALF,
        _FUSED_EXPERT_UP_BYTES_PER_COL_QUAD,
    )
    # Reorder the raw row-major [64, 768] tile into the exact 48 KiB K-major
    # lane slab consumed by compute_mma_kiter_fused_expert_up().
    permute_order = (
        *range(num_prefix_dims),
        num_prefix_dims,
        num_prefix_dims + 4,
        num_prefix_dims + 5,
        num_prefix_dims + 1,
        num_prefix_dims + 6,
        num_prefix_dims + 3,
        num_prefix_dims + 8,
        num_prefix_dims + 7,
        num_prefix_dims + 2,
        num_prefix_dims + 9,
    )
    return (
        reshaped.permute(permute_order)
        .contiguous()
        .reshape(*prefix_shape, sub_rows, _FUSED_EXPERT_UP_NUM_K_ITER, _FUSED_EXPERT_UP_TILE_BYTES)
    )


def prepack_fused_expert_up_shared_gate_up_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [2 * I, H], stored as [gate, up].
    gate_weight, up_weight = weight.chunk(2, dim=0)
    gate_packed = _prepack_fused_expert_up_weight_side_pytorch(gate_weight)
    up_packed = _prepack_fused_expert_up_weight_side_pytorch(up_weight)
    packed = torch.empty((2, *gate_packed.shape), device=weight.device, dtype=weight.dtype)
    # packed: torch.float8_e4m3fn, [2, I / 64, 8, 49152], stored as [gate, up].
    packed[0].copy_(gate_packed)
    packed[1].copy_(up_packed)
    return packed


def prepack_fused_expert_up_routed_w3_w1_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [E, 2 * I, H], stored as [up, gate].
    up_weight, gate_weight = weight.chunk(2, dim=1)
    gate_packed = _prepack_fused_expert_up_weight_side_pytorch(gate_weight)
    up_packed = _prepack_fused_expert_up_weight_side_pytorch(up_weight)
    packed = torch.empty(
        (weight.shape[0], 2, *gate_packed.shape[1:]),
        device=weight.device,
        dtype=weight.dtype,
    )
    # packed: torch.float8_e4m3fn, [E, 2, I / 64, 8, 49152], stored as [gate, up].
    packed[:, 0].copy_(gate_packed)
    packed[:, 1].copy_(up_packed)
    return packed


def _should_use_cute_fused_expert_up_weight_pack(weight: torch.Tensor) -> bool:
    return weight.is_cuda and torch.cuda.is_available()


def prepack_fused_expert_up_shared_gate_up_weight(weight: torch.Tensor) -> torch.Tensor:
    if _should_use_cute_fused_expert_up_weight_pack(weight):
        try:
            from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
                deepseekv3_fused_moe_weight_pack,
            )

            return deepseekv3_fused_moe_weight_pack.pack_fused_expert_up_shared_gate_up_weight(
                weight
            )
        except ImportError:
            pass
    return prepack_fused_expert_up_shared_gate_up_weight_pytorch(weight)


def prepack_fused_expert_up_routed_w3_w1_weight(weight: torch.Tensor) -> torch.Tensor:
    if _should_use_cute_fused_expert_up_weight_pack(weight):
        try:
            from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
                deepseekv3_fused_moe_weight_pack,
            )

            return deepseekv3_fused_moe_weight_pack.pack_fused_expert_up_routed_w3_w1_weight(weight)
        except ImportError:
            pass
    return prepack_fused_expert_up_routed_w3_w1_weight_pytorch(weight)


def _fused_expert_down_packed_indices(
    device: torch.device, k_local: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_local % _FUSED_EXPERT_DOWN_BLOCK_K != 0:
        raise ValueError(
            f"fused expert down prepack expects K_local divisible by {_FUSED_EXPERT_DOWN_BLOCK_K}, got {k_local}"
        )

    tile_idx = torch.arange(_FUSED_EXPERT_DOWN_PACKED_ROW_TILES, device=device, dtype=torch.int64)
    row_base = (tile_idx // _FUSED_EXPERT_DOWN_ROW_TILES_PER_CTA) * _FUSED_EXPERT_DOWN_ROWS_PER_CTA
    row_base += (tile_idx % _FUSED_EXPERT_DOWN_ROW_TILES_PER_CTA) * _FUSED_EXPERT_DOWN_MMA_M

    elem = torch.arange(_FUSED_EXPERT_DOWN_TILE_BYTES, device=device, dtype=torch.int64)
    row = elem // _FUSED_EXPERT_DOWN_BLOCK_K
    phys_col = elem - row * _FUSED_EXPERT_DOWN_BLOCK_K
    chunk_phys = phys_col // 16
    byte = phys_col - chunk_phys * 16
    logical_col_in_kb = ((chunk_phys ^ (row & 7)) * 16) + byte

    src_rows = row_base[:, None] + row[None, :]
    src_rows = torch.where(
        src_rows < _FUSED_EXPERT_DOWN_HIDDEN_SIZE,
        src_rows,
        torch.full_like(src_rows, _FUSED_EXPERT_DOWN_HIDDEN_SIZE),
    )
    kb = torch.arange(k_local // _FUSED_EXPERT_DOWN_BLOCK_K, device=device, dtype=torch.int64)
    src_cols = kb[:, None] * _FUSED_EXPERT_DOWN_BLOCK_K + logical_col_in_kb[None, :]
    return src_rows, src_cols


def prepack_fused_expert_down_shared_down_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [H, I].
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"fused expert down prepack expects float8_e4m3fn weights, got {weight.dtype}"
        )
    if weight.ndim != 2 or weight.shape[0] != _FUSED_EXPERT_DOWN_HIDDEN_SIZE:
        raise ValueError(f"shared fused expert down prepack expects [H, I], got {weight.shape}")

    weight = weight.contiguous()
    src_rows, src_cols = _fused_expert_down_packed_indices(weight.device, weight.shape[1])
    padded = torch.empty(
        (_FUSED_EXPERT_DOWN_HIDDEN_SIZE + _FUSED_EXPERT_DOWN_MMA_M, weight.shape[1]),
        device=weight.device,
        dtype=weight.dtype,
    )
    padded[:_FUSED_EXPERT_DOWN_HIDDEN_SIZE].copy_(weight)
    padded[_FUSED_EXPERT_DOWN_HIDDEN_SIZE:].zero_()
    # packed: torch.float8_e4m3fn, [444, I / 128, 2048].
    return padded[src_rows[:, None, :], src_cols[None, :, :]].contiguous()


def prepack_fused_expert_down_routed_w2_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [E, H, I].
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(
            f"fused expert down prepack expects float8_e4m3fn weights, got {weight.dtype}"
        )
    if weight.ndim != 3 or weight.shape[1] != _FUSED_EXPERT_DOWN_HIDDEN_SIZE:
        raise ValueError(f"routed fused expert down prepack expects [E, H, I], got {weight.shape}")

    weight = weight.contiguous()
    src_rows, src_cols = _fused_expert_down_packed_indices(weight.device, weight.shape[2])
    padded = torch.empty(
        (
            weight.shape[0],
            _FUSED_EXPERT_DOWN_HIDDEN_SIZE + _FUSED_EXPERT_DOWN_MMA_M,
            weight.shape[2],
        ),
        device=weight.device,
        dtype=weight.dtype,
    )
    padded[:, :_FUSED_EXPERT_DOWN_HIDDEN_SIZE].copy_(weight)
    padded[:, _FUSED_EXPERT_DOWN_HIDDEN_SIZE:].zero_()
    # packed: torch.float8_e4m3fn, [E, 444, I / 128, 2048].
    return padded[:, src_rows[:, None, :], src_cols[None, :, :]].contiguous()


def prepack_fused_expert_down_shared_down_weight(weight: torch.Tensor) -> torch.Tensor:
    return prepack_fused_expert_down_shared_down_weight_pytorch(weight)


def prepack_fused_expert_down_routed_w2_weight(weight: torch.Tensor) -> torch.Tensor:
    return prepack_fused_expert_down_routed_w2_weight_pytorch(weight)


class Deepseekv3FusedMoE(nn.Module):
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
        del aux_stream_dict, dtype, override_quant_config
        if get_fused_moe_mode() == FUSED_MOE_MODE_BASELINE:
            raise RuntimeError(
                "TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE=baseline is handled by "
                "constructing Deepseekv3MoE directly."
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.model_config = model_config
        self.mapping = model_config.mapping
        self.use_dp = model_config.mapping.enable_attention_dp
        if (
            self.use_dp
            or self.mapping.cp_size > 1
            or self.mapping.pp_size > 1
            or self.mapping.moe_ep_size > 1
            or self.mapping.attn_cp_size > 1
            or self.mapping.moe_cluster_size > 1
            or self.mapping.moe_tp_size != self.mapping.tp_size
            or self.mapping.attn_tp_size != self.mapping.tp_size
        ):
            raise RuntimeError(
                "Deepseekv3FusedMoE currently supports single-node TP-only "
                "execution with matching TP, attention TP, and MoE TP sizes, got "
                f"{self.mapping.to_dict()}"
            )
        self.layer_idx = layer_idx
        self._wip_weights_loaded = False

        from ..distributed import AllReduce

        config = model_config.pretrained_config
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.swiglu_limit = Deepseekv3MoE._create_swiglu_limit_tensor(model_config, num_experts)
        self.swiglu_limit_scalar = None
        self.experts_stub = SimpleNamespace(has_nvfp4=False)
        self.moe_tp_size = model_config.mapping.moe_tp_size
        self.moe_tp_rank = model_config.mapping.moe_tp_rank
        self.moe_ep_size = model_config.mapping.moe_ep_size
        self.moe_ep_rank = model_config.mapping.moe_ep_rank
        self.routed_intermediate_size_per_partition = intermediate_size // self.moe_tp_size

        shared_quant_config = Deepseekv3MoE._get_shared_experts_quant_config(
            model_config, layer_idx
        )
        block_size = 1
        if (
            shared_quant_config is not None
            and shared_quant_config.quant_algo is not None
            and shared_quant_config.group_size is not None
        ):
            block_size = shared_quant_config.group_size
        self.shared_tp_size, self.shared_output_scale = (
            Deepseekv3MoE._compute_shared_expert_tp_size(
                self, shared_expert_intermediate_size, block_size
            )
        )
        self.shared_tp_rank = self.mapping.rank % self.shared_tp_size
        self.shared_intermediate_size_per_partition = (
            shared_expert_intermediate_size // self.shared_tp_size
        )

        self.allreduce = None
        if not self.use_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping, strategy=model_config.allreduce_strategy
            )
        self.register_buffer("_fused_expert_down_allreduce_workspace", None, persistent=False)
        self.register_buffer("_fused_expert_down_local_output", None, persistent=False)
        self.register_buffer("_fused_expert_down_residual_output", None, persistent=False)
        self.register_buffer("_fused_expert_down_hidden_output", None, persistent=False)
        self.register_buffer("_fused_expert_down_rms_sums", None, persistent=False)

    @property
    def experts(self) -> MoE:
        return self.experts_stub

    _FUSED_MOE_MODE_BASELINE = FUSED_MOE_MODE_BASELINE
    _FUSED_MOE_MODE_WIP = FUSED_MOE_MODE_WIP
    _WIP_DOWN_PROJECT_MAX_NUM_TOKENS = 4

    @staticmethod
    def _set_nonpersistent_buffer(
        module: nn.Module, name: str, tensor: torch.Tensor | None
    ) -> None:
        if name in module._buffers:
            module._buffers[name] = tensor
        elif hasattr(module, name):
            if getattr(module, name) is None:
                delattr(module, name)
                module.register_buffer(name, tensor, persistent=False)
            else:
                setattr(module, name, tensor)
        else:
            module.register_buffer(name, tensor, persistent=False)

    def _ensure_fused_expert_down_finalize_buffers(self, hidden_states: torch.Tensor) -> None:
        max_tokens = self._WIP_DOWN_PROJECT_MAX_NUM_TOKENS
        hidden_shape = (max_tokens, hidden_states.shape[-1])
        buffer_specs = (
            ("_fused_expert_down_local_output", hidden_shape, hidden_states.dtype),
            ("_fused_expert_down_residual_output", hidden_shape, hidden_states.dtype),
            ("_fused_expert_down_hidden_output", hidden_shape, hidden_states.dtype),
            ("_fused_expert_down_rms_sums", (max_tokens,), torch.float32),
        )
        for name, shape, dtype in buffer_specs:
            buffer = getattr(self, name)
            if (
                buffer is None
                or tuple(buffer.shape) != tuple(shape)
                or buffer.device != hidden_states.device
                or buffer.dtype != dtype
            ):
                self._set_nonpersistent_buffer(
                    self,
                    name,
                    torch.empty(shape, device=hidden_states.device, dtype=dtype),
                )

        if self.mapping.tp_size > 1 and self._fused_expert_down_allreduce_workspace is None:
            self._set_nonpersistent_buffer(
                self,
                "_fused_expert_down_allreduce_workspace",
                get_allreduce_workspace(self.mapping),
            )

    def _can_finalize_with_fused_expert_down(
        self,
        final_all_reduce_params: AllReduceParams | None,
        num_tokens: int,
    ) -> bool:
        return (
            final_all_reduce_params is not None
            and final_all_reduce_params.enable_allreduce
            and final_all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
            and final_all_reduce_params.residual is not None
            and final_all_reduce_params.norm_weight is not None
            and self.mapping.tp_size > 1
            and self.allreduce is not None
            and num_tokens <= self._WIP_DOWN_PROJECT_MAX_NUM_TOKENS
        )

    def _run_dsv3_fused_expert_down_finalize(
        self,
        slot_swiglu_output: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        routed_w2_weight: torch.Tensor,
        routed_w2_weight_scale: torch.Tensor,
        shared_down_weight: torch.Tensor,
        shared_down_weight_scale_org: torch.Tensor,
        final_all_reduce_params: AllReduceParams,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = slot_swiglu_output.size(0)
        self._ensure_fused_expert_down_finalize_buffers(final_all_reduce_params.residual)
        local_output = self._fused_expert_down_local_output[:num_tokens]
        residual_out = self._fused_expert_down_residual_output[:num_tokens]
        hidden_out = self._fused_expert_down_hidden_output[:num_tokens]
        rms_sums = self._fused_expert_down_rms_sums[:num_tokens]
        workspace = self._fused_expert_down_allreduce_workspace
        if workspace is None:
            raise RuntimeError("Deepseekv3FusedMoE fused finalize workspace is not initialized")

        hidden_out = torch.ops.trtllm.dsv3_fused_expert_down_ar_residual_rms_norm(
            slot_swiglu_output.contiguous(),
            expert_indices.contiguous(),
            expert_weights.contiguous(),
            routed_w2_weight,
            routed_w2_weight_scale,
            shared_down_weight,
            shared_down_weight_scale_org,
            final_all_reduce_params.residual,
            final_all_reduce_params.norm_weight,
            workspace,
            self.mapping.tp_rank,
            self.mapping.tp_size,
            final_all_reduce_params.eps,
            local_output,
            residual_out,
            hidden_out,
            rms_sums,
        )
        return hidden_out, residual_out

    @staticmethod
    def _checkpoint_tensor(weights, key: str) -> torch.Tensor:
        if key not in weights:
            raise KeyError(f"Missing Deepseekv3FusedMoE WIP weight: {key}")
        tensor = weights[key]
        return tensor[:] if hasattr(tensor, "__getitem__") else tensor

    @staticmethod
    def _normalize_fp8_block_scale(scale: torch.Tensor) -> torch.Tensor:
        if scale.dim() == 4:
            scale = scale.squeeze(1).squeeze(-1)
        return scale

    @classmethod
    def _load_checkpoint_shard(
        cls,
        weights,
        key: str,
        tensor_parallel_size: int,
        tensor_parallel_rank: int,
        tensor_parallel_mode: TensorParallelMode,
        *,
        normalize_scale: bool = False,
    ) -> torch.Tensor:
        tensor = cls._checkpoint_tensor(weights, key)
        if normalize_scale:
            tensor = cls._normalize_fp8_block_scale(tensor)
        return load_weight_shard(
            tensor,
            tensor_parallel_size,
            tensor_parallel_rank,
            tensor_parallel_mode,
            torch.device("cuda"),
        )

    @staticmethod
    def _as_fp8_weight(weight: torch.Tensor) -> torch.Tensor:
        weight = weight.contiguous()
        if weight.dtype != torch.float8_e4m3fn:
            weight = weight.view(torch.float8_e4m3fn)
        return weight

    @staticmethod
    def _scale_suffix(weights, prefix: str) -> str:
        if f"{prefix}.weight_scale_inv" in weights:
            return "weight_scale_inv"
        return "weight_scale"

    def _load_shared_gate_up_weight(
        self, prefix: str, weights
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gate_prefix = f"{prefix}.shared_experts.gate_proj"
        up_prefix = f"{prefix}.shared_experts.up_proj"
        scale_suffix = self._scale_suffix(weights, gate_prefix)
        gate_weight = self._load_checkpoint_shard(
            weights,
            f"{gate_prefix}.weight",
            self.shared_tp_size,
            self.shared_tp_rank,
            TensorParallelMode.COLUMN,
        )
        up_weight = self._load_checkpoint_shard(
            weights,
            f"{up_prefix}.weight",
            self.shared_tp_size,
            self.shared_tp_rank,
            TensorParallelMode.COLUMN,
        )
        gate_scale = self._load_checkpoint_shard(
            weights,
            f"{gate_prefix}.{scale_suffix}",
            self.shared_tp_size,
            self.shared_tp_rank,
            TensorParallelMode.COLUMN,
            normalize_scale=True,
        )
        up_scale = self._load_checkpoint_shard(
            weights,
            f"{up_prefix}.{scale_suffix}",
            self.shared_tp_size,
            self.shared_tp_rank,
            TensorParallelMode.COLUMN,
            normalize_scale=True,
        )
        return (
            torch.cat((self._as_fp8_weight(gate_weight), self._as_fp8_weight(up_weight))),
            torch.cat((gate_scale.to(torch.float32), up_scale.to(torch.float32))),
        )

    def _load_shared_down_weight(self, prefix: str, weights) -> tuple[torch.Tensor, torch.Tensor]:
        down_prefix = f"{prefix}.shared_experts.down_proj"
        scale_suffix = self._scale_suffix(weights, down_prefix)
        weight = self._load_checkpoint_shard(
            weights,
            f"{down_prefix}.weight",
            self.shared_tp_size,
            self.shared_tp_rank,
            TensorParallelMode.ROW,
        )
        scale = self._load_checkpoint_shard(
            weights,
            f"{down_prefix}.{scale_suffix}",
            self.shared_tp_size,
            self.shared_tp_rank,
            TensorParallelMode.ROW,
            normalize_scale=True,
        )
        return self._as_fp8_weight(weight), scale.to(torch.float32)

    @staticmethod
    def _expert_proj_prefix(prefix: str, expert_id: int, proj_name: str) -> str:
        return f"{prefix}.experts.{expert_id}.{proj_name}"

    def _load_routed_expert_weights(
        self, prefix: str, weights
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.moe_ep_size != 1:
            raise RuntimeError(
                "TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE=wip currently requires "
                f"moe_ep_size=1, got {self.moe_ep_size}"
            )

        hidden_size = self.hidden_size
        intermediate_size = self.routed_intermediate_size_per_partition
        block_scale_hidden_size = (hidden_size + 127) // 128
        block_scale_intermediate_size = (intermediate_size + 127) // 128
        routed_w3_w1_weight = torch.empty(
            (self.num_experts, 2 * intermediate_size, hidden_size),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        routed_w2_weight = torch.empty(
            (self.num_experts, hidden_size, intermediate_size),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        routed_w3_w1_weight_scale = torch.empty(
            (self.num_experts, 2 * block_scale_intermediate_size, block_scale_hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        routed_w2_weight_scale = torch.empty(
            (self.num_experts, block_scale_hidden_size, block_scale_intermediate_size),
            device="cuda",
            dtype=torch.float32,
        )

        for expert_id in range(self.num_experts):
            gate_prefix = self._expert_proj_prefix(prefix, expert_id, "gate_proj")
            up_prefix = self._expert_proj_prefix(prefix, expert_id, "up_proj")
            down_prefix = self._expert_proj_prefix(prefix, expert_id, "down_proj")
            gate_scale_suffix = self._scale_suffix(weights, gate_prefix)
            up_scale_suffix = self._scale_suffix(weights, up_prefix)
            down_scale_suffix = self._scale_suffix(weights, down_prefix)

            w1_weight = self._load_checkpoint_shard(
                weights,
                f"{gate_prefix}.weight",
                self.moe_tp_size,
                self.moe_tp_rank,
                TensorParallelMode.COLUMN,
            )
            w3_weight = self._load_checkpoint_shard(
                weights,
                f"{up_prefix}.weight",
                self.moe_tp_size,
                self.moe_tp_rank,
                TensorParallelMode.COLUMN,
            )
            w2_weight = self._load_checkpoint_shard(
                weights,
                f"{down_prefix}.weight",
                self.moe_tp_size,
                self.moe_tp_rank,
                TensorParallelMode.ROW,
            )
            w1_scale = self._load_checkpoint_shard(
                weights,
                f"{gate_prefix}.{gate_scale_suffix}",
                self.moe_tp_size,
                self.moe_tp_rank,
                TensorParallelMode.COLUMN,
                normalize_scale=True,
            )
            w3_scale = self._load_checkpoint_shard(
                weights,
                f"{up_prefix}.{up_scale_suffix}",
                self.moe_tp_size,
                self.moe_tp_rank,
                TensorParallelMode.COLUMN,
                normalize_scale=True,
            )
            w2_scale = self._load_checkpoint_shard(
                weights,
                f"{down_prefix}.{down_scale_suffix}",
                self.moe_tp_size,
                self.moe_tp_rank,
                TensorParallelMode.ROW,
                normalize_scale=True,
            )

            routed_w3_w1_weight[expert_id, :intermediate_size].copy_(
                self._as_fp8_weight(w3_weight), non_blocking=True
            )
            routed_w3_w1_weight[expert_id, intermediate_size:].copy_(
                self._as_fp8_weight(w1_weight), non_blocking=True
            )
            routed_w2_weight[expert_id].copy_(self._as_fp8_weight(w2_weight), non_blocking=True)
            routed_w3_w1_weight_scale[expert_id, :block_scale_intermediate_size].copy_(
                w3_scale.to(torch.float32), non_blocking=True
            )
            routed_w3_w1_weight_scale[expert_id, block_scale_intermediate_size:].copy_(
                w1_scale.to(torch.float32), non_blocking=True
            )
            routed_w2_weight_scale[expert_id].copy_(w2_scale.to(torch.float32), non_blocking=True)

        return (
            routed_w3_w1_weight,
            routed_w3_w1_weight_scale,
            routed_w2_weight,
            routed_w2_weight_scale,
        )

    def precompile_dsv3_fused_expert_weight_pack(self) -> None:
        if not torch.cuda.is_available():
            return
        try:
            from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
                deepseekv3_fused_moe_weight_pack,
            )

            deepseekv3_fused_moe_weight_pack.precompile_fused_expert_up_weight_pack_kernels(
                self.shared_intermediate_size_per_partition,
                self.num_experts,
                self.routed_intermediate_size_per_partition,
                torch.device("cuda", torch.cuda.current_device()),
            )
        except ImportError:
            pass

    def load_dsv3_fused_expert_weights(self, prefix: str, weights) -> None:
        router_weight = self._checkpoint_tensor(weights, f"{prefix}.gate.weight").to(
            device="cuda", dtype=torch.bfloat16
        )
        routing_bias = self._checkpoint_tensor(
            weights, f"{prefix}.gate.e_score_correction_bias"
        ).to(device="cuda", dtype=torch.bfloat16)
        shared_gate_up_weight, shared_gate_up_weight_scale = self._load_shared_gate_up_weight(
            prefix, weights
        )
        shared_down_weight, shared_down_weight_scale = self._load_shared_down_weight(
            prefix, weights
        )
        (
            routed_w3_w1_weight,
            routed_w3_w1_weight_scale,
            routed_w2_weight,
            routed_w2_weight_scale,
        ) = self._load_routed_expert_weights(prefix, weights)

        self._set_nonpersistent_buffer(self, "router_weight", router_weight)
        self._set_nonpersistent_buffer(self, "routing_bias", routing_bias)
        self._set_nonpersistent_buffer(
            self, "shared_gate_up_weight_scale_org", shared_gate_up_weight_scale
        )
        self._set_nonpersistent_buffer(
            self,
            "shared_gate_up_weight_packed_fused_expert_up",
            prepack_fused_expert_up_shared_gate_up_weight(shared_gate_up_weight),
        )
        self._set_nonpersistent_buffer(
            self, "routed_w3_w1_weight_scaling_factor", routed_w3_w1_weight_scale
        )
        self._set_nonpersistent_buffer(
            self,
            "routed_w3_w1_weight_packed_fused_expert_up",
            prepack_fused_expert_up_routed_w3_w1_weight(routed_w3_w1_weight),
        )
        self._set_nonpersistent_buffer(
            self, "routed_w2_weight_scaling_factor", routed_w2_weight_scale
        )
        self._set_nonpersistent_buffer(
            self, "shared_down_weight_scale_org", shared_down_weight_scale
        )
        if _fused_moe_prepack_fused_expert_down_enabled():
            self._set_nonpersistent_buffer(
                self,
                "shared_down_weight_packed_fused_expert_down",
                prepack_fused_expert_down_shared_down_weight(shared_down_weight),
            )
            self._set_nonpersistent_buffer(
                self,
                "routed_w2_weight_packed_fused_expert_down",
                prepack_fused_expert_down_routed_w2_weight(routed_w2_weight),
            )
        else:
            self._set_nonpersistent_buffer(self, "shared_down_weight", shared_down_weight)
            self._set_nonpersistent_buffer(self, "routed_w2_weight", routed_w2_weight)

        self._wip_weights_loaded = True

    @classmethod
    def _run_dsv3_fused_expert_up(
        cls,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        routing_bias: torch.Tensor,
        shared_gate_up_weight_packed_fused_expert_up: torch.Tensor,
        shared_gate_up_weight_scale_org: torch.Tensor,
        routed_w3_w1_weight_packed_fused_expert_up: torch.Tensor,
        routed_w3_w1_weight_scale: torch.Tensor,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # fp32, [num_tokens, num_router_experts]
        router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states, router_weight.t(), bias=None, out_dtype=torch.float32
        )
        num_tokens = hidden_states.shape[0]
        num_router_experts = 256
        check_data(
            router_logits,
            "dsv3_fused_expert_up.router_logits",
            torch.float32,
            (num_tokens, num_router_experts),
        )
        assert top_k == 8, f"dsv3_fused_expert_up only supports top_k=8, got {top_k}"
        assert n_group == 1, f"dsv3_fused_expert_up only supports n_group=1, got {n_group}"
        assert topk_group == 1, f"dsv3_fused_expert_up only supports topk_group=1, got {topk_group}"

        expert_weights, expert_indices, slot_swiglu_output = torch.ops.trtllm.dsv3_fused_expert_up(
            router_logits.contiguous(),
            hidden_states.contiguous(),
            routing_bias.contiguous(),
            shared_gate_up_weight_packed_fused_expert_up,
            shared_gate_up_weight_scale_org,
            routed_w3_w1_weight_packed_fused_expert_up,
            routed_w3_w1_weight_scale,
            routed_scaling_factor,
        )
        check_data(
            expert_indices,
            "dsv3_fused_expert_up.expert_indices",
            torch.int32,
            (num_tokens, top_k),
        )
        check_data(
            expert_weights,
            "dsv3_fused_expert_up.expert_weights",
            torch.float32,
            (num_tokens, top_k),
        )
        check_data(
            slot_swiglu_output,
            "dsv3_fused_expert_up.slot_swiglu_output",
            torch.float16,
            (num_tokens, top_k + 1, -1),
        )
        return expert_indices, expert_weights, slot_swiglu_output

    @classmethod
    def _run_dsv3_fused_expert_down_chunked(
        cls,
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
        for token_start in range(0, num_tokens, cls._WIP_DOWN_PROJECT_MAX_NUM_TOKENS):
            token_end = min(token_start + cls._WIP_DOWN_PROJECT_MAX_NUM_TOKENS, num_tokens)
            torch.ops.trtllm.dsv3_fused_expert_down(
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

    def _forward_dsv3_fused_expert(
        self,
        hidden_states: torch.Tensor,
        final_all_reduce_params: AllReduceParams | None,
        num_tokens: int,
        hidden_size: int,
        num_router_experts: int,
        expert_intermediate_size: int,
        block_scale_fp32_hidden_size: int,
        block_scale_fp32_expert_intermediate_size: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self._wip_weights_loaded:
            raise RuntimeError(
                "TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE=wip requires "
                "Deepseekv3FusedMoE fused expert weights to be loaded before forward()"
            )
        if (
            self.shared_intermediate_size_per_partition
            != self.routed_intermediate_size_per_partition
        ):
            raise RuntimeError(
                "Deepseekv3FusedMoE fused expert kernels require matching shared and routed "
                "intermediate partitions, got "
                f"{self.shared_intermediate_size_per_partition} and "
                f"{self.routed_intermediate_size_per_partition}"
            )
        assert self.shared_output_scale is None, (
            f"shared_output_scale must be None, got {self.shared_output_scale}"
        )
        assert self.num_experts == num_router_experts
        assert self.top_k == 8, f"dsv3_fused_expert_up only supports top_k=8, got {self.top_k}"
        assert self.n_group == 1, (
            f"dsv3_fused_expert_up only supports n_group=1, got {self.n_group}"
        )
        assert self.topk_group == 1, (
            f"dsv3_fused_expert_up only supports topk_group=1, got {self.topk_group}"
        )

        router_weight = self.router_weight
        routing_bias = self.routing_bias
        shared_gate_up_weight_packed_fused_expert_up = (
            self.shared_gate_up_weight_packed_fused_expert_up
        )
        shared_gate_up_weight_scale_org = self.shared_gate_up_weight_scale_org
        routed_w3_w1_weight_packed_fused_expert_up = self.routed_w3_w1_weight_packed_fused_expert_up
        routed_w3_w1_weight_scale = self.routed_w3_w1_weight_scaling_factor
        routed_w2_weight_scale = self.routed_w2_weight_scaling_factor
        shared_down_weight_scale_org = self.shared_down_weight_scale_org
        routed_w2_weight_for_fused_expert_down = getattr(
            self, "routed_w2_weight_packed_fused_expert_down", None
        )
        if routed_w2_weight_for_fused_expert_down is None:
            routed_w2_weight_for_fused_expert_down = self.routed_w2_weight
        shared_down_weight_for_fused_expert_down = getattr(
            self, "shared_down_weight_packed_fused_expert_down", None
        )
        if shared_down_weight_for_fused_expert_down is None:
            shared_down_weight_for_fused_expert_down = self.shared_down_weight

        check_data(
            router_weight,
            "dsv3_fused_expert_up.router_weight",
            torch.bfloat16,
            (num_router_experts, hidden_size),
        )
        check_data(
            routing_bias,
            "dsv3_fused_expert_up.router_logit_offset",
            torch.bfloat16,
            (num_router_experts,),
        )
        check_data(
            shared_gate_up_weight_packed_fused_expert_up,
            "dsv3_fused_expert_up.shared_gate_up_weight_packed",
            torch.float8_e4m3fn,
            (
                2,
                expert_intermediate_size // _FUSED_EXPERT_UP_CTA_OUT_ROWS,
                _FUSED_EXPERT_UP_NUM_K_ITER,
                _FUSED_EXPERT_UP_TILE_BYTES,
            ),
        )
        check_data(
            shared_gate_up_weight_scale_org,
            "dsv3_fused_expert_up.shared_gate_up_weight_scale",
            torch.float32,
            (2 * block_scale_fp32_expert_intermediate_size, block_scale_fp32_hidden_size),
        )
        check_data(
            routed_w3_w1_weight_packed_fused_expert_up,
            "dsv3_fused_expert_up.routed_w3_w1_weight_packed",
            torch.float8_e4m3fn,
            (
                num_router_experts,
                2,
                expert_intermediate_size // _FUSED_EXPERT_UP_CTA_OUT_ROWS,
                _FUSED_EXPERT_UP_NUM_K_ITER,
                _FUSED_EXPERT_UP_TILE_BYTES,
            ),
        )
        check_data(
            routed_w3_w1_weight_scale,
            "dsv3_fused_expert_up.routed_w3_w1_weight_scale",
            torch.float32,
            (
                num_router_experts,
                2 * block_scale_fp32_expert_intermediate_size,
                block_scale_fp32_hidden_size,
            ),
        )
        if getattr(self, "routed_w2_weight_packed_fused_expert_down", None) is not None:
            check_data(
                routed_w2_weight_for_fused_expert_down,
                "dsv3_fused_expert_down.routed_w2_weight_packed",
                torch.float8_e4m3fn,
                (
                    num_router_experts,
                    _FUSED_EXPERT_DOWN_PACKED_ROW_TILES,
                    block_scale_fp32_expert_intermediate_size,
                    _FUSED_EXPERT_DOWN_TILE_BYTES,
                ),
            )
        else:
            check_data(
                routed_w2_weight_for_fused_expert_down,
                "dsv3_fused_expert_down.routed_w2_weight",
                torch.float8_e4m3fn,
                (num_router_experts, hidden_size, expert_intermediate_size),
            )
        check_data(
            routed_w2_weight_scale,
            "dsv3_fused_expert_down.routed_w2_weight_scale",
            torch.float32,
            (
                num_router_experts,
                block_scale_fp32_hidden_size,
                block_scale_fp32_expert_intermediate_size,
            ),
        )
        if getattr(self, "shared_down_weight_packed_fused_expert_down", None) is not None:
            check_data(
                shared_down_weight_for_fused_expert_down,
                "dsv3_fused_expert_down.shared_down_weight_packed",
                torch.float8_e4m3fn,
                (
                    _FUSED_EXPERT_DOWN_PACKED_ROW_TILES,
                    block_scale_fp32_expert_intermediate_size,
                    _FUSED_EXPERT_DOWN_TILE_BYTES,
                ),
            )
        else:
            check_data(
                shared_down_weight_for_fused_expert_down,
                "dsv3_fused_expert_down.shared_down_weight",
                torch.float8_e4m3fn,
                (hidden_size, expert_intermediate_size),
            )
        check_data(
            shared_down_weight_scale_org,
            "dsv3_fused_expert_down.shared_down_weight_scale",
            torch.float32,
            (block_scale_fp32_hidden_size, block_scale_fp32_expert_intermediate_size),
        )

        expert_indices, expert_weights, slot_swiglu_output = self._run_dsv3_fused_expert_up(
            hidden_states,
            router_weight,
            routing_bias,
            shared_gate_up_weight_packed_fused_expert_up,
            shared_gate_up_weight_scale_org,
            routed_w3_w1_weight_packed_fused_expert_up,
            routed_w3_w1_weight_scale,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
        )
        check_data(
            slot_swiglu_output,
            "dsv3_fused_expert_up.slot_swiglu_output",
            torch.float16,
            (num_tokens, self.top_k + 1, expert_intermediate_size),
        )

        if self._can_finalize_with_fused_expert_down(final_all_reduce_params, num_tokens):
            final_hidden_states, residual_out = self._run_dsv3_fused_expert_down_finalize(
                slot_swiglu_output,
                expert_indices,
                expert_weights,
                routed_w2_weight_for_fused_expert_down,
                routed_w2_weight_scale,
                shared_down_weight_for_fused_expert_down,
                shared_down_weight_scale_org,
                final_all_reduce_params,
            )
            check_data(
                final_hidden_states,
                "dsv3_fused_expert_down.final_hidden_states",
                torch.bfloat16,
                tuple(hidden_states.shape),
            )
            check_data(
                residual_out,
                "dsv3_fused_expert_down.residual_out",
                torch.bfloat16,
                tuple(hidden_states.shape),
            )
            return final_hidden_states, residual_out

        output_tensor = None
        if not self.use_dp and self.mapping.tp_size > 1:
            w, actual_kind = torch.ops.trtllm.allocate_output(
                hidden_states, self.allreduce.output_buffer_kind, self.mapping.tp_group
            )
            if actual_kind == int(BufferKind.NCCL_WINDOW):
                output_tensor = w
        if output_tensor is None:
            output_tensor = torch.empty_like(hidden_states)

        local_hidden_states = self._run_dsv3_fused_expert_down_chunked(
            slot_swiglu_output,
            expert_indices,
            expert_weights,
            routed_w2_weight_for_fused_expert_down,
            routed_w2_weight_scale,
            shared_down_weight_for_fused_expert_down,
            shared_down_weight_scale_org,
            output_tensor,
        )
        check_data(
            local_hidden_states,
            "dsv3_fused_expert_down.final_hidden_states",
            torch.bfloat16,
            tuple(hidden_states.shape),
        )

        final_hidden_states: torch.Tensor | tuple[torch.Tensor, ...] = local_hidden_states
        if final_all_reduce_params is not None:
            if self.allreduce is None:
                if final_all_reduce_params.enable_allreduce:
                    raise RuntimeError(
                        "Deepseekv3FusedMoE received enabled final_all_reduce_params "
                        "without an AllReduce module"
                    )
            else:
                final_hidden_states = self.allreduce(
                    local_hidden_states,
                    all_reduce_params=final_all_reduce_params,
                )
        return final_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Fp4QuantizedTensor | None = None,
        all_rank_num_tokens: list[int] | None = None,
        final_all_reduce_params: AllReduceParams | None = None,
        do_finalize: bool | None = True,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert do_finalize is True, "Deepseekv3FusedMoE only supports do_finalize=True"
        del hidden_states_fp4

        num_tokens: int = hidden_states.size(0)
        selected_fused_moe_mode = get_fused_moe_mode()
        hidden_size: int = hidden_states.size(1)
        block_scale_fp32_hidden_Size: int = int(triton.cdiv(hidden_size, 128))

        if selected_fused_moe_mode != FUSED_MOE_MODE_WIP:
            raise RuntimeError(
                f"Deepseekv3FusedMoE does not implement mode {selected_fused_moe_mode!r}; "
                "baseline mode is handled by Deepseekv3MoE."
            )

        check_data(hidden_states, "hidden_states", torch.bfloat16, (-1, 6144))
        assert is_sm_100f(), "fp8_quantize_1x128_packed_ue8m0 requires SM100-family Blackwell"
        num_router_experts = self.num_experts
        expert_intermediate_size = self.shared_intermediate_size_per_partition
        block_scale_fp32_expert_intermediate_size = int(triton.cdiv(expert_intermediate_size, 128))
        return self._forward_dsv3_fused_expert(
            hidden_states=hidden_states,
            final_all_reduce_params=final_all_reduce_params,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_router_experts=num_router_experts,
            expert_intermediate_size=expert_intermediate_size,
            block_scale_fp32_hidden_size=block_scale_fp32_hidden_Size,
            block_scale_fp32_expert_intermediate_size=(block_scale_fp32_expert_intermediate_size),
        )
