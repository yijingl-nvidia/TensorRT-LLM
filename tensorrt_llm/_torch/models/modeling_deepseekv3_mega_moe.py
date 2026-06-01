import os
import threading
from types import SimpleNamespace
from typing import Dict, Optional

import torch
import triton
from torch import nn

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.fused_moe import MoE
from ..modules.linear import TensorParallelMode, load_weight_shard
from ..utils import AuxStreamType, Fp4QuantizedTensor
from .modeling_deepseekv3_moe import Deepseekv3MoE

_MEGA_MOE_PREPACK_V110_ENV = "TRTLLM_DEEPSEEKV3_MEGAMOE_PREPACK_V110"
_MEGA_MOE_WIP_WEIGHT_LOAD_LOCK = threading.Lock()
_V68_HIDDEN_SIZE = 6144
_V68_CTA_OUT_ROWS = 64
_V68_M_TILES_PER_CTA = 4
_V68_ROW_HALVES_PER_M_TILE = 2
_V68_ROWS_PER_HALF = 8
_V68_NUM_K_ITER = 8
_V68_K_THIRDS_PER_ITER = 6
_V68_K_SUBS_PER_THIRD = 4
_V68_COL_HALVES_PER_K_SUB = 2
_V68_COL_QUADS_PER_HALF = 4
_V68_BYTES_PER_COL_QUAD = 4
_V68_TILE_BYTES = 49152
_V110_HIDDEN_SIZE = 6144
_V110_NUM_CTAS = 148
_V110_ROWS_PER_CTA = 42
_V110_MMA_M = 16
_V110_ROW_TILES_PER_CTA = 3
_V110_PACKED_ROW_TILES = _V110_NUM_CTAS * _V110_ROW_TILES_PER_CTA
_V110_BLOCK_K = 128
_V110_TILE_BYTES = 2048


def _mega_moe_prepack_v110_enabled() -> bool:
    value = os.environ.get(_MEGA_MOE_PREPACK_V110_ENV, "0").strip().lower()
    return value in ("1", "true", "yes", "on")


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


def _prepack_v68_weight_side_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [..., I, H].
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"v68 prepack expects float8_e4m3fn weights, got {weight.dtype}")
    if weight.shape[-1] != _V68_HIDDEN_SIZE:
        raise ValueError(
            f"v68 prepack expects hidden size {_V68_HIDDEN_SIZE}, got {weight.shape[-1]}"
        )
    inter_per_tp = weight.shape[-2]
    if inter_per_tp % _V68_CTA_OUT_ROWS != 0:
        raise ValueError(
            f"v68 prepack expects I to be divisible by {_V68_CTA_OUT_ROWS}, got {inter_per_tp}"
        )

    prefix_shape = tuple(weight.shape[:-2])
    num_prefix_dims = len(prefix_shape)
    sub_rows = inter_per_tp // _V68_CTA_OUT_ROWS

    reshaped = weight.contiguous().reshape(
        *prefix_shape,
        sub_rows,
        _V68_M_TILES_PER_CTA,
        _V68_ROW_HALVES_PER_M_TILE,
        _V68_ROWS_PER_HALF,
        _V68_NUM_K_ITER,
        _V68_K_THIRDS_PER_ITER,
        _V68_K_SUBS_PER_THIRD,
        _V68_COL_HALVES_PER_K_SUB,
        _V68_COL_QUADS_PER_HALF,
        _V68_BYTES_PER_COL_QUAD,
    )
    # Reorder the raw row-major [64, 768] tile into the exact 48 KiB K-major
    # lane slab consumed by compute_mma_kiter_v68().
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
        .reshape(*prefix_shape, sub_rows, _V68_NUM_K_ITER, _V68_TILE_BYTES)
    )


def prepack_v68_shared_gate_up_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [2 * I, H], stored as [gate, up].
    gate_weight, up_weight = weight.chunk(2, dim=0)
    gate_packed = _prepack_v68_weight_side_pytorch(gate_weight)
    up_packed = _prepack_v68_weight_side_pytorch(up_weight)
    packed = torch.empty((2, *gate_packed.shape), device=weight.device, dtype=weight.dtype)
    # packed: torch.float8_e4m3fn, [2, I / 64, 8, 49152], stored as [gate, up].
    packed[0].copy_(gate_packed)
    packed[1].copy_(up_packed)
    return packed


def prepack_v68_routed_w3_w1_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [E, 2 * I, H], stored as [up, gate].
    up_weight, gate_weight = weight.chunk(2, dim=1)
    gate_packed = _prepack_v68_weight_side_pytorch(gate_weight)
    up_packed = _prepack_v68_weight_side_pytorch(up_weight)
    packed = torch.empty(
        (weight.shape[0], 2, *gate_packed.shape[1:]),
        device=weight.device,
        dtype=weight.dtype,
    )
    # packed: torch.float8_e4m3fn, [E, 2, I / 64, 8, 49152], stored as [gate, up].
    packed[:, 0].copy_(gate_packed)
    packed[:, 1].copy_(up_packed)
    return packed


def _should_use_cute_v68_weight_pack(weight: torch.Tensor) -> bool:
    return weight.is_cuda and torch.cuda.is_available()


def prepack_v68_shared_gate_up_weight(weight: torch.Tensor) -> torch.Tensor:
    if _should_use_cute_v68_weight_pack(weight):
        try:
            from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
                deepseekv3_mega_moe_weight_pack,
            )

            return deepseekv3_mega_moe_weight_pack.pack_v68_shared_gate_up_weight(weight)
        except ImportError:
            pass
    return prepack_v68_shared_gate_up_weight_pytorch(weight)


def prepack_v68_routed_w3_w1_weight(weight: torch.Tensor) -> torch.Tensor:
    if _should_use_cute_v68_weight_pack(weight):
        try:
            from tensorrt_llm._torch.cute_dsl_kernels.blackwell import (
                deepseekv3_mega_moe_weight_pack,
            )

            return deepseekv3_mega_moe_weight_pack.pack_v68_routed_w3_w1_weight(weight)
        except ImportError:
            pass
    return prepack_v68_routed_w3_w1_weight_pytorch(weight)


def _v110_packed_indices(device: torch.device, k_local: int) -> tuple[torch.Tensor, torch.Tensor]:
    if k_local % _V110_BLOCK_K != 0:
        raise ValueError(
            f"v110 prepack expects K_local divisible by {_V110_BLOCK_K}, got {k_local}"
        )

    tile_idx = torch.arange(_V110_PACKED_ROW_TILES, device=device, dtype=torch.int64)
    row_base = (tile_idx // _V110_ROW_TILES_PER_CTA) * _V110_ROWS_PER_CTA
    row_base += (tile_idx % _V110_ROW_TILES_PER_CTA) * _V110_MMA_M

    elem = torch.arange(_V110_TILE_BYTES, device=device, dtype=torch.int64)
    row = elem // _V110_BLOCK_K
    phys_col = elem - row * _V110_BLOCK_K
    chunk_phys = phys_col // 16
    byte = phys_col - chunk_phys * 16
    logical_col_in_kb = ((chunk_phys ^ (row & 7)) * 16) + byte

    src_rows = row_base[:, None] + row[None, :]
    src_rows = torch.where(
        src_rows < _V110_HIDDEN_SIZE,
        src_rows,
        torch.full_like(src_rows, _V110_HIDDEN_SIZE),
    )
    kb = torch.arange(k_local // _V110_BLOCK_K, device=device, dtype=torch.int64)
    src_cols = kb[:, None] * _V110_BLOCK_K + logical_col_in_kb[None, :]
    return src_rows, src_cols


def prepack_v110_shared_down_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [H, I].
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"v110 prepack expects float8_e4m3fn weights, got {weight.dtype}")
    if weight.ndim != 2 or weight.shape[0] != _V110_HIDDEN_SIZE:
        raise ValueError(f"shared v110 prepack expects [H, I], got {weight.shape}")

    weight = weight.contiguous()
    src_rows, src_cols = _v110_packed_indices(weight.device, weight.shape[1])
    padded = torch.empty(
        (_V110_HIDDEN_SIZE + _V110_MMA_M, weight.shape[1]),
        device=weight.device,
        dtype=weight.dtype,
    )
    padded[:_V110_HIDDEN_SIZE].copy_(weight)
    padded[_V110_HIDDEN_SIZE:].zero_()
    # packed: torch.float8_e4m3fn, [444, I / 128, 2048].
    return padded[src_rows[:, None, :], src_cols[None, :, :]].contiguous()


def prepack_v110_routed_w2_weight_pytorch(weight: torch.Tensor) -> torch.Tensor:
    # weight: torch.float8_e4m3fn, [E, H, I].
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"v110 prepack expects float8_e4m3fn weights, got {weight.dtype}")
    if weight.ndim != 3 or weight.shape[1] != _V110_HIDDEN_SIZE:
        raise ValueError(f"routed v110 prepack expects [E, H, I], got {weight.shape}")

    weight = weight.contiguous()
    src_rows, src_cols = _v110_packed_indices(weight.device, weight.shape[2])
    padded = torch.empty(
        (weight.shape[0], _V110_HIDDEN_SIZE + _V110_MMA_M, weight.shape[2]),
        device=weight.device,
        dtype=weight.dtype,
    )
    padded[:, :_V110_HIDDEN_SIZE].copy_(weight)
    padded[:, _V110_HIDDEN_SIZE:].zero_()
    # packed: torch.float8_e4m3fn, [E, 444, I / 128, 2048].
    return padded[:, src_rows[:, None, :], src_cols[None, :, :]].contiguous()


def prepack_v110_shared_down_weight(weight: torch.Tensor) -> torch.Tensor:
    return prepack_v110_shared_down_weight_pytorch(weight)


def prepack_v110_routed_w2_weight(weight: torch.Tensor) -> torch.Tensor:
    return prepack_v110_routed_w2_weight_pytorch(weight)


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
        del aux_stream_dict, dtype, override_quant_config
        mega_moe_mode = self._mega_moe_mode()
        if mega_moe_mode == self._MEGA_MOE_MODE_BASELINE:
            raise RuntimeError(
                "TRTLLM_DEEPSEEKV3_MEGAMOE_MODE=baseline is handled by "
                "constructing Deepseekv3MoE directly."
            )
        self._wip_owns_weights = mega_moe_mode == self._MEGA_MOE_MODE_WIP
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.model_config = model_config
        self.mapping = model_config.mapping
        self.use_dp = model_config.mapping.enable_attention_dp
        self.use_cute_dsl_blockscaling_mm = model_config.use_cute_dsl_blockscaling_mm
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

    @property
    def experts(self) -> MoE:
        return self.experts_stub

    _MEGA_MOE_MODE_BASELINE = "baseline"
    _MEGA_MOE_MODE_WIP = "wip_mega_kernel"
    _WIP_DOWN_PROJECT_MAX_NUM_TOKENS = 4

    @classmethod
    def _mega_moe_mode(cls) -> str:
        mode = os.environ.get("TRTLLM_DEEPSEEKV3_MEGAMOE_MODE", "").strip().lower()
        if not mode:
            return cls._MEGA_MOE_MODE_BASELINE

        mode_aliases = {
            "baseline": cls._MEGA_MOE_MODE_BASELINE,
            "trtllm": cls._MEGA_MOE_MODE_BASELINE,
            "trtllm_baseline": cls._MEGA_MOE_MODE_BASELINE,
            "mega": cls._MEGA_MOE_MODE_WIP,
            "mega_kernel": cls._MEGA_MOE_MODE_WIP,
            "wip": cls._MEGA_MOE_MODE_WIP,
            "wip_mega_kernel": cls._MEGA_MOE_MODE_WIP,
        }
        if mode not in mode_aliases:
            allowed_modes = ", ".join(
                (
                    cls._MEGA_MOE_MODE_BASELINE,
                    cls._MEGA_MOE_MODE_WIP,
                )
            )
            raise ValueError(
                f"Unsupported TRTLLM_DEEPSEEKV3_MEGAMOE_MODE={mode!r}; "
                f"expected one of: {allowed_modes}"
            )
        return mode_aliases[mode]

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

    def uses_wip_mega_kernel_weights(self) -> bool:
        return self._wip_owns_weights

    @staticmethod
    def _checkpoint_tensor(weights, key: str) -> torch.Tensor:
        if key not in weights:
            raise KeyError(f"Missing Deepseekv3MegaMoE WIP weight: {key}")
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
                "TRTLLM_DEEPSEEKV3_MEGAMOE_MODE=wip_mega_kernel currently requires "
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

    def load_wip_mega_kernel_weights(self, prefix: str, weights) -> None:
        # Weight loading is parallelized by layer; serialize WIP packing to avoid
        # duplicate first-use CuTe DSL compilation and lower peak temporary memory.
        with _MEGA_MOE_WIP_WEIGHT_LOAD_LOCK:
            self._load_wip_mega_kernel_weights_locked(prefix, weights)

    def _load_wip_mega_kernel_weights_locked(self, prefix: str, weights) -> None:
        if not self._wip_owns_weights:
            return

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
            "shared_gate_up_weight_packed_v68",
            prepack_v68_shared_gate_up_weight(shared_gate_up_weight),
        )
        self._set_nonpersistent_buffer(
            self, "routed_w3_w1_weight_scaling_factor", routed_w3_w1_weight_scale
        )
        self._set_nonpersistent_buffer(
            self,
            "routed_w3_w1_weight_packed_v68",
            prepack_v68_routed_w3_w1_weight(routed_w3_w1_weight),
        )
        self._set_nonpersistent_buffer(
            self, "routed_w2_weight_scaling_factor", routed_w2_weight_scale
        )
        self._set_nonpersistent_buffer(
            self, "shared_down_weight_scale_org", shared_down_weight_scale
        )
        if _mega_moe_prepack_v110_enabled():
            self._set_nonpersistent_buffer(
                self,
                "shared_down_weight_packed_v110",
                prepack_v110_shared_down_weight(shared_down_weight),
            )
            self._set_nonpersistent_buffer(
                self,
                "routed_w2_weight_packed_v110",
                prepack_v110_routed_w2_weight(routed_w2_weight),
            )
        else:
            self._set_nonpersistent_buffer(self, "shared_down_weight_org", shared_down_weight)
            self._set_nonpersistent_buffer(self, "routed_w2_weight", routed_w2_weight)

        self._wip_weights_loaded = True

    @classmethod
    def _run_wip_packed_mega_kernel(
        cls,
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
        routing_bias: torch.Tensor,
        shared_gate_up_weight_packed_v68: torch.Tensor,
        shared_gate_up_weight_scale_org: torch.Tensor,
        routed_w3_w1_weight_packed_v68: torch.Tensor,
        routed_w3_w1_weight_scale: torch.Tensor,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        shared_swiglu_limit: Optional[float],
        routed_swiglu_limit: Optional[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # fp32, [num_tokens, num_router_experts]
        router_logits = torch.ops.trtllm.dsv3_router_gemm_op(
            hidden_states, router_weight.t(), bias=None, out_dtype=torch.float32
        )
        num_tokens = hidden_states.shape[0]
        num_router_experts = 256
        check_data(
            router_logits,
            "wip_packed_mega_kernels.router_logits",
            torch.float32,
            (num_tokens, num_router_experts),
        )
        assert top_k == 8, f"v68 WIP mega kernel only supports top_k=8, got {top_k}"
        assert n_group == 1, f"v68 WIP mega kernel only supports n_group=1, got {n_group}"
        assert topk_group == 1, f"v68 WIP mega kernel only supports topk_group=1, got {topk_group}"

        expert_weights, expert_indices, slot_swiglu_output = (
            torch.ops.trtllm.glm5_expert_select_up_gate_silu_packed(
                router_logits.contiguous(),
                hidden_states.contiguous(),
                routing_bias.contiguous(),
                shared_gate_up_weight_packed_v68,
                shared_gate_up_weight_scale_org,
                routed_w3_w1_weight_packed_v68,
                routed_w3_w1_weight_scale,
                routed_scaling_factor,
            )
        )
        check_data(
            expert_indices,
            "wip_packed_mega_kernels.expert_indices",
            torch.int32,
            (num_tokens, top_k),
        )
        check_data(
            expert_weights,
            "wip_packed_mega_kernels.expert_weights",
            torch.float32,
            (num_tokens, top_k),
        )
        check_data(
            slot_swiglu_output,
            "wip_packed_mega_kernels.slot_swiglu_output",
            torch.float16,
            (num_tokens, top_k + 1, -1),
        )
        return expert_indices, expert_weights, slot_swiglu_output

    @classmethod
    def _run_wip_down_project_chunked(
        cls,
        slot_swiglu_output: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        routed_w2_weight: torch.Tensor,
        routed_w2_weight_scale: torch.Tensor,
        shared_down_weight_org: torch.Tensor,
        shared_down_weight_scale_org: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> torch.Tensor:
        num_tokens = slot_swiglu_output.size(0)
        for token_start in range(0, num_tokens, cls._WIP_DOWN_PROJECT_MAX_NUM_TOKENS):
            token_end = min(token_start + cls._WIP_DOWN_PROJECT_MAX_NUM_TOKENS, num_tokens)
            torch.ops.trtllm.glm5_expert_down_project(
                slot_swiglu_output[token_start:token_end].contiguous(),
                expert_indices[token_start:token_end].contiguous(),
                expert_weights[token_start:token_end].contiguous(),
                routed_w2_weight,
                routed_w2_weight_scale,
                shared_down_weight_org,
                shared_down_weight_scale_org,
                output_tensor[token_start:token_end],
            )
        return output_tensor

    def _forward_wip_mega_kernel_owned(
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
        del all_rank_num_tokens, block_scale_int32_hidden_size, gate_up_output_size
        if not self._wip_weights_loaded:
            raise RuntimeError(
                "TRTLLM_DEEPSEEKV3_MEGAMOE_MODE=wip_mega_kernel requires "
                "Deepseekv3MegaMoE WIP weights to be loaded before forward()"
            )
        if (
            self.shared_intermediate_size_per_partition
            != self.routed_intermediate_size_per_partition
        ):
            raise RuntimeError(
                "Deepseekv3MegaMoE WIP kernels require matching shared and routed "
                "intermediate partitions, got "
                f"{self.shared_intermediate_size_per_partition} and "
                f"{self.routed_intermediate_size_per_partition}"
            )
        assert self.shared_output_scale is None, (
            f"shared_output_scale must be None, got {self.shared_output_scale}"
        )
        assert self.num_experts == num_router_experts
        assert self.top_k == 8, f"v68 WIP mega kernel only supports top_k=8, got {self.top_k}"
        assert self.n_group == 1, f"v68 WIP mega kernel only supports n_group=1, got {self.n_group}"
        assert self.topk_group == 1, (
            f"v68 WIP mega kernel only supports topk_group=1, got {self.topk_group}"
        )

        router_weight = self.router_weight
        routing_bias = self.routing_bias
        shared_gate_up_weight_packed_v68 = self.shared_gate_up_weight_packed_v68
        shared_gate_up_weight_scale_org = self.shared_gate_up_weight_scale_org
        routed_w3_w1_weight_packed_v68 = self.routed_w3_w1_weight_packed_v68
        routed_w3_w1_weight_scale = self.routed_w3_w1_weight_scaling_factor
        routed_w2_weight_scale = self.routed_w2_weight_scaling_factor
        shared_down_weight_scale_org = self.shared_down_weight_scale_org
        routed_w2_weight_for_v110 = getattr(self, "routed_w2_weight_packed_v110", None)
        if routed_w2_weight_for_v110 is None:
            routed_w2_weight_for_v110 = self.routed_w2_weight
        shared_down_weight_for_v110 = getattr(self, "shared_down_weight_packed_v110", None)
        if shared_down_weight_for_v110 is None:
            shared_down_weight_for_v110 = self.shared_down_weight_org

        check_data(
            router_weight,
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
            shared_gate_up_weight_packed_v68,
            "wip_mega_shared_gate_up_weight_packed_v68",
            torch.float8_e4m3fn,
            (2, expert_intermediate_size // _V68_CTA_OUT_ROWS, _V68_NUM_K_ITER, _V68_TILE_BYTES),
        )
        check_data(
            shared_gate_up_weight_scale_org,
            "wip_mega_shared_gate_up_weight_scale_org",
            torch.float32,
            (2 * block_scale_fp32_expert_intermediate_size, block_scale_fp32_hidden_size),
        )
        check_data(
            routed_w3_w1_weight_packed_v68,
            "wip_mega_routed_w3_w1_weight_packed_v68",
            torch.float8_e4m3fn,
            (
                num_router_experts,
                2,
                expert_intermediate_size // _V68_CTA_OUT_ROWS,
                _V68_NUM_K_ITER,
                _V68_TILE_BYTES,
            ),
        )
        check_data(
            routed_w3_w1_weight_scale,
            "wip_mega_routed_w3_w1_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                2 * block_scale_fp32_expert_intermediate_size,
                block_scale_fp32_hidden_size,
            ),
        )
        if getattr(self, "routed_w2_weight_packed_v110", None) is not None:
            check_data(
                routed_w2_weight_for_v110,
                "wip_mega_routed_w2_weight_packed_v110",
                torch.float8_e4m3fn,
                (
                    num_router_experts,
                    _V110_PACKED_ROW_TILES,
                    block_scale_fp32_expert_intermediate_size,
                    _V110_TILE_BYTES,
                ),
            )
        else:
            check_data(
                routed_w2_weight_for_v110,
                "wip_mega_routed_w2_weight",
                torch.float8_e4m3fn,
                (num_router_experts, hidden_size, expert_intermediate_size),
            )
        check_data(
            routed_w2_weight_scale,
            "wip_mega_routed_w2_weight_scaling_factor",
            torch.float32,
            (
                num_router_experts,
                block_scale_fp32_hidden_size,
                block_scale_fp32_expert_intermediate_size,
            ),
        )
        if getattr(self, "shared_down_weight_packed_v110", None) is not None:
            check_data(
                shared_down_weight_for_v110,
                "wip_mega_shared_down_weight_packed_v110",
                torch.float8_e4m3fn,
                (
                    _V110_PACKED_ROW_TILES,
                    block_scale_fp32_expert_intermediate_size,
                    _V110_TILE_BYTES,
                ),
            )
        else:
            check_data(
                shared_down_weight_for_v110,
                "wip_mega_shared_down_weight_org",
                torch.float8_e4m3fn,
                (hidden_size, expert_intermediate_size),
            )
        check_data(
            shared_down_weight_scale_org,
            "wip_mega_shared_down_weight_scale_org",
            torch.float32,
            (block_scale_fp32_hidden_size, block_scale_fp32_expert_intermediate_size),
        )

        expert_indices, expert_weights, slot_swiglu_output = self._run_wip_packed_mega_kernel(
            hidden_states,
            router_weight,
            routing_bias,
            shared_gate_up_weight_packed_v68,
            shared_gate_up_weight_scale_org,
            routed_w3_w1_weight_packed_v68,
            routed_w3_w1_weight_scale,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            None,
            self.swiglu_limit_scalar,
        )
        check_data(
            slot_swiglu_output,
            "wip_mega_slot_swiglu_output",
            torch.float16,
            (num_tokens, self.top_k + 1, expert_intermediate_size),
        )

        output_tensor = None
        if not self.use_dp and self.mapping.tp_size > 1:
            w, actual_kind = torch.ops.trtllm.allocate_output(
                hidden_states, self.allreduce.output_buffer_kind, self.mapping.tp_group
            )
            if actual_kind == int(BufferKind.NCCL_WINDOW):
                output_tensor = w
        if output_tensor is None:
            output_tensor = torch.empty_like(hidden_states)

        final_hidden_states = self._run_wip_down_project_chunked(
            slot_swiglu_output,
            expert_indices,
            expert_weights,
            routed_w2_weight_for_v110,
            routed_w2_weight_scale,
            shared_down_weight_for_v110,
            shared_down_weight_scale_org,
            output_tensor,
        )
        check_data(
            final_hidden_states,
            "wip_mega_final_hidden_states",
            torch.bfloat16,
            tuple(hidden_states.shape),
        )
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
        return self._forward_wip_mega_kernel_owned(
            hidden_states=hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_router_experts=num_router_experts,
            expert_intermediate_size=expert_intermediate_size,
            block_scale_fp32_hidden_size=block_scale_fp32_hidden_size,
            block_scale_int32_hidden_size=block_scale_int32_hidden_size,
            block_scale_fp32_expert_intermediate_size=(block_scale_fp32_expert_intermediate_size),
            gate_up_output_size=gate_up_output_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Fp4QuantizedTensor | None = None,
        all_rank_num_tokens: list[int] | None = None,
        final_all_reduce_params: AllReduceParams | None = None,
        do_finalize: bool | None = True,
    ) -> torch.Tensor:
        assert do_finalize is True, "Deepseekv3MegaMoE only supports do_finalize=True"
        del hidden_states_fp4, final_all_reduce_params

        num_tokens: int = hidden_states.size(0)
        mega_moe_mode = self._mega_moe_mode()
        hidden_size: int = hidden_states.size(1)
        block_scale_fp32_hidden_Size: int = int(triton.cdiv(hidden_size, 128))
        block_scale_int32_hidden_size: int = int(triton.cdiv(hidden_size, 128 * 4))

        if mega_moe_mode == self._MEGA_MOE_MODE_WIP:
            check_data(hidden_states, "hidden_states", torch.bfloat16, (-1, 6144))
            assert is_sm_100f(), "fp8_quantize_1x128_packed_ue8m0 requires SM100-family Blackwell"
            num_router_experts = self.num_experts
            expert_intermediate_size = self.shared_intermediate_size_per_partition
            block_scale_fp32_expert_intermediate_size = int(
                triton.cdiv(expert_intermediate_size, 128)
            )
            gate_up_output_size = 2 * expert_intermediate_size
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

        raise RuntimeError(
            f"Deepseekv3MegaMoE does not implement mode {mega_moe_mode!r}; "
            "baseline mode is handled by Deepseekv3MoE."
        )
