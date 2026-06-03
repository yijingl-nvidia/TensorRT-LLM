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

from __future__ import annotations

import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

if IS_CUTLASS_DSL_AVAILABLE:
    try:
        import cuda.bindings.driver as cuda
    except ImportError:
        from cuda import cuda

    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

_FUSED_EXPERT_UP_HIDDEN_SIZE = 6144
_FUSED_EXPERT_UP_CTA_OUT_ROWS = 64
_FUSED_EXPERT_UP_NUM_K_ITER = 8
_FUSED_EXPERT_UP_TILE_BYTES = 49152
_FUSED_EXPERT_UP_NUM_THREADS = 256

_pack_compile_cache: dict[tuple[object, ...], object] = {}


class _CUDAGraphCompatibleWrapper:
    """Wrapper that lets CuTe DSL consume tensors without forcing a stream sync."""

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __dlpack__(self, stream=None):
        return self._tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self._tensor.__dlpack_device__()


def _validate_shared_gate_up_weight(weight: torch.Tensor) -> int:
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"fused expert up pack expects float8_e4m3fn weights, got {weight.dtype}")
    if weight.ndim != 2:
        raise ValueError(f"shared fused expert up pack expects rank-2 weight, got {weight.shape}")
    if weight.shape[1] != _FUSED_EXPERT_UP_HIDDEN_SIZE:
        raise ValueError(
            f"shared fused expert up pack expects hidden size {_FUSED_EXPERT_UP_HIDDEN_SIZE}, got {weight.shape[1]}"
        )
    if weight.shape[0] % 2 != 0:
        raise ValueError(f"shared fused expert up pack expects [2 * I, H], got {weight.shape}")
    inter_per_tp = weight.shape[0] // 2
    if inter_per_tp % _FUSED_EXPERT_UP_CTA_OUT_ROWS != 0:
        raise ValueError(
            "shared fused expert up pack expects I to be divisible by "
            f"{_FUSED_EXPERT_UP_CTA_OUT_ROWS}, got {inter_per_tp}"
        )
    return inter_per_tp


def _validate_routed_w3_w1_weight(weight: torch.Tensor) -> tuple[int, int]:
    if weight.dtype != torch.float8_e4m3fn:
        raise TypeError(f"fused expert up pack expects float8_e4m3fn weights, got {weight.dtype}")
    if weight.ndim != 3:
        raise ValueError(f"routed fused expert up pack expects rank-3 weight, got {weight.shape}")
    if weight.shape[2] != _FUSED_EXPERT_UP_HIDDEN_SIZE:
        raise ValueError(
            f"routed fused expert up pack expects hidden size {_FUSED_EXPERT_UP_HIDDEN_SIZE}, got {weight.shape[2]}"
        )
    if weight.shape[1] % 2 != 0:
        raise ValueError(f"routed fused expert up pack expects [E, 2 * I, H], got {weight.shape}")
    inter_per_tp = weight.shape[1] // 2
    if inter_per_tp % _FUSED_EXPERT_UP_CTA_OUT_ROWS != 0:
        raise ValueError(
            "routed fused expert up pack expects I to be divisible by "
            f"{_FUSED_EXPERT_UP_CTA_OUT_ROWS}, got {inter_per_tp}"
        )
    return weight.shape[0], inter_per_tp


if IS_CUTLASS_DSL_AVAILABLE:

    class _PackFusedExpertUpSharedGateUpKernel:
        def __init__(self, inter_per_tp: int):
            self.inter_per_tp = inter_per_tp
            self.sub_rows = inter_per_tp // _FUSED_EXPERT_UP_CTA_OUT_ROWS

        @cute.kernel
        def pack_kernel(self, raw: cute.Tensor, packed: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            tile_idx, _, _ = cute.arch.block_idx()

            k_iter = tile_idx % _FUSED_EXPERT_UP_NUM_K_ITER
            sub_row = (tile_idx // _FUSED_EXPERT_UP_NUM_K_ITER) % self.sub_rows
            side = tile_idx // (self.sub_rows * _FUSED_EXPERT_UP_NUM_K_ITER)

            for elem in range(tidx, _FUSED_EXPERT_UP_TILE_BYTES, _FUSED_EXPERT_UP_NUM_THREADS):
                byte = elem % 4
                row_half = (elem // 4) % 2
                col_half = (elem // 8) % 2
                col_quad = (elem // 16) % 4
                row = (elem // 64) % 8
                k_sub = (elem // 512) % 4
                m_tile = (elem // 2048) % 4
                k_third = elem // 8192

                src_row = side * self.inter_per_tp + sub_row * 64 + m_tile * 16 + row_half * 8 + row
                src_col = (
                    k_iter * 768 + k_third * 128 + k_sub * 32 + col_half * 16 + col_quad * 4 + byte
                )
                src_offset = src_row * _FUSED_EXPERT_UP_HIDDEN_SIZE + src_col
                dst_offset = tile_idx * _FUSED_EXPERT_UP_TILE_BYTES + elem
                packed[dst_offset] = raw[src_offset]

        @cute.jit
        def __call__(self, raw: cute.Tensor, packed: cute.Tensor, stream: cuda.CUstream):
            self.pack_kernel(raw, packed).launch(
                grid=(2 * self.sub_rows * _FUSED_EXPERT_UP_NUM_K_ITER, 1, 1),
                block=(_FUSED_EXPERT_UP_NUM_THREADS, 1, 1),
                stream=stream,
            )

    class _PackFusedExpertUpRoutedW3W1Kernel:
        def __init__(self, num_experts: int, inter_per_tp: int):
            self.num_experts = num_experts
            self.inter_per_tp = inter_per_tp
            self.sub_rows = inter_per_tp // _FUSED_EXPERT_UP_CTA_OUT_ROWS

        @cute.kernel
        def pack_kernel(self, raw: cute.Tensor, packed: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            tile_idx, _, _ = cute.arch.block_idx()

            k_iter = tile_idx % _FUSED_EXPERT_UP_NUM_K_ITER
            sub_row = (tile_idx // _FUSED_EXPERT_UP_NUM_K_ITER) % self.sub_rows
            side = (tile_idx // (self.sub_rows * _FUSED_EXPERT_UP_NUM_K_ITER)) % 2
            expert_idx = tile_idx // (2 * self.sub_rows * _FUSED_EXPERT_UP_NUM_K_ITER)

            for elem in range(tidx, _FUSED_EXPERT_UP_TILE_BYTES, _FUSED_EXPERT_UP_NUM_THREADS):
                byte = elem % 4
                row_half = (elem // 4) % 2
                col_half = (elem // 8) % 2
                col_quad = (elem // 16) % 4
                row = (elem // 64) % 8
                k_sub = (elem // 512) % 4
                m_tile = (elem // 2048) % 4
                k_third = elem // 8192

                raw_side = 1 - side
                src_row = (
                    raw_side * self.inter_per_tp + sub_row * 64 + m_tile * 16 + row_half * 8 + row
                )
                src_col = (
                    k_iter * 768 + k_third * 128 + k_sub * 32 + col_half * 16 + col_quad * 4 + byte
                )
                src_offset = (
                    expert_idx * 2 * self.inter_per_tp * _FUSED_EXPERT_UP_HIDDEN_SIZE
                    + src_row * _FUSED_EXPERT_UP_HIDDEN_SIZE
                    + src_col
                )
                dst_offset = tile_idx * _FUSED_EXPERT_UP_TILE_BYTES + elem
                packed[dst_offset] = raw[src_offset]

        @cute.jit
        def __call__(self, raw: cute.Tensor, packed: cute.Tensor, stream: cuda.CUstream):
            self.pack_kernel(raw, packed).launch(
                grid=(self.num_experts * 2 * self.sub_rows * _FUSED_EXPERT_UP_NUM_K_ITER, 1, 1),
                block=(_FUSED_EXPERT_UP_NUM_THREADS, 1, 1),
                stream=stream,
            )


def _to_cute_tensor(tensor: torch.Tensor):
    return from_dlpack(_CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=16)


def _compile_pack_kernel(kind: str, raw_cute, packed_cute, *shape_key: int):
    compile_key = (kind, *shape_key)
    compiled = _pack_compile_cache.get(compile_key)
    if compiled is None:
        if kind == "shared":
            kernel = _PackFusedExpertUpSharedGateUpKernel(inter_per_tp=shape_key[0])
        elif kind == "routed":
            kernel = _PackFusedExpertUpRoutedW3W1Kernel(
                num_experts=shape_key[0], inter_per_tp=shape_key[1]
            )
        else:
            raise ValueError(f"unknown fused expert up pack kind: {kind}")
        current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        compiled = cute.compile(kernel, raw_cute, packed_cute, current_stream)
        _pack_compile_cache[compile_key] = compiled
    return compiled


def _run_pack_kernel(
    kind: str, raw_flat: torch.Tensor, packed_flat: torch.Tensor, *shape_key: int
) -> None:
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    raw_cute = _to_cute_tensor(raw_flat)
    packed_cute = _to_cute_tensor(packed_flat)
    compiled = _compile_pack_kernel(kind, raw_cute, packed_cute, *shape_key)
    compiled(raw_cute, packed_cute, current_stream)


def precompile_fused_expert_up_weight_pack_kernels(
    shared_inter_per_tp: int,
    routed_num_experts: int,
    routed_inter_per_tp: int,
    device: torch.device | None = None,
) -> None:
    if not IS_CUTLASS_DSL_AVAILABLE:
        raise ImportError("CUTLASS DSL is not available")
    if shared_inter_per_tp % _FUSED_EXPERT_UP_CTA_OUT_ROWS != 0:
        raise ValueError(
            "shared fused expert up pack expects I to be divisible by "
            f"{_FUSED_EXPERT_UP_CTA_OUT_ROWS}, got {shared_inter_per_tp}"
        )
    if routed_inter_per_tp % _FUSED_EXPERT_UP_CTA_OUT_ROWS != 0:
        raise ValueError(
            "routed fused expert up pack expects I to be divisible by "
            f"{_FUSED_EXPERT_UP_CTA_OUT_ROWS}, got {routed_inter_per_tp}"
        )
    if routed_num_experts <= 0:
        raise ValueError(f"routed_num_experts must be positive, got {routed_num_experts}")

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    shared_key = ("shared", shared_inter_per_tp)
    if shared_key not in _pack_compile_cache:
        shared_sub_rows = shared_inter_per_tp // _FUSED_EXPERT_UP_CTA_OUT_ROWS
        raw = torch.empty(
            (2 * shared_inter_per_tp * _FUSED_EXPERT_UP_HIDDEN_SIZE,),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        packed = torch.empty(
            (2 * shared_sub_rows * _FUSED_EXPERT_UP_NUM_K_ITER * _FUSED_EXPERT_UP_TILE_BYTES,),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        _compile_pack_kernel(
            "shared",
            _to_cute_tensor(raw),
            _to_cute_tensor(packed),
            shared_inter_per_tp,
        )

    routed_key = ("routed", routed_num_experts, routed_inter_per_tp)
    if routed_key not in _pack_compile_cache:
        routed_sub_rows = routed_inter_per_tp // _FUSED_EXPERT_UP_CTA_OUT_ROWS
        raw = torch.empty(
            (routed_num_experts * 2 * routed_inter_per_tp * _FUSED_EXPERT_UP_HIDDEN_SIZE,),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        packed = torch.empty(
            (
                routed_num_experts
                * 2
                * routed_sub_rows
                * _FUSED_EXPERT_UP_NUM_K_ITER
                * _FUSED_EXPERT_UP_TILE_BYTES,
            ),
            device=device,
            dtype=torch.float8_e4m3fn,
        )
        _compile_pack_kernel(
            "routed",
            _to_cute_tensor(raw),
            _to_cute_tensor(packed),
            routed_num_experts,
            routed_inter_per_tp,
        )


def pack_fused_expert_up_shared_gate_up_weight(weight: torch.Tensor) -> torch.Tensor:
    if not IS_CUTLASS_DSL_AVAILABLE:
        raise ImportError("CUTLASS DSL is not available")
    if not weight.is_cuda:
        raise ValueError("CuTe DSL fused expert up pack expects a CUDA tensor")
    inter_per_tp = _validate_shared_gate_up_weight(weight)
    weight = weight.contiguous()
    sub_rows = inter_per_tp // _FUSED_EXPERT_UP_CTA_OUT_ROWS
    packed = torch.empty(
        (2, sub_rows, _FUSED_EXPERT_UP_NUM_K_ITER, _FUSED_EXPERT_UP_TILE_BYTES),
        device=weight.device,
        dtype=weight.dtype,
    )
    _run_pack_kernel("shared", weight.reshape(-1), packed.reshape(-1), inter_per_tp)
    return packed


def pack_fused_expert_up_routed_w3_w1_weight(weight: torch.Tensor) -> torch.Tensor:
    if not IS_CUTLASS_DSL_AVAILABLE:
        raise ImportError("CUTLASS DSL is not available")
    if not weight.is_cuda:
        raise ValueError("CuTe DSL fused expert up pack expects a CUDA tensor")
    num_experts, inter_per_tp = _validate_routed_w3_w1_weight(weight)
    weight = weight.contiguous()
    sub_rows = inter_per_tp // _FUSED_EXPERT_UP_CTA_OUT_ROWS
    packed = torch.empty(
        (num_experts, 2, sub_rows, _FUSED_EXPERT_UP_NUM_K_ITER, _FUSED_EXPERT_UP_TILE_BYTES),
        device=weight.device,
        dtype=weight.dtype,
    )
    _run_pack_kernel("routed", weight.reshape(-1), packed.reshape(-1), num_experts, inter_per_tp)
    return packed
