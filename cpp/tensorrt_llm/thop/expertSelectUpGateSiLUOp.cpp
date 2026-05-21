/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// GLM-5 ExpertSelectUpGateSiLU mega-kernel op wrapper (v68).
//
// Forwards to the v68 host launchers in
// tensorrt_llm/kernels/glm5SmallBatch/expert_select_up_gate_silu/mega_kernel_v68.cu
// without modification. v68 is hard-specialized for the GLM-5 deploy config
// (M up to 16, 256 routed experts + 1 shared expert, hidden=6144, fp8 e4m3
// weights with per-K-group pre-folded scales). See the mega_kernel_v68.cu
// header for the full kKTile=768 saturation-test design notes.

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/torch.h>

#include <tuple>

namespace th = torch;
namespace tl = tensorrt_llm;

// Forward declarations of the v68 host launchers defined in the .cu file.
namespace mega_kernel
{
std::tuple<torch::Tensor, torch::Tensor> repack_weights_v68(torch::Tensor w_fp8, torch::Tensor w_scale);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mega_silu_v68(torch::Tensor scores, torch::Tensor hidden_in,
    torch::Tensor bias, torch::Tensor w_gate_packed, torch::Tensor w_up_packed, torch::Tensor group_max_scale_gate,
    torch::Tensor group_max_scale_up, double routed_scaling_factor);
} // namespace mega_kernel

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Thin TRT-LLM-named wrapper around mega_kernel::repack_weights_v68.
// Returns (packed_weights [E*kPackedSlabsPerExpert, 49152] u8,
//          group_max_scale [E, 4, 8] fp32).
std::tuple<th::Tensor, th::Tensor> glm5_repack_weights_up_gate_silu(th::Tensor w_fp8, th::Tensor w_scale)
{
    return mega_kernel::repack_weights_v68(std::move(w_fp8), std::move(w_scale));
}

// Thin TRT-LLM-named wrapper around mega_kernel::mega_silu_v68.
// Returns (topk_values, topk_indices, hidden_out_bf16).
std::tuple<th::Tensor, th::Tensor, th::Tensor> glm5_expert_select_up_gate_silu(th::Tensor scores, th::Tensor hidden_in,
    th::Tensor bias, th::Tensor w_gate_packed, th::Tensor w_up_packed, th::Tensor group_max_scale_gate,
    th::Tensor group_max_scale_up, double routed_scaling_factor)
{
    return mega_kernel::mega_silu_v68(std::move(scores), std::move(hidden_in), std::move(bias),
        std::move(w_gate_packed), std::move(w_up_packed), std::move(group_max_scale_gate),
        std::move(group_max_scale_up), routed_scaling_factor);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm5_repack_weights_up_gate_silu(Tensor w_fp8, Tensor w_scale) -> (Tensor packed, Tensor group_max_scale)");
    m.def(
        "glm5_expert_select_up_gate_silu(Tensor scores, Tensor hidden_in, Tensor bias, Tensor w_gate_packed, Tensor "
        "w_up_packed, Tensor group_max_scale_gate, Tensor group_max_scale_up, float routed_scaling_factor) -> "
        "(Tensor topk_values, Tensor topk_indices, Tensor hidden_out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm5_repack_weights_up_gate_silu", &tensorrt_llm::torch_ext::glm5_repack_weights_up_gate_silu);
    m.impl("glm5_expert_select_up_gate_silu", &tensorrt_llm::torch_ext::glm5_expert_select_up_gate_silu);
}
