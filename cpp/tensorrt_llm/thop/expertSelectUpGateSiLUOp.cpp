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

// GLM-5 ExpertSelectUpGateSiLU mega-kernel op wrapper (v68 integrated).

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/torch.h>

#include <tuple>
#include <utility>

namespace th = torch;

namespace mega_kernel
{
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mega_silu_v68_integrated(torch::Tensor scores,
    torch::Tensor hidden_in, torch::Tensor bias, torch::Tensor shared_gate_up_weight,
    torch::Tensor shared_gate_up_scale, torch::Tensor routed_w3_w1_weight, torch::Tensor routed_w3_w1_scale,
    double routed_scaling_factor);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mega_silu_v68_packed_integrated(torch::Tensor scores,
    torch::Tensor hidden_in, torch::Tensor bias, torch::Tensor shared_gate_up_weight,
    torch::Tensor shared_gate_up_scale, torch::Tensor routed_w3_w1_weight, torch::Tensor routed_w3_w1_scale,
    double routed_scaling_factor);
} // namespace mega_kernel

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<th::Tensor, th::Tensor, th::Tensor> glm5_expert_select_up_gate_silu(th::Tensor scores, th::Tensor hidden_in,
    th::Tensor bias, th::Tensor shared_gate_up_weight, th::Tensor shared_gate_up_scale, th::Tensor routed_w3_w1_weight,
    th::Tensor routed_w3_w1_scale, double routed_scaling_factor)
{
    return mega_kernel::mega_silu_v68_integrated(std::move(scores), std::move(hidden_in), std::move(bias),
        std::move(shared_gate_up_weight), std::move(shared_gate_up_scale), std::move(routed_w3_w1_weight),
        std::move(routed_w3_w1_scale), routed_scaling_factor);
}

std::tuple<th::Tensor, th::Tensor, th::Tensor> glm5_expert_select_up_gate_silu_packed(th::Tensor scores,
    th::Tensor hidden_in, th::Tensor bias, th::Tensor shared_gate_up_weight, th::Tensor shared_gate_up_scale,
    th::Tensor routed_w3_w1_weight, th::Tensor routed_w3_w1_scale, double routed_scaling_factor)
{
    return mega_kernel::mega_silu_v68_packed_integrated(std::move(scores), std::move(hidden_in), std::move(bias),
        std::move(shared_gate_up_weight), std::move(shared_gate_up_scale), std::move(routed_w3_w1_weight),
        std::move(routed_w3_w1_scale), routed_scaling_factor);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm5_expert_select_up_gate_silu(Tensor scores, Tensor hidden_in, Tensor bias, Tensor shared_gate_up_weight, "
        "Tensor shared_gate_up_scale, Tensor routed_w3_w1_weight, Tensor routed_w3_w1_scale, "
        "float routed_scaling_factor) -> "
        "(Tensor topk_values, Tensor topk_indices, Tensor hidden_out)");
    m.def(
        "glm5_expert_select_up_gate_silu_packed(Tensor scores, Tensor hidden_in, Tensor bias, "
        "Tensor shared_gate_up_weight, Tensor shared_gate_up_scale, Tensor routed_w3_w1_weight, "
        "Tensor routed_w3_w1_scale, float routed_scaling_factor) -> "
        "(Tensor topk_values, Tensor topk_indices, Tensor hidden_out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm5_expert_select_up_gate_silu", &tensorrt_llm::torch_ext::glm5_expert_select_up_gate_silu);
    m.impl("glm5_expert_select_up_gate_silu_packed", &tensorrt_llm::torch_ext::glm5_expert_select_up_gate_silu_packed);
}
