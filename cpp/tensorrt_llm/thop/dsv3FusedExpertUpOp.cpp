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

// DeepSeek-V3 fused expert-up op wrapper.

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/torch.h>

#include <tuple>
#include <utility>

namespace th = torch;

namespace dsv3_fused_expert
{
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dsv3_fused_expert_up(torch::Tensor scores,
    torch::Tensor hidden_in, torch::Tensor bias, torch::Tensor expert_gate_up_weight,
    torch::Tensor expert_gate_up_scale, double routed_scaling_factor);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dsv3_fused_expert_up_fp16_mma(torch::Tensor scores,
    torch::Tensor hidden_in, torch::Tensor bias, torch::Tensor expert_gate_up_weight,
    torch::Tensor expert_gate_up_scale, double routed_scaling_factor);
} // namespace dsv3_fused_expert

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

std::tuple<th::Tensor, th::Tensor, th::Tensor> dsv3_fused_expert_up(th::Tensor scores, th::Tensor hidden_in,
    th::Tensor bias, th::Tensor expert_gate_up_weight, th::Tensor expert_gate_up_scale, double routed_scaling_factor)
{
    return dsv3_fused_expert::dsv3_fused_expert_up(std::move(scores), std::move(hidden_in), std::move(bias),
        std::move(expert_gate_up_weight), std::move(expert_gate_up_scale), routed_scaling_factor);
}

std::tuple<th::Tensor, th::Tensor, th::Tensor> dsv3_fused_expert_up_fp16_mma(th::Tensor scores, th::Tensor hidden_in,
    th::Tensor bias, th::Tensor expert_gate_up_weight, th::Tensor expert_gate_up_scale, double routed_scaling_factor)
{
    return dsv3_fused_expert::dsv3_fused_expert_up_fp16_mma(std::move(scores), std::move(hidden_in), std::move(bias),
        std::move(expert_gate_up_weight), std::move(expert_gate_up_scale), routed_scaling_factor);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsv3_fused_expert_up(Tensor scores, Tensor hidden_in, Tensor bias, "
        "Tensor expert_gate_up_weight, Tensor expert_gate_up_scale, float routed_scaling_factor) -> "
        "(Tensor topk_values, Tensor topk_indices, Tensor hidden_out)");
    m.def(
        "dsv3_fused_expert_up_fp16_mma(Tensor scores, Tensor hidden_in, Tensor bias, "
        "Tensor expert_gate_up_weight, Tensor expert_gate_up_scale, float routed_scaling_factor) -> "
        "(Tensor topk_values, Tensor topk_indices, Tensor hidden_out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_expert_up", &tensorrt_llm::torch_ext::dsv3_fused_expert_up);
    m.impl("dsv3_fused_expert_up_fp16_mma", &tensorrt_llm::torch_ext::dsv3_fused_expert_up_fp16_mma);
}
