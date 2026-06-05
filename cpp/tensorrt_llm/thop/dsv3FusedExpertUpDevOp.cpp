/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Dev-only DeepSeek-V3 fused expert-up op wrapper for standalone builds.
//
// This file intentionally registers a unique op name so the in-development
// shared object can be loaded beside libth_common.so without duplicate schema
// registration.

#include <torch/torch.h>

#include <tuple>
#include <utility>

namespace th = torch;

namespace dsv3_fused_expert
{
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> dsv3_fused_expert_up_dev_cuda(torch::Tensor scores,
    torch::Tensor hidden_in, torch::Tensor bias, torch::Tensor expert_gate_up_weight,
    torch::Tensor expert_gate_up_scale, double routed_scaling_factor);
} // namespace dsv3_fused_expert

namespace tensorrt_llm
{

namespace torch_ext
{

std::tuple<th::Tensor, th::Tensor, th::Tensor> dsv3_fused_expert_up_dev(th::Tensor scores, th::Tensor hidden_in,
    th::Tensor bias, th::Tensor expert_gate_up_weight, th::Tensor expert_gate_up_scale, double routed_scaling_factor)
{
    return dsv3_fused_expert::dsv3_fused_expert_up_dev_cuda(std::move(scores), std::move(hidden_in), std::move(bias),
        std::move(expert_gate_up_weight), std::move(expert_gate_up_scale), routed_scaling_factor);
}

} // namespace torch_ext

} // namespace tensorrt_llm

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsv3_fused_expert_up_dev(Tensor scores, Tensor hidden_in, Tensor bias, "
        "Tensor expert_gate_up_weight, Tensor expert_gate_up_scale, float routed_scaling_factor) -> "
        "(Tensor topk_values, Tensor topk_indices, Tensor hidden_out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_expert_up_dev", &tensorrt_llm::torch_ext::dsv3_fused_expert_up_dev);
}
