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

// Dev-only DeepSeek-V3 fused expert-down op wrapper for standalone builds.
//
// This file intentionally registers unique op names so the in-development
// shared object can be loaded beside libth_common.so without duplicate schema
// registration.

#include <torch/torch.h>

torch::Tensor dsv3_fused_expert_down_cuda_dev(torch::Tensor hidden_in, torch::Tensor indices, torch::Tensor scores,
    torch::Tensor routed_w_down, torch::Tensor routed_w_down_scale, torch::Tensor shared_w_down,
    torch::Tensor shared_w_down_scale, torch::Tensor output);

torch::Tensor dsv3_fused_expert_down_ar_residual_cuda_dev(torch::Tensor hidden_in, torch::Tensor indices,
    torch::Tensor scores, torch::Tensor routed_w_down, torch::Tensor routed_w_down_scale, torch::Tensor shared_w_down,
    torch::Tensor shared_w_down_scale, torch::Tensor residual, torch::Tensor workspace, int64_t rank, int64_t nranks,
    torch::Tensor local_output, torch::Tensor residual_out);

torch::Tensor dsv3_fused_expert_down_ar_residual_rms_norm_cuda_dev(torch::Tensor hidden_in, torch::Tensor indices,
    torch::Tensor scores, torch::Tensor routed_w_down, torch::Tensor routed_w_down_scale, torch::Tensor shared_w_down,
    torch::Tensor shared_w_down_scale, torch::Tensor residual, torch::Tensor norm_weight, torch::Tensor workspace,
    int64_t rank, int64_t nranks, double rms_norm_eps, torch::Tensor local_output, torch::Tensor residual_out,
    torch::Tensor hidden_out, torch::Tensor rms_sums);

namespace th = torch;

namespace tensorrt_llm
{

namespace torch_ext
{

th::Tensor dsv3_fused_expert_down_dev(th::Tensor hidden_in, th::Tensor indices, th::Tensor scores,
    th::Tensor routed_w_down, th::Tensor routed_w_down_scale, th::Tensor shared_w_down, th::Tensor shared_w_down_scale,
    th::Tensor output)
{
    return dsv3_fused_expert_down_cuda_dev(std::move(hidden_in), std::move(indices), std::move(scores),
        std::move(routed_w_down), std::move(routed_w_down_scale), std::move(shared_w_down),
        std::move(shared_w_down_scale), std::move(output));
}

th::Tensor dsv3_fused_expert_down_ar_residual_dev(th::Tensor hidden_in, th::Tensor indices, th::Tensor scores,
    th::Tensor routed_w_down, th::Tensor routed_w_down_scale, th::Tensor shared_w_down, th::Tensor shared_w_down_scale,
    th::Tensor residual, th::Tensor workspace, int64_t rank, int64_t nranks, th::Tensor local_output,
    th::Tensor residual_out)
{
    return dsv3_fused_expert_down_ar_residual_cuda_dev(std::move(hidden_in), std::move(indices), std::move(scores),
        std::move(routed_w_down), std::move(routed_w_down_scale), std::move(shared_w_down),
        std::move(shared_w_down_scale), std::move(residual), std::move(workspace), rank, nranks,
        std::move(local_output), std::move(residual_out));
}

th::Tensor dsv3_fused_expert_down_ar_residual_rms_norm_dev(th::Tensor hidden_in, th::Tensor indices, th::Tensor scores,
    th::Tensor routed_w_down, th::Tensor routed_w_down_scale, th::Tensor shared_w_down, th::Tensor shared_w_down_scale,
    th::Tensor residual, th::Tensor norm_weight, th::Tensor workspace, int64_t rank, int64_t nranks,
    double rms_norm_eps, th::Tensor local_output, th::Tensor residual_out, th::Tensor hidden_out, th::Tensor rms_sums)
{
    return dsv3_fused_expert_down_ar_residual_rms_norm_cuda_dev(std::move(hidden_in), std::move(indices),
        std::move(scores), std::move(routed_w_down), std::move(routed_w_down_scale), std::move(shared_w_down),
        std::move(shared_w_down_scale), std::move(residual), std::move(norm_weight), std::move(workspace), rank, nranks,
        rms_norm_eps, std::move(local_output), std::move(residual_out), std::move(hidden_out), std::move(rms_sums));
}

} // namespace torch_ext

} // namespace tensorrt_llm

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsv3_fused_expert_down_dev(Tensor hidden_in, Tensor indices, Tensor scores, Tensor routed_w_down, Tensor "
        "routed_w_down_scale, Tensor shared_down_weight, Tensor shared_down_weight_scale, Tensor output) -> Tensor");
    m.def(
        "dsv3_fused_expert_down_ar_residual_dev(Tensor hidden_in, Tensor indices, Tensor scores, Tensor routed_w_down, "
        "Tensor routed_w_down_scale, Tensor shared_down_weight, Tensor shared_down_weight_scale, Tensor residual, "
        "Tensor workspace, int rank, int nranks, Tensor local_output, Tensor residual_out) -> Tensor");
    m.def(
        "dsv3_fused_expert_down_ar_residual_rms_norm_dev(Tensor hidden_in, Tensor indices, Tensor scores, "
        "Tensor routed_w_down, Tensor routed_w_down_scale, Tensor shared_down_weight, Tensor shared_down_weight_scale, "
        "Tensor residual, Tensor norm_weight, Tensor workspace, int rank, int nranks, float rms_norm_eps, "
        "Tensor local_output, Tensor residual_out, Tensor hidden_out, Tensor rms_sums) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_expert_down_dev", &tensorrt_llm::torch_ext::dsv3_fused_expert_down_dev);
    m.impl("dsv3_fused_expert_down_ar_residual_dev", &tensorrt_llm::torch_ext::dsv3_fused_expert_down_ar_residual_dev);
    m.impl("dsv3_fused_expert_down_ar_residual_rms_norm_dev",
        &tensorrt_llm::torch_ext::dsv3_fused_expert_down_ar_residual_rms_norm_dev);
}
