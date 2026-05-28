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

// GLM-5 ExpertDownProject mega-kernel op wrapper (v110).
//
// Forwards to the v110 host launcher in
// tensorrt_llm/kernels/glm5SmallBatch/expert_down_allreduce/mega_kernel_down_v110.cu
// after removing the branch kernel's residual add and peer allreduce. v110 is
// hard-specialized for the GLM-5 small-batch deploy and supports two TP
// configurations:
//   * TP=4: K_local=512
//   * TP=8: K_local=256

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/torch.h>

// Forward declaration of the v110 host launcher defined in the .cu file.
torch::Tensor mega_down_project_v110(torch::Tensor hidden_in, torch::Tensor indices, torch::Tensor scores,
    torch::Tensor routed_w_down, torch::Tensor routed_w_down_scale, torch::Tensor shared_w_down,
    torch::Tensor shared_w_down_scale, torch::Tensor output);

namespace th = torch;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Thin TRT-LLM-named wrapper around mega_down_project_v110.
// Returns hidden_out_bf16 [M, kHiddenSize] in the caller-provided output tensor.
th::Tensor glm5_expert_down_project(th::Tensor hidden_in, th::Tensor indices, th::Tensor scores,
    th::Tensor routed_w_down, th::Tensor routed_w_down_scale, th::Tensor shared_w_down, th::Tensor shared_w_down_scale,
    th::Tensor output)
{
    return mega_down_project_v110(std::move(hidden_in), std::move(indices), std::move(scores), std::move(routed_w_down),
        std::move(routed_w_down_scale), std::move(shared_w_down), std::move(shared_w_down_scale), std::move(output));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm5_expert_down_project(Tensor hidden_in, Tensor indices, Tensor scores, Tensor routed_w_down, Tensor "
        "routed_w_down_scale, Tensor shared_w_down, Tensor shared_w_down_scale, Tensor output) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm5_expert_down_project", &tensorrt_llm::torch_ext::glm5_expert_down_project);
}
