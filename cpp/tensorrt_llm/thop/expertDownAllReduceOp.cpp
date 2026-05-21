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

// GLM-5 ExpertDownAllReduce mega-kernel op wrapper (v110).
//
// Forwards to the v110 host launcher in
// tensorrt_llm/kernels/glm5SmallBatch/expert_down_allreduce/mega_kernel_down_v110.cu
// without modification. v110 is hard-specialized for the GLM-5 small-batch
// deploy: M=4, kFp8Stages=4, and supports two TP configurations:
//   * TP=4: K_local=512, num_peers=4 (peer_ptr4..7 must be 0)
//   * TP=8: K_local=256, num_peers=8 (all peer_ptr0..7 must be valid)
// It will TORCH_CHECK at the entry if (K_local, num_peers) is off-spec. The
// peer_ptr* / flag arguments come from a Lamport symmetric-heap allocator
// on the Python side; this wrapper only forwards them through as int64.

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <torch/torch.h>

// Forward declaration of the v110 host launcher defined in the .cu file.
torch::Tensor mega_down_v110(torch::Tensor hidden_in, torch::Tensor indices, torch::Tensor scores,
    torch::Tensor residual, torch::Tensor w_down_packed, torch::Tensor w_down_group_scale,
    bool add_residual_on_rank0_only, int64_t rank, int64_t peer_ptr0, int64_t peer_ptr1, int64_t peer_ptr2,
    int64_t peer_ptr3, int64_t peer_ptr4, int64_t peer_ptr5, int64_t peer_ptr6, int64_t peer_ptr7, int64_t num_peers,
    int64_t flag);

std::tuple<torch::Tensor, torch::Tensor> repack_weights_v110(torch::Tensor w_down_fp8, torch::Tensor w_down_scale);

namespace th = torch;
namespace tl = tensorrt_llm;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Thin TRT-LLM-named wrapper around repack_weights_v110.
// Returns (w_packed_fp8, group_max_scale_fp32). Same algorithm as v85/v93/v103,
// but k_blocks_per_group is now derived from K_local (4 for TP=4, 2 for TP=8).
std::tuple<th::Tensor, th::Tensor> glm5_repack_weights_down(th::Tensor w_down_fp8, th::Tensor w_down_scale)
{
    return repack_weights_v110(std::move(w_down_fp8), std::move(w_down_scale));
}

// Thin TRT-LLM-named wrapper around mega_down_v110.
// Supports TP=4 (num_peers=4, K_local=512) and TP=8 (num_peers=8, K_local=256).
// At TP=4 the trailing peer_ptr4..7 should be passed as 0 and are ignored by
// the kernel (peer_bufs[4..7] are never dereferenced because num_peers=4 caps
// the peer loops). Peer pointers + Lamport flag are plumbed through unchanged;
// allocation/lifecycle is the caller's responsibility. Returns hidden_out_bf16
// [M, kHiddenSize].
th::Tensor glm5_expert_down_allreduce(th::Tensor hidden_in, th::Tensor indices, th::Tensor scores, th::Tensor residual,
    th::Tensor w_down_packed, th::Tensor w_down_group_scale, bool add_residual_on_rank0_only, int64_t rank,
    int64_t peer_ptr0, int64_t peer_ptr1, int64_t peer_ptr2, int64_t peer_ptr3, int64_t peer_ptr4, int64_t peer_ptr5,
    int64_t peer_ptr6, int64_t peer_ptr7, int64_t num_peers, int64_t flag)
{
    return mega_down_v110(std::move(hidden_in), std::move(indices), std::move(scores), std::move(residual),
        std::move(w_down_packed), std::move(w_down_group_scale), add_residual_on_rank0_only, rank, peer_ptr0,
        peer_ptr1, peer_ptr2, peer_ptr3, peer_ptr4, peer_ptr5, peer_ptr6, peer_ptr7, num_peers, flag);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("glm5_repack_weights_down(Tensor w_down_fp8, Tensor w_down_scale) -> (Tensor packed, Tensor group_max_scale)");
    m.def(
        "glm5_expert_down_allreduce(Tensor hidden_in, Tensor indices, Tensor scores, Tensor residual, Tensor "
        "w_down_packed, Tensor w_down_group_scale, bool add_residual_on_rank0_only, int rank, int peer_ptr0, int "
        "peer_ptr1, int peer_ptr2, int peer_ptr3, int peer_ptr4, int peer_ptr5, int peer_ptr6, int peer_ptr7, int "
        "num_peers, int flag) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm5_repack_weights_down", &tensorrt_llm::torch_ext::glm5_repack_weights_down);
    m.impl("glm5_expert_down_allreduce", &tensorrt_llm::torch_ext::glm5_expert_down_allreduce);
}
