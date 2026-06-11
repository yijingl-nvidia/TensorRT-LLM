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

#include "tensorrt_llm/common/quantization.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"
#include "tensorrt_llm/runtime/torchUtils.h"
#include "tensorrt_llm/thop/attentionOp.h"

#include <torch/extension.h>

#include <optional>
#include <utility>

namespace th = torch;

namespace dsv3_fused_mla
{
torch::Tensor dsv3_fused_mla_context_cuda(torch::Tensor fused_q, torch::Tensor q_pe, torch::Tensor latent_cache,
    torch::Tensor topk_indices, torch::Tensor rotary_cos_sin, torch::Tensor ctx_cached_token_indptr,
    torch::Tensor ctx_kv_indptr, tensorrt_llm::kernels::KVBlockArray kv_cache,
    std::optional<torch::Tensor> kv_scale_orig_quant, bool has_fp8_kv_cache, double q_scaling);
} // namespace dsv3_fused_mla

namespace tensorrt_llm
{
namespace torch_ext
{

th::Tensor dsv3_fused_mla_context(th::Tensor fused_q, th::Tensor q_pe, th::Tensor latent_cache, th::Tensor topk_indices,
    th::Tensor rotary_cos_sin, th::Tensor ctx_cached_token_indptr, th::Tensor ctx_kv_indptr,
    th::Tensor kv_cache_block_offsets, th::Tensor host_kv_cache_pool_pointers, th::Tensor host_kv_cache_pool_mapping,
    std::optional<th::Tensor> kv_scale_orig_quant, int64_t layer_idx, int64_t tokens_per_block, int64_t quant_mode,
    double q_scaling)
{
    TORCH_CHECK(fused_q.is_cuda(), "fused_q must be a CUDA tensor");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be a CUDA tensor");
    TORCH_CHECK(latent_cache.is_cuda(), "latent_cache must be a CUDA tensor");
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(rotary_cos_sin.is_cuda(), "rotary_cos_sin must be a CUDA tensor");
    TORCH_CHECK(ctx_cached_token_indptr.is_cuda(), "ctx_cached_token_indptr must be a CUDA tensor");
    TORCH_CHECK(ctx_kv_indptr.is_cuda(), "ctx_kv_indptr must be a CUDA tensor");
    TORCH_CHECK(kv_cache_block_offsets.is_cuda(), "kv_cache_block_offsets must be a CUDA tensor");

    TORCH_CHECK(fused_q.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA context currently expects bf16 fused_q");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA context currently expects bf16 q_pe");
    TORCH_CHECK(
        latent_cache.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA context currently expects bf16 latent_cache");
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32, "topk_indices must be int32");
    TORCH_CHECK(ctx_cached_token_indptr.scalar_type() == torch::kInt64, "ctx_cached_token_indptr must be int64");
    TORCH_CHECK(ctx_kv_indptr.scalar_type() == torch::kInt64, "ctx_kv_indptr must be int64");

    TORCH_CHECK(fused_q.dim() == 3, "fused_q must have shape [tokens, heads, 576]");
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must have shape [tokens, heads, 64]");
    TORCH_CHECK(latent_cache.dim() == 2, "latent_cache must have shape [tokens, 576]");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must have shape [tokens, topk]");
    TORCH_CHECK(fused_q.size(0) == q_pe.size(0), "fused_q and q_pe token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == latent_cache.size(0), "fused_q and latent_cache token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == topk_indices.size(0), "fused_q and topk_indices token dimensions must match");
    TORCH_CHECK(fused_q.size(1) == 8, "GLM-5 TP=8 fused MLA context expects 8 local heads");
    TORCH_CHECK(q_pe.size(1) == 8, "GLM-5 TP=8 fused MLA context expects 8 q_pe heads");
    TORCH_CHECK(fused_q.size(2) == 576, "GLM-5 fused MLA context expects fused head size 576");
    TORCH_CHECK(q_pe.size(2) == 64, "GLM-5 fused MLA context expects q_pe head size 64");
    TORCH_CHECK(latent_cache.size(1) == 576, "GLM-5 fused MLA context expects latent cache head size 576");
    TORCH_CHECK(fused_q.stride(2) == 1, "fused_q last dimension must be contiguous");
    TORCH_CHECK(q_pe.stride(2) == 1, "q_pe last dimension must be contiguous");
    TORCH_CHECK(latent_cache.stride(1) == 1, "latent_cache last dimension must be contiguous");
    TORCH_CHECK(topk_indices.stride(1) == 1, "topk_indices last dimension must be contiguous");
    TORCH_CHECK(topk_indices.size(1) <= 4096, "topk_indices top-k dimension is larger than the dev kernel supports");

    auto const quantMode = common::QuantMode{static_cast<uint32_t>(quant_mode)};
    auto const maxAttentionWindow = static_cast<int64_t>(kv_cache_block_offsets.size(-1)) * tokens_per_block;
    auto kvCacheBuffers = buildPagedKvCacheBuffers(std::optional<torch::Tensor>(kv_cache_block_offsets),
        std::optional<torch::Tensor>(host_kv_cache_pool_pointers),
        std::optional<torch::Tensor>(host_kv_cache_pool_mapping), quantMode, layer_idx, /*batch_size=*/1,
        tokens_per_block, /*kv_head_num=*/1, /*size_per_head=*/576,
        /*cyclic_attention_window_size=*/maxAttentionWindow, /*max_attention_window_size=*/maxAttentionWindow,
        /*sink_token_length=*/0, /*beam_width=*/1, /*seq_offset=*/0, /*is_mla_enable=*/true, fused_q.element_size());
    TORCH_CHECK(kvCacheBuffers.kvCacheBuffer.data != nullptr, "KV cache block offsets are required");

    return dsv3_fused_mla::dsv3_fused_mla_context_cuda(std::move(fused_q), std::move(q_pe), std::move(latent_cache),
        std::move(topk_indices), std::move(rotary_cos_sin), std::move(ctx_cached_token_indptr),
        std::move(ctx_kv_indptr), kvCacheBuffers.kvCacheBuffer, std::move(kv_scale_orig_quant),
        quantMode.hasFp8KvCache(), q_scaling);
}

} // namespace torch_ext
} // namespace tensorrt_llm

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsv3_fused_mla_context(Tensor(a!) fused_q, Tensor q_pe, Tensor(b!) latent_cache, Tensor topk_indices, "
        "Tensor rotary_cos_sin, Tensor ctx_cached_token_indptr, Tensor ctx_kv_indptr, Tensor kv_cache_block_offsets, "
        "Tensor host_kv_cache_pool_pointers, Tensor host_kv_cache_pool_mapping, Tensor? kv_scale_orig_quant, "
        "int layer_idx, int tokens_per_block, int quant_mode, float q_scaling) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_mla_context", &tensorrt_llm::torch_ext::dsv3_fused_mla_context);
}
