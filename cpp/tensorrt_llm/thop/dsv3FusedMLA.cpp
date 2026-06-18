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
    torch::Tensor topk_indices_pool, torch::Tensor topk_indices_local, torch::Tensor kv_cache_pool,
    torch::Tensor rotary_cos_sin, torch::Tensor ctx_cached_token_indptr, tensorrt_llm::kernels::KVBlockArray kv_cache,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    bool has_fp8_kv_cache, double q_scaling);
torch::Tensor dsv3_fused_mla_generation_cuda(torch::Tensor fused_q, torch::Tensor q_nope, torch::Tensor k_b_proj_trans,
    torch::Tensor q_pe, torch::Tensor latent_cache, torch::Tensor rotary_cos_sin, torch::Tensor sequence_length,
    tensorrt_llm::kernels::KVBlockArray kv_cache, torch::Tensor topk_indices, torch::Tensor topk_indices_pool,
    torch::Tensor kv_cache_pool, torch::Tensor quant_q_buffer, torch::Tensor mla_bmm1_scale,
    torch::Tensor mla_bmm2_scale, std::optional<torch::Tensor> kv_scale_orig_quant,
    std::optional<torch::Tensor> kv_scale_quant_orig, std::optional<torch::Tensor> spec_decoding_packed_mask,
    double q_scaling, bool is_context, int64_t context_chunk_start, std::optional<torch::Tensor> q_b_proj_input,
    std::optional<torch::Tensor> q_b_proj_weight, std::optional<torch::Tensor> q_b_proj_weight_scale,
    std::optional<torch::Tensor> q_b_proj_output, int64_t q_b_proj_impl);
torch::Tensor dsv3_fused_mla_q_b_proj_cuda(torch::Tensor q_b_proj_input, torch::Tensor q_b_proj_weight,
    torch::Tensor q_b_proj_weight_scale, int64_t q_b_proj_impl);
} // namespace dsv3_fused_mla

namespace tensorrt_llm
{
namespace torch_ext
{

namespace
{

constexpr int64_t kQbProjImplScalar = 0;
constexpr int64_t kQbProjImplMma = 1;
constexpr int64_t kQbProjImplUmma = 2;

void checkQbProjImpl(int64_t q_b_proj_impl)
{
    TORCH_CHECK(
        q_b_proj_impl == kQbProjImplScalar || q_b_proj_impl == kQbProjImplMma || q_b_proj_impl == kQbProjImplUmma,
        "q_b_proj_impl must be 0 (scalar), 1 (mma), or 2 (umma), got ", q_b_proj_impl);
}

} // namespace

th::Tensor dsv3_fused_mla_q_b_proj_impl(
    th::Tensor q_b_proj_input, th::Tensor q_b_proj_weight, th::Tensor q_b_proj_weight_scale, int64_t q_b_proj_impl);

th::Tensor dsv3_fused_mla_context(th::Tensor fused_q, th::Tensor q_pe, th::Tensor latent_cache,
    th::Tensor topk_indices_pool, th::Tensor topk_indices_local, th::Tensor kv_cache_pool, th::Tensor rotary_cos_sin,
    th::Tensor ctx_cached_token_indptr, th::Tensor kv_cache_block_offsets, th::Tensor host_kv_cache_pool_pointers,
    th::Tensor host_kv_cache_pool_mapping, std::optional<th::Tensor> kv_scale_orig_quant,
    std::optional<th::Tensor> kv_scale_quant_orig, int64_t layer_idx, int64_t tokens_per_block, int64_t quant_mode,
    double q_scaling)
{
    TORCH_CHECK(fused_q.is_cuda(), "fused_q must be a CUDA tensor");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be a CUDA tensor");
    TORCH_CHECK(latent_cache.is_cuda(), "latent_cache must be a CUDA tensor");
    TORCH_CHECK(topk_indices_pool.is_cuda(), "topk_indices_pool must be a CUDA tensor");
    TORCH_CHECK(topk_indices_local.is_cuda(), "topk_indices_local must be a CUDA tensor");
    TORCH_CHECK(kv_cache_pool.is_cuda(), "kv_cache_pool must be a CUDA tensor");
    TORCH_CHECK(rotary_cos_sin.is_cuda(), "rotary_cos_sin must be a CUDA tensor");
    TORCH_CHECK(ctx_cached_token_indptr.is_cuda(), "ctx_cached_token_indptr must be a CUDA tensor");
    TORCH_CHECK(kv_cache_block_offsets.is_cuda(), "kv_cache_block_offsets must be a CUDA tensor");

    TORCH_CHECK(fused_q.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA context currently expects bf16 fused_q");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA context currently expects bf16 q_pe");
    TORCH_CHECK(
        latent_cache.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA context currently expects bf16 latent_cache");
    TORCH_CHECK(topk_indices_pool.scalar_type() == torch::kInt32, "topk_indices_pool must be int32");
    TORCH_CHECK(topk_indices_local.scalar_type() == torch::kInt32, "topk_indices_local must be int32");
    TORCH_CHECK(ctx_cached_token_indptr.scalar_type() == torch::kInt64, "ctx_cached_token_indptr must be int64");

    TORCH_CHECK(fused_q.dim() == 3, "fused_q must have shape [tokens, heads, 576]");
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must have shape [tokens, heads, 64]");
    TORCH_CHECK(latent_cache.dim() == 2, "latent_cache must have shape [tokens, 576]");
    TORCH_CHECK(topk_indices_pool.dim() == 2, "topk_indices_pool must have shape [tokens, topk]");
    TORCH_CHECK(topk_indices_local.dim() == 2, "topk_indices_local must have shape [tokens, topk]");
    TORCH_CHECK(kv_cache_pool.dim() == 3, "kv_cache_pool must have shape [pool_tokens, 1, 576]");
    TORCH_CHECK(fused_q.size(0) == q_pe.size(0), "fused_q and q_pe token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == latent_cache.size(0), "fused_q and latent_cache token dimensions must match");
    TORCH_CHECK(
        fused_q.size(0) == topk_indices_pool.size(0), "fused_q and topk_indices_pool token dimensions must match");
    TORCH_CHECK(
        fused_q.size(0) == topk_indices_local.size(0), "fused_q and topk_indices_local token dimensions must match");
    TORCH_CHECK(topk_indices_pool.size(1) == topk_indices_local.size(1),
        "topk_indices_pool and topk_indices_local top-k dimensions must match");
    TORCH_CHECK(fused_q.size(1) == 8, "GLM-5 TP=8 fused MLA context expects 8 local heads");
    TORCH_CHECK(q_pe.size(1) == 8, "GLM-5 TP=8 fused MLA context expects 8 q_pe heads");
    TORCH_CHECK(fused_q.size(2) == 576, "GLM-5 fused MLA context expects fused head size 576");
    TORCH_CHECK(q_pe.size(2) == 64, "GLM-5 fused MLA context expects q_pe head size 64");
    TORCH_CHECK(latent_cache.size(1) == 576, "GLM-5 fused MLA context expects latent cache head size 576");
    TORCH_CHECK(kv_cache_pool.size(1) == 1, "GLM-5 fused MLA context expects one KV head in kv_cache_pool");
    TORCH_CHECK(kv_cache_pool.size(2) == 576, "GLM-5 fused MLA context expects kv_cache_pool head size 576");
    TORCH_CHECK(fused_q.stride(2) == 1, "fused_q last dimension must be contiguous");
    TORCH_CHECK(q_pe.stride(2) == 1, "q_pe last dimension must be contiguous");
    TORCH_CHECK(latent_cache.stride(1) == 1, "latent_cache last dimension must be contiguous");
    TORCH_CHECK(topk_indices_pool.stride(1) == 1, "topk_indices_pool last dimension must be contiguous");
    TORCH_CHECK(topk_indices_local.stride(1) == 1, "topk_indices_local last dimension must be contiguous");
    TORCH_CHECK(kv_cache_pool.stride(2) == 1, "kv_cache_pool last dimension must be contiguous");
    TORCH_CHECK(
        topk_indices_pool.size(1) <= 4096, "topk_indices_pool top-k dimension is larger than the dev kernel supports");

    auto const quantMode = common::QuantMode{static_cast<uint32_t>(quant_mode)};
    TORCH_CHECK(quantMode.hasFp8KvCache(), "GLM-5 fused MLA context currently expects FP8 KV cache");
    TORCH_CHECK(kv_cache_pool.scalar_type() == torch::kFloat8_e4m3fn,
        "GLM-5 fused MLA context currently expects FP8 E4M3 kv_cache_pool");

    auto const maxAttentionWindow = static_cast<int64_t>(kv_cache_block_offsets.size(-1)) * tokens_per_block;
    auto kvCacheBuffers = buildPagedKvCacheBuffers(std::optional<torch::Tensor>(kv_cache_block_offsets),
        std::optional<torch::Tensor>(host_kv_cache_pool_pointers),
        std::optional<torch::Tensor>(host_kv_cache_pool_mapping), quantMode, layer_idx, /*batch_size=*/1,
        tokens_per_block, /*kv_head_num=*/1, /*size_per_head=*/576,
        /*cyclic_attention_window_size=*/maxAttentionWindow, /*max_attention_window_size=*/maxAttentionWindow,
        /*sink_token_length=*/0, /*beam_width=*/1, /*seq_offset=*/0, /*is_mla_enable=*/true, fused_q.element_size());
    TORCH_CHECK(kvCacheBuffers.kvCacheBuffer.data != nullptr, "KV cache block offsets are required");

    return dsv3_fused_mla::dsv3_fused_mla_context_cuda(std::move(fused_q), std::move(q_pe), std::move(latent_cache),
        std::move(topk_indices_pool), std::move(topk_indices_local), std::move(kv_cache_pool),
        std::move(rotary_cos_sin), std::move(ctx_cached_token_indptr), kvCacheBuffers.kvCacheBuffer,
        std::move(kv_scale_orig_quant), std::move(kv_scale_quant_orig), quantMode.hasFp8KvCache(), q_scaling);
}

th::Tensor dsv3_fused_mla_generation(th::Tensor fused_q, th::Tensor q_nope, th::Tensor k_b_proj_trans, th::Tensor q_pe,
    th::Tensor latent_cache, th::Tensor rotary_cos_sin, th::Tensor sequence_length, th::Tensor kv_cache_block_offsets,
    th::Tensor host_kv_cache_pool_pointers, th::Tensor host_kv_cache_pool_mapping, th::Tensor topk_indices,
    th::Tensor topk_indices_pool, th::Tensor kv_cache_pool, std::optional<th::Tensor> kv_scale_orig_quant,
    std::optional<th::Tensor> kv_scale_quant_orig, th::Tensor quant_q_buffer, th::Tensor mla_bmm1_scale,
    th::Tensor mla_bmm2_scale, std::optional<th::Tensor> spec_decoding_packed_mask, int64_t layer_idx,
    int64_t tokens_per_block, int64_t quant_mode, double q_scaling, bool is_context, int64_t context_chunk_start,
    std::optional<th::Tensor> q_b_proj_input, std::optional<th::Tensor> q_b_proj_weight,
    std::optional<th::Tensor> q_b_proj_weight_scale, std::optional<th::Tensor> q_b_proj_output, int64_t q_b_proj_impl)
{
    checkQbProjImpl(q_b_proj_impl);
    TORCH_CHECK(fused_q.is_cuda(), "fused_q must be a CUDA tensor");
    TORCH_CHECK(q_nope.is_cuda(), "q_nope must be a CUDA tensor");
    TORCH_CHECK(k_b_proj_trans.is_cuda(), "k_b_proj_trans must be a CUDA tensor");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be a CUDA tensor");
    TORCH_CHECK(latent_cache.is_cuda(), "latent_cache must be a CUDA tensor");
    TORCH_CHECK(rotary_cos_sin.is_cuda(), "rotary_cos_sin must be a CUDA tensor");
    TORCH_CHECK(sequence_length.is_cuda(), "sequence_length must be a CUDA tensor");
    TORCH_CHECK(kv_cache_block_offsets.is_cuda(), "kv_cache_block_offsets must be a CUDA tensor");
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(topk_indices_pool.is_cuda(), "topk_indices_pool must be a CUDA tensor");
    TORCH_CHECK(kv_cache_pool.is_cuda(), "kv_cache_pool must be a CUDA tensor");
    TORCH_CHECK(quant_q_buffer.is_cuda(), "quant_q_buffer must be a CUDA tensor");
    TORCH_CHECK(mla_bmm1_scale.is_cuda(), "mla_bmm1_scale must be a CUDA tensor");
    TORCH_CHECK(mla_bmm2_scale.is_cuda(), "mla_bmm2_scale must be a CUDA tensor");
    if (kv_scale_orig_quant.has_value())
    {
        TORCH_CHECK(kv_scale_orig_quant.value().is_cuda(), "kv_scale_orig_quant must be a CUDA tensor");
        TORCH_CHECK(kv_scale_orig_quant.value().scalar_type() == torch::kFloat32, "kv_scale_orig_quant must be fp32");
    }
    if (kv_scale_quant_orig.has_value())
    {
        TORCH_CHECK(kv_scale_quant_orig.value().is_cuda(), "kv_scale_quant_orig must be a CUDA tensor");
        TORCH_CHECK(kv_scale_quant_orig.value().scalar_type() == torch::kFloat32, "kv_scale_quant_orig must be fp32");
    }
    if (spec_decoding_packed_mask.has_value())
    {
        TORCH_CHECK(spec_decoding_packed_mask.value().is_cuda(), "spec_decoding_packed_mask must be a CUDA tensor");
        TORCH_CHECK(spec_decoding_packed_mask.value().scalar_type() == torch::kInt32,
            "spec_decoding_packed_mask must be int32");
    }
    bool const hasQbProj = q_b_proj_input.has_value() || q_b_proj_weight.has_value()
        || q_b_proj_weight_scale.has_value() || q_b_proj_output.has_value();
    TORCH_CHECK(!hasQbProj
            || (q_b_proj_input.has_value() && q_b_proj_weight.has_value() && q_b_proj_weight_scale.has_value()
                && q_b_proj_output.has_value()),
        "q_b_proj_input, q_b_proj_weight, q_b_proj_weight_scale, and q_b_proj_output must be provided together");
    if (hasQbProj)
    {
        TORCH_CHECK(q_b_proj_input.value().is_cuda(), "q_b_proj_input must be a CUDA tensor");
        TORCH_CHECK(q_b_proj_weight.value().is_cuda(), "q_b_proj_weight must be a CUDA tensor");
        TORCH_CHECK(q_b_proj_weight_scale.value().is_cuda(), "q_b_proj_weight_scale must be a CUDA tensor");
        TORCH_CHECK(q_b_proj_output.value().is_cuda(), "q_b_proj_output must be a CUDA tensor");
    }

    TORCH_CHECK(fused_q.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA generation expects bf16 fused_q");
    TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA generation expects bf16 q_nope");
    TORCH_CHECK(
        k_b_proj_trans.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA generation expects bf16 k_b_proj_trans");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA generation expects bf16 q_pe");
    TORCH_CHECK(latent_cache.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA generation expects bf16 latent_cache");
    TORCH_CHECK(rotary_cos_sin.scalar_type() == torch::kFloat32, "rotary_cos_sin must be fp32");
    TORCH_CHECK(sequence_length.scalar_type() == torch::kInt32, "sequence_length must be int32");
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32, "topk_indices must be int32");
    TORCH_CHECK(topk_indices_pool.scalar_type() == torch::kInt32, "topk_indices_pool must be int32");
    TORCH_CHECK(kv_cache_pool.scalar_type() == torch::kFloat8_e4m3fn,
        "GLM-5 fused MLA generation currently expects FP8 E4M3 kv_cache_pool");
    TORCH_CHECK(quant_q_buffer.scalar_type() == torch::kUInt8, "quant_q_buffer must be uint8-backed FP8");
    TORCH_CHECK(mla_bmm1_scale.scalar_type() == torch::kFloat32, "mla_bmm1_scale must be float32");
    TORCH_CHECK(mla_bmm2_scale.scalar_type() == torch::kFloat32, "mla_bmm2_scale must be float32");
    if (hasQbProj)
    {
        TORCH_CHECK(q_b_proj_input.value().scalar_type() == torch::kBFloat16, "q_b_proj_input must be bf16");
        TORCH_CHECK(q_b_proj_weight.value().scalar_type() == torch::kFloat8_e4m3fn, "q_b_proj_weight must be FP8 E4M3");
        TORCH_CHECK(q_b_proj_weight_scale.value().scalar_type() == torch::kInt32
                || q_b_proj_weight_scale.value().scalar_type() == torch::kFloat32,
            "q_b_proj_weight_scale must be packed UE8M0 int32 or original FP32 block scales");
        TORCH_CHECK(q_b_proj_output.value().scalar_type() == torch::kBFloat16, "q_b_proj_output must be bf16");
    }

    TORCH_CHECK(fused_q.dim() == 3, "fused_q must have shape [tokens, heads, 576]");
    TORCH_CHECK(q_nope.dim() == 3, "q_nope must have shape [tokens, heads, 192]");
    TORCH_CHECK(k_b_proj_trans.dim() == 3, "k_b_proj_trans must have shape [heads, 512, 192]");
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must have shape [tokens, heads, 64]");
    TORCH_CHECK(latent_cache.dim() == 2, "latent_cache must have shape [tokens, 576]");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must have shape [tokens, topk]");
    TORCH_CHECK(topk_indices_pool.dim() == 2, "topk_indices_pool must have shape [tokens, topk]");
    TORCH_CHECK(kv_cache_pool.dim() == 3, "kv_cache_pool must have shape [pool_tokens, 1, 576]");
    TORCH_CHECK(quant_q_buffer.dim() == 3, "quant_q_buffer must have shape [tokens, heads, 576]");
    TORCH_CHECK(fused_q.size(0) == q_pe.size(0), "fused_q and q_pe token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == q_nope.size(0), "fused_q and q_nope token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == latent_cache.size(0), "fused_q and latent_cache token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == topk_indices.size(0), "fused_q and topk_indices token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == quant_q_buffer.size(0), "fused_q and quant_q_buffer token dimensions must match");
    TORCH_CHECK(quant_q_buffer.size(0) == topk_indices_pool.size(0),
        "quant_q_buffer and topk_indices_pool token dimensions must match");
    TORCH_CHECK(
        topk_indices.size(1) == topk_indices_pool.size(1), "topk_indices and topk_indices_pool top-k must match");
    TORCH_CHECK(fused_q.size(1) == 8, "GLM-5 TP=8 fused MLA generation expects 8 local heads");
    TORCH_CHECK(q_nope.size(1) == 8, "GLM-5 TP=8 fused MLA generation expects 8 q_nope heads");
    TORCH_CHECK(k_b_proj_trans.size(0) == 8, "GLM-5 TP=8 fused MLA generation expects 8 k_b_proj heads");
    TORCH_CHECK(q_pe.size(1) == 8, "GLM-5 TP=8 fused MLA generation expects 8 q_pe heads");
    TORCH_CHECK(quant_q_buffer.size(1) == 8, "GLM-5 TP=8 fused MLA generation expects 8 local heads");
    TORCH_CHECK(fused_q.size(2) == 576, "GLM-5 fused MLA generation expects fused head size 576");
    TORCH_CHECK(q_nope.size(2) == 192, "GLM-5 fused MLA generation expects q_nope head size 192");
    TORCH_CHECK(k_b_proj_trans.size(1) == 512, "GLM-5 fused MLA generation expects k_b_proj dim 512");
    TORCH_CHECK(k_b_proj_trans.size(2) == 192, "GLM-5 fused MLA generation expects k_b_proj reduction dim 192");
    TORCH_CHECK(q_pe.size(2) == 64, "GLM-5 fused MLA generation expects q_pe head size 64");
    TORCH_CHECK(latent_cache.size(1) == 576, "GLM-5 fused MLA generation expects latent cache head size 576");
    TORCH_CHECK(quant_q_buffer.size(2) == 576, "GLM-5 fused MLA generation expects fused head size 576");
    TORCH_CHECK(kv_cache_pool.size(1) == 1, "GLM-5 fused MLA generation expects one KV head in kv_cache_pool");
    TORCH_CHECK(kv_cache_pool.size(2) == 576, "GLM-5 fused MLA generation expects kv_cache_pool head size 576");
    TORCH_CHECK(fused_q.stride(2) == 1, "fused_q last dimension must be contiguous");
    TORCH_CHECK(q_nope.stride(2) == 1, "q_nope last dimension must be contiguous");
    TORCH_CHECK(k_b_proj_trans.stride(2) == 1, "k_b_proj_trans last dimension must be contiguous");
    TORCH_CHECK(q_pe.stride(2) == 1, "q_pe last dimension must be contiguous");
    TORCH_CHECK(latent_cache.stride(1) == 1, "latent_cache last dimension must be contiguous");
    TORCH_CHECK(topk_indices.stride(1) == 1, "topk_indices last dimension must be contiguous");
    TORCH_CHECK(topk_indices_pool.stride(1) == 1, "topk_indices_pool last dimension must be contiguous");
    TORCH_CHECK(kv_cache_pool.stride(2) == 1, "kv_cache_pool last dimension must be contiguous");
    TORCH_CHECK(quant_q_buffer.stride(2) == 1, "quant_q_buffer last dimension must be contiguous");
    TORCH_CHECK(sequence_length.numel() == 1, "GLM-5 fused MLA generation currently expects one sequence");
    TORCH_CHECK(!is_context || context_chunk_start >= 0, "context_chunk_start must be non-negative in context mode");
    TORCH_CHECK(
        topk_indices_pool.size(1) <= 4096, "topk_indices_pool top-k dimension is larger than the dev kernel supports");
    if (hasQbProj)
    {
        TORCH_CHECK(q_b_proj_input.value().dim() == 2, "q_b_proj_input must have shape [tokens, 2048]");
        TORCH_CHECK(q_b_proj_weight.value().dim() == 2, "q_b_proj_weight must have shape [2048, 2048]");
        TORCH_CHECK(q_b_proj_weight_scale.value().dim() == 2,
            "q_b_proj_weight_scale must have shape [2048, 4] packed UE8M0 or [16, 16] FP32 block scales");
        TORCH_CHECK(q_b_proj_input.value().size(0) == fused_q.size(0),
            "q_b_proj_input and fused_q token dimensions must match");
        TORCH_CHECK(q_b_proj_input.value().size(1) == 2048, "q_b_proj_input must have hidden size 2048");
        TORCH_CHECK(q_b_proj_weight.value().size(0) == 2048, "q_b_proj_weight must have 2048 output rows");
        TORCH_CHECK(q_b_proj_weight.value().size(1) == 2048, "q_b_proj_weight must have 2048 input columns");
        if (q_b_proj_weight_scale.value().scalar_type() == torch::kInt32)
        {
            TORCH_CHECK(
                q_b_proj_weight_scale.value().size(0) == 2048, "q_b_proj_weight_scale must have 2048 output rows");
            TORCH_CHECK(q_b_proj_weight_scale.value().size(1) == 4,
                "q_b_proj_weight_scale must pack 16 K-block scales into four int32 values");
        }
        else
        {
            TORCH_CHECK(
                q_b_proj_weight_scale.value().size(0) == 16, "FP32 q_b_proj_weight_scale must have 16 output blocks");
            TORCH_CHECK(
                q_b_proj_weight_scale.value().size(1) == 16, "FP32 q_b_proj_weight_scale must have 16 K blocks");
        }
        TORCH_CHECK(q_b_proj_input.value().stride(1) == 1, "q_b_proj_input last dimension must be contiguous");
        TORCH_CHECK(q_b_proj_weight.value().stride(1) == 1, "q_b_proj_weight last dimension must be contiguous");
        TORCH_CHECK(q_b_proj_output.value().dim() == 2, "q_b_proj_output must have shape [tokens, 2048]");
        TORCH_CHECK(q_b_proj_output.value().size(0) == fused_q.size(0),
            "q_b_proj_output and fused_q token dimensions must match");
        TORCH_CHECK(q_b_proj_output.value().size(1) == 2048, "q_b_proj_output must have hidden size 2048");
        TORCH_CHECK(q_b_proj_output.value().stride(1) == 1, "q_b_proj_output last dimension must be contiguous");
        TORCH_CHECK(q_b_proj_output.value().stride(0) == 2048, "q_b_proj_output must be contiguous");
    }

    auto const quantMode = common::QuantMode{static_cast<uint32_t>(quant_mode)};
    TORCH_CHECK(quantMode.hasFp8KvCache(), "GLM-5 fused MLA generation currently expects FP8 KV cache");

    auto const maxAttentionWindow = static_cast<int64_t>(kv_cache_block_offsets.size(-1)) * tokens_per_block;
    auto kvCacheBuffers = buildPagedKvCacheBuffers(std::optional<torch::Tensor>(kv_cache_block_offsets),
        std::optional<torch::Tensor>(host_kv_cache_pool_pointers),
        std::optional<torch::Tensor>(host_kv_cache_pool_mapping), quantMode, layer_idx, /*batch_size=*/1,
        tokens_per_block, /*kv_head_num=*/1, /*size_per_head=*/576,
        /*cyclic_attention_window_size=*/maxAttentionWindow, /*max_attention_window_size=*/maxAttentionWindow,
        /*sink_token_length=*/0, /*beam_width=*/1, /*seq_offset=*/0, /*is_mla_enable=*/true, fused_q.element_size());
    TORCH_CHECK(kvCacheBuffers.kvCacheBuffer.data != nullptr, "KV cache block offsets are required");

    return dsv3_fused_mla::dsv3_fused_mla_generation_cuda(std::move(fused_q), std::move(q_nope),
        std::move(k_b_proj_trans), std::move(q_pe), std::move(latent_cache), std::move(rotary_cos_sin),
        std::move(sequence_length), kvCacheBuffers.kvCacheBuffer, std::move(topk_indices), std::move(topk_indices_pool),
        std::move(kv_cache_pool), std::move(quant_q_buffer), std::move(mla_bmm1_scale), std::move(mla_bmm2_scale),
        std::move(kv_scale_orig_quant), std::move(kv_scale_quant_orig), std::move(spec_decoding_packed_mask), q_scaling,
        is_context, context_chunk_start, std::move(q_b_proj_input), std::move(q_b_proj_weight),
        std::move(q_b_proj_weight_scale), std::move(q_b_proj_output), q_b_proj_impl);
}

th::Tensor dsv3_fused_mla_q_b_proj(
    th::Tensor q_b_proj_input, th::Tensor q_b_proj_weight, th::Tensor q_b_proj_weight_scale, bool q_b_proj_use_mma)
{
    return dsv3_fused_mla_q_b_proj_impl(std::move(q_b_proj_input), std::move(q_b_proj_weight),
        std::move(q_b_proj_weight_scale), q_b_proj_use_mma ? kQbProjImplMma : kQbProjImplScalar);
}

th::Tensor dsv3_fused_mla_q_b_proj_impl(
    th::Tensor q_b_proj_input, th::Tensor q_b_proj_weight, th::Tensor q_b_proj_weight_scale, int64_t q_b_proj_impl)
{
    checkQbProjImpl(q_b_proj_impl);
    TORCH_CHECK(q_b_proj_input.is_cuda(), "q_b_proj_input must be a CUDA tensor");
    TORCH_CHECK(q_b_proj_weight.is_cuda(), "q_b_proj_weight must be a CUDA tensor");
    TORCH_CHECK(q_b_proj_weight_scale.is_cuda(), "q_b_proj_weight_scale must be a CUDA tensor");
    TORCH_CHECK(q_b_proj_input.scalar_type() == torch::kBFloat16, "q_b_proj_input must be bf16");
    TORCH_CHECK(q_b_proj_weight.scalar_type() == torch::kFloat8_e4m3fn, "q_b_proj_weight must be FP8 E4M3");
    TORCH_CHECK(
        q_b_proj_weight_scale.scalar_type() == torch::kInt32 || q_b_proj_weight_scale.scalar_type() == torch::kFloat32,
        "q_b_proj_weight_scale must be packed UE8M0 int32 or original FP32 block scales");
    TORCH_CHECK(q_b_proj_input.dim() == 2, "q_b_proj_input must have shape [tokens, 2048]");
    TORCH_CHECK(q_b_proj_weight.dim() == 2, "q_b_proj_weight must have shape [2048, 2048]");
    TORCH_CHECK(q_b_proj_weight_scale.dim() == 2,
        "q_b_proj_weight_scale must have shape [2048, 4] packed UE8M0 or [16, 16] FP32 block scales");
    TORCH_CHECK(q_b_proj_input.size(0) > 0, "q_b_proj_input must have at least one token");
    TORCH_CHECK(q_b_proj_input.size(1) == 2048, "q_b_proj_input must have hidden size 2048");
    TORCH_CHECK(q_b_proj_weight.size(0) == 2048, "q_b_proj_weight must have 2048 output rows");
    TORCH_CHECK(q_b_proj_weight.size(1) == 2048, "q_b_proj_weight must have 2048 input columns");
    if (q_b_proj_weight_scale.scalar_type() == torch::kInt32)
    {
        TORCH_CHECK(q_b_proj_weight_scale.size(0) == 2048, "q_b_proj_weight_scale must have 2048 output rows");
        TORCH_CHECK(q_b_proj_weight_scale.size(1) == 4,
            "q_b_proj_weight_scale must pack 16 K-block scales into four int32 values");
    }
    else
    {
        TORCH_CHECK(q_b_proj_weight_scale.size(0) == 16, "FP32 q_b_proj_weight_scale must have 16 output blocks");
        TORCH_CHECK(q_b_proj_weight_scale.size(1) == 16, "FP32 q_b_proj_weight_scale must have 16 K blocks");
    }
    TORCH_CHECK(q_b_proj_input.stride(1) == 1, "q_b_proj_input last dimension must be contiguous");
    TORCH_CHECK(q_b_proj_weight.stride(1) == 1, "q_b_proj_weight last dimension must be contiguous");

    return dsv3_fused_mla::dsv3_fused_mla_q_b_proj_cuda(
        std::move(q_b_proj_input), std::move(q_b_proj_weight), std::move(q_b_proj_weight_scale), q_b_proj_impl);
}

} // namespace torch_ext
} // namespace tensorrt_llm

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dsv3_fused_mla_context(Tensor(a!) fused_q, Tensor q_pe, Tensor(b!) latent_cache, "
        "Tensor topk_indices_pool, Tensor topk_indices_local, Tensor kv_cache_pool, Tensor rotary_cos_sin, "
        "Tensor ctx_cached_token_indptr, Tensor kv_cache_block_offsets, Tensor host_kv_cache_pool_pointers, "
        "Tensor host_kv_cache_pool_mapping, Tensor? kv_scale_orig_quant, Tensor? kv_scale_quant_orig, "
        "int layer_idx, int tokens_per_block, int quant_mode, float q_scaling) -> Tensor");
    m.def(
        "dsv3_fused_mla_generation(Tensor(a!) fused_q, Tensor q_nope, Tensor k_b_proj_trans, Tensor q_pe, "
        "Tensor latent_cache, Tensor rotary_cos_sin, Tensor sequence_length, Tensor kv_cache_block_offsets, "
        "Tensor host_kv_cache_pool_pointers, Tensor host_kv_cache_pool_mapping, Tensor topk_indices, "
        "Tensor topk_indices_pool, Tensor(b!) kv_cache_pool, Tensor? kv_scale_orig_quant, Tensor? kv_scale_quant_orig, "
        "Tensor(c!) quant_q_buffer, Tensor(d!) mla_bmm1_scale, Tensor(e!) mla_bmm2_scale, "
        "Tensor? spec_decoding_packed_mask, int layer_idx, int tokens_per_block, int quant_mode, "
        "float q_scaling, bool is_context, int context_chunk_start, Tensor? q_b_proj_input, "
        "Tensor? q_b_proj_weight, Tensor? q_b_proj_weight_scale, Tensor? q_b_proj_output, "
        "int q_b_proj_impl) -> Tensor");
    m.def(
        "dsv3_fused_mla_q_b_proj(Tensor q_b_proj_input, Tensor q_b_proj_weight, "
        "Tensor q_b_proj_weight_scale, bool q_b_proj_use_mma) -> Tensor");
    m.def(
        "dsv3_fused_mla_q_b_proj_impl(Tensor q_b_proj_input, Tensor q_b_proj_weight, "
        "Tensor q_b_proj_weight_scale, int q_b_proj_impl) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_mla_context", &tensorrt_llm::torch_ext::dsv3_fused_mla_context);
    m.impl("dsv3_fused_mla_generation", &tensorrt_llm::torch_ext::dsv3_fused_mla_generation);
    m.impl("dsv3_fused_mla_q_b_proj", &tensorrt_llm::torch_ext::dsv3_fused_mla_q_b_proj);
    m.impl("dsv3_fused_mla_q_b_proj_impl", &tensorrt_llm::torch_ext::dsv3_fused_mla_q_b_proj_impl);
}
