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
torch::Tensor dsv3_fused_mla_generation_cuda(torch::Tensor fused_q, torch::Tensor topk_indices_pool,
    torch::Tensor topk_indices, torch::Tensor kv_cache_pool, torch::Tensor sequence_length,
    torch::Tensor quant_q_buffer, torch::Tensor mla_bmm1_scale, torch::Tensor mla_bmm2_scale);
} // namespace dsv3_fused_mla

namespace tensorrt_llm
{
namespace torch_ext
{

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

th::Tensor dsv3_fused_mla_generation(th::Tensor fused_q, th::Tensor q_pe, th::Tensor latent_cache,
    th::Tensor topk_indices, th::Tensor topk_indices_pool, th::Tensor kv_cache_pool,
    std::optional<th::Tensor> workspace, th::Tensor sequence_length, th::Tensor host_past_key_value_lengths,
    th::Tensor host_total_kv_lens, th::Tensor context_lengths, th::Tensor host_context_lengths,
    th::Tensor host_request_types, th::Tensor rotary_cos_sin, std::optional<th::Tensor> rotary_inv_freq,
    std::optional<th::Tensor> cache_indirection, std::optional<th::Tensor> block_ids_per_seq,
    th::Tensor kv_cache_block_offsets, th::Tensor host_kv_cache_pool_pointers, th::Tensor host_kv_cache_pool_mapping,
    std::optional<th::Tensor> kv_scale_orig_quant, std::optional<th::Tensor> kv_scale_quant_orig,
    th::Tensor cu_q_seqlens, th::Tensor cu_kv_seqlens, th::Tensor fmha_scheduler_counter,
    std::optional<th::Tensor> mla_bmm1_scale, std::optional<th::Tensor> mla_bmm2_scale,
    std::optional<th::Tensor> quant_q_buffer, bool is_spec_decoding_enabled, bool use_spec_decoding,
    bool is_spec_dec_tree, std::optional<th::Tensor> spec_decoding_generation_lengths,
    std::optional<th::Tensor> spec_decoding_position_offsets, std::optional<th::Tensor> spec_decoding_packed_mask,
    std::optional<th::Tensor> spec_decoding_bl_tree_mask_offset, std::optional<th::Tensor> spec_decoding_bl_tree_mask,
    std::optional<th::Tensor> spec_bl_tree_first_sparse_mask_offset_kv, int64_t layer_idx, int64_t tokens_per_block,
    int64_t quant_mode, double q_scaling, int64_t position_embedding_type, int64_t rotary_embedding_dim,
    double rotary_embedding_base, int64_t rotary_embedding_scale_type, double rotary_embedding_scale,
    double rotary_embedding_short_m_scale, double rotary_embedding_long_m_scale, int64_t rotary_embedding_max_positions,
    int64_t rotary_embedding_original_max_positions, int64_t predicted_tokens_per_seq, int64_t q_lora_rank,
    int64_t qk_nope_head_dim, int64_t max_num_requests, int64_t max_context_length, int64_t attention_window_size,
    int64_t beam_width, int64_t num_contexts, int64_t num_ctx_tokens)
{
    TORCH_CHECK(fused_q.is_cuda(), "fused_q must be a CUDA tensor");
    TORCH_CHECK(q_pe.is_cuda(), "q_pe must be a CUDA tensor");
    TORCH_CHECK(latent_cache.is_cuda(), "latent_cache must be a CUDA tensor");
    TORCH_CHECK(topk_indices.is_cuda(), "topk_indices must be a CUDA tensor");
    TORCH_CHECK(topk_indices_pool.is_cuda(), "topk_indices_pool must be a CUDA tensor");
    TORCH_CHECK(kv_cache_pool.is_cuda(), "kv_cache_pool must be a CUDA tensor");
    if (workspace.has_value())
    {
        TORCH_CHECK(workspace.value().is_cuda(), "workspace must be a CUDA tensor");
    }
    TORCH_CHECK(sequence_length.is_cuda(), "sequence_length must be a CUDA tensor");
    TORCH_CHECK(context_lengths.is_cuda(), "context_lengths must be a CUDA tensor");
    TORCH_CHECK(rotary_cos_sin.is_cuda(), "rotary_cos_sin must be a CUDA tensor");
    TORCH_CHECK(kv_cache_block_offsets.is_cuda(), "kv_cache_block_offsets must be a CUDA tensor");
    TORCH_CHECK(cu_q_seqlens.is_cuda(), "cu_q_seqlens must be a CUDA tensor");
    TORCH_CHECK(cu_kv_seqlens.is_cuda(), "cu_kv_seqlens must be a CUDA tensor");
    TORCH_CHECK(fmha_scheduler_counter.is_cuda(), "fmha_scheduler_counter must be a CUDA tensor");
    TORCH_CHECK(mla_bmm1_scale.has_value(), "mla_bmm1_scale is required for GLM-5 fused MLA generation");
    TORCH_CHECK(mla_bmm1_scale.value().is_cuda(), "mla_bmm1_scale must be a CUDA tensor");
    TORCH_CHECK(mla_bmm2_scale.has_value(), "mla_bmm2_scale is required for GLM-5 fused MLA generation");
    TORCH_CHECK(mla_bmm2_scale.value().is_cuda(), "mla_bmm2_scale must be a CUDA tensor");
    TORCH_CHECK(quant_q_buffer.has_value(), "quant_q_buffer is required for GLM-5 fused MLA generation");
    TORCH_CHECK(quant_q_buffer.value().is_cuda(), "quant_q_buffer must be a CUDA tensor");

    TORCH_CHECK(fused_q.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA generation currently expects bf16 fused_q");
    TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16, "GLM-5 fused MLA generation currently expects bf16 q_pe");
    TORCH_CHECK(latent_cache.scalar_type() == torch::kBFloat16,
        "GLM-5 fused MLA generation currently expects bf16 latent_cache");
    TORCH_CHECK(topk_indices.scalar_type() == torch::kInt32, "topk_indices must be int32");
    TORCH_CHECK(topk_indices_pool.scalar_type() == torch::kInt32, "topk_indices_pool must be int32");
    TORCH_CHECK(kv_cache_pool.scalar_type() == torch::kFloat8_e4m3fn,
        "GLM-5 fused MLA generation currently expects FP8 E4M3 kv_cache_pool");
    TORCH_CHECK(sequence_length.scalar_type() == torch::kInt32, "sequence_length must be int32");
    TORCH_CHECK(
        host_past_key_value_lengths.scalar_type() == torch::kInt32, "host_past_key_value_lengths must be int32");
    TORCH_CHECK(host_total_kv_lens.scalar_type() == torch::kInt32, "host_total_kv_lens must be int32");
    TORCH_CHECK(context_lengths.scalar_type() == torch::kInt32, "context_lengths must be int32");
    TORCH_CHECK(host_context_lengths.scalar_type() == torch::kInt32, "host_context_lengths must be int32");
    TORCH_CHECK(host_request_types.scalar_type() == torch::kInt32, "host_request_types must be int32");
    TORCH_CHECK(cu_q_seqlens.scalar_type() == torch::kInt32, "cu_q_seqlens must be int32");
    TORCH_CHECK(cu_kv_seqlens.scalar_type() == torch::kInt32, "cu_kv_seqlens must be int32");
    TORCH_CHECK(fmha_scheduler_counter.scalar_type() == torch::kUInt32, "fmha_scheduler_counter must be uint32");
    TORCH_CHECK(mla_bmm1_scale.value().scalar_type() == torch::kFloat32, "mla_bmm1_scale must be float32");
    TORCH_CHECK(mla_bmm2_scale.value().scalar_type() == torch::kFloat32, "mla_bmm2_scale must be float32");
    TORCH_CHECK(quant_q_buffer.value().scalar_type() == torch::kUInt8, "quant_q_buffer must be uint8-backed FP8");

    TORCH_CHECK(fused_q.dim() == 3, "fused_q must have shape [tokens, heads, 576]");
    TORCH_CHECK(q_pe.dim() == 3, "q_pe must have shape [tokens, heads, 64]");
    TORCH_CHECK(latent_cache.dim() == 2, "latent_cache must have shape [tokens, 576]");
    TORCH_CHECK(topk_indices.dim() == 2, "topk_indices must have shape [tokens, topk]");
    TORCH_CHECK(topk_indices_pool.dim() == 2, "topk_indices_pool must have shape [tokens, topk]");
    TORCH_CHECK(kv_cache_pool.dim() == 3, "kv_cache_pool must have shape [pool_tokens, 1, 576]");
    TORCH_CHECK(quant_q_buffer.value().dim() == 3, "quant_q_buffer must have shape [tokens, heads, 576]");
    TORCH_CHECK(fused_q.size(0) == q_pe.size(0), "fused_q and q_pe token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == latent_cache.size(0), "fused_q and latent_cache token dimensions must match");
    TORCH_CHECK(fused_q.size(0) == topk_indices.size(0), "fused_q and topk_indices token dimensions must match");
    TORCH_CHECK(
        fused_q.size(0) == topk_indices_pool.size(0), "fused_q and topk_indices_pool token dimensions must match");
    TORCH_CHECK(
        topk_indices.size(1) == topk_indices_pool.size(1), "topk_indices and topk_indices_pool top-k must match");
    TORCH_CHECK(fused_q.sizes() == quant_q_buffer.value().sizes(), "fused_q and quant_q_buffer dimensions must match");
    TORCH_CHECK(fused_q.size(1) == 8, "GLM-5 TP=8 fused MLA generation expects 8 local heads");
    TORCH_CHECK(q_pe.size(1) == 8, "GLM-5 TP=8 fused MLA generation expects 8 q_pe heads");
    TORCH_CHECK(fused_q.size(2) == 576, "GLM-5 fused MLA generation expects fused head size 576");
    TORCH_CHECK(q_pe.size(2) == 64, "GLM-5 fused MLA generation expects q_pe head size 64");
    TORCH_CHECK(latent_cache.size(1) == 576, "GLM-5 fused MLA generation expects latent cache head size 576");
    TORCH_CHECK(kv_cache_pool.size(1) == 1, "GLM-5 fused MLA generation expects one KV head in kv_cache_pool");
    TORCH_CHECK(kv_cache_pool.size(2) == 576, "GLM-5 fused MLA generation expects kv_cache_pool head size 576");
    TORCH_CHECK(fused_q.stride(2) == 1, "fused_q last dimension must be contiguous");
    TORCH_CHECK(q_pe.stride(2) == 1, "q_pe last dimension must be contiguous");
    TORCH_CHECK(latent_cache.stride(1) == 1, "latent_cache last dimension must be contiguous");
    TORCH_CHECK(topk_indices.stride(1) == 1, "topk_indices last dimension must be contiguous");
    TORCH_CHECK(topk_indices_pool.stride(1) == 1, "topk_indices_pool last dimension must be contiguous");
    TORCH_CHECK(kv_cache_pool.stride(2) == 1, "kv_cache_pool last dimension must be contiguous");
    TORCH_CHECK(quant_q_buffer.value().stride(2) == 1, "quant_q_buffer last dimension must be contiguous");
    TORCH_CHECK(
        topk_indices_pool.size(1) <= 4096, "topk_indices_pool top-k dimension is larger than the dev kernel supports");

    auto const quantMode = common::QuantMode{static_cast<uint32_t>(quant_mode)};
    TORCH_CHECK(quantMode.hasFp8KvCache(), "GLM-5 fused MLA generation currently expects FP8 KV cache");

    (void) workspace;
    (void) host_past_key_value_lengths;
    (void) host_total_kv_lens;
    (void) context_lengths;
    (void) host_context_lengths;
    (void) host_request_types;
    (void) rotary_cos_sin;
    (void) rotary_inv_freq;
    (void) cache_indirection;
    (void) block_ids_per_seq;
    (void) kv_cache_block_offsets;
    (void) host_kv_cache_pool_pointers;
    (void) host_kv_cache_pool_mapping;
    (void) kv_scale_orig_quant;
    (void) kv_scale_quant_orig;
    (void) cu_q_seqlens;
    (void) cu_kv_seqlens;
    (void) fmha_scheduler_counter;
    (void) is_spec_decoding_enabled;
    (void) use_spec_decoding;
    (void) is_spec_dec_tree;
    (void) spec_decoding_generation_lengths;
    (void) spec_decoding_position_offsets;
    (void) spec_decoding_packed_mask;
    (void) spec_decoding_bl_tree_mask_offset;
    (void) spec_decoding_bl_tree_mask;
    (void) spec_bl_tree_first_sparse_mask_offset_kv;
    (void) layer_idx;
    (void) tokens_per_block;
    (void) q_scaling;
    (void) position_embedding_type;
    (void) rotary_embedding_dim;
    (void) rotary_embedding_base;
    (void) rotary_embedding_scale_type;
    (void) rotary_embedding_scale;
    (void) rotary_embedding_short_m_scale;
    (void) rotary_embedding_long_m_scale;
    (void) rotary_embedding_max_positions;
    (void) rotary_embedding_original_max_positions;
    (void) predicted_tokens_per_seq;
    (void) q_lora_rank;
    (void) qk_nope_head_dim;
    (void) max_num_requests;
    (void) max_context_length;
    (void) attention_window_size;
    (void) beam_width;
    (void) num_contexts;
    (void) num_ctx_tokens;

    return dsv3_fused_mla::dsv3_fused_mla_generation_cuda(std::move(fused_q), std::move(topk_indices_pool),
        std::move(topk_indices), std::move(kv_cache_pool), std::move(sequence_length),
        std::move(quant_q_buffer.value()), std::move(mla_bmm1_scale.value()), std::move(mla_bmm2_scale.value()));
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
        "dsv3_fused_mla_generation(Tensor fused_q, Tensor q_pe, Tensor latent_cache, Tensor topk_indices, "
        "Tensor topk_indices_pool, Tensor kv_cache_pool, Tensor? workspace, Tensor sequence_length, "
        "Tensor host_past_key_value_lengths, Tensor host_total_kv_lens, Tensor context_lengths, "
        "Tensor host_context_lengths, Tensor host_request_types, "
        "Tensor rotary_cos_sin, "
        "Tensor? rotary_inv_freq, Tensor? cache_indirection, Tensor? block_ids_per_seq, "
        "Tensor kv_cache_block_offsets, Tensor host_kv_cache_pool_pointers, Tensor host_kv_cache_pool_mapping, "
        "Tensor? kv_scale_orig_quant, Tensor? kv_scale_quant_orig, Tensor cu_q_seqlens, Tensor cu_kv_seqlens, "
        "Tensor fmha_scheduler_counter, Tensor? mla_bmm1_scale, Tensor? mla_bmm2_scale, Tensor? quant_q_buffer, "
        "bool is_spec_decoding_enabled, bool use_spec_decoding, bool is_spec_dec_tree, "
        "Tensor? spec_decoding_generation_lengths, Tensor? spec_decoding_position_offsets, "
        "Tensor? spec_decoding_packed_mask, Tensor? spec_decoding_bl_tree_mask_offset, "
        "Tensor? spec_decoding_bl_tree_mask, Tensor? spec_bl_tree_first_sparse_mask_offset_kv, int layer_idx, "
        "int tokens_per_block, int quant_mode, float q_scaling, int position_embedding_type, "
        "int rotary_embedding_dim, float rotary_embedding_base, int rotary_embedding_scale_type, "
        "float rotary_embedding_scale, float rotary_embedding_short_m_scale, float rotary_embedding_long_m_scale, "
        "int rotary_embedding_max_positions, int rotary_embedding_original_max_positions, "
        "int predicted_tokens_per_seq, int q_lora_rank, int qk_nope_head_dim, int max_num_requests, "
        "int max_context_length, int attention_window_size, int beam_width, int num_contexts, "
        "int num_ctx_tokens) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dsv3_fused_mla_context", &tensorrt_llm::torch_ext::dsv3_fused_mla_context);
    m.impl("dsv3_fused_mla_generation", &tensorrt_llm::torch_ext::dsv3_fused_mla_generation);
}
