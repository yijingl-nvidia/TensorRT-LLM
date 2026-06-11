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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <optional>

namespace
{

constexpr int kLocalHeads = 8;
constexpr int kKvLoraRank = 512;
constexpr int kRopeDim = 64;
constexpr int kHeadDim = kKvLoraRank + kRopeDim;
constexpr int kThreads = 256;
constexpr int kMaxTopK = 4096;

// This file is intentionally specialized to the GLM-5/DeepSeek-V3 fused MLA shape used by the
// WIP Python path:
//
//
// Generation input tensors:
// - quantQ:          FP8 E4M3, logical shape [num_gen_tokens, 8 local heads, 576]. It is produced by
//                    mla_rope_generation after applying Q RoPE and quantizing the full absorbed Q.
// - topkIndicesLocal: INT32, logical shape [num_gen_tokens, topK]. Each non-negative entry is the local
//                    KV position inside the single generation sequence. This is used only for causal
//                    validity checks.
// - topkIndicesPool: INT32, logical shape [num_gen_tokens, topK]. Each non-negative entry is the global
//                    kvCachePool row to load after the DSA index conversion.
// - sequenceLength:  INT32, logical shape [1] on the currently enabled custom path. It is the total KV
//                    length after mla_rope_generation appends all num_gen_tokens rows for this step.
// - bmm1Scale:       FP32, logical shape [2]. bmm1Scale[0] is the natural-exp score scale, and
//                    bmm1Scale[1] is bmm1Scale[0] * log2(e), used with exp2f().
// - bmm2Scale:       FP32, logical shape [1]. It dequantizes the FP8 V/cache values before BF16 output.
// - output:          BF16, logical shape [num_tokens, 8 * 512]. It stores latent attention output before
//                    the per-head V-up projection in Python.

__device__ __forceinline__ float toFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
}

__device__ __forceinline__ float toFloat(__nv_fp8_e4m3 value)
{
    return static_cast<float>(value);
}

__device__ __forceinline__ __nv_bfloat16 toBfloat16(float value)
{
    return __float2bfloat16(value);
}

__device__ __forceinline__ float2 rotaryTransform(float2 value, float2 coef)
{
    float2 rotated;
    rotated.x = coef.x * value.x - coef.y * value.y;
    rotated.y = coef.x * value.y + coef.y * value.x;
    return rotated;
}

__device__ __forceinline__ float blockReduceSum(float value, float* shared)
{
    int const tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

// Preprocess one context token row before sparse MLA attention.
//
// Grid:
// - blockIdx.x selects the context token row in [0, numTokens).
// - blockIdx.y selects the preprocessing role:
//   - [0, kLocalHeads) rotates one per-head Q RoPE suffix from qPe into fusedQ.
//   - kLocalHeads rotates the shared latent K RoPE suffix in latentCache, then writes the full latent KV row
//     into the paged KV cache.
// - threadIdx.x is used as a strided lane over RoPE pairs or full 576-dim KV elements.
//
// Inputs and mutable tensors:
// - fusedQ: mutable BF16 [numTokens, kLocalHeads, kHeadDim]. Dims [0, kKvLoraRank) already contain the
//   absorbed q_nope projection. The Q branch fills dims [kKvLoraRank, kHeadDim) with rotated Q RoPE.
// - latentCache: mutable BF16 [numTokens, kHeadDim]. Dims [0, kKvLoraRank) are compressed latent KV/V values.
//   Dims [kKvLoraRank, kHeadDim) are the unrotated shared K RoPE suffix; the KV branch rotates them in-place.
// - qPe: BF16 [numTokens, kLocalHeads, kRopeDim]. Unrotated per-head Q RoPE values.
// - rotaryCosSin: FP32 float2 RoPE table. It is indexed by the absolute KV position, so cached prefixes use
//   the same rotary phase as TRT-LLM's paged attention path.
// - ctxCachedTokenIndptr: INT64 [numContexts + 1]. The current custom path is restricted to one context request;
//   ctxCachedTokenIndptr[1] - ctxCachedTokenIndptr[0] is the already-cached prefix length for sequence 0.
// - kvCache: paged TRT-LLM KV cache view with one MLA KV head and kHeadDim elements per token.
// - kvScaleOrigQuant: optional FP32 [1]. When kFp8KvCache is true, BF16 latentCache values are multiplied by
//   this original-domain to FP8 scale before storing to the paged KV cache.
// - numTokens: number of context tokens in fusedQ/qPe/latentCache.
//
// Side effects:
// - Q branch mutates fusedQ[token, head, kKvLoraRank:kHeadDim].
// - KV branch mutates latentCache[token, kKvLoraRank:kHeadDim] and writes latentCache[token, :] to kvCache.
// - The function does not produce a returned tensor; contextAttentionKernel consumes fusedQ and kvCachePool later.
template <bool kFp8KvCache>
__global__ void preprocessContextKernel(__nv_bfloat16* fusedQ, __nv_bfloat16* latentCache, __nv_bfloat16 const* qPe,
    float2 const* rotaryCosSin, int64_t const* ctxCachedTokenIndptr, tensorrt_llm::kernels::KVBlockArray kvCache,
    float const* kvScaleOrigQuant, int numTokens)
{
    // Grid shape is [numTokens, kLocalHeads + 1].
    // - blockIdx.x selects one context token row.
    // - blockIdx.y in [0, 7] rotates one Q head from qPe into fusedQ.
    // - blockIdx.y == 8 rotates the single shared latent K RoPE suffix and writes the 576-dim KV row to cache.
    int const tokenIdx = static_cast<int>(blockIdx.x);
    int const headIdx = static_cast<int>(blockIdx.y);
    // Current custom context dispatch is restricted to one context request. ctxCachedTokenIndptr has shape
    // [num_contexts + 1] == [2], so this is the number of already-cached prefix tokens for sequence 0.
    int64_t const cachedLen = ctxCachedTokenIndptr[1] - ctxCachedTokenIndptr[0];
    // tokenIdxInKvCache is the absolute KV position for this context token after any cached prefix.
    int const tokenIdxInKvCache = static_cast<int>(cachedLen) + tokenIdx;
    // kvScaleOrigQuant maps BF16 original values to FP8 E4M3 cache values when the paged KV cache is FP8.
    float const kvScale = kvScaleOrigQuant == nullptr ? 1.0F : kvScaleOrigQuant[0];

    if (tokenIdx >= numTokens)
    {
        return;
    }

    if (headIdx < kLocalHeads)
    {
        // Q branch: each block rotates one token/head RoPE suffix. The absorbed q_nope prefix in fusedQ is produced
        // by the Python-side BMM and is left unchanged here.
        for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
        {
            float2 value;
            // dim is the first element of an adjacent RoPE pair in qPe[token, head, dim:dim + 2].
            int const dim = 2 * pairIdx;
            // qPe is BF16 [numTokens, 8, 64] with contiguous last dimension.
            int const qPeOffset = (tokenIdx * kLocalHeads + headIdx) * kRopeDim + dim;
            value.x = toFloat(qPe[qPeOffset]);
            value.y = toFloat(qPe[qPeOffset + 1]);
            // rotaryCosSin is indexed by absolute KV position so cached prefixes get the correct phase.
            float2 const coef = rotaryCosSin[tokenIdxInKvCache * kRopeDim + pairIdx];
            float2 const rotated = rotaryTransform(value, coef);

            // fusedQ is BF16 [numTokens, 8, 576]; write the RoPE suffix at dims [512, 576).
            int const fusedOffset = (tokenIdx * kLocalHeads + headIdx) * kHeadDim + kKvLoraRank + dim;
            fusedQ[fusedOffset] = toBfloat16(rotated.x);
            fusedQ[fusedOffset + 1] = toBfloat16(rotated.y);
        }
        return;
    }

    // KV branch: one block per token rotates the shared latent K RoPE suffix and then writes the complete
    // [latent KV/V, K RoPE] row into TRT-LLM's paged KV cache.
    for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
    {
        float2 value;
        // latentCache is BF16 [numTokens, 576]; the final 64 elements are the shared K RoPE suffix.
        int const dim = 2 * pairIdx;
        int const latentOffset = tokenIdx * kHeadDim + kKvLoraRank + dim;
        value.x = toFloat(latentCache[latentOffset]);
        value.y = toFloat(latentCache[latentOffset + 1]);
        float2 const coef = rotaryCosSin[tokenIdxInKvCache * kRopeDim + pairIdx];
        float2 const rotated = rotaryTransform(value, coef);
        latentCache[latentOffset] = toBfloat16(rotated.x);
        latentCache[latentOffset + 1] = toBfloat16(rotated.y);
    }
    __syncthreads();

    auto* blockPtr = kvCache.getKBlockPtr(/*seqIdx=*/0, tokenIdxInKvCache);
    for (int dim = threadIdx.x; dim < kHeadDim; dim += blockDim.x)
    {
        // cacheOffset maps [absolute token position, one MLA KV head, dim] into TRT-LLM's paged block layout.
        int const cacheOffset = kvCache.getKVLocalIdx(tokenIdxInKvCache, /*headIdx=*/0, kHeadDim, dim);
        __nv_bfloat16 const value = latentCache[tokenIdx * kHeadDim + dim];
        if constexpr (kFp8KvCache)
        {
            reinterpret_cast<__nv_fp8_e4m3*>(blockPtr)[cacheOffset] = __nv_fp8_e4m3(toFloat(value) * kvScale);
        }
        else
        {
            reinterpret_cast<__nv_bfloat16*>(blockPtr)[cacheOffset] = value;
        }
    }
}

// Context attention kernel: compute scores for each head between queries and keys and return
// softmax-weighted sum of value vectors for each token as outputs
//
// Inputs:
// - fusedQ: BF16 [numTokens, kLocalHeads, kHeadDim]. Per-token per-head Q.
//   The first 512 dims in kHeadDim are absorbed q_nope, and the final 64 dims are Q RoPE
//   written by preprocessContextKernel before this kernel runs.
// - kvCachePool: FP8 E4M3 [poolTokens, kHeadDim]. Sparse KV pool rows.
//   All 576 dims are used as K for QK scores; only the first 512 dims are used
//   as latent V for the output.
// - topkIndicesPool: INT32 [numTokens, topK]. For each query token, contains
//   global row indices into kvCachePool. Negative entries are padding and are
//   treated as masked-out KV candidates.
// - kvScaleQuantOrig: optional FP32 scale mapping FP8 cache values back to
//   original units. Used twice for QK score scale and once for V dequant.
// - numTokens/topK: logical sizes.
// - hostScoreScale: host-computed attention scale, currently
//   1 / (q_scaling * sqrt(256)).
//
// Output:
// - output: BF16 [numTokens, kLocalHeads * kKvLoraRank], logically
//   [numTokens, kLocalHeads, 512]. Each row is the softmax-weighted sum of the
//   sparse FP8 latent V rows, dequantized back to BF16 units.
//
// Grid:
// - One CUDA block computes one (tokenIdx, headIdx) output row.
// - blockIdx.x selects query token.
// - blockIdx.y selects local attention head.
// - threadIdx.x cooperates across kHeadDim for QK reductions and across
//   kKvLoraRank for the final V accumulation.
__global__ void contextAttentionKernel(__nv_bfloat16 const* fusedQ, __nv_fp8_e4m3 const* kvCachePool,
    int32_t const* topkIndicesPool, __nv_bfloat16* output, float const* kvScaleQuantOrig, int numTokens, int topK,
    float hostScoreScale)
{
    // One CUDA block computes one (query token, local head) row.
    // Dynamic shared layout:
    // - reduce[0:kThreads] is used by blockReduceSum() for a 576-wide dot product.
    // - weights[0:topK] stores one score/softmax weight per sparse KV candidate.
    extern __shared__ float shared[];
    float* reduce = shared;
    float* weights = shared + blockDim.x;

    int const tokenIdx = static_cast<int>(blockIdx.x);
    int const headIdx = static_cast<int>(blockIdx.y);
    int const tid = threadIdx.x;
    // kvDequantScale maps FP8 cache values back to BF16/FP32 units. Context recomputes Q as FP8-rounded BF16
    // below, so Q and K each contribute one dequant scale to the score.
    float const kvDequantScale = kvScaleQuantOrig == nullptr ? 1.0F : kvScaleQuantOrig[0];
    float const scoreScale = kvDequantScale * kvDequantScale * hostScoreScale;

    if (tokenIdx >= numTokens || headIdx >= kLocalHeads)
    {
        return;
    }

    float maxScore = -INFINITY;

    for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
    {
        // topkIndicesPool is INT32 [numTokens, topK]. Negative values are padding and become -inf scores.
        int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
        float partial = 0.0F;
        if (kvIdx >= 0)
        {
            // kvPtr points at FP8 [576] for one sparse KV row. The first 512 dims are V, all 576 dims are K.
            __nv_fp8_e4m3 const* kvPtr = kvCachePool + kvIdx * kHeadDim;
            for (int dim = tid; dim < kHeadDim; dim += blockDim.x)
            {
                // Match the FP8-KV attention path by rounding BF16 fusedQ to E4M3 before the QK dot.
                float const qFp8 = static_cast<float>(
                    __nv_fp8_e4m3(toFloat(fusedQ[(tokenIdx * kLocalHeads + headIdx) * kHeadDim + dim])));
                partial += qFp8 * toFloat(kvPtr[dim]);
            }
        }
        float const dot = blockReduceSum(partial, reduce);
        if (tid == 0)
        {
            float const score = kvIdx >= 0 ? dot * scoreScale : -INFINITY;
            weights[topkIdx] = score;
            maxScore = fmaxf(maxScore, score);
        }
        __syncthreads();
    }

    __shared__ float denomShared;
    if (tid == 0)
    {
        float denom = 0.0F;
        for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
        {
            float const weight = expf(weights[topkIdx] - maxScore);
            weights[topkIdx] = weight;
            denom += weight;
        }
        denomShared = denom;
    }
    __syncthreads();

    float const invDenom = denomShared > 0.0F ? 1.0F / denomShared : 0.0F;
    for (int dim = tid; dim < kKvLoraRank; dim += blockDim.x)
    {
        // Output dim only spans the latent V rank [0, 512), not the 64 RoPE suffix.
        float acc = 0.0F;
        for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
        {
            int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
            if (kvIdx >= 0)
            {
                acc += weights[topkIdx] * invDenom * toFloat(kvCachePool[kvIdx * kHeadDim + dim]);
            }
        }
        output[(tokenIdx * kLocalHeads + headIdx) * kKvLoraRank + dim] = toBfloat16(acc * kvDequantScale);
    }
}

__global__ void generationAttentionKernel(__nv_fp8_e4m3 const* quantQ, __nv_fp8_e4m3 const* kvCachePool,
    int32_t const* topkIndicesPool, int32_t const* topkIndicesLocal, int32_t const* sequenceLength,
    __nv_bfloat16* output, float const* bmm1Scale, float const* bmm2Scale, int numTokens, int topK)
{
    // One CUDA block computes one (generation query token, local head) row.
    // Current Python dispatch restricts this custom path to one generation sequence, but numTokens may be
    // greater than one for MTP/spec decode. In that case each token row has a different causal KV end.
    extern __shared__ float shared[];
    float* reduce = shared;
    float* weights = shared + blockDim.x;

    int const tokenIdx = static_cast<int>(blockIdx.x);
    int const headIdx = static_cast<int>(blockIdx.y);
    int const tid = threadIdx.x;
    // sequenceLength is INT32 [1] after mla_rope_generation appended the entire numTokens decode chunk.
    // For row tokenIdx, valid local KV positions are:
    //   [0, sequenceLength[0] - numTokens + tokenIdx]
    // Example: old length 100 and numTokens 4 -> rows can attend through 100, 101, 102, 103 respectively.
    int const localKvEnd = sequenceLength[0] - numTokens + tokenIdx;
    // bmm1Scale[1] is already in log2 space. Using exp2f() below matches TRTLLM-Gen's softmax convention.
    float const scoreScaleLog2 = bmm1Scale == nullptr ? 1.4426950408889634F : bmm1Scale[1];
    // bmm2Scale[0] dequantizes FP8 V/cache values before storing BF16 latent output.
    float const outputScale = bmm2Scale == nullptr ? 1.0F : bmm2Scale[0];

    if (tokenIdx >= numTokens || headIdx >= kLocalHeads)
    {
        return;
    }

    float maxScore = -INFINITY;

    for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
    {
        // topkIndicesPool: INT32 [numTokens, topK], converted global pool row for data loads.
        // topkIndicesLocal: INT32 [numTokens, topK], original local KV position for causal validity.
        int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
        int const localKvIdx = topkIndicesLocal[tokenIdx * topK + topkIdx];
        bool const validKv = kvIdx >= 0 && localKvIdx >= 0 && localKvIdx <= localKvEnd;
        float partial = 0.0F;
        if (validKv)
        {
            // quantQ is FP8 [numTokens, 8, 576]. It was produced by mla_rope_generation, not by this kernel.
            __nv_fp8_e4m3 const* qPtr = quantQ + (tokenIdx * kLocalHeads + headIdx) * kHeadDim;
            // kvCachePool is FP8 [poolTokens, 1, 576] viewed as [poolTokens, 576].
            __nv_fp8_e4m3 const* kvPtr = kvCachePool + kvIdx * kHeadDim;
            for (int dim = tid; dim < kHeadDim; dim += blockDim.x)
            {
                partial += toFloat(qPtr[dim]) * toFloat(kvPtr[dim]);
            }
        }
        float const dot = blockReduceSum(partial, reduce);
        if (tid == 0)
        {
            float const score = validKv ? dot * scoreScaleLog2 : -INFINITY;
            weights[topkIdx] = score;
            maxScore = fmaxf(maxScore, score);
        }
        __syncthreads();
    }

    __shared__ float denomShared;
    if (tid == 0)
    {
        float denom = 0.0F;
        for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
        {
            float const weight = exp2f(weights[topkIdx] - maxScore);
            weights[topkIdx] = weight;
            denom += weight;
        }
        denomShared = denom;
    }
    __syncthreads();

    float const invDenom = denomShared > 0.0F ? 1.0F / denomShared : 0.0F;
    for (int dim = tid; dim < kKvLoraRank; dim += blockDim.x)
    {
        float acc = 0.0F;
        for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
        {
            int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
            int const localKvIdx = topkIndicesLocal[tokenIdx * topK + topkIdx];
            bool const validKv = kvIdx >= 0 && localKvIdx >= 0 && localKvIdx <= localKvEnd;
            if (validKv)
            {
                acc += weights[topkIdx] * invDenom * toFloat(kvCachePool[kvIdx * kHeadDim + dim]);
            }
        }
        output[(tokenIdx * kLocalHeads + headIdx) * kKvLoraRank + dim] = toBfloat16(acc * outputScale);
    }
}

} // namespace

namespace dsv3_fused_mla
{

// Fused GLM-5/DeepSeek-V3 sparse MLA context path.
//
// Inputs:
// - fused_q: BF16 [numTokens, 8, 576]. On entry, dims [0, 512) contain the absorbed q_nope projection.
//   This function fills dims [512, 576) with rotated Q RoPE values before attention.
// - q_pe: BF16 [numTokens, 8, 64]. Unrotated per-head Q RoPE values.
// - latent_cache: BF16 [numTokens, 576]. Dims [0, 512) are compressed latent KV/V values; dims [512, 576)
//   are unrotated shared K RoPE values. The K RoPE suffix is rotated in-place.
// - topk_indices_pool: INT32 [numTokens, topK]. Global row indices into kv_cache_pool for sparse attention.
//   Negative entries are padding.
// - kv_cache_pool: FP8 E4M3 [poolTokens, 1, 576], viewed by the kernel as [poolTokens, 576]. All 576 dims
//   are used as K; only dims [0, 512) are used as latent V.
// - rotary_cos_sin: FP32 float2 RoPE table indexed by absolute KV position.
// - ctx_cached_token_indptr: INT64 [numContexts + 1]. The current custom path uses one context;
//   ctx_cached_token_indptr[1] - ctx_cached_token_indptr[0] gives the cached prefix length used for
//   absolute RoPE and cache positions.
// - ctx_kv_indptr: currently unused by this simplified CUDA path.
// - kv_cache: paged TRT-LLM KV cache view. preprocessContextKernel writes the rotated latent KV rows into
//   this cache.
// - kv_scale_orig_quant: optional FP32 [1], original-domain to FP8 scale used when storing BF16
//   latent_cache values into an FP8 paged KV cache.
// - kv_scale_quant_orig: optional FP32 [1], FP8 to original-domain scale used for score/output scaling in
//   sparse attention.
// - has_fp8_kv_cache: selects FP8 vs BF16 writes for the paged KV cache update. The current GLM-5 fused MLA
//   wrapper enforces FP8 kv_cache_pool.
// - q_scaling: model attention scaling divisor; score scale is 1 / (q_scaling * sqrt(256)) before FP8
//   dequant factors.
//
// Side effects:
// - Mutates fused_q by writing rotated Q RoPE suffix.
// - Mutates latent_cache by rotating the K RoPE suffix in-place.
// - Mutates kv_cache by appending/writing the context latent KV rows.
//
// Output:
// - Returns BF16 [numTokens, 8 * 512], logically [numTokens, 8, 512]. Each row is the sparse
//   softmax-weighted sum of FP8 latent V rows from kv_cache_pool, in original/BF16 units.
torch::Tensor dsv3_fused_mla_context_cuda(torch::Tensor fused_q, torch::Tensor q_pe, torch::Tensor latent_cache,
    torch::Tensor topk_indices_pool, torch::Tensor kv_cache_pool, torch::Tensor rotary_cos_sin,
    torch::Tensor ctx_cached_token_indptr, torch::Tensor ctx_kv_indptr, tensorrt_llm::kernels::KVBlockArray kv_cache,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    bool has_fp8_kv_cache, double q_scaling)
{
    (void) ctx_kv_indptr;
    c10::cuda::CUDAGuard const deviceGuard{fused_q.device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());

    int const numTokens = static_cast<int>(fused_q.size(0));
    int const topK = static_cast<int>(topk_indices_pool.size(1));
    TORCH_CHECK(topK <= kMaxTopK, "dsv3_fused_mla_context supports topK <= ", kMaxTopK, ", got ", topK);
    auto output = torch::empty({numTokens, kLocalHeads * kKvLoraRank}, fused_q.options());

    float const* kvScaleOrigQuantPtr
        = kv_scale_orig_quant.has_value() ? kv_scale_orig_quant.value().data_ptr<float>() : nullptr;
    float const* kvScaleQuantOrigPtr
        = kv_scale_quant_orig.has_value() ? kv_scale_quant_orig.value().data_ptr<float>() : nullptr;
    auto* rotaryCosSinPtr = reinterpret_cast<float2 const*>(rotary_cos_sin.data_ptr());
    auto* fusedQPtr = static_cast<__nv_bfloat16*>(fused_q.data_ptr());
    auto* qPePtr = static_cast<__nv_bfloat16*>(q_pe.data_ptr());
    auto* latentCachePtr = static_cast<__nv_bfloat16*>(latent_cache.data_ptr());
    auto* kvCachePoolPtr = reinterpret_cast<__nv_fp8_e4m3 const*>(kv_cache_pool.data_ptr());
    auto* outputPtr = static_cast<__nv_bfloat16*>(output.data_ptr());

    dim3 preprocessGrid(numTokens, kLocalHeads + 1);
    if (has_fp8_kv_cache)
    {
        preprocessContextKernel<true><<<preprocessGrid, kThreads, 0, stream>>>(fusedQPtr, latentCachePtr, qPePtr,
            rotaryCosSinPtr, ctx_cached_token_indptr.data_ptr<int64_t>(), kv_cache, kvScaleOrigQuantPtr, numTokens);
    }
    else
    {
        preprocessContextKernel<false><<<preprocessGrid, kThreads, 0, stream>>>(fusedQPtr, latentCachePtr, qPePtr,
            rotaryCosSinPtr, ctx_cached_token_indptr.data_ptr<int64_t>(), kv_cache, kvScaleOrigQuantPtr, numTokens);
    }

    float const hostScoreScale = 1.0F / (static_cast<float>(q_scaling) * sqrtf(256.0F));
    dim3 attentionGrid(numTokens, kLocalHeads);
    size_t const sharedBytes = (kThreads + std::min(topK, kMaxTopK)) * sizeof(float);
    contextAttentionKernel<<<attentionGrid, kThreads, sharedBytes, stream>>>(fusedQPtr, kvCachePoolPtr,
        topk_indices_pool.data_ptr<int32_t>(), outputPtr, kvScaleQuantOrigPtr, numTokens, topK, hostScoreScale);

    sync_check_cuda_error(stream);
    return output;
}

torch::Tensor dsv3_fused_mla_generation_cuda(torch::Tensor fused_q, torch::Tensor topk_indices_pool,
    torch::Tensor topk_indices, torch::Tensor kv_cache_pool, torch::Tensor sequence_length,
    torch::Tensor quant_q_buffer, torch::Tensor mla_bmm1_scale, torch::Tensor mla_bmm2_scale)
{
    // fused_q is BF16 [numTokens, 8, 576]. The generation attention kernel does not read it directly, but its
    // shape/device/dtype define the output allocation and are checked in the THOP wrapper.
    c10::cuda::CUDAGuard const deviceGuard{fused_q.device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());

    int const numTokens = static_cast<int>(fused_q.size(0));
    int const topK = static_cast<int>(topk_indices_pool.size(1));
    TORCH_CHECK(topK <= kMaxTopK, "dsv3_fused_mla_generation supports topK <= ", kMaxTopK, ", got ", topK);
    auto output = torch::empty({numTokens, kLocalHeads * kKvLoraRank}, fused_q.options());

    auto* quantQPtr = reinterpret_cast<__nv_fp8_e4m3 const*>(quant_q_buffer.data_ptr());
    auto* kvCachePoolPtr = reinterpret_cast<__nv_fp8_e4m3 const*>(kv_cache_pool.data_ptr());
    auto* outputPtr = static_cast<__nv_bfloat16*>(output.data_ptr());

    dim3 attentionGrid(numTokens, kLocalHeads);
    size_t const sharedBytes = (kThreads + topK) * sizeof(float);
    generationAttentionKernel<<<attentionGrid, kThreads, sharedBytes, stream>>>(quantQPtr, kvCachePoolPtr,
        topk_indices_pool.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(), sequence_length.data_ptr<int32_t>(),
        outputPtr, mla_bmm1_scale.data_ptr<float>(), mla_bmm2_scale.data_ptr<float>(), numTokens, topK);

    sync_check_cuda_error(stream);
    return output;
}

} // namespace dsv3_fused_mla
