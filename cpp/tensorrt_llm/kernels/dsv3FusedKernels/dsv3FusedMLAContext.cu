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

__device__ __forceinline__ float toFloat(__nv_bfloat16 value)
{
    return __bfloat162float(value);
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

template <bool kFp8KvCache>
__global__ void preprocessContextKernel(__nv_bfloat16* fusedQ, __nv_bfloat16* latentCache, __nv_bfloat16 const* qPe,
    float2 const* rotaryCosSin, int64_t const* ctxCachedTokenIndptr, tensorrt_llm::kernels::KVBlockArray kvCache,
    float const* kvScaleOrigQuant, int numTokens)
{
    int const tokenIdx = static_cast<int>(blockIdx.x);
    int const headIdx = static_cast<int>(blockIdx.y);
    int64_t const cachedLen = ctxCachedTokenIndptr[1] - ctxCachedTokenIndptr[0];
    int const tokenIdxInKvCache = static_cast<int>(cachedLen) + tokenIdx;
    float const kvScale = kvScaleOrigQuant == nullptr ? 1.0F : kvScaleOrigQuant[0];

    if (tokenIdx >= numTokens)
    {
        return;
    }

    if (headIdx < kLocalHeads)
    {
        for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
        {
            float2 value;
            int const dim = 2 * pairIdx;
            int const qPeOffset = (tokenIdx * kLocalHeads + headIdx) * kRopeDim + dim;
            value.x = toFloat(qPe[qPeOffset]);
            value.y = toFloat(qPe[qPeOffset + 1]);
            float2 const coef = rotaryCosSin[tokenIdxInKvCache * kRopeDim + pairIdx];
            float2 const rotated = rotaryTransform(value, coef);

            int const fusedOffset = (tokenIdx * kLocalHeads + headIdx) * kHeadDim + kKvLoraRank + dim;
            fusedQ[fusedOffset] = toBfloat16(rotated.x);
            fusedQ[fusedOffset + 1] = toBfloat16(rotated.y);
        }
        return;
    }

    for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
    {
        float2 value;
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

__global__ void contextAttentionKernel(__nv_bfloat16 const* fusedQ, __nv_bfloat16 const* latentCache,
    int32_t const* topkIndices, __nv_bfloat16* output, int numTokens, int topK, float scoreScale)
{
    extern __shared__ float shared[];
    float* reduce = shared;
    float* weights = shared + blockDim.x;

    int const tokenIdx = static_cast<int>(blockIdx.x);
    int const headIdx = static_cast<int>(blockIdx.y);
    int const tid = threadIdx.x;

    if (tokenIdx >= numTokens || headIdx >= kLocalHeads)
    {
        return;
    }

    __nv_bfloat16 const* qPtr = fusedQ + (tokenIdx * kLocalHeads + headIdx) * kHeadDim;
    float maxScore = -INFINITY;

    for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
    {
        int const kvIdx = topkIndices[tokenIdx * topK + topkIdx];
        float partial = 0.0F;
        if (kvIdx >= 0 && kvIdx <= tokenIdx && kvIdx < numTokens)
        {
            __nv_bfloat16 const* kvPtr = latentCache + kvIdx * kHeadDim;
            for (int dim = tid; dim < kHeadDim; dim += blockDim.x)
            {
                partial += toFloat(qPtr[dim]) * toFloat(kvPtr[dim]);
            }
        }
        float const dot = blockReduceSum(partial, reduce);
        if (tid == 0)
        {
            float const score = (kvIdx >= 0 && kvIdx <= tokenIdx && kvIdx < numTokens) ? dot * scoreScale : -INFINITY;
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
        float acc = 0.0F;
        for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
        {
            int const kvIdx = topkIndices[tokenIdx * topK + topkIdx];
            if (kvIdx >= 0 && kvIdx <= tokenIdx && kvIdx < numTokens)
            {
                acc += weights[topkIdx] * invDenom * toFloat(latentCache[kvIdx * kHeadDim + dim]);
            }
        }
        output[(tokenIdx * kLocalHeads + headIdx) * kKvLoraRank + dim] = toBfloat16(acc);
    }
}

} // namespace

namespace dsv3_fused_mla
{

torch::Tensor dsv3_fused_mla_context_cuda(torch::Tensor fused_q, torch::Tensor q_pe, torch::Tensor latent_cache,
    torch::Tensor topk_indices, torch::Tensor rotary_cos_sin, torch::Tensor ctx_cached_token_indptr,
    torch::Tensor ctx_kv_indptr, tensorrt_llm::kernels::KVBlockArray kv_cache,
    std::optional<torch::Tensor> kv_scale_orig_quant, bool has_fp8_kv_cache, double q_scaling)
{
    (void) ctx_kv_indptr;
    c10::cuda::CUDAGuard const deviceGuard{fused_q.device()};
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());

    int const numTokens = static_cast<int>(fused_q.size(0));
    int const topK = static_cast<int>(topk_indices.size(1));
    auto output = torch::empty({numTokens, kLocalHeads * kKvLoraRank}, fused_q.options());

    float const* kvScaleOrigQuantPtr
        = kv_scale_orig_quant.has_value() ? kv_scale_orig_quant.value().data_ptr<float>() : nullptr;
    auto* rotaryCosSinPtr = reinterpret_cast<float2 const*>(rotary_cos_sin.data_ptr());
    auto* fusedQPtr = static_cast<__nv_bfloat16*>(fused_q.data_ptr());
    auto* qPePtr = static_cast<__nv_bfloat16*>(q_pe.data_ptr());
    auto* latentCachePtr = static_cast<__nv_bfloat16*>(latent_cache.data_ptr());
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

    float const scoreScale = 1.0F / (static_cast<float>(q_scaling) * sqrtf(256.0F));
    dim3 attentionGrid(numTokens, kLocalHeads);
    size_t const sharedBytes = (kThreads + std::min(topK, kMaxTopK)) * sizeof(float);
    contextAttentionKernel<<<attentionGrid, kThreads, sharedBytes, stream>>>(
        fusedQPtr, latentCachePtr, topk_indices.data_ptr<int32_t>(), outputPtr, numTokens, topK, scoreScale);

    sync_check_cuda_error(stream);
    return output;
}

} // namespace dsv3_fused_mla
