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

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace
{

constexpr int kLocalHeads = 8;
constexpr int kKvLoraRank = 512;
constexpr int kRopeDim = 64;
constexpr int kHeadDim = kKvLoraRank + kRopeDim;
constexpr int kGenerationAttentionThreads = 256;
constexpr int kSparseSplitSize = 64;

// Convert one FP8 E4M3 scalar to FP32.
__device__ __forceinline__ float toFloat(__nv_fp8_e4m3 value)
{
    return static_cast<float>(value);
}

// Round one FP32 scalar to BF16.
__device__ __forceinline__ __nv_bfloat16 toBfloat16(float value)
{
    return __float2bfloat16(value);
}

// Sum one scalar contribution across the current warp.
__device__ __forceinline__ float warpReduceSum(float value)
{
    unsigned const mask = 0xFFFFFFFFU;
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

// Check whether a selected sparse KV row is visible to a generation/MTP query token.
__device__ __forceinline__ bool isGenerationKvVisible(
    int tokenIdx, int localKvIdx, int currentGroupStart, int numTokens)
{
    int const currentGroupOffset = localKvIdx - currentGroupStart;
    bool const historicalKv = localKvIdx >= 0 && localKvIdx < currentGroupStart;
    bool const currentGroupKv = currentGroupOffset >= 0 && currentGroupOffset < numTokens;
    if (historicalKv)
    {
        return true;
    }
    if (!currentGroupKv)
    {
        return false;
    }
    return currentGroupOffset <= tokenIdx;
}

// Synchronize all split CTAs in the standalone generation attention launch.
__device__ __forceinline__ void generationGridBarrier(int32_t* syncScratch, int expectedCtas, bool needsGlobalFence)
{
    __syncthreads();
    if (needsGlobalFence)
    {
        __threadfence();
    }
    __syncthreads();

    __shared__ bool isLastCtaShared;
    if (threadIdx.x == 0)
    {
        int const arrival = atomicAdd(syncScratch, 1);
        isLastCtaShared = arrival == expectedCtas - 1;
    }
    __syncthreads();

    if (isLastCtaShared)
    {
        syncScratch[0] = 0;
        __threadfence();
        atomicExch(syncScratch + 1, 1);
    }
    else
    {
        while (atomicAdd(syncScratch + 1, 0) == 0)
        {
            __nanosleep(2048U);
        }
    }
    __syncthreads();
}

// Standalone TileRT-style split sparse MLA attention kernel for GLM-5 generation.
//
// One CTA owns one [token, sparse top-k split]. Warps 0..7 map one local MLA head per warp, compute split-local QK
// scores, normalize the split with exp2 softmax, write split-local AV results, then grid-sync and combine all splits
// into BF16 output.
__global__ void flashLatentAttention(__nv_fp8_e4m3 const* quantQ, __nv_fp8_e4m3 const* kvCachePool,
    int32_t const* topkIndicesPool, int32_t const* topkIndicesLocal, int32_t const* sequenceLength, float* splitLse,
    float* splitOutput, int32_t* syncScratch, __nv_bfloat16* output, float const* kvScaleQuantOrig, int numTokens,
    int topK, int numSplits, int expectedCtas, float hostScoreScale, bool isContext, int contextChunkStart)
{
    extern __shared__ float splitScores[];
    int const tokenIdx = static_cast<int>(blockIdx.x) / numSplits;
    int const splitIdx = static_cast<int>(blockIdx.x) - tokenIdx * numSplits;
    int const tid = threadIdx.x;
    int const warpIdx = tid / 32;
    int const laneIdx = tid & 31;

    __shared__ int currentGroupStartShared;
    __shared__ float scoreScaleLog2Shared;
    __shared__ float outputScaleShared;
    if (tid == 0)
    {
        int currentGroupStart = isContext ? contextChunkStart : sequenceLength[0] - numTokens;
        if (!isContext && sequenceLength[0] <= numTokens)
        {
            int maxLocalIdx = -1;
            for (int idx = 0; idx < numTokens * topK; ++idx)
            {
                int const localIdx = topkIndicesLocal[idx];
                maxLocalIdx = localIdx > maxLocalIdx ? localIdx : maxLocalIdx;
            }
            if (maxLocalIdx >= numTokens)
            {
                currentGroupStart = maxLocalIdx - (numTokens - 1);
            }
        }
        currentGroupStartShared = currentGroupStart;

        float const dequantScale = kvScaleQuantOrig == nullptr ? 1.0F : kvScaleQuantOrig[0];
        constexpr float kLog2e = 1.4426950408889634074F;
        float const bmm1ScaleValue = dequantScale * dequantScale * hostScoreScale;
        scoreScaleLog2Shared = bmm1ScaleValue * kLog2e;
        outputScaleShared = dequantScale;
    }
    __syncthreads();

    if (tokenIdx >= numTokens || splitIdx >= numSplits)
    {
        return;
    }

    int const currentGroupStart = currentGroupStartShared;
    __shared__ int64_t validKvBaseOffsets[kSparseSplitSize];
    int const splitStart = splitIdx * kSparseSplitSize;
    if (tid < kSparseSplitSize)
    {
        int const topkIdx = splitStart + tid;
        bool const inRange = topkIdx < topK;
        int const kvIdx = inRange ? topkIndicesPool[tokenIdx * topK + topkIdx] : -1;
        int const localKvIdx = inRange ? topkIndicesLocal[tokenIdx * topK + topkIdx] : -1;
        bool const validKv = kvIdx >= 0 && isGenerationKvVisible(tokenIdx, localKvIdx, currentGroupStart, numTokens);
        validKvBaseOffsets[tid] = validKv ? static_cast<int64_t>(kvIdx) * kHeadDim : -1;
    }
    __syncthreads();

    {
        int const headIdx = warpIdx;
        float* headScores = splitScores + headIdx * kSparseSplitSize;
        float const scoreScaleLog2 = scoreScaleLog2Shared;
        float maxScore = -INFINITY;

        __nv_fp8_e4m3 const* qPtr = quantQ + (tokenIdx * kLocalHeads + headIdx) * kHeadDim;
        float const qValue0 = toFloat(qPtr[laneIdx]);
        float const qValue1 = toFloat(qPtr[laneIdx + 32]);
        float const qValue2 = toFloat(qPtr[laneIdx + 64]);
        float const qValue3 = toFloat(qPtr[laneIdx + 96]);
        float const qValue4 = toFloat(qPtr[laneIdx + 128]);
        float const qValue5 = toFloat(qPtr[laneIdx + 160]);
        float const qValue6 = toFloat(qPtr[laneIdx + 192]);
        float const qValue7 = toFloat(qPtr[laneIdx + 224]);
        float const qValue8 = toFloat(qPtr[laneIdx + 256]);
        float const qValue9 = toFloat(qPtr[laneIdx + 288]);
        float const qValue10 = toFloat(qPtr[laneIdx + 320]);
        float const qValue11 = toFloat(qPtr[laneIdx + 352]);
        float const qValue12 = toFloat(qPtr[laneIdx + 384]);
        float const qValue13 = toFloat(qPtr[laneIdx + 416]);
        float const qValue14 = toFloat(qPtr[laneIdx + 448]);
        float const qValue15 = toFloat(qPtr[laneIdx + 480]);
        float const qValue16 = toFloat(qPtr[laneIdx + 512]);
        float const qValue17 = toFloat(qPtr[laneIdx + 544]);

        for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
        {
            int64_t const kvBaseOffset = validKvBaseOffsets[splitOffset];
            bool const validKv = kvBaseOffset >= 0;
            int64_t const kvLoadBaseOffset = validKv ? kvBaseOffset : 0;
            float partial = 0.0F;
            __nv_fp8_e4m3 const* kvPtr = kvCachePool + kvLoadBaseOffset;
            partial += qValue0 * toFloat(kvPtr[laneIdx]);
            partial += qValue1 * toFloat(kvPtr[laneIdx + 32]);
            partial += qValue2 * toFloat(kvPtr[laneIdx + 64]);
            partial += qValue3 * toFloat(kvPtr[laneIdx + 96]);
            partial += qValue4 * toFloat(kvPtr[laneIdx + 128]);
            partial += qValue5 * toFloat(kvPtr[laneIdx + 160]);
            partial += qValue6 * toFloat(kvPtr[laneIdx + 192]);
            partial += qValue7 * toFloat(kvPtr[laneIdx + 224]);
            partial += qValue8 * toFloat(kvPtr[laneIdx + 256]);
            partial += qValue9 * toFloat(kvPtr[laneIdx + 288]);
            partial += qValue10 * toFloat(kvPtr[laneIdx + 320]);
            partial += qValue11 * toFloat(kvPtr[laneIdx + 352]);
            partial += qValue12 * toFloat(kvPtr[laneIdx + 384]);
            partial += qValue13 * toFloat(kvPtr[laneIdx + 416]);
            partial += qValue14 * toFloat(kvPtr[laneIdx + 448]);
            partial += qValue15 * toFloat(kvPtr[laneIdx + 480]);
            partial += qValue16 * toFloat(kvPtr[laneIdx + 512]);
            partial += qValue17 * toFloat(kvPtr[laneIdx + 544]);
            float const dot = warpReduceSum(partial);
            if (laneIdx == 0)
            {
                float const score = validKv ? dot * scoreScaleLog2 : -INFINITY;
                headScores[splitOffset] = score;
                maxScore = fmaxf(maxScore, score);
            }
        }

        if (laneIdx == 0)
        {
            float denom = 0.0F;
            if (maxScore != -INFINITY)
            {
                for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
                {
                    float const unnormalized = exp2f(headScores[splitOffset] - maxScore);
                    denom += unnormalized;
                    headScores[splitOffset] = unnormalized;
                }
                float const invDenom = 1.0F / denom;
                for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
                {
                    headScores[splitOffset] *= invDenom;
                }
                splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + splitIdx] = maxScore + log2f(denom);
            }
            else
            {
                for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
                {
                    headScores[splitOffset] = 0.0F;
                }
                splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + splitIdx] = -INFINITY;
            }
        }
        __syncwarp();

        constexpr int kAvDimsPerLaneGroup = 8;
        static_assert(kKvLoraRank % (32 * kAvDimsPerLaneGroup) == 0);
        for (int dimBase = laneIdx; dimBase < kKvLoraRank; dimBase += 32 * kAvDimsPerLaneGroup)
        {
            float acc0 = 0.0F;
            float acc1 = 0.0F;
            float acc2 = 0.0F;
            float acc3 = 0.0F;
            float acc4 = 0.0F;
            float acc5 = 0.0F;
            float acc6 = 0.0F;
            float acc7 = 0.0F;
            for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
            {
                int64_t const kvBaseOffset = validKvBaseOffsets[splitOffset];
                int64_t const kvLoadBaseOffset = kvBaseOffset >= 0 ? kvBaseOffset : 0;
                float const probability = headScores[splitOffset];
                acc0 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase]);
                acc1 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase + 32]);
                acc2 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase + 64]);
                acc3 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase + 96]);
                acc4 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase + 128]);
                acc5 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase + 160]);
                acc6 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase + 192]);
                acc7 += probability * toFloat(kvCachePool[kvLoadBaseOffset + dimBase + 224]);
            }
            int64_t const splitOutputBase
                = ((tokenIdx * kLocalHeads + headIdx) * numSplits + splitIdx) * kKvLoraRank + dimBase;
            splitOutput[splitOutputBase] = acc0;
            splitOutput[splitOutputBase + 32] = acc1;
            splitOutput[splitOutputBase + 64] = acc2;
            splitOutput[splitOutputBase + 96] = acc3;
            splitOutput[splitOutputBase + 128] = acc4;
            splitOutput[splitOutputBase + 160] = acc5;
            splitOutput[splitOutputBase + 192] = acc6;
            splitOutput[splitOutputBase + 224] = acc7;
        }
    }

    generationGridBarrier(syncScratch, expectedCtas, true);

    if (splitIdx < kLocalHeads)
    {
        __shared__ float globalLseShared;
        for (int headIdx = splitIdx; headIdx < kLocalHeads; headIdx += numSplits)
        {
            if (tid == 0)
            {
                float maxLse = -INFINITY;
                for (int combineSplitIdx = 0; combineSplitIdx < numSplits; ++combineSplitIdx)
                {
                    float const lse = splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + combineSplitIdx];
                    maxLse = fmaxf(maxLse, lse);
                }

                if (maxLse == -INFINITY)
                {
                    globalLseShared = -INFINITY;
                }
                else
                {
                    float denom = 0.0F;
                    for (int combineSplitIdx = 0; combineSplitIdx < numSplits; ++combineSplitIdx)
                    {
                        float const lse = splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + combineSplitIdx];
                        denom += exp2f(lse - maxLse);
                    }
                    globalLseShared = maxLse + log2f(denom);
                }
            }
            __syncthreads();

            float const outputScale = outputScaleShared;
            float const globalLse = globalLseShared;
            float* splitWeights = splitScores;
            for (int combineSplitIdx = tid; combineSplitIdx < numSplits; combineSplitIdx += kGenerationAttentionThreads)
            {
                float const lse = splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + combineSplitIdx];
                splitWeights[combineSplitIdx] = lse == -INFINITY ? 0.0F : exp2f(lse - globalLse);
            }
            __syncthreads();

            constexpr int kCombineDimsPerThread = 4;
            static_assert(kKvLoraRank % kCombineDimsPerThread == 0);
            if (tid < kKvLoraRank / kCombineDimsPerThread)
            {
                int const dim0 = tid;
                int const dim1 = tid + kKvLoraRank / kCombineDimsPerThread;
                int const dim2 = tid + 2 * (kKvLoraRank / kCombineDimsPerThread);
                int const dim3 = tid + 3 * (kKvLoraRank / kCombineDimsPerThread);
                float acc0 = 0.0F;
                float acc1 = 0.0F;
                float acc2 = 0.0F;
                float acc3 = 0.0F;
                int64_t const splitOutputBase = (tokenIdx * kLocalHeads + headIdx) * numSplits * kKvLoraRank;
                float const* splitOutputPtr0 = splitOutput + splitOutputBase + dim0;
                float const* splitOutputPtr1 = splitOutput + splitOutputBase + dim1;
                float const* splitOutputPtr2 = splitOutput + splitOutputBase + dim2;
                float const* splitOutputPtr3 = splitOutput + splitOutputBase + dim3;
                for (int combineSplitIdx = 0; combineSplitIdx < numSplits; ++combineSplitIdx)
                {
                    float const splitWeight = splitWeights[combineSplitIdx];
                    float const splitValue0 = *splitOutputPtr0;
                    float const splitValue1 = *splitOutputPtr1;
                    float const splitValue2 = *splitOutputPtr2;
                    float const splitValue3 = *splitOutputPtr3;
                    acc0 += splitWeight * splitValue0;
                    acc1 += splitWeight * splitValue1;
                    acc2 += splitWeight * splitValue2;
                    acc3 += splitWeight * splitValue3;
                    splitOutputPtr0 += kKvLoraRank;
                    splitOutputPtr1 += kKvLoraRank;
                    splitOutputPtr2 += kKvLoraRank;
                    splitOutputPtr3 += kKvLoraRank;
                }
                int64_t const outputBase = (tokenIdx * kLocalHeads + headIdx) * kKvLoraRank;
                output[outputBase + dim0] = toBfloat16(acc0 * outputScale);
                output[outputBase + dim1] = toBfloat16(acc1 * outputScale);
                output[outputBase + dim2] = toBfloat16(acc2 * outputScale);
                output[outputBase + dim3] = toBfloat16(acc3 * outputScale);
            }
            __syncthreads();
        }
    }
}

} // namespace

namespace dsv3_fused_mla::detail
{

void launchGenerationFlashLatentAttention(dim3 splitGrid, size_t splitSharedBytes, cudaStream_t stream,
    __nv_fp8_e4m3 const* quantQ, __nv_fp8_e4m3 const* kvCachePool, int32_t const* topkIndicesPool,
    int32_t const* topkIndicesLocal, int32_t const* sequenceLength, float* splitLse, float* splitOutput,
    int32_t* syncScratch, __nv_bfloat16* output, float const* kvScaleQuantOrig, int numTokens, int topK, int numSplits,
    int expectedCtas, float hostScoreScale, bool isContext, int contextChunkStart)
{
    flashLatentAttention<<<splitGrid, kGenerationAttentionThreads, splitSharedBytes, stream>>>(quantQ, kvCachePool,
        topkIndicesPool, topkIndicesLocal, sequenceLength, splitLse, splitOutput, syncScratch, output, kvScaleQuantOrig,
        numTokens, topK, numSplits, expectedCtas, hostScoreScale, isContext, contextChunkStart);
}

} // namespace dsv3_fused_mla::detail
