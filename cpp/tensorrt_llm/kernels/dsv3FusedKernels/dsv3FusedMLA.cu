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
// #include "tensorrt_llm/kernels/decoderMaskedMultiheadAttentionUtils.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>

// #include <cmath>
#include <cstdint>
#include <optional>

namespace
{

constexpr int kLocalHeads = 8;
constexpr int kKvLoraRank = 512;
constexpr int kQkNopeHeadDim = 192;
constexpr int kRopeDim = 64;
constexpr int kHeadDim = kKvLoraRank + kRopeDim;
constexpr int kThreads = 256;
constexpr int kTileRtThreads = 384;
constexpr int kTileRtConsumerThreads = 256;
constexpr int kSparseSplitSize = 64;
constexpr int kTileRtMaxResidentSplitCtas = 128;
constexpr int kMaxTopK = 4096;

// This file is intentionally specialized to the GLM-5/DeepSeek-V3 fused MLA shape used by the WIP Python path.
// The constants above are part of that specialization:
// - kLocalHeads is the tensor-parallel local attention head count on TP=8.
// - kKvLoraRank is the absorbed latent KV/V dimension used by MLA.
// - kQkNopeHeadDim is the unabsorbed q_nope dimension before multiplying by k_b_proj.
// - kRopeDim is the per-head RoPE suffix dimension.
// - kHeadDim is the full latent attention dimension, [latent KV/V, RoPE] == 512 + 64.
// - kThreads is the fixed CTA width used by the context path and combine reductions.
// - kTileRtThreads is the TileRT-style generation split CTA width: 12 warps.
// - kTileRtConsumerThreads is the TileRT-style consumer group: 8 warps, one local head per warp.
// - kSparseSplitSize is TileRT's 64 selected-KV-row split size, giving 32 splits for topK=2048.
// - kTileRtMaxResidentSplitCtas is the currently supported split-CTA grid cap for the combined generation kernel.
//   The in-kernel global rendezvous below requires all launched split CTAs to be resident concurrently.
// - kMaxTopK bounds dynamic shared memory for sparse top-k attention weights.
//
//
// Generation input tensors:
// - quantQ:          FP8 E4M3, logical shape [num_gen_tokens, 8 local heads, 576]. The combined generation kernel
//                    writes it after applying Q RoPE and quantizing the full absorbed Q.
// - topkIndicesLocal: INT32, logical shape [num_gen_tokens, topK]. Each non-negative entry is the local
//                    KV position inside the single generation sequence. This is used only for causal
//                    validity checks.
// - topkIndicesPool: INT32, logical shape [num_gen_tokens, topK]. Each non-negative entry is the global
//                    kvCachePool row to load after the DSA index conversion.
// - sequenceLength:  INT32, logical shape [1] on the currently enabled custom path. It is the total KV
//                    length after the current combined kernel appends all num_gen_tokens rows.
// - bmm1Scale:       FP32, logical shape [2]. The combined generation kernel writes both the natural-exp score scale
//                    and the log2-space score scale used with exp2f().
// - bmm2Scale:       FP32, logical shape [1]. The combined generation kernel writes the FP8 V dequantization scale
//                    before consuming it for BF16 output.
// - output:          BF16, logical shape [num_tokens, 8 * 512]. It stores latent attention output before
//                    the per-head V-up projection in Python.

// Convert one BF16 scalar to FP32.
//
// Inputs:
// - value: __nv_bfloat16 scalar in original model units.
//
// Output:
// - float scalar with the exact BF16 value widened to FP32.
//
// Side effects: none.
__device__ __forceinline__ float toFloat(__nv_bfloat16 value)
{
    // CUDA provides the canonical BF16-to-FP32 widening intrinsic.
    return __bfloat162float(value);
}

// Convert one FP8 E4M3 scalar to FP32.
//
// Inputs:
// - value: __nv_fp8_e4m3 scalar, used for quantized Q/K/V cache values.
//
// Output:
// - float scalar after CUDA's FP8 E4M3 dequantization to FP32.
//
// Side effects: none.
__device__ __forceinline__ float toFloat(__nv_fp8_e4m3 value)
{
    // The CUDA FP8 type implements conversion to float through static_cast.
    return static_cast<float>(value);
}

// Round one FP32 scalar to BF16.
//
// Inputs:
// - value: float scalar, typically an accumulator or rotated RoPE component.
//
// Output:
// - __nv_bfloat16 scalar rounded with CUDA's BF16 conversion rules.
//
// Side effects: none.
__device__ __forceinline__ __nv_bfloat16 toBfloat16(float value)
{
    // This is the same scalar rounding primitive used by CUDA BF16 math helpers.
    return __float2bfloat16(value);
}

// Apply one two-dimensional RoPE rotation.
//
// Inputs:
// - value: float2, one adjacent RoPE pair [x_even, x_odd] in original/BF16-widened units.
// - coef: float2, precomputed [cos(theta), sin(theta)] for the absolute token position and pair index.
//
// Output:
// - float2, rotated pair [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos].
//
// Side effects: none.
__device__ __forceinline__ float2 rotaryTransform(float2 value, float2 coef)
{
    // Allocate the two-lane result in registers.
    float2 rotated;
    // Real component of complex multiplication (x + i y) * (cos + i sin).
    rotated.x = coef.x * value.x - coef.y * value.y;
    // Imaginary component of complex multiplication (x + i y) * (cos + i sin).
    rotated.y = coef.x * value.y + coef.y * value.x;
    // Return the rotated adjacent RoPE pair.
    return rotated;
}

// Sum one scalar contribution across all threads in the current CUDA block.
//
// Inputs:
// - value: float scalar contribution owned by the calling thread.
// - shared: float shared-memory buffer with at least blockDim.x elements.
//
// Output:
// - float scalar equal to the sum of value over all threads in the CTA, returned by every thread.
//
// Side effects:
// - Uses shared[0:blockDim.x] as scratch.
// - Synchronizes the whole CTA during the tree reduction.
__device__ __forceinline__ float blockReduceSum(float value, float* shared)
{
    // Cache the lane id because this helper is called many times inside top-k loops.
    int const tid = threadIdx.x;
    // Store each thread's partial dot-product contribution into shared memory.
    shared[tid] = value;
    // Ensure all partials are visible before the reduction tree starts.
    __syncthreads();
    // Halve the active reduction width each iteration: 256 -> 128 -> ... -> 1.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        // Threads in the lower half add the partner value from the upper half.
        if (tid < stride)
        {
            // Accumulate the partner contribution in place.
            shared[tid] += shared[tid + stride];
        }
        // Synchronize after each tree level before the next level reads shared.
        __syncthreads();
    }
    // shared[0] holds the CTA-wide sum; every thread returns it for uniform control flow.
    return shared[0];
}

// Sum one scalar contribution across the current warp.
//
// Inputs:
// - value: float scalar contribution owned by the calling lane.
//
// Output:
// - float scalar equal to the sum of value over all 32 lanes in the warp.
//
// Side effects: none.
__device__ __forceinline__ float warpReduceSum(float value)
{
    // All GLM-5 WIP decode attention CTAs use full warps for head-local reductions.
    unsigned const mask = 0xFFFFFFFFU;
    // Tree-reduce inside the warp: 32 -> 16 -> 8 -> 4 -> 2 -> 1 active values.
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        // Add the partner lane's contribution at the current tree distance.
        value += __shfl_down_sync(mask, value, offset);
    }
    // Lane 0 receives the sum. Other lanes receive partial tree values that are ignored by callers.
    return value;
}

// Check whether a selected sparse KV row is visible to a generation/MTP query token.
//
// Inputs:
// - tokenIdx: int, query token row inside the current decode/MTP group.
// - localKvIdx: int, selected local sequence position from topkIndicesLocal.
// - currentGroupStart: int, first local position of the just-appended decode/MTP group.
// - numTokens: int, number of rows in the current decode/MTP group.
// - specDecodingPackedMask: optional INT32 [numTokens, specMaskWords] view for request 0.
// - specMaskWords: int, number of packed int32 mask words per token.
//
// Output:
// - bool, true when localKvIdx is historical or allowed by the current-group speculative/causal mask.
//
// Side effects: none.
__device__ __forceinline__ bool isGenerationKvVisible(int tokenIdx, int localKvIdx, int currentGroupStart,
    int numTokens, int32_t const* specDecodingPackedMask, int specMaskWords)
{
    // Offset 0 means the first token in the current MTP group; negative means historical KV.
    int const currentGroupOffset = localKvIdx - currentGroupStart;
    // Historical KV rows precede the current MTP group and are always visible when selected.
    bool const historicalKv = localKvIdx >= 0 && localKvIdx < currentGroupStart;
    // Current-group rows are the just-appended decode/MTP tokens.
    bool const currentGroupKv = currentGroupOffset >= 0 && currentGroupOffset < numTokens;

    // Historical rows do not need an intra-group visibility check.
    if (historicalKv)
    {
        return true;
    }
    // Negative padding entries and rows outside the active group are invalid.
    if (!currentGroupKv)
    {
        return false;
    }

    // Prefer explicit speculative/MTP visibility when the packed mask is available.
    if (specDecodingPackedMask != nullptr && specMaskWords > 0)
    {
        // Select which int32 word contains the bit for this current-group offset.
        int const wordIdx = currentGroupOffset / 32;
        // Select the bit inside that int32 word.
        int const bitIdx = currentGroupOffset % 32;
        // The current custom path is restricted to one request, so the request dimension is implicit.
        uint32_t const packed = static_cast<uint32_t>(specDecodingPackedMask[tokenIdx * specMaskWords + wordIdx]);
        // A set bit means query tokenIdx may attend to this current-group KV row.
        return ((packed >> bitIdx) & 1U) != 0U;
    }

    // No packed mask: fall back to local causal order inside the current group.
    return currentGroupOffset <= tokenIdx;
}

// Synchronize all split CTAs in the single combined generation attention launch.
//
// Inputs:
// - syncScratch: INT32 [2]. syncScratch[0] is the arrival counter, syncScratch[1] is the release flag.
// - expectedCtas: int, exact number of split CTAs launched by the kernel grid.
//
// Output: none.
//
// Side effects:
// - Uses global atomics and a spin wait to form a one-shot grid-wide barrier.
// - Requires every launched CTA to be resident concurrently. The host path enforces this for the narrow GLM-5 WIP
//   configuration by capping the split grid to kTileRtMaxResidentSplitCtas.
__device__ __forceinline__ void generationGridBarrier(int32_t* syncScratch, int expectedCtas)
{
    // Ensure splitLse/splitOutput writes are visible to other CTAs before this CTA announces arrival.
    __threadfence();

    // Store whether this CTA is the last arrival so every thread can observe the decision after the CTA barrier.
    __shared__ bool isLastCtaShared;
    if (threadIdx.x == 0)
    {
        // Atomically reserve one arrival slot in the global counter.
        int const arrival = atomicAdd(syncScratch, 1);
        // The last CTA is responsible for releasing all waiting CTAs.
        isLastCtaShared = arrival == expectedCtas - 1;
    }
    // Make the last-CTA decision visible to every thread in this CTA.
    __syncthreads();

    if (isLastCtaShared)
    {
        // Reset the counter for hygiene; this barrier is still only used once per kernel launch.
        syncScratch[0] = 0;
        // Ensure the reset and all prior split scratch writes are globally visible before releasing waiters.
        __threadfence();
        // Publish the release flag. Waiting CTAs observe this with atomicAdd(..., 0).
        atomicExch(syncScratch + 1, 1);
    }
    else
    {
        // Wait until the last CTA publishes the release flag.
        while (atomicAdd(syncScratch + 1, 0) == 0)
        {
            // Back off slightly while polling global memory.
            __nanosleep(64U);
        }
    }
    // Keep all threads in the CTA aligned before entering the combine phase.
    __syncthreads();
}

// Compute one per-token/head absorbed q_nope projection row and quantize it for generation attention.
//
// Inputs:
// - fusedQ: mutable BF16 [numTokens, kLocalHeads, kHeadDim].
// - qNope: BF16 [numTokens, kLocalHeads, kQkNopeHeadDim], possibly a split view.
// - kBProjTrans: BF16 [kLocalHeads, kKvLoraRank, kQkNopeHeadDim].
// - quantQ: mutable FP8 E4M3 [numTokens, kLocalHeads, kHeadDim].
// - quantScale: float, original-domain to FP8 scale.
// - qNopeStrideToken/qNopeStrideHead: runtime strides for qNope.
// - kBProjStrideHead/kBProjStrideDim/kBProjStrideReduction: runtime strides for kBProjTrans.
// - tokenIdx/headIdx: selected token and local attention head.
//
// Outputs: none.
//
// Side effects:
// - Writes fusedQ[tokenIdx, headIdx, 0:kKvLoraRank].
// - Writes quantQ[tokenIdx, headIdx, 0:kKvLoraRank].
__device__ __forceinline__ void projectGenerationQNopeHead(__nv_bfloat16* fusedQ, __nv_bfloat16 const* qNope,
    __nv_bfloat16 const* kBProjTrans, __nv_fp8_e4m3* quantQ, float quantScale, int64_t qNopeStrideToken,
    int64_t qNopeStrideHead, int64_t kBProjStrideHead, int64_t kBProjStrideDim, int64_t kBProjStrideReduction,
    int tokenIdx, int headIdx)
{
    // Each thread owns one or two latent output dimensions for the current [token, head].
    for (int dim = threadIdx.x; dim < kKvLoraRank; dim += blockDim.x)
    {
        // Flattened [token, head, dim] offset for both fusedQ and quantQ.
        int const qOffset = (tokenIdx * kLocalHeads + headIdx) * kHeadDim + dim;
        // Accumulate one BF16 GEMV output element in FP32:
        // fusedQ[token, head, dim] = sum_r qNope[token, head, r] * kBProjTrans[head, dim, r].
        float projected = 0.0F;
        // Base offset for qNope[token, head, 0]. qNope is a split view, so strides are runtime values.
        int64_t const qNopeBaseOffset = tokenIdx * qNopeStrideToken + headIdx * qNopeStrideHead;
        // Base offset for kBProjTrans[head, dim, 0].
        int64_t const kBProjBaseOffset = headIdx * kBProjStrideHead + dim * kBProjStrideDim;
        // Reduction over the unabsorbed q_nope dimension, fixed to 192 for GLM-5.
        for (int reduceDim = 0; reduceDim < kQkNopeHeadDim; ++reduceDim)
        {
            // Load q_nope[token, head, reduceDim] as FP32.
            float const qValue = toFloat(qNope[qNopeBaseOffset + reduceDim]);
            // Load k_b_proj_trans[head, dim, reduceDim] as FP32.
            float const kValue = toFloat(kBProjTrans[kBProjBaseOffset + reduceDim * kBProjStrideReduction]);
            // Add this BF16 x BF16 product to the FP32 accumulator.
            projected += qValue * kValue;
        }
        // Round the absorbed query value to BF16 to match the dtype of the previous BMM output tensor.
        __nv_bfloat16 const projectedBf16 = toBfloat16(projected);
        // Keep fusedQ readable for debug comparisons and for parity with the old Python-side BMM path.
        fusedQ[qOffset] = projectedBf16;
        // Convert original-domain BF16 projected Q to scaled FP8 for the attention dot product.
        quantQ[qOffset] = __nv_fp8_e4m3(toFloat(projectedBf16) * quantScale);
    }
}

// Rotate one per-token/head Q RoPE suffix and quantize it for generation attention.
//
// Inputs:
// - fusedQ: mutable BF16 [numTokens, kLocalHeads, kHeadDim].
// - qPe: BF16 [numTokens, kLocalHeads, kRopeDim], possibly a split view.
// - rotaryCosSin: FP32 float2 RoPE table indexed by absolute KV position.
// - quantQ: mutable FP8 E4M3 [numTokens, kLocalHeads, kHeadDim].
// - quantScale: float, original-domain to FP8 scale.
// - qPeStrideToken/qPeStrideHead: runtime strides for qPe.
// - tokenIdx/tokenIdxInKvCache/headIdx: selected token, absolute position, and local attention head.
//
// Outputs: none.
//
// Side effects:
// - Writes fusedQ[tokenIdx, headIdx, kKvLoraRank:kHeadDim].
// - Writes quantQ[tokenIdx, headIdx, kKvLoraRank:kHeadDim].
__device__ __forceinline__ void rotateGenerationQPeHead(__nv_bfloat16* fusedQ, __nv_bfloat16 const* qPe,
    float2 const* rotaryCosSin, __nv_fp8_e4m3* quantQ, float quantScale, int64_t qPeStrideToken, int64_t qPeStrideHead,
    int tokenIdx, int tokenIdxInKvCache, int headIdx)
{
    // Each thread handles a strided subset of the 32 adjacent RoPE pairs.
    for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
    {
        // Adjacent RoPE scalar pair within the 64-dim suffix.
        int const dim = 2 * pairIdx;
        // qPe may be a non-contiguous view, so use runtime token/head strides.
        int64_t const qPeOffset = tokenIdx * qPeStrideToken + headIdx * qPeStrideHead + dim;
        // Load one BF16 pair as FP32 for RoPE math.
        float2 value;
        value.x = toFloat(qPe[qPeOffset]);
        value.y = toFloat(qPe[qPeOffset + 1]);
        // Use absolute KV position for the current decode/MTP token.
        float2 const coef = rotaryCosSin[tokenIdxInKvCache * kRopeDim + pairIdx];
        // Apply GPT-J style adjacent-pair RoPE.
        float2 const rotated = rotaryTransform(value, coef);
        // Flattened output offset for the first RoPE suffix scalar.
        int const qOffset = (tokenIdx * kLocalHeads + headIdx) * kHeadDim + kKvLoraRank + dim;
        // Match the generic kernel by rounding rotated RoPE values to BF16 before FP8 quantization.
        __nv_bfloat16 const first = toBfloat16(rotated.x);
        // Round the second adjacent component to BF16.
        __nv_bfloat16 const second = toBfloat16(rotated.y);
        // Keep fusedQ readable for debug comparisons even though decode attention consumes quantQ.
        fusedQ[qOffset] = first;
        // Store the second rotated component in fusedQ.
        fusedQ[qOffset + 1] = second;
        // Quantize first rotated component into full FP8 Q.
        quantQ[qOffset] = __nv_fp8_e4m3(toFloat(first) * quantScale);
        // Quantize second rotated component into full FP8 Q.
        quantQ[qOffset + 1] = __nv_fp8_e4m3(toFloat(second) * quantScale);
    }
}

// Append the current decode/MTP token's shared latent KV row to the paged KV cache.
//
// Inputs:
// - latentCache: BF16 [numTokens, kHeadDim]. Dims [0, 512) are latent KV/V; dims [512, 576) are K RoPE.
// - rotaryCosSin: FP32 float2 RoPE table indexed by absolute KV position.
// - kvCache: paged TRT-LLM KV cache view with one MLA KV head.
// - quantScale: float, original-domain to FP8 scale.
// - tokenIdx/tokenIdxInKvCache: selected decode/MTP token and absolute cache position.
//
// Outputs: none.
//
// Side effects:
// - Writes one FP8 row into kvCache for request 0 at tokenIdxInKvCache.
__device__ __forceinline__ void appendGenerationKvCache(__nv_bfloat16 const* latentCache, float2 const* rotaryCosSin,
    tensorrt_llm::kernels::KVBlockArray kvCache, float quantScale, int tokenIdx, int tokenIdxInKvCache)
{
    // Resolve the physical page that contains this token's single MLA KV head.
    auto* blockPtr = kvCache.getKBlockPtr(/*seqIdx=*/0, tokenIdxInKvCache);
    // Copy and quantize latent KV/V prefix [0, 512).
    for (int dim = threadIdx.x; dim < kKvLoraRank; dim += blockDim.x)
    {
        // Scalar offset inside the resolved physical KV page.
        int const cacheOffset = kvCache.getKVLocalIdx(tokenIdxInKvCache, /*headIdx=*/0, kHeadDim, dim);
        // Load BF16 latent KV/V and store scaled FP8.
        reinterpret_cast<__nv_fp8_e4m3*>(blockPtr)[cacheOffset]
            = __nv_fp8_e4m3(toFloat(latentCache[tokenIdx * kHeadDim + dim]) * quantScale);
    }

    // Rotate and quantize the shared K RoPE suffix [512, 576).
    for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
    {
        // Adjacent scalar pair within the K RoPE suffix.
        int const dim = 2 * pairIdx;
        // Flattened latentCache offset for the first suffix scalar.
        int const latentOffset = tokenIdx * kHeadDim + kKvLoraRank + dim;
        // Load one K RoPE pair as FP32 for rotation.
        float2 value;
        value.x = toFloat(latentCache[latentOffset]);
        value.y = toFloat(latentCache[latentOffset + 1]);
        // K uses the same absolute position as Q for the appended token.
        float2 const coef = rotaryCosSin[tokenIdxInKvCache * kRopeDim + pairIdx];
        // Apply RoPE to the K suffix.
        float2 const rotated = rotaryTransform(value, coef);
        // Scalar offset inside the paged KV row.
        int const cacheOffset = kvCache.getKVLocalIdx(tokenIdxInKvCache, /*headIdx=*/0, kHeadDim, kKvLoraRank + dim);
        // Round to BF16 before FP8 quantization to match the generic MLA RoPE kernel.
        __nv_bfloat16 const first = toBfloat16(rotated.x);
        // Round the second component.
        __nv_bfloat16 const second = toBfloat16(rotated.y);
        // Store first rotated K suffix component as FP8.
        reinterpret_cast<__nv_fp8_e4m3*>(blockPtr)[cacheOffset] = __nv_fp8_e4m3(toFloat(first) * quantScale);
        // Store second rotated K suffix component as FP8.
        reinterpret_cast<__nv_fp8_e4m3*>(blockPtr)[cacheOffset + 1] = __nv_fp8_e4m3(toFloat(second) * quantScale);
    }
}

// Preprocess one context token row before sparse MLA attention.
//
// Grid:
// - blockIdx.x selects the context token row in [0, numTokens).
// - blockIdx.y selects the preprocessing role:
//   - [0, kLocalHeads) rotates one per-head Q RoPE suffix from qPe into fusedQ.
//   - kLocalHeads reads the shared latent K RoPE suffix from latentCache, rotates it, then writes the full
//     latent KV row into the paged KV cache.
// - threadIdx.x is used as a strided lane over RoPE pairs or full 576-dim KV elements.
//
// Inputs and mutable tensors:
// - fusedQ: mutable BF16 [numTokens, kLocalHeads, kHeadDim]. Dims [0, kKvLoraRank) already contain the
//   absorbed q_nope projection. The Q branch fills dims [kKvLoraRank, kHeadDim) with rotated Q RoPE.
// - latentCache: BF16 [numTokens, kHeadDim]. Dims [0, kKvLoraRank) are compressed latent KV/V values.
//   Dims [kKvLoraRank, kHeadDim) are the unrotated shared K RoPE suffix. This tensor is read only.
// - qPe: BF16 [numTokens, kLocalHeads, kRopeDim]. Unrotated per-head Q RoPE values.
// - rotaryCosSin: FP32 float2 RoPE table. It is indexed by the absolute KV position, so cached prefixes use
//   the same rotary phase as TRT-LLM's paged attention path.
// - ctxCachedTokenIndptr: INT64 [numContexts + 1]. The current custom path is restricted to one context request;
//   ctxCachedTokenIndptr[1] - ctxCachedTokenIndptr[0] is the already-cached prefix length for sequence 0.
// - kvCache: paged TRT-LLM KV cache view with one MLA KV head and kHeadDim elements per token.
// - kvScaleOrigQuant: optional FP32 [1]. BF16 latentCache values are multiplied by this original-domain to
//   FP8 scale before storing to the paged KV cache when kFp8KvCache is true.
// - numTokens: number of context tokens in fusedQ/qPe/latentCache.
//
// Side effects:
// - Q branch mutates fusedQ[token, head, kKvLoraRank:kHeadDim].
// - KV branch writes latentCache[token, :] to kvCache with a rotated K RoPE suffix.
// - The function does not produce a returned tensor; contextAttentionKernel consumes fusedQ and kvCachePool later.
template <bool kFp8KvCache>
__global__ void preprocessContextKernel(__nv_bfloat16* fusedQ, __nv_bfloat16* latentCache, __nv_bfloat16 const* qPe,
    float2 const* rotaryCosSin, int64_t const* ctxCachedTokenIndptr, tensorrt_llm::kernels::KVBlockArray kvCache,
    float const* kvScaleOrigQuant, int64_t qPeStrideToken, int64_t qPeStrideHead, int numTokens)
{
    // Grid shape is [numTokens, kLocalHeads + 1].
    // - blockIdx.x selects one context token row.
    // - blockIdx.y in [0, 7] rotates one Q head from qPe into fusedQ.
    // - blockIdx.y == 8 rotates the single shared latent K RoPE suffix and writes the 576-dim KV row to cache.
    // Convert the x-grid coordinate to the local context-token index this CTA owns.
    int const tokenIdx = static_cast<int>(blockIdx.x);
    // Convert the y-grid coordinate to either a Q head index [0, 7] or the KV branch sentinel value 8.
    int const headIdx = static_cast<int>(blockIdx.y);
    // Current custom context dispatch is restricted to one context request. ctxCachedTokenIndptr has shape
    // [num_contexts + 1] == [2], so this is the number of already-cached prefix tokens for sequence 0.
    // The subtraction gives the cached prefix length for the only supported request.
    int64_t const cachedLen = ctxCachedTokenIndptr[1] - ctxCachedTokenIndptr[0];
    // tokenIdxInKvCache is the absolute KV position for this context token after any cached prefix.
    // The RoPE table and paged KV cache both use this absolute sequence position.
    int const tokenIdxInKvCache = static_cast<int>(cachedLen) + tokenIdx;
    // kvScaleOrigQuant maps BF16 original values to FP8 E4M3 cache values when the paged KV cache is FP8.
    // Null means the caller is using identity quantization scale.
    float const kvScale = kvScaleOrigQuant == nullptr ? 1.0F : kvScaleOrigQuant[0];

    // A defensive guard for padded launches. Current launches use exactly numTokens x (heads + 1).
    if (tokenIdx >= numTokens)
    {
        // Out-of-range CTAs own no tensor elements.
        return;
    }

    // Q branch: headIdx [0, 7] rotates qPe[tokenIdx, headIdx, :] and writes it into fusedQ.
    if (headIdx < kLocalHeads)
    {
        // Q branch: each block rotates one token/head RoPE suffix. The absorbed q_nope prefix in fusedQ is produced
        // by the Python-side BMM and is left unchanged here.
        // Each thread handles a strided subset of the 32 adjacent RoPE pairs.
        for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
        {
            // Temporary FP32 pair holding one adjacent BF16 RoPE vector component pair.
            float2 value;
            // dim is the first element of an adjacent RoPE pair in qPe[token, head, dim:dim + 2].
            // RoPE pairs adjacent scalar positions as [dim, dim + 1].
            int const dim = 2 * pairIdx;
            // qPe is a split view from [numTokens, 8, 256] in the model path, so only the last dimension
            // is guaranteed contiguous.
            // Use runtime strides instead of assuming a compact [numTokens, 8, 64] allocation.
            int64_t const qPeOffset = tokenIdx * qPeStrideToken + headIdx * qPeStrideHead + dim;
            // Load the first BF16 RoPE component and widen it to FP32 for rotation math.
            value.x = toFloat(qPe[qPeOffset]);
            // Load the second BF16 RoPE component and widen it to FP32 for rotation math.
            value.y = toFloat(qPe[qPeOffset + 1]);
            // rotaryCosSin is indexed by absolute KV position so cached prefixes get the correct phase.
            // Layout is [position, pairIdx] as float2 [cos, sin].
            float2 const coef = rotaryCosSin[tokenIdxInKvCache * kRopeDim + pairIdx];
            // Apply the complex rotation for this absolute position.
            float2 const rotated = rotaryTransform(value, coef);

            // fusedQ is BF16 [numTokens, 8, 576]; write the RoPE suffix at dims [512, 576).
            // The leading 512 dims already hold q_nope_absorbed from the BMM before this kernel.
            int const fusedOffset = (tokenIdx * kLocalHeads + headIdx) * kHeadDim + kKvLoraRank + dim;
            // Store the rotated first component, rounded back to BF16 like the reference path.
            fusedQ[fusedOffset] = toBfloat16(rotated.x);
            // Store the rotated second component, rounded back to BF16 like the reference path.
            fusedQ[fusedOffset + 1] = toBfloat16(rotated.y);
        }
        // Q-branch CTAs do not touch latentCache or the KV cache.
        return;
    }

    // KV branch: one block per token writes the complete [latent KV/V, K RoPE] row into TRT-LLM's paged KV cache.
    // Match the trusted MLA context path by leaving latentCache itself unchanged; only the cache receives rotated K.
    // Resolve the base pointer for the physical KV-cache page containing tokenIdxInKvCache for request 0.
    auto* blockPtr = kvCache.getKBlockPtr(/*seqIdx=*/0, tokenIdxInKvCache);
    // Copy and quantize the latent [0, 512) value prefix. These dims are both latent K content and V content.
    for (int dim = threadIdx.x; dim < kKvLoraRank; dim += blockDim.x)
    {
        // Convert the absolute sequence position and latent dim to the offset inside the resolved KV page.
        int const cacheOffset = kvCache.getKVLocalIdx(tokenIdxInKvCache, /*headIdx=*/0, kHeadDim, dim);
        // Load latentCache[tokenIdx, dim] as BF16; this part does not use RoPE.
        __nv_bfloat16 const value = latentCache[tokenIdx * kHeadDim + dim];
        // Write the value to the real paged KV cache so this context attention and later decode steps can read it.
        if constexpr (kFp8KvCache)
        {
            // Quantize BF16 original-domain value to FP8 E4M3 using the same cache quantization scale as TRTLLM MLA.
            __nv_fp8_e4m3 const quantized = __nv_fp8_e4m3(toFloat(value) * kvScale);
            // FP8 KV-cache mode stores the quantized byte directly.
            reinterpret_cast<__nv_fp8_e4m3*>(blockPtr)[cacheOffset] = quantized;
        }
        else
        {
            // Non-FP8 mode stores the original BF16 value directly.
            reinterpret_cast<__nv_bfloat16*>(blockPtr)[cacheOffset] = value;
        }
    }

    // Rotate and store the shared K RoPE suffix [512, 576). There is one shared K suffix per token.
    for (int pairIdx = threadIdx.x; pairIdx < kRopeDim / 2; pairIdx += blockDim.x)
    {
        // Temporary FP32 pair holding one adjacent K RoPE component pair.
        float2 value;
        // latentCache is BF16 [numTokens, 576]; the final 64 elements are the shared K RoPE suffix.
        // dim is the first scalar of the adjacent RoPE pair within the 64-dim suffix.
        int const dim = 2 * pairIdx;
        // Convert token/suffix pair coordinates into the flattened latentCache offset.
        int const latentOffset = tokenIdx * kHeadDim + kKvLoraRank + dim;
        // Load the first unrotated K RoPE component as FP32.
        value.x = toFloat(latentCache[latentOffset]);
        // Load the second unrotated K RoPE component as FP32.
        value.y = toFloat(latentCache[latentOffset + 1]);
        // Load [cos, sin] for the absolute cache position and this RoPE pair.
        float2 const coef = rotaryCosSin[tokenIdxInKvCache * kRopeDim + pairIdx];
        // Apply RoPE to the shared K suffix.
        float2 const rotated = rotaryTransform(value, coef);

        // Compute the paged-cache offset of the first scalar in this rotated suffix pair.
        int const cacheOffset = kvCache.getKVLocalIdx(tokenIdxInKvCache, /*headIdx=*/0, kHeadDim, kKvLoraRank + dim);
        // Round the first rotated scalar to BF16 before optional FP8 quantization.
        __nv_bfloat16 const first = toBfloat16(rotated.x);
        // Round the second rotated scalar to BF16 before optional FP8 quantization.
        __nv_bfloat16 const second = toBfloat16(rotated.y);
        // Mirror the rotated K suffix into the paged KV cache for later decode.
        if constexpr (kFp8KvCache)
        {
            // Quantize the first rounded scalar for the FP8 paged cache.
            __nv_fp8_e4m3 const quantizedFirst = __nv_fp8_e4m3(toFloat(first) * kvScale);
            // Quantize the second rounded scalar for the FP8 paged cache.
            __nv_fp8_e4m3 const quantizedSecond = __nv_fp8_e4m3(toFloat(second) * kvScale);
            // FP8 KV-cache mode stores quantized rotated suffix values.
            reinterpret_cast<__nv_fp8_e4m3*>(blockPtr)[cacheOffset] = quantizedFirst;
            // Store the adjacent rotated suffix scalar.
            reinterpret_cast<__nv_fp8_e4m3*>(blockPtr)[cacheOffset + 1] = quantizedSecond;
        }
        else
        {
            // Non-FP8 mode stores BF16 rotated suffix values.
            reinterpret_cast<__nv_bfloat16*>(blockPtr)[cacheOffset] = first;
            // Store the adjacent rotated suffix scalar.
            reinterpret_cast<__nv_bfloat16*>(blockPtr)[cacheOffset + 1] = second;
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
// - kvCachePool: FP8 E4M3 [poolTokens, 1, kHeadDim], flattened as [poolTokens, kHeadDim]. This is the full
//   primary KV-cache pool view. It contains cached-prefix rows from earlier prefill chunks and current rows
//   written by preprocessContextKernel just before this kernel launches.
// - topkIndicesPool: INT32 [numTokens, topK]. Global row indices into kvCachePool produced from local top-k.
//   Negative entries are padding or conversion failures.
// - topkIndicesLocal: INT32 [numTokens, topK]. Local sequence positions used only for context causal validity.
//   Negative entries are padding.
// - ctxCachedTokenIndptr: INT64 [numContexts + 1]. For the supported one-context path,
//   ctxCachedTokenIndptr[1] - ctxCachedTokenIndptr[0] is the number of cached-prefix rows before token 0 in
//   this current prefill chunk.
// - kvScaleOrigQuant: optional FP32 scale mapping original BF16 values to FP8.
//   Context uses the same scale for Q and KV quantization in the trusted path.
// - kvScaleQuantOrig: optional FP32 scale mapping FP8 cache values back to
//   original units. Used for Q/K score dequant and V output dequant.
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
    int32_t const* topkIndicesPool, int32_t const* topkIndicesLocal, int64_t const* ctxCachedTokenIndptr,
    __nv_bfloat16* output, float const* kvScaleOrigQuant, float const* kvScaleQuantOrig, int numTokens, int topK,
    float hostScoreScale)
{
    // One CUDA block computes one (query token, local head) row.
    // Dynamic shared layout:
    // - reduce[0:kThreads] is used by blockReduceSum() for a 576-wide dot product.
    // - weights[0:topK] stores one score/softmax weight per sparse KV candidate.
    // Allocate dynamic shared memory provided by the launch.
    extern __shared__ float shared[];
    // The first kThreads floats are reused as reduction scratch for one QK dot product.
    float* reduce = shared;
    // The remaining topK floats store per-candidate logits first, then exp2-normalized weights.
    float* weights = shared + blockDim.x;

    // Select the context query token row owned by this CTA.
    int const tokenIdx = static_cast<int>(blockIdx.x);
    // Select the local MLA attention head owned by this CTA.
    int const headIdx = static_cast<int>(blockIdx.y);
    // Thread-local lane id used for strided loops over head_dim and v_dim.
    int const tid = threadIdx.x;
    // Current dispatch supports one context request. The first current-chunk query token has local sequence
    // position cachedLen, so query token tokenIdx may attend through local position cachedLen + tokenIdx.
    int const cachedLen = static_cast<int>(ctxCachedTokenIndptr[1] - ctxCachedTokenIndptr[0]);
    // Match the trusted context FP8-Q path: Q and KV are quantized with kvScaleOrigQuant, then FMHA dequantizes
    // both operands with kvScaleQuantOrig during BMM1.
    // This scale maps original BF16 Q values to FP8 for the explicit Q rounding below.
    float const kvQuantScale = kvScaleOrigQuant == nullptr ? 1.0F : kvScaleOrigQuant[0];
    // This scale maps FP8 Q/K/V values back to original units in score and output math.
    float const kvDequantScale = kvScaleQuantOrig == nullptr ? 1.0F : kvScaleQuantOrig[0];
    // Softmax is evaluated with exp2f, so natural-log logits are multiplied by log2(e).
    constexpr float kLog2e = 1.4426950408889634074F;
    // QK score = dot(q_fp8, k_fp8) * dq * dk * hostScoreScale.
    // The extra log2(e) converts the exponent base from e to 2 for exp2f.
    float const scoreScaleLog2 = kvDequantScale * kvDequantScale * hostScoreScale * kLog2e;

    // Defensive bounds guard for padded launches. Current launch grid is exact.
    if (tokenIdx >= numTokens || headIdx >= kLocalHeads)
    {
        // This CTA owns no valid output row.
        return;
    }

    // Track max log2 score for numerically stable softmax: exp2(score - maxScore).
    float maxScore = -INFINITY;

    // First pass over sparse candidates computes QK logits and the row max.
    for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
    {
        // topkIndicesPool is INT32 [numTokens, topK]. It resolves the selected local sequence position into
        // the physical pool row that stores the FP8 KV data.
        int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
        // topkIndicesLocal is INT32 [numTokens, topK]. It keeps the original local sequence position so the
        // current chunk can mask future rows while allowing cached-prefix rows.
        int const localKvIdx = topkIndicesLocal[tokenIdx * topK + topkIdx];
        // Query token tokenIdx is located at local position cachedLen + tokenIdx.
        int const maxVisibleLocalIdx = cachedLen + tokenIdx;
        // A local candidate is visible when it is a real position at or before the query's absolute position.
        bool const visibleKv = localKvIdx >= 0 && localKvIdx <= maxVisibleLocalIdx;
        // A candidate is valid only when top-k conversion produced a real pool row and the causal check passes.
        bool const validKv = kvIdx >= 0 && visibleKv;
        // Each thread accumulates a subset of the 576-dimensional dot product.
        float partial = 0.0F;
        // Skip memory reads for invalid/padded sparse candidates.
        if (validKv)
        {
            // Use 64-bit offset math because real bench pool rows can exceed int32_t byte-address ranges.
            int64_t const kvOffset = static_cast<int64_t>(kvIdx) * kHeadDim;
            // kvPtr points at the full 576-dim FP8 KV row in the primary KV-cache pool.
            __nv_fp8_e4m3 const* kvPtr = kvCachePool + kvOffset;
            // Stride threads across the full QK dimension [latent 512 + RoPE 64].
            for (int dim = tid; dim < kHeadDim; dim += blockDim.x)
            {
                // Match the FP8-KV attention path by rounding BF16 fusedQ to E4M3 before the QK dot.
                // fusedQ is BF16 [numTokens, 8, 576]; multiply by orig->quant scale, then round to FP8.
                float const qFp8 = static_cast<float>(
                    __nv_fp8_e4m3(toFloat(fusedQ[(tokenIdx * kLocalHeads + headIdx) * kHeadDim + dim]) * kvQuantScale));
                // Accumulate the unscaled FP8-dot-product contribution in FP32.
                partial += qFp8 * toFloat(kvPtr[dim]);
            }
        }
        // Reduce the 576-dim partial dot product across all threads in this CTA.
        float const dot = blockReduceSum(partial, reduce);
        // One thread writes the candidate logit and updates the max for the softmax.
        if (tid == 0)
        {
            // Invalid sparse entries get -inf so they contribute zero probability.
            float const score = validKv ? dot * scoreScaleLog2 : -INFINITY;
            // Store log2-space score in shared memory. It becomes a softmax weight later.
            weights[topkIdx] = score;
            // Track the largest valid score for numerical stability.
            maxScore = fmaxf(maxScore, score);
        }
        // Ensure weights[topkIdx] is written before any later iteration can reuse reduction scratch.
        __syncthreads();
    }

    // Shared denominator is written by thread 0 and read by all threads during the V pass.
    __shared__ float denomShared;
    // Thread 0 converts logits to unnormalized exp2 weights and sums the denominator.
    if (tid == 0)
    {
        // Accumulate sum_j exp2(score_j - maxScore).
        float denom = 0.0F;
        // If every candidate was invalid, maxScore remains -inf and exp2(-inf - -inf) would be NaN.
        if (maxScore != -INFINITY)
        {
            // Convert each stored log2 score into an unnormalized probability.
            for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
            {
                // Subtracting maxScore implements stable softmax.
                float const weight = exp2f(weights[topkIdx] - maxScore);
                // Reuse the same shared slot for the unnormalized softmax weight.
                weights[topkIdx] = weight;
                // Add to the softmax denominator.
                denom += weight;
            }
        }
        else
        {
            // No valid candidates. Zero all weights so the output row becomes zero.
            for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
            {
                // Explicitly clear the slot that previously held -inf.
                weights[topkIdx] = 0.0F;
            }
        }
        // Publish the denominator to the rest of the CTA.
        denomShared = denom;
    }
    // Wait until all weights and denomShared have been finalized.
    __syncthreads();

    // Compute reciprocal denominator once per thread. A zero denominator produces zero output.
    float const invDenom = denomShared > 0.0F ? 1.0F / denomShared : 0.0F;
    // Second pass computes the weighted sum over the latent V dimensions [0, 512).
    for (int dim = tid; dim < kKvLoraRank; dim += blockDim.x)
    {
        // Output dim only spans the latent V rank [0, 512), not the 64 RoPE suffix.
        // Each thread accumulates one or more output V dimensions independently.
        float acc = 0.0F;
        // Sum over the same sparse candidates used for QK.
        for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
        {
            // Reload the global pool row for the selected KV candidate.
            int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
            // Reload the local sequence position for the same causal check used in QK.
            int const localKvIdx = topkIndicesLocal[tokenIdx * topK + topkIdx];
            // Query token tokenIdx is located at local position cachedLen + tokenIdx.
            int const maxVisibleLocalIdx = cachedLen + tokenIdx;
            // A local candidate is visible when it is a real position at or before the query's absolute position.
            bool const visibleKv = localKvIdx >= 0 && localKvIdx <= maxVisibleLocalIdx;
            // A candidate is valid only when top-k conversion produced a real pool row and the causal check passes.
            bool const validKv = kvIdx >= 0 && visibleKv;
            // Accumulate only real sparse candidates.
            if (validKv)
            {
                // Use 64-bit offset math because real bench pool rows can exceed int32_t byte-address ranges.
                int64_t const kvOffset = static_cast<int64_t>(kvIdx) * kHeadDim;
                // The trusted context path rounds probabilities to BF16 before BMM2.
                float const prob = toFloat(toBfloat16(weights[topkIdx] * invDenom));
                // The trusted context path also rounds FP8 V values through BF16 before multiply-add.
                float const value = toFloat(toBfloat16(toFloat(kvCachePool[kvOffset + dim])));
                // Accumulate prob * V in FP32.
                acc += prob * value;
            }
        }
        // Match the BF16 accumulation/output behavior by rounding the accumulator before dequantization.
        float const accBf16 = toFloat(toBfloat16(acc));
        // Dequantize V with kvScaleQuantOrig and write BF16 output [token, head, dim].
        output[(tokenIdx * kLocalHeads + headIdx) * kKvLoraRank + dim] = toBfloat16(accBf16 * kvDequantScale);
    }
}

// Combined TileRT-style split and combine kernel for GLM-5 generation sparse MLA attention.
//
// This kernel changes the decode attention work decomposition from one CTA per [token, head] to one CTA per
// [token, sparse top-k split]. For the GLM-5 topK=2048 path, kSparseSplitSize=64 gives 32 CTAs per token, matching
// TileRT's documented CTAID layout. The launch starts by using the split CTAs to apply RoPE, quantize Q, and append
// the current KV rows. During attention math, consumer warps 0..7 map one warp to one local MLA head and
// producer/control warps 8..11 stay alive as the reserved TileRT-style group. After all split CTAs finish, the same
// launch grid-syncs and the first eight split CTAs for each token combine one local-head output each.
//
// Inputs:
// - fusedQ: mutable BF16 [numTokens, kLocalHeads, kHeadDim]. The kernel writes dims [0, kKvLoraRank) with
//   q_nope absorbed by k_b_proj and dims [kKvLoraRank, kHeadDim) with rotated Q RoPE.
// - qNope: BF16 [numTokens, kLocalHeads, kQkNopeHeadDim]. Unabsorbed query prefix from q_a/q_b projection.
// - kBProjTrans: BF16 [kLocalHeads, kKvLoraRank, kQkNopeHeadDim]. Per-head absorbed-K projection weight.
// - qPe: BF16 [numTokens, kLocalHeads, kRopeDim]. Unrotated per-head Q RoPE suffix.
// - latentCache: BF16 [numTokens, kHeadDim]. Dims [0, kKvLoraRank) are latent KV/V; dims [kKvLoraRank, kHeadDim)
//   are unrotated shared K RoPE.
// - rotaryCosSin: FP32 float2 RoPE table indexed as [position, rope_pair].
// - kvCache: paged TRT-LLM KV cache view. The kernel writes the current decode/MTP group into sequence 0 before
//   attention reads those rows through kvCachePool.
// - quantQ: FP8 E4M3 [numTokens, kLocalHeads, kHeadDim]. The kernel writes the full rotated/absorbed Q before
//   attention consumes it.
// - kvCachePool: FP8 E4M3 [poolTokens, 1, kHeadDim], flattened as [poolTokens, kHeadDim].
// - topkIndicesPool: INT32 [numTokens, topK]. Global row indices into kvCachePool produced from local top-k.
// - topkIndicesLocal: INT32 [numTokens, topK]. Local sequence positions for MTP visibility checks.
// - sequenceLength: INT32 [1]. Total KV length after the current decode/MTP group has been appended.
// - specDecodingPackedMask: optional INT32 [numTokens, specMaskWords] for request 0.
// - specMaskWords: int, number of packed int32 mask words per query token.
// - bmm1Scale: optional FP32 [2]. The append phase writes natural-log and log2-space QK score scales.
// - bmm2Scale: optional FP32 [1]. The append phase writes the dequantization scale for FP8 V values.
// - kvScaleOrigQuant: optional FP32 [1]. Original-domain to FP8 scale for Q and appended KV.
// - kvScaleQuantOrig: optional FP32 [1]. FP8 to original-domain dequant scale.
// - qNopeStrideToken/qNopeStrideHead: runtime strides for qNope because it is a split view of Q.
// - kBProjStrideHead/kBProjStrideDim/kBProjStrideReduction: runtime strides for kBProjTrans.
// - qPeStrideToken/qPeStrideHead: runtime strides for qPe because it can be a non-contiguous split view.
// - numTokens: int, number of decode/MTP tokens in this call.
// - topK: int, sparse candidate count per query token.
// - numSplits: int, ceil(topK / kSparseSplitSize).
// - expectedCtas: int, exact split CTA grid size used by each one-shot global barrier.
// - hostScoreScale: float, unquantized model attention scale.
//
// Outputs:
// - splitLse: FP32 [numTokens, kLocalHeads, numSplits]. Each element is log2(sum(exp2(score))) for one split.
// - splitOutput: FP32 [numTokens, kLocalHeads, numSplits, kKvLoraRank]. Each element is the split-local
//   softmax-normalized latent V output in raw FP8-cache units.
// - output: BF16 [numTokens, kLocalHeads * kKvLoraRank], logically [numTokens, 8, 512].
//
// Side effects:
// - Writes fusedQ's absorbed prefix and RoPE suffix, quantQ, bmm1Scale, bmm2Scale, the paged KV cache, splitLse,
//   splitOutput, syncScratch, and output.
// - Uses dynamic shared memory [kLocalHeads, kSparseSplitSize] floats for per-head split scores/probabilities.
__global__ void generationAttentionCombinedKernel(__nv_bfloat16* fusedQ, __nv_bfloat16 const* qNope,
    __nv_bfloat16 const* kBProjTrans, __nv_bfloat16 const* qPe, __nv_bfloat16 const* latentCache,
    float2 const* rotaryCosSin, tensorrt_llm::kernels::KVBlockArray kvCache, __nv_fp8_e4m3* quantQ,
    __nv_fp8_e4m3 const* kvCachePool, int32_t const* topkIndicesPool, int32_t const* topkIndicesLocal,
    int32_t const* sequenceLength, int32_t const* specDecodingPackedMask, int specMaskWords, float* splitLse,
    float* splitOutput, int32_t* syncScratch, __nv_bfloat16* output, float* bmm1Scale, float* bmm2Scale,
    float const* kvScaleOrigQuant, float const* kvScaleQuantOrig, int64_t qNopeStrideToken, int64_t qNopeStrideHead,
    int64_t kBProjStrideHead, int64_t kBProjStrideDim, int64_t kBProjStrideReduction, int64_t qPeStrideToken,
    int64_t qPeStrideHead, int numTokens, int topK, int numSplits, int expectedCtas, float hostScoreScale)
{
    // Dynamic shared memory is partitioned as [head, split_offset]. Lane 0 in each consumer warp first writes
    // log2 scores, then rewrites the same slots as split-local normalized probabilities.
    extern __shared__ float splitScores[];

    // blockIdx.x is a linearized [token, split] coordinate: token is the high component, split is the low one.
    int const tokenIdx = static_cast<int>(blockIdx.x) / numSplits;
    // Split id in [0, numSplits). For topK=2048 and split size 64 this is [0, 32).
    int const splitIdx = static_cast<int>(blockIdx.x) - tokenIdx * numSplits;
    // Thread id inside the 384-thread CTA.
    int const tid = threadIdx.x;
    // Warp id inside the CTA. Warps 0..7 are consumer warps, matching the 256-thread consumer group in TileRT.
    int const warpIdx = tid / 32;
    // Lane id inside the current warp.
    int const laneIdx = tid & 31;

    // The group start is shared by all warps in this CTA. It is derived once by thread 0.
    __shared__ int currentGroupStartShared;
    if (tid == 0)
    {
        // Normally: post-append length minus number of just-appended decode tokens.
        int currentGroupStart = sequenceLength[0] - numTokens;
        // CUDA graph warmup can capture a dummy sequenceLength equal to numTokens. If top-k was replay-updated to
        // real positions, infer the group start from the maximum local selected index.
        if (sequenceLength[0] <= numTokens)
        {
            // Initialize to no valid top-k entries.
            int maxLocalIdx = -1;
            // Scan all token rows; the group start is shared by the single supported generation request.
            for (int idx = 0; idx < numTokens * topK; ++idx)
            {
                // Negative padding entries do not increase maxLocalIdx.
                maxLocalIdx = max(maxLocalIdx, topkIndicesLocal[idx]);
            }
            // Recover the first local position of the current decode/MTP group.
            if (maxLocalIdx >= numTokens)
            {
                currentGroupStart = maxLocalIdx - (numTokens - 1);
            }
        }
        // Publish the group start to every warp.
        currentGroupStartShared = currentGroupStart;
    }
    // All 12 warps must observe currentGroupStartShared before consumer warps start validity checks.
    __syncthreads();

    // Defensive guard for padded launches. Current dispatch uses an exact grid.
    if (tokenIdx >= numTokens || splitIdx >= numSplits)
    {
        return;
    }

    // Publish FP8 dequant scales once before any CTA can enter attention.
    if (tokenIdx == 0 && splitIdx == 0 && tid == 0)
    {
        // FP8 to original-domain dequant scale used by BMM1 and BMM2.
        float const dequantScale = kvScaleQuantOrig == nullptr ? 1.0F : kvScaleQuantOrig[0];
        // TRTLLM-Gen softmax uses exp2f, so keep both natural-log and log2-space score scales.
        constexpr float kLog2e = 1.4426950408889634074F;
        // BMM1 scale is dequant_q * dequant_k * attention_scale.
        float const bmm1ScaleValue = dequantScale * dequantScale * hostScoreScale;
        // Natural-log score scale.
        bmm1Scale[0] = bmm1ScaleValue;
        // Log2-space score scale consumed by the generation split-attention phase below.
        bmm1Scale[1] = bmm1ScaleValue * kLog2e;
        // BMM2 dequantizes FP8 V. The generic op multiplies by out_scale when provided; WIP passes no out_scale.
        bmm2Scale[0] = dequantScale;
    }

    // The current group's first absolute KV position. This uses the same fallback as the visibility logic so CUDA
    // graph warmup with a dummy sequenceLength still rotates and appends at the top-k-derived positions.
    int const currentGroupStart = currentGroupStartShared;
    // Original-domain to FP8 scale for Q and KV.
    float const quantScale = kvScaleOrigQuant == nullptr ? 1.0F : kvScaleOrigQuant[0];
    // Absolute KV-cache position for this decode/MTP token.
    int const tokenIdxInKvCache = currentGroupStart + tokenIdx;

    // For the target topK=2048 path there are 32 split CTAs per token. Use the first eight CTAs for q_nope
    // projection, the next eight for Q RoPE, and one more for the shared KV append so independent setup work runs
    // in parallel. Unit tests can run with fewer split CTAs, so keep a strided fallback that preserves coverage.
    bool const useParallelSetup = numSplits >= 2 * kLocalHeads + 1;
    if (useParallelSetup)
    {
        if (splitIdx < kLocalHeads)
        {
            // splitIdx 0..7: one CTA computes the absorbed q_nope projection for one head.
            projectGenerationQNopeHead(fusedQ, qNope, kBProjTrans, quantQ, quantScale, qNopeStrideToken,
                qNopeStrideHead, kBProjStrideHead, kBProjStrideDim, kBProjStrideReduction, tokenIdx, splitIdx);
        }
        else if (splitIdx < 2 * kLocalHeads)
        {
            // splitIdx 8..15: one CTA rotates and quantizes one Q RoPE head, independent of q_nope projection.
            rotateGenerationQPeHead(fusedQ, qPe, rotaryCosSin, quantQ, quantScale, qPeStrideToken, qPeStrideHead,
                tokenIdx, tokenIdxInKvCache, splitIdx - kLocalHeads);
        }
        else if (splitIdx == 2 * kLocalHeads)
        {
            // splitIdx 16: one CTA appends the shared latent KV/V and rotated K RoPE row to the paged cache.
            appendGenerationKvCache(latentCache, rotaryCosSin, kvCache, quantScale, tokenIdx, tokenIdxInKvCache);
        }
    }
    else
    {
        // Small-topK fallback: the available split CTAs stride over heads and splitIdx 0 also owns KV append.
        for (int setupHeadIdx = splitIdx; setupHeadIdx < kLocalHeads; setupHeadIdx += numSplits)
        {
            // Compute the absorbed prefix for this [token, head].
            projectGenerationQNopeHead(fusedQ, qNope, kBProjTrans, quantQ, quantScale, qNopeStrideToken,
                qNopeStrideHead, kBProjStrideHead, kBProjStrideDim, kBProjStrideReduction, tokenIdx, setupHeadIdx);
            // Rotate and quantize the RoPE suffix for the same [token, head].
            rotateGenerationQPeHead(fusedQ, qPe, rotaryCosSin, quantQ, quantScale, qPeStrideToken, qPeStrideHead,
                tokenIdx, tokenIdxInKvCache, setupHeadIdx);
        }

        if (splitIdx == 0)
        {
            // Preserve the old single-CTA cache append behavior when the launch lacks enough split CTAs to split it.
            appendGenerationKvCache(latentCache, rotaryCosSin, kvCache, quantScale, tokenIdx, tokenIdxInKvCache);
        }
    }

    // Wait until Q quantization, bmm scale publication, and paged KV appends are visible to every split CTA.
    generationGridBarrier(syncScratch, expectedCtas);

    // Threads 256..383 correspond to TileRT's producer/control group. This WIP version does not issue TMA gather
    // from them yet; they skip split math but stay alive so the CTA reaches the common grid barrier safely.
    if (tid < kTileRtConsumerThreads)
    {
        // Each of the 8 consumer warps owns one local attention head.
        int const headIdx = warpIdx;
        // Each head gets 64 score/probability slots inside dynamic shared memory.
        float* headScores = splitScores + headIdx * kSparseSplitSize;
        // First top-k index owned by this sparse split.
        int const splitStart = splitIdx * kSparseSplitSize;
        // Current decode/MTP group's first local sequence position.
        int const currentGroupStart = currentGroupStartShared;
        // bmm1Scale[1] is already in log2 space. exp2f below therefore matches the existing generation path.
        float const scoreScaleLog2 = bmm1Scale == nullptr ? 1.4426950408889634F : bmm1Scale[1];
        // Full FP8 query vector for this token/head.
        __nv_fp8_e4m3 const* qPtr = quantQ + (tokenIdx * kLocalHeads + headIdx) * kHeadDim;
        // Track the largest log2 score inside this 64-key split.
        float maxScore = -INFINITY;

        // QK pass: one consumer warp computes all candidate scores for its head over this 64-key split.
        for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
        {
            // Convert split-local offset to global top-k index.
            int const topkIdx = splitStart + splitOffset;
            // Only indices inside topK are real. The tail of a non-multiple-of-64 split is invalid.
            bool const inRange = topkIdx < topK;
            // Load pool and local indices when in range; otherwise use invalid sentinels.
            int const kvIdx = inRange ? topkIndicesPool[tokenIdx * topK + topkIdx] : -1;
            // Local sequence position drives causal/speculative visibility checks.
            int const localKvIdx = inRange ? topkIndicesLocal[tokenIdx * topK + topkIdx] : -1;
            // Candidate must have a real pool row and be visible to this query token.
            bool const validKv = kvIdx >= 0
                && isGenerationKvVisible(
                    tokenIdx, localKvIdx, currentGroupStart, numTokens, specDecodingPackedMask, specMaskWords);
            // Each lane accumulates a strided slice of the 576-dim QK dot product.
            float partial = 0.0F;
            if (validKv)
            {
                // Use 64-bit offset math because real pool row ids can be large in the benchmark.
                int64_t const kvOffset = static_cast<int64_t>(kvIdx) * kHeadDim;
                // Pointer to the selected FP8 latent KV row.
                __nv_fp8_e4m3 const* kvPtr = kvCachePool + kvOffset;
                // Stride the warp lanes over [0, 576).
                for (int dim = laneIdx; dim < kHeadDim; dim += 32)
                {
                    // Accumulate q_fp8[dim] * k_fp8[dim] in FP32.
                    partial += toFloat(qPtr[dim]) * toFloat(kvPtr[dim]);
                }
            }
            // Warp-local reduction produces the dot product in lane 0.
            float const dot = warpReduceSum(partial);
            if (laneIdx == 0)
            {
                // Invalid candidates become -inf so their softmax contribution is exactly zero.
                float const score = validKv ? dot * scoreScaleLog2 : -INFINITY;
                // Store the log2 score in the split-local scratch row.
                headScores[splitOffset] = score;
                // Track the split-local maximum for stable exp2 softmax.
                maxScore = fmaxf(maxScore, score);
            }
        }
        // Make lane-0 score writes visible to the other lanes in this head warp.
        __syncwarp();

        // Lane 0 computes split-local denominator, normalizes probabilities, and publishes this split's LSE.
        if (laneIdx == 0)
        {
            // Sum of exp2(score - maxScore) for valid candidates inside this split.
            float denom = 0.0F;
            if (maxScore != -INFINITY)
            {
                // Convert scores to normalized split-local probabilities in place.
                for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
                {
                    // Subtracting maxScore keeps the largest exponent at exp2(0) == 1.
                    float const unnormalized = exp2f(headScores[splitOffset] - maxScore);
                    // Accumulate the split-local denominator.
                    denom += unnormalized;
                    // Temporarily store the unnormalized probability.
                    headScores[splitOffset] = unnormalized;
                }
                // Normalize each probability and store log2 LSE for the combine phase.
                float const invDenom = 1.0F / denom;
                for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
                {
                    // headScores now stores split-local softmax probability.
                    headScores[splitOffset] *= invDenom;
                }
                // LSE in log2 space: max + log2(sum(exp2(score - max))).
                splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + splitIdx] = maxScore + log2f(denom);
            }
            else
            {
                // No valid candidate in this split: probability is zero and LSE is -inf.
                for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
                {
                    headScores[splitOffset] = 0.0F;
                }
                // Mark this split as empty for the combine phase.
                splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + splitIdx] = -INFINITY;
            }
        }
        // Make normalized probabilities visible to every lane in this head warp.
        __syncwarp();

        // AV pass: write split-local normalized latent output in raw FP8-cache units.
        for (int dim = laneIdx; dim < kKvLoraRank; dim += 32)
        {
            // Accumulator for one latent V dimension for this [token, head, split].
            float acc = 0.0F;
            // Sum split-local probability times selected latent V value.
            for (int splitOffset = 0; splitOffset < kSparseSplitSize; ++splitOffset)
            {
                // Convert split-local offset to global top-k index.
                int const topkIdx = splitStart + splitOffset;
                // Skip tail entries outside topK.
                bool const inRange = topkIdx < topK;
                // Load the selected pool row or invalid sentinel.
                int const kvIdx = inRange ? topkIndicesPool[tokenIdx * topK + topkIdx] : -1;
                // Load the local sequence position or invalid sentinel.
                int const localKvIdx = inRange ? topkIndicesLocal[tokenIdx * topK + topkIdx] : -1;
                // Reuse the exact same visibility rule as the QK pass.
                bool const validKv = kvIdx >= 0
                    && isGenerationKvVisible(
                        tokenIdx, localKvIdx, currentGroupStart, numTokens, specDecodingPackedMask, specMaskWords);
                if (validKv)
                {
                    // Flattened [pool row, latent dim] address. Only dims [0, 512) are latent V.
                    int64_t const kvOffset = static_cast<int64_t>(kvIdx) * kHeadDim + dim;
                    // Accumulate split-softmax probability times raw FP8-cache V value.
                    acc += headScores[splitOffset] * toFloat(kvCachePool[kvOffset]);
                }
            }
            // Store FP32 partial output for the combine phase. Layout is [token, head, split, dim].
            splitOutput[((tokenIdx * kLocalHeads + headIdx) * numSplits + splitIdx) * kKvLoraRank + dim] = acc;
        }
    }

    // Wait until every split CTA has published splitLse and splitOutput. Use a second independent one-shot barrier
    // because the append-phase barrier's release flag remains set for the rest of this kernel launch.
    generationGridBarrier(syncScratch + 2, expectedCtas);

    // Reuse the split CTAs for each token as combine collectors. With the target topK=2048 case there are 32 split
    // CTAs per token, so the first eight CTAs collect one local head each. Small-topK tests may have fewer than eight
    // splits, so each available split CTA strides over the local-head dimension.
    if (splitIdx < kLocalHeads)
    {
        // The global log2 LSE is scalar per [token, head].
        __shared__ float globalLseShared;
        for (int headIdx = splitIdx; headIdx < kLocalHeads; headIdx += numSplits)
        {
            if (tid == 0)
            {
                // First pass finds max split LSE for numerical stability.
                float maxLse = -INFINITY;
                for (int combineSplitIdx = 0; combineSplitIdx < numSplits; ++combineSplitIdx)
                {
                    // Load split LSE in log2 space.
                    float const lse = splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + combineSplitIdx];
                    // Track the largest non-empty split.
                    maxLse = fmaxf(maxLse, lse);
                }

                // If all splits are empty, mark the whole row empty.
                if (maxLse == -INFINITY)
                {
                    globalLseShared = -INFINITY;
                }
                else
                {
                    // Sum exp2(lse - maxLse) over all splits.
                    float denom = 0.0F;
                    for (int combineSplitIdx = 0; combineSplitIdx < numSplits; ++combineSplitIdx)
                    {
                        // Empty splits have lse=-inf and contribute zero.
                        float const lse = splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + combineSplitIdx];
                        denom += exp2f(lse - maxLse);
                    }
                    // Store global log2 LSE for every lane.
                    globalLseShared = maxLse + log2f(denom);
                }
            }
            // Ensure every thread in the collector CTA sees globalLseShared.
            __syncthreads();

            // bmm2Scale[0] dequantizes FP8 V/cache values before storing BF16 latent output.
            float const outputScale = bmm2Scale == nullptr ? 1.0F : bmm2Scale[0];
            // Read global log2 LSE into a register.
            float const globalLse = globalLseShared;

            // Each collector CTA writes the 512 latent V dimensions for one [token, head].
            for (int dim = tid; dim < kKvLoraRank; dim += blockDim.x)
            {
                // Accumulator in raw FP8-cache units before applying outputScale.
                float acc = 0.0F;
                if (globalLse != -INFINITY)
                {
                    // Combine all split-local normalized outputs using exp(lse_s - global_lse).
                    for (int combineSplitIdx = 0; combineSplitIdx < numSplits; ++combineSplitIdx)
                    {
                        // Load split LSE in log2 space.
                        float const lse = splitLse[(tokenIdx * kLocalHeads + headIdx) * numSplits + combineSplitIdx];
                        // Empty splits have zero combine weight.
                        float const splitWeight = lse == -INFINITY ? 0.0F : exp2f(lse - globalLse);
                        // Load split-local normalized latent output.
                        float const splitValue
                            = splitOutput[((tokenIdx * kLocalHeads + headIdx) * numSplits + combineSplitIdx)
                                    * kKvLoraRank
                                + dim];
                        // Add this split's contribution to the global softmax output.
                        acc += splitWeight * splitValue;
                    }
                }
                // Dequantize the FP8 V sum to original units and round to BF16 output.
                output[(tokenIdx * kLocalHeads + headIdx) * kKvLoraRank + dim] = toBfloat16(acc * outputScale);
            }
            // Do not let thread 0 overwrite globalLseShared for the next head before every thread finishes this head.
            __syncthreads();
        }
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
//   are unrotated shared K RoPE values. This tensor is left unchanged; preprocessing writes rotated K to
//   the paged cache and to a temporary FP8 current-context KV tensor.
// - topk_indices_pool: INT32 [numTokens, topK]. Global row indices into kv_cache_pool. This is kept in the
//   op signature for parity with generation and for shape checks; context attention gathers from topk_indices_local.
// - topk_indices_local: INT32 [numTokens, topK]. Local current-context sparse attention row indices.
//   Negative entries are padding.
// - kv_cache_pool: FP8 E4M3 [poolTokens, 1, 576], viewed by the kernel as [poolTokens, 576]. All 576 dims
//   are used as K; only dims [0, 512) are used as latent V.
// - rotary_cos_sin: FP32 float2 RoPE table indexed by absolute KV position.
// - ctx_cached_token_indptr: INT64 [numContexts + 1]. The current custom path uses one context;
//   ctx_cached_token_indptr[1] - ctx_cached_token_indptr[0] gives the cached prefix length used for
//   absolute RoPE and cache positions.
// - kv_cache: paged TRT-LLM KV cache view. preprocessContextKernel writes the rotated latent KV rows into
//   this cache. contextAttentionKernel reads from kv_cache_pool after this append.
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
// - Mutates kv_cache by appending/writing the context latent KV rows with rotated K RoPE.
//
// Output:
// - Returns BF16 [numTokens, 8 * 512], logically [numTokens, 8, 512]. Each row is the sparse
//   softmax-weighted sum of FP8 latent V rows from kv_cache_pool, in original/BF16 units.
torch::Tensor dsv3_fused_mla_context_cuda(torch::Tensor fused_q, torch::Tensor q_pe, torch::Tensor latent_cache,
    torch::Tensor topk_indices_pool, torch::Tensor topk_indices_local, torch::Tensor kv_cache_pool,
    torch::Tensor rotary_cos_sin, torch::Tensor ctx_cached_token_indptr, tensorrt_llm::kernels::KVBlockArray kv_cache,
    std::optional<torch::Tensor> kv_scale_orig_quant, std::optional<torch::Tensor> kv_scale_quant_orig,
    bool has_fp8_kv_cache, double q_scaling)
{
    // Set the active CUDA device to match fused_q so allocations and launches use the correct GPU.
    c10::cuda::CUDAGuard const deviceGuard{fused_q.device()};
    // Reuse PyTorch's current stream so this op stays ordered with surrounding torch operations and graph capture.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());

    // Number of context tokens in this prefill segment.
    int const numTokens = static_cast<int>(fused_q.size(0));
    // Sparse top-k width per query token.
    int const topK = static_cast<int>(topk_indices_pool.size(1));
    // Dynamic shared memory allocates one float per top-k candidate, bounded by kMaxTopK.
    TORCH_CHECK(topK <= kMaxTopK, "dsv3_fused_mla_context supports topK <= ", kMaxTopK, ", got ", topK);
    // Allocate BF16 output [numTokens, 8 * 512] with the same device and dtype options as fused_q.
    auto output = torch::empty({numTokens, kLocalHeads * kKvLoraRank}, fused_q.options());
    // Optional original->FP8 scale. Null means identity scale.
    float const* kvScaleOrigQuantPtr
        = kv_scale_orig_quant.has_value() ? kv_scale_orig_quant.value().data_ptr<float>() : nullptr;
    // Optional FP8->original scale. Null means identity scale.
    float const* kvScaleQuantOrigPtr
        = kv_scale_quant_orig.has_value() ? kv_scale_quant_orig.value().data_ptr<float>() : nullptr;
    // rotary_cos_sin stores adjacent [cos, sin] pairs, so reinterpret the raw storage as float2.
    auto* rotaryCosSinPtr = reinterpret_cast<float2 const*>(rotary_cos_sin.data_ptr());
    // Raw BF16 pointer for mutable fusedQ [numTokens, 8, 576].
    auto* fusedQPtr = static_cast<__nv_bfloat16*>(fused_q.data_ptr());
    // Raw BF16 pointer for qPe [numTokens, 8, 64] or an equivalent strided view.
    auto* qPePtr = static_cast<__nv_bfloat16*>(q_pe.data_ptr());
    // Raw BF16 pointer for latentCache [numTokens, 576].
    auto* latentCachePtr = static_cast<__nv_bfloat16*>(latent_cache.data_ptr());
    // Raw FP8 pointer for flattened primary KV-cache pool [poolTokens, 576].
    auto* kvCachePoolPtr = reinterpret_cast<__nv_fp8_e4m3*>(kv_cache_pool.data_ptr());
    // Raw BF16 pointer for output [numTokens, 8 * 512].
    auto* outputPtr = static_cast<__nv_bfloat16*>(output.data_ptr());

    // Launch one CTA per token for each of 8 Q-head rotations plus one KV-cache write branch.
    dim3 preprocessGrid(numTokens, kLocalHeads + 1);
    // q_pe may be a non-contiguous split view; pass token stride explicitly.
    int64_t const qPeStrideToken = q_pe.stride(0);
    // q_pe may be a non-contiguous split view; pass head stride explicitly.
    int64_t const qPeStrideHead = q_pe.stride(1);
    // Instantiate the preprocessing kernel according to the paged KV-cache storage dtype.
    if (has_fp8_kv_cache)
    {
        // FP8 KV-cache mode writes quantized rows to kv_cache.
        preprocessContextKernel<true><<<preprocessGrid, kThreads, 0, stream>>>(fusedQPtr, latentCachePtr, qPePtr,
            rotaryCosSinPtr, ctx_cached_token_indptr.data_ptr<int64_t>(), kv_cache, kvScaleOrigQuantPtr, qPeStrideToken,
            qPeStrideHead, numTokens);
    }
    else
    {
        // BF16 KV-cache mode writes BF16 rows to kv_cache. This mode is not used by the current GLM-5 FP8 wrapper.
        preprocessContextKernel<false><<<preprocessGrid, kThreads, 0, stream>>>(fusedQPtr, latentCachePtr, qPePtr,
            rotaryCosSinPtr, ctx_cached_token_indptr.data_ptr<int64_t>(), kv_cache, kvScaleOrigQuantPtr, qPeStrideToken,
            qPeStrideHead, numTokens);
    }

    // Context score scale follows the MLA absorbed-Q reference: 1 / (q_scaling * sqrt(qk_head_dim)).
    // qk_head_dim is 256 for GLM-5 because absorbed q_nope is 192 and RoPE is 64.
    float const hostScoreScale = 1.0F / (static_cast<float>(q_scaling) * sqrtf(256.0F));
    // Launch one CTA per [token, head] output row.
    dim3 attentionGrid(numTokens, kLocalHeads);
    // Shared memory stores kThreads reduction floats plus topK score/weight floats.
    size_t const sharedBytes = (kThreads + std::min(topK, kMaxTopK)) * sizeof(float);
    // Compute sparse context attention from the primary KV-cache pool so chunked prefill can read cached prefixes.
    contextAttentionKernel<<<attentionGrid, kThreads, sharedBytes, stream>>>(fusedQPtr, kvCachePoolPtr,
        topk_indices_pool.data_ptr<int32_t>(), topk_indices_local.data_ptr<int32_t>(),
        ctx_cached_token_indptr.data_ptr<int64_t>(), outputPtr, kvScaleOrigQuantPtr, kvScaleQuantOrigPtr, numTokens,
        topK, hostScoreScale);

    // Surface any asynchronous launch/runtime errors before returning the tensor to Python.
    sync_check_cuda_error(stream);
    // Return BF16 [numTokens, 8 * 512] latent attention output.
    return output;
}

// Fused GLM-5/DeepSeek-V3 sparse MLA generation/decode path.
//
// Inputs:
// - fused_q: mutable BF16 [numTokens, 8, 576]. The combined kernel fills dims [0, 512) with q_nope absorbed
//   by k_b_proj_trans and dims [512, 576) with rotated Q RoPE.
// - q_nope: BF16 [numTokens, 8, 192]. Unabsorbed query prefix from the Q projection split view.
// - k_b_proj_trans: BF16 [8, 512, 192]. Per-head K absorption projection weight.
// - q_pe: BF16 [numTokens, 8, 64]. Unrotated per-head Q RoPE suffix.
// - latent_cache: BF16 [numTokens, 576]. Current decode/MTP latent KV/V plus unrotated shared K RoPE.
// - rotary_cos_sin: FP32 float2 RoPE table.
// - sequence_length: INT32 [1], total KV length after appending this decode/MTP group.
// - kv_cache: paged TRT-LLM KV cache view to receive the current decode/MTP rows.
// - topk_indices: INT32 [numTokens, topK], local sequence positions used for MTP visibility checks.
// - topk_indices_pool: INT32 [numTokens, topK], global KV-cache pool rows.
// - kv_cache_pool: FP8 E4M3 [poolTokens, 1, 576], flattened by the kernel as [poolTokens, 576].
// - quant_q_buffer: UINT8-backed FP8 E4M3 [numTokens, 8, 576].
// - mla_bmm1_scale: FP32 [2], with index 1 holding log2-space QK score scale.
// - mla_bmm2_scale: FP32 [1], FP8 V dequantization scale.
// - kv_scale_orig_quant: optional FP32 [1], original-domain to FP8 scale.
// - kv_scale_quant_orig: optional FP32 [1], FP8 to original-domain scale.
// - spec_decoding_packed_mask: optional INT32 [maxRequests, numTokens, packedWords].
// - q_scaling: model attention scaling divisor.
//
// Output:
// - Returns BF16 [numTokens, 8 * 512], logically [numTokens, 8, 512].
//
// Side effects:
// - Mutates fused_q absorbed prefix and RoPE suffix for debug visibility.
// - Mutates quant_q_buffer, mla_bmm1_scale, mla_bmm2_scale, and kv_cache before running attention.
// - Writes the returned output tensor.
torch::Tensor dsv3_fused_mla_generation_cuda(torch::Tensor fused_q, torch::Tensor q_nope, torch::Tensor k_b_proj_trans,
    torch::Tensor q_pe, torch::Tensor latent_cache, torch::Tensor rotary_cos_sin, torch::Tensor sequence_length,
    tensorrt_llm::kernels::KVBlockArray kv_cache, torch::Tensor topk_indices, torch::Tensor topk_indices_pool,
    torch::Tensor kv_cache_pool, torch::Tensor quant_q_buffer, torch::Tensor mla_bmm1_scale,
    torch::Tensor mla_bmm2_scale, std::optional<torch::Tensor> kv_scale_orig_quant,
    std::optional<torch::Tensor> kv_scale_quant_orig, std::optional<torch::Tensor> spec_decoding_packed_mask,
    double q_scaling)
{
    // Set the active CUDA device to match fused_q so the launch uses the correct GPU.
    c10::cuda::CUDAGuard const deviceGuard{fused_q.device()};
    // Use PyTorch's current stream for stream-ordering and CUDA graph capture compatibility.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());

    // Number of decode/MTP query rows in this call.
    int const numTokens = static_cast<int>(fused_q.size(0));
    // Sparse top-k width per query row.
    int const topK = static_cast<int>(topk_indices_pool.size(1));
    // Dynamic shared memory uses topK floats for weights, so enforce the development-kernel bound.
    TORCH_CHECK(topK <= kMaxTopK, "dsv3_fused_mla_generation supports topK <= ", kMaxTopK, ", got ", topK);
    // Optional original->FP8 scale pointer.
    float const* kvScaleOrigQuantPtr
        = kv_scale_orig_quant.has_value() ? kv_scale_orig_quant.value().data_ptr<float>() : nullptr;
    // Optional FP8->original scale pointer.
    float const* kvScaleQuantOrigPtr
        = kv_scale_quant_orig.has_value() ? kv_scale_quant_orig.value().data_ptr<float>() : nullptr;
    // RoPE table is stored as adjacent [cos, sin] pairs.
    auto* rotaryCosSinPtr = reinterpret_cast<float2 const*>(rotary_cos_sin.data_ptr());
    // Raw BF16 fused Q pointer [numTokens, 8, 576].
    auto* fusedQPtr = static_cast<__nv_bfloat16*>(fused_q.data_ptr());
    // Raw BF16 q_nope pointer [numTokens, 8, 192]. It may be a strided split view.
    auto* qNopePtr = static_cast<__nv_bfloat16 const*>(q_nope.data_ptr());
    // Raw BF16 k_b_proj_trans pointer [8, 512, 192].
    auto* kBProjTransPtr = static_cast<__nv_bfloat16 const*>(k_b_proj_trans.data_ptr());
    // Raw BF16 qPe pointer. It may be a strided view.
    auto* qPePtr = static_cast<__nv_bfloat16 const*>(q_pe.data_ptr());
    // Raw BF16 latent cache pointer [numTokens, 576].
    auto* latentCachePtr = static_cast<__nv_bfloat16 const*>(latent_cache.data_ptr());
    // quant_q_buffer is a uint8 tensor whose bytes are FP8 E4M3 values.
    auto* quantQPtr = reinterpret_cast<__nv_fp8_e4m3*>(quant_q_buffer.data_ptr());
    // q_nope may be a split view, so pass runtime strides.
    int64_t const qNopeStrideToken = q_nope.stride(0);
    // Head stride for q_nope.
    int64_t const qNopeStrideHead = q_nope.stride(1);
    // Head stride for k_b_proj_trans.
    int64_t const kBProjStrideHead = k_b_proj_trans.stride(0);
    // Output-dimension stride for k_b_proj_trans.
    int64_t const kBProjStrideDim = k_b_proj_trans.stride(1);
    // Reduction-dimension stride for k_b_proj_trans.
    int64_t const kBProjStrideReduction = k_b_proj_trans.stride(2);
    // q_pe may be a split view, so pass runtime strides.
    int64_t const qPeStrideToken = q_pe.stride(0);
    // Head stride for q_pe.
    int64_t const qPeStrideHead = q_pe.stride(1);
    // Generic MLA RoPE uses 1 / (q_scaling * sqrt(qk_nope + qk_rope)); GLM-5 has 192 + 64.
    float const hostScoreScale = 1.0F / (static_cast<float>(q_scaling) * sqrtf(256.0F));

    // Allocate BF16 output [numTokens, 8 * 512] on the same device as fused_q.
    auto output = torch::empty({numTokens, kLocalHeads * kKvLoraRank}, fused_q.options());
    // TileRT splits topK=2048 into 32 chunks of 64 selected KV rows. Keep the same split size and support
    // smaller/larger topK by rounding up while preserving the exact GLM-5 32-CTA/token case.
    int const numSplits = (topK + kSparseSplitSize - 1) / kSparseSplitSize;
    // The in-kernel global barrier is safe only while every split CTA can be resident concurrently. This WIP path
    // assumes one resident split CTA per SM, so guard both the TileRT target cap and the actual GPU SM count.
    cudaDeviceProp deviceProperties;
    TLLM_CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, fused_q.get_device()));
    int const safeResidentSplitCtas = deviceProperties.multiProcessorCount < kTileRtMaxResidentSplitCtas
        ? deviceProperties.multiProcessorCount
        : kTileRtMaxResidentSplitCtas;
    TORCH_CHECK(numTokens * numSplits <= safeResidentSplitCtas,
        "dsv3_fused_mla_generation combined kernel supports at most ", safeResidentSplitCtas,
        " resident split CTAs on this GPU, got ", numTokens * numSplits);
    // Global FP32 scratch mirrors TileRT's split-softmax scratch concept. The combined kernel writes partials,
    // grid-syncs, and then reads them back from this scratch in the same launch.
    auto const scratchOptions = fused_q.options().dtype(torch::kFloat32);
    // splitLse: FP32 [numTokens, 8, numSplits], log2 LSE for each sparse split.
    auto splitLse = torch::empty({numTokens, kLocalHeads, numSplits}, scratchOptions);
    // splitOutput: FP32 [numTokens, 8, numSplits, 512], split-local normalized latent output.
    auto splitOutput = torch::empty({numTokens, kLocalHeads, numSplits, kKvLoraRank}, scratchOptions);
    // syncScratch: INT32 [4]. Entries [0, 1] are the append-phase barrier state, and entries [2, 3] are the
    // split-output barrier state.
    auto syncScratch = torch::empty({4}, fused_q.options().dtype(torch::kInt32));
    // Start every combined launch with both barrier states clear.
    TLLM_CUDA_CHECK(cudaMemsetAsync(syncScratch.data_ptr<int32_t>(), 0, 4 * sizeof(int32_t), stream));
    // kv_cache_pool is an FP8 E4M3 tensor [poolTokens, 1, 576].
    auto* kvCachePoolPtr = reinterpret_cast<__nv_fp8_e4m3 const*>(kv_cache_pool.data_ptr());
    // Raw BF16 output pointer [numTokens, 8 * 512].
    auto* outputPtr = static_cast<__nv_bfloat16*>(output.data_ptr());
    // Optional packed mask pointer. Null means use causal current-group rule.
    int32_t const* specDecodingPackedMaskPtr
        = spec_decoding_packed_mask.has_value() ? spec_decoding_packed_mask.value().data_ptr<int32_t>() : nullptr;
    // Number of int32 mask words per query token row.
    int const specMaskWords
        = spec_decoding_packed_mask.has_value() ? static_cast<int>(spec_decoding_packed_mask.value().size(-1)) : 0;

    // Launch one CTA per [decode token, 64-key sparse split], matching TileRT's 32 CTAs/token for topK=2048.
    dim3 splitGrid(numTokens * numSplits);
    // Shared memory stores [8 heads, 64 split candidates] score/probability slots.
    size_t const splitSharedBytes = kLocalHeads * kSparseSplitSize * sizeof(float);
    // The combined launch first performs RoPE/Q quantization/KV append, grid-syncs, computes split-local partials,
    // grid-syncs again, and then combines into final latent outputs.
    generationAttentionCombinedKernel<<<splitGrid, kTileRtThreads, splitSharedBytes, stream>>>(fusedQPtr, qNopePtr,
        kBProjTransPtr, qPePtr, latentCachePtr, rotaryCosSinPtr, kv_cache, quantQPtr, kvCachePoolPtr,
        topk_indices_pool.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(), sequence_length.data_ptr<int32_t>(),
        specDecodingPackedMaskPtr, specMaskWords, splitLse.data_ptr<float>(), splitOutput.data_ptr<float>(),
        syncScratch.data_ptr<int32_t>(), outputPtr, mla_bmm1_scale.data_ptr<float>(), mla_bmm2_scale.data_ptr<float>(),
        kvScaleOrigQuantPtr, kvScaleQuantOrigPtr, qNopeStrideToken, qNopeStrideHead, kBProjStrideHead, kBProjStrideDim,
        kBProjStrideReduction, qPeStrideToken, qPeStrideHead, numTokens, topK, numSplits, numTokens * numSplits,
        hostScoreScale);

    // Report any asynchronous CUDA error before returning to Python.
    sync_check_cuda_error(stream);
    // Return BF16 [numTokens, 8 * 512] latent attention output.
    return output;
}

} // namespace dsv3_fused_mla
