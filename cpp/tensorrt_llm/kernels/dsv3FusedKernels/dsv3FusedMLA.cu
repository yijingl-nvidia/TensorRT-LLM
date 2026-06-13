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
#include <torch/extension.h>

// #include <cmath>
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

// This file is intentionally specialized to the GLM-5/DeepSeek-V3 fused MLA shape used by the WIP Python path.
// The constants above are part of that specialization:
// - kLocalHeads is the tensor-parallel local attention head count on TP=8.
// - kKvLoraRank is the absorbed latent KV/V dimension used by MLA.
// - kRopeDim is the per-head RoPE suffix dimension.
// - kHeadDim is the full latent attention dimension, [latent KV/V, RoPE] == 512 + 64.
// - kThreads is the fixed CTA width used for reductions and strided vector loops.
// - kMaxTopK bounds dynamic shared memory for sparse top-k attention weights.
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

// Preprocess one context token row before sparse MLA attention.
//
// Grid:
// - blockIdx.x selects the context token row in [0, numTokens).
// - blockIdx.y selects the preprocessing role:
//   - [0, kLocalHeads) rotates one per-head Q RoPE suffix from qPe into fusedQ.
//   - kLocalHeads reads the shared latent K RoPE suffix from latentCache, rotates it, then writes the full
//     latent KV row into the temporary current-context KV tensor and the paged KV cache.
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
// - contextKvFp8: mutable FP8 E4M3 [numTokens, kHeadDim]. The KV branch writes the compact current-context
//   cache consumed by contextAttentionKernel.
// - kvScaleOrigQuant: optional FP32 [1]. BF16 latentCache values are multiplied by this original-domain to
//   FP8 scale before storing to contextKvFp8 and, when kFp8KvCache is true, the paged KV cache.
// - numTokens: number of context tokens in fusedQ/qPe/latentCache.
//
// Side effects:
// - Q branch mutates fusedQ[token, head, kKvLoraRank:kHeadDim].
// - KV branch writes latentCache[token, :] to contextKvFp8 and kvCache with a rotated K RoPE suffix.
// - The function does not produce a returned tensor; contextAttentionKernel consumes fusedQ and contextKvFp8 later.
template <bool kFp8KvCache>
__global__ void preprocessContextKernel(__nv_bfloat16* fusedQ, __nv_bfloat16* latentCache, __nv_bfloat16 const* qPe,
    float2 const* rotaryCosSin, int64_t const* ctxCachedTokenIndptr, tensorrt_llm::kernels::KVBlockArray kvCache,
    __nv_fp8_e4m3* contextKvFp8, float const* kvScaleOrigQuant, int64_t qPeStrideToken, int64_t qPeStrideHead,
    int numTokens)
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
        // Quantize BF16 original-domain value to FP8 E4M3 using the same cache quantization scale as TRTLLM MLA.
        __nv_fp8_e4m3 const quantized = __nv_fp8_e4m3(toFloat(value) * kvScale);
        // Save an FP8 current-context copy for contextAttentionKernel, avoiding reads from paged cache pages.
        contextKvFp8[tokenIdx * kHeadDim + dim] = quantized;
        // Write the same value to the real paged KV cache so later decode steps can attend to this prefix.
        if constexpr (kFp8KvCache)
        {
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
        // Quantize the first rounded scalar for the temporary FP8 context KV and FP8 paged cache.
        __nv_fp8_e4m3 const quantizedFirst = __nv_fp8_e4m3(toFloat(first) * kvScale);
        // Quantize the second rounded scalar for the temporary FP8 context KV and FP8 paged cache.
        __nv_fp8_e4m3 const quantizedSecond = __nv_fp8_e4m3(toFloat(second) * kvScale);
        // Save the first rotated K suffix component in the temporary current-context FP8 buffer.
        contextKvFp8[tokenIdx * kHeadDim + kKvLoraRank + dim] = quantizedFirst;
        // Save the second rotated K suffix component in the temporary current-context FP8 buffer.
        contextKvFp8[tokenIdx * kHeadDim + kKvLoraRank + dim + 1] = quantizedSecond;
        // Mirror the rotated K suffix into the paged KV cache for later decode.
        if constexpr (kFp8KvCache)
        {
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
// - contextKvFp8: FP8 E4M3 [numTokens, kHeadDim]. Rotated current-context KV rows
//   produced by preprocessContextKernel. All 576 dims are used as K for QK scores;
//   only the first 512 dims are used as latent V for the output.
// - topkIndicesLocal: INT32 [numTokens, topK]. For each query token, contains
//   local current-context row indices into contextKvFp8. Negative entries are padding.
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
__global__ void contextAttentionKernel(__nv_bfloat16 const* fusedQ, __nv_fp8_e4m3 const* contextKvFp8,
    int32_t const* topkIndicesLocal, __nv_bfloat16* output, float const* kvScaleOrigQuant,
    float const* kvScaleQuantOrig, int numTokens, int topK, float hostScoreScale)
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
        // topkIndicesLocal is INT32 [numTokens, topK]. Negative values are padding and become -inf scores.
        // localKvIdxRaw is the current-context row selected for this sparse candidate.
        int const localKvIdxRaw = topkIndicesLocal[tokenIdx * topK + topkIdx];
        // Only non-negative top-k entries point at a real context KV row.
        bool const validKv = localKvIdxRaw >= 0;
        // Clamp invalid entries so address arithmetic is safe when validKv is false.
        int const localKvIdx = min(max(localKvIdxRaw, 0), numTokens - 1);
        // Each thread accumulates a subset of the 576-dimensional dot product.
        float partial = 0.0F;
        // Skip memory reads for invalid/padded sparse candidates.
        if (validKv)
        {
            // kvPtr points at FP8 [576] for one sparse current-context KV row.
            // contextKvFp8 is compact current-context storage, so 32-bit local row math is enough here.
            __nv_fp8_e4m3 const* kvPtr = contextKvFp8 + localKvIdx * kHeadDim;
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
            // Reload the local context row index for this candidate.
            int const localKvIdxRaw = topkIndicesLocal[tokenIdx * topK + topkIdx];
            // Negative entries are padding and must not contribute.
            bool const validKv = localKvIdxRaw >= 0;
            // Clamp invalid entries to keep address arithmetic safe in the guarded block.
            int const localKvIdx = min(max(localKvIdxRaw, 0), numTokens - 1);
            // Accumulate only real sparse candidates.
            if (validKv)
            {
                // The trusted context path rounds probabilities to BF16 before BMM2.
                float const prob = toFloat(toBfloat16(weights[topkIdx] * invDenom));
                // The trusted context path also rounds FP8 V values through BF16 before multiply-add.
                float const value = toFloat(toBfloat16(toFloat(contextKvFp8[localKvIdx * kHeadDim + dim])));
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

// Generation/decode sparse MLA attention kernel.
//
// This kernel computes the decode-stage latent attention output for the GLM-5 FP8 BS=1, MTP=3 path. The
// RoPE/append kernel has already produced FP8 Q and has already appended the current decode/MTP-group latent KV
// rows into the paged KV cache. This kernel only reads those buffers and performs sparse QK softmax and V
// accumulation.
//
// Inputs:
// - quantQ: FP8 E4M3 [numTokens, kLocalHeads, kHeadDim]. Rotated and quantized Q from mla_rope_generation.
// - kvCachePool: FP8 E4M3 [poolTokens, 1, kHeadDim], flattened as [poolTokens, kHeadDim]. This is the full
//   primary KV-cache pool view across pages/layers, so pool row indices can be millions of rows.
// - topkIndicesPool: INT32 [numTokens, topK]. Global row indices into kvCachePool produced from local top-k.
// - topkIndicesLocal: INT32 [numTokens, topK]. Local sequence positions used only to decide whether a candidate
//   is valid for the current MTP token. Negative entries are padding.
// - sequenceLength: INT32 [1]. Total KV length after the current numTokens decode group has been appended.
// - specDecodingPackedMask: optional INT32 [maxRequests, numTokens, specMaskWords]. For current-group rows, bit
//   currentGroupOffset tells whether query token tokenIdx can attend to that current-group KV row.
// - specMaskWords: int number of packed int32 words per token row in specDecodingPackedMask.
// - bmm1Scale: optional FP32 [2]. bmm1Scale[1] is the log2-space QK score scale used with exp2f.
// - bmm2Scale: optional FP32 [1]. Dequantization scale for FP8 V values before BF16 output.
// - numTokens: int, number of decode/MTP tokens in this call. For GLM-5 MTP=3 this is 4.
// - topK: int, sparse candidate count per query token.
//
// Output:
// - output: BF16 [numTokens, kLocalHeads * kKvLoraRank], logically [numTokens, 8, 512].
//
// Side effects:
// - Writes output only. It does not mutate Q, KV cache, or top-k tensors.
// - Uses dynamic shared memory [kThreads + topK] floats for reductions and softmax weights.
__global__ void generationAttentionKernel(__nv_fp8_e4m3 const* quantQ, __nv_fp8_e4m3 const* kvCachePool,
    int32_t const* topkIndicesPool, int32_t const* topkIndicesLocal, int32_t const* sequenceLength,
    int32_t const* specDecodingPackedMask, int specMaskWords, __nv_bfloat16* output, float const* bmm1Scale,
    float const* bmm2Scale, int numTokens, int topK)
{
    // One CUDA block computes one (generation query token, local head) row.
    // Current Python dispatch restricts this custom path to one generation sequence, but numTokens may be
    // greater than one for MTP/spec decode. In that case each token row has a different causal KV end.
    // Allocate dynamic shared memory provided by the launch.
    extern __shared__ float shared[];
    // The first kThreads floats are used by blockReduceSum for QK dot-product reductions.
    float* reduce = shared;
    // The remaining topK floats store log2 scores first, then unnormalized softmax weights.
    float* weights = shared + blockDim.x;

    // Decode/MTP token row selected by this CTA.
    int const tokenIdx = static_cast<int>(blockIdx.x);
    // Local attention head selected by this CTA.
    int const headIdx = static_cast<int>(blockIdx.y);
    // Thread lane id used for all strided loops in this CTA.
    int const tid = threadIdx.x;
    // sequenceLength is INT32 [1] after mla_rope_generation appended the entire numTokens decode chunk.
    // Historical rows before currentGroupStart are always valid when selected by topK. Rows inside the current
    // decode/MTP group are filtered either by the packed spec-decoding mask or by the local causal order.
    // The group start is shared because all token/head CTAs in this launch use the same sequence.
    __shared__ int currentGroupStartShared;
    // One lane computes the current decode/MTP group's first local sequence position.
    if (tid == 0)
    {
        // Normally: post-append length minus number of just-appended decode tokens.
        int currentGroupStart = sequenceLength[0] - numTokens;
        // CUDA graph warmup can capture a dummy sequenceLength equal to numTokens. If the top-k tensor was
        // replay-updated to real sequence positions, infer the group start from the max local selected index.
        if (sequenceLength[0] <= numTokens)
        {
            // Initialize to no valid top-k entries.
            int maxLocalIdx = -1;
            // Scan the full [numTokens, topK] local-index tensor. This is only used for dummy-length graph cases.
            for (int idx = 0; idx < numTokens * topK; ++idx)
            {
                // Negative padding entries do not increase maxLocalIdx.
                maxLocalIdx = max(maxLocalIdx, topkIndicesLocal[idx]);
            }
            // If top-k contains real sequence positions, maxLocalIdx should be currentGroupStart + numTokens - 1.
            if (maxLocalIdx >= numTokens)
            {
                // Recover the first local position of the current decode/MTP group.
                currentGroupStart = maxLocalIdx - (numTokens - 1);
            }
        }
        // Publish the group start to every lane in this CTA.
        currentGroupStartShared = currentGroupStart;
    }
    // Ensure every lane sees the group start before candidate validity checks.
    __syncthreads();
    // Read the shared current-group start into a register.
    int const currentGroupStart = currentGroupStartShared;
    // bmm1Scale[1] is already in log2 space. Using exp2f() below matches TRTLLM-Gen's softmax convention.
    // Null scale is a defensive fallback; production FP8 path passes a valid bmm1Scale tensor.
    float const scoreScaleLog2 = bmm1Scale == nullptr ? 1.4426950408889634F : bmm1Scale[1];
    // bmm2Scale[0] dequantizes FP8 V/cache values before storing BF16 latent output.
    // Null scale is a defensive fallback; production FP8 path passes a valid bmm2Scale tensor.
    float const outputScale = bmm2Scale == nullptr ? 1.0F : bmm2Scale[0];

    // Defensive guard for padded launches. Current launch grid is exact.
    if (tokenIdx >= numTokens || headIdx >= kLocalHeads)
    {
        // This CTA owns no output row.
        return;
    }

    // Track the largest log2 score for stable softmax.
    float maxScore = -INFINITY;

    // First pass over sparse candidates computes QK logits and the row maximum.
    for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
    {
        // topkIndicesPool: INT32 [numTokens, topK], converted global pool row for data loads.
        // topkIndicesLocal: INT32 [numTokens, topK], original local KV position for causal validity.
        // kvIdx is a global row in the flattened pool view. It can be large, so later offset math uses int64_t.
        int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
        // localKvIdx is the local sequence position used to decide if tokenIdx may attend to this candidate.
        int const localKvIdx = topkIndicesLocal[tokenIdx * topK + topkIdx];
        // Offset 0 means the first token in the current MTP group; negative means historical KV.
        int const currentGroupOffset = localKvIdx - currentGroupStart;
        // Historical KV rows precede the current MTP group and are always visible when selected.
        bool const historicalKv = localKvIdx >= 0 && localKvIdx < currentGroupStart;
        // Current-group rows are the just-appended decode/MTP tokens
        // [currentGroupStart, currentGroupStart + numTokens).
        bool const currentGroupKv = currentGroupOffset >= 0 && currentGroupOffset < numTokens;
        // Start as invalid; set true only when the current-group mask or causal rule allows it.
        bool currentGroupValid = false;
        // Only current-group candidates need an intra-group visibility check.
        if (currentGroupKv)
        {
            // Spec/MTP path provides an explicit packed mask when available.
            if (specDecodingPackedMask != nullptr && specMaskWords > 0)
            {
                // Select which int32 word contains the bit for currentGroupOffset.
                int const wordIdx = currentGroupOffset / 32;
                // Select the bit inside that int32 word.
                int const bitIdx = currentGroupOffset % 32;
                // Load the packed mask word for this query token row.
                uint32_t const packed
                    = static_cast<uint32_t>(specDecodingPackedMask[tokenIdx * specMaskWords + wordIdx]);
                // A set bit means query tokenIdx may attend to this current-group KV row.
                currentGroupValid = ((packed >> bitIdx) & 1U) != 0U;
            }
            else
            {
                // No packed mask: fall back to local causal order inside the current group.
                currentGroupValid = currentGroupOffset <= tokenIdx;
            }
        }
        // A candidate is valid only if it has a real pool row and is either historical or current-group-visible.
        bool const validKv = kvIdx >= 0 && (historicalKv || (currentGroupKv && currentGroupValid));
        // Each thread accumulates part of the QK dot product for this candidate.
        float partial = 0.0F;
        // Skip invalid candidates so padded or future rows produce -inf scores.
        if (validKv)
        {
            // quantQ is FP8 [numTokens, 8, 576]. It was produced by mla_rope_generation, not by this kernel.
            // qPtr points to the full 576-dim FP8 query vector for this token/head.
            __nv_fp8_e4m3 const* qPtr = quantQ + (tokenIdx * kLocalHeads + headIdx) * kHeadDim;
            // kvCachePool is FP8 [poolTokens, 1, 576] viewed as [poolTokens, 576].
            // Use 64-bit offset math: real bench pool rows can exceed 7M, and kvIdx * 576 overflows int32.
            int64_t const kvOffset = static_cast<int64_t>(kvIdx) * kHeadDim;
            // kvPtr points to the full 576-dim FP8 key/value row selected by DSA top-k.
            __nv_fp8_e4m3 const* kvPtr = kvCachePool + kvOffset;
            // Stride the CTA lanes over the full QK dimension [latent 512 + RoPE 64].
            for (int dim = tid; dim < kHeadDim; dim += blockDim.x)
            {
                // Accumulate q_fp8[dim] * k_fp8[dim] in FP32. Dequant scale is folded into scoreScaleLog2.
                partial += toFloat(qPtr[dim]) * toFloat(kvPtr[dim]);
            }
        }
        // Reduce this candidate's QK dot product across all lanes in the CTA.
        float const dot = blockReduceSum(partial, reduce);
        // One lane stores the candidate score and row maximum.
        if (tid == 0)
        {
            // Invalid candidates become -inf so their softmax contribution is exactly zero.
            float const score = validKv ? dot * scoreScaleLog2 : -INFINITY;
            // Store log2-space score in shared memory.
            weights[topkIdx] = score;
            // Track max score for stable exp2 softmax.
            maxScore = fmaxf(maxScore, score);
        }
        // Ensure the stored score and reduction scratch are stable before the next candidate.
        __syncthreads();
    }

    // Shared denominator written by lane 0 and read by all lanes during V accumulation.
    __shared__ float denomShared;
    // Lane 0 converts log2 scores to unnormalized probabilities.
    if (tid == 0)
    {
        // Sum of exp2(score - maxScore) over all valid sparse candidates.
        float denom = 0.0F;
        // Avoid NaN from -inf - -inf if every candidate is invalid.
        if (maxScore != -INFINITY)
        {
            // Convert every candidate score to an unnormalized softmax weight.
            for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
            {
                // Subtracting the max keeps the largest exponent at exp2(0) == 1.
                float const weight = exp2f(weights[topkIdx] - maxScore);
                // Reuse the shared score slot as an unnormalized softmax weight.
                weights[topkIdx] = weight;
                // Accumulate denominator for later normalization.
                denom += weight;
            }
        }
        else
        {
            // No valid candidate means zero output.
            for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
            {
                // Clear all weights to avoid consuming stale shared memory values.
                weights[topkIdx] = 0.0F;
            }
        }
        // Publish the denominator to all lanes.
        denomShared = denom;
    }
    // Wait for all weights and denomShared to be ready.
    __syncthreads();

    // Precompute reciprocal denominator. If denom is zero, all normalized weights are treated as zero.
    float const invDenom = denomShared > 0.0F ? 1.0F / denomShared : 0.0F;
    // Second pass computes output latent V dimensions [0, 512).
    for (int dim = tid; dim < kKvLoraRank; dim += blockDim.x)
    {
        // Accumulator for output[tokenIdx, headIdx, dim].
        float acc = 0.0F;
        // Sum over sparse candidates using the already-computed softmax weights.
        for (int topkIdx = 0; topkIdx < topK; ++topkIdx)
        {
            // Reload the global pool row for data loads.
            int const kvIdx = topkIndicesPool[tokenIdx * topK + topkIdx];
            // Reload local sequence position for the same visibility check used in the QK pass.
            int const localKvIdx = topkIndicesLocal[tokenIdx * topK + topkIdx];
            // Offset inside the current decode/MTP group.
            int const currentGroupOffset = localKvIdx - currentGroupStart;
            // Historical rows are before the current group.
            bool const historicalKv = localKvIdx >= 0 && localKvIdx < currentGroupStart;
            // Current-group rows are the just-appended decode/MTP rows.
            bool const currentGroupKv = currentGroupOffset >= 0 && currentGroupOffset < numTokens;
            // Visibility flag for current-group rows.
            bool currentGroupValid = false;
            // Current-group rows require packed-mask or causal filtering.
            if (currentGroupKv)
            {
                // Prefer explicit spec/MTP packed mask when it is present.
                if (specDecodingPackedMask != nullptr && specMaskWords > 0)
                {
                    // Select mask word for this current-group offset.
                    int const wordIdx = currentGroupOffset / 32;
                    // Select mask bit inside the word.
                    int const bitIdx = currentGroupOffset % 32;
                    // Load the mask word for this query token row.
                    uint32_t const packed
                        = static_cast<uint32_t>(specDecodingPackedMask[tokenIdx * specMaskWords + wordIdx]);
                    // A set bit allows this current-group KV row to contribute.
                    currentGroupValid = ((packed >> bitIdx) & 1U) != 0U;
                }
                else
                {
                    // Fallback causal rule: token t may attend to current-group rows <= t.
                    currentGroupValid = currentGroupOffset <= tokenIdx;
                }
            }
            // Use exactly the same validity rule as the QK pass.
            bool const validKv = kvIdx >= 0 && (historicalKv || (currentGroupKv && currentGroupValid));
            // Only valid candidates contribute probability times value.
            if (validKv)
            {
                // Compute the flattened [pool row, dim] address with 64-bit arithmetic to avoid overflow.
                int64_t const kvOffset = static_cast<int64_t>(kvIdx) * kHeadDim + dim;
                // Accumulate softmax(topkIdx) * V[dim] in FP32. V dequant scale is applied after the sum.
                acc += weights[topkIdx] * invDenom * toFloat(kvCachePool[kvOffset]);
            }
        }
        // Dequantize the FP8 V sum to original units and round to BF16 output.
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
// - Mutates kv_cache by appending/writing the context latent KV rows with rotated K RoPE.
//
// Output:
// - Returns BF16 [numTokens, 8 * 512], logically [numTokens, 8, 512]. Each row is the sparse
//   softmax-weighted sum of FP8 latent V rows from the temporary current-context KV tensor, in original/BF16 units.
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
    // Allocate compact FP8 current-context KV [numTokens, 576] used by contextAttentionKernel.
    auto contextKvFp8 = torch::empty({numTokens, kHeadDim}, kv_cache_pool.options());

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
    // Raw FP8 pointer for compact current-context KV scratch [numTokens, 576].
    auto* contextKvFp8Ptr = reinterpret_cast<__nv_fp8_e4m3*>(contextKvFp8.data_ptr());
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
        // FP8 KV-cache mode writes quantized rows to kv_cache and contextKvFp8.
        preprocessContextKernel<true><<<preprocessGrid, kThreads, 0, stream>>>(fusedQPtr, latentCachePtr, qPePtr,
            rotaryCosSinPtr, ctx_cached_token_indptr.data_ptr<int64_t>(), kv_cache, contextKvFp8Ptr,
            kvScaleOrigQuantPtr, qPeStrideToken, qPeStrideHead, numTokens);
    }
    else
    {
        // BF16 KV-cache mode writes BF16 rows to kv_cache but still creates FP8 contextKvFp8 for attention.
        preprocessContextKernel<false><<<preprocessGrid, kThreads, 0, stream>>>(fusedQPtr, latentCachePtr, qPePtr,
            rotaryCosSinPtr, ctx_cached_token_indptr.data_ptr<int64_t>(), kv_cache, contextKvFp8Ptr,
            kvScaleOrigQuantPtr, qPeStrideToken, qPeStrideHead, numTokens);
    }

    // Context score scale follows the MLA absorbed-Q reference: 1 / (q_scaling * sqrt(qk_head_dim)).
    // qk_head_dim is 256 for GLM-5 because absorbed q_nope is 192 and RoPE is 64.
    float const hostScoreScale = 1.0F / (static_cast<float>(q_scaling) * sqrtf(256.0F));
    // Launch one CTA per [token, head] output row.
    dim3 attentionGrid(numTokens, kLocalHeads);
    // Shared memory stores kThreads reduction floats plus topK score/weight floats.
    size_t const sharedBytes = (kThreads + std::min(topK, kMaxTopK)) * sizeof(float);
    // Compute sparse context attention from the rotated/quantized current-context KV scratch.
    contextAttentionKernel<<<attentionGrid, kThreads, sharedBytes, stream>>>(fusedQPtr, contextKvFp8Ptr,
        topk_indices_local.data_ptr<int32_t>(), outputPtr, kvScaleOrigQuantPtr, kvScaleQuantOrigPtr, numTokens, topK,
        hostScoreScale);

    // Surface any asynchronous launch/runtime errors before returning the tensor to Python.
    sync_check_cuda_error(stream);
    // Return BF16 [numTokens, 8 * 512] latent attention output.
    return output;
}

// Fused GLM-5/DeepSeek-V3 sparse MLA generation/decode path.
//
// Inputs:
// - fused_q: BF16 [numTokens, 8, 576]. Used here for shape/device/output options; Q data is read from
//   quant_q_buffer because mla_rope_generation already rotated and FP8-quantized Q.
// - topk_indices_pool: INT32 [numTokens, topK], global KV-cache pool rows.
// - topk_indices: INT32 [numTokens, topK], local sequence positions used for MTP visibility checks.
// - kv_cache_pool: FP8 E4M3 [poolTokens, 1, 576], flattened by the kernel as [poolTokens, 576].
// - sequence_length: INT32 [1], total KV length after appending this decode/MTP group.
// - quant_q_buffer: UINT8-backed FP8 E4M3 [numTokens, 8, 576].
// - mla_bmm1_scale: FP32 [2], with index 1 holding log2-space QK score scale.
// - mla_bmm2_scale: FP32 [1], FP8 V dequantization scale.
// - spec_decoding_packed_mask: optional INT32 [maxRequests, numTokens, packedWords].
//
// Output:
// - Returns BF16 [numTokens, 8 * 512], logically [numTokens, 8, 512].
//
// Side effects:
// - Writes only the returned output tensor. The paged KV cache was already updated by mla_rope_generation.
torch::Tensor dsv3_fused_mla_generation_cuda(torch::Tensor fused_q, torch::Tensor topk_indices_pool,
    torch::Tensor topk_indices, torch::Tensor kv_cache_pool, torch::Tensor sequence_length,
    torch::Tensor quant_q_buffer, torch::Tensor mla_bmm1_scale, torch::Tensor mla_bmm2_scale,
    std::optional<torch::Tensor> spec_decoding_packed_mask)
{
    // fused_q is BF16 [numTokens, 8, 576]. The generation attention kernel does not read it directly, but its
    // shape/device/dtype define the output allocation and are checked in the THOP wrapper.
    // Set active CUDA device from fused_q.
    c10::cuda::CUDAGuard const deviceGuard{fused_q.device()};
    // Use PyTorch's current stream for stream-ordering and CUDA graph capture compatibility.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(fused_q.get_device());

    // Number of decode/MTP query rows in this call.
    int const numTokens = static_cast<int>(fused_q.size(0));
    // Sparse top-k width per query row.
    int const topK = static_cast<int>(topk_indices_pool.size(1));
    // Dynamic shared memory uses topK floats for weights, so enforce the development-kernel bound.
    TORCH_CHECK(topK <= kMaxTopK, "dsv3_fused_mla_generation supports topK <= ", kMaxTopK, ", got ", topK);
    // Allocate BF16 output [numTokens, 8 * 512] with the same device/dtype options as fused_q.
    auto output = torch::empty({numTokens, kLocalHeads * kKvLoraRank}, fused_q.options());

    // quant_q_buffer is a uint8 tensor whose bytes are FP8 E4M3 values.
    auto* quantQPtr = reinterpret_cast<__nv_fp8_e4m3 const*>(quant_q_buffer.data_ptr());
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

    // Launch one CTA per [decode token, local head] output row.
    dim3 attentionGrid(numTokens, kLocalHeads);
    // Shared memory stores kThreads reduction floats plus topK score/weight floats.
    size_t const sharedBytes = (kThreads + topK) * sizeof(float);
    // Compute sparse decode attention using FP8 Q, FP8 paged KV, and DSA top-k indices.
    generationAttentionKernel<<<attentionGrid, kThreads, sharedBytes, stream>>>(quantQPtr, kvCachePoolPtr,
        topk_indices_pool.data_ptr<int32_t>(), topk_indices.data_ptr<int32_t>(), sequence_length.data_ptr<int32_t>(),
        specDecodingPackedMaskPtr, specMaskWords, outputPtr, mla_bmm1_scale.data_ptr<float>(),
        mla_bmm2_scale.data_ptr<float>(), numTokens, topK);

    // Report any asynchronous CUDA error before returning to Python.
    sync_check_cuda_error(stream);
    // Return BF16 [numTokens, 8 * 512] latent attention output.
    return output;
}

} // namespace dsv3_fused_mla
