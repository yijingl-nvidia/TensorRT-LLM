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

// Step-5z mega kernel - v68 = v65 + kKTile=768 + per-K-block (no-rescale) scales.
//
// Numerical-fix variant (in-place edit, same kernel name "v68_integrated"):
// previously a single per-K-iter scale ("group max") was applied at runtime
// and a compensating fp8-rescale was folded into packed weights - that
// rescale re-quantized every fp8 value through a sub-unity ratio, leaking
// ~5 bits per weight. The current path consumes the raw model weights and
// per-128-col K-block scales directly. No pre-fold rescale and no offline
// packed weight allocation are required.
//
// Activation-quant fix (also in-place edit, kernel name unchanged):
// the per-tensor activation FP8 quant (single amax + scalar fold) is
// replaced by TRTLLM's per-128-col activation quant (matching
// fp8_quantize_1x128). The kernel now produces 48 fp32 per-K-block
// activation dequant scales internally, computed during Phase 1
// overlapped with the noaux_tc top-K reduction using the 10 warps that
// are otherwise idle during top-K (warps kNumExpertWarps..kNumWarps-1).
// Each idle warp covers a stride-10 share of the 48 K-blocks (5 each).
// The MMA fold becomes: c[i] += c_kb_frag[i] * (S_w[kb] * S_act[kb_global])
// where S_act[kb_global] = amax_kb / 448 is the per-128-col dequant scale.
// External API is UNCHANGED - activation quant remains internal to the
// kernel; we still consume bf16 hidden_in directly.
//
// ============================================================================
// DESIGN INTENT (v68) - SATURATION CONFIRMATION
// ============================================================================
//
// K-axis sweep results so far (M=4):
//   kKTile=128 (v34): 26.66 us
//   kKTile=256 (v51): 24.60 us  (-7.7% vs 128)
//   kKTile=384 (v59): 18.49 us  (-18.0% vs 256, LB=1)
//   kKTile=512 (v65): 18.12 us  (-2.4% vs 384, LB=1)  - MATCHES TileRT 18.11
//
// K-axis appears saturated by v65. v68 tests kKTile=768 to CONFIRM saturation:
// likely outcome ~= v65 (wash) or small regression (register/inner-loop cliff).
// This is a deliberate diminishing-returns probe; ships only if it wins.
//
// kKTile=768 (NEW):
//   * K-iters: 6144 / 768 = 8 (vs v65's 12, -33%)
//   * Per-slab smem: 48 KiB x 2 sides x 1 raw-load tile = 96 KiB weight smem
//     + activation/top-k scratch. Legacy TMA double-buffering is no longer used.
//   * Per-K-iter HMMA work: 24 m16n8k32 fragments (was 16 in v65)
//   * BSSY/BSYNC/WARPSYNC drop another ~33%
//
// kKTile must be multiple of 32 (HMMA.16816 k-dim = 32 fp8 bytes). 768/32 = 24.
// 6144 / 768 = 8 (integer divisor, OK). 768 / 128 = 6 sub-slabs per K-iter.
//
// kStages is 1 in the raw-layout path because each K tile is synchronously
// staged from the model weights before the MMA workers consume it.
//
// __launch_bounds__(384, V68_LB_BLOCKS_PER_SM) -- v65 found LB=1 wins. v68's
// 24-HMMA inner loop likely needs ~140-160 regs, still within LB=1's 170-reg
// cap (slim). Default LB=1.
//
// Per-K-block scale handling: kKTile=768 means each K-iter spans 6 underlying
// 128-col K-blocks (kWeightScaleKBlocksPerKIter = 48/8 = 6). The mega kernel
// loads the 6 matching fp32 scales per K-iter from the original scale tensors
// and folds each 128-col K-block's MMA accumulator into the per-K-iter sum
// using its own scale; no pre-fold rescale is required.
//
// ============================================================================
// LAYOUT (v68: 6 stacked sub-slabs of 128-col-wide K-major lane-contiguous).
//
// The kernel builds this legacy MMA-consumer slab directly in shared memory
// from the existing row-major model weights. It still uses 6 stacked
// 8192-byte "K-sixth" sub-slabs, each covering 128 K-cols, because
// compute_mma_kiter_v68 consumes that layout.
//
//   Slab byte_off [k_sixth, m_tile, k_sub_in_sixth, lane, b]:   // 49152 total
//     k_sixth in [0..5], m_tile in [0..3], k_sub_in_sixth in [0..3],
//     lane in [0..31], b in [0..15]
//   byte_off = k_sixth * 8192 + m_tile * 2048 + k_sub_in_sixth * 512 + lane * 16 + b
//
//   Within each k_sixth (mirror of v34):
//     b in [0..3]   : (row = m_tile*16 + (lane>>2),     col = k_sub*32 + 4*(lane&3) + 0..3)
//     b in [4..7]   : (row = m_tile*16 + (lane>>2) + 8, col = k_sub*32 + 4*(lane&3) + 0..3)
//     b in [8..11]  : (row = m_tile*16 + (lane>>2),     col = k_sub*32 + 4*(lane&3) + 16..19)
//     b in [12..15] : (row = m_tile*16 + (lane>>2) + 8, col = k_sub*32 + 4*(lane&3) + 16..19)
//   where k_sub = k_sixth * 4 + k_sub_in_sixth.
//
// Raw input layouts:
//   shared_gate_up_weight [2 * inter_per_tp, hidden] fp8: gate, then up.
//   shared_gate_up_scale  [2 * inter_per_tp / 128, 48] fp32: gate, then up.
//   routed_w3_w1_weight   [256, 2 * inter_per_tp, hidden] fp8: up, then gate.
//   routed_w3_w1_scale    [256, 2 * inter_per_tp / 128, 48] fp32: up, then gate.
//
// Variable names `k_third` / `kKThirdsPerIter` / `kKSubsPerThird` are kept
// (vs renaming to "sixth") for minimal-diff vs v65 - the constants below now
// express 6 sub-slabs and 4 k_subs per sub-slab.
//
// ============================================================================

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <string>
#include <torch/extension.h>
#include <tuple>
#include <utility>

#include "topk_reduce.cuh"

namespace cg = cooperative_groups;

namespace
{

// ---- Constants ----
// Model-architecture constants (unchanged across TP):
constexpr int kNumExperts = 256;
constexpr int kSharedExpert = 256;
constexpr int kTopK = 8;
// kSlotsPerToken is GLM-5-architecture-locked: 1 shared expert + 8 routed top-k.
// This does NOT depend on TP, despite the legacy name suggesting otherwise.
constexpr int kSlotsPerToken = kSharedExpert / kNumExperts + kTopK; // 1 + 8 = 9
constexpr int kThreadsPerCta = 384;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kThreadsPerCta / kWarpSize;
constexpr int kMaxNumExpertsUnit = 128;
constexpr int kNumExpertWarps = (kNumExperts - 1) / kMaxNumExpertsUnit + 1;
constexpr int kMaxNumTopGroups = 4;
constexpr int kNumInterTopK = kNumExpertWarps * kTopK;

constexpr int kHidden = 6144;
constexpr int kCtaOutRows = 64;

// TP-dependent constants. The kernel is templated on kInterPerTpParam; the
// host-side dispatcher selects {256, 512} based on input tensor shape.
//   TP=8: kInterPerTp = 2048/8 = 256 -> kSubRowsPerExpert=4, kCtasPerToken=36
//   TP=4: kInterPerTp = 2048/4 = 512 -> kSubRowsPerExpert=8, kCtasPerToken=72
constexpr int kInterPerTp_TP8 = 256;
constexpr int kInterPerTp_TP4 = 512;

// Helper constexpr functions (used by both host and __global__ template kernels).
__host__ __device__ constexpr int sub_rows_per_expert(int kInterPerTp)
{
    return kInterPerTp / kCtaOutRows;
}

__host__ __device__ constexpr int ctas_per_token(int kInterPerTp)
{
    return kSlotsPerToken * sub_rows_per_expert(kInterPerTp);
}

__host__ __device__ constexpr int weight_scale_m_blocks(int kInterPerTp)
{
    return kInterPerTp / 128;
}

// Invariant: kSlotsPerToken = 1 + kTopK assumes exactly one shared expert.
// The expert_slot==0 branch later (`my_expert = kSharedExpert`) hard-codes
// this; if a future model has 2+ shared experts, that branch and the
// kSlotsPerToken formula above need to be revisited together.
static_assert(kSharedExpert == kNumExperts,
    "kSlotsPerToken derivation assumes exactly one shared expert "
    "(kSharedExpert == kNumExperts).");

// Invariant: each FP8 weight scale m-block covers 128 rows; each CTA covers
// kCtaOutRows=64 rows; so exactly 2 CTAs share one scale m-block. The raw
// scale lookup computes m_block_idx = sr / (kSubRowsPerExpert /
// kWeightScaleMBlocks) and relies on that ratio being 2 at every supported TP.
static_assert(sub_rows_per_expert(kInterPerTp_TP8) / weight_scale_m_blocks(kInterPerTp_TP8) == 2,
    "TP=8: kCtaOutRows must be half the 128-row FP8 scale m-block "
    "(2 CTAs per scale m-block).");
static_assert(sub_rows_per_expert(kInterPerTp_TP4) / weight_scale_m_blocks(kInterPerTp_TP4) == 2,
    "TP=4: kCtaOutRows must be half the 128-row FP8 scale m-block "
    "(2 CTAs per scale m-block).");

// K-axis constants are functions of kHidden ONLY (NOT TP) - unchanged at TP=4.
// v68 delta: kKTile 512 -> 768.
constexpr int kKTile = 768;
constexpr int kNumKIter = kHidden / kKTile; // 8
constexpr int kKSubsPerIter = kKTile / 32;  // 24

// Legacy v68 had "kNumKGroups = kNumKIter = 8, kKBlocksPerGroup = 1" - the
// per-K-block-scaling refactor removed the K-iter->K-group indirection
// entirely (each K-iter directly carries its 6 K-block scales).

// w_scale tensor original shape: [E, kWeightScaleMBlocks, 48]. kWeightScaleMBlocks
// depends on TP (kInterPerTp / 128). kWeightScaleKBlocks is a function of kHidden only.
constexpr int kWeightScaleKBlocks = 48; // original 128-col K-blocks (kHidden / 128)
constexpr int kWeightScaleKBlocksPerKIter = kWeightScaleKBlocks / kNumKIter; // 6

constexpr int kGateWorkerWarpBase = 4;
constexpr int kUpWorkerWarpBase = 8;
constexpr int kNumGateWorkers = 4;
constexpr int kNumUpWorkers = 4;

constexpr int kTileBytes = kCtaOutRows * kKTile; // 49152 (48 KiB)

constexpr int kStages = 1;

constexpr float kInvalidScore = -INFINITY;
constexpr float kFp8Max = 448.f;

constexpr int kActBytes = kHidden * 2;                               // 12288
constexpr int kActCpAsyncs = kActBytes / 16;                         // 768
constexpr int kActCpAsyncsPerThread = kActCpAsyncs / kThreadsPerCta; // 2

// K-major slab offset constants. v68 stacks 6 v34-style sub-slabs along Z
// (k_sixth axis) so each box load stays within the TMA SWIZZLE_NONE 256-byte
// inner-dim cap. Variable names kept (k_third / kKThirdsPerIter / kKSubsPerThird)
// for minimal-diff vs v65.
constexpr int kLaneBytes = 16;                                       // 16 B per lane
constexpr int kKsubBytes = kWarpSize * kLaneBytes;                   // 512 B per K-sub
constexpr int kKSubsPerThird = 4;                                    // 4 k_subs per k_sixth
constexpr int kKThirdsPerIter = 6;                                   // 768 / 128 = 6 k_sixths per K-iter
constexpr int kMtileSubslabBytes = kKSubsPerThird * kKsubBytes;      // 2048 B per m_tile within a k_sixth
constexpr int kSubslabBytes = kCtaOutRows / 16 * kMtileSubslabBytes; // 4 * 2048 = 8192 B per k_sixth

static __device__ inline float sigmoid_accurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// -------------------------------------------------------------------------
// `ld.shared.v2.b32` = LDS.64 (8 bytes / lane).
__device__ __forceinline__ void lds64_b32x2(uint32_t& r0, uint32_t& r1, __nv_fp8_e4m3 const* smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(const_cast<__nv_fp8_e4m3*>(smem_ptr));
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// `ld.shared.v4.b32` = LDS.128 (16 bytes / lane).
__device__ __forceinline__ void lds128_b32x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, __nv_fp8_e4m3 const* smem_ptr)
{
    uint32_t addr = __cvta_generic_to_shared(const_cast<__nv_fp8_e4m3*>(smem_ptr));
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

// -------------------------------------------------------------------------
// v68 consumer addressing: 6 stacked v34-style sub-slabs (k_sixth dim outer,
// then m_tile, then k_sub_in_sixth, then lane). SWIZZLE_NONE.
// Variable names `k_third` / `kKSubsPerThird` kept (vs renaming to sixth) for
// minimal-diff vs v65. The constants now express 6 sub-slabs of 128 K-cols each.
// -------------------------------------------------------------------------
__device__ __forceinline__ int v68_lane_offset(int m_tile, int k_sub, int lane)
{
    int const k_third = k_sub / kKSubsPerThird;        // 0..5
    int const k_sub_in_third = k_sub % kKSubsPerThird; // 0..3
    return k_third * kSubslabBytes + m_tile * kMtileSubslabBytes + k_sub_in_third * kKsubBytes + lane * kLaneBytes;
}

// Compute MMA for ONE K-iter (kKTile = 768 cols => 24 m16n8k32 MMAs per fragment).
//
// Per-K-block-scaled variant: the 24 K-subs split into 6 K-blocks of 4 K-subs
// each (one per 128-col fp8-scale m-block). Each block's accumulator is
// scaled by `per_block_scale[kb] * act_block_scale[kb]` and reduced into
// `d_out[]`. The per-K-block activation scale (dequant) is supplied per
// K-block (6 fp32 per K-iter) - there is NO global act_scale_val anymore
// because activation quant is now per-128-col-K-block (TRTLLM 1x128 scheme).
//
__device__ __forceinline__ void compute_mma_kiter_v68(__nv_fp8_e4m3 const* __restrict__ smem_tile,
    __nv_fp8_e4m3 const* __restrict__ smem_act_fp8, int k_iter, int my_m, int lane,
    float const (&per_block_scale)[kWeightScaleKBlocksPerKIter],
    float const (&act_block_scale)[kWeightScaleKBlocksPerKIter], float (&d_out)[4])
{
#pragma unroll
    for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
    {
        float c_frag[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int ks_in_kb = 0; ks_in_kb < kKSubsPerThird; ++ks_in_kb)
        {
            int const k_sub = kb * kKSubsPerThird + ks_in_kb;
            // Per-lane base - natural K-major layout, 16 contiguous bytes/lane.
            __nv_fp8_e4m3 const* lane_base = smem_tile + v68_lane_offset(my_m, k_sub, lane);

            // 2 x LDS.64 fetch the 4 b32 A-frag chunks. ptxas typically merges
            // these into a single LDS.128 in the SASS.
            uint32_t a_frag[4];
            lds64_b32x2(a_frag[0], a_frag[1], lane_base);     // chunks 0+1
            lds64_b32x2(a_frag[2], a_frag[3], lane_base + 8); // chunks 2+3

            // B-fragment (activations). Lanes 0..3 hold the K-contiguous 16
            // bytes/half-K-sub; other lanes hold zeros. Unchanged from v34.
            uint32_t b_frag[2];
            if (lane < 4)
            {
                int const k_base = k_iter * kKTile + k_sub * 32 + (lane & 3) * 4;
                uint32_t b_lo_pair[2];
                uint32_t b_hi_pair[2];
                lds64_b32x2(b_lo_pair[0], b_lo_pair[1], smem_act_fp8 + k_base);
                lds64_b32x2(b_hi_pair[0], b_hi_pair[1], smem_act_fp8 + k_base + 16);
                b_frag[0] = b_lo_pair[0];
                b_frag[1] = b_hi_pair[0];
            }
            else
            {
                b_frag[0] = 0;
                b_frag[1] = 0;
            }

            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(c_frag[0]), "+f"(c_frag[1]), "+f"(c_frag[2]), "+f"(c_frag[3])
                : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
        }

        // Fold this 128-col K-block's accumulator into the per-K-iter sum.
        // Per-K-block activation scale + per-K-block weight scale - both
        // are dequant scales (mul to convert fp8 -> fp32 magnitude).
        float const fs_kb = per_block_scale[kb] * act_block_scale[kb];
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            d_out[i] += c_frag[i] * fs_kb;
        }
    }
}

template <int kInterPerTpParam, bool kGate>
__device__ __forceinline__ __nv_fp8_e4m3 const* raw_weight_ptr(__nv_fp8_e4m3 const* __restrict__ shared_gate_up_weight,
    __nv_fp8_e4m3 const* __restrict__ routed_w3_w1_weight, int expert, int row, int col)
{
    constexpr int kInterPerTp = kInterPerTpParam;
    if (expert == kSharedExpert)
    {
        // Shared expert is stored as one [gate, up] matrix.
        int const row_offset = (kGate ? 0 : kInterPerTp) + row;
        return shared_gate_up_weight + static_cast<int64_t>(row_offset) * kHidden + col;
    }

    // Routed experts are stored as one [up, gate] matrix per expert.
    int const row_offset = (kGate ? kInterPerTp : 0) + row;
    return routed_w3_w1_weight + (static_cast<int64_t>(expert) * (2 * kInterPerTp) + row_offset) * kHidden + col;
}

template <int kInterPerTpParam, bool kGate>
__device__ __forceinline__ float const* raw_scale_ptr(float const* __restrict__ shared_gate_up_scale,
    float const* __restrict__ routed_w3_w1_scale, int expert, int sub_row, int k_iter)
{
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTpParam);
    constexpr int kWeightScaleMBlocks = weight_scale_m_blocks(kInterPerTpParam);
    int const m_block_idx = sub_row / (kSubRowsPerExpert / kWeightScaleMBlocks);
    if (expert == kSharedExpert)
    {
        // Shared expert scales follow [gate, up] order.
        int const m_offset = (kGate ? 0 : kWeightScaleMBlocks) + m_block_idx;
        return shared_gate_up_scale + static_cast<int64_t>(m_offset) * kWeightScaleKBlocks
            + k_iter * kWeightScaleKBlocksPerKIter;
    }

    // Routed expert scales follow [up, gate] order.
    int const m_offset = (kGate ? kWeightScaleMBlocks : 0) + m_block_idx;
    return routed_w3_w1_scale
        + (static_cast<int64_t>(expert) * (2 * kWeightScaleMBlocks) + m_offset) * kWeightScaleKBlocks
        + k_iter * kWeightScaleKBlocksPerKIter;
}

template <int kInterPerTpParam, bool kGate>
__device__ __forceinline__ void load_raw_weight_tile_v68(__nv_fp8_e4m3* __restrict__ smem_tile,
    __nv_fp8_e4m3 const* __restrict__ shared_gate_up_weight, __nv_fp8_e4m3 const* __restrict__ routed_w3_w1_weight,
    int expert, int sub_row, int k_iter, int tidx)
{
    // Fill the same 48 KiB K-major slab layout that the legacy packed path
    // produced, but source directly from the existing row-major
    // model weights. 3072 lane records * 16 bytes = 49152 bytes.
#pragma unroll 1
    for (int tri = tidx; tri < 3072; tri += kThreadsPerCta)
    {
        int const m_tile = tri / (kKSubsPerIter * kWarpSize);
        int const tri_in_mtile = tri % (kKSubsPerIter * kWarpSize);
        int const k_sub = tri_in_mtile / kWarpSize;
        int const lane = tri_in_mtile & 31;

        int const k_third = k_sub / kKSubsPerThird;
        int const k_sub_in_third = k_sub % kKSubsPerThird;

        int const row_lo = m_tile * 16 + (lane >> 2);
        int const row_hi = row_lo + 8;
        int const col_lo_in_block = k_sub_in_third * 32 + ((lane & 3) << 2);
        int const col_hi_in_block = col_lo_in_block + 16;

        int const gm_lo = sub_row * kCtaOutRows + row_lo;
        int const gm_hi = sub_row * kCtaOutRows + row_hi;
        int const gk_lo = k_iter * kKTile + k_third * 128 + col_lo_in_block;
        int const gk_hi = k_iter * kKTile + k_third * 128 + col_hi_in_block;

        uint32_t const a = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, kGate>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm_lo, gk_lo));
        uint32_t const b = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, kGate>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm_hi, gk_lo));
        uint32_t const c = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, kGate>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm_lo, gk_hi));
        uint32_t const d = *reinterpret_cast<uint32_t const*>(
            raw_weight_ptr<kInterPerTpParam, kGate>(shared_gate_up_weight, routed_w3_w1_weight, expert, gm_hi, gk_hi));

        uint8_t* dst = reinterpret_cast<uint8_t*>(smem_tile) + k_third * kSubslabBytes + m_tile * kMtileSubslabBytes
            + k_sub_in_third * kKsubBytes + lane * kLaneBytes;
        *reinterpret_cast<uint32_t*>(dst) = a;
        *reinterpret_cast<uint32_t*>(dst + 4) = b;
        *reinterpret_cast<uint32_t*>(dst + 8) = c;
        *reinterpret_cast<uint32_t*>(dst + 12) = d;
    }
}

// -------------------------------------------------------------------------
// v68 kernel: v65 with kKTile=768 + __launch_bounds__(384, V68_LB_BLOCKS_PER_SM).
//
// Defaults to 1 (mirrors v65's winning LB hint at M=4). v68's 24-HMMA inner
// loop likely wants ~140-160 registers; LB=1 cap is floor(65536/(1*384)) = 170.
// Slim margin -- spill risk possible. Override via -DV68_LB_BLOCKS_PER_SM=N.
// -------------------------------------------------------------------------

#ifndef V68_LB_BLOCKS_PER_SM
#define V68_LB_BLOCKS_PER_SM 1
#endif

template <int kInterPerTpParam>
__global__ __launch_bounds__(384, V68_LB_BLOCKS_PER_SM) void mega_kernel_v68(float const* __restrict__ scores,
    __nv_bfloat16 const* __restrict__ hidden_in, __nv_bfloat16 const* __restrict__ bias,
    __nv_fp8_e4m3 const* __restrict__ shared_gate_up_weight, float const* __restrict__ shared_gate_up_scale,
    __nv_fp8_e4m3 const* __restrict__ routed_w3_w1_weight, float const* __restrict__ routed_w3_w1_scale,
    float* __restrict__ topk_weights, int32_t* __restrict__ topk_indices,
    // [iter5-fp16-hidden] hidden_out is fp16 (10-bit mantissa) - kept from
    // iter-2 (the bf16 revert of iter-4 caused a bench hang; iter-5 isolates
    // the fp64 renorm fix). v110's TMA descriptor matches FLOAT16.
    __half* __restrict__ hidden_out, int64_t num_tokens, float routed_scaling_factor)
{

    constexpr int kInterPerTp = kInterPerTpParam;
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTp);

    int const token = blockIdx.x;
    int const cta_y = blockIdx.y;
    int const expert_slot = cta_y / kSubRowsPerExpert;
    int const sub_row = cta_y % kSubRowsPerExpert;
    int const row_stripe_start = sub_row * kCtaOutRows;

    int const tidx = threadIdx.x;
    int const lane = tidx & (kWarpSize - 1);
    int const warp_idx = __shfl_sync(0xffffffff, tidx / kWarpSize, 0);

    extern __shared__ __align__(128) unsigned char smem_buf[];

    __nv_fp8_e4m3* const smem_gate_tiles = reinterpret_cast<__nv_fp8_e4m3*>(smem_buf);
    __nv_fp8_e4m3* const smem_up_tiles = smem_gate_tiles + kStages * kTileBytes;

    __nv_bfloat16* const smem_act_bf16 = reinterpret_cast<__nv_bfloat16*>(smem_up_tiles + kStages * kTileBytes);
    // In-kernel per-128-col act quant writes fp8 to a SEPARATE buffer
    // (not aliased over bf16) so multiple warps can read bf16 / write fp8
    // in parallel without aliasing races. Costs 6 KiB.
    __nv_fp8_e4m3* const smem_act_fp8 = reinterpret_cast<__nv_fp8_e4m3*>(smem_act_bf16 + kHidden);

    auto align_up_128 = [](uintptr_t p) -> uintptr_t { return (p + 127u) & ~uintptr_t(127); };
    uintptr_t rs_base = align_up_128(reinterpret_cast<uintptr_t>(smem_act_fp8 + kHidden));

    float* const smem_score_sigmoid = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kNumExperts;
    rs_base = align_up_128(rs_base);

    float* const smem_score_bias = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kNumExperts;
    rs_base = align_up_128(rs_base);

    float* const smem_inter_scores = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kNumInterTopK;
    rs_base = align_up_128(rs_base);

    int32_t* const smem_inter_experts = reinterpret_cast<int32_t*>(rs_base);
    rs_base += sizeof(int32_t) * kNumInterTopK;
    rs_base = align_up_128(rs_base);

    int32_t* const smem_topk_i = reinterpret_cast<int32_t*>(rs_base);
    rs_base += sizeof(int32_t) * kTopK;
    rs_base = align_up_128(rs_base);

    // Per-128-col activation dequant scales (TRTLLM 1x128 quant scheme).
    // 48 fp32 = one scale per 128-col K-block over kHidden=6144.
    // Computed in-kernel during Phase 1 (overlapped with top-K) and consumed
    // in the K-loop fold (replacing the old single per-tensor scalar).
    float* const smem_act_block_scales = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kWeightScaleKBlocks;
    rs_base = align_up_128(rs_base);

    float* const smem_gate_acc = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kCtaOutRows;
    rs_base = align_up_128(rs_base);

    float* const smem_up_acc = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kCtaOutRows;
    rs_base = align_up_128(rs_base);

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<kWarpSize>(block);

    // ===== Phase 0: activation cp.async prefetch =====
    {
        __nv_bfloat16 const* x_ptr = hidden_in + static_cast<int64_t>(token) * kHidden;
#pragma unroll
        for (int ii = 0; ii < kActCpAsyncsPerThread; ++ii)
        {
            int const byte_off = (ii * kThreadsPerCta + tidx) * 16;
            __nv_bfloat16 const* src
                = reinterpret_cast<__nv_bfloat16 const*>(reinterpret_cast<char const*>(x_ptr) + byte_off);
            __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(reinterpret_cast<char*>(smem_act_bf16) + byte_off);
            unsigned const dst_smem = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem), "l"(src));
        }
        asm volatile("cp.async.commit_group;\n" :::);
    }
    // Wait for activation cp.async completion BEFORE we let any warp read
    // smem_act_bf16. We move the wait+sync to the top so the per-128-col
    // quant (running in parallel with top-K) sees the activation immediately.
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    // ===== Phase 1: noaux_tc score prep - every thread =====
    // (Independent of activation; could be done in parallel with cp.async,
    // but the sync at the top is needed anyway for activation visibility.)
    {
        int const expert = tidx;
        bool const valid = expert < kNumExperts;
        float const bias_val = valid ? __bfloat162float(bias[expert]) : kInvalidScore;
        float const score = valid ? scores[static_cast<int64_t>(token) * kNumExperts + expert] : kInvalidScore;
        float const score_sigmoid = sigmoid_accurate(score);
        if (valid)
        {
            smem_score_sigmoid[expert] = score_sigmoid;
            smem_score_bias[expert] = score_sigmoid + bias_val;
        }
    }
    __syncthreads();

    // ===== Phase 1+2 FUSED: top-K (warps 0..kNumExpertWarps-1)
    //                       || per-128-col activation FP8 quant (warps kNumExpertWarps..kNumWarps-1)
    //
    // Activation quant uses the warps that are otherwise idle while warps
    // 0..kNumExpertWarps-1 run the noaux_tc top-K reduction. 48 K-blocks of
    // 128 bf16 elements each are partitioned round-robin across the 10
    // (= kNumWarps - kNumExpertWarps) quant warps. Each warp performs
    // amax -> quant_scale -> fp8 cast (writing to a separate, non-aliased
    // fp8 smem buffer), then stores the dequant scale into
    // smem_act_block_scales[kb_global] for the K-loop to consume per block.
    //
    // Layout:
    //   smem_act_bf16[0..6143]  - source (12 KiB bf16)
    //   smem_act_fp8 [0..6143]  - destination (6 KiB fp8, SEPARATE buffer)
    //
    // We deliberately use a separate fp8 buffer (rather than reusing the
    // bf16 buffer's lower half) so multiple quant warps can read bf16 and
    // write fp8 concurrently without cross-warp byte-level aliasing races.
    // The 6 KiB extra smem is acceptable (we have ~21 KiB of slack at LB=1).
    // -------------------------------------------------------------------
    float top_scores[kTopK];
    int32_t top_experts[kTopK];

    // Quant constants: 48 K-blocks split round-robin across (kNumWarps -
    // kNumExpertWarps) = 10 warps. Each quant warp does kQuantRounds = 5
    // strided blocks (covers 50 >= 48 with bounds check).
    constexpr int kQuantWarps = kNumWarps - kNumExpertWarps;                            // 10
    constexpr int kQuantRounds = (kWeightScaleKBlocks + kQuantWarps - 1) / kQuantWarps; // 5
    constexpr int kElemsPerLane128Block = 128 / kWarpSize;                              // 4

    if (warp_idx < kNumExpertWarps)
    {
        // ----- Top-K stage 1 -----
        int const offset = warp_idx * kWarpSize * kMaxNumTopGroups;
        float in_value[kMaxNumTopGroups];
        int32_t in_idx[kMaxNumTopGroups];
#pragma unroll
        for (int ii = 0; ii < kMaxNumTopGroups; ++ii)
        {
            int const e = ii * kWarpSize + lane;
            in_idx[ii] = offset + e;
            in_value[ii] = (offset + e) < kNumExperts ? smem_score_bias[offset + e] : kInvalidScore;
        }
        mega_topk::reduceTopK<kTopK, float, kMaxNumTopGroups>(
            warp, top_scores, top_experts, in_value, in_idx, kInvalidScore, kTopK);
        if (lane < kTopK)
        {
            smem_inter_scores[warp_idx * kTopK + lane] = top_scores[lane];
            smem_inter_experts[warp_idx * kTopK + lane] = top_experts[lane];
        }
    }
    else
    {
        // ----- Per-128-col activation FP8 quant (warps kNumExpertWarps..) -----
        // Each "quant warp" processes kQuantRounds blocks stride-kQuantWarps.
        int const q_warp = warp_idx - kNumExpertWarps; // 0..9

#pragma unroll
        for (int r = 0; r < kQuantRounds; ++r)
        {
            int const kb_global = q_warp + r * kQuantWarps;
            if (kb_global >= kWeightScaleKBlocks)
                break;

            int const block_off = kb_global * 128; // bf16 element offset

            // Each lane reads 4 contiguous bf16 elements (8 B vector load).
            __nv_bfloat16 const* lane_src = smem_act_bf16 + block_off + lane * kElemsPerLane128Block;
            __nv_bfloat162 v01;
            __nv_bfloat162 v23;
            v01 = *reinterpret_cast<__nv_bfloat162 const*>(lane_src);
            v23 = *reinterpret_cast<__nv_bfloat162 const*>(lane_src + 2);
            float const f0 = __bfloat162float(__low2bfloat16(v01));
            float const f1 = __bfloat162float(__high2bfloat16(v01));
            float const f2 = __bfloat162float(__low2bfloat16(v23));
            float const f3 = __bfloat162float(__high2bfloat16(v23));

            float lane_max = fmaxf(fmaxf(fabsf(f0), fabsf(f1)), fmaxf(fabsf(f2), fabsf(f3)));
            float amax = cg::reduce(warp, lane_max, cg::greater<float>{});
            amax = fmaxf(amax, 1e-10f);
            float const quant_scale = kFp8Max / amax; // bf16 * quant_scale -> fp8
            // [v68-precision] Compute dequant as 1/quant_scale rather than
            // amax/kFp8Max so the (quant_scale * dequant_scale) chain
            // simplifies to EXACTLY 1.0 in fp32 (IEEE 754 x*(1/x)==1 for
            // most finite x; the two-division form `amax/448 * 448/amax`
            // accumulates a 1-ULP residual that propagates into the MMA
            // fold). Matches parent fp8CS1x128's formulation
            // (fp8_blockscale_gemm_kernel.cuh:247).
            float const dequant_scale = 1.f / quant_scale; // fp8 * dequant -> bf16

            if (lane == 0)
            {
                smem_act_block_scales[kb_global] = dequant_scale;
            }

            // Quantize and write 4 fp8 elements per lane.
            __nv_fp8_e4m3* lane_dst = smem_act_fp8 + block_off + lane * kElemsPerLane128Block;
            float const q0 = fmaxf(-kFp8Max, fminf(kFp8Max, f0 * quant_scale));
            float const q1 = fmaxf(-kFp8Max, fminf(kFp8Max, f1 * quant_scale));
            float const q2 = fmaxf(-kFp8Max, fminf(kFp8Max, f2 * quant_scale));
            float const q3 = fmaxf(-kFp8Max, fminf(kFp8Max, f3 * quant_scale));
            __nv_fp8_e4m3 const fp8_0 = __nv_fp8_e4m3(q0);
            __nv_fp8_e4m3 const fp8_1 = __nv_fp8_e4m3(q1);
            __nv_fp8_e4m3 const fp8_2 = __nv_fp8_e4m3(q2);
            __nv_fp8_e4m3 const fp8_3 = __nv_fp8_e4m3(q3);
            // Pack 4 fp8 = 4 bytes = uint32 store.
            uint32_t const packed = (static_cast<uint32_t>(static_cast<uint8_t>(fp8_0.__x)) << 0)
                | (static_cast<uint32_t>(static_cast<uint8_t>(fp8_1.__x)) << 8)
                | (static_cast<uint32_t>(static_cast<uint8_t>(fp8_2.__x)) << 16)
                | (static_cast<uint32_t>(static_cast<uint8_t>(fp8_3.__x)) << 24);
            *reinterpret_cast<uint32_t*>(lane_dst) = packed;
        }
    }
    __syncthreads();

    // ----- Top-K stage 2 (warp 0 only); quant warps + warp 1 idle here. -----
    if (warp_idx == 0)
    {
        float cand_val = (lane < kNumInterTopK) ? smem_inter_scores[lane] : kInvalidScore;
        int32_t cand_idx = (lane < kNumInterTopK) ? smem_inter_experts[lane] : (kNumExperts - 1);
        mega_topk::reduceTopK<kTopK, float>(warp, top_scores, top_experts, cand_val, cand_idx, kInvalidScore, kTopK);

        int32_t const expert_idx = (lane < kTopK) ? top_experts[lane] : (kNumExperts - 1);
        float const score_norm = (lane < kTopK) ? smem_score_sigmoid[expert_idx] : 0.f;
        // Match noAuxTcKernels.cu: the warp reduction itself is fp32, then
        // the double scaling factor and double literal promote the final
        // division expression before the OutputT cast.
        float const red_norm = cg::reduce(warp, score_norm, cg::plus<float>{});
        double const final_score_d = (double) score_norm * (double) routed_scaling_factor / ((double) red_norm + 1e-20);
        float const final_score = static_cast<float>(final_score_d);

        if (lane < kTopK)
        {
            smem_topk_i[lane] = expert_idx;
            if (blockIdx.y == 0)
            {
                int64_t out_off = static_cast<int64_t>(token) * kTopK + lane;
                topk_weights[out_off] = final_score;
                topk_indices[out_off] = expert_idx;
            }
        }
    }
    __syncthreads();

    int const my_expert = (expert_slot == 0) ? kSharedExpert : smem_topk_i[expert_slot - 1];

    bool const is_gate_worker = (warp_idx >= kGateWorkerWarpBase && warp_idx < (kGateWorkerWarpBase + kNumGateWorkers));
    bool const is_up_worker = (warp_idx >= kUpWorkerWarpBase && warp_idx < (kUpWorkerWarpBase + kNumUpWorkers));
    int const my_m_gate = is_gate_worker ? (warp_idx - kGateWorkerWarpBase) : 0;
    int const my_m_up = is_up_worker ? (warp_idx - kUpWorkerWarpBase) : 0;

    float d_gate[4] = {0.f, 0.f, 0.f, 0.f};
    float d_up[4] = {0.f, 0.f, 0.f, 0.f};

    // ---- v68 K-LOOP - per-K-BLOCK scaling (6 scales per K-iter, applied
    //      block-wise inside the inner MMA loop). ----
    for (int k = 0; k < kNumKIter; ++k)
    {
        // Per-K-iter, per-K-block scale load: 6 fp32 weight scales per K-iter.
        // Also load the corresponding 6 per-128-col activation dequant scales
        // (produced by the in-kernel 1x128 quant during Phase 1) so the
        // compute_mma helper can fold both per-K-block.
        float gate_block_scales[kWeightScaleKBlocksPerKIter];
        float up_block_scales[kWeightScaleKBlocksPerKIter];
        float act_block_scales[kWeightScaleKBlocksPerKIter];
        if (is_gate_worker || is_up_worker)
        {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
            {
                act_block_scales[kb] = smem_act_block_scales[k * kWeightScaleKBlocksPerKIter + kb];
            }
        }
        else
        {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
            {
                act_block_scales[kb] = 0.f;
            }
        }
        if (is_gate_worker)
        {
            float const* const gate_block_scale_base = raw_scale_ptr<kInterPerTpParam, true>(
                shared_gate_up_scale, routed_w3_w1_scale, my_expert, sub_row, k);
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
            {
                gate_block_scales[kb] = __ldg(gate_block_scale_base + kb);
            }
        }
        else
        {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
            {
                gate_block_scales[kb] = 0.f;
            }
        }
        if (is_up_worker)
        {
            float const* const up_block_scale_base = raw_scale_ptr<kInterPerTpParam, false>(
                shared_gate_up_scale, routed_w3_w1_scale, my_expert, sub_row, k);
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
            {
                up_block_scales[kb] = __ldg(up_block_scale_base + kb);
            }
        }
        else
        {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb)
            {
                up_block_scales[kb] = 0.f;
            }
        }

        load_raw_weight_tile_v68<kInterPerTpParam, true>(
            smem_gate_tiles, shared_gate_up_weight, routed_w3_w1_weight, my_expert, sub_row, k, tidx);
        load_raw_weight_tile_v68<kInterPerTpParam, false>(
            smem_up_tiles, shared_gate_up_weight, routed_w3_w1_weight, my_expert, sub_row, k, tidx);
        __syncthreads();

        if (is_gate_worker)
        {
            compute_mma_kiter_v68(
                smem_gate_tiles, smem_act_fp8, k, my_m_gate, lane, gate_block_scales, act_block_scales, d_gate);
        }
        if (is_up_worker)
        {
            compute_mma_kiter_v68(
                smem_up_tiles, smem_act_fp8, k, my_m_up, lane, up_block_scales, act_block_scales, d_up);
        }
        __syncthreads();
    }

    __syncthreads();

    // ===== Phase 5: SiLU*x writer =====
    if (is_gate_worker && (lane & 3) == 0)
    {
        int const row_top = lane >> 2;
        int const row_bot = row_top + 8;
        smem_gate_acc[my_m_gate * 16 + row_top] = d_gate[0];
        smem_gate_acc[my_m_gate * 16 + row_bot] = d_gate[2];
    }
    if (is_up_worker && (lane & 3) == 0)
    {
        int const row_top = lane >> 2;
        int const row_bot = row_top + 8;
        smem_up_acc[my_m_up * 16 + row_top] = d_up[0];
        smem_up_acc[my_m_up * 16 + row_bot] = d_up[2];
    }
    __syncthreads();

    // Routed baseline applies fp8 quantize/dequantize to the gate/up output
    // before SwiGLU. This local 64-row scale is the closest match available
    // inside one CTA; the reference uses 128-row blocks.
    if (expert_slot != 0)
    {
        float const gate_abs = tidx < kCtaOutRows ? fabsf(smem_gate_acc[tidx]) : 0.f;
        float const warp_amax = cg::reduce(warp, gate_abs, cg::greater<float>{});
        if (lane == 0 && warp_idx < 2)
        {
            smem_score_sigmoid[warp_idx] = warp_amax;
        }
        __syncthreads();
        if (tidx == 0)
        {
            smem_score_sigmoid[0] = fmaxf(smem_score_sigmoid[0], smem_score_sigmoid[1]);
        }
        __syncthreads();
        if (tidx < kCtaOutRows)
        {
            float const amax = smem_score_sigmoid[0];
            float const dequant_scale = amax > 0.f ? (amax / kFp8Max) : 0.f;
            float const quant_scale = amax > 0.f ? (kFp8Max / amax) : 1.f;
            float const q = fmaxf(-kFp8Max, fminf(kFp8Max, smem_gate_acc[tidx] * quant_scale));
            smem_gate_acc[tidx] = static_cast<float>(__nv_fp8_e4m3(q)) * dequant_scale;
        }
    }

    if (tidx < kCtaOutRows)
    {
        float const g = smem_gate_acc[tidx];
        float const u = smem_up_acc[tidx];
        float const silu_g = g * sigmoid_accurate(g);
        float const h = silu_g * u;
        int const global_row = row_stripe_start + tidx;
        int64_t const out_off = static_cast<int64_t>(token) * kSlotsPerToken * kInterPerTp
            + static_cast<int64_t>(expert_slot) * kInterPerTp + global_row;
        // [iter5-fp16-hidden] Keep iter-2's fp16 store. Iter-5 reverts the
        // iter-4 bf16-revert to isolate the fp64 renorm fix as a single
        // variable. v110 consumes this handoff as FLOAT16.
        hidden_out[out_off] = __float2half(h);
    }
}

// Dynamic smem sizing.
static inline size_t v68_smem_bytes()
{
    auto align_up_128 = [](size_t p) -> size_t { return (p + 127u) & ~size_t(127); };
    size_t bytes = 0;
    bytes += static_cast<size_t>(kStages) * kTileBytes;
    bytes += static_cast<size_t>(kStages) * kTileBytes;
    bytes += static_cast<size_t>(kHidden) * sizeof(__nv_bfloat16);
    // Separate fp8 act buffer (no longer aliased over bf16). +6 KiB.
    bytes += static_cast<size_t>(kHidden) * sizeof(__nv_fp8_e4m3);
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumExperts;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumExperts;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumInterTopK;
    bytes = align_up_128(bytes);
    bytes += sizeof(int32_t) * kNumInterTopK;
    bytes = align_up_128(bytes);
    bytes += sizeof(int32_t) * kTopK;
    bytes = align_up_128(bytes);
    // smem_act_block_scales[48]: per-128-col activation dequant scales.
    bytes += sizeof(float) * kWeightScaleKBlocks;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kCtaOutRows;
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kCtaOutRows;
    bytes = align_up_128(bytes);
    bytes = (bytes + 15u) & ~size_t(15);
    return bytes;
}

template <int kInterPerTpParam>
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mega_silu_v68_impl(torch::Tensor scores,
    torch::Tensor hidden_in, torch::Tensor bias, torch::Tensor shared_gate_up_weight,
    torch::Tensor shared_gate_up_scale, torch::Tensor routed_w3_w1_weight, torch::Tensor routed_w3_w1_scale,
    double routed_scaling_factor)
{
    constexpr int kInterPerTp = kInterPerTpParam;
    constexpr int kCtasPerToken = ctas_per_token(kInterPerTp);
    constexpr int kWeightScaleMBlocks = weight_scale_m_blocks(kInterPerTp);

    auto const M = scores.size(0);

    TORCH_CHECK(shared_gate_up_weight.size(0) == 2 * kInterPerTp && shared_gate_up_weight.size(1) == kHidden,
        "shared_gate_up_weight shape mismatch");
    TORCH_CHECK(
        shared_gate_up_scale.size(0) == 2 * kWeightScaleMBlocks && shared_gate_up_scale.size(1) == kWeightScaleKBlocks,
        "shared_gate_up_scale shape mismatch");
    TORCH_CHECK(routed_w3_w1_weight.size(0) == kNumExperts && routed_w3_w1_weight.size(1) == 2 * kInterPerTp
            && routed_w3_w1_weight.size(2) == kHidden,
        "routed_w3_w1_weight shape mismatch");
    TORCH_CHECK(routed_w3_w1_scale.size(0) == kNumExperts && routed_w3_w1_scale.size(1) == 2 * kWeightScaleMBlocks
            && routed_w3_w1_scale.size(2) == kWeightScaleKBlocks,
        "routed_w3_w1_scale shape mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());

    static bool s_logged = false;

    if (!s_logged)
    {
        printf(
            "[v68_integrated] kKTile=%d, kStages=%d, LB(384,%d), "
            "kInterPerTp=%d (raw shared gate/up, routed up/gate)\n",
            kKTile, kStages, V68_LB_BLOCKS_PER_SM, kInterPerTp);
        s_logged = true;
    }

    auto topk_weights = torch::empty({M, kTopK}, torch::dtype(torch::kFloat32).device(scores.device()));
    auto topk_indices = torch::empty({M, kTopK}, torch::dtype(torch::kInt32).device(scores.device()));
    // [iter5-fp16-hidden] hidden_out is fp16 (kept from iter-2). See kernel
    // signature comment for rationale.
    auto hidden_out
        = torch::empty({M, kSlotsPerToken, kInterPerTp}, torch::dtype(torch::kHalf).device(scores.device()));

    dim3 grid(static_cast<unsigned>(M), kCtasPerToken, 1);
    dim3 block(kThreadsPerCta, 1, 1);

    size_t const smem_bytes = v68_smem_bytes();

    static bool s_smem_opt_done = false;
    if (!s_smem_opt_done)
    {
        cudaError_t err = cudaFuncSetAttribute(mega_kernel_v68<kInterPerTpParam>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
        TORCH_CHECK(err == cudaSuccess, "cudaFuncSetAttribute(v68, maxDynSmem) failed: ", cudaGetErrorString(err));
        s_smem_opt_done = true;
    }

    mega_kernel_v68<kInterPerTpParam><<<grid, block, smem_bytes, stream>>>(scores.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 const*>(hidden_in.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_fp8_e4m3 const*>(shared_gate_up_weight.data_ptr<at::Float8_e4m3fn>()),
        shared_gate_up_scale.data_ptr<float>(),
        reinterpret_cast<__nv_fp8_e4m3 const*>(routed_w3_w1_weight.data_ptr<at::Float8_e4m3fn>()),
        routed_w3_w1_scale.data_ptr<float>(), topk_weights.data_ptr<float>(), topk_indices.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(hidden_out.data_ptr<at::Half>()), M, static_cast<float>(routed_scaling_factor));

    return std::make_tuple(topk_weights, topk_indices, hidden_out);
}

} // anonymous namespace

namespace mega_kernel
{

// Keep the exported symbols suffixed with `_integrated` so this file can
// coexist with older v68 experiments that used unsuffixed launcher names.
// The TRT-LLM thop wrapper dispatches the Python entry points to these names.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mega_silu_v68_integrated(torch::Tensor scores,
    torch::Tensor hidden_in, torch::Tensor bias, torch::Tensor shared_gate_up_weight,
    torch::Tensor shared_gate_up_scale, torch::Tensor routed_w3_w1_weight, torch::Tensor routed_w3_w1_scale,
    double routed_scaling_factor)
{

    TORCH_CHECK(scores.is_cuda() && hidden_in.is_cuda() && bias.is_cuda() && shared_gate_up_weight.is_cuda()
            && shared_gate_up_scale.is_cuda() && routed_w3_w1_weight.is_cuda() && routed_w3_w1_scale.is_cuda(),
        "inputs must be CUDA");
    TORCH_CHECK(scores.dtype() == torch::kFloat32 && hidden_in.dtype() == torch::kBFloat16
            && bias.dtype() == torch::kBFloat16 && shared_gate_up_weight.dtype() == torch::kFloat8_e4m3fn
            && shared_gate_up_scale.dtype() == torch::kFloat32 && routed_w3_w1_weight.dtype() == torch::kFloat8_e4m3fn
            && routed_w3_w1_scale.dtype() == torch::kFloat32,
        "dtype mismatch");

    auto const M = scores.size(0);
    TORCH_CHECK(scores.size(1) == kNumExperts);
    TORCH_CHECK(hidden_in.size(0) == M && hidden_in.size(1) == kHidden);
    TORCH_CHECK(bias.size(0) == kNumExperts);

    TORCH_CHECK(scores.is_contiguous() && hidden_in.is_contiguous() && bias.is_contiguous()
            && shared_gate_up_weight.is_contiguous() && shared_gate_up_scale.is_contiguous()
            && routed_w3_w1_weight.is_contiguous() && routed_w3_w1_scale.is_contiguous(),
        "inputs must be contiguous");

    TORCH_CHECK(shared_gate_up_weight.dim() == 2, "shared_gate_up_weight must be 2D [2 * inter_per_tp, hidden]");
    TORCH_CHECK(
        shared_gate_up_scale.dim() == 2, "shared_gate_up_scale must be 2D [2 * inter_per_tp / 128, hidden / 128]");
    TORCH_CHECK(
        routed_w3_w1_weight.dim() == 3, "routed_w3_w1_weight must be 3D [num_experts, 2 * inter_per_tp, hidden]");
    TORCH_CHECK(routed_w3_w1_scale.dim() == 3,
        "routed_w3_w1_scale must be 3D [num_experts, 2 * inter_per_tp / 128, hidden / 128]");
    TORCH_CHECK(routed_w3_w1_weight.size(0) == kNumExperts && routed_w3_w1_weight.size(2) == kHidden,
        "routed_w3_w1_weight shape mismatch");
    TORCH_CHECK(routed_w3_w1_scale.size(0) == kNumExperts && routed_w3_w1_scale.size(2) == kWeightScaleKBlocks,
        "routed_w3_w1_scale shape mismatch");

    int64_t const inter_per_tp = routed_w3_w1_weight.size(1) / 2;
    TORCH_CHECK(routed_w3_w1_weight.size(1) == 2 * inter_per_tp, "routed_w3_w1_weight.size(1) must be even");
    TORCH_CHECK(shared_gate_up_weight.size(0) == 2 * inter_per_tp && shared_gate_up_weight.size(1) == kHidden,
        "shared_gate_up_weight shape mismatch");
    TORCH_CHECK(
        shared_gate_up_scale.size(0) == 2 * (inter_per_tp / 128) && shared_gate_up_scale.size(1) == kWeightScaleKBlocks,
        "shared_gate_up_scale shape mismatch");
    TORCH_CHECK(routed_w3_w1_scale.size(1) == 2 * (inter_per_tp / 128), "routed_w3_w1_scale shape mismatch");

    if (inter_per_tp == kInterPerTp_TP8)
    {
        return mega_silu_v68_impl<kInterPerTp_TP8>(std::move(scores), std::move(hidden_in), std::move(bias),
            std::move(shared_gate_up_weight), std::move(shared_gate_up_scale), std::move(routed_w3_w1_weight),
            std::move(routed_w3_w1_scale), routed_scaling_factor);
    }
    else if (inter_per_tp == kInterPerTp_TP4)
    {
        return mega_silu_v68_impl<kInterPerTp_TP4>(std::move(scores), std::move(hidden_in), std::move(bias),
            std::move(shared_gate_up_weight), std::move(shared_gate_up_scale), std::move(routed_w3_w1_weight),
            std::move(routed_w3_w1_scale), routed_scaling_factor);
    }
    else
    {
        TORCH_CHECK(false,
            "v68 only supports TP=4 (kInterPerTp=512) or TP=8 "
            "(kInterPerTp=256); inferred kInterPerTp=",
            inter_per_tp);
        return {};
    }
}

} // namespace mega_kernel
