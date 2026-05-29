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

// v110 ExpertDownProject mega kernel — dv103 base + TileRT SASS pattern match.
// This local-development variant strips the residual add and peer allreduce from
// the original branch kernel. It computes only:
//   shared_down(slot 0) + sum_i routed_weight_i * routed_down(slot i + 1).
//
// dv103's K-loop body has 320 SEL + 263 SHF.R.U32.HI integer ops to do
// byte-selection from 2 wide LDS.128 loads. dv110 bakes col_lo into the
// LDS address and uses 4 narrow LDS.U16 per K-iter instead, matching TileRT's
// BS=4 pattern (narrow per-lane loads after LDGSTS prefetch).
//
// Predicted SASS delta per-spec K-loop body: -320 SEL, -263 SHF.R.U32.HI,
// -128 LDS.128, +256 LDS.U16 = -455 integer/wide-LDS ops, +256 narrow-LDS
// = net 199 fewer ops, ~4× narrower smem traffic.
//
// Current WIP delta: hidden_in is staged with generic LDG instead of TMA, while
// preserving dv22's [K_chunks, M, 9, 128] SMEM layout. This keeps the v68->v110
// handoff in the generic proxy and avoids expensive cross-proxy/system fences.
//
// Per D107/D108: mechanism levers null lately; this is a SASS-pattern lever
// targeting the largest opcode-count delta vs TileRT BS=4. Honest a-priori:
// 0.5-2% if compiler issues are smem-pipe-throttled, may regress if narrow
// LDS bank conflicts dominate.
//
// All other dv103 levers preserved unchanged.
//
// =========================================================================
// Original dv103 header (PRESERVED FOR CONTEXT):
// =========================================================================
// v103 ExpertDownAllReduce mega kernel — dv93 base + add_residual TEMPLATE
// + my_rank PER-RANK SPECIALIZATION (per dv100 audit C01 + C02).
//
// On top of dv93's K_local=512/num_peers=4/kFp8Stages=4 hard-spec, dv110 adds:
//   * `bool kAddRes` template parameter — replaces the runtime `add_residual`
//     bool. Each rank-0 launch instantiates kAddRes=true; rank>0 launches
//     instantiate kAddRes=false. Eliminates the `if (add_residual)` branch
//     and 2 residual LDG.E.U16 + 2 BF16->F32 + 2 FADD per pair-cell + tail.
//
//   * `int kMyRank` template parameter (0..3) — replaces the runtime
//     `my_rank` int32. Each rank launches its own kernel instantiation.
//     The compiler folds the writer_stride offset to a compile-time literal
//     and unrolls the `for (p=0; p<4; ++p) if (p==my_rank) continue;` peer
//     loops into 3 straight-line iterations with literal peer indices,
//     eliminating the per-iter ISETP+BRA skip-self compare.
//
// Total kernel specializations: 2 (kAddRes) x 4 (kMyRank) = 8 variants,
// dispatched at launch time. Production-deploy ONLY (same fixed config
// as dv93: M=4, K_local=512, TP=4).
//
// =========================================================================
// Original dv93 header (PRESERVED FOR CONTEXT):
// =========================================================================
// v93 ExpertDownAllReduce mega kernel — dv85 base + EXTENDED hard-spec
// to additional deploy constants (K_local=512, num_peers=4 [TP=4],
// kFp8Stages=4). Tests D89 compile-time-spec stall-budget expansion with
// more constant folding.
//
// On top of dv85's M=4 specialization, dv93 hard-codes:
//   * K_local = 512  (TP=4, GLM-5 deploy)
//   * k_blocks = K_local / kBlockK = 4
//   * n_groups = k_blocks / kKBlocksPerGroup = 1
//   * hidden_k_chunks = K_local / kHiddenKChunk = 4
//   * num_peers = 4  (TP=4 test rig)
//   * kFp8Stages = 4 (always — k_blocks=4 always supports it; launcher
//     drops the runtime stage-selection logic)
//   * kb_stride_bytes, m_stride_bytes_h, slot_stride_bytes_h fold to
//     compile-time u32 literals.
//
// All M-dependent foldings preserved from dv85.
//
// Estimated SASS savings (audit-predicted):
//   * Eliminates n_groups=1 outer kg-loop (was 1 trip but produced a loop
//     counter, kg-stride add, scale-table index calc each kig).
//   * Steady-state `next_kb < k_blocks` check: with stages=4 and
//     k_blocks=4, next_kb (= kb+4) is always >= 4, so the entire
//     steady-state re-issue is dead code and can be removed.
//   * Prologue computes min(kFp8Stages, k_blocks) = 4: simplifies to
//     unconditional 4-iter prologue.
//   * `if (num_peers <= 1)` single-rank fast-path: dead at TP=4.
//   * `if (p >= num_peers) break;` in peer loops: replaced with
//     `p < 4` direct bound; eliminates per-iteration compare-break.
//   * AR poll/publish loops: 4 iters each; the conditional skip-self
//     (`p == my_rank`) stays runtime (varies per rank).
//   * Launcher: only `mega_down_v110_kernel<4>` compiled; the `<2>` /
//     `<1>` stage variants and chosen_stages selector are deleted.
//
// Trade-off: ONLY works at the FIXED deploy config (M=4, K_local=512,
// num_peers=4). Launcher TORCH_CHECKs enforce these.
//
// All dv56 levers (fp16 MMA, fp16 unpack, bf16->fp16 in-place narrow,
// TMA SWIZZLE_128B, kStages=4, batched-publish AR, etc.) preserved.
//
// =========================================================================
// Original dv56 header (PRESERVED FOR CONTEXT):
// =========================================================================
// v56 ExpertDownAllReduce mega kernel — dv53 base + fp16 MMA retry.
//
// dv56 takes dv53 (TMA SWIZZLE_128B + 1024B alignment + STS reduction)
// and swaps the MMA inputs from bf16 to fp16. Three mechanism levers
// (same as dv12 on dv9, applied on the new dv53 base):
//
//   1) fp8 -> fp16 unpack: single instr `cvt.rn.f16x2.e4m3x2`
//      (the bf16x2 variant isn't supported on sm_103a; the f16x2
//      variant is).  Replaces dv53's two-step f32 -> bf16x2 path.
//
//   2) MMA dtype: `m16n8k16.row.col.f32.f16.f16.f32` replaces the
//      bf16.bf16 variant. fp32 accumulator unchanged.
//
//   3) hidden_in narrowing: bf16 -> fp16 in smem AFTER the dv22 TMA
//      bulk staging (same byte footprint, in-place reinterpret).
//      B-fragment pack: fp16x2 pairs sourced from the f16 smem.
//
// Risk per D34/D47: dv12 (fp16 MMA) + dv14 (kStages=4) interfered on
// dv9 base via HMMA-dispatch slack. dv53 already includes kStages=4.
// May again interfere. Empirical test.
//
// Honest a-priori range: -3% to +3% at M=4. The hypothesis is that
// dv53's new smem profile (SWIZZLE_128B + STS reduction + L2_256B)
// re-exposes the F2FP unpack as a binding cost, in which case fp16
// (single-cycle latency) + single-instr unpack saves cycles.
//
// fp16 dynamic range is ±65504. Per-block scale pre-fold (dv8b) keeps
// activations in a controlled range. MMA accumulator stays fp32.
//
// dv53 details preserved verbatim:
// dv53 changes vs dv30: eliminate the bf16-mini staging path. dv30
// converts fp8→bf16 into a 16×16 per-warp smem mini-buffer (1 STS.128
// per lane per K-iter), then `ldmatrix.x4.b16` reads it into A-frag.
// dv53 reads the 4 specific 2-byte (fp8x2) chunks each lane needs for
// the m16n8k16 A-fragment DIRECTLY from the fp8 stage via 4 LDS.U16
// per lane per K-iter, converts in registers, and skips the mini buffer
// and the LDSM.x4 entirely.
//
// Per-K-iter smem-op delta per warp:
//   * REMOVED: 1 LDS.64 (cvt source) + 1 STS.128 (mini) + 1 LDSM.x4 (A-frag)
//   * ADDED:   4 LDS.U16 (direct fp8 reads for a_frag[0..3])
// Net: -1 LDS.64 -1 STS.128 -1 LDSM.x4 +4 LDS.U16 per warp per K-iter.
// STS.128 count from this site drops to zero. Total smem-op width drops
// (LDS.U16 narrow but no STS pipe pressure). Brief D55: throughput-bound,
// so reducing STS count and total smem-op count relieves L1/TEX pipe.
//
// The mini-buffer smem allocation is kept zero-size (helper deleted).
//
// All other dv30 levers preserved unchanged:
//   * Grid (148, 1, 1), block (384, 1, 1)
//   * 3-phase AR tail (Phase A compute / B publish / C poll)
//   * dv11 self-publish elimination, dv14 kStages=4, dv16 L2_256B,
//     dv21 ldmatrix.x2 B-frag (B-frag still uses LDSM.x2 — same-warp
//     register-feasible was NOT possible there because the B operand
//     reads from a wide bf16 row in smem_hidden that requires the
//     ldmatrix lane permutation, not a per-lane register pack).
//   * dv22 TMA bulk hidden_in.
//   * dv30 batched-publish AR.
//
// SASS audit (dv30 baseline):
//   * 648 STS.128 — ALL from cvt_fp8_to_bf16_mini → eliminated in dv53.
//   * 357 plain STS — smem_part writes (cross-warp, CANNOT eliminate).
//   * 45 STS.U16 — smem_bucket_pairs (cross-warp, CANNOT eliminate).
//   * 45 STS.U8 — smem_bucket_count atomic helpers (cross-warp).
//   * 6 STS.64 — smem_expert_ids/smem_weights table fill (cross-warp).
//
// Note (D17 retrospective context): dv6a hoisted publish INTO compute,
//
// dv30 changes vs dv25: restructure the AR tail from
//   per-cell { compute → publish-to-all-peers → poll-all-peers → FADD → store }
// into a 3-phase pipeline:
//   Phase A: each thread computes ALL its owned cells' acc values; caches
//            them to a local smem `smem_ar_acc` buffer keyed by per-thread
//            cell index. NO STG/LDG here.
//   Phase B: TIGHT publish-loop — each thread iterates its cells and
//            issues STG.E.128.STRONG.SYS to every remote peer in a
//            no-branch back-to-back burst. NVLink can hold many STGs in
//            flight per SM; bursting all owned cells' peer-publishes
//            keeps the wire saturated.
//   Phase C: poll-loop — each thread iterates its cells, spin-polls each
//            remote peer's slot, FADDs into local acc, then writes the
//            final bf16 output.
//
// Single CTA-scope `__threadfence_system()` is hoisted from per-cell
// (dv25 dynamic count = N_cells per thread) to one-per-CTA before the
// publish burst. Skipping self-rank is preserved from dv11.
//
// The hypothesis (per the brief): NVLink intra-node latency per STG is
// ~1-2 µs but the wire can hold many in flight. dv25's per-cell
// publish→poll serialization pays the full RTT before issuing the next
// cell's STG. dv30's burst-publish PIPELINES the publishes so the
// AGGREGATE wall-time approaches max(per-STG latency, total / throughput)
// rather than the sum.
//
// CAVEAT (from dv6a/D16 retrospective). At M=4, total_cells = 42×4 = 168
// across 384 threads → only 168 threads active in the AR tail; each does
// at most 1 cell in the pair-cell path. dv6a confirmed simple burst at
// M=4 is wash. dv30 differentiates from dv6a by doing the FULL 3-phase
// split (cache the acc value so the publish-loop can stay purely
// STG-only), and by keeping the v25 super-stack compute base. The
// pair-cell loop already amortizes 2 cells per thread → publish loop
// issues 2 × (num_peers-1) = 6 STGs back-to-back per thread, which is
// what we want NVLink to pipeline.
//
// All other dv25 levers (dv11/dv14/dv16/dv21/dv22) PRESERVED unchanged.
//
// SUPER STACK from dv25 retained:
//   1) [dv11] AR self-publish elimination
//   2) [dv14] kStages=4 deeper TMA pipeline
//   3) [dv16] TMA L2 cache promotion = 256B
//   4) [dv21] B-fragment via ldmatrix.sync.aligned.m8n8.x2.b16
//   5) [dv22] TMA bulk activation prologue (hidden_in)
//
// SMEM accounting (worst case M=4, kStages=4):
//   * hidden buffer: 4 chunks × M × 9 × 128 × 2 = 4 × 4 × 9 × 128 × 2
//     = 36 KiB at M=4 (same total as dv9's per-K layout).
//   * partial accumulator: 9 × 3 × 16 × 16 × 4 = ~28 KiB.
//   * mbarriers: 12 × kFp8Stages × 8 + 8 (hidden mbarrier).
//   * fp8 W-tile ring: 12 warps × kStages × 2048 = 96 KiB at kStages=4.
//   * bf16 mini: 12 × 512 = 6 KiB.
//   * bucket tables, expert ids, weights: ~6 KiB.
//   Total ≈ 172 KiB at M=4, well within 232 KiB cap.
//
// Integration notes:
//   * dv22 changes smem_hidden to [K_chunks, M, 9, 128] row-major.
//     dv21's ldmatrix.x2 row-address scheme is updated to compute
//     per-(kb, m, slot) row offsets from this new layout.
//   * dv22's hidden mbarrier sits AFTER all per-warp W_down mbarriers
//     in the smem ring; index = kNumWarps * kFp8Stages.
//   * dv16's L2_256B applies to the W_down map; the new hidden map
//     uses L2_128B by default (kept narrow because hidden is small).
//
// Unchanged from dv9 base:
//   * Grid (148, 1, 1), block (384, 1, 1).
//   * HMMA.16816.F32.BF16 (NOT fp16).
//   * Routed-slot MMA dedup (atomicAdd-bucket by expert_id).
//   * 16-byte STG/LDG.E.128 AR vec.
//   * Sym-heap sentinel-flag AR primitive.
//   * Per-K-block fp32 scale pre-fold (dv8b).
//   * ldmatrix.x4.b16 A-fragment load (dv8c, from cvt fp8→bf16 mini).

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <list>
#include <mutex>
#include <torch/extension.h>
#include <unordered_map>

namespace
{

constexpr int kHiddenSize = 6144;
constexpr int kTopKPlusShared = 9;
constexpr int kRoutedSlots = 8;
constexpr int kSharedExpertIdx = 256;
constexpr int kNumExpertsTotal = 257;

constexpr int kThreadsPerCta = 384;
constexpr int kWarpSize = 32;
constexpr int kNumWarps = kThreadsPerCta / kWarpSize;                // 12
constexpr int kNumCtas = 148;
constexpr int kRowsPerCta = (kHiddenSize + kNumCtas - 1) / kNumCtas; // 42

constexpr int kBlockK = 128;
constexpr int kBlockN = 128;

constexpr int kMmaM = 16;
constexpr int kMmaK = 16;
constexpr int kKItersPerBlock = kBlockK / kMmaK;                   // 8
constexpr int kRowTilesPerCta = (kRowsPerCta + kMmaM - 1) / kMmaM; // 3
// [dv85] M=4 hard-specialization: kMaxM is now compile-time 4.
// Every M-dependent literal (loop bounds, smem offsets, buffer sizes)
// shrinks accordingly. Kernel ONLY accepts M=4.
constexpr int kMaxM = 4;

// [dv93/dv110-tp8] EXTENDED hard-spec for kFp8Stages. Other constants
// (kKLocal, kNumPeers, kKBlocksPerGroup, etc.) are now per-kernel template
// parameters to support both TP=4 (kKLocal=512, kNumPeers=4) and TP=8
// (kKLocal=256, kNumPeers=8).
constexpr int kSpecFp8Stages = 4; // always 4

// fp8 TMA staging: 16 rows × 128 fp8 bytes = 2048 bytes per stage,
// 2 stages per warp.
constexpr int kWtRows = kMmaM;
constexpr int kWtKChunk = kBlockK;
constexpr int kFp8BytesPerStage = kWtRows * kWtKChunk; // 2048

// Per-K-block grouping for pre-folded scale. K_local=512 (TP=4) -> 4
// K-blocks -> 1 K-group with kKBlocksPerGroup=4.
// K_local=256 (TP=8) -> 2 K-blocks -> 1 K-group with kKBlocksPerGroup=2.
// Inside the templated kernel this is bound to (kKLocal / 128) (since the
// design always yields a single K-group per (N-block, expert)).
// Host-side TORCH_CHECK enforces divisibility.

// [dv85] At M=4: kMaxRoutedPairs = 4 * 8 = 32 (was 128 at M=16).
constexpr int kMaxRoutedPairs = kMaxM * kRoutedSlots; // 32
constexpr int kMaxBuckets = kMaxRoutedPairs;          // 32

// [dv22] Hidden TMA box K-chunk granularity (innermost K bytes per TMA tile).
constexpr int kHiddenKChunk = kBlockK; // = 128

// -----------------------------------------------------------------------------
// fp8 (e4m3) -> fp16 conversion (single instr — was the lever-killer
// for bf16 on sm_103a, but fp16 path is supported).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint16_t fp8_pair)
{
    uint32_t out;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(out) : "h"(fp8_pair));
    return out;
}

__device__ __forceinline__ void prefetchGlobalL2(void const* ptr)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("prefetch.global.L2::evict_last [%0];\n" ::"l"(ptr));
#else
    (void) ptr;
#endif
}

__device__ __forceinline__ uint4 loadGlobalUint4L2(uint4 const* ptr)
{
    uint4 value;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("ld.global.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
                 : "l"(ptr));
#else
    value = *ptr;
#endif
    return value;
}

// -----------------------------------------------------------------------------
// HMMA.16816 f16xf16 -> fp32.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_f16(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
    uint32_t b1, float& c0, float& c1, float& c2, float& c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// -----------------------------------------------------------------------------
// ldmatrix.sync.aligned.x2.b16 — [dv21].
//
// 2 8x8 b16 matrices. Lanes 0..7 supply rows of matrix 0; lanes 8..15
// supply rows of matrix 1; lanes 16..31 are ignored for addressing
// (but the address must still be a legal smem address).
//
// Output per lane (m16n8k16 B-frag, K-major B operand):
//   r0[T] = bf16x2 at matrix_0[row T/4, cols 2*(T%4)..+1]
//   r1[T] = bf16x2 at matrix_1[row T/4, cols 2*(T%4)..+1]
// -----------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x2_b16(uint32_t smem_addr, uint32_t& r0, uint32_t& r1)
{
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

// -----------------------------------------------------------------------------
// mbarrier + cp.async.bulk.tensor.3d wrappers (TMA).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cvt_smem_addr(void const* smem_ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, int arrive_count)
{
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(addr), "r"(arrive_count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(uint64_t* mbar, uint32_t bytes)
{
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(uint64_t* mbar, uint32_t phase)
{
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile(
        "{\n"
        " .reg .pred P;\n"
        " WAIT_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "  @P bra DONE_%=;\n"
        "  bra WAIT_%=;\n"
        " DONE_%=:\n"
        "}\n" ::"r"(addr),
        "r"(phase));
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d(
    void* smem_dst, CUtensorMap const* tmap, int32_t coord_x, int32_t coord_y, int32_t coord_z, uint64_t* mbar)
{
    uint32_t smem_addr = cvt_smem_addr(smem_dst);
    uint32_t mbar_addr = cvt_smem_addr(mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n" ::"r"(smem_addr),
        "l"(tmap), "r"(coord_x), "r"(coord_y), "r"(coord_z), "r"(mbar_addr));
}

__device__ __forceinline__ void fence_proxy_async_shared()
{
    asm volatile("fence.proxy.async.shared::cta;\n" :::);
}

// -----------------------------------------------------------------------------
// [dv53] Direct fp8 -> A-fragment loader (STS-elimination).
//
// Replaces the dv30 sequence:
//   cvt_fp8_to_bf16_mini  -> STS.128 the bf16 mini
//   __syncwarp
//   ldmatrix.x4.b16       -> LDSM into A-frag
// with 2 LDS.U32 per lane that fetch the exact fp8 bytes the
// m16n8k16 A-fragment requires, plus 4 fp8x2->bf16x2 cvts.
//
// m16n8k16 A-fragment layout (per lane T in warp):
//   a_frag[0] = bf16x2 at (row T/4,     cols 2*(T%4)..2*(T%4)+1)  -> 2 bf16
//   a_frag[1] = bf16x2 at (row T/4 + 8, cols 2*(T%4)..2*(T%4)+1)
//   a_frag[2] = bf16x2 at (row T/4,     cols 2*(T%4)+8..2*(T%4)+9)
//   a_frag[3] = bf16x2 at (row T/4 + 8, cols 2*(T%4)+8..2*(T%4)+9)
//
// Observation: a_frag[0] (cols col_lo..col_lo+1) and a_frag[2]
// (cols col_lo+8..col_lo+9) come from the SAME row (row_lo). The
// pair spans 10 fp8 bytes. We can read these by ONE LDS.64 (8 B)
// from a re-arranged 16-byte tile... but actually we can do it
// with just 2 LDS.32 — one of (row_lo, [col_lo..col_lo+3]) and one
// of (row_lo, [col_lo+8..col_lo+11]). Reading 4 bytes (LDS.32) but
// using only the low 2 bytes is wasteful; smarter is one LDS.64
// per row, taking bytes [col_lo, col_lo+1] (low half) and
// [col_lo+8, col_lo+9] (high half from byte 8..11). Wait — the
// 8-byte stride is exactly an LDS.64 worth, so within a single
// LDS.64 reading bytes [col_lo..col_lo+7], we only get col_lo+1
// adjacent bytes, NOT col_lo+8.
//
// Simpler approach: 2 LDS.U32 per row. Lane T reads col_lo..col_lo+3
// (LDS.32 #1) and col_lo+8..col_lo+11 (LDS.32 #2). From #1 take the
// low 2 bytes as fp8x2_a0; from #2 take low 2 bytes as fp8x2_a2.
// Total: 4 LDS.U32 per K-iter per lane (2 rows × 2 reads each).
//
// [dv110] TileRT SASS pattern match: eliminate the 320 SEL + 263 SHF.R.U32.HI
// per-K-iter byte-extraction chain that dominates dv103's K-loop body.
//
// dv103 used 2 LDS.128 (32 bytes total per K-iter per lane) + 4× SEL +
// 4× SHF.R.U32.HI + 4× LOP3 to extract 8 bytes from 32. The bake-in of
// col_lo into the LDS address removes the runtime byte-select entirely.
//
// Per-lane needed bytes (= 8 bytes total per K-iter):
//   p0_u16 (row_lo, byte col_lo..col_lo+1)
//   p2_u16 (row_lo, byte col_lo+8..col_lo+9)
//   p1_u16 (row_hi, byte col_lo..col_lo+1)
//   p3_u16 (row_hi, byte col_lo+8..col_lo+9)
//
// dv110 loads 2 LDS.U16 per row (one at col_lo, one at col_lo+8), 4 LDS.U16
// total per K-iter per lane. Address fold:
//   base + row*kWtKChunk + ki_phys*kMmaK + col_lo
//   base + row*kWtKChunk + ki_phys*kMmaK + col_lo + 8
// where col_lo = (lane & 3) << 1 is pre-computed and stays in a register
// once per K-iter (not 4× per K-iter as in dv103's SEL chain).
//
// Predicted SASS delta vs dv103 K-loop body:
//   -320 SEL (R, R, R, !P0)         → 0
//   -263 SHF.R.U32.HI               → 0
//   -128 LDS.128 (2 per K-iter × 64)→ 0
//   +256 LDS.U16 (4 per K-iter × 64)→ +256
//   net opcode count delta: -455 ops removed, +256 ops added = -199 ops.
//   Byte-traffic delta: 32 bytes/K-iter → 8 bytes/K-iter (4× narrower).
//
// Trade-off: more smem-load issues but narrower bandwidth + zero
// integer-pipe pressure for the extract. Matches TileRT's BS=4 pattern
// where post-LDGSTS layout is consumed via per-lane narrow LDS reads
// rather than wide-load+extract.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void cvt_fp8_to_afrag_direct(
    uint32_t fp8_stage_smem, int ki, int lane, uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3)
{
    int const row_lo = (lane >> 2);     // 0..7
    int const row_hi = row_lo + 8;      // 8..15
    int const col_lo = (lane & 3) << 1; // 0,2,4,6

    // [dv53] SWIZZLE_128B de-swizzle: TMA wrote fp8 weight tile with
    // CU_TENSOR_MAP_SWIZZLE_128B (period 8 rows × 8 chunks of 16B each).
    // For logical (row, chunk=ki), physical chunk = ki XOR (row & 7).
    int const ki_phys_lo = ki ^ (row_lo & 7);
    int const ki_phys_hi = ki ^ (row_hi & 7); // == ki_phys_lo

    // [dv110] LDS.U16 with col_lo baked into the address. No SEL, no SHF.R.
    const uint32_t row_lo_base = fp8_stage_smem + (uint32_t) (row_lo * (int) kWtKChunk + ki_phys_lo * (int) kMmaK);
    const uint32_t row_hi_base = fp8_stage_smem + (uint32_t) (row_hi * (int) kWtKChunk + ki_phys_hi * (int) kMmaK);
    const uint32_t addr_p0 = row_lo_base + (uint32_t) col_lo;
    const uint32_t addr_p2 = row_lo_base + (uint32_t) col_lo + 8u;
    const uint32_t addr_p1 = row_hi_base + (uint32_t) col_lo;
    const uint32_t addr_p3 = row_hi_base + (uint32_t) col_lo + 8u;

    uint16_t p0_u16, p1_u16, p2_u16, p3_u16;
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p0_u16) : "r"(addr_p0));
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p2_u16) : "r"(addr_p2));
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p1_u16) : "r"(addr_p1));
    asm volatile("ld.shared.b16 %0, [%1];\n" : "=h"(p3_u16) : "r"(addr_p3));

    a0 = fp8x2_to_f16x2(p0_u16);
    a1 = fp8x2_to_f16x2(p1_u16);
    a2 = fp8x2_to_f16x2(p2_u16);
    a3 = fp8x2_to_f16x2(p3_u16);
}

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------
template <int kKLocal>
__global__ __launch_bounds__(kThreadsPerCta, 1) void mega_down_v110_kernel(
    __grid_constant__ const CUtensorMap routed_w_down_map, __grid_constant__ const CUtensorMap shared_w_down_map,
    // Read hidden_in via plain LDG (generic memory proxy) instead of TMA.
    // v68 writes hidden_out through normal global stores, so same-stream
    // kernel ordering is sufficient for this handoff without proxy fences.
    // [v68-fp16-hidden] Upstream v68 now writes hidden_in as fp16 (was bf16).
    // The downstream MMA already runs in fp16 so we accept fp16 directly,
    // skipping the bf16->fp16 narrowing pass that this kernel used to do.
    __half const* __restrict__ hidden_in_raw, int32_t const* __restrict__ indices, float const* __restrict__ scores,
    float const* __restrict__ routed_w_down_scale, // [256, 48, k_blocks]
    float const* __restrict__ shared_w_down_scale, // [48, k_blocks]
    __nv_bfloat16* __restrict__ output, int M)
{
    // [dv110-runtimeM] M is now a runtime kernel argument in [1, kMaxM].
    // kMaxM still sizes all smem buffers (compile-time upper bound). M
    // bounds only the active-token loops; trailing rows in smem may be
    // uninitialized and must NOT be read by any consumer.
    // [dv93] EXTENDED hard-spec — runtime args stripped:
    //   * num_peers parameter removed; replaced with constexpr (template).
    //   * K_local parameter removed; replaced with constexpr (template).
    //   * kFp8Stages template parameter removed; replaced with constexpr 4.
    constexpr int kFp8Stages = kSpecFp8Stages; // == 4
    constexpr int K_local = kKLocal;           // 512 (TP=4) or 256 (TP=8)
    // [dv110-tp8] Per-template derived counts. Same formulas as before;
    // values change for TP=8.
    //   TP=4 (kKLocal=512): kKBlocks=4
    //   TP=8 (kKLocal=256): kKBlocks=2
    constexpr int kKBlocks = kKLocal / kBlockK; // 4 or 2

    int const cta_id = blockIdx.x;
    int const tid = threadIdx.x;
    int const warp_id = tid >> 5;
    int const lane = tid & 31;

    int const row_lo = cta_id * kRowsPerCta;
    if (row_lo >= kHiddenSize)
        return;
    int const row_hi = min(row_lo + kRowsPerCta, kHiddenSize);
    int const rows_here = row_hi - row_lo;

    // [dv93/dv110-tp8] All derived loop counts fold to compile-time literals.
    constexpr int k_blocks = kKBlocks; // 4 (TP=4) or 2 (TP=8)

    extern __shared__ unsigned char smem_raw[];
    // [v68-fp16-hidden] Upstream v68 writes fp16 directly; the smem destination
    // is fp16 and we skip the bf16->fp16 narrowing pass.
    __half* smem_hidden = reinterpret_cast<__half*>(smem_raw);
    int const hidden_elems = M * kTopKPlusShared * K_local;
    const size_t hidden_bytes = sizeof(__half) * (size_t) hidden_elems;

    // Cooperatively warm L2 for the small v68->v110 handoff before the per-CTA
    // shared-memory staging below. The buffer is at most 36 KiB for M <= 4.
    if (tid == 0)
    {
        constexpr int kPrefetchBytes = 128;
        int const hidden_prefetch_lines = static_cast<int>((hidden_bytes + kPrefetchBytes - 1) / kPrefetchBytes);
        char const* hidden_prefetch_base = reinterpret_cast<char const*>(hidden_in_raw);
        for (int line = cta_id; line < hidden_prefetch_lines; line += kNumCtas)
        {
            prefetchGlobalL2(hidden_prefetch_base + static_cast<size_t>(line) * kPrefetchBytes);
        }
        if (cta_id == 0)
        {
            prefetchGlobalL2(indices);
            prefetchGlobalL2(scores);
        }
    }

    int32_t* smem_expert_ids = reinterpret_cast<int32_t*>(smem_raw + hidden_bytes);
    float* smem_weights = reinterpret_cast<float*>(smem_expert_ids + M * kTopKPlusShared);

    const size_t partial_base
        = hidden_bytes + sizeof(int32_t) * (size_t) M * kTopKPlusShared + sizeof(float) * (size_t) M * kTopKPlusShared;
    float* smem_part = reinterpret_cast<float*>(smem_raw + partial_base);

    int const part_elems = kTopKPlusShared * kRowTilesPerCta * kMmaM * kMaxM;
    const size_t partial_bytes = sizeof(float) * (size_t) part_elems;

    // ---- Bucketing tables (same as dv7/dv8a). ----
    size_t bucket_count_base = partial_base + partial_bytes;
    bucket_count_base = (bucket_count_base + 15) & ~size_t(15);
    int32_t* smem_bucket_count = reinterpret_cast<int32_t*>(smem_raw + bucket_count_base);

    size_t bucket_pairs_base = bucket_count_base + sizeof(int32_t) * (size_t) kNumExpertsTotal;
    bucket_pairs_base = (bucket_pairs_base + 15) & ~size_t(15);
    uint8_t* smem_bucket_pairs = reinterpret_cast<uint8_t*>(smem_raw + bucket_pairs_base);

    size_t unique_eid_base = bucket_pairs_base + (size_t) kNumExpertsTotal * kMaxM;
    unique_eid_base = (unique_eid_base + 15) & ~size_t(15);
    int16_t* smem_unique_eid = reinterpret_cast<int16_t*>(smem_raw + unique_eid_base);

    size_t num_unique_base = unique_eid_base + sizeof(int16_t) * (size_t) kMaxBuckets;
    num_unique_base = (num_unique_base + 15) & ~size_t(15);
    int32_t* smem_num_unique = reinterpret_cast<int32_t*>(smem_raw + num_unique_base);

    // ---- TMA mbarrier ring (one per (warp, stage) for W_down). 8B aligned. ----
    size_t mbar_base = num_unique_base + sizeof(int32_t) * 4;
    mbar_base = (mbar_base + 15) & ~size_t(15);
    uint64_t* smem_mbar = reinterpret_cast<uint64_t*>(smem_raw + mbar_base);
    int const kWdMbarCount = kNumWarps * kFp8Stages;
    int const kMbarCount = kWdMbarCount;

    // ---- fp8 TMA W-tile staging.
    //
    // [dv53] For TMA SWIZZLE_128B, the destination smem region must be
    // aligned to the swizzle natural period = 8 rows × 128 B = 1024 B.
    // The dv48 lever-2 attempt aligned only to 128 B; that caused the
    // swizzle's chunk-XOR pattern to land at a different row-modulo
    // depending on `fp8_base % 1024`, which varies with M (since the
    // hidden_in / per-token tables preceding fp8_base have M-dependent
    // sizes). Aligning to 1024 B makes the consumer's
    // `chunk_phys = chunk_logical XOR (row & 7)` formula correct for
    // ALL M values.
    size_t fp8_base = mbar_base + sizeof(uint64_t) * kMbarCount;
    fp8_base = (fp8_base + 1023) & ~size_t(1023); // [dv53] 1024 B for SWIZZLE_128B
    uint8_t* smem_fp8_stages = smem_raw + fp8_base;

    // ---- bf16 mini-buffers (per warp). ----
    // [dv53] bf16 mini-buffer eliminated. The lane>=16 fallback for
    // ldmatrix.x2 B-frag uses warp_fp8_addr instead (valid smem address).
    // We still consume fp8 stage memory in the same place.

    // ---- [dv22] Init mbarriers FIRST (W_down ring). ----
    if (tid < kMbarCount)
    {
        mbarrier_init(&smem_mbar[tid], 1);
    }
    if (tid == 0)
    {
        fence_proxy_async_shared();
    }
    __syncthreads();

    // ---- Stage hidden_in into SMEM through the generic memory proxy. ----
    {
        constexpr int kHalfElemsPerVec = sizeof(uint4) / sizeof(__half);
        constexpr int kHiddenVecsPerKChunk = kHiddenKChunk / kHalfElemsPerVec;
        constexpr int kKLocalVecs = K_local / kHalfElemsPerVec;
        static_assert(kHiddenKChunk % kHalfElemsPerVec == 0);
        static_assert(K_local % kHalfElemsPerVec == 0);
        int const chunk_elems = M * kTopKPlusShared * kHiddenKChunk;
        int const chunk_vecs = chunk_elems / kHalfElemsPerVec;
        uint4 const* hidden_src = reinterpret_cast<uint4 const*>(hidden_in_raw);
        uint4* hidden_dst = reinterpret_cast<uint4*>(smem_hidden);
#pragma unroll
        for (int chunk = 0; chunk < k_blocks; ++chunk)
        {
            uint4* dst_chunk = hidden_dst + (size_t) chunk * chunk_vecs;
            int const src_chunk_vec = chunk * kHiddenVecsPerKChunk;
            for (int chunk_vec = tid; chunk_vec < chunk_vecs; chunk_vec += kThreadsPerCta)
            {
                int const slot_vec = chunk_vec / kHiddenVecsPerKChunk;
                int const k_vec = chunk_vec - slot_vec * kHiddenVecsPerKChunk;
                int const m = slot_vec / kTopKPlusShared;
                int const slot = slot_vec - m * kTopKPlusShared;
                int const src_vec = (m * kTopKPlusShared + slot) * kKLocalVecs + src_chunk_vec + k_vec;

                dst_chunk[chunk_vec] = loadGlobalUint4L2(hidden_src + src_vec);
            }
        }
    }

    // ---- Build expert-id and weight tables [M, 9] in SMEM. ----
    {
        int const total_slots = M * kTopKPlusShared;
        for (int i = tid; i < total_slots; i += kThreadsPerCta)
        {
            int const m = i / kTopKPlusShared;
            int const s = i - m * kTopKPlusShared;
            int32_t eid;
            float w;
            if (s == 0)
            {
                eid = kSharedExpertIdx;
                w = 1.0f;
            }
            else
            {
                eid = indices[m * kRoutedSlots + (s - 1)];
                w = scores[m * kRoutedSlots + (s - 1)];
            }
            smem_expert_ids[i] = eid;
            smem_weights[i] = w;
        }
    }

    // ---- Initialise partial accumulator. ----
    for (int i = tid; i < part_elems; i += kThreadsPerCta)
    {
        smem_part[i] = 0.0f;
    }

    // ---- Init routed-dedup bucket tables. ----
    for (int i = tid; i < kNumExpertsTotal; i += kThreadsPerCta)
    {
        smem_bucket_count[i] = 0;
    }
    if (tid == 0)
    {
        *smem_num_unique = 0;
    }

    // Mbarriers are already initialised before weight TMA staging.
    __syncthreads();

    // ---- Bucket routed pairs by expert_id (dv7). ----
    {
        int const total_routed = M * kRoutedSlots;
        for (int i = tid; i < total_routed; i += kThreadsPerCta)
        {
            int const m = i / kRoutedSlots;
            int const s = i - m * kRoutedSlots;
            int const e_id = indices[m * kRoutedSlots + s];
            if (e_id < 0 || e_id >= kNumExpertsTotal)
                continue;

            uint8_t packed = static_cast<uint8_t>((m << 4) | (s + 1));

            int old_count = atomicAdd(&smem_bucket_count[e_id], 1);
            if (old_count < kMaxM)
            {
                smem_bucket_pairs[e_id * kMaxM + old_count] = packed;
            }
            if (old_count == 0)
            {
                int slot_idx = atomicAdd(smem_num_unique, 1);
                if (slot_idx < kMaxBuckets)
                {
                    smem_unique_eid[slot_idx] = static_cast<int16_t>(e_id);
                }
            }
        }
    }
    __syncthreads();

    // The previous __syncthreads waits for the LDG-based hidden staging,
    // table initialization, and bucket construction before any reader runs.

    int const num_unique = *smem_num_unique;

    // Per-warp mbarrier base index, fp8 stage base, bf16 mini base.
    int const warp_mbar_base = warp_id * kFp8Stages;
    uint8_t* warp_fp8_base = smem_fp8_stages + warp_id * (kFp8Stages * kFp8BytesPerStage);
    // [dv53] warp_mini_addr was used as safe-smem-addr fallback for
    // lane>=16 in ldmatrix.x2 B-frag — replaced by warp_fp8_addr.
    uint32_t warp_fp8_addr = cvt_smem_addr(warp_fp8_base);
    const uint32_t warp_mini_addr = warp_fp8_addr; // [dv53] safe-smem alias

    // [dv21+dv22] smem_hidden base addr for ldmatrix.x2 B-frag.
    //
    // dv22 smem layout = [K_chunks, M, 9, 128] row-major (innermost = 128
    // fp16 elements). So per-(kb, m, slot) row offset in bytes is:
    //   kb * (M * 9 * 128 * 2)
    // + m  * (9 * 128 * 2)
    // + s  * (128 * 2)
    // The fp16 row stride for a single (kb, m, slot) is 128 elements *
    // 2 bytes = 256 bytes — much larger than a single ldmatrix row
    // (16 bytes = 8 fp16). ldmatrix.x2 only uses per-lane addresses
    // (not strides), so the wide fp16 row works fine as the row source.
    const uint32_t smem_hidden_addr = cvt_smem_addr(smem_hidden);
    // [dv110-runtimeM] kb_stride depends on runtime M (hidden staging writes M-packed
    // rows per K-chunk). m_stride and slot_stride remain compile-time.
    const uint32_t kb_stride_bytes = (uint32_t) (M * kTopKPlusShared * kHiddenKChunk * 2);  // M*9*128*2
    constexpr uint32_t m_stride_bytes_h = (uint32_t) (kTopKPlusShared * kHiddenKChunk * 2); // 9*128*2 = 2304
    constexpr uint32_t slot_stride_bytes_h = (uint32_t) (kHiddenKChunk * 2);                // 128*2 = 256

    // TMA issue helper. Routed weights use z=e_id. The shared expert is a
    // separate standalone [N, K] matrix represented as a one-expert map with
    // z=0, matching the existing TRT-LLM weight layout without repacking.
    auto issue_tma_load = [&](int e_id, int row_base, int k_off, int stage_idx)
    {
        if (lane == 0)
        {
            int mbar_idx = warp_mbar_base + stage_idx;
            uint8_t* stage_smem = warp_fp8_base + stage_idx * kFp8BytesPerStage;
            mbarrier_arrive_expect_tx(&smem_mbar[mbar_idx], kFp8BytesPerStage);
            if (e_id == kSharedExpertIdx)
            {
                cp_async_bulk_tensor_3d(stage_smem, &shared_w_down_map,
                    /*x=*/k_off, /*y=*/row_base, /*z=*/0, &smem_mbar[mbar_idx]);
            }
            else
            {
                cp_async_bulk_tensor_3d(stage_smem, &routed_w_down_map,
                    /*x=*/k_off, /*y=*/row_base, /*z=*/e_id, &smem_mbar[mbar_idx]);
            }
        }
    };

    // Per-warp mbarrier phase tracking (toggled at each wait).
    // kFp8Stages now spans {1, 2, 3, 4} — size for max-of-template.
    constexpr int kMaxStages = 4;
    uint32_t mbar_phase[kMaxStages] = {0u, 0u, 0u, 0u};

    auto wait_stage = [&](int stage_idx)
    {
        int mbar_idx = warp_mbar_base + stage_idx;
        mbarrier_wait_parity(&smem_mbar[mbar_idx], mbar_phase[stage_idx]);
        mbar_phase[stage_idx] ^= 1u;
    };

    // ---- Phase A: shared-expert path. ----
    {
        int const shared_work = kRowTilesPerCta;
        for (int w = warp_id; w < shared_work; w += kNumWarps)
        {
            int const tile = w;
            int const row_base_in_cta = tile * kMmaM;
            int const row_base = row_lo + row_base_in_cta;

            int const rows_active = min(kMmaM, row_hi - row_base);
            if (rows_active <= 0)
                continue;

            int const e_id = kSharedExpertIdx;
            // [dv85] n_tiles_m = (4+7)/8 = 1 — single iteration, m_base=0.
            constexpr int kNTilesM = 1;
#pragma unroll
            for (int nt = 0; nt < kNTilesM; ++nt)
            {
                constexpr int m_base = 0;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                // Prologue: pre-issue stages 0..min(kFp8Stages-1, k_blocks-1).
                // For kFp8Stages==1 this issues nothing here (issue-on-demand
                // in the inner loop below mirrors dv9's single-stage path).
                if constexpr (kFp8Stages > 1)
                {
                    int const prologue_n = kFp8Stages < k_blocks ? kFp8Stages : k_blocks;
#pragma unroll
                    for (int s = 0; s < kMaxStages; ++s)
                    {
                        if (s < prologue_n)
                        {
                            issue_tma_load(e_id, row_base, s * kBlockK, s);
                        }
                    }
                }

                // [v110-perkb-scale] Per-K-block scale fold INSIDE the K-loop
                // (was: one group-max scale applied AFTER the loop, with the
                // ratio s_orig/s_max pre-folded into the FP8 weights — that
                // pre-fold rescaled every fp8 value through a sub-unity ratio
                // and re-quantized, leaking ~5 bits per weight). The new path
                // packs weights bit-identical to the source and applies the
                // per-K-block scale at MMA-fold time, matching v68's design.
                //
                // Per-lane row->n-block mapping is invariant over kb, so we
                // hoist it out of the kb loop.
                int const row0_lane_pkb = row_base + (lane >> 2);
                int const row8_lane_pkb = row_base + (lane >> 2) + 8;
                int const nb0_pkb = row0_lane_pkb / kBlockN;
                int const nb8_pkb = row8_lane_pkb / kBlockN;
                // Shared scale shape: [48, k_blocks]. Hoist the per-n-block
                // base; per-kb scales live at consecutive memory.
                float const* const s0_base_pkb = shared_w_down_scale + (size_t) nb0_pkb * (size_t) k_blocks;
                float const* const s8_base_pkb
                    = (nb8_pkb == nb0_pkb) ? s0_base_pkb : (shared_w_down_scale + (size_t) nb8_pkb * (size_t) k_blocks);

#pragma unroll
                for (int kb = 0; kb < k_blocks; ++kb)
                {
                    int const stage = (kFp8Stages == 1) ? 0 : (kb % kFp8Stages);

                    if constexpr (kFp8Stages == 1)
                    {
                        // 1-stage: issue current then wait.
                        issue_tma_load(e_id, row_base, kb * kBlockK, 0);
                    }
                    wait_stage(stage);
                    __syncwarp();

                    uint32_t fp8_stage_ptr = warp_fp8_addr + (uint32_t) (stage * kFp8BytesPerStage);

                    float c_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
                    for (int ki = 0; ki < kKItersPerBlock; ++ki)
                    {
                        // [dv53] Direct fp8 -> A-frag, no STS/LDSM.x4.
                        uint32_t a_frag[4];
                        cvt_fp8_to_afrag_direct(fp8_stage_ptr, ki, lane, a_frag[0], a_frag[1], a_frag[2], a_frag[3]);

                        // [dv21+dv22] B-frag via ldmatrix.x2.b16
                        // from smem layout [K_chunks, M, 9, 128].
                        // Lanes 0..7 supply mat0 rows (k=0..7 of ki);
                        // lanes 8..15 supply mat1 rows (k=8..15);
                        // lanes 16..31 use warp_mini_addr as a safe
                        // smem address (ignored by ldmatrix.x2).
                        uint32_t b_frag[2];
                        {
                            int const idx_in_row = lane & 7;                               // 0..7 (per-N row)
                            int const mat_id_b = (lane >> 3) & 1;                          // 0 or 1
                            int const m_for_lane = m_base + idx_in_row;
                            int const m_clamped = (m_for_lane < M) ? m_for_lane : (M - 1); // safe in-range
                            int const k_in_off = ki * kMmaK + mat_id_b * 8;
                            const uint32_t row_off = (uint32_t) kb * kb_stride_bytes
                                + (uint32_t) m_clamped * m_stride_bytes_h + 0u * slot_stride_bytes_h // shared slot=0
                                + (uint32_t) k_in_off * 2u;
                            uint32_t b_addr = (lane < 16) ? (smem_hidden_addr + row_off) : warp_mini_addr;
                            ldmatrix_x2_b16(b_addr, b_frag[0], b_frag[1]);
                        }

                        mma_m16n8k16_f16(a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag[0], b_frag[1], c_block[0],
                            c_block[1], c_block[2], c_block[3]);

                        // [dv53] no mini-buffer, so no syncwarp needed here.
                    }

                    // Steady-state: pre-issue (kb + kFp8Stages) into the
                    // just-consumed stage slot.
                    if constexpr (kFp8Stages > 1)
                    {
                        int const next_kb = kb + kFp8Stages;
                        if (next_kb < k_blocks)
                        {
                            issue_tma_load(e_id, row_base, next_kb * kBlockK, stage);
                        }
                    }

                    // [v110-perkb-scale] Per-K-block FFMA fold. Each K-block
                    // has its own scale (raw scale from the source, no rescale).
                    // The fp32 accumulator preserves intermediate precision
                    // across kb's, matching v68's per-K-block design and the
                    // parent fp8_block_scale_moe runner's precision profile.
                    float const s0_kb = s0_base_pkb[kb];
                    float const s8_kb = (nb8_pkb == nb0_pkb) ? s0_kb : s8_base_pkb[kb];
                    c[0] += c_block[0] * s0_kb;
                    c[1] += c_block[1] * s0_kb;
                    c[2] += c_block[2] * s8_kb;
                    c[3] += c_block[3] * s8_kb;
                }

                {
                    int const row0 = (lane >> 2);
                    int const row1 = row0 + 8;
                    int const col0_local = (lane & 3) * 2;
                    int const col1_local = col0_local + 1;
                    int const col0 = m_base + col0_local;
                    int const col1 = m_base + col1_local;
                    int const slot_off = 0 * kRowTilesPerCta * kMmaM * kMaxM + tile * kMmaM * kMaxM;
                    if (col0 < M && row0 < rows_active)
                        smem_part[slot_off + row0 * kMaxM + col0] = c[0];
                    if (col1 < M && row0 < rows_active)
                        smem_part[slot_off + row0 * kMaxM + col1] = c[1];
                    if (col0 < M && row1 < rows_active)
                        smem_part[slot_off + row1 * kMaxM + col0] = c[2];
                    if (col1 < M && row1 < rows_active)
                        smem_part[slot_off + row1 * kMaxM + col1] = c[3];
                }
            }
        }
    }

    // ---- Phase B: routed-expert dedup path. ----
    {
        int const total_outer = kRowTilesPerCta * num_unique;
        for (int w_outer = warp_id; w_outer < total_outer; w_outer += kNumWarps)
        {
            int const tile = w_outer / num_unique;
            int const b_idx = w_outer - tile * num_unique;
            int const row_base_in_cta = tile * kMmaM;
            int const row_base = row_lo + row_base_in_cta;
            int const rows_active = min(kMmaM, row_hi - row_base);
            if (rows_active <= 0)
                continue;

            int const e_id = smem_unique_eid[b_idx];
            int const count = static_cast<int>(smem_bucket_count[e_id]);
            int const n_groups_routed = (count + 7) >> 3; // 1 or 2

            for (int g = 0; g < n_groups_routed; ++g)
            {
                int const group_start = g * 8;
                int const group_count = min(8, count - group_start);

                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                // [dv21+dv22] Pre-compute per-lane base offset (in bytes)
                // into smem_hidden for ldmatrix.x2.b16 B-frag loads.
                // Lanes 0..7 address mat0 rows (one pair each); lanes 8..15
                // address mat1 rows (same pair as lane-8); lanes 16..31
                // are unused — fall back to warp_mini_addr.
                //
                // Layout = [K_chunks, M, 9, 128]: the kb stride is added
                // later (per K-iter); here we collect M*9 base only.
                uint32_t b_row_base_bytes = 0u;
                bool lane_has_pair = false;
                {
                    int const pair_idx_in_group = lane & 7;
                    if (lane < 16 && pair_idx_in_group < group_count)
                    {
                        uint8_t packed = smem_bucket_pairs[e_id * kMaxM + group_start + pair_idx_in_group];
                        int const mm = (packed >> 4) & 0xF;
                        int const ss = packed & 0xF;
                        b_row_base_bytes = (uint32_t) mm * m_stride_bytes_h + (uint32_t) ss * slot_stride_bytes_h;
                        lane_has_pair = true;
                    }
                }
                // Prologue: pre-issue stages 0..min(kFp8Stages-1, k_blocks-1).
                if constexpr (kFp8Stages > 1)
                {
                    int const prologue_n = kFp8Stages < k_blocks ? kFp8Stages : k_blocks;
#pragma unroll
                    for (int s = 0; s < kMaxStages; ++s)
                    {
                        if (s < prologue_n)
                        {
                            issue_tma_load(e_id, row_base, s * kBlockK, s);
                        }
                    }
                }

                // [v110-perkb-scale] Per-K-block scale fold INSIDE the K-loop
                // (see Phase A for design rationale). Routed path mirrors the
                // shared path's per-K-block FFMA fold.
                int const row0_lane_pkbR = row_base + (lane >> 2);
                int const row8_lane_pkbR = row_base + (lane >> 2) + 8;
                int const nb0_pkbR = row0_lane_pkbR / kBlockN;
                int const nb8_pkbR = row8_lane_pkbR / kBlockN;
                float const* const s0_base_pkbR
                    = routed_w_down_scale + ((size_t) e_id * (kHiddenSize / kBlockN) + nb0_pkbR) * (size_t) k_blocks;
                float const* const s8_base_pkbR = (nb8_pkbR == nb0_pkbR)
                    ? s0_base_pkbR
                    : (routed_w_down_scale + ((size_t) e_id * (kHiddenSize / kBlockN) + nb8_pkbR) * (size_t) k_blocks);

#pragma unroll
                for (int kb = 0; kb < k_blocks; ++kb)
                {
                    int const stage = (kFp8Stages == 1) ? 0 : (kb % kFp8Stages);

                    if constexpr (kFp8Stages == 1)
                    {
                        issue_tma_load(e_id, row_base, kb * kBlockK, 0);
                    }
                    wait_stage(stage);
                    __syncwarp();

                    uint32_t fp8_stage_ptr = warp_fp8_addr + (uint32_t) (stage * kFp8BytesPerStage);

                    float c_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
                    for (int ki = 0; ki < kKItersPerBlock; ++ki)
                    {
                        // [dv53] Direct fp8 -> A-frag, no STS/LDSM.x4.
                        uint32_t a_frag[4];
                        cvt_fp8_to_afrag_direct(fp8_stage_ptr, ki, lane, a_frag[0], a_frag[1], a_frag[2], a_frag[3]);

                        // [dv21+dv22] B-frag via ldmatrix.x2.b16
                        // from smem [K_chunks, M, 9, 128].
                        uint32_t b_frag[2];
                        {
                            int const mat_id_b = (lane >> 3) & 1;
                            int const k_in_off = ki * kMmaK + mat_id_b * 8;
                            const uint32_t k_off_bytes = (uint32_t) kb * kb_stride_bytes + (uint32_t) k_in_off * 2u;
                            uint32_t b_addr
                                = lane_has_pair ? (smem_hidden_addr + b_row_base_bytes + k_off_bytes) : warp_mini_addr;
                            ldmatrix_x2_b16(b_addr, b_frag[0], b_frag[1]);
                        }

                        mma_m16n8k16_f16(a_frag[0], a_frag[1], a_frag[2], a_frag[3], b_frag[0], b_frag[1], c_block[0],
                            c_block[1], c_block[2], c_block[3]);
                        // [dv53] no mini-buffer, so no syncwarp needed here.
                    }

                    // Steady-state: pre-issue (kb + kFp8Stages) into the
                    // freed stage slot.
                    if constexpr (kFp8Stages > 1)
                    {
                        int const next_kb = kb + kFp8Stages;
                        if (next_kb < k_blocks)
                        {
                            issue_tma_load(e_id, row_base, next_kb * kBlockK, stage);
                        }
                    }

                    // [v110-perkb-scale] Per-K-block FFMA fold.
                    float const s0_kbR = s0_base_pkbR[kb];
                    float const s8_kbR = (nb8_pkbR == nb0_pkbR) ? s0_kbR : s8_base_pkbR[kb];
                    c[0] += c_block[0] * s0_kbR;
                    c[1] += c_block[1] * s0_kbR;
                    c[2] += c_block[2] * s8_kbR;
                    c[3] += c_block[3] * s8_kbR;
                }

                {
                    int const row0 = (lane >> 2);
                    int const row1 = row0 + 8;
                    int const p0_local = (lane & 3) * 2;
                    int const p1_local = p0_local + 1;
                    int const p0 = group_start + p0_local;
                    int const p1 = group_start + p1_local;

                    auto write_cell = [&](int p, int row_in_tile, float val)
                    {
                        if (p < count && row_in_tile < rows_active)
                        {
                            uint8_t packed = smem_bucket_pairs[e_id * kMaxM + p];
                            const int m = (packed >> 4) & 0xF;
                            const int s = packed & 0xF;
                            const int off
                                = s * kRowTilesPerCta * kMmaM * kMaxM + tile * kMmaM * kMaxM + row_in_tile * kMaxM + m;
                            smem_part[off] = val;
                        }
                    };
                    write_cell(p0, row0, c[0]);
                    write_cell(p1, row0, c[1]);
                    write_cell(p0, row1, c[2]);
                    write_cell(p1, row1, c[3]);
                }
            }
        }
    }

    __syncthreads();

    // -------------------------------------------------------------------
    // Final local reduction. The original branch kernel added residual and
    // then all-reduced across TP peers here. This local-development variant
    // stores only this rank's weighted shared+routed down projection so the
    // existing post-MoE allreduce path can handle cross-rank reduction.
    // -------------------------------------------------------------------
    int const rows_pairs = rows_here >> 1;
    int const tail_rows = rows_here - (rows_pairs << 1);

    int const pair_cells = rows_pairs * M;

    // Each thread owns at most one pair-cell (pair_cells <= 336 < 384).
    bool const has_pair_cell = (tid < pair_cells);

    if (has_pair_cell)
    {
        int const cell = tid;
        int const pair_idx_in_cta = cell / M;
        int const m = cell - pair_idx_in_cta * M;
        int const row_in_cta0 = pair_idx_in_cta << 1;
        int const row_in_cta1 = row_in_cta0 + 1;
        int const row0 = row_lo + row_in_cta0;
        int const row1 = row0 + 1;

        int const tile0 = row_in_cta0 / kMmaM;
        int const rit0 = row_in_cta0 - tile0 * kMmaM;
        int const tile1 = row_in_cta1 / kMmaM;
        int const rit1 = row_in_cta1 - tile1 * kMmaM;

        float acc0 = 0.0f, acc1 = 0.0f;
#pragma unroll
        for (int s = 0; s < kTopKPlusShared; ++s)
        {
            float const w = smem_weights[m * kTopKPlusShared + s];
            int const off0 = s * kRowTilesPerCta * kMmaM * kMaxM + tile0 * kMmaM * kMaxM + rit0 * kMaxM + m;
            int const off1 = s * kRowTilesPerCta * kMmaM * kMaxM + tile1 * kMmaM * kMaxM + rit1 * kMaxM + m;
            acc0 += smem_part[off0] * w;
            acc1 += smem_part[off1] * w;
        }
        output[(size_t) m * kHiddenSize + row0] = __float2bfloat16(acc0);
        output[(size_t) m * kHiddenSize + row1] = __float2bfloat16(acc1);
    }

    // Tail-row owner (at most one per thread since M ≤ kMaxM=16 < 384).
    bool const has_tail_cell = (tail_rows > 0) && (tid < M);

    if (has_tail_cell)
    {
        int const row_in_cta = rows_here - 1;
        int const row = row_lo + row_in_cta;
        int const tile = row_in_cta / kMmaM;
        int const rit = row_in_cta - tile * kMmaM;
        int const m = tid;

        float acc = 0.0f;
#pragma unroll
        for (int s = 0; s < kTopKPlusShared; ++s)
        {
            int const off = s * kRowTilesPerCta * kMmaM * kMaxM + tile * kMmaM * kMaxM + rit * kMaxM + m;
            acc += smem_part[off] * smem_weights[m * kTopKPlusShared + s];
        }
        output[(size_t) m * kHiddenSize + row] = __float2bfloat16(acc);
    }
}

// -----------------------------------------------------------------------------
// CUtensorMap build helper.
// -----------------------------------------------------------------------------
// W_down has shape [E, N=6144, K=K_local], dtype = fp8_e4m3 = 1 byte.
//   * dim 0 (x = K) — element stride 1 byte
//   * dim 1 (y = N rows)
//   * dim 2 (z = E experts)
// Box dim per tile = [kBlockK, kMmaM, 1] = [128, 16, 1].
// [dv53] SWIZZLE_128B mode. 128-byte rows × 16 rows = 2 swizzle periods
// of 8 rows each. For each row r, the 8 16-byte chunks are permuted by
// chunk_phys = chunk_logical XOR (r & 7). Bank-conflict-free for the
// consumer's LDS.128 (see cvt_fp8_to_afrag_direct).
static CUtensorMap make_w_down_tmap(void* base_ptr, int num_experts, int K_local, CUresult* out_err)
{
    CUtensorMap map = {};
    cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(K_local),
        static_cast<cuuint64_t>(kHiddenSize),
        static_cast<cuuint64_t>(num_experts),
    };
    cuuint64_t global_stride[2] = {
        static_cast<cuuint64_t>(K_local),
        static_cast<cuuint64_t>(K_local) * static_cast<cuuint64_t>(kHiddenSize),
    };
    cuuint32_t box_dim[3] = {
        static_cast<cuuint32_t>(kBlockK),
        static_cast<cuuint32_t>(kMmaM),
        1u,
    };
    cuuint32_t elem_stride[3] = {1u, 1u, 1u};

    *out_err = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /*rank=*/3, base_ptr, global_dim, global_stride, box_dim, elem_stride, CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B, // [dv53] bank-conflict-free fp8 stage reads
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

constexpr size_t kWDownTmaDescCacheCap = 256;
constexpr int kMaxCudaDevicesForSmemAttr = 16;

struct WDownTmaDescKey
{
    void const* base;
    int numExperts;
    int kLocal;
    int deviceId;

    bool operator==(WDownTmaDescKey const& o) const noexcept
    {
        return base == o.base && numExperts == o.numExperts && kLocal == o.kLocal && deviceId == o.deviceId;
    }
};

struct WDownTmaDescKeyHash
{
    size_t operator()(WDownTmaDescKey const& k) const noexcept
    {
        size_t h = reinterpret_cast<uintptr_t>(k.base);
        h = h * 1099511628211ull + static_cast<size_t>(k.numExperts);
        h = h * 1099511628211ull + static_cast<size_t>(k.kLocal);
        h = h * 1099511628211ull + static_cast<size_t>(k.deviceId);
        return h;
    }
};

struct WDownTmaDescCache
{
    using ListIt = std::list<std::pair<WDownTmaDescKey, CUtensorMap>>::iterator;
    std::list<std::pair<WDownTmaDescKey, CUtensorMap>> order;
    std::unordered_map<WDownTmaDescKey, ListIt, WDownTmaDescKeyHash> index;
};

static CUtensorMap get_cached_w_down_tmap(
    void* base_ptr, int num_experts, int K_local, int device_id, CUresult* out_err)
{
    static thread_local WDownTmaDescCache cache;
    WDownTmaDescKey const key{base_ptr, num_experts, K_local, device_id};
    auto it = cache.index.find(key);
    if (it != cache.index.end())
    {
        cache.order.splice(cache.order.begin(), cache.order, it->second);
        *out_err = CUDA_SUCCESS;
        return it->second->second;
    }

    CUtensorMap const map = make_w_down_tmap(base_ptr, num_experts, K_local, out_err);
    if (*out_err != CUDA_SUCCESS)
    {
        return map;
    }
    if (cache.order.size() >= kWDownTmaDescCacheCap)
    {
        cache.index.erase(cache.order.back().first);
        cache.order.pop_back();
    }
    cache.order.emplace_front(key, map);
    cache.index.emplace(key, cache.order.begin());
    return map;
}

// -----------------------------------------------------------------------------
} // anonymous namespace

// -----------------------------------------------------------------------------
// Host launcher
// -----------------------------------------------------------------------------
torch::Tensor mega_down_project_v110(torch::Tensor hidden_in, torch::Tensor indices, torch::Tensor scores,
    torch::Tensor routed_w_down,       // fp8 [256, 6144, K_local]
    torch::Tensor routed_w_down_scale, // fp32 [256, 48, k_blocks]
    torch::Tensor shared_w_down,       // fp8 [6144, K_local]
    torch::Tensor shared_w_down_scale, // fp32 [48, k_blocks]
    torch::Tensor output)
{
    TORCH_CHECK(hidden_in.is_cuda(), "hidden_in must be CUDA");
    TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
    TORCH_CHECK(scores.is_cuda(), "scores must be CUDA");
    TORCH_CHECK(routed_w_down.is_cuda(), "routed_w_down must be CUDA");
    TORCH_CHECK(routed_w_down_scale.is_cuda(), "routed_w_down_scale must be CUDA");
    TORCH_CHECK(shared_w_down.is_cuda(), "shared_w_down must be CUDA");
    TORCH_CHECK(shared_w_down_scale.is_cuda(), "shared_w_down_scale must be CUDA");
    TORCH_CHECK(output.is_cuda(), "output must be CUDA");

    // [v68-fp16-hidden] hidden_in is now fp16 (was bf16).
    TORCH_CHECK(hidden_in.dtype() == torch::kHalf, "hidden_in must be fp16 (was bf16; v68 now emits fp16)");
    TORCH_CHECK(indices.dtype() == torch::kInt32, "indices must be int32");
    TORCH_CHECK(scores.dtype() == torch::kFloat32, "scores must be fp32");
    TORCH_CHECK(routed_w_down.dtype() == torch::kFloat8_e4m3fn, "routed_w_down must be fp8 e4m3");
    TORCH_CHECK(routed_w_down_scale.dtype() == torch::kFloat32, "routed_w_down_scale must be fp32");
    TORCH_CHECK(shared_w_down.dtype() == torch::kFloat8_e4m3fn, "shared_w_down must be fp8 e4m3");
    TORCH_CHECK(shared_w_down_scale.dtype() == torch::kFloat32, "shared_w_down_scale must be fp32");
    TORCH_CHECK(output.dtype() == torch::kBFloat16, "output must be bf16");

    TORCH_CHECK(hidden_in.dim() == 3, "hidden_in must be [M, 9, K_local]");
    TORCH_CHECK(hidden_in.size(1) == kTopKPlusShared, "hidden_in dim1 must = 9");
    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == kRoutedSlots, "indices must be [M, 8]");
    TORCH_CHECK(scores.dim() == 2 && scores.size(1) == kRoutedSlots, "scores must be [M, 8]");
    TORCH_CHECK(
        routed_w_down.dim() == 3 && routed_w_down.size(0) == kSharedExpertIdx && routed_w_down.size(1) == kHiddenSize,
        "routed_w_down must be [256, 6144, K_local]");
    TORCH_CHECK(routed_w_down_scale.dim() == 3 && routed_w_down_scale.size(0) == kSharedExpertIdx,
        "routed_w_down_scale must be [256, 48, k_blocks]");
    TORCH_CHECK(
        shared_w_down.dim() == 2 && shared_w_down.size(0) == kHiddenSize, "shared_w_down must be [6144, K_local]");
    TORCH_CHECK(shared_w_down_scale.dim() == 2, "shared_w_down_scale must be [48, k_blocks]");

    int const M = static_cast<int>(hidden_in.size(0));
    int const K_local = static_cast<int>(hidden_in.size(2));
    // [dv110-runtimeM] M is now a runtime argument in [1, kMaxM]. Smem
    // buffers are sized for kMaxM (compile-time upper bound); only the
    // active-token loops use the runtime M.
    TORCH_CHECK(M >= 1 && M <= kMaxM, "v110 supports M in [1, ", kMaxM, "]; got M=", M);
    TORCH_CHECK(K_local == 512 || K_local == 256,
        "v110 supports K_local=512 [TP=4] or K_local=256 [TP=8]; got K_local=", K_local);
    TORCH_CHECK(output.dim() == 2 && output.size(0) == M && output.size(1) == kHiddenSize, "output must be [M, 6144]");
    TORCH_CHECK(routed_w_down.size(2) == K_local, "routed_w_down last dim mismatch");
    TORCH_CHECK(shared_w_down.size(1) == K_local, "shared_w_down last dim mismatch");
    int const k_blocks = K_local / kBlockK; // 4 (TP=4) or 2 (TP=8)
    TORCH_CHECK(routed_w_down_scale.size(1) == kHiddenSize / kBlockN, "routed_w_down_scale dim1 must = 48");
    TORCH_CHECK(routed_w_down_scale.size(2) == k_blocks,
        "routed_w_down_scale dim2 must = k_blocks; got=", routed_w_down_scale.size(2), " expected=", k_blocks);
    TORCH_CHECK(shared_w_down_scale.size(0) == kHiddenSize / kBlockN, "shared_w_down_scale dim0 must = 48");
    TORCH_CHECK(shared_w_down_scale.size(1) == k_blocks,
        "shared_w_down_scale dim1 must = k_blocks; got=", shared_w_down_scale.size(1), " expected=", k_blocks);

    TORCH_CHECK(hidden_in.is_contiguous(), "hidden_in must be contiguous");
    TORCH_CHECK(indices.is_contiguous(), "indices must be contiguous");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(routed_w_down.is_contiguous(), "routed_w_down must be contiguous");
    TORCH_CHECK(routed_w_down_scale.is_contiguous(), "routed_w_down_scale must be contiguous");
    TORCH_CHECK(shared_w_down.is_contiguous(), "shared_w_down must be contiguous");
    TORCH_CHECK(shared_w_down_scale.is_contiguous(), "shared_w_down_scale must be contiguous");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

    at::cuda::OptionalCUDAGuard const device_guard(hidden_in.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    int device_id = 0;
    AT_CUDA_CHECK(cudaGetDevice(&device_id));

    // Build or reuse the TMA descriptors for routed and shared down weights.
    CUresult routed_tma_err = CUDA_SUCCESS;
    CUtensorMap routed_w_down_map
        = get_cached_w_down_tmap(routed_w_down.data_ptr(), kSharedExpertIdx, K_local, device_id, &routed_tma_err);
    TORCH_CHECK(routed_tma_err == CUDA_SUCCESS,
        "cuTensorMapEncodeTiled (routed_w_down) failed: CUresult=", (int) routed_tma_err);
    CUresult shared_tma_err = CUDA_SUCCESS;
    CUtensorMap shared_w_down_map
        = get_cached_w_down_tmap(shared_w_down.data_ptr(), 1, K_local, device_id, &shared_tma_err);
    TORCH_CHECK(shared_tma_err == CUDA_SUCCESS,
        "cuTensorMapEncodeTiled (shared_w_down) failed: CUresult=", (int) shared_tma_err);

    // [dv93] kFp8Stages is constexpr 4 (deploy config). Stage selection
    // and the <1>/<2> variants are eliminated. Smem footprint is fixed.
    constexpr int chosen_stages = kSpecFp8Stages;
    auto compute_smem = [&](int stages, int m_for_smem) -> size_t
    {
        size_t hidden_bytes = sizeof(__half) * (size_t) m_for_smem * kTopKPlusShared * K_local;
        size_t tables_bytes = sizeof(int32_t) * (size_t) m_for_smem * kTopKPlusShared
            + sizeof(float) * (size_t) m_for_smem * kTopKPlusShared;
        size_t partial_bytes = sizeof(float) * (size_t) kTopKPlusShared * kRowTilesPerCta * kMmaM * kMaxM;

        size_t bucket_count_base = hidden_bytes + tables_bytes + partial_bytes;
        bucket_count_base = (bucket_count_base + 15) & ~size_t(15);
        size_t bucket_count_bytes = sizeof(int32_t) * (size_t) kNumExpertsTotal;

        size_t bucket_pairs_base = bucket_count_base + bucket_count_bytes;
        bucket_pairs_base = (bucket_pairs_base + 15) & ~size_t(15);
        size_t bucket_pairs_bytes = (size_t) kNumExpertsTotal * kMaxM;

        size_t unique_eid_base = bucket_pairs_base + bucket_pairs_bytes;
        unique_eid_base = (unique_eid_base + 15) & ~size_t(15);
        size_t unique_eid_bytes = sizeof(int16_t) * (size_t) kMaxBuckets;

        size_t num_unique_base = unique_eid_base + unique_eid_bytes;
        num_unique_base = (num_unique_base + 15) & ~size_t(15);
        size_t num_unique_bytes = sizeof(int32_t) * 4;

        size_t mb = num_unique_base + num_unique_bytes;
        mb = (mb + 15) & ~size_t(15);
        size_t mb_bytes = sizeof(uint64_t) * (size_t) (kNumWarps * stages);
        size_t fp8b = mb + mb_bytes;
        fp8b = (fp8b + 1023) & ~size_t(1023);
        size_t fp8b_bytes = (size_t) kNumWarps * stages * kFp8BytesPerStage;
        return fp8b + fp8b_bytes;
    };
    size_t smem_bytes = compute_smem(chosen_stages, M);
    size_t max_smem_bytes = compute_smem(chosen_stages, kMaxM);

    const size_t kSmemCapBytes = 232448; // B200/GB300 maxSharedMemoryPerBlockOptin
    TORCH_CHECK(
        max_smem_bytes <= kSmemCapBytes, "dv110 smem footprint ", max_smem_bytes, " exceeds cap ", kSmemCapBytes);

    using KernelFn = void (*)(const CUtensorMap, const CUtensorMap,
        __half const*, // [v68-fp16-hidden] hidden_in_raw (was bf16)
        int32_t const*, float const*, float const*, float const*, __nv_bfloat16*, int);

    KernelFn kfn = nullptr;
    if (K_local == 512)
    {
        kfn = &mega_down_v110_kernel<512>;
    }
    else if (K_local == 256)
    {
        kfn = &mega_down_v110_kernel<256>;
    }
    else
    {
        TORCH_CHECK(false, "v110 only supports K_local=512 or K_local=256; got K_local=", K_local);
    }

    auto set_smem_attribute_once = [&](std::once_flag& flag)
    {
        std::call_once(flag,
            [&]()
            {
                if (max_smem_bytes > 48 * 1024)
                {
                    cudaError_t set_err = cudaFuncSetAttribute(
                        kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_smem_bytes));
                    TORCH_CHECK(set_err == cudaSuccess,
                        "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=", max_smem_bytes,
                        ") failed: ", cudaGetErrorString(set_err));
                }
            });
    };
    TORCH_CHECK(device_id >= 0 && device_id < kMaxCudaDevicesForSmemAttr, "unsupported CUDA device id ", device_id);
    static std::once_flag s_smem_attr_512_per_device[kMaxCudaDevicesForSmemAttr];
    static std::once_flag s_smem_attr_256_per_device[kMaxCudaDevicesForSmemAttr];
    if (K_local == 512)
    {
        set_smem_attribute_once(s_smem_attr_512_per_device[device_id]);
    }
    else
    {
        set_smem_attribute_once(s_smem_attr_256_per_device[device_id]);
    }

    dim3 grid(kNumCtas, 1, 1);
    dim3 block(kThreadsPerCta, 1, 1);

    void* args[] = {
        (void*) &routed_w_down_map,
        (void*) &shared_w_down_map,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
    };
    // Hidden_in is staged in the kernel with generic LDG instead of TMA.
    // [v68-fp16-hidden] hidden_in dtype is now __half (was __nv_bfloat16).
    __half const* hidden_in_ptr = reinterpret_cast<__half const*>(hidden_in.data_ptr<at::Half>());
    int32_t const* indices_ptr = indices.data_ptr<int32_t>();
    float const* scores_ptr = scores.data_ptr<float>();
    float const* routed_wscale_ptr = routed_w_down_scale.data_ptr<float>();
    float const* shared_wscale_ptr = shared_w_down_scale.data_ptr<float>();
    __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    int m_arg = M;
    args[2] = &hidden_in_ptr;
    args[3] = &indices_ptr;
    args[4] = &scores_ptr;
    args[5] = &routed_wscale_ptr;
    args[6] = &shared_wscale_ptr;
    args[7] = &output_ptr;
    args[8] = &m_arg;

    cudaError_t launch_err = cudaLaunchKernel((void const*) kfn, grid, block, args, smem_bytes, stream);
    TORCH_CHECK(launch_err == cudaSuccess, "dv110 cudaLaunchKernel failed: ", cudaGetErrorString(launch_err));

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}
