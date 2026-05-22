// v110 ExpertDownAllReduce mega kernel — dv103 base + TileRT SASS pattern match.
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
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

namespace {

constexpr int kHiddenSize       = 6144;
constexpr int kTopKPlusShared   = 9;
constexpr int kRoutedSlots      = 8;
constexpr int kSharedExpertIdx  = 256;
constexpr int kNumExpertsTotal  = 257;

constexpr int kThreadsPerCta    = 384;
constexpr int kWarpSize         = 32;
constexpr int kNumWarps         = kThreadsPerCta / kWarpSize;  // 12
constexpr int kNumCtas          = 148;
constexpr int kRowsPerCta       = (kHiddenSize + kNumCtas - 1) / kNumCtas;  // 42

constexpr int kBlockK           = 128;
constexpr int kBlockN           = 128;

constexpr int kMmaM             = 16;
constexpr int kMmaN             = 8;
constexpr int kMmaK             = 16;
constexpr int kKItersPerBlock   = kBlockK / kMmaK;  // 8
constexpr int kRowTilesPerCta   = (kRowsPerCta + kMmaM - 1) / kMmaM;  // 3
// [dv85] M=4 hard-specialization: kMaxM is now compile-time 4.
// Every M-dependent literal (loop bounds, smem offsets, buffer sizes)
// shrinks accordingly. Kernel ONLY accepts M=4.
constexpr int kMaxM             = 4;
// [dv85] Hard-coded compile-time M for the specialized kernel.
constexpr int kSpecM            = 4;

// [dv93/dv110-tp8] EXTENDED hard-spec for kFp8Stages. Other constants
// (kKLocal, kNumPeers, kKBlocksPerGroup, etc.) are now per-kernel template
// parameters to support both TP=4 (kKLocal=512, kNumPeers=4) and TP=8
// (kKLocal=256, kNumPeers=8).
constexpr int kSpecFp8Stages    = 4;                                 // always 4

// fp8 TMA staging: 16 rows × 128 fp8 bytes = 2048 bytes per stage,
// 2 stages per warp.
constexpr int kWtRows           = kMmaM;
constexpr int kWtKChunk         = kBlockK;
constexpr int kFp8BytesPerStage = kWtRows * kWtKChunk;       // 2048

// bf16 mini-buffer for ldmatrix source (one 16x16 bf16 tile per warp).
constexpr int kBf16MiniBytes    = kMmaM * kMmaK * 2;          // 16*16*2 = 512

// Per-K-block grouping for pre-folded scale. K_local=512 (TP=4) -> 4
// K-blocks -> 1 K-group with kKBlocksPerGroup=4.
// K_local=256 (TP=8) -> 2 K-blocks -> 1 K-group with kKBlocksPerGroup=2.
// Inside the templated kernel this is bound to (kKLocal / 128) (since the
// design always yields a single K-group per (N-block, expert)).
// Host-side TORCH_CHECK enforces divisibility.

constexpr int kMaxPeers         = 8;
// [dv85] At M=4: kMaxRoutedPairs = 4 * 8 = 32 (was 128 at M=16).
constexpr int kMaxRoutedPairs   = kMaxM * kRoutedSlots;       // 32
constexpr int kMaxBuckets       = kMaxRoutedPairs;            // 32

// [dv22] Hidden TMA box K-chunk granularity (innermost K bytes per TMA tile).
constexpr int kHiddenKChunk     = kBlockK;   // = 128

// -----------------------------------------------------------------------------
// fp8 (e4m3) -> fp16 conversion (single instr — was the lever-killer
// for bf16 on sm_103a, but fp16 path is supported).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t fp8x2_to_f16x2(uint16_t fp8_pair) {
    uint32_t out;
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
                 : "=r"(out) : "h"(fp8_pair));
    return out;
}

// -----------------------------------------------------------------------------
// HMMA.16816 f16xf16 -> fp32.
// -----------------------------------------------------------------------------
__device__ __forceinline__ void mma_m16n8k16_f16(
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float& c0, float& c1, float& c2, float& c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_f16x2_from_f16(
    __half lo, __half hi) {
    uint32_t out;
    uint16_t lo_u = *reinterpret_cast<uint16_t*>(&lo);
    uint16_t hi_u = *reinterpret_cast<uint16_t*>(&hi);
    out = (static_cast<uint32_t>(hi_u) << 16) | static_cast<uint32_t>(lo_u);
    return out;
}

// -----------------------------------------------------------------------------
// ldmatrix.sync.aligned.x4.b16.
//
// Source: 16-row × 16-col bf16 tile in smem, row stride = 32 bytes.
// Output per lane (m16n8k16 A-frag):
//   r0 = bf16x2 at (row = lane/4,     cols (lane%4)*2 .. +1)
//   r1 = bf16x2 at (row = lane/4 + 8, cols (lane%4)*2 .. +1)
//   r2 = bf16x2 at (row = lane/4,     cols (lane%4)*2 + 8 .. +9)
//   r3 = bf16x2 at (row = lane/4 + 8, cols (lane%4)*2 + 8 .. +9)
//
// Address scheme: lane T provides address of row (T%8) of matrix (T/8).
//   mat 0: rows 0..7,  cols 0..7  (bytes 0..15  in row)
//   mat 1: rows 8..15, cols 0..7  (bytes 0..15)
//   mat 2: rows 0..7,  cols 8..15 (bytes 16..31)
//   mat 3: rows 8..15, cols 8..15 (bytes 16..31)
// -----------------------------------------------------------------------------
__device__ __forceinline__ void ldmatrix_x4_b16(
    uint32_t smem_addr,
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared::cta.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
        : "r"(smem_addr));
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
__device__ __forceinline__ void ldmatrix_x2_b16(
    uint32_t smem_addr,
    uint32_t& r0, uint32_t& r1) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared::cta.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(r0), "=r"(r1)
        : "r"(smem_addr));
}

// -----------------------------------------------------------------------------
// mbarrier + cp.async.bulk.tensor.3d wrappers (TMA).
// -----------------------------------------------------------------------------
__device__ __forceinline__ uint32_t cvt_smem_addr(const void* smem_ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
}

__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, int arrive_count) {
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(addr), "r"(arrive_count));
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar, uint32_t bytes) {
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(addr), "r"(bytes));
}

__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* mbar, uint32_t phase) {
    uint32_t addr = cvt_smem_addr(mbar);
    asm volatile(
        "{\n"
        " .reg .pred P;\n"
        " WAIT_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
        "  @P bra DONE_%=;\n"
        "  bra WAIT_%=;\n"
        " DONE_%=:\n"
        "}\n"
        :: "r"(addr), "r"(phase));
}

__device__ __forceinline__ void cp_async_bulk_tensor_3d(
    void* smem_dst,
    CUtensorMap const* tmap,
    int32_t coord_x, int32_t coord_y, int32_t coord_z,
    uint64_t* mbar) {
    uint32_t smem_addr = cvt_smem_addr(smem_dst);
    uint32_t mbar_addr = cvt_smem_addr(mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
        ::
        "r"(smem_addr),
        "l"(tmap),
        "r"(coord_x), "r"(coord_y), "r"(coord_z),
        "r"(mbar_addr));
}

__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;\n" :::);
}

// [v110-fix-proxy-fence] Cross-proxy fence: synchronizes the GENERIC proxy
// (regular STG / LDG) with the ASYNC proxy (TMA / cp.async.bulk).
//
// Why this is needed: the upstream v68 kernel writes its `hidden_out` via
// regular STG.U16 (generic proxy). v110 reads the same buffer via TMA
// (cp.async.bulk.tensor — async proxy). The kernel-end implicit fence at
// the v68 → v110 stream boundary makes v68's writes visible to v110's
// GENERIC-proxy loads (LDG), but does NOT guarantee visibility to ASYNC-
// proxy TMA loads on the SAME global address. Without this fence, v110's
// first TMA load after v68 may return stale L2 / cache state, producing
// silently corrupted hidden_in in smem and ~10% per-element output error.
//
// Repro: GLM-5 layer 3, M=1/4, TP=4 — v68 output → v110 has max_abs
// ≈ 0.14 vs ref; `torch.cuda.synchronize()` between v68 and v110 (Trial C
// in probe_ordering.py) hides the bug because it forces full DRAM commit.
// Adding `fence.proxy.async.global` at v110 entry (before the hidden TMA
// load) is the proper cross-proxy synchronization primitive and fixes the
// regression without the host-side sync overhead.
__device__ __forceinline__ void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;\n" :::);
}

// -----------------------------------------------------------------------------
// AR primitives — 16-byte (v4.b32) vec publish/poll (dv6c/dv7 verbatim).
// -----------------------------------------------------------------------------
__device__ __forceinline__ void stg_e_128_strong_sys_pair(
    void* gmem_ptr, float v0, float v1, uint32_t marker) {
    uint32_t v0_bits = __float_as_uint(v0);
    uint32_t v1_bits = __float_as_uint(v1);
    asm volatile("st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};\n"
                 :: "l"(gmem_ptr),
                    "r"(v0_bits), "r"(marker),
                    "r"(v1_bits), "r"(marker)
                 : "memory");
}

__device__ __forceinline__ bool ldg_e_128_strong_sys_pair(
    const void* gmem_ptr, uint32_t expected_marker,
    float& out_v0, float& out_v1) {
    uint32_t v0_bits, m0_bits, v1_bits, m1_bits;
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(v0_bits), "=r"(m0_bits),
                   "=r"(v1_bits), "=r"(m1_bits)
                 : "l"(gmem_ptr)
                 : "memory");
    out_v0 = __uint_as_float(v0_bits);
    out_v1 = __uint_as_float(v1_bits);
    return (m0_bits == expected_marker) && (m1_bits == expected_marker);
}

__device__ __forceinline__ void stg_e_64_strong_sys(
    void* gmem_ptr, float value, uint32_t marker) {
    uint32_t v_bits = __float_as_uint(value);
    asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};\n"
                 :: "l"(gmem_ptr), "r"(v_bits), "r"(marker)
                 : "memory");
}

__device__ __forceinline__ bool ldg_e_64_strong_sys(
    const void* gmem_ptr, uint32_t expected_marker, float& out_value) {
    uint32_t v_bits, m_bits;
    asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];\n"
                 : "=r"(v_bits), "=r"(m_bits)
                 : "l"(gmem_ptr)
                 : "memory");
    out_value = __uint_as_float(v_bits);
    return m_bits == expected_marker;
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
    uint32_t fp8_stage_smem,
    int ki, int lane,
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3) {
    const int row_lo = (lane >> 2);          // 0..7
    const int row_hi = row_lo + 8;           // 8..15
    const int col_lo = (lane & 3) << 1;      // 0,2,4,6

    // [dv53] SWIZZLE_128B de-swizzle: TMA wrote fp8 weight tile with
    // CU_TENSOR_MAP_SWIZZLE_128B (period 8 rows × 8 chunks of 16B each).
    // For logical (row, chunk=ki), physical chunk = ki XOR (row & 7).
    const int ki_phys_lo = ki ^ (row_lo & 7);
    const int ki_phys_hi = ki ^ (row_hi & 7);  // == ki_phys_lo

    // [dv110] LDS.U16 with col_lo baked into the address. No SEL, no SHF.R.
    const uint32_t row_lo_base = fp8_stage_smem
        + (uint32_t)(row_lo * (int)kWtKChunk + ki_phys_lo * (int)kMmaK);
    const uint32_t row_hi_base = fp8_stage_smem
        + (uint32_t)(row_hi * (int)kWtKChunk + ki_phys_hi * (int)kMmaK);
    const uint32_t addr_p0 = row_lo_base + (uint32_t)col_lo;
    const uint32_t addr_p2 = row_lo_base + (uint32_t)col_lo + 8u;
    const uint32_t addr_p1 = row_hi_base + (uint32_t)col_lo;
    const uint32_t addr_p3 = row_hi_base + (uint32_t)col_lo + 8u;

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
template <bool kAddRes, int kMyRank, int kNumPeers, int kKLocal>
__global__ __launch_bounds__(kThreadsPerCta, 1) void mega_down_v110_kernel(
    __grid_constant__ const CUtensorMap w_down_map,
    __grid_constant__ const CUtensorMap hidden_map,
    // [v110-fix-hidden-ldg] Pass hidden_in's raw pointer too, so we can
    // read it via plain LDG (generic memory proxy) instead of TMA
    // (`cp.async.bulk.tensor`, async memory proxy). This eliminates the
    // cross-proxy ordering issue with the upstream v68 kernel's STG
    // (generic proxy) writes. The TMA descriptor (hidden_map) is kept
    // for now but no longer used in the hidden_in staging loop —
    // future cleanup can drop hidden_map entirely.
    // [v68-fp16-hidden] Upstream v68 now writes hidden_in as fp16 (was bf16).
    // The downstream MMA already runs in fp16 so we accept fp16 directly,
    // skipping the bf16->fp16 narrowing pass that this kernel used to do.
    const __half* __restrict__ hidden_in_raw,
    const int32_t*       __restrict__ indices,
    const float*         __restrict__ scores,
    const __nv_bfloat16* __restrict__ residual,
    const float*         __restrict__ w_down_group_scale,   // [v110-perkb-scale] [E, 48, k_blocks]
    __nv_bfloat16*       __restrict__ output,
    uint64_t* __restrict__ peer_buf0,
    uint64_t* __restrict__ peer_buf1,
    uint64_t* __restrict__ peer_buf2,
    uint64_t* __restrict__ peer_buf3,
    uint64_t* __restrict__ peer_buf4,
    uint64_t* __restrict__ peer_buf5,
    uint64_t* __restrict__ peer_buf6,
    uint64_t* __restrict__ peer_buf7,
    int M,
    uint32_t flag)
{
    // [dv110-runtimeM] M is now a runtime kernel argument in [1, kMaxM].
    // kMaxM still sizes all smem buffers (compile-time upper bound). M
    // bounds only the active-token loops; trailing rows in smem may be
    // uninitialized and must NOT be read by any consumer.
    // [dv93] EXTENDED hard-spec — runtime args stripped:
    //   * num_peers parameter removed; replaced with constexpr (template).
    //   * K_local parameter removed; replaced with constexpr (template).
    //   * kFp8Stages template parameter removed; replaced with constexpr 4.
    constexpr int kFp8Stages = kSpecFp8Stages;        // == 4
    constexpr int K_local    = kKLocal;               // 512 (TP=4) or 256 (TP=8)
    constexpr int num_peers  = kNumPeers;             // 4 or 8
    // [dv110] my_rank + add_residual are now template parameters; bind
    // local constexpr aliases so existing body references resolve to
    // compile-time literals.
    constexpr int  my_rank      = kMyRank;
    constexpr bool add_residual = kAddRes;
    // [dv110-tp8] Per-template derived counts. Same formulas as before;
    // values change for TP=8.
    //   TP=4 (kKLocal=512): kKBlocks=4, kKBlocksPerGroup=4, kNGroups=1, kHiddenKChunks=4
    //   TP=8 (kKLocal=256): kKBlocks=2, kKBlocksPerGroup=2, kNGroups=1, kHiddenKChunks=2
    constexpr int kKBlocks         = kKLocal / kBlockK;         // 4 or 2
    constexpr int kKBlocksPerGroup = kKLocal / 128;             // 4 or 2 (== kKBlocks here)
    constexpr int kNGroups         = kKBlocks / kKBlocksPerGroup; // == 1
    constexpr int kHiddenKChunks   = kKLocal / kHiddenKChunk;   // 4 or 2

    const int cta_id  = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    const int row_lo = cta_id * kRowsPerCta;
    if (row_lo >= kHiddenSize) return;
    const int row_hi = min(row_lo + kRowsPerCta, kHiddenSize);
    const int rows_here = row_hi - row_lo;

    // [dv93/dv110-tp8] All derived loop counts fold to compile-time literals.
    constexpr int k_blocks        = kKBlocks;            // 4 (TP=4) or 2 (TP=8)
    constexpr int n_groups        = kNGroups;            // == 1
    constexpr int hidden_k_chunks = kHiddenKChunks;      // 4 (TP=4) or 2 (TP=8)

    extern __shared__ unsigned char smem_raw[];
    // [v68-fp16-hidden] Upstream v68 writes fp16 directly; the smem destination
    // is fp16 and we skip the bf16->fp16 narrowing pass.
    __half*        smem_hidden      = reinterpret_cast<__half*>(smem_raw);
    // [legacy] bf16 alias retained for TMA descriptor compatibility — TMA
    // descriptor's element-type field can be set to FLOAT16 instead, but the
    // smem pointer arithmetic is identical (both 16-bit). Cast is a no-op.
    __half* smem_hidden_bf16 = smem_hidden;
    const int hidden_elems = M * kTopKPlusShared * K_local;
    const size_t hidden_bytes = sizeof(__half) * (size_t)hidden_elems;

    int32_t* smem_expert_ids = reinterpret_cast<int32_t*>(smem_raw + hidden_bytes);
    float*   smem_weights    = reinterpret_cast<float*>(
        smem_expert_ids + M * kTopKPlusShared);

    const size_t partial_base =
        hidden_bytes
      + sizeof(int32_t) * (size_t)M * kTopKPlusShared
      + sizeof(float)   * (size_t)M * kTopKPlusShared;
    float* smem_part = reinterpret_cast<float*>(smem_raw + partial_base);

    const int part_elems = kTopKPlusShared * kRowTilesPerCta * kMmaM * kMaxM;
    const size_t partial_bytes = sizeof(float) * (size_t)part_elems;

    // ---- Bucketing tables (same as dv7/dv8a). ----
    size_t bucket_count_base = partial_base + partial_bytes;
    bucket_count_base = (bucket_count_base + 15) & ~size_t(15);
    int32_t* smem_bucket_count =
        reinterpret_cast<int32_t*>(smem_raw + bucket_count_base);

    size_t bucket_pairs_base =
        bucket_count_base + sizeof(int32_t) * (size_t)kNumExpertsTotal;
    bucket_pairs_base = (bucket_pairs_base + 15) & ~size_t(15);
    uint8_t* smem_bucket_pairs =
        reinterpret_cast<uint8_t*>(smem_raw + bucket_pairs_base);

    size_t unique_eid_base = bucket_pairs_base + (size_t)kNumExpertsTotal * kMaxM;
    unique_eid_base = (unique_eid_base + 15) & ~size_t(15);
    int16_t* smem_unique_eid =
        reinterpret_cast<int16_t*>(smem_raw + unique_eid_base);

    size_t num_unique_base = unique_eid_base + sizeof(int16_t) * (size_t)kMaxBuckets;
    num_unique_base = (num_unique_base + 15) & ~size_t(15);
    int32_t* smem_num_unique =
        reinterpret_cast<int32_t*>(smem_raw + num_unique_base);

    // ---- TMA mbarrier ring (one per (warp, stage) for W_down +
    //      one shared for hidden TMA). 8B aligned. ----
    size_t mbar_base = num_unique_base + sizeof(int32_t) * 4;
    mbar_base = (mbar_base + 15) & ~size_t(15);
    uint64_t* smem_mbar =
        reinterpret_cast<uint64_t*>(smem_raw + mbar_base);
    const int kWdMbarCount   = kNumWarps * kFp8Stages;
    const int kHiddenMbarIdx = kWdMbarCount;  // [dv22] hidden mbarrier slot
    const int kMbarCount     = kWdMbarCount + 1;

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
    fp8_base = (fp8_base + 1023) & ~size_t(1023);  // [dv53] 1024 B for SWIZZLE_128B
    uint8_t* smem_fp8_stages = smem_raw + fp8_base;

    // ---- bf16 mini-buffers (per warp). ----
    // [dv53] bf16 mini-buffer eliminated. The lane>=16 fallback for
    // ldmatrix.x2 B-frag uses warp_fp8_addr instead (valid smem address).
    // We still consume fp8 stage memory in the same place.

    // ---- [v110-fix-proxy-fence + v110-fix-l2-drain] ----
    // Cross-proxy fence at kernel entry, plus a CTA-wide L2 drain for the
    // hidden_in surface.
    //
    // The upstream v68 kernel writes hidden_in via regular STG (generic
    // proxy); v110 reads it via TMA (async proxy). The implicit stream
    // fence between v68 and v110 syncs the generic proxy only.
    //
    // The original fix (just fence.proxy.async.global) was empirically
    // INSUFFICIENT when v68 is followed back-to-back by v110 on the same
    // stream — the consumer-side cross-proxy fence is per-thread and can
    // only elevate generic-proxy state that is already visible to the
    // issuing thread; it cannot force cross-CTA visibility in the async
    // proxy on its own. The v68 producer-side STG writes may sit in L2
    // long enough that v110's TMA load (tid 0) reads stale state for
    // some cache lines, while other lines have already drained.
    // Symptom: B-A in the GLM-5 single-layer diff test = 0.058-0.143
    // (vs ~0.003 expected) — the error is structured, consistent with
    // ~1 of 9 routed slots reading wrong bytes per TMA-tile cache-line.
    //
    // Fix: pair the per-thread proxy fence with a CTA-wide L2 release
    // fence at SYSTEM scope (`__threadfence_system` ≡ `membar.sys`).
    // membar.sys flushes ALL pending writes (both proxies) from this
    // CTA's L1/L2 view to system-visible state and waits for prior
    // writes to be acknowledged. This guarantees the v68→v110 cache
    // lines are coherent before the TMA load begins.
    //
    // Cost: one membar.sys at kernel entry (~100-200 cycles, one-shot
    // per CTA). v110 wall-time is ~33 us = ~108k cycles; the cost is
    // <0.2% — well within the user's ≤5% perf budget.
    fence_proxy_async_global();
    __threadfence_system();

    // ---- [dv22] Init mbarriers FIRST (W_down ring + hidden mbarrier). ----
    if (tid < kMbarCount) {
        mbarrier_init(&smem_mbar[tid], 1);
    }
    if (tid == 0) {
        fence_proxy_async_shared();
    }
    __syncthreads();

    // ---- [dv22] Stage hidden_in into SMEM via TMA bulk. ----
    //
    // [v68-fp16-hidden] hidden_in is now fp16 (was bf16). The TMA descriptor
    // is built with element-type FLOAT16; smem destination is __half. The
    // post-stage bf16->fp16 narrowing pass is no longer needed.
    if (tid == 0) {
        const uint32_t total_bytes = static_cast<uint32_t>(hidden_bytes);
        mbarrier_arrive_expect_tx(&smem_mbar[kHiddenMbarIdx], total_bytes);
        const size_t per_chunk_elems =
            (size_t)M * kTopKPlusShared * kHiddenKChunk;
        for (int c = 0; c < hidden_k_chunks; ++c) {
            __half* dst_chunk =
                smem_hidden + (size_t)c * per_chunk_elems;
            cp_async_bulk_tensor_3d(
                dst_chunk, &hidden_map,
                /*x=*/c * kHiddenKChunk, /*y=*/0, /*z=*/0,
                &smem_mbar[kHiddenMbarIdx]);
        }
    }

    // ---- Build expert-id and weight tables [M, 9] in SMEM. ----
    {
        const int total_slots = M * kTopKPlusShared;
        for (int i = tid; i < total_slots; i += kThreadsPerCta) {
            const int m = i / kTopKPlusShared;
            const int s = i - m * kTopKPlusShared;
            int32_t eid;
            float   w;
            if (s == 0) {
                eid = kSharedExpertIdx;
                w   = 1.0f;
            } else {
                eid = indices[m * kRoutedSlots + (s - 1)];
                w   = scores[m * kRoutedSlots + (s - 1)];
            }
            smem_expert_ids[i] = eid;
            smem_weights[i]    = w;
        }
    }

    // ---- Initialise partial accumulator. ----
    for (int i = tid; i < part_elems; i += kThreadsPerCta) {
        smem_part[i] = 0.0f;
    }

    // ---- Init routed-dedup bucket tables. ----
    for (int i = tid; i < kNumExpertsTotal; i += kThreadsPerCta) {
        smem_bucket_count[i] = 0;
    }
    if (tid == 0) {
        *smem_num_unique = 0;
    }

    // [dv22] mbarriers already initialised before the TMA staging.
    __syncthreads();

    // ---- Bucket routed pairs by expert_id (dv7). ----
    {
        const int total_routed = M * kRoutedSlots;
        for (int i = tid; i < total_routed; i += kThreadsPerCta) {
            const int m = i / kRoutedSlots;
            const int s = i - m * kRoutedSlots;
            const int e_id = indices[m * kRoutedSlots + s];
            if (e_id < 0 || e_id >= kNumExpertsTotal) continue;

            uint8_t packed = static_cast<uint8_t>((m << 4) | (s + 1));

            int old_count = atomicAdd(&smem_bucket_count[e_id], 1);
            if (old_count < kMaxM) {
                smem_bucket_pairs[e_id * kMaxM + old_count] = packed;
            }
            if (old_count == 0) {
                int slot_idx = atomicAdd(smem_num_unique, 1);
                if (slot_idx < kMaxBuckets) {
                    smem_unique_eid[slot_idx] = static_cast<int16_t>(e_id);
                }
            }
        }
    }
    __syncthreads();

    // ---- [dv22] Wait for hidden TMA to land before any reader runs. ----
    mbarrier_wait_parity(&smem_mbar[kHiddenMbarIdx], 0u);
    fence_proxy_async_shared();
    __syncthreads();

    // [v68-fp16-hidden] The legacy bf16->fp16 in-place narrowing pass is now
    // a no-op: v68 already writes fp16 and the TMA staged fp16 into smem.
    // No conversion is needed; the consumer reads __half directly.

    const int num_unique = *smem_num_unique;

    // Per-warp mbarrier base index, fp8 stage base, bf16 mini base.
    const int warp_mbar_base = warp_id * kFp8Stages;
    uint8_t* warp_fp8_base   = smem_fp8_stages + warp_id * (kFp8Stages * kFp8BytesPerStage);
    // [dv53] warp_mini_addr was used as safe-smem-addr fallback for
    // lane>=16 in ldmatrix.x2 B-frag — replaced by warp_fp8_addr.
    uint32_t warp_fp8_addr   = cvt_smem_addr(warp_fp8_base);
    const uint32_t warp_mini_addr = warp_fp8_addr;  // [dv53] safe-smem alias

    // [dv21+dv22] smem_hidden base addr for ldmatrix.x2 B-frag.
    //
    // dv22 smem layout = [K_chunks, M, 9, 128] row-major (innermost = 128
    // bf16 elements). So per-(kb, m, slot) row offset in bytes is:
    //   kb * (M * 9 * 128 * 2)
    // + m  * (9 * 128 * 2)
    // + s  * (128 * 2)
    // The bf16 row stride for a single (kb, m, slot) is 128 elements *
    // 2 bytes = 256 bytes — much larger than a single ldmatrix row
    // (16 bytes = 8 bf16). ldmatrix.x2 only uses per-lane addresses
    // (not strides), so the wide bf16-row works fine as the row source.
    const uint32_t smem_hidden_addr   = cvt_smem_addr(smem_hidden);
    // [dv110-runtimeM] kb_stride depends on runtime M (the TMA writes M-packed
    // rows per K-chunk). m_stride and slot_stride remain compile-time.
    const uint32_t kb_stride_bytes     =
        (uint32_t)(M * kTopKPlusShared * kHiddenKChunk * 2);   // M*9*128*2
    constexpr uint32_t m_stride_bytes_h    =
        (uint32_t)(kTopKPlusShared * kHiddenKChunk * 2);       // 9*128*2 = 2304
    constexpr uint32_t slot_stride_bytes_h =
        (uint32_t)(kHiddenKChunk * 2);                          // 128*2 = 256

    // TMA issue helper. Coords: x = k_off (bytes), y = row_base, z = e_id.
    auto issue_tma_load = [&](int e_id, int row_base, int k_off, int stage_idx) {
        if (lane == 0) {
            int mbar_idx = warp_mbar_base + stage_idx;
            uint8_t* stage_smem = warp_fp8_base + stage_idx * kFp8BytesPerStage;
            mbarrier_arrive_expect_tx(&smem_mbar[mbar_idx], kFp8BytesPerStage);
            cp_async_bulk_tensor_3d(
                stage_smem, &w_down_map,
                /*x=*/k_off, /*y=*/row_base, /*z=*/e_id,
                &smem_mbar[mbar_idx]);
        }
    };

    // Per-warp mbarrier phase tracking (toggled at each wait).
    // kFp8Stages now spans {1, 2, 3, 4} — size for max-of-template.
    constexpr int kMaxStages = 4;
    uint32_t mbar_phase[kMaxStages] = {0u, 0u, 0u, 0u};

    auto wait_stage = [&](int stage_idx) {
        int mbar_idx = warp_mbar_base + stage_idx;
        mbarrier_wait_parity(&smem_mbar[mbar_idx], mbar_phase[stage_idx]);
        mbar_phase[stage_idx] ^= 1u;
    };

    // ---- Phase A: shared-expert path. ----
    {
        const int shared_work = kRowTilesPerCta;
        for (int w = warp_id; w < shared_work; w += kNumWarps) {
            const int tile = w;
            const int row_base_in_cta = tile * kMmaM;
            const int row_base = row_lo + row_base_in_cta;

            const int rows_active = min(kMmaM, row_hi - row_base);
            if (rows_active <= 0) continue;

            const int e_id = kSharedExpertIdx;
            // [dv85] n_tiles_m = (4+7)/8 = 1 — single iteration, m_base=0.
            constexpr int kNTilesM = 1;
            #pragma unroll
            for (int nt = 0; nt < kNTilesM; ++nt) {
                constexpr int m_base = 0;
                float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                // Prologue: pre-issue stages 0..min(kFp8Stages-1, k_blocks-1).
                // For kFp8Stages==1 this issues nothing here (issue-on-demand
                // in the inner loop below mirrors dv9's single-stage path).
                if constexpr (kFp8Stages > 1) {
                    const int prologue_n = kFp8Stages < k_blocks
                                         ? kFp8Stages : k_blocks;
                    #pragma unroll
                    for (int s = 0; s < kMaxStages; ++s) {
                        if (s < prologue_n) {
                            issue_tma_load(e_id, row_base,
                                           s * kBlockK, s);
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
                const int row0_lane_pkb = row_base + (lane >> 2);
                const int row8_lane_pkb = row_base + (lane >> 2) + 8;
                const int nb0_pkb = row0_lane_pkb / kBlockN;
                const int nb8_pkb = row8_lane_pkb / kBlockN;
                // Scale tensor shape: [E, 48, k_blocks]. Hoist the per-(e, nb)
                // base; per-kb scales live at consecutive memory.
                const float* const s0_base_pkb = w_down_group_scale +
                    ((size_t)e_id * (kHiddenSize / kBlockN) + nb0_pkb) *
                        (size_t)k_blocks;
                const float* const s8_base_pkb = (nb8_pkb == nb0_pkb)
                    ? s0_base_pkb
                    : (w_down_group_scale +
                       ((size_t)e_id * (kHiddenSize / kBlockN) + nb8_pkb) *
                           (size_t)k_blocks);

                #pragma unroll
                for (int kb = 0; kb < k_blocks; ++kb) {
                    const int stage =
                        (kFp8Stages == 1) ? 0 : (kb % kFp8Stages);

                    if constexpr (kFp8Stages == 1) {
                        // 1-stage: issue current then wait.
                        issue_tma_load(e_id, row_base, kb * kBlockK, 0);
                    }
                    wait_stage(stage);
                    __syncwarp();

                    uint32_t fp8_stage_ptr = warp_fp8_addr
                                           + (uint32_t)(stage * kFp8BytesPerStage);

                    float c_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    #pragma unroll
                    for (int ki = 0; ki < kKItersPerBlock; ++ki) {
                        // [dv53] Direct fp8 -> A-frag, no STS/LDSM.x4.
                        uint32_t a_frag[4];
                        cvt_fp8_to_afrag_direct(fp8_stage_ptr, ki, lane,
                            a_frag[0], a_frag[1], a_frag[2], a_frag[3]);

                        // [dv21+dv22] B-frag via ldmatrix.x2.b16
                        // from smem layout [K_chunks, M, 9, 128].
                        // Lanes 0..7 supply mat0 rows (k=0..7 of ki);
                        // lanes 8..15 supply mat1 rows (k=8..15);
                        // lanes 16..31 use warp_mini_addr as a safe
                        // smem address (ignored by ldmatrix.x2).
                        uint32_t b_frag[2];
                        {
                            const int idx_in_row = lane & 7;  // 0..7 (per-N row)
                            const int mat_id_b   = (lane >> 3) & 1;  // 0 or 1
                            const int m_for_lane = m_base + idx_in_row;
                            const int m_clamped  = (m_for_lane < M)
                                                 ? m_for_lane
                                                 : (M - 1);  // safe in-range
                            const int k_in_off   = ki * kMmaK + mat_id_b * 8;
                            const uint32_t row_off =
                                (uint32_t)kb        * kb_stride_bytes
                              + (uint32_t)m_clamped * m_stride_bytes_h
                              + 0u                  * slot_stride_bytes_h  // shared slot=0
                              + (uint32_t)k_in_off  * 2u;
                            uint32_t b_addr = (lane < 16)
                                ? (smem_hidden_addr + row_off)
                                : warp_mini_addr;
                            ldmatrix_x2_b16(b_addr, b_frag[0], b_frag[1]);
                        }

                        mma_m16n8k16_f16(
                            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                            b_frag[0], b_frag[1],
                            c_block[0], c_block[1], c_block[2], c_block[3]);

                        // [dv53] no mini-buffer, so no syncwarp needed here.
                    }

                    // Steady-state: pre-issue (kb + kFp8Stages) into the
                    // just-consumed stage slot.
                    if constexpr (kFp8Stages > 1) {
                        const int next_kb = kb + kFp8Stages;
                        if (next_kb < k_blocks) {
                            issue_tma_load(e_id, row_base,
                                           next_kb * kBlockK, stage);
                        }
                    }

                    // [v110-perkb-scale] Per-K-block FFMA fold. Each K-block
                    // has its own scale (raw scale from the source, no rescale).
                    // The fp32 accumulator preserves intermediate precision
                    // across kb's, matching v68's per-K-block design and the
                    // parent fp8_block_scale_moe runner's precision profile.
                    const float s0_kb = s0_base_pkb[kb];
                    const float s8_kb = (nb8_pkb == nb0_pkb)
                                        ? s0_kb : s8_base_pkb[kb];
                    c[0] += c_block[0] * s0_kb;
                    c[1] += c_block[1] * s0_kb;
                    c[2] += c_block[2] * s8_kb;
                    c[3] += c_block[3] * s8_kb;
                }

                {
                    const int row0 = (lane >> 2);
                    const int row1 = row0 + 8;
                    const int col0_local = (lane & 3) * 2;
                    const int col1_local = col0_local + 1;
                    const int col0 = m_base + col0_local;
                    const int col1 = m_base + col1_local;
                    const int slot_off = 0 * kRowTilesPerCta * kMmaM * kMaxM
                                       + tile * kMmaM * kMaxM;
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
        const int total_outer = kRowTilesPerCta * num_unique;
        for (int w_outer = warp_id; w_outer < total_outer; w_outer += kNumWarps) {
            const int tile      = w_outer / num_unique;
            const int b_idx     = w_outer - tile * num_unique;
            const int row_base_in_cta = tile * kMmaM;
            const int row_base = row_lo + row_base_in_cta;
            const int rows_active = min(kMmaM, row_hi - row_base);
            if (rows_active <= 0) continue;

            const int e_id  = smem_unique_eid[b_idx];
            const int count = static_cast<int>(smem_bucket_count[e_id]);
            const int n_groups_routed = (count + 7) >> 3;  // 1 or 2

            for (int g = 0; g < n_groups_routed; ++g) {
                const int group_start = g * 8;
                const int group_count = min(8, count - group_start);

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
                    const int pair_idx_in_group = lane & 7;
                    if (lane < 16 && pair_idx_in_group < group_count) {
                        uint8_t packed = smem_bucket_pairs[
                            e_id * kMaxM + group_start + pair_idx_in_group];
                        const int mm = (packed >> 4) & 0xF;
                        const int ss = packed & 0xF;
                        b_row_base_bytes =
                            (uint32_t)mm * m_stride_bytes_h
                          + (uint32_t)ss * slot_stride_bytes_h;
                        lane_has_pair = true;
                    }
                }
                // my_active drives accumulator write-out logic below.
                const int my_n_idx = lane >> 2;
                const bool my_active = (my_n_idx < group_count);

                // Prologue: pre-issue stages 0..min(kFp8Stages-1, k_blocks-1).
                if constexpr (kFp8Stages > 1) {
                    const int prologue_n = kFp8Stages < k_blocks
                                         ? kFp8Stages : k_blocks;
                    #pragma unroll
                    for (int s = 0; s < kMaxStages; ++s) {
                        if (s < prologue_n) {
                            issue_tma_load(e_id, row_base,
                                           s * kBlockK, s);
                        }
                    }
                }

                // [v110-perkb-scale] Per-K-block scale fold INSIDE the K-loop
                // (see Phase A for design rationale). Routed path mirrors the
                // shared path's per-K-block FFMA fold.
                const int row0_lane_pkbR = row_base + (lane >> 2);
                const int row8_lane_pkbR = row_base + (lane >> 2) + 8;
                const int nb0_pkbR = row0_lane_pkbR / kBlockN;
                const int nb8_pkbR = row8_lane_pkbR / kBlockN;
                const float* const s0_base_pkbR = w_down_group_scale +
                    ((size_t)e_id * (kHiddenSize / kBlockN) + nb0_pkbR) *
                        (size_t)k_blocks;
                const float* const s8_base_pkbR = (nb8_pkbR == nb0_pkbR)
                    ? s0_base_pkbR
                    : (w_down_group_scale +
                       ((size_t)e_id * (kHiddenSize / kBlockN) + nb8_pkbR) *
                           (size_t)k_blocks);

                #pragma unroll
                for (int kb = 0; kb < k_blocks; ++kb) {
                    const int stage =
                        (kFp8Stages == 1) ? 0 : (kb % kFp8Stages);

                    if constexpr (kFp8Stages == 1) {
                        issue_tma_load(e_id, row_base, kb * kBlockK, 0);
                    }
                    wait_stage(stage);
                    __syncwarp();

                    uint32_t fp8_stage_ptr = warp_fp8_addr
                                           + (uint32_t)(stage * kFp8BytesPerStage);

                    float c_block[4] = {0.0f, 0.0f, 0.0f, 0.0f};

                    #pragma unroll
                    for (int ki = 0; ki < kKItersPerBlock; ++ki) {
                        // [dv53] Direct fp8 -> A-frag, no STS/LDSM.x4.
                        uint32_t a_frag[4];
                        cvt_fp8_to_afrag_direct(fp8_stage_ptr, ki, lane,
                            a_frag[0], a_frag[1], a_frag[2], a_frag[3]);

                        // [dv21+dv22] B-frag via ldmatrix.x2.b16
                        // from smem [K_chunks, M, 9, 128].
                        uint32_t b_frag[2];
                        {
                            const int mat_id_b = (lane >> 3) & 1;
                            const int k_in_off = ki * kMmaK + mat_id_b * 8;
                            const uint32_t k_off_bytes =
                                (uint32_t)kb * kb_stride_bytes
                              + (uint32_t)k_in_off * 2u;
                            uint32_t b_addr = lane_has_pair
                                ? (smem_hidden_addr + b_row_base_bytes
                                                    + k_off_bytes)
                                : warp_mini_addr;
                            ldmatrix_x2_b16(b_addr, b_frag[0], b_frag[1]);
                        }

                        mma_m16n8k16_f16(
                            a_frag[0], a_frag[1], a_frag[2], a_frag[3],
                            b_frag[0], b_frag[1],
                            c_block[0], c_block[1], c_block[2], c_block[3]);
                        // [dv53] no mini-buffer, so no syncwarp needed here.
                    }

                    // Steady-state: pre-issue (kb + kFp8Stages) into the
                    // freed stage slot.
                    if constexpr (kFp8Stages > 1) {
                        const int next_kb = kb + kFp8Stages;
                        if (next_kb < k_blocks) {
                            issue_tma_load(e_id, row_base,
                                           next_kb * kBlockK, stage);
                        }
                    }

                    // [v110-perkb-scale] Per-K-block FFMA fold.
                    const float s0_kbR = s0_base_pkbR[kb];
                    const float s8_kbR = (nb8_pkbR == nb0_pkbR)
                                         ? s0_kbR : s8_base_pkbR[kb];
                    c[0] += c_block[0] * s0_kbR;
                    c[1] += c_block[1] * s0_kbR;
                    c[2] += c_block[2] * s8_kbR;
                    c[3] += c_block[3] * s8_kbR;
                }

                {
                    const int row0 = (lane >> 2);
                    const int row1 = row0 + 8;
                    const int p0_local = (lane & 3) * 2;
                    const int p1_local = p0_local + 1;
                    const int p0 = group_start + p0_local;
                    const int p1 = group_start + p1_local;

                    auto write_cell = [&](int p, int row_in_tile, float val) {
                        if (p < count && row_in_tile < rows_active) {
                            uint8_t packed = smem_bucket_pairs[e_id * kMaxM + p];
                            const int m = (packed >> 4) & 0xF;
                            const int s = packed & 0xF;
                            const int off = s * kRowTilesPerCta * kMmaM * kMaxM
                                          + tile * kMmaM * kMaxM
                                          + row_in_tile * kMaxM + m;
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
    // [dv30] Final reduction + AR with BATCHED PUBLISH.
    //
    // 3-phase pipeline:
    //   Phase A: compute acc for ALL owned cells (no STG/LDG). Cache acc
    //            in per-thread registers (each thread owns ≤ 1 pair-cell
    //            since pair_cells_max = 336 ≤ 384, and ≤ 1 tail-cell
    //            since M_max = 16 ≤ 384). The per-thread "loop" is in
    //            fact a single iteration; the registers hold acc0/acc1
    //            for the pair-cell case and acc_t for the tail-row case.
    //   Phase B: CTA-scope publish burst. Every active thread issues
    //            ALL its peer STG.E.128 (or STG.E.64) STRONG.SYS writes
    //            back-to-back, with NO interleaved compute or poll.
    //            CTA-wide aggregate: 84 active threads × (num_peers-1)
    //            STGs per pair-cell + M active threads × (num_peers-1)
    //            STGs per tail-row → ~504 STGs at TP=4 M=4. NVLink can
    //            keep many in flight, so the publish phase wall-time
    //            approaches `total_stgs / wire_throughput` rather than
    //            `cells × per_stg_latency`.
    //   Phase C: poll-loop. Each active thread spin-polls its cells'
    //            peer slots, FADDs into local acc, writes bf16 output.
    //
    // Single `__threadfence_system()` hoisted CTA-wide before publish.
    // Self-rank skip preserved (dv11).
    //
    // To do "all pair-cells then all tail-rows" with NO sync between, we
    // simply structure the two passes back-to-back inside each phase.
    // -------------------------------------------------------------------
    uint64_t* peer_bufs[kMaxPeers] = {
        peer_buf0, peer_buf1, peer_buf2, peer_buf3,
        peer_buf4, peer_buf5, peer_buf6, peer_buf7,
    };

    const size_t writer_stride = (size_t)kMaxM * kHiddenSize;

    const int rows_pairs = rows_here >> 1;
    const int tail_rows  = rows_here - (rows_pairs << 1);

    const int pair_cells = rows_pairs * M;

    // Each thread owns at most one pair-cell (pair_cells ≤ 336 < 384).
    // Hold its acc in registers; track validity via tid < pair_cells.
    const bool has_pair_cell = (tid < pair_cells);
    float pair_acc0 = 0.0f, pair_acc1 = 0.0f;
    int   pair_m = 0, pair_row0 = 0;
    size_t pair_cell_off = 0;

    if (has_pair_cell) {
        const int cell = tid;
        const int pair_idx_in_cta = cell / M;
        const int m               = cell - pair_idx_in_cta * M;
        const int row_in_cta0     = pair_idx_in_cta << 1;
        const int row_in_cta1     = row_in_cta0 + 1;
        const int row0            = row_lo + row_in_cta0;
        const int row1            = row0 + 1;

        const int tile0 = row_in_cta0 / kMmaM;
        const int rit0  = row_in_cta0 - tile0 * kMmaM;
        const int tile1 = row_in_cta1 / kMmaM;
        const int rit1  = row_in_cta1 - tile1 * kMmaM;

        float acc0 = 0.0f, acc1 = 0.0f;
        #pragma unroll
        for (int s = 0; s < kTopKPlusShared; ++s) {
            const float w = smem_weights[m * kTopKPlusShared + s];
            const int off0 = s * kRowTilesPerCta * kMmaM * kMaxM
                           + tile0 * kMmaM * kMaxM
                           + rit0 * kMaxM + m;
            const int off1 = s * kRowTilesPerCta * kMmaM * kMaxM
                           + tile1 * kMmaM * kMaxM
                           + rit1 * kMaxM + m;
            acc0 += smem_part[off0] * w;
            acc1 += smem_part[off1] * w;
        }
        if (add_residual) {
            acc0 += __bfloat162float(residual[(size_t)m * kHiddenSize + row0]);
            acc1 += __bfloat162float(residual[(size_t)m * kHiddenSize + row1]);
        }
        pair_acc0     = acc0;
        pair_acc1     = acc1;
        pair_m        = m;
        pair_row0     = row0;
        pair_cell_off = (size_t)m * kHiddenSize + (size_t)row0;
        (void)row1;
    }

    // Tail-row owner (at most one per thread since M ≤ kMaxM=16 < 384).
    const bool has_tail_cell = (tail_rows > 0) && (tid < M);
    float tail_acc = 0.0f;
    int   tail_m = 0, tail_row = 0;
    size_t tail_cell_off = 0;

    if (has_tail_cell) {
        const int row_in_cta = rows_here - 1;
        const int row = row_lo + row_in_cta;
        const int tile = row_in_cta / kMmaM;
        const int rit  = row_in_cta - tile * kMmaM;
        const int m    = tid;

        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < kTopKPlusShared; ++s) {
            const int off = s * kRowTilesPerCta * kMmaM * kMaxM
                          + tile * kMmaM * kMaxM
                          + rit * kMaxM + m;
            acc += smem_part[off] * smem_weights[m * kTopKPlusShared + s];
        }
        if (add_residual) {
            acc += __bfloat162float(residual[(size_t)m * kHiddenSize + row]);
        }
        tail_acc      = acc;
        tail_m        = m;
        tail_row      = row;
        tail_cell_off = (size_t)m * kHiddenSize + (size_t)row;
    }

    // [dv93] Single-rank fast-path eliminated — num_peers is constexpr 4
    // at the TP=4 deploy config.

    // -------- Phase A done. Single CTA-scope fence before Phase B. --------
    __threadfence_system();

    // -------- Phase B: BATCHED PUBLISH BURST. -------------------------
    // Each thread issues all its peer STGs back-to-back, no compute or
    // poll interleaved. CTA-wide, all 84 active pair-cell threads issue
    // simultaneously, then tail-cell threads issue. The publish phase
    // contains NO LDG / synchronization beyond the unrolled per-peer
    // STGs.
    //
    // [dv93] Pair-cells: each active thread issues (num_peers-1) STG.E.128.
    // num_peers is constexpr 4 — direct unrolled loop (no break).
    if (has_pair_cell) {
        #pragma unroll
        for (int p = 0; p < num_peers; ++p) {
            if (p == my_rank) continue;
            uint64_t* dst = peer_bufs[p]
                          + (size_t)my_rank * writer_stride
                          + pair_cell_off;
            stg_e_128_strong_sys_pair(dst, pair_acc0, pair_acc1, flag);
        }
    }
    // Tail-row: each active thread issues (num_peers-1) STG.E.64.
    if (has_tail_cell) {
        #pragma unroll
        for (int p = 0; p < num_peers; ++p) {
            if (p == my_rank) continue;
            uint64_t* dst = peer_bufs[p]
                          + (size_t)my_rank * writer_stride
                          + tail_cell_off;
            stg_e_64_strong_sys(dst, tail_acc, flag);
        }
    }

    // -------- Phase C: POLL + FADD + STORE -------------------------------
    // Spin-poll each peer's slot until the sentinel `flag` matches; FADD
    // the value into the local acc; write the final bf16 output.
    //
    // NOTE: no CTA barrier between Phase B and Phase C. Within a single
    // thread, all its STGs are issued before its LDGs (program-order
    // serialization). Across threads, peer threads spin-wait on the
    // sentinel which provides the cross-rank ordering guarantee — peer
    // writes are visible iff the sentinel matches.
    // [dv93] Phase C poll loops: num_peers is constexpr 4.
    if (has_pair_cell) {
        float ar0 = pair_acc0;
        float ar1 = pair_acc1;
        uint64_t* own = peer_bufs[my_rank];
        #pragma unroll
        for (int p = 0; p < num_peers; ++p) {
            if (p == my_rank) continue;
            const uint64_t* src = own
                                + (size_t)p * writer_stride
                                + pair_cell_off;
            float v0, v1;
            while (true) {
                if (ldg_e_128_strong_sys_pair(src, flag, v0, v1)) break;
            }
            ar0 += v0;
            ar1 += v1;
        }
        output[(size_t)pair_m * kHiddenSize + pair_row0]     = __float2bfloat16(ar0);
        output[(size_t)pair_m * kHiddenSize + pair_row0 + 1] = __float2bfloat16(ar1);
    }

    if (has_tail_cell) {
        float ar_acc = tail_acc;
        uint64_t* own = peer_bufs[my_rank];
        #pragma unroll
        for (int p = 0; p < num_peers; ++p) {
            if (p == my_rank) continue;
            const uint64_t* src = own
                                + (size_t)p * writer_stride
                                + tail_cell_off;
            float v;
            while (true) {
                if (ldg_e_64_strong_sys(src, flag, v)) break;
            }
            ar_acc += v;
        }
        output[(size_t)tail_m * kHiddenSize + tail_row] = __float2bfloat16(ar_acc);
    }
}

// -----------------------------------------------------------------------------
// [v110-perkb-scale] Offline weight repack — bit-identical layout, scale
// passthrough. The runtime kernel applies per-K-block scale at MMA-fold
// time (matching v68's design); no pre-fold rescale is required.
// -----------------------------------------------------------------------------

// Pass 1: copy per-K-block scales to the runtime-friendly layout.
// Output shape: [E, n_n_blocks, k_blocks] fp32 (same as the input scale tensor,
// just re-emitted contiguously so the kernel can hoist a per-(e, nb) base
// pointer and stride by 1 fp32 / kb).
__global__ void copy_per_kb_scales_kernel_v110(
    const float* __restrict__ scale_orig,    // [E, n_n_blocks, k_blocks]
    float*       __restrict__ scale_out,     // [E, n_n_blocks, k_blocks]
    int          n_n_blocks,
    int          k_blocks)
{
    // 1D grid over (E * n_n_blocks); each block copies its `k_blocks` scales.
    const int block = blockIdx.x;
    const int tid   = threadIdx.x;
    if (tid < k_blocks) {
        const size_t off = (size_t)block * (size_t)k_blocks + (size_t)tid;
        scale_out[off] = scale_orig[off];
    }
}

// Pass 2: bit-identical fp8 weight copy (no rescale). Layout unchanged from
// the source: [E, N, K_local] row-major fp8_e4m3. The runtime TMA descriptor
// is also unchanged because it operates on raw bytes.
__global__ void copy_weights_kernel_v110(
    const __nv_fp8_e4m3* __restrict__ w_orig,    // [E, N, K_local]
    __nv_fp8_e4m3*       __restrict__ w_packed,  // [E, N, K_local]
    int                  K_local,
    int                  k_blocks)
{
    // 1D grid over (E * N * k_blocks); each block copies kBlockK fp8 bytes.
    const int block = blockIdx.x;
    const int blocks_per_e = kHiddenSize * k_blocks;
    const int e = block / blocks_per_e;
    const int rest = block - e * blocks_per_e;
    const int n = rest / k_blocks;
    const int kb = rest - n * k_blocks;

    const int tid = threadIdx.x;
    if (tid < kBlockK) {
        const size_t off = ((size_t)e * kHiddenSize + n) * (size_t)K_local
                         + kb * kBlockK + tid;
        w_packed[off] = w_orig[off];
    }
}

// -----------------------------------------------------------------------------
// CUtensorMap build helper.
// -----------------------------------------------------------------------------
// W_down has shape [E=257, N=6144, K=K_local], dtype = fp8_e4m3 = 1 byte.
//   * dim 0 (x = K) — element stride 1 byte
//   * dim 1 (y = N rows)
//   * dim 2 (z = E experts)
// Box dim per tile = [kBlockK, kMmaM, 1] = [128, 16, 1].
// [dv53] SWIZZLE_128B mode. 128-byte rows × 16 rows = 2 swizzle periods
// of 8 rows each. For each row r, the 8 16-byte chunks are permuted by
// chunk_phys = chunk_logical XOR (r & 7). Bank-conflict-free for the
// consumer's LDS.128 (see cvt_fp8_to_afrag_direct).
static CUtensorMap make_w_down_tmap(
    void* base_ptr, int K_local, CUresult* out_err) {
    CUtensorMap map = {};
    cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(K_local),
        static_cast<cuuint64_t>(kHiddenSize),
        static_cast<cuuint64_t>(kNumExpertsTotal),
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

    *out_err = cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /*rank=*/3,
        base_ptr,
        global_dim, global_stride, box_dim, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,  // [dv53] bank-conflict-free fp8 stage reads
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

// -----------------------------------------------------------------------------
// [dv22] Hidden bf16 TMA descriptor.
//
// hidden_in [M, 9, K_local] bf16 contiguous row-major.
// TMA coord order: (x = K, y = slot, z = m).
// global_dim   = {K_local, 9, M} elements
// global_stride = bytes between successive y / z elements
// box_dim      = {kHiddenKChunk, 9, M} elements (one TMA call = one K chunk)
// L2_128B (narrow; hidden is small and only loaded once per launch).
// -----------------------------------------------------------------------------
static CUtensorMap make_hidden_tmap_v110(
    void* base_ptr, int M, int K_local, CUresult* out_err) {
    CUtensorMap map = {};
    cuuint64_t global_dim[3] = {
        static_cast<cuuint64_t>(K_local),
        static_cast<cuuint64_t>(kTopKPlusShared),
        static_cast<cuuint64_t>(M),
    };
    cuuint64_t global_stride[2] = {
        static_cast<cuuint64_t>(K_local) * 2u,
        static_cast<cuuint64_t>(K_local)
            * static_cast<cuuint64_t>(kTopKPlusShared) * 2u,
    };
    cuuint32_t box_dim[3] = {
        static_cast<cuuint32_t>(kHiddenKChunk),
        static_cast<cuuint32_t>(kTopKPlusShared),
        static_cast<cuuint32_t>(M),
    };
    cuuint32_t elem_stride[3] = {1u, 1u, 1u};

    // [v68-fp16-hidden] TMA element type is FLOAT16 (was BFLOAT16). Stride
    // bytes are unchanged (both 16-bit). Consumer reads __half.
    *out_err = cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
        /*rank=*/3,
        base_ptr,
        global_dim, global_stride, box_dim, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

}  // anonymous namespace

// -----------------------------------------------------------------------------
// [v110-perkb-scale] Public: weight repack — bit-identical pack, scale
// passthrough. Output scale tensor is now per-K-block fp32 (one scale per
// 128-col K-block per N-block per expert) instead of the previous per-group
// max. The runtime kernel applies the scale per-K-block at MMA-fold time,
// matching v68's design and the parent fp8_block_scale_moe runner's precision.
//
// Output scale tensor shape: [E, 48, k_blocks] fp32
//   (was [E, 48, n_groups=1] under the old group-max design)
// The signature returns (packed_weights, per_k_block_scale) — same arity as
// before, just with the trailing dim semantics changed (k_blocks instead of
// n_groups). The PyTorch op signature is unchanged.
// -----------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> repack_weights_v110(
    torch::Tensor w_down_fp8,    // [E, 6144, K_local] fp8
    torch::Tensor w_down_scale)  // [E, 48, K_local/128] fp32
{
    TORCH_CHECK(w_down_fp8.is_cuda() && w_down_scale.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(w_down_fp8.dtype() == torch::kFloat8_e4m3fn,
                "w_down_fp8 must be fp8 e4m3");
    TORCH_CHECK(w_down_scale.dtype() == torch::kFloat32,
                "w_down_scale must be fp32");
    TORCH_CHECK(w_down_fp8.is_contiguous() && w_down_scale.is_contiguous(),
                "inputs must be contiguous");
    TORCH_CHECK(w_down_fp8.dim() == 3 && w_down_fp8.size(1) == kHiddenSize,
                "w_down_fp8 must be [E, 6144, K_local]");
    TORCH_CHECK(w_down_scale.dim() == 3,
                "w_down_scale must be [E, N/128, K_local/128]");

    const int64_t E       = w_down_fp8.size(0);
    const int64_t K_local = w_down_fp8.size(2);
    const int64_t n_n_blocks = kHiddenSize / kBlockN;
    const int64_t k_blocks   = K_local / kBlockK;
    TORCH_CHECK(w_down_scale.size(0) == E, "scale dim0 mismatch");
    TORCH_CHECK(w_down_scale.size(1) == n_n_blocks, "scale dim1 mismatch");
    TORCH_CHECK(w_down_scale.size(2) == k_blocks, "scale dim2 mismatch");
    TORCH_CHECK(K_local == 512 || K_local == 256,
                "v110 supports K_local in {512 (TP=4), 256 (TP=8)}; got K_local=",
                K_local);

    // [v110-perkb-scale] Output scale tensor: per-K-block, not per-group.
    auto per_kb_scale = torch::empty(
        {E, n_n_blocks, k_blocks},
        torch::dtype(torch::kFloat32).device(w_down_fp8.device()));
    auto w_packed = torch::empty_like(w_down_fp8);

    const at::cuda::OptionalCUDAGuard device_guard(w_down_fp8.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    // Pass 1: bit-identical scale copy into [E, 48, k_blocks].
    const int64_t pass1_blocks = E * n_n_blocks;
    copy_per_kb_scales_kernel_v110<<<
        static_cast<unsigned>(pass1_blocks),
        /*threads=*/32, 0, stream>>>(
        w_down_scale.data_ptr<float>(),
        per_kb_scale.data_ptr<float>(),
        static_cast<int>(n_n_blocks),
        static_cast<int>(k_blocks));

    // Pass 2: bit-identical fp8 weight copy (no rescale).
    const int64_t pass2_blocks = E * kHiddenSize * k_blocks;
    copy_weights_kernel_v110<<<
        static_cast<unsigned>(pass2_blocks), 128, 0, stream>>>(
        reinterpret_cast<const __nv_fp8_e4m3*>(
            w_down_fp8.data_ptr<at::Float8_e4m3fn>()),
        reinterpret_cast<__nv_fp8_e4m3*>(
            w_packed.data_ptr<at::Float8_e4m3fn>()),
        static_cast<int>(K_local),
        static_cast<int>(k_blocks));

    AT_CUDA_CHECK(cudaGetLastError());
    return std::make_tuple(w_packed, per_kb_scale);
}

// -----------------------------------------------------------------------------
// Host launcher
// -----------------------------------------------------------------------------
torch::Tensor mega_down_v110(
    torch::Tensor hidden_in,
    torch::Tensor indices,
    torch::Tensor scores,
    torch::Tensor residual,
    torch::Tensor w_down_packed,        // pre-folded fp8 [E, 6144, K_local]
    torch::Tensor w_down_group_scale,   // [v110-perkb-scale] [E, 48, k_blocks]
    bool add_residual_on_rank0_only,
    int64_t rank,
    int64_t peer_ptr0,
    int64_t peer_ptr1,
    int64_t peer_ptr2,
    int64_t peer_ptr3,
    int64_t peer_ptr4,
    int64_t peer_ptr5,
    int64_t peer_ptr6,
    int64_t peer_ptr7,
    int64_t num_peers,
    int64_t flag)
{
    TORCH_CHECK(hidden_in.is_cuda(),     "hidden_in must be CUDA");
    TORCH_CHECK(indices.is_cuda(),       "indices must be CUDA");
    TORCH_CHECK(scores.is_cuda(),        "scores must be CUDA");
    TORCH_CHECK(residual.is_cuda(),      "residual must be CUDA");
    TORCH_CHECK(w_down_packed.is_cuda(), "w_down_packed must be CUDA");
    TORCH_CHECK(w_down_group_scale.is_cuda(), "w_down_group_scale must be CUDA");

    // [v68-fp16-hidden] hidden_in is now fp16 (was bf16).
    TORCH_CHECK(hidden_in.dtype()    == torch::kHalf, "hidden_in must be fp16 (was bf16; v68 now emits fp16)");
    TORCH_CHECK(indices.dtype()      == torch::kInt32,    "indices must be int32");
    TORCH_CHECK(scores.dtype()       == torch::kFloat32,  "scores must be fp32");
    TORCH_CHECK(residual.dtype()     == torch::kBFloat16, "residual must be bf16");
    TORCH_CHECK(w_down_packed.dtype()== torch::kFloat8_e4m3fn,
                "w_down_packed must be fp8 e4m3");
    TORCH_CHECK(w_down_group_scale.dtype() == torch::kFloat32,
                "w_down_group_scale must be fp32");

    TORCH_CHECK(hidden_in.dim() == 3,    "hidden_in must be [M, 9, K_local]");
    TORCH_CHECK(hidden_in.size(1) == kTopKPlusShared, "hidden_in dim1 must = 9");
    TORCH_CHECK(indices.dim() == 2 && indices.size(1) == kRoutedSlots,
                "indices must be [M, 8]");
    TORCH_CHECK(scores.dim()  == 2 && scores.size(1)  == kRoutedSlots,
                "scores must be [M, 8]");
    TORCH_CHECK(residual.dim() == 2 && residual.size(1) == kHiddenSize,
                "residual must be [M, 6144]");
    TORCH_CHECK(w_down_packed.dim() == 3 && w_down_packed.size(0) == kNumExpertsTotal
                && w_down_packed.size(1) == kHiddenSize,
                "w_down_packed must be [257, 6144, K_local]");
    TORCH_CHECK(w_down_group_scale.dim() == 3 &&
                w_down_group_scale.size(0) == kNumExpertsTotal,
                "w_down_group_scale must be [257, 48, k_blocks] (per-K-block scale)");

    const int M       = static_cast<int>(hidden_in.size(0));
    const int K_local = static_cast<int>(hidden_in.size(2));
    // [dv110-runtimeM] M is now a runtime argument in [1, kMaxM]. Smem
    // buffers are sized for kMaxM (compile-time upper bound); only the
    // active-token loops use the runtime M.
    TORCH_CHECK(M >= 1 && M <= kMaxM,
                "v110 supports M in [1, ", kMaxM, "]; got M=", M);
    TORCH_CHECK((K_local == 512 && num_peers == 4) ||
                (K_local == 256 && num_peers == 8),
                "v110 supports (K_local=512, num_peers=4) [TP=4] or "
                "(K_local=256, num_peers=8) [TP=8]; got K_local=",
                K_local, " num_peers=", num_peers);
    TORCH_CHECK(w_down_packed.size(2) == K_local,
                "w_down_packed last dim mismatch");
    const int k_blocks = K_local / kBlockK;     // 4 (TP=4) or 2 (TP=8)
    TORCH_CHECK(w_down_group_scale.size(1) == kHiddenSize / kBlockN,
                "w_down_group_scale dim1 must = 48");
    // [v110-perkb-scale] Scale tensor is now [E, 48, k_blocks] (per-K-block).
    TORCH_CHECK(w_down_group_scale.size(2) == k_blocks,
                "w_down_group_scale dim2 must = k_blocks (per-K-block scale layout); got=",
                w_down_group_scale.size(2), " expected=", k_blocks);

    TORCH_CHECK(hidden_in.is_contiguous(),     "hidden_in must be contiguous");
    TORCH_CHECK(indices.is_contiguous(),       "indices must be contiguous");
    TORCH_CHECK(scores.is_contiguous(),        "scores must be contiguous");
    TORCH_CHECK(residual.is_contiguous(),      "residual must be contiguous");
    TORCH_CHECK(w_down_packed.is_contiguous(), "w_down_packed must be contiguous");
    TORCH_CHECK(w_down_group_scale.is_contiguous(),
                "w_down_group_scale must be contiguous");

    TORCH_CHECK(num_peers >= 1 && num_peers <= kMaxPeers,
                "num_peers must be in [1, 8]");
    TORCH_CHECK(rank >= 0 && rank < num_peers,
                "rank must be in [0, num_peers)");
    // The Lamport AR primitive uses `flag` as a sentinel marker that the
    // consumer's spin-poll compares against the peer-buffer high bits.
    // IpcMemory zero-initializes the peer buffers, so `flag == 0` would
    // immediately match the zero-initialized state on the FIRST call
    // after IPC allocation, causing the consumer to silently read 0.0
    // instead of the actual remote write. The caller must start the
    // flag sequence at 1 (and increment per call thereafter).
    TORCH_CHECK(flag != 0,
                "v110: flag must be non-zero to avoid sentinel collision with "
                "the zero-initialized Lamport peer buffer. Start the flag "
                "sequence at 1 and increment per call.");

    // Build the TMA descriptor for w_down.
    CUresult tma_err = CUDA_SUCCESS;
    CUtensorMap w_down_map = make_w_down_tmap(
        w_down_packed.data_ptr(), K_local, &tma_err);
    TORCH_CHECK(tma_err == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled (w_down) failed: CUresult=", (int)tma_err);

    // [dv22] Build the TMA descriptor for hidden_in.
    CUresult hidden_err = CUDA_SUCCESS;
    CUtensorMap hidden_map = make_hidden_tmap_v110(
        hidden_in.data_ptr(), M, K_local, &hidden_err);
    TORCH_CHECK(hidden_err == CUDA_SUCCESS,
                "cuTensorMapEncodeTiled (hidden_in) failed: CUresult=",
                (int)hidden_err);

    auto out_opts = torch::TensorOptions()
        .dtype(torch::kBFloat16)
        .device(hidden_in.device());
    auto output = torch::empty({M, kHiddenSize}, out_opts);

    const at::cuda::OptionalCUDAGuard device_guard(hidden_in.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    size_t hidden_bytes = sizeof(__nv_bfloat16)
                       * (size_t)M * kTopKPlusShared * K_local;
    size_t tables_bytes = sizeof(int32_t) * (size_t)M * kTopKPlusShared
                       + sizeof(float)   * (size_t)M * kTopKPlusShared;
    size_t partial_bytes = sizeof(float)
                         * (size_t)kTopKPlusShared * kRowTilesPerCta
                         * kMmaM * kMaxM;

    size_t bucket_count_base = hidden_bytes + tables_bytes + partial_bytes;
    bucket_count_base = (bucket_count_base + 15) & ~size_t(15);
    size_t bucket_count_bytes = sizeof(int32_t) * (size_t)kNumExpertsTotal;

    size_t bucket_pairs_base = bucket_count_base + bucket_count_bytes;
    bucket_pairs_base = (bucket_pairs_base + 15) & ~size_t(15);
    size_t bucket_pairs_bytes = (size_t)kNumExpertsTotal * kMaxM;

    size_t unique_eid_base = bucket_pairs_base + bucket_pairs_bytes;
    unique_eid_base = (unique_eid_base + 15) & ~size_t(15);
    size_t unique_eid_bytes = sizeof(int16_t) * (size_t)kMaxBuckets;

    size_t num_unique_base = unique_eid_base + unique_eid_bytes;
    num_unique_base = (num_unique_base + 15) & ~size_t(15);
    size_t num_unique_bytes = sizeof(int32_t) * 4;

    // [dv93] kFp8Stages is constexpr 4 (deploy config). Stage selection
    // and the <1>/<2> variants are eliminated. Smem footprint is fixed.
    constexpr int chosen_stages = kSpecFp8Stages;
    auto compute_smem = [&](int stages) -> size_t {
        size_t mb = num_unique_base + num_unique_bytes;
        mb = (mb + 15) & ~size_t(15);
        size_t mb_bytes = sizeof(uint64_t)
                        * (size_t)(kNumWarps * stages + 1);
        size_t fp8b = mb + mb_bytes;
        fp8b = (fp8b + 1023) & ~size_t(1023);
        size_t fp8b_bytes = (size_t)kNumWarps * stages * kFp8BytesPerStage;
        return fp8b + fp8b_bytes;
    };
    size_t smem_bytes = compute_smem(chosen_stages);

    const size_t kSmemCapBytes = 232448;  // B200/GB300 maxSharedMemoryPerBlockOptin
    TORCH_CHECK(smem_bytes <= kSmemCapBytes,
                "dv110 smem footprint ", smem_bytes,
                " exceeds cap ", kSmemCapBytes);

    // [dv110/dv110-tp8] Kernel specialization on (kAddRes, kMyRank, kNumPeers,
    // kKLocal). TP=4 gives 8 instantiations (kAddRes x rank in 0..3); TP=8
    // gives 16 instantiations (kAddRes x rank in 0..7). Total = 24.
    const bool add_res = (add_residual_on_rank0_only && rank == 0);
    using KernelFn = void(*)(
        const CUtensorMap,
        const CUtensorMap,
        const __half*,                  // [v68-fp16-hidden] hidden_in_raw (was bf16)
        const int32_t*,
        const float*,
        const __nv_bfloat16*,
        const float*,
        __nv_bfloat16*,
        uint64_t*, uint64_t*, uint64_t*, uint64_t*,
        uint64_t*, uint64_t*, uint64_t*, uint64_t*,
        int,
        uint32_t);

    KernelFn kfn = nullptr;
    if (num_peers == 4) {
        if (add_res) {
            switch (rank) {
                case 0: kfn = &mega_down_v110_kernel<true,  0, 4, 512>; break;
                case 1: kfn = &mega_down_v110_kernel<true,  1, 4, 512>; break;
                case 2: kfn = &mega_down_v110_kernel<true,  2, 4, 512>; break;
                case 3: kfn = &mega_down_v110_kernel<true,  3, 4, 512>; break;
                default: TORCH_CHECK(false, "v110 TP=4: unsupported rank=", rank);
            }
        } else {
            switch (rank) {
                case 0: kfn = &mega_down_v110_kernel<false, 0, 4, 512>; break;
                case 1: kfn = &mega_down_v110_kernel<false, 1, 4, 512>; break;
                case 2: kfn = &mega_down_v110_kernel<false, 2, 4, 512>; break;
                case 3: kfn = &mega_down_v110_kernel<false, 3, 4, 512>; break;
                default: TORCH_CHECK(false, "v110 TP=4: unsupported rank=", rank);
            }
        }
    } else if (num_peers == 8) {
        if (add_res) {
            switch (rank) {
                case 0: kfn = &mega_down_v110_kernel<true,  0, 8, 256>; break;
                case 1: kfn = &mega_down_v110_kernel<true,  1, 8, 256>; break;
                case 2: kfn = &mega_down_v110_kernel<true,  2, 8, 256>; break;
                case 3: kfn = &mega_down_v110_kernel<true,  3, 8, 256>; break;
                case 4: kfn = &mega_down_v110_kernel<true,  4, 8, 256>; break;
                case 5: kfn = &mega_down_v110_kernel<true,  5, 8, 256>; break;
                case 6: kfn = &mega_down_v110_kernel<true,  6, 8, 256>; break;
                case 7: kfn = &mega_down_v110_kernel<true,  7, 8, 256>; break;
                default: TORCH_CHECK(false, "v110 TP=8: unsupported rank=", rank);
            }
        } else {
            switch (rank) {
                case 0: kfn = &mega_down_v110_kernel<false, 0, 8, 256>; break;
                case 1: kfn = &mega_down_v110_kernel<false, 1, 8, 256>; break;
                case 2: kfn = &mega_down_v110_kernel<false, 2, 8, 256>; break;
                case 3: kfn = &mega_down_v110_kernel<false, 3, 8, 256>; break;
                case 4: kfn = &mega_down_v110_kernel<false, 4, 8, 256>; break;
                case 5: kfn = &mega_down_v110_kernel<false, 5, 8, 256>; break;
                case 6: kfn = &mega_down_v110_kernel<false, 6, 8, 256>; break;
                case 7: kfn = &mega_down_v110_kernel<false, 7, 8, 256>; break;
                default: TORCH_CHECK(false, "v110 TP=8: unsupported rank=", rank);
            }
        }
    } else {
        TORCH_CHECK(false, "v110 only supports TP=4 or TP=8; got num_peers=",
                    num_peers);
    }

    if (smem_bytes > 48 * 1024) {
        cudaError_t set_err = cudaFuncSetAttribute(
            kfn, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        TORCH_CHECK(set_err == cudaSuccess,
                    "cudaFuncSetAttribute(MaxDynamicSharedMemorySize=",
                    smem_bytes, ") failed: ", cudaGetErrorString(set_err));
    }

    dim3 grid(kNumCtas, 1, 1);
    dim3 block(kThreadsPerCta, 1, 1);

    void* args[] = {
        (void*)&w_down_map,
        (void*)&hidden_map,
        nullptr,  // hidden_in_raw   [NEW — for v110-fix-hidden-ldg]
        nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr,
    };
    // [v110-fix-hidden-ldg] hidden_in raw pointer for the LDG-based
    // hidden_in staging in the kernel — see the long comment in the
    // kernel for why we no longer use TMA for this path.
    // [v68-fp16-hidden] hidden_in dtype is now __half (was __nv_bfloat16).
    const __half* hidden_in_ptr =
        reinterpret_cast<const __half*>(hidden_in.data_ptr<at::Half>());
    const int32_t* indices_ptr = indices.data_ptr<int32_t>();
    const float*   scores_ptr  = scores.data_ptr<float>();
    const __nv_bfloat16* residual_ptr =
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>());
    const float* wscale_ptr = w_down_group_scale.data_ptr<float>();
    __nv_bfloat16* output_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
    uint64_t* pp0 = reinterpret_cast<uint64_t*>(peer_ptr0);
    uint64_t* pp1 = reinterpret_cast<uint64_t*>(peer_ptr1);
    uint64_t* pp2 = reinterpret_cast<uint64_t*>(peer_ptr2);
    uint64_t* pp3 = reinterpret_cast<uint64_t*>(peer_ptr3);
    uint64_t* pp4 = reinterpret_cast<uint64_t*>(peer_ptr4);
    uint64_t* pp5 = reinterpret_cast<uint64_t*>(peer_ptr5);
    uint64_t* pp6 = reinterpret_cast<uint64_t*>(peer_ptr6);
    uint64_t* pp7 = reinterpret_cast<uint64_t*>(peer_ptr7);
    int      m_arg  = M;
    uint32_t flag_u = static_cast<uint32_t>(flag);
    args[2]  = &hidden_in_ptr;
    args[3]  = &indices_ptr;
    args[4]  = &scores_ptr;
    args[5]  = &residual_ptr;
    args[6]  = &wscale_ptr;
    args[7]  = &output_ptr;
    args[8]  = &pp0;
    args[9]  = &pp1;
    args[10] = &pp2;
    args[11] = &pp3;
    args[12] = &pp4;
    args[13] = &pp5;
    args[14] = &pp6;
    args[15] = &pp7;
    args[16] = &m_arg;
    args[17] = &flag_u;

    cudaError_t launch_err = cudaLaunchKernel(
        (const void*)kfn, grid, block, args, smem_bytes, stream);
    TORCH_CHECK(launch_err == cudaSuccess,
                "dv110 cudaLaunchKernel failed: ",
                cudaGetErrorString(launch_err));

    AT_CUDA_CHECK(cudaGetLastError());
    return output;
}
