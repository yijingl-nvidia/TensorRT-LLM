// Step-5z mega kernel — v68 = v65 + kKTile=768 + per-K-block (no-rescale) scales.
//
// Numerical-fix variant (in-place edit, same kernel name "v68_integrated"):
// previously a single per-K-iter scale ("group max") was applied at runtime
// and a compensating fp8-rescale was folded into the packed weights — that
// rescale re-quantized every fp8 value through a sub-unity ratio, leaking
// ~5 bits per weight. The new path stores the raw per-128-col K-block
// scales (6 per K-iter) and scales the MMA accumulator per K-block at
// runtime. No pre-fold rescale; the packed weights are bit-identical to
// the original fp8 source (modulo layout). External API is unchanged
// except the auxiliary scale tensor is now 4-D
// ([E, kSubRowsPerExpert, kNumKIter, kWeightScaleKBlocksPerKIter])
// instead of 3-D.
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
// External API is UNCHANGED — activation quant remains internal to the
// kernel; we still consume bf16 hidden_in directly.
//
// ============================================================================
// DESIGN INTENT (v68) — SATURATION CONFIRMATION
// ============================================================================
//
// K-axis sweep results so far (M=4):
//   kKTile=128 (v34): 26.66 us
//   kKTile=256 (v51): 24.60 us  (-7.7% vs 128)
//   kKTile=384 (v59): 18.49 us  (-18.0% vs 256, LB=1)
//   kKTile=512 (v65): 18.12 us  (-2.4% vs 384, LB=1)  — MATCHES TileRT 18.11
//
// K-axis appears saturated by v65. v68 tests kKTile=768 to CONFIRM saturation:
// likely outcome ≈ v65 (wash) or small regression (register/inner-loop cliff).
// This is a deliberate diminishing-returns probe; ships only if it wins.
//
// kKTile=768 (NEW):
//   * K-iters: 6144 / 768 = 8 (vs v65's 12, -33%)
//   * Per-slab smem: 48 KiB × 2 sides × 2 stages = 192 KiB weight smem
//     + ~15 KiB other = ~207 KiB. Fits at M=4 (1 CTA/SM, 228 KiB cap, slim).
//   * Per-K-iter HMMA work: 24 m16n8k32 fragments (was 16 in v65)
//   * BSSY/BSYNC/WARPSYNC drop another ~33%
//
// kKTile must be multiple of 32 (HMMA.16816 k-dim = 32 fp8 bytes). 768/32 = 24.
// 6144 / 768 = 8 (integer divisor, OK). 768 / 128 = 6 sub-slabs per K-iter.
//
// kStages stays at 2.
//
// __launch_bounds__(384, V68_LB_BLOCKS_PER_SM) -- v65 found LB=1 wins. v68's
// 24-HMMA inner loop likely needs ~140-160 regs, still within LB=1's 170-reg
// cap (slim). Default LB=1.
//
// Per-K-block scale handling: kKTile=768 means each K-iter spans 6 underlying
// 128-col K-blocks (kWeightScaleKBlocksPerKIter = 48/8 = 6). The block-scale
// tensor shape: [E, kSubRowsPerExpert, kNumKIter, kWeightScaleKBlocksPerKIter]
// = [E, sub_rows, 8, 6] fp32. The mega kernel loads 6 fp32 scales per K-iter
// and folds each 128-col K-block's MMA accumulator into the per-K-iter sum
// using its own scale; no pre-fold rescale is required.
//
// ============================================================================
// LAYOUT (v68: 6 stacked sub-slabs of 128-col-wide K-major lane-contiguous).
//
// TMA SWIZZLE_NONE caps box_dim[0]*sizeof(elem) at 256 bytes. Since kKTile=768
// exceeds this limit, the slab is structured as 6 stacked 8192-byte
// "K-sixth" sub-slabs (each 128 K-cols wide, identical to v34's layout)
// reachable via box_dim[2]=6. One cp.async.bulk.tensor still fills the full
// 48 KiB slab in a single call (6 z-planes loaded contiguously into smem).
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
// TMA descriptor (rank-3, SWIZZLE_NONE):
//   global_dim = (128, 64, total_slabs * 6)
//   box_dim    = (128, 64, 6)   -- box_dim[0]=128 within the 256-byte cap
//   coord_z    = slab_idx * 6   -- one prefix per slab
//
// Auxiliary tensor (per-K-block scales — see design note above):
//   group_max_scale_gate [E, sub_rows, 8, 6] fp32 (raw per-128-col scales).
//   group_max_scale_up   [E, sub_rows, 8, 6] fp32
//
// Variable names `k_third` / `kKThirdsPerIter` / `kKSubsPerThird` are kept
// (vs renaming to "sixth") for minimal-diff vs v65 — the constants below now
// express 6 sub-slabs and 4 k_subs per sub-slab.
//
// ============================================================================

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <cstdlib>
#include <string>

#include "topk_reduce.cuh"

namespace cg = cooperative_groups;

namespace {

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
__host__ __device__ constexpr int sub_rows_per_expert(int kInterPerTp) {
    return kInterPerTp / kCtaOutRows;
}
__host__ __device__ constexpr int ctas_per_token(int kInterPerTp) {
    return kSlotsPerToken * sub_rows_per_expert(kInterPerTp);
}
__host__ __device__ constexpr int weight_scale_m_blocks(int kInterPerTp) {
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
// kCtaOutRows=64 rows; so exactly 2 CTAs share one scale m-block. The repack
// kernels compute m_block_idx = sr / (kSubRowsPerExpert / kWeightScaleMBlocks)
// and rely on that ratio being 2 at every supported TP.
static_assert(sub_rows_per_expert(kInterPerTp_TP8) /
                  weight_scale_m_blocks(kInterPerTp_TP8) == 2,
              "TP=8: kCtaOutRows must be half the 128-row FP8 scale m-block "
              "(2 CTAs per scale m-block).");
static_assert(sub_rows_per_expert(kInterPerTp_TP4) /
                  weight_scale_m_blocks(kInterPerTp_TP4) == 2,
              "TP=4: kCtaOutRows must be half the 128-row FP8 scale m-block "
              "(2 CTAs per scale m-block).");

// K-axis constants are functions of kHidden ONLY (NOT TP) — unchanged at TP=4.
// v68 delta: kKTile 512 -> 768.
constexpr int kKTile = 768;
constexpr int kNumKIter = kHidden / kKTile;                       // 8
constexpr int kKSubsPerIter = kKTile / 32;                        // 24

// Legacy v68 had "kNumKGroups = kNumKIter = 8, kKBlocksPerGroup = 1" — the
// per-K-block-scaling refactor removed the K-iter→K-group indirection
// entirely (each K-iter directly carries its 6 K-block scales).

// w_scale tensor original shape: [E, kWeightScaleMBlocks, 48]. kWeightScaleMBlocks
// depends on TP (kInterPerTp / 128). kWeightScaleKBlocks is a function of kHidden only.
constexpr int kWeightScaleKBlocks = 48;  // original 128-col K-blocks (kHidden / 128)
constexpr int kWeightScaleKBlocksPerKIter = kWeightScaleKBlocks / kNumKIter;  // 6

constexpr int kNumProducerWarps  = 4;
constexpr int kGateWorkerWarpBase = 4;
constexpr int kUpWorkerWarpBase   = 8;
constexpr int kNumGateWorkers = 4;
constexpr int kNumUpWorkers   = 4;

constexpr int kTileBytes = kCtaOutRows * kKTile;                  // 49152 (48 KiB)
// No inline scale in v68 slab; group scales live in the auxiliary tensor.
constexpr int kPackedSlabBytes = kTileBytes;                      // 49152
__host__ __device__ constexpr int packed_slabs_per_expert(int kInterPerTp) {
    return sub_rows_per_expert(kInterPerTp) * kNumKIter;          // 32 (TP=8), 64 (TP=4)
}
constexpr int kPackedExpertCount = kSharedExpert + 1;             // 257

// v68 brief: kStages = 2 (same as v59 / v65).
constexpr int kStages = 2;

constexpr float kInvalidScore = -INFINITY;
constexpr float kFp8Max = 448.f;

constexpr int kActBytes = kHidden * 2;                            // 12288
constexpr int kActCpAsyncs = kActBytes / 16;                      // 768
constexpr int kActCpAsyncsPerThread = kActCpAsyncs / kThreadsPerCta;  // 2

// K-major slab offset constants. v68 stacks 6 v34-style sub-slabs along Z
// (k_sixth axis) so each box load stays within the TMA SWIZZLE_NONE 256-byte
// inner-dim cap. Variable names kept (k_third / kKThirdsPerIter / kKSubsPerThird)
// for minimal-diff vs v65.
constexpr int kLaneBytes = 16;                                       // 16 B per lane
constexpr int kKsubBytes = kWarpSize * kLaneBytes;                   // 512 B per K-sub
constexpr int kKSubsPerThird = 4;                                    // 4 k_subs per k_sixth
constexpr int kKThirdsPerIter = 6;                                   // 768 / 128 = 6 k_sixths per K-iter
constexpr int kMtileSubslabBytes = kKSubsPerThird * kKsubBytes;      // 2048 B per m_tile within a k_sixth
constexpr int kSubslabBytes =
    kCtaOutRows / 16 * kMtileSubslabBytes;                           // 4 * 2048 = 8192 B per k_sixth
constexpr int kTmaInnerCols = 128;                                   // TMA box X = 128 (≤ 256-byte cap)

static __device__ inline float sigmoid_accurate(float x) {
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

// ---- [v68-fix-proxy-fence] Producer-side cross-proxy fence helper. ----
// v68 writes hidden_out via regular STG (generic memory proxy) and the
// downstream v110 kernel reads the same address via TMA / cp.async.bulk
// (async memory proxy). The implicit stream-order fence between v68 and
// v110 guarantees generic-proxy visibility of v68's writes to v110's
// generic-proxy reads, but it does NOT promote those writes into the
// async memory proxy. v110 already issues `fence.proxy.async.global` at
// kernel entry, but that consumer-side fence is per-thread and can only
// uplift writes the issuing thread can already observe in the generic
// proxy. With the v68 per-128-col activation-quant rewrite, the generic
// writes from v68's Phase-5 STG no longer drain reliably through L2 to
// be visible in async-proxy view by the time v110's tid 0 issues its
// hidden TMA — empirically observed as B-A=0.143 (vs ~0.003 expected).
//
// The fix: each producer thread also emits `fence.proxy.async.global`
// AFTER its STG. This per-thread fence uplifts the thread's STG into
// the async proxy before the kernel-end implicit fence, so the
// subsequent kernel sees consistent state in BOTH proxies.
//
// Cost: one extra FENCE.VIEW.ASYNC.G per active producer thread (64
// threads per CTA, 4-8 cycles each). Negligible (<0.05% of v68 wall
// time at M=1).
__device__ __forceinline__ void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;\n" :::);
}

// -------------------------------------------------------------------------
// mbarrier + cp.async.bulk wrappers (identical to v34 / v51 / v55).
// -------------------------------------------------------------------------
__device__ __forceinline__ void mbarrier_init(uint64_t* mbar, int arrive_count) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n"
                 :: "r"(addr), "r"(arrive_count));
}
__device__ __forceinline__ void mbarrier_arrive_expect_tx(
    uint64_t* mbar, uint32_t bytes) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(addr), "r"(bytes));
}
__device__ __forceinline__ void mbarrier_arrive(uint64_t* mbar) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "mbarrier.arrive.shared::cta.b64 _, [%0];\n"
        :: "r"(addr));
}
__device__ __forceinline__ void mbarrier_wait_parity(
    uint64_t* mbar, uint32_t phase) {
    uint32_t addr = __cvta_generic_to_shared(mbar);
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

// 3D TMA tensor copy.
__device__ __forceinline__ void cp_async_bulk_tensor_3d(
    void* smem_dst,
    CUtensorMap const* tmap,
    int32_t coord_x, int32_t coord_y, int32_t coord_z,
    uint64_t* mbar) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_dst);
    uint32_t mbar_addr = __cvta_generic_to_shared(mbar);
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
        "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5];\n"
        ::
        "r"(smem_addr),
        "l"(tmap),
        "r"(coord_x), "r"(coord_y), "r"(coord_z),
        "r"(mbar_addr));
}

// `ld.shared.v2.b32` = LDS.64 (8 bytes / lane).
__device__ __forceinline__ void lds64_b32x2(
    uint32_t& r0, uint32_t& r1,
    __nv_fp8_e4m3 const* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(
        const_cast<__nv_fp8_e4m3*>(smem_ptr));
    asm volatile("ld.shared.v2.b32 {%0, %1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

// `ld.shared.v4.b32` = LDS.128 (16 bytes / lane).
__device__ __forceinline__ void lds128_b32x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3,
    __nv_fp8_e4m3 const* smem_ptr) {
    uint32_t addr = __cvta_generic_to_shared(
        const_cast<__nv_fp8_e4m3*>(smem_ptr));
    asm volatile("ld.shared.v4.b32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

// -------------------------------------------------------------------------
// v68 consumer addressing: 6 stacked v34-style sub-slabs (k_sixth dim outer,
// then m_tile, then k_sub_in_sixth, then lane). SWIZZLE_NONE.
// Variable names `k_third` / `kKSubsPerThird` kept (vs renaming to sixth) for
// minimal-diff vs v65. The constants now express 6 sub-slabs of 128 K-cols each.
// -------------------------------------------------------------------------
__device__ __forceinline__ int v68_lane_offset(int m_tile, int k_sub, int lane) {
    int const k_third       = k_sub / kKSubsPerThird;       // 0..5
    int const k_sub_in_third = k_sub % kKSubsPerThird;      // 0..3
    return k_third * kSubslabBytes
         + m_tile * kMtileSubslabBytes
         + k_sub_in_third * kKsubBytes
         + lane * kLaneBytes;
}

// Compute MMA for ONE K-iter (kKTile = 768 cols => 24 m16n8k32 MMAs per fragment).
//
// Per-K-block-scaled variant: the 24 K-subs split into 6 K-blocks of 4 K-subs
// each (one per 128-col fp8-scale m-block). Each block's accumulator is
// scaled by `per_block_scale[kb] * act_block_scale[kb]` and reduced into
// `d_out[]`. The per-K-block activation scale (dequant) is supplied per
// K-block (6 fp32 per K-iter) — there is NO global act_scale_val anymore
// because activation quant is now per-128-col-K-block (TRTLLM 1×128 scheme).
//
// [v68-slot0-residual] When `use_residual` is true (shared expert / slot 0),
// the kernel additionally performs a SECOND MMA over the residual fp8
// activation, recovering ~7 bits of precision lost by the primary fp8 quant.
// Cost: 2× MMA work for slot 0 only (1/9 of total work => ~11% more compute).
// The shared expert's hidden_out has weight=1.0 in v110's combine while
// routed slots get topk_w ≈ 0.3, so slot 0 dominates the noise budget;
// halving slot 0's fp8 quant error closes ~75% of the v68 residual gap.
__device__ __forceinline__ void compute_mma_kiter_v68(
    __nv_fp8_e4m3 const* __restrict__ smem_tile,
    __nv_fp8_e4m3 const* __restrict__ smem_act_fp8,
    __nv_fp8_e4m3 const* __restrict__ smem_act_fp8_lo,
    int k_iter,
    int my_m,
    int lane,
    float const (&per_block_scale)[kWeightScaleKBlocksPerKIter],
    float const (&act_block_scale)[kWeightScaleKBlocksPerKIter],
    float const (&act_block_scale_lo)[kWeightScaleKBlocksPerKIter],
    bool use_residual,
    float (&d_out)[4]) {
#pragma unroll
    for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb) {
        float c_frag[4] = {0.f, 0.f, 0.f, 0.f};
        float c_frag_lo[4] = {0.f, 0.f, 0.f, 0.f};

#pragma unroll
        for (int ks_in_kb = 0; ks_in_kb < kKSubsPerThird; ++ks_in_kb) {
            int const k_sub = kb * kKSubsPerThird + ks_in_kb;
            // Per-lane base — natural K-major layout, 16 contiguous bytes/lane.
            __nv_fp8_e4m3 const* lane_base =
                smem_tile + v68_lane_offset(my_m, k_sub, lane);

            // 2 × LDS.64 fetch the 4 b32 A-frag chunks. ptxas typically merges
            // these into a single LDS.128 in the SASS.
            uint32_t a_frag[4];
            lds64_b32x2(a_frag[0], a_frag[1], lane_base);          // chunks 0+1
            lds64_b32x2(a_frag[2], a_frag[3], lane_base + 8);      // chunks 2+3

            // B-fragment (activations). Lanes 0..3 hold the K-contiguous 16
            // bytes/half-K-sub; other lanes hold zeros. Unchanged from v34.
            uint32_t b_frag[2];
            uint32_t b_frag_lo[2];
            if (lane < 4) {
                int const k_base =
                    k_iter * kKTile + k_sub * 32 + (lane & 3) * 4;
                uint32_t b_lo_pair[2];
                uint32_t b_hi_pair[2];
                lds64_b32x2(b_lo_pair[0], b_lo_pair[1],
                            smem_act_fp8 + k_base);
                lds64_b32x2(b_hi_pair[0], b_hi_pair[1],
                            smem_act_fp8 + k_base + 16);
                b_frag[0] = b_lo_pair[0];
                b_frag[1] = b_hi_pair[0];
                if (use_residual) {
                    lds64_b32x2(b_lo_pair[0], b_lo_pair[1],
                                smem_act_fp8_lo + k_base);
                    lds64_b32x2(b_hi_pair[0], b_hi_pair[1],
                                smem_act_fp8_lo + k_base + 16);
                    b_frag_lo[0] = b_lo_pair[0];
                    b_frag_lo[1] = b_hi_pair[0];
                } else {
                    b_frag_lo[0] = 0;
                    b_frag_lo[1] = 0;
                }
            } else {
                b_frag[0] = 0;
                b_frag[1] = 0;
                b_frag_lo[0] = 0;
                b_frag_lo[1] = 0;
            }

            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                "{%0, %1, %2, %3}, "
                "{%4, %5, %6, %7}, "
                "{%8, %9}, "
                "{%0, %1, %2, %3};\n"
                : "+f"(c_frag[0]), "+f"(c_frag[1]),
                  "+f"(c_frag[2]), "+f"(c_frag[3])
                : "r"(a_frag[0]), "r"(a_frag[1]),
                  "r"(a_frag[2]), "r"(a_frag[3]),
                  "r"(b_frag[0]), "r"(b_frag[1]));

            if (use_residual) {
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0, %1, %2, %3}, "
                    "{%4, %5, %6, %7}, "
                    "{%8, %9}, "
                    "{%0, %1, %2, %3};\n"
                    : "+f"(c_frag_lo[0]), "+f"(c_frag_lo[1]),
                      "+f"(c_frag_lo[2]), "+f"(c_frag_lo[3])
                    : "r"(a_frag[0]), "r"(a_frag[1]),
                      "r"(a_frag[2]), "r"(a_frag[3]),
                      "r"(b_frag_lo[0]), "r"(b_frag_lo[1]));
            }
        }

        // Fold this 128-col K-block's accumulator into the per-K-iter sum.
        // Per-K-block activation scale + per-K-block weight scale — both
        // are dequant scales (mul to convert fp8 → fp32 magnitude).
        float const fs_kb = per_block_scale[kb] * act_block_scale[kb];
        float const fs_kb_lo = use_residual
            ? (per_block_scale[kb] * act_block_scale_lo[kb])
            : 0.f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            d_out[i] += c_frag[i] * fs_kb;
            if (use_residual) {
                d_out[i] += c_frag_lo[i] * fs_kb_lo;
            }
        }
    }
}

template <int kInterPerTpParam>
__device__ __forceinline__ int packed_slab_idx(
    int expert, int sub_row, int k_iter) {
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTpParam);
    return (expert * kSubRowsPerExpert + sub_row) * kNumKIter + k_iter;
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
__global__ __launch_bounds__(384, V68_LB_BLOCKS_PER_SM) void mega_kernel_v68(
    __grid_constant__ const CUtensorMap w_gate_map,
    __grid_constant__ const CUtensorMap w_up_map,
    float const* __restrict__ scores,
    __nv_bfloat16 const* __restrict__ hidden_in,
    __nv_bfloat16 const* __restrict__ bias,
    uint8_t const* __restrict__ w_gate_packed,
    uint8_t const* __restrict__ w_up_packed,
    float const* __restrict__ group_max_scale_gate,  // [E, kSubRowsPerExpert, 8, 6] fp32 (per-K-block)
    float const* __restrict__ group_max_scale_up,    // [E, kSubRowsPerExpert, 8, 6] fp32 (per-K-block)
    float* __restrict__ topk_weights,
    int32_t* __restrict__ topk_indices,
    // [v68-fp16-hidden] hidden_out is fp16 (10-bit mantissa) instead of bf16
    // (7-bit). The fp32 silu(g)*u result has tighter rounding error in fp16,
    // and v110 narrows bf16 -> fp16 internally anyway, so emitting fp16 from
    // v68 avoids one quantization step. Value range (silu(g)*u for our
    // activation magnitudes) is well within fp16's +/-65504 limit.
    __half* __restrict__ hidden_out,
    int64_t num_tokens,
    float routed_scaling_factor) {

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

    __nv_fp8_e4m3* const smem_gate_tiles =
        reinterpret_cast<__nv_fp8_e4m3*>(smem_buf);
    __nv_fp8_e4m3* const smem_up_tiles =
        smem_gate_tiles + kStages * kTileBytes;

    __nv_bfloat16* const smem_act_bf16 =
        reinterpret_cast<__nv_bfloat16*>(smem_up_tiles + kStages * kTileBytes);
    // In-kernel per-128-col act quant writes fp8 to a SEPARATE buffer
    // (not aliased over bf16) so multiple warps can read bf16 / write fp8
    // in parallel without aliasing races. Costs 6 KiB.
    __nv_fp8_e4m3* const smem_act_fp8 =
        reinterpret_cast<__nv_fp8_e4m3*>(smem_act_bf16 + kHidden);
    // [v68-slot0-residual] Second fp8 activation buffer holding the residual
    // (act - dequant(act_fp8)) — used by slot 0 (shared expert) only to
    // recover ~7 bits of precision lost to the primary fp8 quant. Costs 6 KiB.
    __nv_fp8_e4m3* const smem_act_fp8_lo =
        smem_act_fp8 + kHidden;

    auto align_up_128 = [] (uintptr_t p) -> uintptr_t {
        return (p + 127u) & ~uintptr_t(127);
    };
    uintptr_t rs_base = align_up_128(reinterpret_cast<uintptr_t>(
        smem_act_fp8_lo + kHidden));

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

    // (smem_warp_max removed — no longer needed now that activation amax is
    // computed per-128-col within each quant warp and never reduced across warps.)

    // Per-128-col activation dequant scales (TRTLLM 1×128 quant scheme).
    // 48 fp32 = one scale per 128-col K-block over kHidden=6144.
    // Computed in-kernel during Phase 1 (overlapped with top-K) and consumed
    // in the K-loop fold (replacing the old single per-tensor scalar).
    float* const smem_act_block_scales = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kWeightScaleKBlocks;
    rs_base = align_up_128(rs_base);

    // [v68-slot0-residual] Second per-K-block dequant scales for the residual
    // fp8 quant (slot 0 only). 48 fp32 + alignment.
    float* const smem_act_block_scales_lo = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kWeightScaleKBlocks;
    rs_base = align_up_128(rs_base);

    float* const smem_gate_acc = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kCtaOutRows;
    rs_base = align_up_128(rs_base);

    float* const smem_up_acc = reinterpret_cast<float*>(rs_base);
    rs_base += sizeof(float) * kCtaOutRows;
    rs_base = align_up_128(rs_base);

    uint64_t* const mbar_full_gate = reinterpret_cast<uint64_t*>(rs_base);
    rs_base += sizeof(uint64_t) * kStages;
    uint64_t* const mbar_full_up   = reinterpret_cast<uint64_t*>(rs_base);
    rs_base += sizeof(uint64_t) * kStages;
    uint64_t* const mbar_empty     = reinterpret_cast<uint64_t*>(rs_base);
    rs_base += sizeof(uint64_t) * kStages;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<kWarpSize>(block);

    // ---- Init mbarriers ----
    if (tidx == 0) {
#pragma unroll
        for (int s = 0; s < kStages; ++s) {
            mbarrier_init(&mbar_full_gate[s], 1);
            mbarrier_init(&mbar_full_up[s], 1);
            mbarrier_init(&mbar_empty[s], kNumGateWorkers + kNumUpWorkers);
        }
        asm volatile("fence.proxy.async.shared::cta;\n" :::);
    }

    // ===== Phase 0: activation cp.async prefetch =====
    {
        __nv_bfloat16 const* x_ptr =
            hidden_in + static_cast<int64_t>(token) * kHidden;
#pragma unroll
        for (int ii = 0; ii < kActCpAsyncsPerThread; ++ii) {
            int const byte_off = (ii * kThreadsPerCta + tidx) * 16;
            __nv_bfloat16 const* src =
                reinterpret_cast<__nv_bfloat16 const*>(
                    reinterpret_cast<const char*>(x_ptr) + byte_off);
            __nv_bfloat16* dst =
                reinterpret_cast<__nv_bfloat16*>(
                    reinterpret_cast<char*>(smem_act_bf16) + byte_off);
            unsigned const dst_smem = __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                         "r"(dst_smem), "l"(src));
        }
        asm volatile("cp.async.commit_group;\n" :::);
    }
    // Wait for activation cp.async completion BEFORE we let any warp read
    // smem_act_bf16. We move the wait+sync to the top so the per-128-col
    // quant (running in parallel with top-K) sees the activation immediately.
    asm volatile("cp.async.wait_all;\n" :::);
    __syncthreads();

    // ===== Phase 1: noaux_tc score prep — every thread =====
    // (Independent of activation; could be done in parallel with cp.async,
    // but the sync at the top is needed anyway for activation visibility.)
    {
        int const expert = tidx;
        bool const valid = expert < kNumExperts;
        float const bias_val = valid ? __bfloat162float(bias[expert]) : kInvalidScore;
        float const score = valid
            ? scores[static_cast<int64_t>(token) * kNumExperts + expert]
            : kInvalidScore;
        float const score_sigmoid = sigmoid_accurate(score);
        if (valid) {
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
    // amax → quant_scale → fp8 cast (writing to a separate, non-aliased
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
    // strided blocks (covers 50 ≥ 48 with bounds check).
    constexpr int kQuantWarps = kNumWarps - kNumExpertWarps;        // 10
    constexpr int kQuantRounds =
        (kWeightScaleKBlocks + kQuantWarps - 1) / kQuantWarps;      // 5
    constexpr int kElemsPerLane128Block = 128 / kWarpSize;          // 4

    if (warp_idx < kNumExpertWarps) {
        // ----- Top-K stage 1 -----
        int const offset = warp_idx * kWarpSize * kMaxNumTopGroups;
        float in_value[kMaxNumTopGroups];
        int32_t in_idx[kMaxNumTopGroups];
#pragma unroll
        for (int ii = 0; ii < kMaxNumTopGroups; ++ii) {
            int const e = ii * kWarpSize + lane;
            in_idx[ii] = offset + e;
            in_value[ii] = (offset + e) < kNumExperts
                ? smem_score_bias[offset + e]
                : kInvalidScore;
        }
        mega_topk::reduceTopK<kTopK, float, kMaxNumTopGroups>(
            warp, top_scores, top_experts, in_value, in_idx, kInvalidScore, kTopK);
        if (lane < kTopK) {
            smem_inter_scores[warp_idx * kTopK + lane] = top_scores[lane];
            smem_inter_experts[warp_idx * kTopK + lane] = top_experts[lane];
        }
    } else {
        // ----- Per-128-col activation FP8 quant (warps kNumExpertWarps..) -----
        // Each "quant warp" processes kQuantRounds blocks stride-kQuantWarps.
        int const q_warp = warp_idx - kNumExpertWarps;  // 0..9

#pragma unroll
        for (int r = 0; r < kQuantRounds; ++r) {
            int const kb_global = q_warp + r * kQuantWarps;
            if (kb_global >= kWeightScaleKBlocks) break;

            int const block_off = kb_global * 128;  // bf16 element offset

            // Each lane reads 4 contiguous bf16 elements (8 B vector load).
            __nv_bfloat16 const* lane_src =
                smem_act_bf16 + block_off + lane * kElemsPerLane128Block;
            __nv_bfloat162 v01;
            __nv_bfloat162 v23;
            v01 = *reinterpret_cast<__nv_bfloat162 const*>(lane_src);
            v23 = *reinterpret_cast<__nv_bfloat162 const*>(lane_src + 2);
            float const f0 = __bfloat162float(__low2bfloat16(v01));
            float const f1 = __bfloat162float(__high2bfloat16(v01));
            float const f2 = __bfloat162float(__low2bfloat16(v23));
            float const f3 = __bfloat162float(__high2bfloat16(v23));

            float lane_max = fmaxf(fmaxf(fabsf(f0), fabsf(f1)),
                                   fmaxf(fabsf(f2), fabsf(f3)));
            float amax = cg::reduce(warp, lane_max, cg::greater<float>{});
            amax = fmaxf(amax, 1e-10f);
            float const quant_scale = kFp8Max / amax;     // bf16 * quant_scale → fp8
            // [v68-precision] Compute dequant as 1/quant_scale rather than
            // amax/kFp8Max so the (quant_scale * dequant_scale) chain
            // simplifies to EXACTLY 1.0 in fp32 (IEEE 754 x*(1/x)==1 for
            // most finite x; the two-division form `amax/448 * 448/amax`
            // accumulates a 1-ULP residual that propagates into the MMA
            // fold). Matches parent fp8CS1x128's formulation
            // (fp8_blockscale_gemm_kernel.cuh:247).
            float const dequant_scale = 1.f / quant_scale;  // fp8 * dequant → bf16

            if (lane == 0) {
                smem_act_block_scales[kb_global] = dequant_scale;
            }

            // Quantize and write 4 fp8 elements per lane.
            __nv_fp8_e4m3* lane_dst =
                smem_act_fp8 + block_off + lane * kElemsPerLane128Block;
            float const q0 = fmaxf(-kFp8Max, fminf(kFp8Max, f0 * quant_scale));
            float const q1 = fmaxf(-kFp8Max, fminf(kFp8Max, f1 * quant_scale));
            float const q2 = fmaxf(-kFp8Max, fminf(kFp8Max, f2 * quant_scale));
            float const q3 = fmaxf(-kFp8Max, fminf(kFp8Max, f3 * quant_scale));
            __nv_fp8_e4m3 const fp8_0 = __nv_fp8_e4m3(q0);
            __nv_fp8_e4m3 const fp8_1 = __nv_fp8_e4m3(q1);
            __nv_fp8_e4m3 const fp8_2 = __nv_fp8_e4m3(q2);
            __nv_fp8_e4m3 const fp8_3 = __nv_fp8_e4m3(q3);
            // Pack 4 fp8 = 4 bytes = uint32 store.
            uint32_t const packed =
                (static_cast<uint32_t>(static_cast<uint8_t>(fp8_0.__x)) <<  0) |
                (static_cast<uint32_t>(static_cast<uint8_t>(fp8_1.__x)) <<  8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(fp8_2.__x)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(fp8_3.__x)) << 24);
            *reinterpret_cast<uint32_t*>(lane_dst) = packed;

            // [v68-slot0-residual] Compute the residual quant for slot 0
            // (shared expert) use. residual_i = f_i - dequant(fp8_i),
            // then quant the residual with its own scale.
            float const r0 = f0 - static_cast<float>(fp8_0) * dequant_scale;
            float const r1 = f1 - static_cast<float>(fp8_1) * dequant_scale;
            float const r2 = f2 - static_cast<float>(fp8_2) * dequant_scale;
            float const r3 = f3 - static_cast<float>(fp8_3) * dequant_scale;

            float const r_lane_max = fmaxf(fmaxf(fabsf(r0), fabsf(r1)),
                                           fmaxf(fabsf(r2), fabsf(r3)));
            float r_amax = cg::reduce(warp, r_lane_max, cg::greater<float>{});
            r_amax = fmaxf(r_amax, 1e-10f);
            float const quant_scale_lo = kFp8Max / r_amax;
            float const dequant_scale_lo = 1.f / quant_scale_lo;

            if (lane == 0) {
                smem_act_block_scales_lo[kb_global] = dequant_scale_lo;
            }

            __nv_fp8_e4m3* lane_dst_lo =
                smem_act_fp8_lo + block_off + lane * kElemsPerLane128Block;
            float const q0_lo = fmaxf(-kFp8Max, fminf(kFp8Max, r0 * quant_scale_lo));
            float const q1_lo = fmaxf(-kFp8Max, fminf(kFp8Max, r1 * quant_scale_lo));
            float const q2_lo = fmaxf(-kFp8Max, fminf(kFp8Max, r2 * quant_scale_lo));
            float const q3_lo = fmaxf(-kFp8Max, fminf(kFp8Max, r3 * quant_scale_lo));
            uint32_t const packed_lo =
                (static_cast<uint32_t>(static_cast<uint8_t>(__nv_fp8_e4m3(q0_lo).__x)) <<  0) |
                (static_cast<uint32_t>(static_cast<uint8_t>(__nv_fp8_e4m3(q1_lo).__x)) <<  8) |
                (static_cast<uint32_t>(static_cast<uint8_t>(__nv_fp8_e4m3(q2_lo).__x)) << 16) |
                (static_cast<uint32_t>(static_cast<uint8_t>(__nv_fp8_e4m3(q3_lo).__x)) << 24);
            *reinterpret_cast<uint32_t*>(lane_dst_lo) = packed_lo;
        }
    }
    __syncthreads();

    // ----- Top-K stage 2 (warp 0 only); quant warps + warp 1 idle here. -----
    if (warp_idx == 0) {
        float cand_val = (lane < kNumInterTopK) ? smem_inter_scores[lane] : kInvalidScore;
        int32_t cand_idx = (lane < kNumInterTopK) ? smem_inter_experts[lane]
                                                   : (kNumExperts - 1);
        mega_topk::reduceTopK<kTopK, float>(
            warp, top_scores, top_experts, cand_val, cand_idx, kInvalidScore, kTopK);

        int32_t const expert_idx = (lane < kTopK) ? top_experts[lane] : (kNumExperts - 1);
        float const score_norm = (lane < kTopK) ? smem_score_sigmoid[expert_idx] : 0.f;
        // [iter-16 Fix A] Match C++ noaux_tc's fp64 routing renorm. The
        // reference path computes static_cast<float>(scoreNorm * factor /
        // (redNorm + 1e-20)) where factor is double and 1e-20 is a double
        // literal, so the whole RHS evaluates in fp64. fp32 ULP drift here
        // compounds across 75 MoE layers and shaves AL.
        double const red_norm_d = cg::reduce(warp, (double)score_norm, cg::plus<double>{});
        double const final_score_d =
            (double)score_norm * (double)routed_scaling_factor / (red_norm_d + 1e-20);
        float const final_score = static_cast<float>(final_score_d);

        if (lane < kTopK) {
            smem_topk_i[lane] = expert_idx;
            if (blockIdx.y == 0) {
                int64_t out_off = static_cast<int64_t>(token) * kTopK + lane;
                topk_weights[out_off] = final_score;
                topk_indices[out_off] = expert_idx;
            }
        }
    }
    __syncthreads();

    int const my_expert =
        (expert_slot == 0) ? kSharedExpert : smem_topk_i[expert_slot - 1];

    // Per-(expert, sub_row) base pointers for the per-K-block scale tensors.
    // Shape: [E, kSubRowsPerExpert, kNumKIter, kWeightScaleKBlocksPerKIter].
    // Total per (e, sr): kNumKIter * kWeightScaleKBlocksPerKIter = 48 fp32.
    float const* const gate_block_scale_base =
        group_max_scale_gate +
        (static_cast<int64_t>(my_expert) * kSubRowsPerExpert + sub_row) *
            (kNumKIter * kWeightScaleKBlocksPerKIter);
    float const* const up_block_scale_base =
        group_max_scale_up +
        (static_cast<int64_t>(my_expert) * kSubRowsPerExpert + sub_row) *
            (kNumKIter * kWeightScaleKBlocksPerKIter);

    bool const is_producer = (warp_idx < kNumProducerWarps);
    bool const is_gate_worker =
        (warp_idx >= kGateWorkerWarpBase &&
         warp_idx < (kGateWorkerWarpBase + kNumGateWorkers));
    bool const is_up_worker =
        (warp_idx >= kUpWorkerWarpBase &&
         warp_idx < (kUpWorkerWarpBase + kNumUpWorkers));
    int const my_m_gate =
        is_gate_worker ? (warp_idx - kGateWorkerWarpBase) : 0;
    int const my_m_up =
        is_up_worker ? (warp_idx - kUpWorkerWarpBase) : 0;

    float d_gate[4] = {0.f, 0.f, 0.f, 0.f};
    float d_up[4] = {0.f, 0.f, 0.f, 0.f};

    // Producer prologue (issuer = warp 0 lane 0). kKTile=768 -> tx bytes = 49152.
    // v68 TMA descriptor uses 6 sub-slabs per slab along Z; coord_z = slab*6.
    if (is_producer && lane == 0 && warp_idx == 0) {
        for (int s = 0; s < kStages && s < kNumKIter; ++s) {
            int slab = packed_slab_idx<kInterPerTpParam>(my_expert, sub_row, s);
            int const slab_z = slab * kKThirdsPerIter;
            mbarrier_arrive_expect_tx(&mbar_full_gate[s], kTileBytes);
            cp_async_bulk_tensor_3d(
                smem_gate_tiles + s * kTileBytes,
                &w_gate_map, /*x=*/0, /*y=*/0, /*z=*/slab_z,
                &mbar_full_gate[s]);
            mbarrier_arrive_expect_tx(&mbar_full_up[s], kTileBytes);
            cp_async_bulk_tensor_3d(
                smem_up_tiles + s * kTileBytes,
                &w_up_map, /*x=*/0, /*y=*/0, /*z=*/slab_z,
                &mbar_full_up[s]);
        }
    }

    // ---- v68 K-LOOP — per-K-BLOCK scaling (6 scales per K-iter, applied
    //      block-wise inside the inner MMA loop). ----
    for (int k = 0; k < kNumKIter; ++k) {
        int const cur_stage  = k % kStages;
        int const cur_phase  = (k / kStages) & 1;

        // Per-K-iter, per-K-block scale load: 6 fp32 weight scales per K-iter.
        // Also load the corresponding 6 per-128-col activation dequant scales
        // (produced by the in-kernel 1×128 quant during Phase 1) so the
        // compute_mma helper can fold both per-K-block.
        float gate_block_scales[kWeightScaleKBlocksPerKIter];
        float up_block_scales[kWeightScaleKBlocksPerKIter];
        float act_block_scales[kWeightScaleKBlocksPerKIter];
        // [v68-slot0-residual] Second per-K-block activation dequant scales
        // for slot 0 (shared expert) residual MMA path.
        float act_block_scales_lo[kWeightScaleKBlocksPerKIter];
        if (is_gate_worker || is_up_worker) {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb) {
                act_block_scales[kb] =
                    smem_act_block_scales[k * kWeightScaleKBlocksPerKIter + kb];
                act_block_scales_lo[kb] =
                    smem_act_block_scales_lo[k * kWeightScaleKBlocksPerKIter + kb];
            }
        } else {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb) {
                act_block_scales[kb] = 0.f;
                act_block_scales_lo[kb] = 0.f;
            }
        }
        if (is_gate_worker) {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb) {
                gate_block_scales[kb] = __ldg(
                    gate_block_scale_base +
                    k * kWeightScaleKBlocksPerKIter + kb);
            }
        } else {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb) {
                gate_block_scales[kb] = 0.f;
            }
        }
        if (is_up_worker) {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb) {
                up_block_scales[kb] = __ldg(
                    up_block_scale_base +
                    k * kWeightScaleKBlocksPerKIter + kb);
            }
        } else {
#pragma unroll
            for (int kb = 0; kb < kWeightScaleKBlocksPerKIter; ++kb) {
                up_block_scales[kb] = 0.f;
            }
        }

        int const k_load = k + kStages;
        if (k_load < kNumKIter && is_producer && lane == 0 && warp_idx == 0) {
            int const load_stage = k_load % kStages;
            int const load_phase = (k_load / kStages) & 1;
            uint32_t const wait_phase = load_phase ^ 1u;
            mbarrier_wait_parity(&mbar_empty[load_stage], wait_phase);

            int slab = packed_slab_idx<kInterPerTpParam>(my_expert, sub_row, k_load);
            int const slab_z = slab * kKThirdsPerIter;
            mbarrier_arrive_expect_tx(&mbar_full_gate[load_stage], kTileBytes);
            cp_async_bulk_tensor_3d(
                smem_gate_tiles + load_stage * kTileBytes,
                &w_gate_map, /*x=*/0, /*y=*/0, /*z=*/slab_z,
                &mbar_full_gate[load_stage]);
            mbarrier_arrive_expect_tx(&mbar_full_up[load_stage], kTileBytes);
            cp_async_bulk_tensor_3d(
                smem_up_tiles + load_stage * kTileBytes,
                &w_up_map, /*x=*/0, /*y=*/0, /*z=*/slab_z,
                &mbar_full_up[load_stage]);
        }

        // [v68-slot0-residual] Slot 0 is the shared expert; enable the
        // residual fp8 MMA path to recover ~7 bits of precision lost by the
        // primary fp8 quant. The shared expert's hidden_out has weight=1.0
        // in v110's combine while routed slots get topk_w ≈ 0.3, so the
        // extra MMA work for slot 0 (1/9 of total CTAs) closes most of
        // v68's noise residual.
        bool const use_residual = (expert_slot == 0);
        if (is_gate_worker) {
            mbarrier_wait_parity(&mbar_full_gate[cur_stage], cur_phase);
            compute_mma_kiter_v68(
                smem_gate_tiles + cur_stage * kTileBytes,
                smem_act_fp8, smem_act_fp8_lo, k, my_m_gate, lane,
                gate_block_scales, act_block_scales, act_block_scales_lo,
                use_residual, d_gate);
            if (lane == 0) mbarrier_arrive(&mbar_empty[cur_stage]);
        }
        if (is_up_worker) {
            mbarrier_wait_parity(&mbar_full_up[cur_stage], cur_phase);
            compute_mma_kiter_v68(
                smem_up_tiles + cur_stage * kTileBytes,
                smem_act_fp8, smem_act_fp8_lo, k, my_m_up, lane,
                up_block_scales, act_block_scales, act_block_scales_lo,
                use_residual, d_up);
            if (lane == 0) mbarrier_arrive(&mbar_empty[cur_stage]);
        }
    }

    __syncthreads();

    // ===== Phase 5: SiLU·× writer =====
    if (is_gate_worker && (lane & 3) == 0) {
        int const row_top = lane >> 2;
        int const row_bot = row_top + 8;
        smem_gate_acc[my_m_gate * 16 + row_top] = d_gate[0];
        smem_gate_acc[my_m_gate * 16 + row_bot] = d_gate[2];
    }
    if (is_up_worker && (lane & 3) == 0) {
        int const row_top = lane >> 2;
        int const row_bot = row_top + 8;
        smem_up_acc[my_m_up * 16 + row_top] = d_up[0];
        smem_up_acc[my_m_up * 16 + row_bot] = d_up[2];
    }
    __syncthreads();

    if (tidx < kCtaOutRows) {
        float const g = smem_gate_acc[tidx];
        float const u = smem_up_acc[tidx];
        float const silu_g = g * sigmoid_accurate(g);
        float const h = silu_g * u;
        int const global_row = row_stripe_start + tidx;
        int64_t const out_off =
            static_cast<int64_t>(token) * kSlotsPerToken * kInterPerTp
            + static_cast<int64_t>(expert_slot) * kInterPerTp
            + global_row;
        // [v68-fp16-hidden] fp32 -> fp16 narrowing has ~8x less quant noise
        // than fp32 -> bf16 for our activation magnitudes (10-bit mantissa
        // vs 7-bit).
        hidden_out[out_off] = __float2half(h);
    }
    // [v68-fix-proxy-fence] Producer-side ordering for the v68 → v110
    // kernel chain. After all hidden_out STGs by Phase 5 threads have
    // issued, we drain them to system-visible state and uplift them
    // into the async memory proxy before kernel exit. Two-step pattern:
    //
    //   __threadfence_system()       -> drains generic-proxy STGs to L2
    //                                   and waits for system-scope ack
    //                                   (release semantics, CTA-wide).
    //   fence.proxy.async.global     -> elevates the just-drained writes
    //                                   into the ASYNC proxy view, so
    //                                   that v110's `cp.async.bulk.tensor`
    //                                   load of the SAME global address
    //                                   on the SAME stream observes them.
    //
    // Critical: the implicit stream-order kernel-end fence between v68
    // and v110 synchronizes the GENERIC proxy. With the v68 actquant
    // rewrite, the L2 timing of Phase-5 STGs no longer fully drains by
    // the time v110's tid 0 issues its hidden TMA on the same VA — we
    // observe B-A in the chained diff test = 0.058-0.143 instead of
    // the ~0.003 noise floor. Pairing the two fences here at v68 exit
    // (release + proxy-uplift) makes v68's STGs system-visible AND
    // async-proxy-visible across all threads / CTAs / proxies before
    // kernel-end implicit synchronization, eliminating the residue.
    //
    // Cost: one membar.sys + one FENCE.VIEW.ASYNC.G per active Phase-5
    // thread (64 of 384), at kernel exit (one-shot). Single-digit-cycle
    // membar latency, well within the user's <=5% TPOT budget (the
    // kernel runs ~25-30 us; the fences add ~50 ns at most).
    fence_proxy_async_global();
    __threadfence_system();
}

// Dynamic smem sizing.
static inline size_t v68_smem_bytes() {
    auto align_up_128 = [] (size_t p) -> size_t {
        return (p + 127u) & ~size_t(127);
    };
    size_t bytes = 0;
    bytes += static_cast<size_t>(kStages) * kTileBytes;
    bytes += static_cast<size_t>(kStages) * kTileBytes;
    bytes += static_cast<size_t>(kHidden) * sizeof(__nv_bfloat16);
    // Separate fp8 act buffer (no longer aliased over bf16). +6 KiB.
    bytes += static_cast<size_t>(kHidden) * sizeof(__nv_fp8_e4m3);
    // [v68-slot0-residual] Second fp8 act buffer for residual quant. +6 KiB.
    bytes += static_cast<size_t>(kHidden) * sizeof(__nv_fp8_e4m3);
    bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumExperts; bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumExperts; bytes = align_up_128(bytes);
    bytes += sizeof(float) * kNumInterTopK; bytes = align_up_128(bytes);
    bytes += sizeof(int32_t) * kNumInterTopK; bytes = align_up_128(bytes);
    bytes += sizeof(int32_t) * kTopK; bytes = align_up_128(bytes);
    // smem_act_block_scales[48]: per-128-col activation dequant scales.
    bytes += sizeof(float) * kWeightScaleKBlocks; bytes = align_up_128(bytes);
    // [v68-slot0-residual] smem_act_block_scales_lo[48].
    bytes += sizeof(float) * kWeightScaleKBlocks; bytes = align_up_128(bytes);
    bytes += sizeof(float) * kCtaOutRows; bytes = align_up_128(bytes);
    bytes += sizeof(float) * kCtaOutRows; bytes = align_up_128(bytes);
    bytes += sizeof(uint64_t) * kStages;
    bytes += sizeof(uint64_t) * kStages;
    bytes += sizeof(uint64_t) * kStages;
    bytes = (bytes + 15u) & ~size_t(15);
    return bytes;
}

// -------------------------------------------------------------------------
// CUtensorMap construction — rank-3 (128, 64, slabs*6) box, SWIZZLE_NONE.
// v68 stacks 6 v34-style sub-slabs along Z to keep box_dim[0] within the
// SWIZZLE_NONE 256-byte inner-dim cap. box_dim[2]=6 -> one TMA call per slab
// loads all 48 KiB into smem contiguously.
// -------------------------------------------------------------------------
static CUtensorMap make_packed_tensor_map(
    void* base_ptr, int64_t total_slabs, CUresult* out_err) {
    CUtensorMap map = {};
    int64_t const total_subslabs = total_slabs * kKThirdsPerIter;
    cuuint64_t global_dim[3]   = {
        static_cast<cuuint64_t>(kTmaInnerCols),  // 128
        static_cast<cuuint64_t>(kCtaOutRows),    // 64
        static_cast<cuuint64_t>(total_subslabs),
    };
    cuuint64_t global_stride[2] = {
        static_cast<cuuint64_t>(kTmaInnerCols),     // 128 bytes/row
        static_cast<cuuint64_t>(kSubslabBytes),     // 8192 bytes/sub-slab
    };
    cuuint32_t box_dim[3]      = {
        static_cast<cuuint32_t>(kTmaInnerCols),  // 128
        static_cast<cuuint32_t>(kCtaOutRows),    // 64
        static_cast<cuuint32_t>(kKThirdsPerIter), // 6 sub-slabs per box
    };
    cuuint32_t elem_stride[3]  = {1u, 1u, 1u};

    *out_err = cuTensorMapEncodeTiled(
        &map,
        CU_TENSOR_MAP_DATA_TYPE_UINT8,
        /*rank=*/3,
        base_ptr,
        global_dim, global_stride, box_dim, elem_stride,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    return map;
}

// =========================================================================
// v68 OFFLINE WEIGHT REPACK — same conceptual design as v65's
// (per-K-iter pre-fold + K-major layout), but the slab now spans 768 K-cols
// instead of 512, with 6 underlying 128-col K-blocks per K-iter.
//
// Two-pass kernel:
//   Pass 1 (gather_block_scales_kernel_v68): copies the per-K-block fp32
//       scales into the broadcast-to-sub-rows layout that the mega-kernel
//       loads at runtime. Writes block_scales
//       [E, kSubRowsPerExpert, kNumKIter, kWeightScaleKBlocksPerKIter]
//       — i.e. 48 fp32 per (e, sr) (== 6 per K-iter × 8 K-iters).
//   Pass 2 (fold_and_pack_kernel_v68): pure layout transform — copies the
//       original fp8 weights into the v68 stacked sub-slab layout. No
//       per-block rescale (the runtime applies per-block scales directly),
//       hence no re-quantization noise.
// =========================================================================

// Pass 1: gather per-K-block fp32 scales into the runtime-friendly layout.
// Output[e, sr, k_iter, kb_off] = scale_orig[e, sr/2, k_iter*6 + kb_off].
// (sr/2 because 2 CTAs share each scale m-block — same broadcast as before.)
template <int kInterPerTpParam>
__global__ void gather_block_scales_kernel_v68(
    float const* __restrict__ scale_orig,   // [E, kWeightScaleMBlocks, 48] fp32
    float* __restrict__ block_scales_out) { // [E, kSubRowsPerExpert, 8, 6] fp32
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTpParam);
    constexpr int kWeightScaleMBlocks = weight_scale_m_blocks(kInterPerTpParam);
    constexpr int kPerSr = kNumKIter * kWeightScaleKBlocksPerKIter;  // 48

    int const block = blockIdx.x;
    int const e  = block / kSubRowsPerExpert;
    int const sr = block % kSubRowsPerExpert;
    int const m_block_idx = sr / (kSubRowsPerExpert / kWeightScaleMBlocks);

    int const tid = threadIdx.x;  // 64 threads — first 48 carry one fp32 each.
    if (tid < kPerSr) {
        int const kb = tid;  // 0..47 within (e, mb)
        float const v = scale_orig[
            (static_cast<int64_t>(e) * kWeightScaleMBlocks + m_block_idx) *
                kWeightScaleKBlocks + kb];
        block_scales_out[
            (static_cast<int64_t>(e) * kSubRowsPerExpert + sr) * kPerSr + tid] = v;
    }
}

// Pass 2: pure layout transform — copy the original fp8 weights into the v68
// stacked sub-slab layout. No re-quantization (the runtime applies per-K-block
// scales directly, so no ratio compensation is needed at pack time).
//
// 4*24*32 = 3072 (m_tile, k_sub, lane) triples per slab. Each thread writes
// 12 such triples -> 256 threads/block.
template <int kInterPerTpParam>
__global__ void fold_and_pack_kernel_v68(
    __nv_fp8_e4m3 const* __restrict__ w_orig,   // [E, kInterPerTp, 6144] fp8
    uint8_t* __restrict__ w_packed) {           // [E, kSubRowsPerExpert, 8, 49152] u8
    constexpr int kInterPerTp = kInterPerTpParam;
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTp);

    int const slab = blockIdx.x;
    int const e  = slab / (kSubRowsPerExpert * kNumKIter);
    int const rest = slab % (kSubRowsPerExpert * kNumKIter);
    int const sr = rest / kNumKIter;
    int const k_iter = rest % kNumKIter;

    uint8_t* slab_ptr = w_packed +
        static_cast<int64_t>(slab) * kPackedSlabBytes;

    int const tid = threadIdx.x;
    // 12 writes per thread, 256 threads = 3072 triples per slab.
#pragma unroll
    for (int idx = 0; idx < 12; ++idx) {
        int const tri = idx * blockDim.x + tid;  // 0..3071
        int const m_tile = tri / (kKSubsPerIter * kWarpSize);
        int const tri_in_mtile = tri % (kKSubsPerIter * kWarpSize);
        int const k_sub  = tri_in_mtile / kWarpSize;
        int const lane   = tri_in_mtile & 31;

        int const k_third = k_sub / kKSubsPerThird;       // 0..5
        int const k_sub_in_third = k_sub % kKSubsPerThird;  // 0..3

        int const row_lo = m_tile * 16 + (lane >> 2);
        int const row_hi = row_lo + 8;
        int const col_lo_in_block = k_sub_in_third * 32 + ((lane & 3) << 2);
        int const col_hi_in_block = col_lo_in_block + 16;

        int const gm_lo = sr * kCtaOutRows + row_lo;
        int const gm_hi = sr * kCtaOutRows + row_hi;
        int const gk_lo = k_iter * kKTile + k_third * 128 + col_lo_in_block;
        int const gk_hi = k_iter * kKTile + k_third * 128 + col_hi_in_block;

        __nv_fp8_e4m3 const* src_a =
            reinterpret_cast<__nv_fp8_e4m3 const*>(w_orig) +
            (static_cast<int64_t>(e) * kInterPerTp + gm_lo) *
                static_cast<int64_t>(kHidden) + gk_lo;
        __nv_fp8_e4m3 const* src_b =
            reinterpret_cast<__nv_fp8_e4m3 const*>(w_orig) +
            (static_cast<int64_t>(e) * kInterPerTp + gm_hi) *
                static_cast<int64_t>(kHidden) + gk_lo;
        __nv_fp8_e4m3 const* src_c =
            reinterpret_cast<__nv_fp8_e4m3 const*>(w_orig) +
            (static_cast<int64_t>(e) * kInterPerTp + gm_lo) *
                static_cast<int64_t>(kHidden) + gk_hi;
        __nv_fp8_e4m3 const* src_d =
            reinterpret_cast<__nv_fp8_e4m3 const*>(w_orig) +
            (static_cast<int64_t>(e) * kInterPerTp + gm_hi) *
                static_cast<int64_t>(kHidden) + gk_hi;

        uint8_t* dst = slab_ptr + k_third * kSubslabBytes
                       + m_tile * kMtileSubslabBytes
                       + k_sub_in_third * kKsubBytes
                       + lane * kLaneBytes;

#pragma unroll
        for (int b = 0; b < 4; ++b) {
            dst[b]      = *reinterpret_cast<uint8_t const*>(&src_a[b]);
            dst[4 + b]  = *reinterpret_cast<uint8_t const*>(&src_b[b]);
            dst[8 + b]  = *reinterpret_cast<uint8_t const*>(&src_c[b]);
            dst[12 + b] = *reinterpret_cast<uint8_t const*>(&src_d[b]);
        }
    }
}

// Per-kInterPerTp launcher helpers, instantiated for {256 (TP=8), 512 (TP=4)}.
template <int kInterPerTpParam>
static std::tuple<torch::Tensor, torch::Tensor> repack_weights_v68_impl(
    torch::Tensor w_fp8,
    torch::Tensor w_scale) {
    constexpr int kInterPerTp = kInterPerTpParam;
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTp);
    constexpr int kWeightScaleMBlocks = weight_scale_m_blocks(kInterPerTp);
    constexpr int kPackedSlabsPerExpert = packed_slabs_per_expert(kInterPerTp);

    TORCH_CHECK(w_fp8.size(0) == kPackedExpertCount &&
                w_fp8.size(1) == kInterPerTp &&
                w_fp8.size(2) == kHidden, "w_fp8 shape mismatch");
    TORCH_CHECK(w_scale.size(0) == kPackedExpertCount &&
                w_scale.size(1) == kWeightScaleMBlocks &&
                w_scale.size(2) == kWeightScaleKBlocks,
                "w_scale shape mismatch");

    int64_t const num_slabs =
        static_cast<int64_t>(kPackedExpertCount) * kPackedSlabsPerExpert;
    auto packed = torch::empty(
        {static_cast<long>(num_slabs), static_cast<long>(kPackedSlabBytes)},
        torch::dtype(torch::kUInt8).device(w_fp8.device()));

    // Auxiliary scale tensor: per-K-block (not group-max).
    // Shape [E, kSubRowsPerExpert, kNumKIter, kWeightScaleKBlocksPerKIter].
    auto block_scales = torch::empty(
        {kPackedExpertCount, kSubRowsPerExpert, kNumKIter,
         kWeightScaleKBlocksPerKIter},
        torch::dtype(torch::kFloat32).device(w_fp8.device()));

    auto stream = at::cuda::getCurrentCUDAStream();

    // Pass 1: gather per-K-block scales into the broadcast-to-sub-rows layout.
    int64_t const num_pass1_blocks =
        static_cast<int64_t>(kPackedExpertCount) * kSubRowsPerExpert;
    gather_block_scales_kernel_v68<kInterPerTpParam><<<
        static_cast<unsigned>(num_pass1_blocks), 64, 0, stream>>>(
        w_scale.data_ptr<float>(),
        block_scales.data_ptr<float>());

    // Pass 2: layout transform (no rescale) into 49152-byte slabs.
    fold_and_pack_kernel_v68<kInterPerTpParam><<<
        static_cast<unsigned>(num_slabs), 256, 0, stream>>>(
        reinterpret_cast<__nv_fp8_e4m3 const*>(
            w_fp8.data_ptr<at::Float8_e4m3fn>()),
        packed.data_ptr<uint8_t>());

    return std::make_tuple(packed, block_scales);
}

template <int kInterPerTpParam>
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mega_silu_v68_impl(
    torch::Tensor scores,
    torch::Tensor hidden_in,
    torch::Tensor bias,
    torch::Tensor w_gate_packed,
    torch::Tensor w_up_packed,
    torch::Tensor group_max_scale_gate,
    torch::Tensor group_max_scale_up,
    double routed_scaling_factor) {
    constexpr int kInterPerTp = kInterPerTpParam;
    constexpr int kSubRowsPerExpert = sub_rows_per_expert(kInterPerTp);
    constexpr int kCtasPerToken = ctas_per_token(kInterPerTp);
    constexpr int kPackedSlabsPerExpert = packed_slabs_per_expert(kInterPerTp);

    auto const M = scores.size(0);

    int64_t const expected_slabs =
        static_cast<int64_t>(kPackedExpertCount) * kPackedSlabsPerExpert;
    TORCH_CHECK(w_gate_packed.numel() == expected_slabs * kPackedSlabBytes,
                "w_gate_packed bytes mismatch");
    TORCH_CHECK(w_up_packed.numel() == expected_slabs * kPackedSlabBytes,
                "w_up_packed bytes mismatch");

    int64_t const expected_block_scale =
        static_cast<int64_t>(kPackedExpertCount) * kSubRowsPerExpert *
        kNumKIter * kWeightScaleKBlocksPerKIter;
    TORCH_CHECK(group_max_scale_gate.numel() == expected_block_scale,
                "group_max_scale_gate (block scales) shape mismatch");
    TORCH_CHECK(group_max_scale_up.numel() == expected_block_scale,
                "group_max_scale_up (block scales) shape mismatch");

    auto stream = at::cuda::getCurrentCUDAStream();
    const at::cuda::OptionalCUDAGuard device_guard(scores.device());

    static bool s_logged = false;

    CUresult err_gate = CUDA_SUCCESS;
    CUtensorMap w_gate_map = make_packed_tensor_map(
        w_gate_packed.data_ptr(), expected_slabs, &err_gate);
    TORCH_CHECK(err_gate == CUDA_SUCCESS,
                "v68 cuTensorMapEncodeTiled(gate) failed: CUresult=", (int)err_gate);

    CUresult err_up = CUDA_SUCCESS;
    CUtensorMap w_up_map = make_packed_tensor_map(
        w_up_packed.data_ptr(), expected_slabs, &err_up);
    TORCH_CHECK(err_up == CUDA_SUCCESS,
                "v68 cuTensorMapEncodeTiled(up) failed: CUresult=", (int)err_up);

    if (!s_logged) {
        printf("[v68_integrated] kKTile=%d, kStages=%d, LB(384,%d), kInterPerTp=%d (per-K-block scaling; no pre-fold rescale)\n",
               kKTile, kStages, V68_LB_BLOCKS_PER_SM, kInterPerTp);
        s_logged = true;
    }

    auto topk_weights = torch::empty(
        {M, kTopK}, torch::dtype(torch::kFloat32).device(scores.device()));
    auto topk_indices = torch::empty(
        {M, kTopK}, torch::dtype(torch::kInt32).device(scores.device()));
    // [v68-fp16-hidden] hidden_out is fp16 (was bf16). See kernel signature.
    auto hidden_out = torch::empty(
        {M, kSlotsPerToken, kInterPerTp},
        torch::dtype(torch::kHalf).device(scores.device()));

    dim3 grid(static_cast<unsigned>(M), kCtasPerToken, 1);
    dim3 block(kThreadsPerCta, 1, 1);

    size_t const smem_bytes = v68_smem_bytes();

    static bool s_smem_opt_done = false;
    if (!s_smem_opt_done) {
        cudaError_t err = cudaFuncSetAttribute(
            mega_kernel_v68<kInterPerTpParam>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_bytes));
        TORCH_CHECK(err == cudaSuccess,
                    "cudaFuncSetAttribute(v68, maxDynSmem) failed: ",
                    cudaGetErrorString(err));
        s_smem_opt_done = true;
    }

    mega_kernel_v68<kInterPerTpParam><<<grid, block, smem_bytes, stream>>>(
        w_gate_map, w_up_map,
        scores.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 const*>(hidden_in.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16 const*>(bias.data_ptr<at::BFloat16>()),
        w_gate_packed.data_ptr<uint8_t>(),
        w_up_packed.data_ptr<uint8_t>(),
        group_max_scale_gate.data_ptr<float>(),
        group_max_scale_up.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        topk_indices.data_ptr<int32_t>(),
        reinterpret_cast<__half*>(hidden_out.data_ptr<at::Half>()),
        M,
        static_cast<float>(routed_scaling_factor));

    return std::make_tuple(topk_weights, topk_indices, hidden_out);
}

}  // anonymous namespace

namespace mega_kernel {

// -------------------------------------------------------------------------
// Public: v68 weight repack.
// Supports TP=8 (kInterPerTp=256) and TP=4 (kInterPerTp=512) — dispatched
// on w_fp8.size(1).
// Returns (packed_weights [num_slabs, 49152] u8,
//          block_scales [E, kSubRowsPerExpert, 8, 6] fp32 per-K-block).
// -------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor> repack_weights_v68(
    torch::Tensor w_fp8,
    torch::Tensor w_scale) {
    TORCH_CHECK(w_fp8.is_cuda() && w_scale.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(w_fp8.dtype() == torch::kFloat8_e4m3fn, "w must be fp8");
    TORCH_CHECK(w_scale.dtype() == torch::kFloat32, "scale must be fp32");
    TORCH_CHECK(w_fp8.is_contiguous() && w_scale.is_contiguous(),
                "inputs must be contiguous");

    int64_t const inter_per_tp = w_fp8.size(1);
    if (inter_per_tp == kInterPerTp_TP8) {
        return repack_weights_v68_impl<kInterPerTp_TP8>(
            std::move(w_fp8), std::move(w_scale));
    } else if (inter_per_tp == kInterPerTp_TP4) {
        return repack_weights_v68_impl<kInterPerTp_TP4>(
            std::move(w_fp8), std::move(w_scale));
    } else {
        TORCH_CHECK(false,
                    "v68 only supports TP=4 (kInterPerTp=512) or TP=8 "
                    "(kInterPerTp=256); got w_fp8.size(1)=", inter_per_tp);
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> mega_silu_v68(
    torch::Tensor scores,
    torch::Tensor hidden_in,
    torch::Tensor bias,
    torch::Tensor w_gate_packed,
    torch::Tensor w_up_packed,
    torch::Tensor group_max_scale_gate,
    torch::Tensor group_max_scale_up,
    double routed_scaling_factor) {

    TORCH_CHECK(scores.is_cuda() && hidden_in.is_cuda() && bias.is_cuda() &&
                w_gate_packed.is_cuda() && w_up_packed.is_cuda() &&
                group_max_scale_gate.is_cuda() && group_max_scale_up.is_cuda(),
                "inputs must be CUDA");
    TORCH_CHECK(scores.dtype() == torch::kFloat32 &&
                hidden_in.dtype() == torch::kBFloat16 &&
                bias.dtype() == torch::kBFloat16 &&
                w_gate_packed.dtype() == torch::kUInt8 &&
                w_up_packed.dtype() == torch::kUInt8 &&
                group_max_scale_gate.dtype() == torch::kFloat32 &&
                group_max_scale_up.dtype() == torch::kFloat32,
                "dtype mismatch");

    auto const M = scores.size(0);
    TORCH_CHECK(scores.size(1) == kNumExperts);
    TORCH_CHECK(hidden_in.size(0) == M && hidden_in.size(1) == kHidden);
    TORCH_CHECK(bias.size(0) == kNumExperts);

    TORCH_CHECK(scores.is_contiguous() && hidden_in.is_contiguous() &&
                bias.is_contiguous() && w_gate_packed.is_contiguous() &&
                w_up_packed.is_contiguous() &&
                group_max_scale_gate.is_contiguous() &&
                group_max_scale_up.is_contiguous(),
                "inputs must be contiguous");

    // The packed weights aren't shape-preserving (collapsed to bytes), but
    // the block-scale tensor IS preserving: shape
    // [E, kSubRowsPerExpert, kNumKIter, kWeightScaleKBlocksPerKIter].
    // kSubRowsPerExpert = 4 at TP=8 and 8 at TP=4. Use this to infer TP.
    TORCH_CHECK(group_max_scale_gate.dim() == 4,
                "group_max_scale_gate (block scales) must be 4D");
    int64_t const sub_rows = group_max_scale_gate.size(1);
    int64_t const inter_per_tp = sub_rows * kCtaOutRows;

    if (inter_per_tp == kInterPerTp_TP8) {
        return mega_silu_v68_impl<kInterPerTp_TP8>(
            std::move(scores), std::move(hidden_in), std::move(bias),
            std::move(w_gate_packed), std::move(w_up_packed),
            std::move(group_max_scale_gate), std::move(group_max_scale_up),
            routed_scaling_factor);
    } else if (inter_per_tp == kInterPerTp_TP4) {
        return mega_silu_v68_impl<kInterPerTp_TP4>(
            std::move(scores), std::move(hidden_in), std::move(bias),
            std::move(w_gate_packed), std::move(w_up_packed),
            std::move(group_max_scale_gate), std::move(group_max_scale_up),
            routed_scaling_factor);
    } else {
        TORCH_CHECK(false,
                    "v68 only supports TP=4 (kInterPerTp=512) or TP=8 "
                    "(kInterPerTp=256); inferred kInterPerTp=", inter_per_tp,
                    " from group_max_scale_gate.size(1)=", sub_rows);
    }
}

}  // namespace mega_kernel
