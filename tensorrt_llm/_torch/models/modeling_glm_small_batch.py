"""GLM-5 small-batch fused-kernel decoder-layer subclass.

Opt-in path activated by `model_config.enable_glm5_small_batch_fused`. Replaces
the default `DeepseekV3DecoderLayer` with `Glm5SmallBatchDecoderLayer`, which
exposes 5 per-kernel feature flags. Each flag enables one fused mega-kernel
mirroring the corresponding TileRT kernel:

  use_mla_front_kernel           ← `MlaGlm5ExecutorImpl`         (PR 2, paired)
  use_unproj_o_allreduce_kernel  ← `UnprojOAllreduceGlm5ExecutorImpl` (PR 2, paired)
  use_rmsnorm_expert_proj_kernel ← `RMSNormExpertProjGlm5ExecutorImpl` (PR 3)
  use_up_gate_silu_kernel        ← v68 ExpertSelectUpGateSiLU (PR 1, paired w/ down-AR)
  use_down_allreduce_kernel      ← v110 ExpertDownAllReduce   (PR 1, paired w/ up-gate-silu)

Note on GLM-5 model class: GLM-5 ships as HF arch `GlmMoeDsaForCausalLM`, which
TRT-LLM registers in `modeling_deepseekv3.py:1810` against `DeepseekV3ForCausalLM`.
The constructor at `:1847` rewrites `model_type='glm_moe_dsa'` to `'deepseek_v32'`
internally. Layer construction therefore goes through `DeepseekV3Model` /
`DeepseekV3DecoderLayer` / `Deepseekv3MoE`, NOT the `Glm4*` classes from
`modeling_glm.py` (those are for the older `Glm4MoeForCausalLM` HF arch, a
different model). This file's classes subclass the DeepSeekV3 family.

PR 1 (in progress): wires v68 (ExpertSelectUpGateSiLU) and v110 (ExpertDownAllReduce)
into Glm5SmallBatchFusedMoE. Both kernels are paired: when M ≤ 4 on TP=4, both
fire; otherwise the parent TRTLLM-Gen MoE handles the path.

v110 hard-codes TP=4 (kSpecNumPeers=4, kSpecKLocal=512 in mega_kernel_down_v110.cu
lines 268-269). PR 1 therefore only fires on TP=4 deployments. TP=8 production
requires a v110 kernel-level extension.

See `nvbugs/6108841/revisit_moe_mega_kernel/INTEGRATION_PLAN.md` for the full
design rationale, per-PR breakdown, and CUDA-graph-safety constraints.

Precedents:
- Layer-class swap pattern: `modeling_llama.py:969-982` (Llama4 min-latency).
- Decoder-layer subclass pattern: `modeling_llama_min_latency.py:641` (`Llama4MinLatencyDecoderLayer`).
- MoE subclass pattern: `modeling_llama_min_latency.py:509` (`Llama4MinLatencyMoE`).
- Backend subclass + FUSED_COMM scheduler: `mega_moe/mega_moe_deepgemm.py:120`.
- IPC sym-heap workspace: `tensorrt_llm/plugin/plugin.py:548` (`IpcMemory`).
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import torch

from tensorrt_llm._ipc_utils import can_access_peer
from tensorrt_llm.plugin.plugin import IpcMemory

from ..attention_backend import AttentionMetadata
from ..model_config import ModelConfig
from ..modules.fused_moe.fused_moe_trtllm_gen import TRTLLMGenFusedMoE
from ..modules.fused_moe.interface import MoESchedulerKind
from ..speculative import SpecMetadata
from .modeling_deepseekv3 import Deepseekv3MoE, DeepseekV3DecoderLayer


# Per-launch logging — flipped on for the "verify kernel is called" debug pass.
# Set GLM5_SMALL_BATCH_LOG=1 in the env to enable. Once verified, unset and re-bench.
_LOG_KERNEL_INVOCATION = os.environ.get("GLM5_SMALL_BATCH_LOG", "0") == "1"


# [iter-9] Parent-down diagnostic toggle. When GLM5_USE_PARENT_DOWN=1, the
# v68 kernel still produces per-expert hidden states [M, 9, K_local], but
# the v110 kernel call is SKIPPED. Instead, the down GEMM + per-expert
# weighted combine + cross-rank all-reduce are executed in pure PyTorch
# using fp32 dequantized w2 (routed) + shared down weights. This is a
# diagnostic experiment: after 8 iterations of single-variable kernel
# changes all yielding AL=1.0, we want to know whether v110's down kernel
# is the AL killer. If AL recovers (>= 2.5) with this flag, v110 is at
# fault. If AL stays at 1.0, v68 (or wider integration) is broken.
#
# Side effect: when this flag is set, load_weights does NOT drop
# self.w2_weight / self.w2_weight_scaling_factor / shared down weights —
# we need them for the PyTorch dequant matmul. That ~doubles HBM usage
# on the routed down weight (the v110 packed copy is also held); fine
# for this diagnostic but unsuitable for production.
#
# Off by default to preserve iter-8 behavior.
_USE_PARENT_DOWN = os.environ.get("GLM5_USE_PARENT_DOWN", "0") == "1"

# [iter-10] PyTorch v68 reference toggle. When GLM5_USE_PYTORCH_V68=1, the
# v68 kernel call is SKIPPED. Instead, the routing (noaux_tc_op),
# per-expert dequant of routed + shared gate/up FP8 weights, per-128-col
# activation FP8 round-trip quant on the input (to match v68's actquant
# noise pattern), and silu(gate@x)*up@x for slot 0 (shared) + slots 1..8
# (top-K routed) are all executed in vectorized PyTorch.
#
# Combined with GLM5_USE_PARENT_DOWN=1, this gives a pure-PyTorch MoE
# chain with no custom kernel calls. If AL recovers >= 2.5 with both
# flags set, the AL killer lives in v68 and/or v110. If AL stays 1.0,
# the kernels are exonerated and the bug is in the Python integration
# glue / class swap / weight loading / MTP handling.
#
# Off by default. Side effect: when this flag is set, load_weights does
# NOT drop self.w3_w1_weight / self.w3_w1_weight_scaling_factor — the
# PyTorch ref needs them every forward call.
_USE_PYTORCH_V68 = os.environ.get("GLM5_USE_PYTORCH_V68", "0") == "1"

# [iter-15] Per-layer isolation toggles to disambiguate channel-2317
# divergence at L77 between v68 and v110. When set (CSV of layer_idx, e.g.
# "77" or "75,77"), the corresponding PyTorch reference replaces ONLY that
# kernel ONLY at the listed layers. The other kernel still fires normally,
# and on layers NOT in the list both kernels still fire. This lets us A/B
# the L77 disagreement: if A (v68→ref on L77 only) recovers AL, v68 owns
# the bug at L77; if B (v110→ref on L77 only) recovers AL, v110 owns it.
#
# Side effects (when EITHER list is non-empty):
#  - The corresponding weight tensors (w3_w1_* for v68, w2_* for v110) are
#    KEPT during load_weights so the PyTorch ref can dequant them every
#    forward. Mirrors the existing GLM5_USE_PYTORCH_V68 / GLM5_USE_PARENT_DOWN
#    weight-retention behavior.
#  - The v68 packed weights stay loaded too (since v68 still fires on the
#    non-isolated layers).
def _parse_layer_csv(env_name: str) -> "set[int]":
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return set()
    out: "set[int]" = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.add(int(tok))
        except ValueError:
            pass
    return out


_PYTORCH_V68_LAYERS: "set[int]" = _parse_layer_csv("GLM5_PYTORCH_V68_LAYERS")
_PARENT_DOWN_LAYERS: "set[int]" = _parse_layer_csv("GLM5_PARENT_DOWN_LAYERS")

# [iter-15] Optional element-wise dump of v68's hidden_out at chosen layer
# indices (CSV). Used by Method-A disambiguator: dump v68's hidden_out at
# L77 and offline-compare to PyTorch ref to see whether v68 is producing
# the wrong values for the dominant outlier channel. Off by default.
_DUMP_V68_HIDDEN_OUT_LAYERS: "set[int]" = _parse_layer_csv(
    "GLM5_DUMP_V68_HIDDEN_OUT_LAYERS")
_DUMP_V68_DIR = os.environ.get("GLM5_DUMP_V68_DIR", "/tmp/glm5_v68_dump")
_DUMP_V68_SKIP = int(os.environ.get("GLM5_DUMP_V68_SKIP", "5"))
_DUMP_V68_LIMIT = int(os.environ.get("GLM5_DUMP_V68_LIMIT", "1"))

# Effective gates for the load_weights drop logic. We must NOT drop the
# original w3_w1 / w2 weights whenever ANY layer needs the PyTorch ref —
# either globally (GLM5_USE_*) or per-layer (_LAYERS env). The same is true
# for the diff-test layer (already handled elsewhere).
_NEED_W3W1 = (_USE_PYTORCH_V68 or len(_PYTORCH_V68_LAYERS) > 0
              or len(_DUMP_V68_HIDDEN_OUT_LAYERS) > 0)
_NEED_W2 = _USE_PARENT_DOWN or len(_PARENT_DOWN_LAYERS) > 0


def _log_once(tag: str, *fields: str) -> None:
    """Print a per-process one-shot tag the first time a code path is hit."""
    flag = f"_glm5_logged_{tag}"
    if getattr(_log_once, flag, False):
        return
    setattr(_log_once, flag, True)
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "?")))
    print(f"[GLM5_SMALL_BATCH rank={rank}] {tag} {' '.join(fields)}", flush=True)


def _v68_op():
    return torch.ops.trtllm.glm5_expert_select_up_gate_silu


def _v110_op():
    return torch.ops.trtllm.glm5_expert_down_allreduce


def _v68_repack_op():
    return getattr(torch.ops.trtllm, "glm5_repack_weights_up_gate_silu", None)


def _v110_repack_op():
    return getattr(torch.ops.trtllm, "glm5_repack_weights_down", None)


# v110 workspace: 1 MB per rank is enough for M ≤ 4 × hidden=6144 × bf16 × 3 buffers.
# Round to 1 MiB for alignment safety.
_V110_WORKSPACE_BYTES = 1 << 20


class Glm5SmallBatchFusedMoE(TRTLLMGenFusedMoE):
    """MoE backend for the GLM-5 small-batch path.

    Each fused kernel hooks in via a paired ``_can_use_<phase>`` /
    ``_run_<phase>`` method. When ``_can_use_<phase>`` returns False (M > 4,
    or TP != 4), ``forward_impl`` delegates to ``super().forward_impl(...)``.

    PR 1 implementation pairs v68 (up-gate-silu) and v110 (down-AR): they are
    co-designed and either both fire or neither. Falling through to the parent
    on a partial path is unsupported because v68's output `[M, 9, K_local]` bf16
    has no clean drop-in seam in TRTLLM-Gen MoE (see INTEGRATION_PLAN.md).
    """

    # Class-level counter — increments per load_weights call. First instance
    # (counter==0 at load) gets to keep its routed/shared weights when
    # GLM5_DIFF_TEST=1 so the diff test can use them as reference. All other
    # instances drop weights as usual to fit HBM budget.
    _layer_load_counter: int = 0

    # PR 1 first-pass uses EXTERNAL_COMM (inherits TRTLLMGenFusedMoE's default).
    # The plan in INTEGRATION_PLAN.md calls for FUSED_COMM once v110 is actually
    # firing — but FUSED_COMM's scheduler calls `backend.run_moe(output_dtype=...)`
    # (moe_scheduler.py:1078), and TRTLLMGenFusedMoE.run_moe() does NOT accept
    # `output_dtype`. Until we also override run_moe to forward to v110, keep
    # EXTERNAL_COMM so the parent path works for the fall-through (when
    # _can_use_*_kernel returns False). Flip to FUSED_COMM in the same PR that
    # adds the run_moe override.
    scheduler_kind = MoESchedulerKind.EXTERNAL_COMM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Cached weight repacks (filled on first load_weights call).
        self._v68_w_gate_packed: Optional[torch.Tensor] = None
        self._v68_w_up_packed: Optional[torch.Tensor] = None
        self._v68_group_max_scale_gate: Optional[torch.Tensor] = None
        self._v68_group_max_scale_up: Optional[torch.Tensor] = None
        self._v110_w_down_packed: Optional[torch.Tensor] = None
        self._v110_w_down_group_scale: Optional[torch.Tensor] = None

        # Backref to the sibling shared_experts module (set by
        # Glm5SmallBatchMoE.__init__). Both v68 and v110 expect
        # `kPackedExpertCount = kNumExpertsTotal = 257` — the kernels assume
        # the shared expert is pre-merged into the same weight tensor at index
        # 256 of the [E, ...] dim. TRTLLMGenFusedMoE stores 256 routed only;
        # shared lives in a sibling `GatedMLP`. We pull the shared weights
        # at repack time via this backref.
        self._shared_experts_ref = None

        # v110 IPC sym-heap workspace — allocated lazily on first forward
        # because mapping isn't fully wired at __init__ time in all paths.
        self._v110_ipc: Optional[IpcMemory] = None
        self._v110_peer_ptrs: Optional[list] = None
        # NB: must start at 1 (not 0) because the v110 Lamport AR primitive
        # uses `flag` as a sentinel marker compared against the peer-buffer
        # high bits — and `IpcMemory.open_ipc_memory(set_to_zero=True)`
        # leaves those bits at 0. If `flag == 0` on the first call, the
        # consumer's spin-poll matches the zero-initialized state and
        # immediately reads value 0.0 instead of waiting for the peer
        # write, producing silently corrupt AR output (B-A=0.14 vs
        # ~0.003 expected in the single-layer diff test).
        self._v110_flag: int = 1

        # Diff-test mode: set in load_weights if this is the FIRST layer to
        # load and GLM5_DIFF_TEST=1. In that case we keep self.w3_w1_weight,
        # self.w2_weight, and their scaling factors instead of dropping them,
        # so _run_diff_test can dequantize them as the PyTorch ground truth.
        self._diff_test_layer: bool = False
        self._diff_test_done: bool = False

        # DeepSeekV3MoeRoutingMethod stores routed_scaling_factor on its
        # `routing_impl` attribute (no top-level alias). Look there first;
        # fall back to the routing method itself for other methods that
        # might expose it directly. Critical: GLM-5 uses 2.5, and getting
        # this wrong (e.g., defaulting to 1.0) scales every routed-expert
        # contribution by the wrong factor — silent correctness bug that
        # blows up MTP acceptance.
        impl = getattr(self.routing_method, "routing_impl", None)
        if impl is not None and hasattr(impl, "routed_scaling_factor"):
            self._routed_scaling_factor: float = float(impl.routed_scaling_factor)
        else:
            self._routed_scaling_factor: float = float(
                getattr(self.routing_method, "routed_scaling_factor", 1.0)
            )

    # ------------------------------------------------------------------
    # ExpertSelectUpGateSiLU (v68)
    # ------------------------------------------------------------------
    def _can_use_up_gate_silu_kernel(self, x: torch.Tensor,
                                     router_logits: torch.Tensor) -> bool:
        # PR 1 gate: M ≤ 4, bf16 activations, TP=4 (paired with v110).
        if x.shape[0] > 4:
            return False
        if x.dtype != torch.bfloat16:
            return False
        if self.parallel_size != 4:
            return False
        # Weights must have been repacked by load_weights.
        if self._v68_w_gate_packed is None:
            return False
        return True

    def _run_up_gate_silu(self, x: torch.Tensor,
                          router_logits: torch.Tensor):
        if _LOG_KERNEL_INVOCATION:
            _log_once(
                "v68",
                f"M={x.shape[0]}",
                f"K={x.shape[-1]}",
                f"tp={self.parallel_size}",
            )
        # mega_silu_v68 takes the pre-routing scores tensor (not post-sigmoid).
        # In TRTLLMGenFusedMoE.forward_impl, `router_logits` IS that pre-routing
        # tensor — it's the output of self.gate(hidden) at modeling_deepseekv3.py
        # (Deepseekv3MoE.compute_routed_output), before any sigmoid/topK.
        bias = self.routing_method.e_score_correction_bias
        topk_w, topk_i, hidden_out = _v68_op()(
            router_logits,                # scores [M, 256] fp32
            x,                            # hidden_in [M, 6144] bf16
            bias,                         # [256] fp32
            self._v68_w_gate_packed,
            self._v68_w_up_packed,
            self._v68_group_max_scale_gate,
            self._v68_group_max_scale_up,
            self._routed_scaling_factor,
        )
        return hidden_out, topk_i, topk_w

    # ------------------------------------------------------------------
    # Parent-down diagnostic path (iter-9)
    # ------------------------------------------------------------------
    # Per-expert on-the-fly dequant when GLM5_USE_PARENT_DOWN=1. We do NOT
    # cache an fp32 dequantized copy of routed w2 across layers — that would
    # be 3 GiB per layer × 75 layers = 225 GiB, far above HBM. Per-call
    # temporary is ~12 MiB per slot (dequant'd `[K, M_local]` fp32),
    # freed immediately after the matmul.
    _BLK = 128

    @staticmethod
    def _block_dequant_3d(w_fp8: torch.Tensor,
                          s_fp32: torch.Tensor) -> torch.Tensor:
        """[E, dim0, dim1] fp8 * [E, dim0/r0, dim1/r1] fp32 -> fp32 dense.

        Auto-detects per-axis repeat ratio so this works for both the
        standard 128x128 block layout (routed) and the per-row 1xN layout
        (shared) without bespoke handling per call site.
        """
        w_fp32 = w_fp8.to(torch.float32)
        E, K, Mw = w_fp32.shape
        sE, sK, sM = s_fp32.shape
        rep_K = K // sK if sK > 0 else 1
        rep_M = Mw // sM if sM > 0 else 1
        if rep_K * sK == K and rep_M * sM == Mw and sE == E:
            s_rep = (s_fp32.repeat_interleave(rep_K, dim=1)
                           .repeat_interleave(rep_M, dim=2))
            return w_fp32 * s_rep
        # Fallback: best-effort broadcast.
        return w_fp32 * s_fp32

    @staticmethod
    def _block_dequant_2d(w_fp8: torch.Tensor,
                          s_fp32: torch.Tensor) -> torch.Tensor:
        """[dim0, dim1] fp8 * [dim0/r0, dim1/r1] fp32 -> fp32 dense, with
        auto-detected per-axis repeat ratio (see _block_dequant_3d)."""
        w_fp32 = w_fp8.to(torch.float32)
        Mw, Kw = w_fp32.shape
        sM, sK = s_fp32.shape
        rep_M = Mw // sM if sM > 0 else 1
        rep_K = Kw // sK if sK > 0 else 1
        if rep_M * sM == Mw and rep_K * sK == Kw:
            s_rep = (s_fp32.repeat_interleave(rep_M, dim=0)
                           .repeat_interleave(rep_K, dim=1))
            return w_fp32 * s_rep
        # Fallback.
        return w_fp32 * s_fp32

    def _ensure_parent_down_weights(self) -> None:
        """Validate routed + shared down weights are accessible.

        We do NOT cache a dequantized fp32 copy: that would be
        ~3 GiB / layer × 75 layers = 225 GiB, far above any GPU's HBM.
        Instead we keep the original fp8 weights + fp32 scales on device
        (already there as nn.Parameters) and dequant per-expert on the
        fly inside _run_parent_down. Per-call temporary is ~12 MiB per
        slot, well within budget.

        This method just validates references and logs shapes once.
        """
        if getattr(self, "_parent_down_validated", False):
            return
        w2 = getattr(self, "w2_weight", None)
        w2_s = getattr(self, "w2_weight_scaling_factor", None)
        if w2 is None or w2_s is None:
            raise RuntimeError(
                "GLM5_USE_PARENT_DOWN=1 but self.w2_weight/scale missing — "
                "load_weights must skip the drop when this flag is on."
            )
        shared = self._shared_experts_ref
        if shared is None:
            raise RuntimeError(
                "GLM5_USE_PARENT_DOWN=1 but _shared_experts_ref is None — "
                "class-swap path may not have run."
            )
        shared_w_d = shared.down_proj.weight
        shared_s_d = shared.down_proj.weight_scale
        if shared_w_d is None or shared_s_d is None:
            raise RuntimeError(
                "GLM5_USE_PARENT_DOWN=1 but shared.down_proj weights missing."
            )
        # Log shapes once so we can confirm the layout the loader produced.
        _log_once("parent_down_shapes",
                  f"w2={tuple(w2.shape)}",
                  f"w2_s={tuple(w2_s.shape)}",
                  f"shared_w_d={tuple(shared_w_d.shape)}",
                  f"shared_s_d={tuple(shared_s_d.shape)}")
        # Pre-compute the auto-detected repeat ratios for the shared scale
        # tensor since its layout (K_blocks, 1) differs from the routed's
        # (K_blocks, M_blocks). Cache these as ints so we can do the
        # dequant in two interleave passes per call without re-detecting.
        K_sh, Msh = shared_w_d.shape
        sK_sh, sM_sh = shared_s_d.shape
        self._sh_rep_K = K_sh // sK_sh if sK_sh > 0 else 1
        self._sh_rep_M = Msh // sM_sh if sM_sh > 0 else 1
        Er, Kr, Mr = w2.shape
        sEr, sKr, sMr = w2_s.shape
        self._r_rep_K = Kr // sKr if sKr > 0 else 1
        self._r_rep_M = Mr // sMr if sMr > 0 else 1
        self._parent_down_validated = True

    def _run_parent_down(self,
                         hidden_out: torch.Tensor,
                         topk_indices: torch.Tensor,
                         topk_weights: torch.Tensor) -> torch.Tensor:
        """Parent-style down GEMM + weighted combine + AR (fp32 PyTorch).

        Replaces v110's fused down+combine+AR. Input:
          hidden_out [M, 9, K_local] bf16 or fp16 — v68's output.
            slot 0 = shared expert, slots 1-8 = top-K routed.
          topk_indices [M, 8] int — routed expert ids (slot 1-8 mapping).
          topk_weights [M, 8] fp32 — routed expert combine weights.

        Output: [M, K=6144] bf16, AR'd across TP ranks.

        Vectorized form: for each chunk-of-M tokens, gather the unique
        routed experts' fp8 weights, dequant them in one shot, and run
        a single bmm against all (token, slot) pairs at once. ~9× fewer
        Python iterations than the per-slot loop, no .item() syncs.

        Memory footprint per call:
          - routed dequant: M_chunk × 8 × K × M_local × 4 bytes = at
            M_chunk=4: 4 × 8 × 6144 × 512 × 4 = 384 MiB peak (freed
            after the bmm).
          - shared dequant: K × M_local × 4 bytes = 12 MiB.
        For M_chunk > 4 (prefill paths through this code) the bmm tensor
        gets larger. Acceptable for diagnostic; not production.
        """
        self._ensure_parent_down_weights()
        # FP8 weights + fp32 scales kept on the module (already on device).
        w2 = self.w2_weight                              # [E=256, K=6144, M_local=512] fp8
        w2_s = self.w2_weight_scaling_factor             # [E, K_blocks, M_blocks] fp32
        shared = self._shared_experts_ref
        shared_w_d = shared.down_proj.weight             # [K=6144, M_local=512] fp8
        shared_s_d = shared.down_proj.weight_scale       # [K, 1] or [K_blocks, M_blocks] fp32

        # Cached repeat ratios from _ensure_parent_down_weights.
        r_rep_K, r_rep_M = self._r_rep_K, self._r_rep_M
        sh_rep_K, sh_rep_M = self._sh_rep_K, self._sh_rep_M

        M = hidden_out.shape[0]
        K = shared_w_d.shape[0]
        M_local = shared_w_d.shape[1]
        TOP_K = topk_indices.shape[1]   # 8
        device = hidden_out.device
        # fp32 hidden states for the matmul.
        h32 = hidden_out.to(torch.float32)               # [M, 9, M_local]

        # ----- Slot 0 (shared expert) -----
        # Dequant shared down once per call (~12 MiB temporary).
        sh_w_fp32 = shared_w_d.to(torch.float32)
        sh_s_rep = (shared_s_d.repeat_interleave(sh_rep_K, dim=0)
                              .repeat_interleave(sh_rep_M, dim=1))
        w_down_sh_fp32 = sh_w_fp32 * sh_s_rep            # [K, M_local]
        # Shared-expert contribution for all M tokens in one matmul.
        out = h32[:, 0, :] @ w_down_sh_fp32.t()          # [M, K]

        # ----- Slots 1-8 (top-K routed) -----
        # Gather routed expert ids for the whole chunk (M × TOP_K of them).
        # Stay on device — avoids the .item() sync that dominated the
        # per-slot Python loop.
        idx_flat = topk_indices.to(torch.long).reshape(-1)        # [M*TOP_K]
        # Gather + dequant routed down weights for the experts this chunk
        # actually uses. Note: experts may repeat across tokens, and we
        # don't dedup (slot-wise dedup would save memory at small M when
        # tokens share experts; ignore for simplicity).
        w_gathered = w2.index_select(0, idx_flat).to(torch.float32)
        # [M*TOP_K, K, M_local] fp32
        s_gathered = w2_s.index_select(0, idx_flat)               # [M*TOP_K, sK, sM]
        # [iter-10] Vectorized dequant via reshape + broadcast (no
        # repeat_interleave materialization). Standard 128x128 block layout:
        # w[MTK, sK*rK, sM*rM] view as [MTK, sK, rK, sM, rM]; s view as
        # [MTK, sK, 1, sM, 1]; multiply broadcast.
        MTK = w_gathered.shape[0]
        K_dim = w_gathered.shape[1]
        Ml_dim = w_gathered.shape[2]
        sK = s_gathered.shape[1]
        sM = s_gathered.shape[2]
        if sK * r_rep_K == K_dim and sM * r_rep_M == Ml_dim:
            w_view = w_gathered.view(MTK, sK, r_rep_K, sM, r_rep_M)
            s_view = s_gathered.view(MTK, sK, 1, sM, 1)
            w_dq_routed = (w_view * s_view).view(MTK, K_dim, Ml_dim)
        else:
            # Fallback for non-standard layouts.
            s_rep = (s_gathered.repeat_interleave(r_rep_K, dim=1)
                               .repeat_interleave(r_rep_M, dim=2))
            w_dq_routed = w_gathered * s_rep                      # [M*TOP_K, K, M_local]
        # Hidden for routed slots: [M, TOP_K, M_local] → [M*TOP_K, M_local].
        h_routed = h32[:, 1:9, :].reshape(-1, M_local)            # [M*TOP_K, M_local]
        # Batched matmul: each row of w_dq_routed @ h_routed.unsqueeze(-1)
        # → [M*TOP_K, K, 1] → squeeze → [M*TOP_K, K].
        routed_partial = torch.bmm(w_dq_routed,
                                   h_routed.unsqueeze(-1)).squeeze(-1)
        # Apply topk weights and sum across the TOP_K slots per token.
        w_flat = topk_weights.to(torch.float32).reshape(-1, 1)    # [M*TOP_K, 1]
        routed_partial = routed_partial * w_flat                  # [M*TOP_K, K]
        routed_partial = routed_partial.reshape(M, TOP_K, K).sum(dim=1)
        out = out + routed_partial                                # [M, K]

        # Cross-rank AR (replaces v110's fused Lamport AR).
        import torch.distributed as dist
        if dist.is_initialized() and self.mapping.tp_size > 1:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out.to(torch.bfloat16)

    # ------------------------------------------------------------------
    # PyTorch v68 diagnostic path (iter-10)
    # ------------------------------------------------------------------
    # On-the-fly fp32 reference for v68's routing + up + gate + silu chain.
    # When GLM5_USE_PYTORCH_V68=1, skip the v68 op and compute
    # (topk_w, topk_i, hidden_out) in vectorized PyTorch — matching the
    # diff test's _ref_up_gate_silu math but vectorized (no .item() syncs,
    # one bmm per stage).
    def _ensure_pytorch_v68_weights(self) -> None:
        """Validate routed + shared gate/up weights are accessible.

        Per-call temporary is dominated by the routed dequant
        `[M*TOP_K, M_local, K]` fp32 = 4*8*512*6144*4 ≈ 400 MiB at M=4 chunk
        — same order as parent-down, fine for diagnostic.
        """
        if getattr(self, "_pytorch_v68_validated", False):
            return
        w3w1 = getattr(self, "w3_w1_weight", None)
        s3s1 = getattr(self, "w3_w1_weight_scaling_factor", None)
        if w3w1 is None or s3s1 is None:
            raise RuntimeError(
                "GLM5_USE_PYTORCH_V68=1 but self.w3_w1_weight/scale missing — "
                "load_weights must skip the drop when this flag is on."
            )
        shared = self._shared_experts_ref
        if shared is None:
            raise RuntimeError(
                "GLM5_USE_PYTORCH_V68=1 but _shared_experts_ref is None — "
                "class-swap path may not have run."
            )
        sh_gu = shared.gate_up_proj.weight
        sh_sgu = shared.gate_up_proj.weight_scale
        if sh_gu is None or sh_sgu is None:
            raise RuntimeError(
                "GLM5_USE_PYTORCH_V68=1 but shared.gate_up_proj weights missing."
            )
        _log_once("pytorch_v68_shapes",
                  f"w3w1={tuple(w3w1.shape)}",
                  f"s3s1={tuple(s3s1.shape)}",
                  f"sh_gu={tuple(sh_gu.shape)}",
                  f"sh_sgu={tuple(sh_sgu.shape)}")
        # Pre-compute the auto-detected repeat ratios. Standard 128x128 block
        # for routed; shared gate_up may use a different ratio per iter-9.
        Er, Mr, Kr = w3w1.shape
        sEr, sMr, sKr = s3s1.shape
        self._gu_r_rep_M = Mr // sMr if sMr > 0 else 1
        self._gu_r_rep_K = Kr // sKr if sKr > 0 else 1
        Msh, Ksh = sh_gu.shape
        sMsh, sKsh = sh_sgu.shape
        self._gu_sh_rep_M = Msh // sMsh if sMsh > 0 else 1
        self._gu_sh_rep_K = Ksh // sKsh if sKsh > 0 else 1
        self._pytorch_v68_validated = True

    def _fp8_actquant_roundtrip(self, x: torch.Tensor) -> torch.Tensor:
        """Per-128-col FP8 activation quant round-trip.

        Mirrors v68's internal actquant pattern (fp32 -> fp8 -> fp32 via
        per-block amax scale). Each contiguous 128 elements along the K
        (last) dim share a single scale = amax/448. Quantize to e4m3, then
        dequantize back to fp32. Result has the same noise pattern as v68's
        first stage so the downstream matmul reproduces v68's GEMM precision.

        Uses torch.ops.trtllm.fp8_quantize_1x128 when available; otherwise
        hand-rolls the same math.
        """
        # Hand-rolled per-128-col fp32->fp8->fp32 round-trip. The exposed
        # fp8_quantize_1x128 op returns packed bytes + scales but no inverse
        # path; rolling our own is simpler than reverse-engineering its scale
        # layout. Operates on the LAST dim assuming K is a multiple of 128.
        BLK = 128
        orig_shape = x.shape
        K = orig_shape[-1]
        assert K % BLK == 0, f"K={K} not divisible by {BLK}"
        x_blocked = x.reshape(-1, K // BLK, BLK).to(torch.float32)
        # amax per 128-col block.
        amax = x_blocked.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        # FP8 e4m3 max value is 448.
        scale = amax / 448.0
        scale_inv = 1.0 / scale
        # Quantize: multiply by 1/scale, clamp into e4m3 range, cast to fp8.
        q = (x_blocked * scale_inv).to(torch.float8_e4m3fn)
        # Dequantize: cast back to fp32, multiply by scale.
        deq = q.to(torch.float32) * scale
        return deq.reshape(orig_shape)

    def _run_pytorch_v68(self,
                        x: torch.Tensor,
                        router_logits: torch.Tensor,
                        bias_bf16: torch.Tensor):
        """PyTorch fp32 ref for v68 — routing + dequant + silu(gate)*up.

        Returns (topk_w, topk_i, hidden_out) matching v68's op signature:
          - topk_w [M, 8] fp32
          - topk_i [M, 8] int32
          - hidden_out [M, 9, K_local] fp16

        Routing: bf16-bias noaux_tc_op (matches v68's bias precision contract).
        Slot 0: shared expert. Slots 1-8: top-K routed.

        Vectorized: a single torch.bmm across (M*TOP_K) routed experts, a
        single linear for shared. No .item() syncs.

        Includes per-128-col FP8 activation quant round-trip on x BEFORE the
        gate/up matmuls to match v68's actquant noise pattern (this matters
        for v68/v110 noise-match diff testing; for the pure-PyTorch end-to-end
        bench it's defensive but harmless).
        """
        self._ensure_pytorch_v68_weights()
        w3w1 = self.w3_w1_weight                # [E=256, 2*M_local, K] fp8
        s3s1 = self.w3_w1_weight_scaling_factor # [E, 2*M_blocks, K_blocks] fp32
        shared = self._shared_experts_ref
        sh_gu = shared.gate_up_proj.weight      # [2*M_local, K] fp8
        sh_sgu = shared.gate_up_proj.weight_scale

        M = x.shape[0]
        K = x.shape[-1]
        E_routed, two_M_local, _ = w3w1.shape
        M_local = two_M_local // 2
        TOP_K = 8

        # ----- Routing: noaux_tc_op with bf16 bias (matches v68) -----
        scores_fp32 = router_logits.to(torch.float32).contiguous()
        bias_fp32 = bias_bf16.to(torch.float32).contiguous()
        topk_w, topk_i = torch.ops.trtllm.noaux_tc_op(
            scores_fp32, bias_fp32,
            1, 1, TOP_K, self._routed_scaling_factor,
        )
        # noaux_tc_op returns: topk_w fp32 [M, TOP_K], topk_i int32 [M, TOP_K]

        # ----- FP8 activation quant round-trip on x -----
        # Match v68's per-128-col actquant noise: fp32 -> fp8 -> fp32 via
        # per-block scale. Input x is bf16; cast to fp32 first.
        x_fp32 = x.to(torch.float32).contiguous()
        x_aq = self._fp8_actquant_roundtrip(x_fp32)   # [M, K] fp32 (post round-trip)

        # ----- Slot 0 (shared expert) -----
        # Shared gate_up layout: [2*M_local, K] with first half = GATE,
        # second half = UP (per loader convention notes in load_weights).
        half_sh = sh_gu.shape[0] // 2
        sh_gu_fp32 = sh_gu.to(torch.float32)
        sh_sgu_rep = (sh_sgu.repeat_interleave(self._gu_sh_rep_M, dim=0)
                            .repeat_interleave(self._gu_sh_rep_K, dim=1))
        sh_w_dq = sh_gu_fp32 * sh_sgu_rep             # [2*M_local, K]
        w_gate_sh = sh_w_dq[:half_sh, :]              # [M_local, K]
        w_up_sh = sh_w_dq[half_sh:, :]                # [M_local, K]
        # Matmul: x_aq [M, K] @ w_gate_sh.T [K, M_local] -> [M, M_local].
        g_sh = x_aq @ w_gate_sh.t()
        u_sh = x_aq @ w_up_sh.t()
        slot0 = torch.nn.functional.silu(g_sh) * u_sh   # [M, M_local]

        # ----- Slots 1-8 (top-K routed) -----
        # Routed layout: [E, 2*M_local, K] with first half = UP, second half = GATE.
        # Gather routed experts the chunk actually uses: [M*TOP_K] expert IDs.
        idx_flat = topk_i.to(torch.long).reshape(-1)             # [M*TOP_K]
        w_routed = w3w1.index_select(0, idx_flat).to(torch.float32)
        # [M*TOP_K, 2*M_local, K] fp32
        s_routed = s3s1.index_select(0, idx_flat)                # [M*TOP_K, 2*M_blocks, K_blocks]

        # Vectorized dequant via reshape + broadcast, avoiding the
        # `repeat_interleave` of `s_routed` to the full `[M*TOP_K, 2*M_local, K]`
        # which is a ~10x slower op in practice. Standard 128x128 block layout:
        # reshape w into (MTK, M_blocks, BLK_M, K_blocks, BLK_K), broadcast-mul
        # with s[..., None, ..., None], reshape back. Single elementwise op.
        rM, rK = self._gu_r_rep_M, self._gu_r_rep_K
        MTK = w_routed.shape[0]
        two_M_local_dim = w_routed.shape[1]
        s_M_blocks = s_routed.shape[1]
        s_K_blocks = s_routed.shape[2]
        # Sanity: factors line up.
        if s_M_blocks * rM == two_M_local_dim and s_K_blocks * rK == K:
            w_view = w_routed.view(MTK, s_M_blocks, rM, s_K_blocks, rK)
            s_view = s_routed.view(MTK, s_M_blocks, 1, s_K_blocks, 1)
            w_routed_dq = (w_view * s_view).view(MTK, two_M_local_dim, K)
        else:
            # Fallback: explicit repeat_interleave if the shape factoring fails.
            s_routed_rep = (s_routed.repeat_interleave(rM, dim=1)
                                    .repeat_interleave(rK, dim=2))
            w_routed_dq = w_routed * s_routed_rep
        half_r = w_routed_dq.shape[1] // 2
        w_up_r = w_routed_dq[:, :half_r, :]                      # [M*TOP_K, M_local, K]
        w_gate_r = w_routed_dq[:, half_r:, :]                    # [M*TOP_K, M_local, K]

        # x repeated per slot for batched matmul: [M, K] -> [M*TOP_K, K, 1]
        x_aq_per_slot = x_aq.unsqueeze(1).expand(M, TOP_K, K).reshape(M * TOP_K, K, 1)
        # bmm: [M*TOP_K, M_local, K] @ [M*TOP_K, K, 1] -> [M*TOP_K, M_local, 1]
        g_r = torch.bmm(w_gate_r, x_aq_per_slot).squeeze(-1)     # [M*TOP_K, M_local]
        u_r = torch.bmm(w_up_r, x_aq_per_slot).squeeze(-1)
        slots_routed = torch.nn.functional.silu(g_r) * u_r       # [M*TOP_K, M_local]
        slots_routed = slots_routed.reshape(M, TOP_K, M_local)

        # ----- Assemble [M, 9, M_local] hidden_out -----
        out_dtype = torch.float16
        hidden_out = torch.empty(M, 9, M_local, dtype=out_dtype, device=x.device)
        hidden_out[:, 0, :] = slot0.to(out_dtype)
        hidden_out[:, 1:, :] = slots_routed.to(out_dtype)
        return topk_w, topk_i.to(torch.int32), hidden_out

    # ------------------------------------------------------------------
    # ExpertDownAllReduce (v110)
    # ------------------------------------------------------------------
    def _can_use_down_allreduce_kernel(self,
                                       inter: torch.Tensor) -> bool:
        # Paired with up-gate-silu; if we got here, up-gate-silu already
        # passed its gate, so most checks are redundant. Verify weights present.
        if self._v110_w_down_packed is None:
            return False
        return True

    def _ensure_v110_workspace(self) -> None:
        """Allocate the v110 IPC sym-heap on first launch."""
        if self._v110_ipc is not None:
            return
        mapping = self.mapping
        is_p2p = can_access_peer(mapping)
        self._v110_ipc = IpcMemory(mapping, _V110_WORKSPACE_BYTES, is_p2p)
        self._v110_peer_ptrs = self._v110_ipc.serialize()
        if _LOG_KERNEL_INVOCATION:
            _log_once(
                "v110_workspace",
                f"tp={mapping.tp_size}",
                f"peer_ptrs={len(self._v110_peer_ptrs)}",
            )

    def _run_down_allreduce(self, inter: torch.Tensor,
                            topk_indices: torch.Tensor,
                            topk_weights: torch.Tensor,
                            residual: torch.Tensor):
        self._ensure_v110_workspace()
        # v110 thop now takes 8 peer_ptr args (extended to support TP=8).
        # At TP=4 the trailing 4 are zero and never dereferenced by the kernel
        # (the peer loops use `for(p < num_peers)`). At TP=8 all 8 are used.
        peers = list(self._v110_peer_ptrs)
        while len(peers) < 8:
            peers.append(0)
        rank = int(self.mapping.tp_rank)
        # add_residual_on_rank0_only: only rank 0 actually fuses the residual
        # add into the AR (others contribute zero residual). 16 template
        # instantiations now: (kAddRes ∈ {false,true}) × (kMyRank ∈ {0..3}) for
        # TP=4, plus (kMyRank ∈ {0..7}) for TP=8.
        add_res = (rank == 0)
        if _LOG_KERNEL_INVOCATION:
            _log_once(
                "v110",
                f"M={inter.shape[0]}",
                f"rank={rank}",
                f"tp={self.mapping.tp_size}",
                f"flag={self._v110_flag}",
            )
        out = _v110_op()(
            inter,                              # [M, 9, K_local] fp16
            topk_indices.to(torch.int32),       # [M, 8] int32
            topk_weights.to(torch.float32),     # [M, 8] fp32
            residual,                           # [M, 6144] bf16
            self._v110_w_down_packed,           # fp8
            self._v110_w_down_group_scale,      # fp32
            add_res,                            # add_residual_on_rank0_only
            rank,
            int(peers[0]), int(peers[1]), int(peers[2]), int(peers[3]),
            int(peers[4]), int(peers[5]), int(peers[6]), int(peers[7]),
            int(self.mapping.tp_size),
            int(self._v110_flag),
        )
        # Lamport sequence number — monotonically increasing per call (the
        # kernel reads `flag` to discriminate fresh vs stale peer writes). Do
        # NOT mod-down here; v110 uses the raw int value for sequencing.
        self._v110_flag += 1
        return out

    # ------------------------------------------------------------------
    # Glue
    # ------------------------------------------------------------------
    # ConfigurableMoE's scheduler dispatches to backend.run_moe(...) directly
    # (moe_scheduler.py:471), bypassing forward_impl entirely. So the v68/v110
    # dispatch MUST live in run_moe — not forward_impl. (We keep forward_impl
    # as a tiny shim that just delegates upward, in case any code path comes
    # in through ConfigurableMoE.forward_impl directly.)
    def forward_impl(self, x, router_logits, *args, **kwargs):
        return super().forward_impl(x, router_logits, *args, **kwargs)

    # ---- Force fused (in-kernel) routing -----------------------------
    # _supports_load_balancer() = False tells the scheduler NOT to call
    # routing_method.apply() — v68 does routing internally and outputs its
    # own topk_indices/topk_weights. Skipping the scheduler-side routing
    # also gives us the raw `router_logits` (fp32 pre-sigmoid scores) in
    # run_moe, which is what v68 expects.
    #
    # However, when packed weights aren't loaded (e.g., MTP draft layer
    # constructed via DeepseekV3MTP, OR main layers when GLM5_MAIN_USE_PARENT=1
    # makes the layer use plain DeepseekV3DecoderLayer), run_moe falls
    # through to parent TRTLLMGenFusedMoE. The parent's in-kernel routing
    # path uses the kernel's own routing precision, which empirically
    # diverges from the scheduler-applied routing (e_score_correction_bias
    # dtype + fp32 sigmoid+topK) enough to cost ~30% MTP acceptance.
    # When the v68 packed kernels are not going to fire, defer routing to
    # the scheduler (load-balancer path) so the fallback matches the
    # baseline TRTLLM path exactly.
    def _supports_load_balancer(self) -> bool:
        # Packed-weights set during load_weights (see _pack_v68_weights).
        # If absent, we'll fall through in run_moe — let the scheduler route.
        # [iter-15-FIX] Try the lazy repack first so the deferred-repack path
        # doesn't lose its packed-weight gate to this early check.
        self._lazy_repack_if_needed()
        return self._v68_w_gate_packed is None or self._v110_w_down_packed is None

    # ---- No upstream quantization ------------------------------------
    # The scheduler calls backend.quantize_input(x) before run_moe
    # (moe_scheduler.py:467), which for the FP8 block-scale path would
    # convert x to fp8. v68 needs bf16 input, so we override to pass-through.
    #
    # When packed weights aren't loaded, we fall through to parent which
    # expects either fp8 + x_sf (load-balancer path) or bf16 + x_sf=None
    # (fused-routing path). To keep the fallback bit-identical to the
    # baseline TRTLLM path, delegate to the parent's quantize_input when
    # packed kernels won't fire.
    def quantize_input(self, x, post_quant_comm: bool = True):
        # [iter-15-FIX] Lazy repack may also need to fire here — quantize_input
        # is called BEFORE run_moe each scheduler step, and the `_v68_*_packed
        # is None` check below would otherwise route us to the parent path
        # (slow + breaks the v68/v110 chain).
        self._lazy_repack_if_needed()
        if self._v68_w_gate_packed is None or self._v110_w_down_packed is None:
            return super().quantize_input(x, post_quant_comm=post_quant_comm)
        return x, None

    # ------------------------------------------------------------------
    # [iter-15-FIX] Lazy repack — finishes the v68/v110 repacks that were
    # deferred at load_weights time because the sibling shared_experts
    # weight_scale was still an empty placeholder. The TRTLLM weight loader
    # walks named_modules() in registration order: `experts` is registered
    # BEFORE `shared_experts` in DeepseekV3MoE.__init__, so when our
    # Glm5SmallBatchFusedMoE.load_weights runs, the sibling
    # shared.{gate_up_proj,down_proj}.weight_scale is still the
    # `torch.empty(...)` placeholder (all zeros / arbitrary memory).
    #
    # Without this fix, the v68 repack packs ZEROS as the shared expert's
    # gate/up scales, and v68's slot 0 (shared) produces zero hidden_out;
    # similarly v110's slot 0 down GEMM dequantizes shared weights with
    # zero scales and contributes zero to the output. That zero shared
    # contribution is the L77 channel-2317 "systematic suppression" found
    # in iter-14 (channel 2317 is fed heavily by the shared expert at L77).
    #
    # By the time forward fires, shared.load_weights has run and the scale
    # values are the real loaded amax/448 factors. We re-read them now and
    # finish the repack.
    # ------------------------------------------------------------------
    @staticmethod
    def _requantize_to_128x128_blocks(
            w_fp8: torch.Tensor,
            s_arbitrary: torch.Tensor) -> "tuple[torch.Tensor, torch.Tensor]":
        """[iter-15-FIX] Re-quantize an FP8 weight with arbitrary scale
        layout into a 128x128-block-scaled FP8 weight that the v68/v110
        repack ops accept.

        The shared expert's `gate_up_proj.weight_scale` and
        `down_proj.weight_scale` have layouts that differ from the routed
        experts' `[K_blocks=K/128, M_blocks=M/128]` per-128x128-block
        scale:
          - GLM-5 shared gate_up: `[M=2*M_local, K_blocks=K/512]`
            (per-row × per-512-K-block).
          - GLM-5 shared down: `[K=K, M_blocks=1]` (per-row, single col block).
        To pack the shared expert into the 257-expert tensor that the v68
        / v110 C++ repack ops consume, we have to convert layouts.

        Procedure (handles any rectangular `s` via auto-detect):
          - dequantize: `W_f32 = w_fp8.f32 * s_broadcast` where
            `s_broadcast` is `s_arbitrary` repeated-interleaved up to
            `w_fp8.shape`.
          - block amax: `S_blk[kb, mb] = amax(W_f32[kb*128:.., mb*128:..])
            / 448`.
          - re-quantize: `Q_new_fp8 = round((W_f32 / S_blk_full).clamp(-448, 448))`.

        Returns (new fp8 weight, fp32 [out/128, in/128] scale).

        Lossy vs the original scale (a block-max upper-bounds the per-row
        scale within the block, so values shrink slightly under
        re-quantization). The alternative — teaching repack_v110/v68 to
        accept mixed layouts — is far more invasive.
        """
        BLK = 128
        assert w_fp8.dim() == 2, "expected 2-D weight"
        out_dim, in_dim = w_fp8.shape
        assert out_dim % BLK == 0 and in_dim % BLK == 0, (
            f"shape ({out_dim},{in_dim}) not divisible by 128")

        # Auto-detect repeat ratios in both dims so the dequant works for
        # ANY rectangular scale tensor (per-row, per-col, per-block, etc.).
        sO, sI = s_arbitrary.shape if s_arbitrary.dim() == 2 else (
            s_arbitrary.shape[0], 1)
        if s_arbitrary.dim() == 1:
            s_arbitrary = s_arbitrary.unsqueeze(-1)
        rep_O = out_dim // sO if sO > 0 else 1
        rep_I = in_dim // sI if sI > 0 else 1
        if rep_O * sO != out_dim or rep_I * sI != in_dim:
            raise RuntimeError(
                f"Cannot align shared scale {tuple(s_arbitrary.shape)} "
                f"to weight {tuple(w_fp8.shape)}")
        s_full = (s_arbitrary.to(torch.float32)
                  .repeat_interleave(rep_O, dim=0)
                  .repeat_interleave(rep_I, dim=1))
        w_f32 = w_fp8.to(torch.float32) * s_full
        # block amax via view; shape (out_dim//BLK, BLK, in_dim//BLK, BLK).
        Ob, Ib = out_dim // BLK, in_dim // BLK
        w_view = w_f32.view(Ob, BLK, Ib, BLK)
        block_amax = w_view.abs().amax(dim=(1, 3))  # [Ob, Ib]
        block_scale = block_amax / 448.0
        block_scale = torch.where(block_scale > 0, block_scale,
                                  torch.ones_like(block_scale))
        # Broadcast block scale up to full shape for the re-quant.
        bs_full = block_scale.unsqueeze(1).unsqueeze(3).expand(
            Ob, BLK, Ib, BLK).reshape(out_dim, in_dim)
        q_new = (w_f32 / bs_full).clamp(min=-448.0, max=448.0).to(
            torch.float8_e4m3fn)
        return q_new.contiguous(), block_scale.contiguous()

    def post_load_weights(self) -> None:
        """TRTLLM calls this on every module after the full weight load
        finishes (model_loader.py:491). At this point both the experts
        AND shared_experts have completed their per-Linear load_weights,
        so the sibling shared.{gate_up,down}_proj.weight_scale is valid.
        Finish the deferred repack here, BEFORE any forward pass / CUDA
        graph capture.
        """
        self._lazy_repack_if_needed()
        super().post_load_weights()

    def _lazy_repack_if_needed(self) -> None:
        """If load_weights deferred either repack (placeholder shared scale),
        complete it now using the now-loaded shared weight_scale.

        Safe under CUDA graph capture: it skips itself and waits for a
        non-capturing call. Normally fires from post_load_weights() once,
        before any forward pass / capture.
        """
        needs_v68 = (self._v68_w_gate_packed is None
                     and getattr(self, "_v68_pending_w_gu_routed", None)
                     is not None)
        needs_v110 = (self._v110_w_down_packed is None
                      and getattr(self, "_v110_pending_w_down_routed", None)
                      is not None)
        if not (needs_v68 or needs_v110):
            return
        try:
            if torch.cuda.is_current_stream_capturing():
                return
        except Exception:
            pass
        shared = self._shared_experts_ref
        if shared is None:
            _log_once("iter15_lazy_repack_no_shared",
                      "_shared_experts_ref is None — cannot complete lazy "
                      "repack")
            return
        # Re-read shared weights (Parameter values were overwritten in-place
        # by shared.load_weights after our load_weights deferral).
        repack_v68 = _v68_repack_op()
        repack_v110 = _v110_repack_op()
        if needs_v68:
            try:
                shared_w_gu = shared.gate_up_proj.weight
                shared_s_gu = shared.gate_up_proj.weight_scale
                if (shared_w_gu is None or shared_s_gu is None
                        or float(shared_s_gu.abs().max()) == 0.0):
                    _log_once(
                        "iter15_lazy_v68_scale_still_zero",
                        f"shared.gate_up_proj.weight_scale still "
                        f"all-zero (shape={tuple(shared_s_gu.shape)}); "
                        "v68 will fall back to parent. "
                        "Investigate shared load order.")
                else:
                    w_gu = self._v68_pending_w_gu_routed
                    s_gu = self._v68_pending_s_gu_routed
                    half = w_gu.shape[1] // 2
                    w_up_routed   = w_gu[:, :half, :].contiguous()
                    w_gate_routed = w_gu[:, half:, :].contiguous()
                    half_s = s_gu.shape[1] // 2
                    s_up_routed   = s_gu[:, :half_s, :].contiguous()
                    s_gate_routed = s_gu[:, half_s:, :].contiguous()
                    shared_half = shared_w_gu.shape[0] // 2
                    w_gate_shared_raw = shared_w_gu[:shared_half, :].contiguous()
                    w_up_shared_raw   = shared_w_gu[shared_half:, :].contiguous()
                    _log_once(
                        "iter15_lazy_v68_shared_shapes",
                        f"shared_w_gu={tuple(shared_w_gu.shape)} "
                        f"shared_s_gu={tuple(shared_s_gu.shape)} "
                        f"routed_s_up_target={tuple(s_up_routed.shape[1:])}")
                    # Try to split the shared scale at the same offset as
                    # the weight (whichever axis matches `shared_half`).
                    s_per_gate = s_per_up = None
                    if shared_s_gu.shape[0] == shared_w_gu.shape[0]:
                        # Per-row scale on the M axis (e.g., GLM-5
                        # FP8Rowwise with per-K-block: [1024, K_blocks]).
                        s_per_gate = shared_s_gu[:shared_half, :].contiguous()
                        s_per_up   = shared_s_gu[shared_half:, :].contiguous()
                    elif shared_s_gu.dim() == 2 and (
                            shared_s_gu.shape[0] * 2 == shared_w_gu.shape[0]
                            or shared_s_gu.shape[0] * 256 == shared_w_gu.shape[0]
                            or shared_s_gu.shape[0] * 128 == shared_w_gu.shape[0]):
                        # Per-block on the M axis (e.g., [8, 48]).
                        h_s = shared_s_gu.shape[0] // 2
                        s_per_gate = shared_s_gu[:h_s, :].contiguous()
                        s_per_up   = shared_s_gu[h_s:, :].contiguous()
                    else:
                        # Fallback: try halving along last dim.
                        h_s = shared_s_gu.shape[-1] // 2
                        s_per_gate = shared_s_gu[..., :h_s].contiguous()
                        s_per_up   = shared_s_gu[..., h_s:].contiguous()

                    # Re-quantize each shared half into the routed
                    # `[M_local/128, K/128]` per-128x128-block layout.
                    # `_requantize_to_128x128_blocks` handles any
                    # rectangular `s_per_*` via auto-detect.
                    w_gate_shared_blk, s_gate_shared_blk = \
                        self._requantize_to_128x128_blocks(
                            w_gate_shared_raw, s_per_gate)
                    w_up_shared_blk, s_up_shared_blk = \
                        self._requantize_to_128x128_blocks(
                            w_up_shared_raw, s_per_up)
                    w_gate_shared = w_gate_shared_blk.unsqueeze(0).contiguous()
                    w_up_shared   = w_up_shared_blk.unsqueeze(0).contiguous()
                    s_gate_shared = s_gate_shared_blk.unsqueeze(0).contiguous()
                    s_up_shared   = s_up_shared_blk.unsqueeze(0).contiguous()
                    _log_once(
                        "iter15_v68_shared_requant",
                        f"shared gate scale {tuple(s_per_gate.shape)} → "
                        f"block {tuple(s_gate_shared_blk.shape)}; "
                        f"up scale {tuple(s_per_up.shape)} → "
                        f"block {tuple(s_up_shared_blk.shape)}")
                    w_gate_257 = torch.cat([w_gate_routed, w_gate_shared], dim=0).contiguous()
                    w_up_257   = torch.cat([w_up_routed,   w_up_shared],   dim=0).contiguous()
                    s_gate_257 = torch.cat([s_gate_routed, s_gate_shared], dim=0).contiguous()
                    s_up_257   = torch.cat([s_up_routed,   s_up_shared],   dim=0).contiguous()
                    self._v68_w_gate_packed, self._v68_group_max_scale_gate = \
                        repack_v68(w_gate_257, s_gate_257)
                    self._v68_w_up_packed, self._v68_group_max_scale_up = \
                        repack_v68(w_up_257, s_up_257)
                    _log_once("iter15_lazy_v68_repack_ok",
                              f"gate_input={tuple(w_gate_257.shape)} "
                              f"shared_scale_min={float(shared_s_gu.abs().min()):.5g} "
                              f"shared_scale_max={float(shared_s_gu.abs().max()):.5g}")
                    del w_gate_routed, w_up_routed, s_gate_routed, s_up_routed
                    del w_gate_shared, w_up_shared, s_gate_shared, s_up_shared
                    del w_gate_257, w_up_257, s_gate_257, s_up_257
                    # Drop the pending routed refs to release HBM (~1.5 GiB/layer
                    # at TP=4). Skip on diff-test layer or when
                    # GLM5_USE_PYTORCH_V68 path needs them.
                    if not self._diff_test_layer and not _NEED_W3W1:
                        if hasattr(self, "w3_w1_weight"):
                            self.w3_w1_weight = None
                        if hasattr(self, "w3_w1_weight_scaling_factor"):
                            self.w3_w1_weight_scaling_factor = None
                    self._v68_pending_w_gu_routed = None
                    self._v68_pending_s_gu_routed = None
                    torch.cuda.empty_cache()
            except Exception as e:
                _log_once("iter15_lazy_v68_repack_failed", str(e)[:160])
        if needs_v110:
            try:
                shared_w_down = shared.down_proj.weight
                shared_s_down = shared.down_proj.weight_scale
                if (shared_w_down is None or shared_s_down is None
                        or float(shared_s_down.abs().max()) == 0.0):
                    _log_once(
                        "iter15_lazy_v110_scale_still_zero",
                        f"shared.down_proj.weight_scale still all-zero "
                        f"(shape={tuple(shared_s_down.shape)}); v110 will "
                        "fall back to parent. Investigate shared load order.")
                else:
                    w_down = self._v110_pending_w_down_routed
                    s_down = self._v110_pending_s_down_routed
                    # Shared.down_proj uses a DIFFERENT scale layout from
                    # routed: per-row `[K, 1]` (FP8Rowwise) vs per-128-block
                    # `[K_blocks=48, M_blocks=4]` (FP8BlockScales). The v110
                    # repack op expects all 257 experts in the same per-block
                    # layout. Re-quantize the shared expert into block scales
                    # so the cat / repack works.
                    if (shared_s_down.shape == s_down.shape[1:]):
                        # Already block-scale; just unsqueeze and cat.
                        w_down_shared = shared_w_down.unsqueeze(0).contiguous()
                        s_down_shared = shared_s_down.unsqueeze(0).contiguous()
                    else:
                        _log_once(
                            "iter15_shared_down_requant",
                            f"shared down scale layout {tuple(shared_s_down.shape)} "
                            f"!= routed {tuple(s_down.shape[1:])} — "
                            "re-quantizing shared into 128x128-block layout")
                        w_new, s_new = self._requantize_to_128x128_blocks(
                            shared_w_down, shared_s_down)
                        w_down_shared = w_new.unsqueeze(0).contiguous()
                        s_down_shared = s_new.unsqueeze(0).contiguous()
                    w_down_257 = torch.cat([w_down, w_down_shared], dim=0).contiguous()
                    s_down_257 = torch.cat([s_down, s_down_shared], dim=0).contiguous()
                    self._v110_w_down_packed, self._v110_w_down_group_scale = \
                        repack_v110(w_down_257, s_down_257)
                    _log_once("iter15_lazy_v110_repack_ok",
                              f"down_input={tuple(w_down_257.shape)} "
                              f"scale_cat={tuple(s_down_257.shape)} "
                              f"shared_scale_min={float(shared_s_down.abs().min()):.5g} "
                              f"shared_scale_max={float(shared_s_down.abs().max()):.5g}")
                    del w_down_shared, s_down_shared
                    del w_down_257, s_down_257
                    if not self._diff_test_layer and not _NEED_W2:
                        if hasattr(self, "w2_weight"):
                            self.w2_weight = None
                        if hasattr(self, "w2_weight_scaling_factor"):
                            self.w2_weight_scaling_factor = None
                    self._v110_pending_w_down_routed = None
                    self._v110_pending_s_down_routed = None
                    torch.cuda.empty_cache()
            except Exception as e:
                _log_once("iter15_lazy_v110_repack_failed", str(e)[:160])

    # ---- Chunked run_moe (M=4 per chunk) -----------------------------
    def run_moe(self,
                x,
                token_selected_experts=None,
                token_final_scales=None,
                x_sf=None,
                router_logits=None,
                do_finalize: bool = True,
                moe_output=None):
        """v68 + v110 dispatch, chunked at M=4.

        v68 hard-bound at M <= 4 per launch (its persistent-CTA grid is
        sized for small M). For prefill (M >> 4) we slice x and
        router_logits into M=4 sub-batches, run v68+v110 per chunk, and
        concatenate the outputs. Decode (M=4 already) is one chunk.

        If the v68/v110 packed weights aren't loaded for this layer (e.g.,
        OOM during load_weights skipped the repack), we still try — if it
        crashes the bench, that's a real bug; we want to know rather than
        silently fall back.
        """
        # [iter-15-FIX] Lazy repack: if load_weights deferred the repack
        # because the shared scale was an empty placeholder, finish the
        # repack now that the model is fully loaded. This MUST happen
        # before the dispatch check below so we don't fallback unnecessarily.
        self._lazy_repack_if_needed()

        if self._v68_w_gate_packed is None or self._v110_w_down_packed is None:
            # Packed weights missing — this happens for the MTP draft layers
            # constructed via DeepseekV3MTP (not Glm5SmallBatchDecoderLayer),
            # which skip our class-swap + backref setup, so their
            # load_weights merge is skipped. For those layers we fall through
            # to the parent TRTLLMGenFusedMoE path. Main MoE layers always
            # have packed weights and run the new kernels.
            _log_once("fallback_no_packed",
                      "run_moe fall-through (likely MTP draft layer)")
            return super().run_moe(x,
                                   token_selected_experts=token_selected_experts,
                                   token_final_scales=token_final_scales,
                                   x_sf=x_sf,
                                   router_logits=router_logits,
                                   do_finalize=do_finalize,
                                   moe_output=moe_output)

        # router_logits is required for v68; with _supports_load_balancer()
        # returning False, the scheduler is supposed to pass it through.
        if router_logits is None:
            raise RuntimeError(
                "Glm5SmallBatchFusedMoE.run_moe: router_logits is None — "
                "expected the scheduler to pass it (we returned False from "
                "_supports_load_balancer)."
            )

        # Pad x to bf16 if it isn't (defensive — quantize_input override should
        # have left it bf16).
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        # v68 requires bias in bf16 (TORCH_CHECK at mega_kernel_v68.cu:1179).
        # TRT-LLM upcasts e_score_correction_bias to fp32 on load — cast it
        # back here. Cached on the instance to avoid re-casting every call.
        if getattr(self, "_cached_bias_bf16", None) is None:
            bias_src = self.routing_method.e_score_correction_bias
            object.__setattr__(self, "_cached_bias_bf16",
                               bias_src.to(torch.bfloat16).contiguous())
        bias = self._cached_bias_bf16

        # Diff-test one-shot: only on the first MoE layer + first call.
        # Compares routing, v68 per-slot hidden_out, and v110 final output
        # against PyTorch references built from the dequantized weights.
        # Skipped under CUDA graph capture because the diff path mixes Python
        # control flow with kernel launches that the capture would record
        # incorrectly (use cuda_graph_config.batch_sizes=[] for the test).
        # Runs AFTER _cached_bias_bf16 init so the diff path can use it.
        if (self._diff_test_layer and not self._diff_test_done
                and not torch.cuda.is_current_stream_capturing()):
            self._diff_test_done = True
            try:
                self._run_diff_test(x, router_logits)
            except Exception as e:
                rank = int(self.mapping.tp_rank)
                print(f"[GLM5_DIFF rank={rank}] diff test crashed: "
                      f"{type(e).__name__}: {e}", flush=True)

        chunk_size = 4   # max v110 supports per launch.
        total_m = x.shape[0]
        chunks = []
        # [iter-15] Per-layer overrides — when the global flag is off but the
        # layer-CSV lists this layer, route through the PyTorch ref for that
        # kernel only at this layer. `self.layer_idx` is set by the
        # `FusedMoE.__init__` chain (see interface.py:272). Defensive guard
        # for layer_idx is None (would not normally hit but keeps the path
        # safe under odd module construction orders).
        _li = getattr(self, "layer_idx", None)
        _use_pytorch_v68_here = _USE_PYTORCH_V68 or (
            _li is not None and _li in _PYTORCH_V68_LAYERS)
        _use_parent_down_here = _USE_PARENT_DOWN or (
            _li is not None and _li in _PARENT_DOWN_LAYERS)
        if (_li is not None
                and (_li in _PYTORCH_V68_LAYERS or _li in _PARENT_DOWN_LAYERS)):
            _log_once(
                f"iter15_per_layer_override_l{_li}",
                f"layer_idx={_li}",
                f"pytorch_v68={_use_pytorch_v68_here}",
                f"parent_down={_use_parent_down_here}",
            )
        for start in range(0, total_m, chunk_size):
            end = min(start + chunk_size, total_m)
            x_chunk = x[start:end].contiguous()
            scores_chunk = router_logits[start:end].contiguous().to(torch.float32)

            # v68: routing + up + gate + silu (fused).
            # [iter-10] GLM5_USE_PYTORCH_V68=1 skips the v68 op and computes
            # routing + up + gate + silu in vectorized PyTorch. Combined with
            # GLM5_USE_PARENT_DOWN=1 this gives a pure-PyTorch MoE chain.
            # [iter-15] _use_pytorch_v68_here also honors per-layer CSV.
            if _use_pytorch_v68_here:
                if _LOG_KERNEL_INVOCATION:
                    _log_once("pytorch_v68_inference",
                              f"M={x_chunk.shape[0]}",
                              f"K={x_chunk.shape[-1]}",
                              f"tp={self.parallel_size}",
                              "(v68 skipped)")
                topk_w, topk_i, hidden_out = self._run_pytorch_v68(
                    x_chunk, scores_chunk, bias)
            else:
                if _LOG_KERNEL_INVOCATION:
                    _log_once("v68_inference",
                              f"M={x_chunk.shape[0]}",
                              f"K={x_chunk.shape[-1]}",
                              f"tp={self.parallel_size}")
                topk_w, topk_i, hidden_out = (
                    _v68_op()(
                        scores_chunk,
                        x_chunk,
                        bias,
                        self._v68_w_gate_packed,
                        self._v68_w_up_packed,
                        self._v68_group_max_scale_gate,
                        self._v68_group_max_scale_up,
                        self._routed_scaling_factor,
                    )
                )

            # [iter-15] Method-A dump: save v68's hidden_out + topk_i/w to
            # disk at chosen layers so we can offline-compare element-wise to
            # the PyTorch ref. Captures the kernel's actual output, NOT the
            # PyTorch substitute. Only fires when this layer is in the dump
            # list AND we're in the skip/limit window AND outside cuda-graph
            # capture (saving touches CPU memory). Rank-0 only.
            if (_li is not None and _li in _DUMP_V68_HIDDEN_OUT_LAYERS
                    and not _use_pytorch_v68_here
                    and not torch.cuda.is_current_stream_capturing()):
                try:
                    import torch.distributed as _dist
                    _rk = (_dist.get_rank()
                           if _dist.is_initialized() else 0)
                except Exception:
                    _rk = 0
                if _rk == 0:
                    if not hasattr(self, "_v68_dump_count"):
                        self._v68_dump_count = 0
                    _cc_l = self._v68_dump_count
                    _in_w = (_cc_l >= _DUMP_V68_SKIP
                             and _cc_l < _DUMP_V68_SKIP + _DUMP_V68_LIMIT)
                    if _in_w:
                        try:
                            import os as _os
                            _os.makedirs(_DUMP_V68_DIR, exist_ok=True)
                            _x = x_chunk.detach().to(torch.float32).cpu()
                            _ho = hidden_out.detach().to(torch.float32).cpu()
                            _ti = topk_i.detach().to(torch.int32).cpu()
                            _tw = topk_w.detach().to(torch.float32).cpu()
                            _stub = (
                                f"{_DUMP_V68_DIR}/layer_{_li}"
                                f"_cc{_cc_l}_chunk{start}")
                            torch.save(_ho, f"{_stub}_v68_hidden_out.pt")
                            torch.save(_x, f"{_stub}_v68_x.pt")
                            torch.save(_ti, f"{_stub}_v68_topk_i.pt")
                            torch.save(_tw, f"{_stub}_v68_topk_w.pt")
                            # Also compute the PyTorch ref hidden_out on the
                            # same input and dump it. This requires the
                            # routed w3_w1 weights to be live — caller must
                            # set GLM5_PYTORCH_V68_LAYERS (any value
                            # containing _li suffices, but we just check
                            # _NEED_W3W1) OR GLM5_USE_PYTORCH_V68=1 to keep
                            # them loaded. If unavailable, skip the ref dump
                            # gracefully so production runs aren't broken.
                            if _NEED_W3W1 or getattr(
                                    self, "w3_w1_weight", None) is not None:
                                try:
                                    with torch.no_grad():
                                        _tw_ref, _ti_ref, _ho_ref = (
                                            self._run_pytorch_v68(
                                                x_chunk, scores_chunk,
                                                bias))
                                    torch.save(
                                        _ho_ref.detach().to(
                                            torch.float32).cpu(),
                                        f"{_stub}_ref_hidden_out.pt")
                                    torch.save(
                                        _ti_ref.detach().to(
                                            torch.int32).cpu(),
                                        f"{_stub}_ref_topk_i.pt")
                                    torch.save(
                                        _tw_ref.detach().to(
                                            torch.float32).cpu(),
                                        f"{_stub}_ref_topk_w.pt")
                                except Exception as _ex2:
                                    print(
                                        f"[GLM5_DUMP_V68] layer={_li} ref "
                                        f"compute failed: {_ex2}",
                                        flush=True)
                            print(
                                f"[GLM5_DUMP_V68 layer={_li} cc={_cc_l} "
                                f"start={start}] saved hidden_out "
                                f"shape={tuple(hidden_out.shape)} "
                                f"dtype={hidden_out.dtype} -> {_stub}_*.pt",
                                flush=True)
                        except Exception as _ex:
                            print(
                                f"[GLM5_DUMP_V68] layer={_li} dump failed: "
                                f"{_ex}", flush=True)
                    self._v68_dump_count += 1

            # v110: down + per-expert weighted combine + residual + AR.
            # Residual is the pre-MoE-block hidden. We don't have that here
            # (run_moe sees post-norm x). Pass zeros so v110 outputs MoE-only;
            # the layer's external residual add (around modeling_deepseekv3.py
            # Deepseekv3MoE.forward) still happens.
            #
            # [iter-9] When GLM5_USE_PARENT_DOWN=1, skip v110 entirely and
            # run the down GEMM + combine + AR via fp32 PyTorch dequant. This
            # is the v68-only diagnostic path: if AL recovers (>= 2.5), v110
            # is the AL killer; if AL stays at 1.0, v68 or wider integration
            # is broken.
            # [iter-15] _use_parent_down_here also honors per-layer CSV.
            if _use_parent_down_here:
                if _LOG_KERNEL_INVOCATION:
                    _log_once("parent_down_inference",
                              f"M={hidden_out.shape[0]}",
                              f"rank={int(self.mapping.tp_rank)}",
                              "(v110 skipped)")
                out_chunk = self._run_parent_down(
                    hidden_out, topk_i, topk_w)
                chunks.append(out_chunk)
                continue
            self._ensure_v110_workspace()
            peers = list(self._v110_peer_ptrs)
            while len(peers) < 8:
                peers.append(0)
            rank = int(self.mapping.tp_rank)
            add_res = (rank == 0)
            residual_chunk = torch.zeros((x_chunk.shape[0], x_chunk.shape[-1]),
                                         dtype=torch.bfloat16, device=x.device)
            if _LOG_KERNEL_INVOCATION:
                _log_once("v110_inference",
                          f"M={hidden_out.shape[0]}",
                          f"rank={rank}",
                          f"flag={self._v110_flag}")
            out_chunk = _v110_op()(
                hidden_out,
                topk_i.to(torch.int32),
                topk_w.to(torch.float32),
                residual_chunk,
                self._v110_w_down_packed,
                self._v110_w_down_group_scale,
                add_res,
                rank,
                int(peers[0]), int(peers[1]), int(peers[2]), int(peers[3]),
                int(peers[4]), int(peers[5]), int(peers[6]), int(peers[7]),
                int(self.mapping.tp_size),
                int(self._v110_flag),
            )
            self._v110_flag += 1
            chunks.append(out_chunk)

        if len(chunks) == 1:
            return chunks[0]
        return torch.cat(chunks, dim=0)

    # ------------------------------------------------------------------
    # Single-layer diff test (GLM5_DIFF_TEST=1)
    # ------------------------------------------------------------------
    def _run_diff_test(self, x: torch.Tensor,
                       router_logits: torch.Tensor) -> None:
        """Compare v68+v110 outputs to a PyTorch reference, stage-by-stage.

        Stage 1 — routing: v68's topk_i/topk_w vs `torch.ops.trtllm.noaux_tc_op`.
        Stage 2 — v68 hidden_out per slot: vs silu(gate@x)*up@x from dequant'd
                  fp8 weights (routed for slots 1-8, shared for slot 0).
        Stage 3 — v110 output: vs sum-over-slots(weight*(down@hidden)) + AR.

        Localizes the AL=1.0 bug to one of:
          * v68 routing (Stage 1 fails)
          * v68 FP8 GEMM or repack (Stage 2 fails)
          * v110 down+combine or AR (Stage 3 fails)
          * our weight repack mapping (any stage fails on first-layer-only).
        """
        import torch.distributed as dist

        rank = int(self.mapping.tp_rank)
        M, K = x.shape[0], x.shape[-1]

        # ---------- Stage 4 (PRE-FLIGHT): direct baseline vs v68+v110 ----------
        # User asked: how different is our new kernel's MoE output from the
        # baseline parent kernel's output, on the SAME input?
        # We invoke the parent's full MoE path (which uses fp8_block_scale_moe
        # for the GEMMs, run on the same routed weights) and compare against
        # what v68+v110 will produce later in this same run_moe call.
        # Done FIRST (before stages 1/2/3) because v68 mutates internal state
        # (the v110 flag counter), and we want a clean baseline.
        try:
            # Save & temporarily disable packed weights so super().run_moe()
            # routes through the parent's fp8_block_scale_moe path (the
            # "baseline"). Restore after.
            saved_v68_g = self._v68_w_gate_packed
            saved_v68_u = self._v68_w_up_packed
            saved_v110 = self._v110_w_down_packed
            self._v68_w_gate_packed = None
            self._v68_w_up_packed = None
            self._v110_w_down_packed = None

            parent_out = super().run_moe(
                x.contiguous(),
                router_logits=router_logits,
                do_finalize=True,
            )

            self._v68_w_gate_packed = saved_v68_g
            self._v68_w_up_packed = saved_v68_u
            self._v110_w_down_packed = saved_v110

            # Run our v68+v110 path on the same input — duplicate of the
            # chunk loop's first chunk so the flag sequence matches what the
            # production path will see.
            scores_fp32 = router_logits.contiguous().to(torch.float32)
            topk_w_v68, topk_i_v68, hidden_out_v68 = (
                _v68_op()(
                    scores_fp32, x.contiguous(), self._cached_bias_bf16,
                    self._v68_w_gate_packed, self._v68_w_up_packed,
                    self._v68_group_max_scale_gate, self._v68_group_max_scale_up,
                    self._routed_scaling_factor,
                )
            )
            self._ensure_v110_workspace()
            peers = list(self._v110_peer_ptrs)
            while len(peers) < 8:
                peers.append(0)
            residual_zero = torch.zeros((x.shape[0], x.shape[-1]),
                                        dtype=torch.bfloat16, device=x.device)
            ours_out = _v110_op()(
                hidden_out_v68,
                topk_i_v68.to(torch.int32),
                topk_w_v68.to(torch.float32),
                residual_zero,
                self._v110_w_down_packed,
                self._v110_w_down_group_scale,
                (rank == 0),
                rank,
                int(peers[0]), int(peers[1]), int(peers[2]), int(peers[3]),
                int(peers[4]), int(peers[5]), int(peers[6]), int(peers[7]),
                int(self.mapping.tp_size),
                int(self._v110_flag),
            )
            self._v110_flag += 1

            # parent_out includes shared expert (since parent doesn't skip it)
            # via shared_experts.forward in the surrounding Deepseekv3MoE.
            # But wait — parent's run_moe ONLY computes routed; shared is
            # added externally. So `parent_out` is the routed-only output;
            # `ours_out` is shared + routed (v68/v110 fused).
            # To compare apples to apples, add shared_experts(x) to parent_out.
            shared_out = self._shared_experts_ref(x.contiguous())
            parent_full = parent_out + shared_out  # both bf16, broadcast-add
            # Parent skipped its in-MoE AR (Glm5SmallBatchFusedMoE returned
            # EXTERNAL_COMM with enable_allreduce in final_all_reduce_params
            # but run_moe direct call bypasses that). Manually AR to match
            # v110's AR'd output.
            import torch.distributed as dist
            if dist.is_initialized():
                parent_full = parent_full.float().contiguous()
                dist.all_reduce(parent_full)
                parent_full = parent_full.to(torch.bfloat16)

            ours_fp32 = ours_out.to(torch.float32)
            parent_fp32 = parent_full.to(torch.float32)
            max_abs = (ours_fp32 - parent_fp32).abs().max().item()
            denom = parent_fp32.abs().clamp(min=1e-6)
            max_rel = ((ours_fp32 - parent_fp32).abs() / denom).max().item()
            mean_abs = (ours_fp32 - parent_fp32).abs().mean().item()
            print(f"[GLM5_DIFF rank={rank}] STAGE4 baseline-vs-v68/v110: "
                  f"max_abs={max_abs:.4f}  mean_abs={mean_abs:.4f}  "
                  f"max_rel={max_rel:.4f}",
                  flush=True)
            if rank == 0:
                print(f"[GLM5_DIFF rank=0]   parent[0,:8] = {[f'{v:.4f}' for v in parent_fp32[0,:8].tolist()]}",
                      flush=True)
                print(f"[GLM5_DIFF rank=0]   ours  [0,:8] = {[f'{v:.4f}' for v in ours_fp32[0,:8].tolist()]}",
                      flush=True)
        except Exception as e:
            print(f"[GLM5_DIFF rank={rank}] STAGE4 failed: {type(e).__name__}: {e}",
                  flush=True)
            # Make sure packed weights are restored if exception happened
            # before the restore lines.
            if self._v68_w_gate_packed is None and saved_v68_g is not None:
                self._v68_w_gate_packed = saved_v68_g
                self._v68_w_up_packed = saved_v68_u
                self._v110_w_down_packed = saved_v110

        # ---------- Stage 1: routing ----------
        bias_fp32 = self.routing_method.e_score_correction_bias.to(torch.float32)
        bias_bf16_via_cast = bias_fp32.to(torch.bfloat16)
        scores_fp32 = router_logits.contiguous().to(torch.float32)

        # Reference A: noaux_tc with full fp32 bias (matches parent backend
        # which passes fp32 bias through to its router kernel).
        topk_w_ref32, topk_i_ref32 = torch.ops.trtllm.noaux_tc_op(
            scores_fp32, bias_fp32, 1, 1, 8, self._routed_scaling_factor,
        )

        # Reference B: noaux_tc with bf16-cast bias. If this matches v68 but
        # diverges from Reference A → the bf16 bias cast is the bug. If it
        # ALSO diverges from v68 → v68 has additional internal numerical
        # divergence beyond the cast.
        topk_w_ref16, topk_i_ref16 = torch.ops.trtllm.noaux_tc_op(
            scores_fp32, bias_bf16_via_cast.to(torch.float32),
            1, 1, 8, self._routed_scaling_factor,
        )

        topk_w_v68, topk_i_v68, hidden_out_v68 = (
            torch.ops.trtllm.glm5_expert_select_up_gate_silu(
                scores_fp32, x.contiguous(), self._cached_bias_bf16,
                self._v68_w_gate_packed, self._v68_w_up_packed,
                self._v68_group_max_scale_gate, self._v68_group_max_scale_up,
                self._routed_scaling_factor,
            )
        )

        # Helpers to compare two (indices, weights) pairs.
        def _set_match(i_a: torch.Tensor, i_b: torch.Tensor) -> bool:
            a_sorted, _ = torch.sort(i_a.to(torch.int32), dim=-1)
            b_sorted, _ = torch.sort(i_b.to(torch.int32), dim=-1)
            return bool((a_sorted == b_sorted).all().item())

        def _max_w_diff(i_a, w_a, i_b, w_b) -> float:
            a_perm = torch.argsort(i_a.to(torch.int32), dim=-1)
            b_perm = torch.argsort(i_b.to(torch.int32), dim=-1)
            wa = torch.gather(w_a.to(torch.float32), -1, a_perm)
            wb = torch.gather(w_b.to(torch.float32), -1, b_perm)
            return (wa - wb).abs().max().item()

        # Three pairwise comparisons.
        idx_fp32_v68 = _set_match(topk_i_ref32, topk_i_v68)
        idx_bf16_v68 = _set_match(topk_i_ref16, topk_i_v68)
        idx_fp32_bf16 = _set_match(topk_i_ref32, topk_i_ref16)
        w_fp32_v68 = _max_w_diff(topk_i_ref32, topk_w_ref32, topk_i_v68, topk_w_v68)
        w_bf16_v68 = _max_w_diff(topk_i_ref16, topk_w_ref16, topk_i_v68, topk_w_v68)
        w_fp32_bf16 = _max_w_diff(topk_i_ref32, topk_w_ref32, topk_i_ref16, topk_w_ref16)

        print(f"[GLM5_DIFF rank={rank}] STAGE1 routing summary", flush=True)
        print(f"[GLM5_DIFF rank={rank}]   fp32-bias-ref vs v68:           idx_match={idx_fp32_v68}  w_diff={w_fp32_v68:.4e}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   bf16-bias-ref vs v68:           idx_match={idx_bf16_v68}  w_diff={w_bf16_v68:.4e}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   fp32-bias-ref vs bf16-bias-ref: idx_match={idx_fp32_bf16}  w_diff={w_fp32_bf16:.4e}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   x.shape={tuple(x.shape)}  router_logits.shape={tuple(router_logits.shape)}",
              flush=True)
        if rank == 0:
            m0 = 0
            print(f"[GLM5_DIFF rank=0]   topk_i_ref32[{m0}] = {topk_i_ref32[m0].tolist()}", flush=True)
            print(f"[GLM5_DIFF rank=0]   topk_i_ref16[{m0}] = {topk_i_ref16[m0].tolist()}", flush=True)
            print(f"[GLM5_DIFF rank=0]   topk_i_v68[{m0}]   = {topk_i_v68[m0].tolist()}", flush=True)
            print(f"[GLM5_DIFF rank=0]   topk_w_ref32[{m0}] = {[f'{w:.4f}' for w in topk_w_ref32[m0].tolist()]}",
                  flush=True)
            print(f"[GLM5_DIFF rank=0]   topk_w_v68[{m0}]   = {[f'{w:.4f}' for w in topk_w_v68[m0].tolist()]}",
                  flush=True)

        # Keep ref32 as the canonical reference for stages 2/3 (because the
        # rest of the pipeline uses fp32 bias).
        topk_w_ref, topk_i_ref = topk_w_ref32, topk_i_ref32

        # If routing already diverges, stages 2+3 are moot — but they may
        # still be informative if Stage 1 only diverges in weights (not idx).

        # ---------- Stage 2: v68 hidden_out per slot ----------
        # Need dequantized routed + shared weights. Routed is [E=256,2*M/tp,K]
        # with first half = up (w3), second half = gate (w1). Shared is
        # [2*M/tp,K] with first half = gate, second half = up. Scales are
        # block-fp8 with block_size=128.
        BLK = 128

        def _block_dequant_3d(w_fp8: torch.Tensor, s_fp32: torch.Tensor) -> torch.Tensor:
            """[E, M, K] fp8 * [E, M/BLK, K/BLK] fp32 -> [E, M, K] fp32."""
            w_fp32 = w_fp8.to(torch.float32)
            E, Mw, Kw = w_fp32.shape
            s_rep = s_fp32.repeat_interleave(BLK, dim=1).repeat_interleave(BLK, dim=2)
            return w_fp32 * s_rep[:, :Mw, :Kw]

        def _block_dequant_2d(w_fp8: torch.Tensor, s_fp32: torch.Tensor) -> torch.Tensor:
            w_fp32 = w_fp8.to(torch.float32)
            Mw, Kw = w_fp32.shape
            s_rep = s_fp32.repeat_interleave(BLK, dim=0).repeat_interleave(BLK, dim=1)
            return w_fp32 * s_rep[:Mw, :Kw]

        w3_w1 = self.w3_w1_weight
        w3_w1_s = self.w3_w1_weight_scaling_factor
        if w3_w1 is None or w3_w1_s is None:
            print(f"[GLM5_DIFF rank={rank}] STAGE2 skipped: routed weights "
                  f"were dropped (diff_test_layer marker missed?)", flush=True)
            return

        shared = self._shared_experts_ref
        shared_w_gu = shared.gate_up_proj.weight
        shared_s_gu = shared.gate_up_proj.weight_scale
        shared_w_d = shared.down_proj.weight
        shared_s_d = shared.down_proj.weight_scale
        print(f"[GLM5_DIFF rank={rank}] STAGE2 shapes+dtypes:",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   w3_w1.shape={tuple(w3_w1.shape)} dtype={w3_w1.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   w3_w1_s.shape={tuple(w3_w1_s.shape)} dtype={w3_w1_s.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   w2.shape={tuple(self.w2_weight.shape)} dtype={self.w2_weight.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   w2_s.shape={tuple(self.w2_weight_scaling_factor.shape)} dtype={self.w2_weight_scaling_factor.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   shared_gu.shape={tuple(shared_w_gu.shape)} dtype={shared_w_gu.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   shared_gu_s.shape={tuple(shared_s_gu.shape)} dtype={shared_s_gu.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   shared_d.shape={tuple(shared_w_d.shape)} dtype={shared_w_d.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   shared_d_s.shape={tuple(shared_s_d.shape)} dtype={shared_s_d.dtype}",
              flush=True)
        print(f"[GLM5_DIFF rank={rank}]   x.shape={tuple(x.shape)} dtype={x.dtype}",
              flush=True)
        # Also print the gate_up_proj class for context.
        gu_cls = type(shared.gate_up_proj).__name__
        gu_qm_cls = type(getattr(shared.gate_up_proj, "quant_method", None)).__name__
        print(f"[GLM5_DIFF rank={rank}]   shared.gate_up_proj class={gu_cls} quant_method={gu_qm_cls}",
              flush=True)

        # Sanity: x must have K matching the weights' K. If not, stage 2/3
        # are nonsensical — abort with a print.
        K_w = w3_w1.shape[2]
        if x.shape[-1] != K_w:
            print(f"[GLM5_DIFF rank={rank}] STAGE2 skipped: x.shape[-1]={x.shape[-1]} "
                  f"but w3_w1 K={K_w} — input may be TP-sharded or pre-quantized "
                  f"unexpectedly. Investigate before continuing.", flush=True)
            return

        half_w = w3_w1.shape[1] // 2
        half_s = w3_w1_s.shape[1] // 2
        # Routed layout: first half = up (w3), second half = gate (w1).
        w_up_r_dq = _block_dequant_3d(w3_w1[:, :half_w, :], w3_w1_s[:, :half_s, :])
        w_gate_r_dq = _block_dequant_3d(w3_w1[:, half_w:, :], w3_w1_s[:, half_s:, :])

        shared = self._shared_experts_ref
        shared_w_gu = shared.gate_up_proj.weight
        shared_s_gu = shared.gate_up_proj.weight_scale
        shared_half_w = shared_w_gu.shape[0] // 2
        shared_half_s = shared_s_gu.shape[0] // 2
        # Shared layout: first half = gate, second half = up.
        w_gate_sh_dq = _block_dequant_2d(shared_w_gu[:shared_half_w, :],
                                         shared_s_gu[:shared_half_s, :])
        w_up_sh_dq = _block_dequant_2d(shared_w_gu[shared_half_w:, :],
                                       shared_s_gu[shared_half_s:, :])

        x_fp32 = x.to(torch.float32)
        silu = torch.nn.functional.silu

        max_slot_diff = 0.0
        slot_diffs_by_slot = [0.0] * 9
        for m in range(M):
            for slot in range(9):
                if slot == 0:
                    g = w_gate_sh_dq @ x_fp32[m]
                    u = w_up_sh_dq @ x_fp32[m]
                else:
                    e_id = int(topk_i_v68[m, slot - 1].item())
                    g = w_gate_r_dq[e_id] @ x_fp32[m]
                    u = w_up_r_dq[e_id] @ x_fp32[m]
                h_ref = silu(g) * u
                h_v68 = hidden_out_v68[m, slot, :].to(torch.float32)
                d = (h_ref - h_v68).abs().max().item()
                if d > slot_diffs_by_slot[slot]:
                    slot_diffs_by_slot[slot] = d
                if d > max_slot_diff:
                    max_slot_diff = d

        slot_str = " ".join(f"s{i}={d:.3f}" for i, d in enumerate(slot_diffs_by_slot))
        print(f"[GLM5_DIFF rank={rank}] STAGE2 v68 hidden_out: "
              f"max_diff={max_slot_diff:.4f}  per_slot[{slot_str}]",
              flush=True)
        if rank == 0:
            # Show one slot's first 8 values for visual sanity.
            m0, s0 = 0, 1   # token 0, first routed slot
            e_id = int(topk_i_v68[m0, 0].item())
            g = w_gate_r_dq[e_id] @ x_fp32[m0]
            u = w_up_r_dq[e_id] @ x_fp32[m0]
            h_ref = (silu(g) * u)[:8].tolist()
            h_v68 = hidden_out_v68[m0, s0, :8].to(torch.float32).tolist()
            print(f"  m=0 slot=1 expert={e_id}", flush=True)
            print(f"  h_ref[:8]  = {[f'{v:.4f}' for v in h_ref]}", flush=True)
            print(f"  h_v68[:8]  = {[f'{v:.4f}' for v in h_v68]}", flush=True)

        # ---------- Stage 3: v110 full output ----------
        w2 = self.w2_weight
        w2_s = self.w2_weight_scaling_factor
        if w2 is None or w2_s is None:
            print(f"[GLM5_DIFF rank={rank}] STAGE3 skipped: routed down weight "
                  f"was dropped", flush=True)
            return
        w2_dq = _block_dequant_3d(w2, w2_s)  # [E, K, M/tp]

        shared_w_d_dq = _block_dequant_2d(shared.down_proj.weight,
                                          shared.down_proj.weight_scale)

        # Reference: per-rank partial = sum over slots(weight * down @ slot),
        # then AR via NCCL.
        out_ref = torch.zeros(M, K, dtype=torch.float32, device=x.device)
        for m in range(M):
            h_slot0 = hidden_out_v68[m, 0, :].to(torch.float32)
            out_ref[m] += shared_w_d_dq @ h_slot0   # shared, weight 1.0
            for slot in range(1, 9):
                e_id = int(topk_i_v68[m, slot - 1].item())
                w = topk_w_v68[m, slot - 1].to(torch.float32).item()
                h_slot = hidden_out_v68[m, slot, :].to(torch.float32)
                out_ref[m] += w * (w2_dq[e_id] @ h_slot)
        if dist.is_initialized():
            dist.all_reduce(out_ref)

        # v110 output
        self._ensure_v110_workspace()
        peers = list(self._v110_peer_ptrs)
        while len(peers) < 8:
            peers.append(0)
        residual_zero = torch.zeros((M, K), dtype=torch.bfloat16, device=x.device)
        # Consume the next flag slot in the monotonic sequence — Lamport
        # sym-heap reuse hazard if we skip ahead and then resume from 0.
        diff_flag = int(self._v110_flag)
        self._v110_flag += 1
        out_v110 = torch.ops.trtllm.glm5_expert_down_allreduce(
            hidden_out_v68,
            topk_i_v68.to(torch.int32),
            topk_w_v68.to(torch.float32),
            residual_zero,
            self._v110_w_down_packed,
            self._v110_w_down_group_scale,
            (rank == 0),   # add_residual_on_rank0_only
            rank,
            int(peers[0]), int(peers[1]), int(peers[2]), int(peers[3]),
            int(peers[4]), int(peers[5]), int(peers[6]), int(peers[7]),
            int(self.mapping.tp_size),
            diff_flag,
        )
        out_v110_fp32 = out_v110.to(torch.float32)
        max_diff = (out_ref - out_v110_fp32).abs().max().item()
        denom = out_ref.abs().clamp(min=1e-6)
        rel_diff = ((out_ref - out_v110_fp32).abs() / denom).max().item()
        print(f"[GLM5_DIFF rank={rank}] STAGE3 v110 output: "
              f"max_abs_diff={max_diff:.4f} max_rel_diff={rel_diff:.4f}",
              flush=True)
        if rank == 0:
            print(f"  out_ref[0, :8]  = {[f'{v:.4f}' for v in out_ref[0, :8].tolist()]}",
                  flush=True)
            print(f"  out_v110[0, :8] = {[f'{v:.4f}' for v in out_v110_fp32[0, :8].tolist()]}",
                  flush=True)

    # ------------------------------------------------------------------
    # Weight repack hook
    # ------------------------------------------------------------------
    def load_weights(self, *args, **kwargs):
        """Call parent loader, then run the v68 and v110 layout repacks once.

        PR 1 first-pass: surfaces the actual TRTLLMGenFusedMoE weight tensor
        shapes via logging on first call, then attempts the repack ops. If the
        op signature doesn't match (shape mismatch / wrong dtype), the kernel
        gate stays closed (`_can_use_*` returns False because the packed
        weights stay None) and the parent backend handles the forward. That
        gives the bench a fall-through path while we figure out the right
        layout mapping.
        """
        super().load_weights(*args, **kwargs)

        # Diff-test mode: keep raw routed/shared weights on the FIRST layer
        # only, so _run_diff_test can dequantize them as PyTorch ground truth
        # without doubling the HBM budget for all 78 layers.
        diff_test_enabled = os.environ.get("GLM5_DIFF_TEST", "0") == "1"
        is_first_layer = Glm5SmallBatchFusedMoE._layer_load_counter == 0
        Glm5SmallBatchFusedMoE._layer_load_counter += 1
        self._diff_test_layer = diff_test_enabled and is_first_layer
        if self._diff_test_layer:
            _log_once("diff_test_layer_marked",
                      f"this instance keeps routed/shared weights for diff test")

        # Skip silently if the new ops aren't registered (e.g., development
        # build pending or wrong library loaded).
        repack_v68 = _v68_repack_op()
        repack_v110 = _v110_repack_op()
        if repack_v68 is None or repack_v110 is None:
            _log_once(
                "load_weights_no_ops",
                "glm5 thop ops not registered — falling back to parent backend",
            )
            return
        if _USE_PARENT_DOWN:
            _log_once(
                "use_parent_down",
                "GLM5_USE_PARENT_DOWN=1 — v68 fires, v110 SKIPPED, "
                "down GEMM/combine/AR via fp32 PyTorch dequant; "
                "self.w2_weight + w2_weight_scaling_factor retained")
        if _USE_PYTORCH_V68:
            _log_once(
                "use_pytorch_v68",
                "GLM5_USE_PYTORCH_V68=1 — v68 SKIPPED, "
                "routing+up+gate+silu via fp32 PyTorch dequant + per-128-col actquant; "
                "self.w3_w1_weight + w3_w1_weight_scaling_factor retained")

        # Always log the parent's weight shapes once so future iterations can
        # check whether the layout matches v68/v110 expectations.
        weight_attrs = (
            "w3_w1_weight", "w2_weight",
            "w3_w1_weight_scaling_factor", "w2_weight_scaling_factor",
            "fc31_weight", "fc2_weight",
            "fc31_weight_scale", "fc2_weight_scale",
        )
        for name in weight_attrs:
            t = getattr(self, name, None)
            if isinstance(t, torch.Tensor):
                _log_once(
                    f"weight_shape_{name}",
                    f"shape={tuple(t.shape)}",
                    f"dtype={t.dtype}",
                )

        # Pull the routed weights (256 experts).
        w_gu = getattr(self, "w3_w1_weight", None)
        s_gu = getattr(self, "w3_w1_weight_scaling_factor", None)
        w_down = getattr(self, "w2_weight", None)
        s_down = getattr(self, "w2_weight_scaling_factor", None)

        # Pull the shared expert weights (1 expert, sibling module).
        # The kernel layout expects E=257 = 256 routed + 1 shared (at index 256).
        # See mega_kernel_v68.cu:134 (kPackedExpertCount=257) and
        # mega_kernel_down_v110.cu:243 (kNumExpertsTotal=257).
        shared = self._shared_experts_ref
        shared_w_gu = shared_s_gu = shared_w_down = shared_s_down = None
        if shared is not None:
            try:
                shared_w_gu = shared.gate_up_proj.weight        # [2*M/tp, K=6144] fp8
                shared_s_gu = shared.gate_up_proj.weight_scale  # [2*M_blocks/tp, K_blocks] fp32
                shared_w_down = shared.down_proj.weight         # [K=6144, M/tp] fp8
                shared_s_down = shared.down_proj.weight_scale   # [K_blocks, M_blocks/tp] fp32
            except AttributeError as e:
                _log_once("shared_weights_missing", str(e)[:120])
        else:
            _log_once("no_shared_backref", "shared_experts_ref not set by Glm5SmallBatchMoE")

        # ----- Build 257-expert tensors for v68 (gate + up halves) -----
        # CONVENTION MISMATCH between routed and shared storage:
        #   - Routed: TRTLLM stores as `w3_w1_weight: [E, 2*M/tp, K]` where
        #     chunk[0] = w3 = UP, chunk[1] = w1 = GATE
        #     (see quantization.py:587 `_, dst_w1_weight = chunk(2)` and
        #     quantization.py:596 `dst_w3_weight, _ = chunk(2)`).
        #   - Shared (GatedMLP.gate_up_proj): chunk[0] = GATE, chunk[1] = UP
        #     (gated_mlp.py:67 `gateup_shard_indices_mapping={'gate':(0,M),'up':(M,M)}`).
        # The convention is REVERSED between the two storage formats. We must
        # extract correctly from each before concatenating into the 257-expert
        # tensor.
        if (isinstance(w_gu, torch.Tensor) and isinstance(s_gu, torch.Tensor)
                and shared_w_gu is not None and shared_s_gu is not None):
            try:
                # [iter-15-FIX] Same placeholder-scale hazard as the v110
                # block below. shared.gate_up_proj.weight_scale at this point
                # in the loader is the empty `torch.empty(...)` placeholder
                # (all zeros / arbitrary memory) — shared_experts hasn't run
                # its load_weights yet because experts is registered first
                # in DeepseekV3MoE.__init__. If we repack now we pack ZEROS
                # as the shared-expert gate/up scales, and v68's slot 0
                # produces zero hidden_out forever (catastrophic).
                _log_once("iter15_debug_shared_s_gu_at_load",
                          f"shared_w_gu={tuple(shared_w_gu.shape)} "
                          f"shared_s_gu={tuple(shared_s_gu.shape)} "
                          f"shared_s_gu_min={float(shared_s_gu.min())} "
                          f"shared_s_gu_max={float(shared_s_gu.max())} "
                          "(if min=max=0, placeholder — v68 repack DEFERRED)")
                _shared_s_gu_is_placeholder = (
                    float(shared_s_gu.abs().max()) == 0.0)
                if _shared_s_gu_is_placeholder:
                    _log_once(
                        "iter15_defer_v68_repack",
                        "shared gate_up scale is the empty PLACEHOLDER "
                        "— v68 repack DEFERRED to first forward")
                    # Store unrepacked routed refs; lazy repack on first
                    # forward will re-read shared via _shared_experts_ref.
                    self._v68_pending_w_gu_routed = w_gu
                    self._v68_pending_s_gu_routed = s_gu
                    self._v68_w_gate_packed = None
                    self._v68_w_up_packed = None
                    # Skip the rest of the v68 block — drop unused locals
                    # to keep memory clean and continue to the v110 block
                    # (which has its own placeholder check).
                else:
                    half = w_gu.shape[1] // 2
                    # Routed layout: first half = up, second half = gate.
                    w_up_routed   = w_gu[:, :half, :].contiguous()    # [256, M/tp, K]
                    w_gate_routed = w_gu[:, half:, :].contiguous()    # [256, M/tp, K]
                    half_s = s_gu.shape[1] // 2
                    s_up_routed   = s_gu[:, :half_s, :].contiguous()  # [256, M_blocks/tp, K_blocks]
                    s_gate_routed = s_gu[:, half_s:, :].contiguous()  # [256, M_blocks/tp, K_blocks]

                    # Shared layout: first half = gate, second half = up.
                    shared_half = shared_w_gu.shape[0] // 2
                    w_gate_shared = shared_w_gu[:shared_half, :].unsqueeze(0).contiguous()  # [1, M/tp, K]
                    w_up_shared   = shared_w_gu[shared_half:, :].unsqueeze(0).contiguous()
                    shared_half_s = shared_s_gu.shape[0] // 2
                    s_gate_shared = shared_s_gu[:shared_half_s, :].unsqueeze(0).contiguous()  # [1, M_blocks/tp, K_blocks]
                    s_up_shared   = shared_s_gu[shared_half_s:, :].unsqueeze(0).contiguous()

                    # Concat to 257 experts (shared at index 256, last).
                    w_gate_257 = torch.cat([w_gate_routed, w_gate_shared], dim=0).contiguous()
                    w_up_257   = torch.cat([w_up_routed,   w_up_shared],   dim=0).contiguous()
                    s_gate_257 = torch.cat([s_gate_routed, s_gate_shared], dim=0).contiguous()
                    s_up_257   = torch.cat([s_up_routed,   s_up_shared],   dim=0).contiguous()

                    self._v68_w_gate_packed, self._v68_group_max_scale_gate = \
                        repack_v68(w_gate_257, s_gate_257)
                    self._v68_w_up_packed, self._v68_group_max_scale_up = \
                        repack_v68(w_up_257, s_up_257)
                    _log_once("v68_repack_ok",
                              f"gate_input={tuple(w_gate_257.shape)} "
                              f"scale_input={tuple(s_gate_257.shape)}")
                    # Free intermediates immediately to avoid OOM at the next layer.
                    del w_gate_routed, w_up_routed, s_gate_routed, s_up_routed
                    del w_gate_shared, w_up_shared, s_gate_shared, s_up_shared
                    del w_gate_257, w_up_257, s_gate_257, s_up_257
                    # Drop the original routed gate+up weight on this module —
                    # we no longer need it (v68 owns the packed copy). At TP=4
                    # this is ~1.5 GiB / layer recovered, which is mandatory to
                    # fit 78 layers' packed weights in HBM. Skip the drop on the
                    # diff-test layer so _run_diff_test can use raw weights as
                    # ground truth.
                    #
                    # [iter-10] Also skip the drop when GLM5_USE_PYTORCH_V68=1 —
                    # _run_pytorch_v68 needs w3_w1_weight + scale every forward
                    # to dequant routed gate/up weights for the silu(gate@x)*up@x
                    # math. Shared gate_up weights stay live regardless (sibling
                    # GatedMLP owns their lifecycle).
                    if not self._diff_test_layer and not _NEED_W3W1:
                        if hasattr(self, "w3_w1_weight"):
                            self.w3_w1_weight = None
                        if hasattr(self, "w3_w1_weight_scaling_factor"):
                            self.w3_w1_weight_scaling_factor = None
                        torch.cuda.empty_cache()
            except Exception as e:
                _log_once("v68_repack_failed", str(e)[:160])
                self._v68_w_gate_packed = None
                self._v68_w_up_packed = None
                torch.cuda.empty_cache()

        # ----- Build 257-expert tensors for v110 (down) -----
        if (isinstance(w_down, torch.Tensor) and isinstance(s_down, torch.Tensor)
                and shared_w_down is not None and shared_s_down is not None):
            try:
                # [iter-15-FIX] CRITICAL: at this point in the weight loader,
                # the SHARED expert's `down_proj.weight_scale` may still be
                # an uninitialized PLACEHOLDER tensor — for FP8BlockScales it's
                # `torch.empty(ceil(K/128), ceil(M_local/128)) = (48, 4)` with
                # arbitrary memory contents (the verifier dump showed all
                # zeros). The TRTLLM weight loader walks named_modules() in
                # registration order, and `experts` is registered BEFORE
                # `shared_experts` in DeepseekV3MoE.__init__ — so when our
                # Glm5SmallBatchFusedMoE.load_weights runs, the sibling
                # shared.down_proj hasn't been loaded yet. Its scale Parameter
                # exists at the correct shape but the values are garbage.
                #
                # If we proceeded to repack here, v110 would pack ZEROS as the
                # shared expert's per-128-block scales, and slot 0 (shared)
                # would contribute literal ZERO to v110's output forever.
                # That explains iter-14's channel-2317 systematic suppression:
                # the shared expert produces large values on a small set of
                # channels (iter-3 specialist: shared mag ~4x routed), and
                # under the zero-scale bug v110 emits zero where parent
                # emits the full shared contribution.
                #
                # Fix: defer the v110 repack to the FIRST forward call (via
                # _lazy_repack_if_needed in run_moe). At that point shared has
                # been loaded by its own load_weights and weight_scale is the
                # real value.
                _log_once("iter15_debug_shared_s_down_at_load",
                          f"shared_w_down={tuple(shared_w_down.shape)} "
                          f"shared_s_down={tuple(shared_s_down.shape)} "
                          f"routed_w_down={tuple(w_down.shape)} "
                          f"routed_s_down={tuple(s_down.shape)} "
                          f"shared_s_down_isnan={bool(torch.isnan(shared_s_down).any())} "
                          f"shared_s_down_min={float(shared_s_down.min())} "
                          f"shared_s_down_max={float(shared_s_down.max())} "
                          "(if min=max=0, placeholder is uninitialized "
                          "— v110 repack DEFERRED to first forward)")
                # Detect placeholder: an FP8 scale tensor where every element
                # is exactly 0.0 is almost certainly the
                # `torch.empty(...)` allocation that hasn't yet been
                # `copy_weight`'d. Real loaded scales are strictly > 0
                # (they come from `amax / FP8_E4M3_MAX` which is positive).
                _shared_s_is_placeholder = (
                    float(shared_s_down.abs().max()) == 0.0)
                if _shared_s_is_placeholder:
                    _log_once(
                        "iter15_defer_v110_repack",
                        "shared down scale is the empty PLACEHOLDER (all "
                        "zeros) — v110 repack DEFERRED to first forward")
                    # Store unrepacked tensor refs; lazy repack on first
                    # forward.  The actual `shared_s_down` Parameter on
                    # shared.down_proj WILL be overwritten in-place by
                    # shared.load_weights later — we just keep a backref
                    # via self._shared_experts_ref and re-read the
                    # Parameter at lazy-repack time.
                    self._v110_pending_w_down_routed = w_down
                    self._v110_pending_s_down_routed = s_down
                    self._v110_w_down_packed = None
                    self._v110_w_down_group_scale = None
                    return
                w_down_shared = shared_w_down.unsqueeze(0).contiguous()  # [1, K, M/tp]
                s_down_shared = shared_s_down.unsqueeze(0).contiguous()  # [1, K_blocks, M_blocks/tp]
                w_down_257 = torch.cat([w_down, w_down_shared], dim=0).contiguous()
                s_down_257 = torch.cat([s_down, s_down_shared], dim=0).contiguous()

                self._v110_w_down_packed, self._v110_w_down_group_scale = \
                    repack_v110(w_down_257, s_down_257)
                _log_once("v110_repack_ok",
                          f"down_input={tuple(w_down_257.shape)} "
                          f"packed={tuple(self._v110_w_down_packed.shape)} "
                          f"scale_cat={tuple(s_down_257.shape)}")
                del w_down_shared, s_down_shared
                del w_down_257, s_down_257
                # Drop the original routed down weight (v110 has the packed
                # copy). Skip on the diff-test layer so the raw weight remains
                # available as PyTorch ground truth.
                #
                # [iter-9] Also skip the drop when GLM5_USE_PARENT_DOWN=1 —
                # _run_parent_down needs w2_weight + w2_weight_scaling_factor
                # to compute the fp32 dequant + matmul on every forward.
                # Shared down weights stay live because they live on the
                # sibling GatedMLP (we don't manage their lifecycle here).
                if not self._diff_test_layer and not _NEED_W2:
                    if hasattr(self, "w2_weight"):
                        self.w2_weight = None
                    if hasattr(self, "w2_weight_scaling_factor"):
                        self.w2_weight_scaling_factor = None
                    torch.cuda.empty_cache()
            except Exception as e:
                _log_once("v110_repack_failed", str(e)[:160])
                self._v110_w_down_packed = None
                torch.cuda.empty_cache()


class Glm5SmallBatchMoE(Deepseekv3MoE):
    """Deepseekv3MoE subclass for the GLM-5 small-batch path.

    PR 1: when self.experts is a Glm5SmallBatchFusedMoE (selected via
    `moe_config.backend: GLM5_SMALL_BATCH` -> create_moe), set a backref so
    the experts module can pull the sibling self.shared_experts weights into
    the v68/v110 257-expert layout at load_weights time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Locate the actual Glm5SmallBatchFusedMoE backend behind the
        # ConfigurableMoE wrapper (when ENABLE_CONFIGURABLE_MOE=1) and give
        # it a backref to our shared_experts module. Use object.__setattr__
        # to bypass nn.Module's auto-submodule-registration — see
        # Glm5SmallBatchDecoderLayer.__init__ for why this matters.
        backend = getattr(self.experts, "backend", self.experts)
        if isinstance(backend, Glm5SmallBatchFusedMoE):
            object.__setattr__(backend, "_shared_experts_ref",
                               self.shared_experts)

    def _backend_has_packed_weights(self) -> bool:
        backend = getattr(self.experts, "backend", self.experts)
        return (isinstance(backend, Glm5SmallBatchFusedMoE)
                and backend._v68_w_gate_packed is not None
                and backend._v110_w_down_packed is not None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4=None,
        all_rank_num_tokens=None,
        final_all_reduce_params=None,
        do_finalize: bool = True,
    ) -> torch.Tensor:
        # When the v68+v110 packed weights are loaded, the fused kernels
        # already cover (1) the shared expert (slot 0 of v68's hidden_out)
        # and (2) the cross-rank all-reduce (v110 is ExpertDownAllReduce —
        # the AR is fused into the down-GEMM). Both must therefore be
        # SKIPPED on the Python side, otherwise:
        #   - shared_experts(hidden) gets added on top of v110's already-
        #     shared output → shared counted twice;
        #   - self.allreduce(final) (or the outer forward_MoE AR when
        #     POST_MOE_FUSION=True) sums replicated values across 4 ranks
        #     → 4× wrong values.
        # When packed weights aren't loaded (e.g. MTP draft layers, which
        # don't hit our class-swap), fall through to the vanilla path.
        if not self._backend_has_packed_weights():
            return super().forward(
                hidden_states=hidden_states,
                hidden_states_fp4=hidden_states_fp4,
                all_rank_num_tokens=all_rank_num_tokens,
                final_all_reduce_params=final_all_reduce_params,
                do_finalize=do_finalize,
            )
        if not do_finalize:
            raise NotImplementedError(
                "Glm5SmallBatchMoE: do_finalize=False is not supported on "
                "the fused-kernel path — v110 always finalizes.")
        return self.compute_routed_output(
            hidden_states,
            hidden_states_fp4,
            all_rank_num_tokens,
            do_finalize=True,
        )


class Glm5SmallBatchDecoderLayer(DeepseekV3DecoderLayer):
    """Decoder-layer subclass for the GLM-5 small-batch fused-kernel path.

    PR 1 (in progress): default-on `use_up_gate_silu_kernel` and
    `use_down_allreduce_kernel` so the MoE path uses Glm5SmallBatchFusedMoE.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        aux_stream_dict: Dict,
        is_separate_draft_engine: bool = False,
        mapping_with_cp=None,
    ):
        super().__init__(
            model_config,
            layer_idx,
            aux_stream_dict,
            is_separate_draft_engine=is_separate_draft_engine,
            mapping_with_cp=mapping_with_cp,
        )

        # Per-kernel feature flags. PR 0: all False. PR 1: up-gate-silu +
        # down-allreduce flip on together when enable_glm5_small_batch_fused.
        # PR 2/3: MLA pair / RMSNormExpertProj.
        self.use_mla_front_kernel: bool = False
        self.use_unproj_o_allreduce_kernel: bool = False
        self.use_rmsnorm_expert_proj_kernel: bool = False
        self.use_up_gate_silu_kernel: bool = True   # PR 1
        self.use_down_allreduce_kernel: bool = True # PR 1 (paired)

        self._assert_mla_flags_paired()

        # PR 1: when self.mlp is a Deepseekv3MoE (i.e., this is an MoE layer,
        # not a dense first_k_dense_replace layer), swap its class to
        # Glm5SmallBatchMoE so its experts get a backref to shared_experts
        # for the 257-expert weight merge at load_weights time.
        #
        # CRITICAL: we use object.__setattr__ to bypass nn.Module's __setattr__,
        # which would otherwise register shared_experts as a submodule of
        # `backend` (the Glm5SmallBatchFusedMoE instance). Auto-registering a
        # sibling module as a child corrupts the weight-loader's module tree
        # walk (DeepseekV3WeightLoader iterates named_modules and the
        # double-registration causes the loader to consume shared-expert
        # weights twice → AssertionError in load_weights_fused_gate_up_helper).
        if isinstance(self.mlp, Deepseekv3MoE):
            self.mlp.__class__ = Glm5SmallBatchMoE
            backend = getattr(self.mlp.experts, "backend", self.mlp.experts)
            if isinstance(backend, Glm5SmallBatchFusedMoE):
                object.__setattr__(backend, "_shared_experts_ref",
                                   self.mlp.shared_experts)
                # v110 is ExpertDownAllReduce — the cross-rank AR is fused
                # INTO the down-GEMM. Disable the outer post-MoE fused AR so
                # forward_MoE takes the else-branch at line ~1524: plain
                # next_layer_layernorm(hidden, residual) — RMSNorm + residual
                # add, no AR. PRE_MOE_FUSION stays True; v68 still needs the
                # post-attn AR + RMSNorm output as its input.
                self.fusion_config.POST_MOE_FUSION = False

    def _assert_mla_flags_paired(self) -> None:
        if self.use_mla_front_kernel != self.use_unproj_o_allreduce_kernel:
            raise ValueError(
                "use_mla_front_kernel and use_unproj_o_allreduce_kernel must be "
                "set together (both True or both False). The TileRT split is at "
                "the attention-output boundary: front kernel returns pre-W_O "
                "attention output; back kernel does V-unproj + W_O + AR. "
                "Running one without the other has no defined output contract."
            )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: torch.Tensor,
        spec_metadata: Optional[SpecMetadata] = None,
        **kwargs,
    ):
        # PR 1: pure delegation. The MoE side fires inside self.mlp.experts
        # (Glm5SmallBatchFusedMoE.forward_impl) — no changes to the decoder
        # forward needed yet. PR 2 will branch here on the paired MLA flags
        # to drive the front-then-back kernel sequence.
        return super().forward(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            residual=residual,
            spec_metadata=spec_metadata,
            **kwargs,
        )
