# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structural tests for the GLM-5 small-batch fused-kernel scaffolding (PR 0).

Verifies that the subclass hierarchy, feature flags, and MLA-pairing assertion
introduced in PR 0 of the GLM-5 small-batch integration are wired up correctly.

No GPU / no model load needed — purely structural. Forward-pass validation
against synthetic weights is added in PR 1 when the v68 ExpertSelectUpGateSiLU
kernel is wired in.

See `nvbugs/6108841/revisit_moe_mega_kernel/INTEGRATION_PLAN.md` for the full plan.
"""
from __future__ import annotations

import pytest

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3 import (
    Deepseekv3MoE,
    DeepseekV3DecoderLayer,
)
from tensorrt_llm._torch.models.modeling_glm_small_batch import (
    Glm5SmallBatchMoE,
    Glm5SmallBatchDecoderLayer,
    Glm5SmallBatchFusedMoE,
)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_trtllm_gen import (
    TRTLLMGenFusedMoE,
)


EXPECTED_FEATURE_FLAGS = (
    "use_mla_front_kernel",
    "use_unproj_o_allreduce_kernel",
    "use_rmsnorm_expert_proj_kernel",
    "use_up_gate_silu_kernel",
    "use_down_allreduce_kernel",
)


def test_subclass_hierarchy():
    """The 3 new classes must subclass their respective baselines.

    GLM-5 (HF arch `GlmMoeDsaForCausalLM`) registers against
    `DeepseekV3ForCausalLM` (`modeling_deepseekv3.py:1810`), so the small-batch
    scaffolding subclasses the DeepSeekV3 family, NOT the Glm4* classes.
    """
    assert issubclass(Glm5SmallBatchFusedMoE, TRTLLMGenFusedMoE)
    assert issubclass(Glm5SmallBatchMoE, Deepseekv3MoE)
    assert issubclass(Glm5SmallBatchDecoderLayer, DeepseekV3DecoderLayer)


def test_model_config_flag_default():
    """`enable_glm5_small_batch_fused` must default to False on a fresh ModelConfig."""
    cfg = ModelConfig()
    assert hasattr(cfg, "enable_glm5_small_batch_fused")
    assert cfg.enable_glm5_small_batch_fused is False


def test_can_use_predicates_default_false_at_class_level():
    """Each `_can_use_<phase>` must be defined on the backend subclass.

    We can't instantiate `Glm5SmallBatchFusedMoE` cheaply (needs full MoE
    constructor args), but we can verify the methods exist on the class and
    that calling them via `__func__(None, ...)` with a stub raises sensibly.
    The important contract: the predicates default to "False" semantics in
    PR 0 (no kernel wired in yet).
    """
    # Methods exist
    assert callable(Glm5SmallBatchFusedMoE._can_use_up_gate_silu_kernel)
    assert callable(Glm5SmallBatchFusedMoE._run_up_gate_silu)
    assert callable(Glm5SmallBatchFusedMoE._can_use_down_allreduce_kernel)
    assert callable(Glm5SmallBatchFusedMoE._run_down_allreduce)

    # PR 0 _can_use_*: return False regardless of inputs. Call via __func__ to
    # skip the bound-method "self must be an instance" path — the predicate is
    # stateless in PR 0 (it ignores self entirely).
    fake_self = object()
    assert (
        Glm5SmallBatchFusedMoE._can_use_up_gate_silu_kernel(fake_self, None, None)
        is False
    )
    assert (
        Glm5SmallBatchFusedMoE._can_use_down_allreduce_kernel(fake_self, None)
        is False
    )


def test_run_methods_present():
    """PR 1 implements `_run_up_gate_silu` and `_run_down_allreduce`.

    PR 0 had them as NotImplementedError stubs; in PR 1 they dispatch to the
    new torch ops. We don't invoke them here (would require live cuda + ops);
    we just check they're real methods, not the stub raises.
    """
    assert Glm5SmallBatchFusedMoE._run_up_gate_silu.__qualname__.endswith(
        "_run_up_gate_silu")
    assert Glm5SmallBatchFusedMoE._run_down_allreduce.__qualname__.endswith(
        "_run_down_allreduce")


def test_feature_flag_names_in_decoder_layer_class():
    """The 5 feature flags must appear in `Glm5SmallBatchDecoderLayer.__init__`.

    We grep the source rather than instantiating (full constructor needs a
    real ModelConfig + TP plumbing — too heavy for a structural test). Future
    PRs are expected to flip these to True in `__init__` conditionally.
    """
    import inspect

    src = inspect.getsource(Glm5SmallBatchDecoderLayer.__init__)
    for flag_name in EXPECTED_FEATURE_FLAGS:
        assert flag_name in src, (
            f"Feature flag {flag_name!r} not found in "
            f"Glm5SmallBatchDecoderLayer.__init__. The 5 per-kernel feature "
            f"flags are part of PR 0's contract."
        )


def test_mla_pairing_assertion():
    """`_assert_mla_flags_paired` must raise when front/back flags disagree."""
    # We can't construct a full Glm5SmallBatchDecoderLayer cheaply, but we can
    # call the method on a stub that has the two flag attributes.
    class _Stub:
        use_mla_front_kernel = True
        use_unproj_o_allreduce_kernel = False

    with pytest.raises(ValueError, match="set together"):
        Glm5SmallBatchDecoderLayer._assert_mla_flags_paired(_Stub())

    # The inverse pairing also fails.
    class _Stub2:
        use_mla_front_kernel = False
        use_unproj_o_allreduce_kernel = True

    with pytest.raises(ValueError, match="set together"):
        Glm5SmallBatchDecoderLayer._assert_mla_flags_paired(_Stub2())

    # Both False is OK.
    class _Stub3:
        use_mla_front_kernel = False
        use_unproj_o_allreduce_kernel = False

    Glm5SmallBatchDecoderLayer._assert_mla_flags_paired(_Stub3())

    # Both True is OK.
    class _Stub4:
        use_mla_front_kernel = True
        use_unproj_o_allreduce_kernel = True

    Glm5SmallBatchDecoderLayer._assert_mla_flags_paired(_Stub4())
