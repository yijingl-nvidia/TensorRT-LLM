# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from mpi4py import MPI

from tensorrt_llm._torch.distributed import AllReduce, AllReduceParams
from tensorrt_llm._torch.distributed.ops import get_allreduce_workspace
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_deepseekv3_fused_moe import Deepseekv3FusedMoE
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

_NUM_RANKS = 8
_HELPER_MODULE_NAME = "_deepseekv3_fused_moe_test_helpers"
_HELPER_FILE = Path(__file__).with_name("test_modeling_deepseekv3_fused_moe.py")
_BASELINE_ABS_THRESHOLD_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_BASELINE_ABS_THRESHOLD"
_FUSED_ABS_THRESHOLD_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_FUSED_ABS_THRESHOLD"
_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD_ENV = (
    "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD"
)
_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD_ENV = (
    "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD"
)
_FUSED_EXPERT_DOWN_AR_RESIDUAL_OP_ENV = (
    "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_DOWN_AR_RESIDUAL_OP"
)
_FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_OP_ENV = (
    "TRTLLM_DEEPSEEKV3_FUSED_MOE_TEST_FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_OP"
)
_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV = "TRTLLM_DEEPSEEKV3_FUSED_EXPERT_DOWN_FINALIZE_MODE"
_RESIDUAL_SEED_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_RESIDUAL_SEED"
_RMS_NORM_EPS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_RMS_NORM_EPS"
_PROFILE_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_PROFILE"
_PROFILE_WARMUP_ITERS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_PROFILE_WARMUP_ITERS"
_PROFILE_ITERS_ENV = "TRTLLM_DEEPSEEKV3_FUSED_MOE_AR_TEST_PROFILE_ITERS"
_DEFAULT_BASELINE_ABS_THRESHOLD = 0.02
_DEFAULT_FUSED_ABS_THRESHOLD = 0.02
_DEFAULT_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD = 3.90625e-03
_DEFAULT_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD = 1.5625e-02
_DEFAULT_RESIDUAL_SEED = 20260602
_DEFAULT_RMS_NORM_EPS = 1e-6
_DEFAULT_PROFILE_WARMUP_ITERS = 20
_DEFAULT_PROFILE_ITERS = 100
_PRODUCTION_FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_OP = (
    "dsv3_fused_expert_down_ar_residual_rms_norm"
)


def _helpers() -> ModuleType:
    module = sys.modules.get(_HELPER_MODULE_NAME)
    if module is not None:
        return module

    spec = importlib.util.spec_from_file_location(_HELPER_MODULE_NAME, _HELPER_FILE)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to import fused MoE test helpers from {_HELPER_FILE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_HELPER_MODULE_NAME] = module
    spec.loader.exec_module(module)
    return module


def _float_env(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _int_env(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    return value.strip().lower() in ("1", "true", "yes", "on")


def _rms_norm_eps() -> float:
    return _float_env(_RMS_NORM_EPS_ENV, _DEFAULT_RMS_NORM_EPS)


def _profile_enabled() -> bool:
    return _bool_env(_PROFILE_ENV, False)


def _profile_warmup_iters() -> int:
    return _int_env(_PROFILE_WARMUP_ITERS_ENV, _DEFAULT_PROFILE_WARMUP_ITERS)


def _profile_iters() -> int:
    return _int_env(_PROFILE_ITERS_ENV, _DEFAULT_PROFILE_ITERS)


def _fused_expert_down_ar_residual_op_name() -> str:
    return os.environ.get(
        _FUSED_EXPERT_DOWN_AR_RESIDUAL_OP_ENV,
        "dsv3_fused_expert_down_ar_residual",
    )


def _fused_expert_down_ar_residual_rms_norm_op_name() -> str:
    return os.environ.get(
        _FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_OP_ENV,
        _PRODUCTION_FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_OP,
    )


def _has_production_fused_expert_down_ar_residual_rms_norm_op() -> bool:
    return hasattr(
        torch.ops.trtllm,
        _PRODUCTION_FUSED_EXPERT_DOWN_AR_RESIDUAL_RMS_NORM_OP,
    )


def _distributed_world_size() -> int:
    return MPI.COMM_WORLD.Get_size()


def _distributed_rank() -> int:
    return MPI.COMM_WORLD.Get_rank()


def _make_residual_and_norm_weight(shape: torch.Size) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_int_env(_RESIDUAL_SEED_ENV, _DEFAULT_RESIDUAL_SEED))
    residual = (0.25 * torch.randn(shape, dtype=torch.float32, generator=generator)).to(
        torch.bfloat16
    )
    norm_weight = (0.5 + torch.rand((shape[-1],), dtype=torch.float32, generator=generator)).to(
        torch.bfloat16
    )
    return residual.cuda(), norm_weight.cuda()


def _rms_norm(tensor: torch.Tensor, norm_weight: torch.Tensor, eps: float) -> torch.Tensor:
    tensor_fp32 = tensor.to(torch.float32)
    output = tensor_fp32 * torch.rsqrt(tensor_fp32.pow(2).mean(-1, keepdim=True) + eps)
    output *= norm_weight.to(torch.float32)
    return output.to(torch.bfloat16)


def _post_moe_allreduce_residual_rms_norm_reference(
    local_outputs: list[torch.Tensor],
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # This mirrors DeepseekV3DecoderLayer's post-MoE production call:
    # AllReduceFusionOp.RESIDUAL_RMS_NORM first sums the TP-local MoE outputs,
    # then writes the residual-add output and RMS-normalized hidden states.
    fusion_op = AllReduceFusionOp.RESIDUAL_RMS_NORM
    assert fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM
    allreduced_output = torch.stack(
        [local_output.to(torch.float32) for local_output in local_outputs],
        dim=0,
    ).sum(dim=0)
    allreduced_output = allreduced_output.to(torch.bfloat16)
    residual_out = (allreduced_output + residual).to(torch.bfloat16)
    hidden_states = _rms_norm(residual_out, norm_weight, _rms_norm_eps())
    return hidden_states, residual_out


def _require_fused_expert_down_ar_residual_op() -> None:
    op_name = _fused_expert_down_ar_residual_op_name()
    if not hasattr(torch.ops.trtllm, op_name):
        pytest.skip(f"missing torch.ops.trtllm.{op_name}")


def _require_fused_expert_down_ar_residual_rms_norm_op() -> None:
    op_name = _fused_expert_down_ar_residual_rms_norm_op_name()
    if not hasattr(torch.ops.trtllm, op_name):
        pytest.skip(f"missing torch.ops.trtllm.{op_name}")


def _select_fused_expert_down_weights(
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    helpers = _helpers()
    helpers._ensure_fused_expert_up_prepacked_tensors(tensors)
    if helpers._prepack_fused_expert_down_enabled():
        helpers._ensure_fused_expert_down_prepacked_tensors(tensors)
        return (
            tensors["routed_w2_weight_packed_fused_expert_down"],
            tensors["shared_down_weight_packed_fused_expert_down"],
        )
    return tensors["routed_w2_weight"], tensors["shared_down_weight_org"]


def _run_fused_expert_up_outputs(
    tensors: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    helpers = _helpers()
    helpers._ensure_fused_expert_up_prepacked_tensors(tensors)
    expert_indices, expert_weights, slot_swiglu_output = (
        Deepseekv3FusedMoE._run_dsv3_fused_expert_up(
            tensors["hidden_states"],
            tensors["router_weight"],
            tensors["routing_bias"],
            tensors["expert_gate_up_weight_packed_fused_expert_up"],
            tensors["expert_gate_up_scale"],
            helpers._top_k(),
            helpers._n_group(),
            helpers._topk_group(),
            helpers._routed_scaling_factor(),
            fused_expert_up_op_name=helpers._fused_expert_up_op_name(),
        )
    )
    return expert_indices, expert_weights, slot_swiglu_output


def _run_fused_expert_down_local(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    helpers = _helpers()
    expert_indices, expert_weights, slot_swiglu_output = _run_fused_expert_up_outputs(tensors)
    routed_w2_weight, shared_down_weight = _select_fused_expert_down_weights(tensors)
    output = torch.empty_like(tensors["hidden_states"])
    return helpers._run_fused_expert_down_chunked(
        slot_swiglu_output,
        expert_indices,
        expert_weights,
        routed_w2_weight,
        tensors["routed_w2_weight_scale"],
        shared_down_weight,
        tensors["shared_down_weight_scale_org"],
        output,
    )


def _run_fused_expert_down_ar_residual(
    tensors: dict[str, torch.Tensor],
    residual: torch.Tensor,
    workspace: torch.Tensor,
    rank: int,
    nranks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    expert_indices, expert_weights, slot_swiglu_output = _run_fused_expert_up_outputs(tensors)
    routed_w2_weight, shared_down_weight = _select_fused_expert_down_weights(tensors)
    local_output = torch.empty_like(tensors["hidden_states"])
    residual_out = torch.empty_like(tensors["hidden_states"])
    fused_expert_down_ar_residual_op = getattr(
        torch.ops.trtllm, _fused_expert_down_ar_residual_op_name()
    )
    residual_out = fused_expert_down_ar_residual_op(
        slot_swiglu_output,
        expert_indices,
        expert_weights,
        routed_w2_weight,
        tensors["routed_w2_weight_scale"],
        shared_down_weight,
        tensors["shared_down_weight_scale_org"],
        residual,
        workspace,
        rank,
        nranks,
        local_output,
        residual_out,
    )
    return local_output, residual_out


def _run_fused_expert_down_ar_residual_rms_norm(
    tensors: dict[str, torch.Tensor],
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    workspace: torch.Tensor,
    rank: int,
    nranks: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expert_indices, expert_weights, slot_swiglu_output = _run_fused_expert_up_outputs(tensors)
    routed_w2_weight, shared_down_weight = _select_fused_expert_down_weights(tensors)
    local_output = torch.empty_like(tensors["hidden_states"])
    residual_out = torch.empty_like(tensors["hidden_states"])
    hidden_out = torch.empty_like(tensors["hidden_states"])
    rms_sums = torch.empty((tensors["hidden_states"].shape[0],), device="cuda", dtype=torch.float32)
    fused_expert_down_ar_residual_rms_norm_op = getattr(
        torch.ops.trtllm,
        _fused_expert_down_ar_residual_rms_norm_op_name(),
    )
    hidden_out = fused_expert_down_ar_residual_rms_norm_op(
        slot_swiglu_output,
        expert_indices,
        expert_weights,
        routed_w2_weight,
        tensors["routed_w2_weight_scale"],
        shared_down_weight,
        tensors["shared_down_weight_scale_org"],
        residual,
        norm_weight,
        workspace,
        rank,
        nranks,
        _rms_norm_eps(),
        local_output,
        residual_out,
        hidden_out,
        rms_sums,
    )
    return local_output, residual_out, hidden_out


def _build_fused_moe_module(
    tensors: dict[str, torch.Tensor],
    rank: int,
    world_size: int,
) -> Deepseekv3FusedMoE:
    helpers = _helpers()
    helpers._ensure_fused_expert_up_prepacked_tensors(tensors)

    hidden_size = tensors["hidden_states"].shape[-1]
    local_intermediate_size = tensors["routed_w2_weight"].shape[-1]
    full_intermediate_size = local_intermediate_size * world_size
    num_experts = tensors["router_weight"].shape[0]
    pretrained_config = SimpleNamespace(
        hidden_size=hidden_size,
        intermediate_size=full_intermediate_size,
        moe_intermediate_size=full_intermediate_size,
        n_group=helpers._n_group(),
        num_experts=num_experts,
        routed_scaling_factor=helpers._routed_scaling_factor(),
        swiglu_limit=helpers._routed_swiglu_limit(),
        topk_group=helpers._topk_group(),
        torch_dtype=torch.bfloat16,
    )
    quant_config = QuantConfig(quant_algo=QuantAlgo.FP8_BLOCK_SCALES)
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=world_size,
            tp_size=world_size,
            pp_size=1,
        ),
        quant_config=quant_config,
        max_num_tokens=helpers._max_fused_kernel_num_tokens(),
        moe_backend="TRTLLM",
        allreduce_strategy=AllReduceStrategy.NCCL,
    )

    old_mode = os.environ.get("TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE")
    os.environ["TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE"] = "wip"
    try:
        fused_moe = Deepseekv3FusedMoE(
            num_experts=num_experts,
            top_k=helpers._top_k(),
            hidden_size=hidden_size,
            intermediate_size=full_intermediate_size,
            shared_expert_intermediate_size=full_intermediate_size,
            aux_stream_dict={},
            layer_idx=0,
            dtype=torch.bfloat16,
            model_config=model_config,
            override_quant_config=quant_config,
        )
    finally:
        if old_mode is None:
            os.environ.pop("TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE", None)
        else:
            os.environ["TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE"] = old_mode

    fused_moe._set_nonpersistent_buffer(fused_moe, "router_weight", tensors["router_weight"])
    fused_moe._set_nonpersistent_buffer(fused_moe, "routing_bias", tensors["routing_bias"])
    fused_moe._set_nonpersistent_buffer(
        fused_moe,
        "expert_gate_up_weight_packed_fused_expert_up",
        tensors["expert_gate_up_weight_packed_fused_expert_up"],
    )
    fused_moe._set_nonpersistent_buffer(
        fused_moe,
        "expert_gate_up_scale",
        tensors["expert_gate_up_scale"],
    )
    fused_moe._set_nonpersistent_buffer(
        fused_moe,
        "routed_w2_weight_scaling_factor",
        tensors["routed_w2_weight_scale"],
    )
    fused_moe._set_nonpersistent_buffer(
        fused_moe,
        "shared_down_weight_scale_org",
        tensors["shared_down_weight_scale_org"],
    )
    if helpers._prepack_fused_expert_down_enabled():
        helpers._ensure_fused_expert_down_prepacked_tensors(tensors)
        fused_moe._set_nonpersistent_buffer(
            fused_moe,
            "shared_down_weight_packed_fused_expert_down",
            tensors["shared_down_weight_packed_fused_expert_down"],
        )
        fused_moe._set_nonpersistent_buffer(
            fused_moe,
            "routed_w2_weight_packed_fused_expert_down",
            tensors["routed_w2_weight_packed_fused_expert_down"],
        )
    else:
        fused_moe._set_nonpersistent_buffer(
            fused_moe,
            "shared_down_weight",
            tensors["shared_down_weight_org"],
        )
        fused_moe._set_nonpersistent_buffer(
            fused_moe,
            "routed_w2_weight",
            tensors["routed_w2_weight"],
        )
    fused_moe._wip_weights_loaded = True
    fused_moe.eval()
    return fused_moe


def _run_fused_moe_module_wip(
    fused_moe: Deepseekv3FusedMoE,
    hidden_states: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    old_mode = os.environ.get("TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE")
    old_finalize_mode = os.environ.get(_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV)
    os.environ["TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE"] = "wip"
    os.environ[_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV] = "allreduce_residual_rms_norm"
    try:
        output = fused_moe(
            hidden_states,
            all_rank_num_tokens=None,
            final_all_reduce_params=AllReduceParams(
                fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
                residual=residual,
                norm_weight=norm_weight,
                eps=_rms_norm_eps(),
                trigger_completion_at_end=False,
            ),
        )
    finally:
        if old_mode is None:
            os.environ.pop("TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE", None)
        else:
            os.environ["TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE"] = old_mode
        if old_finalize_mode is None:
            os.environ.pop(_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV, None)
        else:
            os.environ[_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV] = old_finalize_mode
    assert isinstance(output, tuple)
    return output


def _run_fused_moe_module_local_output(
    fused_moe: Deepseekv3FusedMoE,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    old_mode = os.environ.get("TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE")
    old_finalize_mode = os.environ.get(_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV)
    os.environ["TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE"] = "wip"
    os.environ[_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV] = "local"
    try:
        output = fused_moe(
            hidden_states,
            all_rank_num_tokens=None,
            final_all_reduce_params=AllReduceParams(enable_allreduce=False),
        )
    finally:
        if old_mode is None:
            os.environ.pop("TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE", None)
        else:
            os.environ["TRTLLM_DEEPSEEKV3_FUSED_MOE_MODE"] = old_mode
        if old_finalize_mode is None:
            os.environ.pop(_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV, None)
        else:
            os.environ[_FUSED_EXPERT_DOWN_FINALIZE_MODE_ENV] = old_finalize_mode
    assert isinstance(output, torch.Tensor)
    return output


def _max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    helpers = _helpers()
    rel_error, abs_error, _, _ = helpers._max_errors(actual, expected)
    return rel_error, abs_error


def _write_summary(
    baseline_rel: float,
    baseline_abs: float,
    baseline_residual_rel: float,
    baseline_residual_abs: float,
    fused_rel: float,
    fused_abs: float,
    fused_residual_rel: float,
    fused_residual_abs: float,
) -> None:
    helpers = _helpers()
    path = helpers._debug_output_dir() / "fused_moe_allreduce_reference_errors.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        " ".join(
            [
                "Test allreduce:",
                f"baseline_rel {baseline_rel:.6e}",
                f"baseline_abs {baseline_abs:.6e}",
                f"baseline_residual_rel {baseline_residual_rel:.6e}",
                f"baseline_residual_abs {baseline_residual_abs:.6e}",
                f"fused_rel {fused_rel:.6e}",
                f"fused_abs {fused_abs:.6e}",
                f"fused_residual_rel {fused_residual_rel:.6e}",
                f"fused_residual_abs {fused_residual_abs:.6e}",
            ]
        )
        + "\n"
    )


def test_deepseekv3_fused_moe_post_moe_allreduce_reference() -> None:
    if _distributed_world_size() > 1:
        pytest.skip("single-process reference test is disabled under distributed launch")

    helpers = _helpers()
    helpers._require_cuda_and_ops(require_baseline_ops=True, require_fused_ops=True)

    baseline_local_outputs = []
    pytorch_local_outputs = []
    fused_local_outputs = []
    residual = None
    norm_weight = None

    with torch.inference_mode():
        for rank in range(_NUM_RANKS):
            group = helpers._dump_group(rank)
            tensors = helpers._load_inputs(
                group,
                max_num_tokens=helpers._max_fused_kernel_num_tokens(),
                include_fused_kernel_tensors=True,
            )
            if residual is None or norm_weight is None:
                residual, norm_weight = _make_residual_and_norm_weight(
                    tensors["hidden_states"].shape
                )
            else:
                assert tuple(tensors["hidden_states"].shape) == tuple(residual.shape)

            old_moe = helpers._build_old_moe_baseline(tensors, group)
            baseline_local_outputs.append(helpers._run_multi_stream_baseline(old_moe, tensors))
            pytorch_local_outputs.append(helpers._run_pytorch_reference(tensors))
            fused_local_outputs.append(_run_fused_expert_down_local(tensors))
            torch.cuda.synchronize()
            del old_moe, tensors
            torch.cuda.empty_cache()

        assert residual is not None
        assert norm_weight is not None
        baseline_hidden_states, baseline_residual = _post_moe_allreduce_residual_rms_norm_reference(
            baseline_local_outputs,
            residual,
            norm_weight,
        )
        pytorch_hidden_states, pytorch_residual = _post_moe_allreduce_residual_rms_norm_reference(
            pytorch_local_outputs,
            residual,
            norm_weight,
        )
        fused_hidden_states, fused_residual = _post_moe_allreduce_residual_rms_norm_reference(
            fused_local_outputs,
            residual,
            norm_weight,
        )
        torch.cuda.synchronize()

    baseline_rel, baseline_abs = _max_errors(baseline_hidden_states, pytorch_hidden_states)
    baseline_residual_rel, baseline_residual_abs = _max_errors(
        baseline_residual,
        pytorch_residual,
    )
    fused_rel, fused_abs = _max_errors(fused_hidden_states, pytorch_hidden_states)
    fused_residual_rel, fused_residual_abs = _max_errors(fused_residual, pytorch_residual)
    _write_summary(
        baseline_rel,
        baseline_abs,
        baseline_residual_rel,
        baseline_residual_abs,
        fused_rel,
        fused_abs,
        fused_residual_rel,
        fused_residual_abs,
    )

    baseline_abs_threshold = _float_env(
        _BASELINE_ABS_THRESHOLD_ENV,
        _DEFAULT_BASELINE_ABS_THRESHOLD,
    )
    fused_abs_threshold = _float_env(
        _FUSED_ABS_THRESHOLD_ENV,
        _DEFAULT_FUSED_ABS_THRESHOLD,
    )
    print(
        "DeepSeekV3 fused MoE post-MoE AllReduce reference: "
        f"baseline_abs {baseline_abs:.6e}, "
        f"baseline_residual_abs {baseline_residual_abs:.6e}, "
        f"fused_abs {fused_abs:.6e}, "
        f"fused_residual_abs {fused_residual_abs:.6e}, "
        f"baseline_abs_threshold {baseline_abs_threshold:.6e}, "
        f"fused_abs_threshold {fused_abs_threshold:.6e}"
    )

    assert baseline_abs <= baseline_abs_threshold
    assert baseline_residual_abs <= baseline_abs_threshold
    assert fused_abs <= fused_abs_threshold
    assert fused_residual_abs <= fused_abs_threshold


def _init_mpi_for_trtllm_allreduce() -> tuple[int, int, MPI.Comm]:
    world_size = _distributed_world_size()
    if world_size == 1:
        pytest.skip(
            "run this test with: mpirun --allow-run-as-root --oversubscribe -n 8 "
            "python3 -m pytest -q "
            "tests/unittest/_torch/models/test_modeling_deepseekv3_fused_moe_allreduce.py"
        )
    if world_size != _NUM_RANKS:
        pytest.skip(f"expected WORLD_SIZE={_NUM_RANKS}, got {world_size}")
    if os.environ.get("TLLM_DISABLE_MPI") == "1":
        pytest.skip("MPI-backed TRT-LLM AllReduce test requires TLLM_DISABLE_MPI to be unset")

    rank = _distributed_rank()
    torch.cuda.set_device(rank)
    return rank, world_size, MPI.COMM_WORLD


def _run_trtllm_allreduce_residual_rms_norm(
    allreduce: AllReduce,
    local_output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return allreduce(
        local_output,
        all_reduce_params=AllReduceParams(
            fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
            residual=residual,
            norm_weight=norm_weight,
            eps=_rms_norm_eps(),
            trigger_completion_at_end=False,
        ),
    )


def _write_wip_summary(rank_results: list[dict[str, float]]) -> None:
    helpers = _helpers()
    path = helpers._debug_output_dir() / "fused_moe_allreduce_wip_vs_baseline_errors.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    module_path_checked = any(
        result.get("module_path_checked", 0.0) > 0.5 for result in rank_results
    )
    lines = []
    for result in sorted(rank_results, key=lambda item: item["rank"]):
        fields = [
            f"Test {int(result['rank'])}:",
            f"wip_vs_baseline_residual_rel {result['residual_rel']:.6e}",
            f"wip_vs_baseline_residual_abs {result['residual_abs']:.6e}",
            f"wip_vs_baseline_hidden_rel {result['hidden_rel']:.6e}",
            f"wip_vs_baseline_hidden_abs {result['hidden_abs']:.6e}",
            f"wip_ar_vs_trtllm_ar_residual_rel {result['local_ar_residual_rel']:.6e}",
            f"wip_ar_vs_trtllm_ar_residual_abs {result['local_ar_residual_abs']:.6e}",
            f"wip_ar_vs_trtllm_ar_hidden_rel {result['local_ar_hidden_rel']:.6e}",
            f"wip_ar_vs_trtllm_ar_hidden_abs {result['local_ar_hidden_abs']:.6e}",
            f"fused_finalize_vs_trtllm_ar_residual_rel {result['fused_finalize_residual_rel']:.6e}",
            f"fused_finalize_vs_trtllm_ar_residual_abs {result['fused_finalize_residual_abs']:.6e}",
            f"wip_rms_vs_python_rms_hidden_rel {result['rms_hidden_rel']:.6e}",
            f"wip_rms_vs_python_rms_hidden_abs {result['rms_hidden_abs']:.6e}",
        ]
        if module_path_checked:
            fields.extend(
                [
                    f"module_vs_direct_residual_rel {result['module_residual_rel']:.6e}",
                    f"module_vs_direct_residual_abs {result['module_residual_abs']:.6e}",
                    f"module_vs_direct_hidden_rel {result['module_hidden_rel']:.6e}",
                    f"module_vs_direct_hidden_abs {result['module_hidden_abs']:.6e}",
                    f"module_fused_vs_local_residual_rel {result['module_fused_vs_local_residual_rel']:.6e}",
                    f"module_fused_vs_local_residual_abs {result['module_fused_vs_local_residual_abs']:.6e}",
                    f"module_fused_vs_local_hidden_rel {result['module_fused_vs_local_hidden_rel']:.6e}",
                    f"module_fused_vs_local_hidden_abs {result['module_fused_vs_local_hidden_abs']:.6e}",
                ]
            )
        lines.append(" ".join(fields))
    max_residual_abs = max(result["residual_abs"] for result in rank_results)
    max_hidden_abs = max(result["hidden_abs"] for result in rank_results)
    max_local_ar_residual_abs = max(result["local_ar_residual_abs"] for result in rank_results)
    max_local_ar_hidden_abs = max(result["local_ar_hidden_abs"] for result in rank_results)
    max_fused_finalize_residual_abs = max(
        result["fused_finalize_residual_abs"] for result in rank_results
    )
    max_rms_hidden_abs = max(result["rms_hidden_abs"] for result in rank_results)
    fields = [
        "Average over 8 ranks:",
        f"max_wip_vs_baseline_residual_abs {max_residual_abs:.6e}",
        f"max_wip_vs_baseline_hidden_abs {max_hidden_abs:.6e}",
        f"max_wip_ar_vs_trtllm_ar_residual_abs {max_local_ar_residual_abs:.6e}",
        f"max_wip_ar_vs_trtllm_ar_hidden_abs {max_local_ar_hidden_abs:.6e}",
        f"max_fused_finalize_vs_trtllm_ar_residual_abs {max_fused_finalize_residual_abs:.6e}",
        f"max_wip_rms_vs_python_rms_hidden_abs {max_rms_hidden_abs:.6e}",
    ]
    if module_path_checked:
        max_module_residual_abs = max(result["module_residual_abs"] for result in rank_results)
        max_module_hidden_abs = max(result["module_hidden_abs"] for result in rank_results)
        max_module_fused_vs_local_residual_abs = max(
            result["module_fused_vs_local_residual_abs"] for result in rank_results
        )
        max_module_fused_vs_local_hidden_abs = max(
            result["module_fused_vs_local_hidden_abs"] for result in rank_results
        )
        fields.extend(
            [
                f"max_module_vs_direct_residual_abs {max_module_residual_abs:.6e}",
                f"max_module_vs_direct_hidden_abs {max_module_hidden_abs:.6e}",
                "max_module_fused_vs_local_residual_abs "
                f"{max_module_fused_vs_local_residual_abs:.6e}",
                f"max_module_fused_vs_local_hidden_abs {max_module_fused_vs_local_hidden_abs:.6e}",
            ]
        )
    lines.append(" ".join(fields))
    path.write_text("\n".join(lines) + "\n")


def _profile_cuda_events(fn, warmup_iters: int, profile_iters: int) -> dict[str, float]:
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")
    if profile_iters <= 0:
        raise ValueError(f"profile_iters must be positive, got {profile_iters}")

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]
    for idx in range(profile_iters):
        starts[idx].record()
        fn()
        ends[idx].record()
    torch.cuda.synchronize()

    times_us = torch.tensor(
        [starts[idx].elapsed_time(ends[idx]) * 1000.0 for idx in range(profile_iters)],
        dtype=torch.float32,
    )
    return {
        "mean_us": float(times_us.mean().item()),
        "median_us": float(times_us.median().item()),
        "min_us": float(times_us.min().item()),
        "max_us": float(times_us.max().item()),
    }


def _profile_cuda_graph_events(fn, warmup_iters: int, profile_iters: int) -> dict[str, float]:
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")
    if profile_iters <= 0:
        raise ValueError(f"profile_iters must be positive, got {profile_iters}")

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for _ in range(max(warmup_iters, 1)):
            fn()
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_outputs = fn()
    torch.cuda.synchronize()

    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(profile_iters)]
    for idx in range(profile_iters):
        starts[idx].record()
        graph.replay()
        ends[idx].record()
    torch.cuda.synchronize()

    times_us = torch.tensor(
        [starts[idx].elapsed_time(ends[idx]) * 1000.0 for idx in range(profile_iters)],
        dtype=torch.float32,
    )
    # Keep tensors allocated during capture alive until all graph replays finish.
    del captured_outputs
    return {
        "mean_us": float(times_us.mean().item()),
        "median_us": float(times_us.median().item()),
        "min_us": float(times_us.min().item()),
        "max_us": float(times_us.max().item()),
    }


def _format_profile_average_line(rank_results: list[dict[str, float]]) -> str:
    metric_names = [
        "old_baseline_full_mean_us",
        "trtllm_finalize_mean_us",
        "fused_ar_residual_mean_us",
        "fused_finalize_mean_us",
        "fused_ar_residual_speedup_vs_trtllm_finalize",
        "fused_finalize_speedup_vs_trtllm_finalize",
        "trtllm_finalize_cuda_graph_mean_us",
        "fused_finalize_cuda_graph_mean_us",
        "fused_finalize_cuda_graph_speedup_vs_trtllm_finalize_cuda_graph",
        "module_local_finalize_cuda_graph_mean_us",
        "module_fused_finalize_cuda_graph_mean_us",
        "module_fused_finalize_cuda_graph_speedup_vs_module_local_finalize_cuda_graph",
    ]
    fields = ["Average over 8 ranks:"]
    for name in metric_names:
        value = sum(result[name] for result in rank_results) / len(rank_results)
        suffix = "x" if "speedup_vs" in name else " us"
        fields.append(f"{name} {value:.3f}{suffix}")
    return " ".join(fields)


def test_deepseekv3_fused_moe_post_moe_allreduce_profile() -> None:
    if not _profile_enabled():
        pytest.skip(f"set {_PROFILE_ENV}=1 to enable profiling")

    rank, world_size, comm = _init_mpi_for_trtllm_allreduce()
    helpers = _helpers()
    helpers._require_cuda_and_ops(require_baseline_ops=True, require_fused_ops=True)
    _require_fused_expert_down_ar_residual_op()
    _require_fused_expert_down_ar_residual_rms_norm_op()

    with torch.inference_mode():
        group = helpers._dump_group(rank)
        tensors = helpers._load_inputs(
            group,
            max_num_tokens=helpers._max_fused_kernel_num_tokens(),
            include_fused_kernel_tensors=True,
        )
        residual, norm_weight = _make_residual_and_norm_weight(tensors["hidden_states"].shape)
        old_moe = helpers._build_old_moe_baseline(tensors, group)
        fused_moe = _build_fused_moe_module(tensors, rank, world_size)
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=world_size,
            tp_size=world_size,
            pp_size=1,
        )
        allreduce = AllReduce(
            mapping=mapping,
            strategy=AllReduceStrategy.NCCL,
            dtype=torch.bfloat16,
        )
        workspace = get_allreduce_workspace(mapping)

        expert_indices, expert_weights, slot_swiglu_output = _run_fused_expert_up_outputs(tensors)
        routed_w2_weight, shared_down_weight = _select_fused_expert_down_weights(tensors)
        local_output = torch.empty_like(tensors["hidden_states"])
        residual_out = torch.empty_like(tensors["hidden_states"])
        hidden_out = torch.empty_like(tensors["hidden_states"])
        rms_sums = torch.empty(
            (tensors["hidden_states"].shape[0],), device="cuda", dtype=torch.float32
        )
        fused_expert_down_op = getattr(torch.ops.trtllm, helpers._fused_expert_down_op_name())
        fused_expert_down_ar_residual_op = getattr(
            torch.ops.trtllm,
            _fused_expert_down_ar_residual_op_name(),
        )
        fused_expert_down_ar_residual_rms_norm_op = getattr(
            torch.ops.trtllm,
            _fused_expert_down_ar_residual_rms_norm_op_name(),
        )

        def run_local_down() -> torch.Tensor:
            return fused_expert_down_op(
                slot_swiglu_output,
                expert_indices,
                expert_weights,
                routed_w2_weight,
                tensors["routed_w2_weight_scale"],
                shared_down_weight,
                tensors["shared_down_weight_scale_org"],
                local_output,
            )

        def run_trtllm_finalize() -> tuple[torch.Tensor, torch.Tensor]:
            run_local_down()
            return _run_trtllm_allreduce_residual_rms_norm(
                allreduce,
                local_output,
                residual,
                norm_weight,
            )

        def run_fused_ar_residual() -> torch.Tensor:
            return fused_expert_down_ar_residual_op(
                slot_swiglu_output,
                expert_indices,
                expert_weights,
                routed_w2_weight,
                tensors["routed_w2_weight_scale"],
                shared_down_weight,
                tensors["shared_down_weight_scale_org"],
                residual,
                workspace,
                rank,
                world_size,
                local_output,
                residual_out,
            )

        def run_fused_finalize() -> torch.Tensor:
            return fused_expert_down_ar_residual_rms_norm_op(
                slot_swiglu_output,
                expert_indices,
                expert_weights,
                routed_w2_weight,
                tensors["routed_w2_weight_scale"],
                shared_down_weight,
                tensors["shared_down_weight_scale_org"],
                residual,
                norm_weight,
                workspace,
                rank,
                world_size,
                _rms_norm_eps(),
                local_output,
                residual_out,
                hidden_out,
                rms_sums,
            )

        def run_old_baseline_full() -> tuple[torch.Tensor, torch.Tensor]:
            baseline_local_output = helpers._run_multi_stream_baseline(old_moe, tensors)
            return _run_trtllm_allreduce_residual_rms_norm(
                allreduce,
                baseline_local_output,
                residual,
                norm_weight,
            )

        def run_module_local_finalize() -> tuple[torch.Tensor, torch.Tensor]:
            module_local_output = _run_fused_moe_module_local_output(
                fused_moe,
                tensors["hidden_states"],
            )
            return _run_trtllm_allreduce_residual_rms_norm(
                allreduce,
                module_local_output,
                residual,
                norm_weight,
            )

        def run_module_fused_finalize() -> tuple[torch.Tensor, torch.Tensor]:
            return _run_fused_moe_module_wip(
                fused_moe,
                tensors["hidden_states"],
                residual,
                norm_weight,
            )

        warmup_iters = _profile_warmup_iters()
        profile_iters = _profile_iters()
        torch.cuda.synchronize()
        comm.Barrier()
        old_baseline_full_stats = _profile_cuda_events(
            run_old_baseline_full,
            warmup_iters,
            profile_iters,
        )
        comm.Barrier()
        trtllm_finalize_stats = _profile_cuda_events(
            run_trtllm_finalize,
            warmup_iters,
            profile_iters,
        )
        comm.Barrier()
        fused_ar_residual_stats = _profile_cuda_events(
            run_fused_ar_residual,
            warmup_iters,
            profile_iters,
        )
        comm.Barrier()
        fused_finalize_stats = _profile_cuda_events(
            run_fused_finalize,
            warmup_iters,
            profile_iters,
        )
        comm.Barrier()
        trtllm_finalize_cuda_graph_stats = _profile_cuda_graph_events(
            run_trtllm_finalize,
            warmup_iters,
            profile_iters,
        )
        comm.Barrier()
        fused_finalize_cuda_graph_stats = _profile_cuda_graph_events(
            run_fused_finalize,
            warmup_iters,
            profile_iters,
        )
        comm.Barrier()
        module_local_finalize_cuda_graph_stats = _profile_cuda_graph_events(
            run_module_local_finalize,
            warmup_iters,
            profile_iters,
        )
        comm.Barrier()
        module_fused_finalize_cuda_graph_stats = _profile_cuda_graph_events(
            run_module_fused_finalize,
            warmup_iters,
            profile_iters,
        )
        torch.cuda.synchronize()

    result = {
        "rank": float(rank),
        "old_baseline_full_mean_us": old_baseline_full_stats["mean_us"],
        "trtllm_finalize_mean_us": trtllm_finalize_stats["mean_us"],
        "fused_ar_residual_mean_us": fused_ar_residual_stats["mean_us"],
        "fused_finalize_mean_us": fused_finalize_stats["mean_us"],
        "fused_ar_residual_speedup_vs_trtllm_finalize": trtllm_finalize_stats["mean_us"]
        / fused_ar_residual_stats["mean_us"],
        "fused_finalize_speedup_vs_trtllm_finalize": trtllm_finalize_stats["mean_us"]
        / fused_finalize_stats["mean_us"],
        "trtllm_finalize_cuda_graph_mean_us": trtllm_finalize_cuda_graph_stats["mean_us"],
        "fused_finalize_cuda_graph_mean_us": fused_finalize_cuda_graph_stats["mean_us"],
        "fused_finalize_cuda_graph_speedup_vs_trtllm_finalize_cuda_graph": trtllm_finalize_cuda_graph_stats[
            "mean_us"
        ]
        / fused_finalize_cuda_graph_stats["mean_us"],
        "module_local_finalize_cuda_graph_mean_us": module_local_finalize_cuda_graph_stats[
            "mean_us"
        ],
        "module_fused_finalize_cuda_graph_mean_us": module_fused_finalize_cuda_graph_stats[
            "mean_us"
        ],
        "module_fused_finalize_cuda_graph_speedup_vs_module_local_finalize_cuda_graph": (
            module_local_finalize_cuda_graph_stats["mean_us"]
            / module_fused_finalize_cuda_graph_stats["mean_us"]
        ),
    }
    gathered_results = [item for item in comm.allgather(result) if item is not None]
    if rank == 0:
        print(_format_profile_average_line(gathered_results))
    comm.Barrier()


def test_deepseekv3_fused_moe_post_moe_allreduce_wip_path() -> None:
    rank, world_size, comm = _init_mpi_for_trtllm_allreduce()
    helpers = _helpers()
    helpers._require_cuda_and_ops(require_baseline_ops=True, require_fused_ops=True)
    _require_fused_expert_down_ar_residual_op()
    _require_fused_expert_down_ar_residual_rms_norm_op()

    with torch.inference_mode():
        group = helpers._dump_group(rank)
        tensors = helpers._load_inputs(
            group,
            max_num_tokens=helpers._max_fused_kernel_num_tokens(),
            include_fused_kernel_tensors=True,
        )
        residual, norm_weight = _make_residual_and_norm_weight(tensors["hidden_states"].shape)
        old_moe = helpers._build_old_moe_baseline(tensors, group)
        mapping = Mapping(
            world_size=world_size,
            rank=rank,
            gpus_per_node=world_size,
            tp_size=world_size,
            pp_size=1,
        )
        allreduce = AllReduce(
            mapping=mapping,
            strategy=AllReduceStrategy.NCCL,
            dtype=torch.bfloat16,
        )
        workspace = get_allreduce_workspace(mapping)

        baseline_local_output = helpers._run_multi_stream_baseline(old_moe, tensors)
        wip_local_output, wip_residual, wip_hidden_states = (
            _run_fused_expert_down_ar_residual_rms_norm(
                tensors,
                residual,
                norm_weight,
                workspace,
                rank,
                world_size,
            )
        )
        stage1_local_output, stage1_residual = _run_fused_expert_down_ar_residual(
            tensors,
            residual,
            workspace,
            rank,
            world_size,
        )
        baseline_hidden_states, baseline_residual = _run_trtllm_allreduce_residual_rms_norm(
            allreduce,
            baseline_local_output,
            residual,
            norm_weight,
        )
        wip_python_rms_hidden_states = _rms_norm(wip_residual, norm_weight, _rms_norm_eps())
        wip_local_allreduce_hidden_states, wip_local_allreduce_residual = (
            _run_trtllm_allreduce_residual_rms_norm(
                allreduce,
                wip_local_output,
                residual,
                norm_weight,
            )
        )
        _, stage1_local_allreduce_residual = _run_trtllm_allreduce_residual_rms_norm(
            allreduce,
            stage1_local_output,
            residual,
            norm_weight,
        )
        module_path_checked = _has_production_fused_expert_down_ar_residual_rms_norm_op()
        module_hidden_states = None
        module_residual = None
        module_local_hidden_states = None
        module_local_residual = None
        if module_path_checked:
            fused_moe = _build_fused_moe_module(tensors, rank, world_size)
            module_hidden_states, module_residual = _run_fused_moe_module_wip(
                fused_moe,
                tensors["hidden_states"],
                residual,
                norm_weight,
            )
            module_local_output = _run_fused_moe_module_local_output(
                fused_moe,
                tensors["hidden_states"],
            )
            module_local_hidden_states, module_local_residual = (
                _run_trtllm_allreduce_residual_rms_norm(
                    allreduce,
                    module_local_output,
                    residual,
                    norm_weight,
                )
            )
        torch.cuda.synchronize()

    local_ar_residual_rel, local_ar_residual_abs = _max_errors(
        wip_residual,
        wip_local_allreduce_residual,
    )
    local_ar_hidden_rel, local_ar_hidden_abs = _max_errors(
        wip_hidden_states,
        wip_local_allreduce_hidden_states,
    )
    fused_finalize_residual_rel, fused_finalize_residual_abs = _max_errors(
        stage1_residual,
        stage1_local_allreduce_residual,
    )
    rms_hidden_rel, rms_hidden_abs = _max_errors(wip_hidden_states, wip_python_rms_hidden_states)
    residual_rel, residual_abs = _max_errors(wip_residual, baseline_residual)
    hidden_rel, hidden_abs = _max_errors(wip_hidden_states, baseline_hidden_states)
    if module_path_checked:
        assert module_hidden_states is not None
        assert module_residual is not None
        assert module_local_hidden_states is not None
        assert module_local_residual is not None
        module_residual_rel, module_residual_abs = _max_errors(module_residual, wip_residual)
        module_hidden_rel, module_hidden_abs = _max_errors(
            module_hidden_states,
            wip_hidden_states,
        )
        module_fused_vs_local_residual_rel, module_fused_vs_local_residual_abs = _max_errors(
            module_residual,
            module_local_residual,
        )
        module_fused_vs_local_hidden_rel, module_fused_vs_local_hidden_abs = _max_errors(
            module_hidden_states,
            module_local_hidden_states,
        )
    else:
        module_residual_rel = 0.0
        module_residual_abs = 0.0
        module_hidden_rel = 0.0
        module_hidden_abs = 0.0
        module_fused_vs_local_residual_rel = 0.0
        module_fused_vs_local_residual_abs = 0.0
        module_fused_vs_local_hidden_rel = 0.0
        module_fused_vs_local_hidden_abs = 0.0
    result = {
        "rank": float(rank),
        "residual_rel": residual_rel,
        "residual_abs": residual_abs,
        "hidden_rel": hidden_rel,
        "hidden_abs": hidden_abs,
        "local_ar_residual_rel": local_ar_residual_rel,
        "local_ar_residual_abs": local_ar_residual_abs,
        "local_ar_hidden_rel": local_ar_hidden_rel,
        "local_ar_hidden_abs": local_ar_hidden_abs,
        "fused_finalize_residual_rel": fused_finalize_residual_rel,
        "fused_finalize_residual_abs": fused_finalize_residual_abs,
        "rms_hidden_rel": rms_hidden_rel,
        "rms_hidden_abs": rms_hidden_abs,
        "module_path_checked": 1.0 if module_path_checked else 0.0,
        "module_residual_rel": module_residual_rel,
        "module_residual_abs": module_residual_abs,
        "module_hidden_rel": module_hidden_rel,
        "module_hidden_abs": module_hidden_abs,
        "module_fused_vs_local_residual_rel": module_fused_vs_local_residual_rel,
        "module_fused_vs_local_residual_abs": module_fused_vs_local_residual_abs,
        "module_fused_vs_local_hidden_rel": module_fused_vs_local_hidden_rel,
        "module_fused_vs_local_hidden_abs": module_fused_vs_local_hidden_abs,
    }
    gathered_results = [item for item in comm.allgather(result) if item is not None]
    if rank == 0:
        _write_wip_summary(gathered_results)
        max_residual_abs = max(item["residual_abs"] for item in gathered_results)
        max_hidden_abs = max(item["hidden_abs"] for item in gathered_results)
        max_rms_hidden_abs = max(item["rms_hidden_abs"] for item in gathered_results)
        message = (
            "DeepSeekV3 fused MoE post-MoE AllReduce WIP path: "
            f"max_wip_vs_baseline_residual_abs {max_residual_abs:.6e}, "
            f"max_wip_vs_baseline_hidden_abs {max_hidden_abs:.6e}, "
            f"max_wip_rms_vs_python_rms_hidden_abs {max_rms_hidden_abs:.6e}"
        )
        if any(item["module_path_checked"] > 0.5 for item in gathered_results):
            max_module_residual_abs = max(item["module_residual_abs"] for item in gathered_results)
            max_module_hidden_abs = max(item["module_hidden_abs"] for item in gathered_results)
            message += (
                f", max_module_vs_direct_residual_abs {max_module_residual_abs:.6e}, "
                f"max_module_vs_direct_hidden_abs {max_module_hidden_abs:.6e}"
            )
            max_module_fused_vs_local_residual_abs = max(
                item["module_fused_vs_local_residual_abs"] for item in gathered_results
            )
            max_module_fused_vs_local_hidden_abs = max(
                item["module_fused_vs_local_hidden_abs"] for item in gathered_results
            )
            message += (
                ", max_module_fused_vs_local_residual_abs "
                f"{max_module_fused_vs_local_residual_abs:.6e}, "
                f"max_module_fused_vs_local_hidden_abs {max_module_fused_vs_local_hidden_abs:.6e}"
            )
        print(message)
    comm.Barrier()

    residual_abs_threshold = _float_env(
        _WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD_ENV,
        _DEFAULT_WIP_VS_BASELINE_RESIDUAL_ABS_THRESHOLD,
    )
    hidden_abs_threshold = _float_env(
        _WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD_ENV,
        _DEFAULT_WIP_VS_BASELINE_HIDDEN_ABS_THRESHOLD,
    )
    assert residual_abs <= residual_abs_threshold
    assert hidden_abs <= hidden_abs_threshold
    assert fused_finalize_residual_abs <= residual_abs_threshold
    assert rms_hidden_abs <= hidden_abs_threshold
    assert module_residual_abs <= residual_abs_threshold
    assert module_hidden_abs <= hidden_abs_threshold
    assert module_fused_vs_local_residual_abs <= residual_abs_threshold
    assert module_fused_vs_local_hidden_abs <= hidden_abs_threshold
