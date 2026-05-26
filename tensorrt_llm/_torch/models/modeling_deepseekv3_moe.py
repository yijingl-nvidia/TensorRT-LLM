import copy
import math
from typing import Dict, List, Optional

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.bindings.internal.thop import BufferKind
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig

from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.fused_moe import DeepSeekV3MoeRoutingMethod, MoE, MoEWeightLoadingMode, create_moe
from ..modules.fused_moe.fused_moe_wide_ep import WideEPMoE

# isort: off
from ..modules.fused_moe.routing import Deepseekv3RoutingImpl

# isort: on
from ..modules.gated_mlp import GatedMLP
from ..modules.multi_stream_utils import maybe_execute_in_parallel
from ..utils import AuxStreamType, EventType, Fp4QuantizedTensor


class DeepseekV3Gate(nn.Module):
    """
    MoE router / gating module for DeepSeek-V3's mixture-of-experts (MoE) layers.

    It produces the logits over experts that the MoE block then uses to dispatch tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        n_group: int,
        topk_group: int,
        routed_scaling_factor: float,
        dtype: Optional[torch.dtype] = None,
        fuse_routing_kernel: bool = True,
        apply_routing: bool = False,
        moe_backend: str = "CUTLASS",
        use_cute_dsl_bf16_gemm: bool = False,
    ):
        super().__init__()
        self.use_cute_dsl_bf16_gemm = use_cute_dsl_bf16_gemm
        self.weight = nn.Parameter(
            torch.empty((num_experts, hidden_size), dtype=dtype), requires_grad=False
        )
        self.moe_backend = moe_backend
        if moe_backend == "TRTLLM":
            bias_dtype = torch.bfloat16
        else:
            bias_dtype = torch.float32
        self.e_score_correction_bias = nn.Parameter(
            torch.empty((num_experts), dtype=bias_dtype), requires_grad=False
        )

        assert not apply_routing, "DeepseekV3Gate routing is called inside MoE"

        # NOTE: e_score_correction_bias belongs in this gate class but is required by the routing impl.
        self.routing_impl = Deepseekv3RoutingImpl(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            is_fused=fuse_routing_kernel,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_cute_dsl_bf16_gemm and is_sm_100f() and self.weight.dtype == torch.bfloat16:
            input_2d = hidden_states.view(-1, hidden_states.shape[-1])
            m, k = input_2d.shape
            n = self.weight.shape[0]
            output = torch.empty(m, n, dtype=torch.float32, device=hidden_states.device)
            torch.ops.trtllm.cute_dsl_bf16_gemm_blackwell(
                input_2d.contiguous(), self.weight, output
            )
            logits = output.view(*hidden_states.shape[:-1], n)
        else:
            logits = torch.ops.trtllm.dsv3_router_gemm_op(
                hidden_states, self.weight.t(), bias=None, out_dtype=torch.float32
            )
        return logits

    def load_weights(self, weights: List[Dict]):
        assert len(weights) == 1

        self.weight.copy_(weights[0]["weight"][:])

        self.e_score_correction_bias.copy_(
            weights[0]["e_score_correction_bias"][:].to(self.e_score_correction_bias.dtype)
        )

    @property
    def routing_method(self) -> DeepSeekV3MoeRoutingMethod:
        return DeepSeekV3MoeRoutingMethod(
            top_k=self.routing_impl.top_k,
            n_group=self.routing_impl.n_group,
            topk_group=self.routing_impl.topk_group,
            routed_scaling_factor=self.routing_impl.routed_scaling_factor,
            is_fused=self.routing_impl.is_fused,
            # Pass a callable to fetch the tensor from DeepseekV3Gate at runtime, ensuring it is on the correct device
            callable_e_score_correction_bias=lambda: self.e_score_correction_bias,
        )

    def apply(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # topk routing
        return self.routing_method.apply(logits)

    def get_experts_per_token(self):
        return self.routing_method.top_k


@torch.compile(dynamic=True)
def moe_reduce_add_shared_output(
    routed_output: torch.Tensor, shared_output: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Reduce the routed output along the expert dimension and add it to the shared output.

    Args:
        routed_output (torch.Tensor): [num_total_tokens, num_topk_experts, hidden_size]
        shared_output (torch.Tensor): [num_total_tokens, hidden_size]
        out (torch.Tensor | None): [num_total_tokens, hidden_size]

    Returns:
        torch.Tensor: [num_total_tokens, hidden_size]
    """
    routed_reduced = torch.sum(routed_output, dim=1, keepdim=False)
    if out is not None:
        torch.add(shared_output, routed_reduced, out=out)
        return out
    # In-place add to avoid allocating a temporary tensor, reducing peak memory
    return shared_output.add_(routed_reduced)


class Deepseekv3MoE(nn.Module):
    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        shared_expert_intermediate_size: int,
        aux_stream_dict: Dict[AuxStreamType, torch.cuda.Stream],
        layer_idx: int,
        dtype: Optional[torch.dtype] = None,
        model_config: ModelConfig = ModelConfig(),
        override_quant_config: Optional[QuantConfig] = None,
    ):
        from ..distributed import AllReduce

        super().__init__()
        config = model_config.pretrained_config
        self.top_k = top_k
        self.use_dp = model_config.mapping.enable_attention_dp
        self.use_cute_dsl_blockscaling_mm = model_config.use_cute_dsl_blockscaling_mm
        self.swiglu_limit = self._create_swiglu_limit_tensor(model_config, num_experts)
        gate_cls = DeepseekV3Gate
        if hasattr(model_config.pretrained_config, "gate_cls"):
            gate_cls = model_config.pretrained_config.gate_cls
        # gate module that creates the expert router logits
        self.gate = gate_cls(
            hidden_size,
            num_experts,
            top_k=top_k,
            n_group=config.n_group,
            topk_group=config.topk_group,
            routed_scaling_factor=config.routed_scaling_factor,
            dtype=dtype,
            fuse_routing_kernel=True,
            apply_routing=False,
            moe_backend=model_config.moe_backend,
            use_cute_dsl_bf16_gemm=model_config.use_cute_dsl_bf16_gemm,
        )
        # Fused MoE module. It can be different MoE backends based on
        # `model_config.moe_backend`.
        self.experts: MoE = create_moe(
            num_experts=num_experts,
            routing_method=self.gate.routing_method,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=False,  # In both low‑latency and attention‑DP modes, FusedMoE skips the in‑op all‑reduce.
            model_config=model_config,
            override_quant_config=override_quant_config,
            aux_stream_dict=aux_stream_dict,
            layer_idx=layer_idx,
            # DS-R1 W4A8 is only supported through custom quantization script from
            # examples/quantization/quantize_mixed_precision_moe.py
            weight_loading_mode=(
                MoEWeightLoadingMode.W4A8_CUSTOM
                if self._get_experts_quant_config(
                    model_config, layer_idx
                ).layer_quant_mode.is_int4_weight_only_per_group()
                else MoEWeightLoadingMode.VANILLA
            ),
            swiglu_limit=self.swiglu_limit,
        )
        # mapping about how this model is sharded across GPUs/nodes.
        self.mapping: Mapping = model_config.mapping

        shared_quant_config: QuantConfig = self._get_shared_experts_quant_config(
            model_config, layer_idx
        )
        shared_model_config = model_config
        if shared_quant_config is not model_config.quant_config:
            shared_model_config = copy.copy(model_config)
            shared_model_config.quant_config = shared_quant_config

        # For shared experts, use the block size implied by their quant config.
        block_size = 1
        if (
            shared_quant_config is not None
            and shared_quant_config.quant_algo is not None
            and shared_quant_config.group_size is not None
        ):
            block_size = shared_quant_config.group_size

        shared_tp_size, self.shared_output_scale = self._compute_shared_expert_tp_size(
            shared_expert_intermediate_size, block_size
        )

        self.shared_experts = GatedMLP(
            hidden_size=hidden_size,
            intermediate_size=shared_expert_intermediate_size,
            bias=False,
            dtype=dtype,
            config=shared_model_config,
            overridden_tp_size=shared_tp_size,
            reduce_output=False,
            use_cute_dsl_blockscaling_mm=self.use_cute_dsl_blockscaling_mm,
        )
        self.shared_experts_use_fp4: bool = (
            shared_quant_config is not None and shared_quant_config.layer_quant_mode.has_nvfp4()
        )

        self.allreduce: AllReduce | None = None
        if not self.use_dp and self.mapping.tp_size > 1:
            self.allreduce = AllReduce(
                mapping=model_config.mapping, strategy=model_config.allreduce_strategy
            )
        self.aux_stream = aux_stream_dict[AuxStreamType.MoeShared]
        self.event_dict = {key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeShared]}

    def _compute_shared_expert_tp_size(
        self, intermediate_size: int, block_size: int
    ) -> tuple[int, float | None]:
        """
        In the case of Deepseek-R1, the TP size of MLP is capped by intermediate_size // block_size.
        For example, when the intermediate_size is 2048 and block scaling size is 128,
        TP sizes are limited to {1, 2, 4, 8, 16} because of 2048/128 = 16.

        Args:
            intermediate_size (int): MLP intermediate size.
            block_size (int): The quantization block scale size. In the case of Deepseek FP8 recipe,
                it's 128. For NVFP4, it's 16.

        Returns:
            tuple[int, float | None]: A tuple containing (shared_tp_size, shared_output_scale).
                - shared_tp_size: The computed TP size.
                - shared_output_scale: The output scale factor, or None if not needed.
        """

        assert intermediate_size % block_size == 0, (
            "intermediate_size must be divisible by block_size."
        )

        shared_output_scale = None
        # The block scale size is 128, which requires shared_expert_intermediate_size
        # to be divisible by 128.
        if self.use_dp:
            # If using attention DP, the shared experts also use DP instead of TP.
            shared_tp_size = 1
        else:
            # Due to the restriction of block scale size (i.e., 128),
            # the supported TP sizes only include 1, 2, 4, 8, and 16.
            # The math.gcd operation ensures that shared_tp_size falls in the supported
            # TP sizes.
            shared_tp_size = math.gcd(
                intermediate_size // block_size,
                self.mapping.tp_size,
            )
            # If shared_tp_size has been overridden, the output of shared experts needs
            # to be scaled down accordingly before all-reduce.
            if shared_tp_size != self.mapping.tp_size:
                shared_output_scale = shared_tp_size / self.mapping.tp_size

        return shared_tp_size, shared_output_scale

    @staticmethod
    def _create_swiglu_limit_tensor(
        model_config: ModelConfig, num_experts: int
    ) -> Optional[torch.Tensor]:
        swiglu_limit = getattr(model_config.pretrained_config, "swiglu_limit", None)
        if swiglu_limit is None or math.isinf(float(swiglu_limit)):
            return None

        moe_load_balancer_config = model_config.moe_load_balancer
        num_slots = num_experts
        if moe_load_balancer_config is not None and moe_load_balancer_config.num_slots is not None:
            num_slots = moe_load_balancer_config.num_slots

        local_num_slots = num_slots // model_config.mapping.moe_ep_size
        return torch.full(
            (local_num_slots,), float(swiglu_limit), dtype=torch.float32, device="cuda"
        )

    @staticmethod
    def _get_experts_quant_config(model_config, layer_idx: int) -> QuantConfig:
        if getattr(model_config, "quant_config_dict", None) is None:
            return model_config.quant_config
        return model_config.quant_config_dict.get(
            f"model.layers.{layer_idx}.mlp.experts", model_config.quant_config
        )

    @staticmethod
    def _get_shared_experts_quant_config(model_config, layer_idx: int) -> QuantConfig:
        # Prefer explicit per-layer quant config if provided.
        if getattr(model_config, "quant_config_dict", None) is not None:
            base_name = f"model.layers.{layer_idx}.mlp.shared_experts"
            qcfg = model_config.quant_config_dict.get(base_name, None)
            if qcfg is not None:
                return qcfg
            for name, qcfg in model_config.quant_config_dict.items():
                if name.startswith(base_name + "."):
                    return qcfg

        quant_config = model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return quant_config

        base_name = f"model.layers.{layer_idx}.mlp.shared_experts"
        candidates = [
            base_name,
            f"{base_name}.gate_up_proj",
            f"{base_name}.down_proj",
            f"{base_name}.gate_proj",
            f"{base_name}.up_proj",
        ]
        if any(quant_config.is_module_excluded_from_quantization(name) for name in candidates):
            return QuantConfig(
                quant_algo=None, kv_cache_quant_algo=quant_config.kv_cache_quant_algo
            )
        return quant_config

    def compute_routed_output(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Fp4QuantizedTensor | None,
        all_rank_num_tokens: list[int] | None,
        do_finalize: bool,
    ) -> torch.Tensor | list[torch.Tensor]:
        """
        Run the routed experts of MoE.
        Args:
            hidden_states (torch.Tensor): Input RMS-normalied hidden states of MoE.
            hidden_states_fp4 (Fp4QuantizedTensor | None): Optional: input
                RMS-normalized hidden states of MoE in FP4 format.
            all_rank_num_tokens (list[int] | None): Optional: the number of tokens
                in each rank.
            do_finalize (bool): whether to finalize the output.

        Returns:
            torch.Tensor | list[torch.Tensor]: The routed experts outputs if
            do_finalize is True, otherwise returns a list of non-combined routed
            expert outputs for a downstream fused kernel.
        """
        # max-throughput
        use_dp_padding = False
        # Add DP padding on SM120 (Blackwell) for context comm performance
        # TODO: Move this model-agonostic part to MoE
        if self.use_dp and self.mapping.tp_size > 1 and get_sm_version() == 120:
            assert all_rank_num_tokens is not None, (
                "all_rank_num_tokens is required when using attention DP and TP > 1 on Blackwell"
            )
            use_dp_padding = True
            hidden_states = torch.nn.functional.pad(
                hidden_states, (0, 0, 0, max(all_rank_num_tokens) - hidden_states.shape[0])
            )

        # dtype usually float32, [num_total_tokens, num_experts]
        router_logits: torch.Tensor = self.gate(hidden_states)
        return self.experts(
            hidden_states_fp4 if hidden_states_fp4 is not None else hidden_states,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=hidden_states.dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            **({"alltoall_result_do_sum": False} if isinstance(self.experts, WideEPMoE) else {}),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        hidden_states_fp4: Fp4QuantizedTensor | None = None,
        all_rank_num_tokens: list[int] | None = None,
        final_all_reduce_params: AllReduceParams | None = None,
        do_finalize: bool | None = True,
    ) -> torch.Tensor:
        if not do_finalize:
            assert not self.use_dp

        def _compute_shared_output() -> torch.Tensor:
            shared_input = (
                hidden_states_fp4
                if (hidden_states_fp4 is not None and self.shared_experts_use_fp4)
                else hidden_states
            )
            # [num_total_tokens, hidden_size]
            shared_output = self.shared_experts(shared_input)
            if self.shared_output_scale is not None:
                shared_output *= self.shared_output_scale
            return shared_output

        def _compute_routed_output():
            # if do_finalize: return torch.Tensor of shape [num_total_tokens, hidden_size]
            # else, return a list of non-combined router expert outputs for a downstream fused kernel.
            routed_output = self.compute_routed_output(
                hidden_states, hidden_states_fp4, all_rank_num_tokens, do_finalize
            )
            return routed_output

        # NOTE: define compiled helpers at module scope to avoid defining decorators inside compiled frames

        routed_output, shared_output = maybe_execute_in_parallel(
            _compute_routed_output,
            _compute_shared_output,
            self.event_dict[EventType.Main],
            self.event_dict[EventType.MoeShared],
            self.aux_stream,
            disable_on_compile=True,
        )

        if not do_finalize:
            # shared_output: [num_total_tokens, hidden_size]
            # routed_outputs: a list of non-combined router expert outputs for a downstream fused kernel.
            return [shared_output, *routed_output]
        else:
            # shared_output: [num_total_tokens, hidden_size]
            # routed_outputs: [num_total_tokens, (num_topk_experts,) hidden_size]
            if not isinstance(shared_output, torch.Tensor):
                final_hidden_states = shared_output + routed_output
                if not self.use_dp and self.mapping.tp_size > 1:
                    final_hidden_states = self.allreduce(
                        final_hidden_states, all_reduce_params=final_all_reduce_params
                    )
                return final_hidden_states
            output_tensor = None
            if not self.use_dp and self.mapping.tp_size > 1:
                # w: [num_total_tokens, hidden_size]
                w, actual_kind = torch.ops.trtllm.allocate_output(
                    shared_output, self.allreduce.output_buffer_kind, self.mapping.tp_group
                )
                if actual_kind == int(BufferKind.NCCL_WINDOW):
                    output_tensor = w
            if routed_output.dim() == 3:
                # routed_output: [num_total_tokens, num_topk_experts, hidden_size]
                assert shared_output.numel() * self.top_k == routed_output.numel(), (
                    "unmatched tensor shape"
                )
                # reduce along the topk expert dimension:
                # final_hidden_states: [num_total_tokens, hidden_size]
                final_hidden_states = moe_reduce_add_shared_output(
                    routed_output, shared_output, out=output_tensor
                )
            else:
                # routed_output: [num_total_tokens, hidden_size]
                assert shared_output.size() == routed_output.size(), "unmatched tensor shape"
                if output_tensor is not None:
                    final_hidden_states = torch.add(shared_output, routed_output, out=output_tensor)
                else:
                    # In-place add to avoid allocating a temporary tensor, reducing peak memory
                    final_hidden_states = shared_output.add_(routed_output)

            if not self.use_dp and self.mapping.tp_size > 1:
                final_hidden_states = self.allreduce(
                    final_hidden_states, all_reduce_params=final_all_reduce_params
                )

            return final_hidden_states
