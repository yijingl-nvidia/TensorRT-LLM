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
"""Architecture-name dispatcher for the PyTorch backend.

Maps the architecture string from the HF checkpoint's config.json (the
"architectures" field, accessed at runtime as
ModelConfig.pretrained_config.architectures[0] -- e.g. "LlamaForCausalLM",
"Glm4MoeForCausalLM") to the matching nn.Module subclass registered in
MODEL_CLASS_MAPPING via @register_auto_model (see modeling_utils.py).
The registry is populated at import time -- models/__init__.py imports every
modeling_*.py so each decorator runs and adds its entry.

Typical use: AutoModelForCausalLM.from_config(model_config) returns an
instantiated decoder model.
"""

from typing import Generic, Optional, Type

from ..model_config import ModelConfig
from ..utils import model_extra_attrs
from .modeling_utils import (MODEL_CLASS_MAPPING,
                             MODEL_CLASS_VISION_ENCODER_MAPPING,
                             DecoderModelForCausalLM, TConfig, TModel)


class AutoModelForCausalLM(Generic[TModel, TConfig]):
    """Entry point for instantiating a registered model from a parsed config.

    Mirrors HuggingFace's AutoModelForCausalLM.from_config shape, but resolves
    against TRT-LLM's own MODEL_CLASS_MAPPING rather than the HF registry.
    Adding a new model is a two-step change: write modeling_<arch>.py with
    @register_auto_model("ArchName"), then import it in models/__init__.py
    so the decorator runs.
    """

    @staticmethod
    def _resolve_class(config: ModelConfig) -> Optional[Type]:
        """Resolve a registered model class from a parsed config without instantiating.

        Looks up config.pretrained_config.architectures[0] (the string that
        appears under the "architectures" key in the HF checkpoint's
        config.json) in MODEL_CLASS_MAPPING. Two architecture-string rewrites
        happen before the lookup:

        1. Eagle3 detection -- checkpoints that expose draft_vocab_size are
           routed to an EAGLE3<Arch> variant. Marked as a hack pending cleaner
           first-class checkpoint tagging.
        2. MTP draft re-routing -- when an MoE arch (DeepSeek-V3, Glm4-MoE,
           Exaone-MoE) is paired with a spec config whose max_draft_len == 0
           (i.e. the draft pass), routing redirects to the shared
           MTPDraftModelForCausalLM class.

        For multimodal encoder-only mode (config.mm_encoder_only), looks up
        MODEL_CLASS_VISION_ENCODER_MAPPING instead.

        Args:
            config: ModelConfig whose pretrained_config (the parsed HF
                config.json, a transformers.PretrainedConfig instance) carries
                the architectures field used as the lookup key.

        Returns:
            The registered model class, or None if pretrained_config is
            missing/empty or the architecture is not registered.
        """
        pretrained_config = config.pretrained_config
        if pretrained_config is None or not pretrained_config.architectures:
            return None

        model_arch = pretrained_config.architectures[0]

        if config.mm_encoder_only:
            vision_encoder_info = MODEL_CLASS_VISION_ENCODER_MAPPING.get(
                model_arch)
            if vision_encoder_info is None:
                return None
            vision_encoder_cls, _ = vision_encoder_info
            return vision_encoder_cls

        # Hack to detect eagle3 checkpoints. TODO: should we provide
        # our own checkpoints with the correct arch? It would let us
        # avoid nasty stuff like this.
        if hasattr(pretrained_config, "draft_vocab_size"):
            model_arch = model_arch.replace("Eagle3",
                                            "")  # Strip the appended EAGLE3
            model_arch = "EAGLE3" + model_arch

        if model_arch in (
                "DeepseekV3ForCausalLM", "Glm4MoeForCausalLM",
                "ExaoneMoEForCausalLM"
        ) and config.spec_config is not None and config.spec_config.max_draft_len == 0:
            model_arch = "MTPDraftModelForCausalLM"

        return MODEL_CLASS_MAPPING.get(model_arch)

    @staticmethod
    def from_config(
        config: ModelConfig[TConfig],
    ) -> DecoderModelForCausalLM[TModel, TConfig]:
        """Resolve and instantiate the registered model class for config.

        For DecoderModelForCausalLM subclasses, weight creation is deferred
        out of __init__ (skip_create_weights_in_init = True); the executor
        will call the model later to materialize weights once parallel layout
        and quant config are finalized.

        Args:
            config: ModelConfig whose pretrained_config is the parsed HF
                config.json. Its architectures[0] is used to look up the
                model class.

        Returns:
            An instantiated DecoderModelForCausalLM (or, in multimodal
            encoder-only mode, the registered vision encoder).

        Raises:
            ValueError: If pretrained_config or its architectures field is
                missing, or if the architecture string is unknown -- typically
                because the matching modeling_*.py is not imported in
                models/__init__.py (decorator never ran), or the HF
                config.json names an architecture TRT-LLM does not yet
                support.
        """
        if config.pretrained_config is None:
            raise ValueError(
                "pretrained_config from HF checkpoint config.json is required")
        if not config.pretrained_config.architectures:
            raise ValueError(
                "architectures field is required in pretrained_config from HF checkpoint config.json"
            )
        if config.mm_encoder_only:
            model_arch = config.pretrained_config.architectures[0]
            vision_encoder_info = MODEL_CLASS_VISION_ENCODER_MAPPING.get(
                model_arch)
            if vision_encoder_info is None:
                raise ValueError(
                    f"Unknown architecture for AutoModelForMultimodalEncoder: {model_arch}"
                )
            vision_encoder_cls, vlm_base_model = vision_encoder_info
            return vision_encoder_cls(config, vlm_base_model)
        cls = AutoModelForCausalLM._resolve_class(config)
        if cls is None:
            raise ValueError(
                f"Unknown architecture for AutoModelForCausalLM: {config.pretrained_config.architectures[0]}"
            )
        if issubclass(cls, DecoderModelForCausalLM):
            config._frozen = False
            config.skip_create_weights_in_init = True
            config._frozen = True
        extra_attrs = config.extra_attrs
        # store extra_attrs as thread-local data for cls to use in __init__()
        with model_extra_attrs(extra_attrs):
            model = cls(config)
        model.extra_attrs = extra_attrs
        return model
