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

import torch

from tests.unittest._torch.models.test_modeling_deepseekv3_attention import (
    _KV_LORA_RANK,
    _LOCAL_NUM_HEADS,
    _QK_HEAD_DIM,
    _QK_NOPE_HEAD_DIM,
)


def _assert_bf16_cuda_finite(name: str, tensor: torch.Tensor, num_tokens: int) -> None:
    """
    Assert that a tensor is finite BF16 CUDA data with the expected token count.

    The smoke test uses this for every major output tensor. Assertion failures
    identify the named tensor in the failure message. It has no side effects
    beyond test assertions.

    Args
    - name: str, human-readable tensor name used in assertion messages.
    - tensor: torch.Tensor, shape [M, ...], bf16, CUDA tensor to validate.
    - num_tokens: int, expected leading dimension M.

    Returns
    - None: successful return means all validation assertions passed.
    """
    assert tensor.is_cuda, f"{name} must be a CUDA tensor"
    assert tensor.dtype == torch.bfloat16, f"{name} has dtype {tensor.dtype}"
    assert tensor.shape[0] == num_tokens, f"{name} has shape {tuple(tensor.shape)}"
    assert torch.isfinite(tensor.float()).all().item(), f"{name} contains NaN or Inf"


def _assert_q_b_selector_prefix_close(
    fused_q: torch.Tensor,
    q_b_proj_output: torch.Tensor,
) -> None:
    """
    Assert that selector k_b exposes q_b output exactly in fused_q.

    With _selector_k_b_proj_trans, the first 192 latent-prefix dimensions should
    be exactly the q_nope slice from q_b_proj_output and the remaining 320
    dimensions should be exactly zero. This is intentionally stricter than the
    full attention output tolerance so a one-BF16-step q_b projection mismatch
    fails the test.

    Args
    - fused_q: torch.Tensor, shape [4, 8, 576], bf16, output buffer from the
        fused decode kernel.
    - q_b_proj_output: torch.Tensor, shape [4, 2048], bf16, existing TRTLLM q_b
        projection output before splitting into heads.

    Returns
    - None: successful return means the q_b projection path is bit-close enough
        to the Linear reference for all exposed q_nope values.
    """
    # [num_tokens, 8, 256]
    q_heads = q_b_proj_output.view(q_b_proj_output.shape[0], _LOCAL_NUM_HEADS, _QK_HEAD_DIM)
    # [num_tokens, 8, 192]
    q_nope = q_heads[..., :_QK_NOPE_HEAD_DIM]
    # [num_tokens, 8, 512]
    expected_prefix = torch.zeros_like(fused_q[..., :_KV_LORA_RANK])
    # [num_tokens, 8, 192]
    expected_prefix[..., :_QK_NOPE_HEAD_DIM] = q_nope
    torch.testing.assert_close(
        fused_q[..., :_KV_LORA_RANK],
        expected_prefix,
        rtol=0.0,
        atol=0.0,
    )
