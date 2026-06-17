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

import pytest
import torch

from tensorrt_llm._torch.autotuner import autotune
from tests.unittest._torch.models.deepseekv3_fused_mla.assertions import _assert_bf16_cuda_finite
from tests.unittest._torch.models.deepseekv3_fused_mla.dump_utils import (
    _NUM_RANKS,
    _dump_group,
    _load_float,
    _load_hidden_states,
    _load_int,
    _load_tensor,
    _max_num_tokens,
    _require_cuda_and_ops,
)
from tests.unittest._torch.models.deepseekv3_fused_mla.reference_ops import (
    _apply_trtllm_mla_rope,
    _dummy_position_ids,
    _run_bf16_bmm,
    _run_fp8_block_scale_linear,
    _run_rms_norm,
)


@pytest.mark.parametrize("rank", range(_NUM_RANKS))
def test_deepseekv3_fused_mla_dump_projection_smoke(rank: int) -> None:
    """
    Smoke test dumped fused MLA projection and downstream local attention flow.

    For each rank, this test loads a realistic dumped MLA activation/weight
    group, runs kv_a_proj_with_mqa, q/kv RMSNorm, q_b_proj, TRTLLM MLA RoPE
    checked against a dummy PyTorch RoPE reference, absorption-style local latent
    attention, v_b_proj, and o_proj. The attention section is a local smoke path
    rather than a reconstruction of production paged-KV AttentionMetadata. Side
    effects include disk reads, CUDA tensor allocations, custom op execution,
    and possible pytest skips when dumps or runtime support are missing.

    Args
    - rank: int, tensor-parallel rank id selected by pytest parametrization.

    Returns
    - None: successful return means the dumped rank flow produced finite tensors
        with expected shapes.
    """
    _require_cuda_and_ops()
    group = _dump_group(rank)

    # Shape symbols used below:
    # M: token count after optional slicing.
    # H: hidden size.
    # Qrank: q LoRA rank.
    # KVrank: compressed KV LoRA rank.
    # Nh: local TP attention heads.
    # Nhcp: local TP/CP attention heads.
    # Dq: q/k head dim.
    # Dnope: q/k non-RoPE head dim.
    # Drope: q/k RoPE head dim.
    # Dv: value head dim.
    hidden_states = _load_hidden_states(group).to(torch.bfloat16)
    # [M_dump, H]
    max_num_tokens = _max_num_tokens()
    if max_num_tokens is not None:
        hidden_states = hidden_states[:max_num_tokens].contiguous()
        # [M, H]
    num_tokens = hidden_states.shape[0]
    assert hidden_states.dim() == 2
    assert num_tokens > 0

    q_lora_rank = _load_int(group, "q_lora_rank")
    kv_lora_rank = _load_int(group, "kv_lora_rank")
    num_heads_tp = _load_int(group, "num_heads_tp")
    num_heads_tp_cp = _load_int(group, "num_heads_tp_cp")
    qk_head_dim = _load_int(group, "qk_head_dim")
    qk_nope_head_dim = _load_int(group, "qk_nope_head_dim")
    qk_rope_head_dim = _load_int(group, "qk_rope_head_dim")
    v_head_dim = _load_int(group, "v_head_dim")
    softmax_scale = _load_float(group, "softmax_scale")
    kv_a_output_width = q_lora_rank + kv_lora_rank + qk_rope_head_dim

    with torch.inference_mode(), autotune():
        # [M, Q_lora_r + kv_lora_r + rope_dim]
        kv_a_output = _run_fp8_block_scale_linear(
            hidden_states,
            _load_tensor(group, "kv_a_proj_with_mqa_weight"),
            _load_tensor(group, "kv_a_proj_with_mqa_weight_scale"),
        )
        assert kv_a_output.shape[-1] == kv_a_output_width
        # q: [M, Q_lora_r]
        # compressed_kv: [M, kv_lora_r]
        # k_pe: [M, rope_dim]
        q, compressed_kv, k_pe = kv_a_output.split(
            [q_lora_rank, kv_lora_rank, qk_rope_head_dim], dim=-1
        )
        # q: [M, Q_lora_r]
        q = _run_rms_norm(
            q,
            _load_tensor(group, "q_a_layernorm_weight"),
            _load_float(group, "q_a_layernorm_variance_epsilon"),
        )
        # [M, kv_lora_r]
        compressed_kv = _run_rms_norm(
            compressed_kv,
            _load_tensor(group, "kv_a_layernorm_weight"),
            _load_float(group, "kv_a_layernorm_variance_epsilon"),
        )
        # [M, n_heads * q_dim]
        q_output = _run_fp8_block_scale_linear(
            q,
            _load_tensor(group, "q_b_proj_weight"),
            _load_tensor(group, "q_b_proj_weight_scale"),
        )
        assert q_output.shape[-1] == num_heads_tp * qk_head_dim
        assert num_heads_tp == num_heads_tp_cp

        # [M, n_heads, q_dim]
        q_heads = q_output.view(num_tokens, num_heads_tp, qk_head_dim)
        # [M], starts at [4, 5, 6, 7]
        dummy_position_ids = _dummy_position_ids(num_tokens, q_heads.device)
        # q_nope: [M, n_heads, q_nope_dim]
        # q_pe: [M, n_heads, rope_dim]
        # k_pe: [M, rope_dim]
        q_nope, q_pe, k_pe = _apply_trtllm_mla_rope(
            q_heads,
            k_pe,
            dummy_position_ids,
            qk_nope_head_dim,
            qk_rope_head_dim,
        )
        # [M, kv_lora_r + rope_dim]
        latent_cache = torch.concat([compressed_kv, k_pe], dim=-1)
        # [n_heads, kv_lora_r, q_nope_dim]
        k_b_proj_trans = _load_tensor(group, "k_b_proj_trans")
        # absorp into query vector
        # [M, n_heads, kv_lora_r]
        q_nope_latent = _run_bf16_bmm(
            q_nope.transpose(0, 1),  # [n_heads, M, q_nope_dim]
            k_b_proj_trans.transpose(1, 2),  # [n_heads, q_nope_dim, kv_lora_r]
        ).transpose(0, 1)
        # [M, n_heads, kv_lora_r + rope_dim]
        fused_q = torch.concat([q_nope_latent, q_pe], dim=-1)

        # ===== TRTLLM C++ MLA Kernel =====
        # dot product of each token's query head to each token's kv latent cache
        # [M, n_heads, M]
        scores = torch.matmul(fused_q.float(), latent_cache.float().transpose(0, 1))
        # [M, n_heads, M]
        scores = scores * softmax_scale
        # [M, n_heads, M]
        weights = torch.softmax(scores, dim=-1)
        # compressed_kv: [M, kv_lora_r]
        # compute weighted latent output for each head: [M, n_heads, kv_lora_r]
        attn_out_latent = torch.matmul(
            weights.float(), compressed_kv[..., :kv_lora_rank].float()
        ).to(torch.bfloat16)
        # ===== TRTLLM C++ MLA Kernel Ends =====

        # [n_heads, v_dim, kv_lora_r]
        v_b_proj = _load_tensor(group, "v_b_proj")
        # [M, n_heads, v_dim]
        attn_output = _run_bf16_bmm(
            attn_out_latent.transpose(0, 1),  # [n_heads, M, kv_lora_r]
            v_b_proj.transpose(1, 2),  # [n_heads, kv_lora_r, v_dim]
        ).transpose(0, 1)
        # [M, n_heads*v_dim]
        attn_output = attn_output.contiguous().view(num_tokens, num_heads_tp_cp * v_head_dim)
        # [M, hidden_size]
        final_output = _run_fp8_block_scale_linear(
            attn_output,
            _load_tensor(group, "o_proj_weight"),
            _load_tensor(group, "o_proj_weight_scale"),
        )

    torch.cuda.synchronize()

    _assert_bf16_cuda_finite("kv_a_output", kv_a_output, num_tokens)
    _assert_bf16_cuda_finite("q_output", q_output, num_tokens)
    _assert_bf16_cuda_finite("latent_cache", latent_cache, num_tokens)
    _assert_bf16_cuda_finite("attn_out_latent", attn_out_latent, num_tokens)
    _assert_bf16_cuda_finite("attn_output", attn_output, num_tokens)
    _assert_bf16_cuda_finite("final_output", final_output, num_tokens)
    assert latent_cache.shape[-1] == kv_lora_rank + qk_rope_head_dim
    assert attn_output.shape[-1] == num_heads_tp_cp * v_head_dim
    assert final_output.shape[-1] == hidden_states.shape[-1]
