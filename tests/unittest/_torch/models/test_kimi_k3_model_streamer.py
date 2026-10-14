# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the installed Model Streamer tensor-name API used by Kimi K3."""

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.kimi_k3_model_streamer import (
    checkpoint_paths_and_keys,
    node_tensor_selection,
    stream_selected_safetensors,
)


def test_node_selection_covers_different_expert_layers(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather(entries, local):
        entries[:] = [local, (local[0], ("layer1.expert3",))]

    monkeypatch.setattr(dist, "all_gather_object", gather)
    selected, ranks = node_tensor_selection({"layer0.expert0"}, rank=0, world_size=2)
    assert selected == {"layer0.expert0", "layer1.expert3"}
    assert ranks == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Model Streamer uses CUDA tensors")
def test_public_tensor_selection(tmp_path):
    pytest.importorskip("runai_model_streamer")
    path = tmp_path / "model.safetensors"
    save_file({"needed": torch.arange(8), "unused": torch.arange(4)}, str(path))
    paths, keys = checkpoint_paths_and_keys(str(tmp_path))
    assert keys == {"needed", "unused"}

    received = {}
    metrics = stream_selected_safetensors(
        paths,
        {"needed"},
        lambda name, tensor: received.setdefault(name, tensor.cpu().clone()),
        distributed=False,
    )
    assert set(received) == {"needed"}
    torch.testing.assert_close(received["needed"], torch.arange(8))
    assert metrics["model_streamer_node_tensors"] == 1
