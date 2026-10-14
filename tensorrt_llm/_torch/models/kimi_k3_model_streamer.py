# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint selection and public Model Streamer API for Kimi K3."""

from __future__ import annotations

import json
import os
import socket
import time
from collections.abc import Callable, Collection
from pathlib import Path

import torch


def checkpoint_paths_and_keys(checkpoint_dir: str) -> tuple[list[str], set[str]]:
    """Read checkpoint names without materializing any weight tensors."""
    directory = Path(checkpoint_dir)
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        with index.open() as index_file:
            weight_map = json.load(index_file)["weight_map"]
        paths = [str(directory / name) for name in sorted(set(weight_map.values()))]
        keys = set(weight_map)
    else:
        from safetensors import safe_open

        paths = sorted(str(path) for path in directory.glob("*.safetensors"))
        keys = set()
        for path in paths:
            with safe_open(path, framework="pt", device="cpu") as file:
                overlap = keys.intersection(file.keys())
                if overlap:
                    raise ValueError(f"Duplicate safetensors keys: {sorted(overlap)[:10]}")
                keys.update(file.keys())
    if not paths or any(not os.path.isfile(path) for path in paths):
        raise FileNotFoundError(f"No complete safetensors checkpoint in {checkpoint_dir}")
    return paths, keys


def node_tensor_selection(
    local_keys: Collection[str],
    *,
    rank: int,
    world_size: int,
) -> tuple[set[str], int]:
    """Give every rank on a node the same selection for distributed streaming."""
    if world_size == 1:
        return set(local_keys), 1

    import torch.distributed as dist

    if dist.is_initialized():
        if dist.get_rank() != rank or dist.get_world_size() != world_size:
            raise RuntimeError("Torch distributed group does not match the TRT-LLM mapping")
    else:
        if not os.environ.get("MASTER_ADDR") or not os.environ.get("MASTER_PORT"):
            raise RuntimeError("MODEL_STREAMER requires MASTER_ADDR and MASTER_PORT")
        dist.init_process_group("nccl", init_method="env://", rank=rank, world_size=world_size)

    hostname = socket.gethostname()
    local = (hostname, tuple(sorted(local_keys)))
    gathered: list[tuple[str, tuple[str, ...]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local)
    peers = [entry for entry in gathered if entry is not None and entry[0] == hostname]
    if not peers:
        raise RuntimeError(f"No Model Streamer peers found on {hostname}")
    node_keys = set().union(*(entry[1] for entry in peers))
    return node_keys, len(peers)


def stream_selected_safetensors(
    paths: list[str],
    selected_keys: Collection[str],
    consume_tensor: Callable[[str, torch.Tensor], None],
    *,
    distributed: bool,
) -> dict[str, float]:
    """Stream one node-wide key set through Model Streamer's public API.

    The callback must copy or finish reading each tensor before the streamer
    may reuse its buffer. Callers retain tensors needed for grouped transforms.
    """
    try:
        from runai_model_streamer import SafetensorsStreamer
    except ImportError as error:
        raise ImportError(
            "MODEL_STREAMER requires a runai-model-streamer wheel with "
            "SafetensorsStreamer.stream_files(tensor_names=...)"
        ) from error

    selected = set(selected_keys)
    if not selected:
        raise ValueError("MODEL_STREAMER selected no checkpoint tensors")
    seen: set[str] = set()
    selected_bytes = 0
    started = time.perf_counter()
    with SafetensorsStreamer() as streamer:
        streamer.stream_files(
            paths, device="cuda", is_distributed=distributed, tensor_names=selected
        )
        if distributed and not streamer.file_streamer.is_distributed:
            raise RuntimeError("Model Streamer could not activate node-distributed loading")
        for name, tensor in streamer.get_tensors():
            if name not in selected or name in seen:
                raise ValueError(f"Unexpected or duplicate Model Streamer tensor: {name}")
            seen.add(name)
            selected_bytes += tensor.numel() * tensor.element_size()
            consume_tensor(name, tensor)
            # The streamer can reuse its GPU staging buffer when iteration
            # resumes. Finish queued copies before requesting another tensor.
            torch.cuda.synchronize()
    if seen != selected:
        raise RuntimeError(f"Model Streamer omitted tensors: {sorted(selected - seen)[:10]}")
    return {
        "model_streamer_seconds": time.perf_counter() - started,
        "model_streamer_node_tensors": float(len(selected)),
        "model_streamer_node_bytes": float(selected_bytes),
    }
