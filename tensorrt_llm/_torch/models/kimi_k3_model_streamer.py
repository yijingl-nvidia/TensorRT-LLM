# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental selected-tensor ModelStreamer support for Kimi K3."""

from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class _SelectedRange:
    """One contiguous run of selected tensors in a safetensors shard."""

    request_id: int
    path: str
    tensor_metadata: tuple[Any, ...]
    offset: int
    sizes: tuple[int, ...]


@dataclass(frozen=True)
class _NodeRankNeeds:
    """Union and intersection of checkpoint needs for one node."""

    non_expert_union: frozenset[str]
    non_expert_intersection: frozenset[str]
    expert_union: frozenset[int]
    expert_intersection: frozenset[int]
    ranks_on_node: int


def _combine_node_rank_needs(
    gathered_needs: Sequence[tuple[str, Sequence[str], Sequence[int]]],
    hostname: str,
) -> _NodeRankNeeds:
    """Combine compact rank requirements gathered across the world group."""
    node_needs = [entry for entry in gathered_needs if entry[0] == hostname]
    if not node_needs:
        raise RuntimeError(f"No gathered ModelStreamer requirements for host {hostname!r}.")

    non_expert_sets = [set(entry[1]) for entry in node_needs]
    expert_sets = [set(entry[2]) for entry in node_needs]
    return _NodeRankNeeds(
        non_expert_union=frozenset().union(*non_expert_sets),
        non_expert_intersection=frozenset.intersection(
            *(frozenset(keys) for keys in non_expert_sets)
        ),
        expert_union=frozenset().union(*expert_sets),
        expert_intersection=frozenset.intersection(
            *(frozenset(expert_ids) for expert_ids in expert_sets)
        ),
        ranks_on_node=len(node_needs),
    )


def gather_node_rank_needs(
    non_expert_keys: Collection[str], local_expert_ids: Collection[int]
) -> _NodeRankNeeds:
    """Gather compact Kimi K3 checkpoint ownership metadata across ranks."""
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() <= 1:
        raise RuntimeError(
            "Kimi K3 distributed ModelStreamer requires initialized "
            "torch.distributed with more than one rank."
        )

    hostname = socket.gethostname()
    local_needs = (
        hostname,
        tuple(sorted(non_expert_keys)),
        tuple(sorted(local_expert_ids)),
    )
    gathered_needs: list[tuple[str, Sequence[str], Sequence[int]] | None] = [
        None
    ] * dist.get_world_size()
    dist.all_gather_object(gathered_needs, local_needs)
    if any(entry is None for entry in gathered_needs):
        raise RuntimeError("ModelStreamer rank-requirement gathering returned an empty entry.")
    return _combine_node_rank_needs(
        [entry for entry in gathered_needs if entry is not None], hostname
    )


def _select_tensor_ranges(
    paths: Sequence[str],
    files_metadata: Sequence[Any],
    selected_keys: Collection[str],
) -> tuple[list[_SelectedRange], int]:
    """Group rank-local tensors into contiguous ModelStreamer 0.16.1 requests."""
    selected_key_set = set(selected_keys)
    found_keys: set[str] = set()
    selected_ranges = []
    selected_bytes = 0
    next_request_id = 0

    if len(paths) != len(files_metadata):
        raise ValueError(
            "ModelStreamer returned metadata for "
            f"{len(files_metadata)} files, expected {len(paths)}."
        )

    for path, file_metadata in zip(paths, files_metadata):
        tensor_metadata = [
            metadata
            for metadata in file_metadata.tensors_metadata
            if metadata.name in selected_key_set
        ]
        if not tensor_metadata:
            continue

        duplicate_keys = found_keys.intersection(metadata.name for metadata in tensor_metadata)
        if duplicate_keys:
            raise ValueError(
                "Kimi K3 ModelStreamer found duplicate checkpoint tensors: "
                f"{sorted(duplicate_keys)[:10]}"
            )
        found_keys.update(metadata.name for metadata in tensor_metadata)

        range_start = 0
        for tensor_index in range(1, len(tensor_metadata) + 1):
            continues_range = (
                tensor_index < len(tensor_metadata)
                and tensor_metadata[tensor_index].offsets.start
                == tensor_metadata[tensor_index - 1].offsets.end
            )
            if continues_range:
                continue

            range_metadata = tuple(tensor_metadata[range_start:tensor_index])
            sizes = tuple(metadata.get_bytesize() for metadata in range_metadata)
            selected_bytes += sum(sizes)
            selected_ranges.append(
                _SelectedRange(
                    request_id=next_request_id,
                    path=path,
                    tensor_metadata=range_metadata,
                    offset=file_metadata.offset + range_metadata[0].offsets.start,
                    sizes=sizes,
                )
            )
            next_request_id += 1
            range_start = tensor_index

    missing_keys = selected_key_set - found_keys
    if missing_keys:
        raise KeyError(
            f"Kimi K3 ModelStreamer could not find {len(missing_keys)} selected "
            f"checkpoint tensors, e.g. {sorted(missing_keys)[:10]}"
        )
    return selected_ranges, selected_bytes


def _checkpoint_paths(checkpoint_dir: str) -> list[str]:
    checkpoint_path = Path(checkpoint_dir)
    index_path = checkpoint_path / "model.safetensors.index.json"
    if index_path.is_file():
        with index_path.open() as index_file:
            weight_map = json.load(index_file)["weight_map"]
        file_names = sorted(set(weight_map.values()))
        return [str(checkpoint_path / file_name) for file_name in file_names]

    paths = sorted(str(path) for path in checkpoint_path.glob("*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No safetensors files found in {checkpoint_dir}")
    return paths


def stream_selected_safetensors(
    checkpoint_dir: str,
    selected_keys: Collection[str],
    consume_tensor: Callable[[str, torch.Tensor], None],
    *,
    device: str | torch.device = "cpu",
    is_distributed: bool = False,
    metric_prefix: str = "model_streamer",
) -> dict[str, float]:
    """Stream selected Kimi K3 tensors and synchronously pass each to a consumer.

    The consumer must finish using a tensor before returning. ModelStreamer owns
    and reuses the staging buffer after the callback returns.
    """
    try:
        from runai_model_streamer import DistributedStreamer, FileChunks
        from runai_model_streamer.safetensors_streamer.safetensors_pytorch import (
            SafetensorsMetadata,
            create_torch_tensor,
        )
    except ImportError as error:
        raise ImportError(
            "KIMI_K3_MODEL_STREAMER=1 requires runai-model-streamer; install "
            "runai-model-streamer==0.16.1 in the TRT-LLM runtime environment."
        ) from error

    paths = _checkpoint_paths(checkpoint_dir)
    total_start = time.perf_counter()
    with DistributedStreamer() as streamer:
        metadata_start = time.perf_counter()
        files_metadata = SafetensorsMetadata.from_files(streamer, paths, None)
        selected_ranges, selected_bytes = _select_tensor_ranges(
            paths, files_metadata, selected_keys
        )
        metadata_seconds = time.perf_counter() - metadata_start

        requests = [
            FileChunks(
                selected_range.request_id,
                selected_range.path,
                selected_range.offset,
                list(selected_range.sizes),
            )
            for selected_range in selected_ranges
        ]
        metadata_by_request = {
            selected_range.request_id: selected_range.tensor_metadata
            for selected_range in selected_ranges
        }

        stream_start = time.perf_counter()
        streamer.stream_files(
            requests,
            credentials=None,
            device=str(device),
            is_distributed=is_distributed,
        )
        if is_distributed and not streamer.is_distributed:
            raise RuntimeError(
                "ModelStreamer rejected distributed mode; check torch.distributed, "
                "the process-group backend, and available staging-buffer memory."
            )
        for request_id, tensor_index, buffer in streamer.get_chunks():
            metadata = metadata_by_request[request_id][tensor_index]
            consume_tensor(metadata.name, create_torch_tensor(buffer, metadata))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        stream_and_load_seconds = time.perf_counter() - stream_start

    return {
        f"{metric_prefix}_metadata_seconds": metadata_seconds,
        f"{metric_prefix}_stream_and_load_seconds": stream_and_load_seconds,
        f"{metric_prefix}_total_seconds": time.perf_counter() - total_start,
        f"{metric_prefix}_selected_tensors": float(len(set(selected_keys))),
        f"{metric_prefix}_selected_bytes": float(selected_bytes),
    }
