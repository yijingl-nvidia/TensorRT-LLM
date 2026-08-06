# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.models.kimi_k3_model_streamer import (
    _combine_node_rank_needs,
    _select_tensor_ranges,
)


def _tensor_metadata(name, start, end):
    return SimpleNamespace(
        name=name,
        offsets=SimpleNamespace(start=start, end=end),
        get_bytesize=lambda: end - start,
    )


def test_select_tensor_ranges_uses_absolute_offsets_and_rank_local_keys():
    files_metadata = [
        SimpleNamespace(
            offset=128,
            tensors_metadata=(
                _tensor_metadata("rank0.weight", 0, 16),
                _tensor_metadata("other_rank.weight", 16, 48),
            ),
        ),
        SimpleNamespace(
            offset=256,
            tensors_metadata=(_tensor_metadata("rank0.scale", 32, 40),),
        ),
    ]

    selected_files, selected_bytes = _select_tensor_ranges(
        ("model-1.safetensors", "model-2.safetensors"),
        files_metadata,
        {"rank0.weight", "rank0.scale"},
    )

    assert [selected_file.request_id for selected_file in selected_files] == [0, 1]
    assert selected_files[0].offset == 128
    assert selected_files[0].sizes == (16,)
    assert selected_files[1].offset == 288
    assert selected_files[1].sizes == (8,)
    assert selected_bytes == 24


def test_select_tensor_ranges_groups_only_adjacent_selected_tensors():
    files_metadata = [
        SimpleNamespace(
            offset=128,
            tensors_metadata=(
                _tensor_metadata("first", 0, 16),
                _tensor_metadata("second", 16, 48),
                _tensor_metadata("skipped", 48, 64),
                _tensor_metadata("third", 64, 72),
            ),
        )
    ]

    selected_ranges, _ = _select_tensor_ranges(
        ("model.safetensors",), files_metadata, {"first", "second", "third"}
    )

    assert len(selected_ranges) == 2
    assert selected_ranges[0].offset == 128
    assert selected_ranges[0].sizes == (16, 32)
    assert selected_ranges[1].offset == 192
    assert selected_ranges[1].sizes == (8,)


def test_select_tensor_ranges_rejects_missing_key():
    files_metadata = [
        SimpleNamespace(
            offset=128,
            tensors_metadata=(_tensor_metadata("present", 0, 16),),
        )
    ]

    with pytest.raises(KeyError, match="could not find 1 selected"):
        _select_tensor_ranges(("model.safetensors",), files_metadata, {"present", "missing"})


def test_combine_node_rank_needs_builds_union_and_intersection():
    needs = _combine_node_rank_needs(
        (
            ("node-a", ("shared", "rank-0"), (0, 1)),
            ("node-a", ("shared", "rank-1"), (2, 3)),
            ("node-b", ("other-node",), (4, 5)),
        ),
        "node-a",
    )

    assert needs.non_expert_union == {"shared", "rank-0", "rank-1"}
    assert needs.non_expert_intersection == {"shared"}
    assert needs.expert_union == {0, 1, 2, 3}
    assert needs.expert_intersection == set()
    assert needs.ranks_on_node == 2


def test_combine_node_rank_needs_rejects_missing_host():
    with pytest.raises(RuntimeError, match="No gathered ModelStreamer requirements"):
        _combine_node_rank_needs((("node-a", (), ()),), "node-b")
