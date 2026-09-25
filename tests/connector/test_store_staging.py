# SPDX-License-Identifier: Apache-2.0
"""Contracts for slot-major snapshots into a caller-owned staging lease."""

import pytest

pytest.importorskip("torch")

import torch

from daser.connector.worker.staging import copy_cross_layer_kv_cache_to_staging


@pytest.mark.parametrize("blocks", [[], [2], [1, 2, 3], [4, 1, 3], [3, 1, 3]])
@pytest.mark.parametrize("prebuilt_index", [False, True])
def test_cross_layer_snapshot_preserves_bytes_and_lease(
    blocks: list[int], prebuilt_index: bool
) -> None:
    """Keep source order, duplicate blocks, lease offsets and guard bytes intact."""
    source = torch.arange(5 * 3 * 2 * 4 * 2 * 8, dtype=torch.int16).reshape(
        5, 3, 2, 4, 2, 8
    )
    original = source.clone()
    slot_bytes = source[0].nbytes
    backing = torch.full((len(blocks) * slot_bytes + 32,), 0xA5, dtype=torch.uint8)
    staging = backing[16:-16]
    index = torch.tensor(blocks, dtype=torch.int64)
    pointer = backing.data_ptr()
    copy_cross_layer_kv_cache_to_staging(
        staging, source, blocks, 3, slot_bytes, index if prebuilt_index else None
    )
    expected = source[index].contiguous().view(torch.uint8).reshape(-1)
    assert torch.equal(staging, expected)
    assert torch.equal(source, original)
    assert backing.data_ptr() == pointer
    assert bool(torch.all(backing[:16] == 0xA5))
    assert bool(torch.all(backing[-16:] == 0xA5))


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_snapshot_uses_no_temporary_payload() -> None:
    """New fragmented shapes must reuse the lease without a full KV allocation."""
    source = (
        torch.arange(8 * 2 * 2 * 128 * 4 * 64, device="cuda")
        .to(torch.int16)
        .reshape(8, 2, 2, 128, 4, 64)
    )
    slot_bytes = source[0].nbytes
    backing = torch.empty((5 * slot_bytes,), dtype=torch.uint8, device="cuda")
    index = torch.tensor([7, 1, 5, 2, 4], device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    for count in (2, 5, 3, 5):
        blocks = [7, 1, 5, 2, 4][:count]
        staging = backing[: count * slot_bytes]
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated()
        with torch.cuda.stream(stream):
            copy_cross_layer_kv_cache_to_staging(
                staging, source, blocks, 2, slot_bytes, index[:count]
            )
        stream.synchronize()
        assert torch.cuda.max_memory_allocated() == allocated
        assert torch.equal(
            staging, source[index[:count]].contiguous().view(torch.uint8).reshape(-1)
        )
