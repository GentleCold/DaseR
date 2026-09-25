# SPDX-License-Identifier: Apache-2.0
"""Public packed placement and FIFO lifetime contracts."""

import random
from typing import Any

import pytest

from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import MetadataStore
from daser.server.packed_layout import OnlinePackedLayout

SLOT = 4 * 4096


def allocate(manager: ChunkManager, key: str, nbytes: int) -> dict[str, Any]:
    """Allocate one live ring record and return its validated wire span."""
    slot = manager.alloc(key, 1, 128, "model", 0)
    return {
        "chunk_key": key,
        "start_slot": slot,
        "num_slots": 1,
        "logical_slot_start": slot,
        "logical_slot_count": 1,
        "file_offset": slot * SLOT,
        "source_offset": 0,
        "nbytes": nbytes,
        "packed": True,
        "mode": "compressed" if nbytes < SLOT else "raw",
    }


def test_tail_aligned_records_keep_partial_retry_placements() -> None:
    """A retry never relocates an earlier record into a live sibling."""
    manager = ChunkManager(8, MetadataStore(8))
    layout = OnlinePackedLayout()
    spans = [allocate(manager, str(i), 4096 * (i + 1)) for i in range(3)]
    spans[1]["source_offset"] = 4096
    spans[2]["source_offset"] = 12288
    assigned = layout.compact(spans, manager, local_slot_size=SLOT)
    assert [span["file_offset"] for span in assigned] == [24576, 28672, 36864]
    assert [span["file_offset"] for span in spans] == [0, SLOT, SLOT * 2]
    for index in (2, 0, 1):
        retry = layout.compact([spans[index]], manager, local_slot_size=SLOT)
        assert retry[0]["file_offset"] == assigned[index]["file_offset"]
    with pytest.raises(ValueError, match="changes an assigned"):
        layout.compact([{**spans[0], "nbytes": 8192}], manager, local_slot_size=SLOT)


def test_tail_boundary_separates_different_fifo_generations() -> None:
    """Numerically adjacent slots on opposite sides of FIFO tail cannot share."""
    manager = ChunkManager(4, MetadataStore(4))
    old = [allocate(manager, f"old{i}", 4096) for i in range(4)]
    new = allocate(manager, "new", 4096)
    old[1]["source_offset"] = 4096
    assert manager.tail_slot == 1
    assigned = OnlinePackedLayout().compact(
        [new, old[1]], manager, local_slot_size=SLOT
    )
    assert assigned[0]["file_offset"] + 4096 == SLOT
    assert assigned[1]["file_offset"] + 4096 == 2 * SLOT


@pytest.mark.parametrize("rank_base", [0, SLOT * 17])
def test_random_fifo_reuse_preserves_all_live_record_bytes(rank_base: int) -> None:
    """Repeated wrap, retries and variable sizes preserve every live payload."""
    rng = random.Random(8173)
    manager = ChunkManager(17, MetadataStore(17))
    layout = OnlinePackedLayout()
    storage = bytearray(rank_base + SLOT * 17)
    live: dict[str, tuple[dict[str, Any], bytes]] = {}
    serial = 0
    for _cycle in range(500):
        spans: list[dict[str, Any]] = []
        payloads: dict[str, bytes] = {}
        cursor = 0
        for _ in range(rng.randint(1, 6)):
            serial += 1
            key = str(serial)
            nbytes = rng.randint(1, 4) * 4096
            span = allocate(manager, key, nbytes)
            span["source_offset"] = cursor
            span["file_offset"] += rank_base
            cursor += nbytes
            spans.append(span)
            payloads[key] = serial.to_bytes(4, "little") * (nbytes // 4)
        assigned = layout.compact(
            spans, manager, local_slot_size=SLOT, rank_base=rank_base
        )
        for span in assigned:
            key = span["chunk_key"]
            start = span["file_offset"]
            end = start + span["nbytes"]
            assert start >= rank_base + span["start_slot"] * SLOT
            assert end <= len(storage)
            storage[start:end] = payloads[key]
            live[key] = (span, payloads[key])
        retry_spans = rng.sample(spans, rng.randint(1, len(spans)))
        retry = layout.compact(
            retry_spans, manager, local_slot_size=SLOT, rank_base=rank_base
        )
        for span in retry:
            expected = live[span["chunk_key"]][0]
            assert span["file_offset"] == expected["file_offset"]
        for key, (span, expected) in list(live.items()):
            if manager.store.get(key) is None:
                del live[key]
                continue
            start = span["file_offset"]
            assert storage[start : start + span["nbytes"]] == expected


def test_invalid_batch_does_not_reserve_its_valid_prefix() -> None:
    """Rejected input is atomic and leaves future placement unconstrained."""
    manager = ChunkManager(4, MetadataStore(4))
    layout = OnlinePackedLayout()
    spans = [allocate(manager, str(i), 4096) for i in range(2)]
    spans[1]["source_offset"] = 4096
    with pytest.raises(ValueError, match="one aligned raw slot"):
        layout.compact(
            [spans[0], {**spans[1], "nbytes": SLOT + 4096}],
            manager,
            local_slot_size=SLOT,
        )
    assigned = layout.compact(spans, manager, local_slot_size=SLOT)
    assert [span["file_offset"] for span in assigned] == [
        2 * SLOT - 8192,
        2 * SLOT - 4096,
    ]


def test_partial_multislot_allocations_stop_at_source_gaps() -> None:
    """Separate regions of one allocation never overwrite already placed peers."""
    manager = ChunkManager(8, MetadataStore(8))
    manager.alloc("chunk", 4, 512, "model", 0)
    spans = [
        {
            "chunk_key": "chunk",
            "start_slot": 0,
            "num_slots": 4,
            "logical_slot_start": i,
            "logical_slot_count": 1,
            "file_offset": i * SLOT,
            "source_offset": i * 4096,
            "nbytes": 4096,
            "packed": True,
        }
        for i in range(4)
    ]
    layout = OnlinePackedLayout()
    first = layout.compact(spans[:2], manager, local_slot_size=SLOT)
    second = layout.compact(spans[2:], manager, local_slot_size=SLOT)
    assert first[-1]["file_offset"] + 4096 == 2 * SLOT
    assert second[0]["file_offset"] >= 2 * SLOT
    assert layout.compact(spans, manager, local_slot_size=SLOT) == first + second

    gapped = OnlinePackedLayout().compact(
        [spans[0], {**spans[1], "source_offset": SLOT}],
        manager,
        local_slot_size=SLOT,
    )
    assert [span["file_offset"] for span in gapped] == [SLOT - 4096, 2 * SLOT - 4096]
