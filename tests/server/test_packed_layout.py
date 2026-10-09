# SPDX-License-Identifier: Apache-2.0
"""Public packed placement and FIFO lifetime contracts."""

import random
from typing import Any

import pytest

from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import MetadataStore
from daser.server.packed_layout import OnlinePackedLayout, PackedArenaFull

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


def test_packed_records_bridge_allocations_and_keep_retry_placements() -> None:
    """A retry never relocates an earlier record into a live sibling."""
    manager = ChunkManager(8, MetadataStore(8))
    layout = OnlinePackedLayout()
    spans = [allocate(manager, str(i), 4096 * (i + 1)) for i in range(3)]
    spans[1]["source_offset"] = 4096
    spans[2]["source_offset"] = 12288
    assigned = layout.compact(spans, manager, local_slot_size=SLOT)
    assert [span["file_offset"] for span in assigned] == [0, 4096, 12288]
    assert [span["file_offset"] for span in spans] == [0, SLOT, SLOT * 2]
    for index in (2, 0, 1):
        retry = layout.compact([spans[index]], manager, local_slot_size=SLOT)
        assert retry[0]["file_offset"] == assigned[index]["file_offset"]
    with pytest.raises(ValueError, match="changes an assigned"):
        layout.compact([{**spans[0], "nbytes": 8192}], manager, local_slot_size=SLOT)


def test_packed_arena_separates_live_fifo_generations_without_raw_tail_gaps() -> None:
    """Different FIFO generations use distinct extents in the shared arena."""
    manager = ChunkManager(4, MetadataStore(4))
    old = [allocate(manager, f"old{i}", 4096) for i in range(4)]
    new = allocate(manager, "new", 4096)
    old[1]["source_offset"] = 4096
    assert manager.tail_slot == 1
    assigned = OnlinePackedLayout().compact(
        [new, old[1]], manager, local_slot_size=SLOT
    )
    assert [span["file_offset"] for span in assigned] == [0, 4096]


def test_packed_arena_reclaims_evicted_owner_extent() -> None:
    """A new generation can reuse bytes after its ring owner is evicted."""
    manager = ChunkManager(1, MetadataStore(1))
    layout = OnlinePackedLayout()
    old = allocate(manager, "old", 3 * 4096)
    assert layout.compact([old], manager, local_slot_size=SLOT)[0]["file_offset"] == 0

    new = allocate(manager, "new", 2 * 4096)
    assigned = layout.compact([new], manager, local_slot_size=SLOT)

    assert assigned[0]["file_offset"] == 0


def test_fragmented_arena_splits_runs_then_reports_deficit() -> None:
    """Runs fall back to per-record holes; a full arena raises a typed deficit."""
    manager = ChunkManager(8, MetadataStore(8))
    layout = OnlinePackedLayout()
    arena = 4 * 4096
    spans = {key: allocate(manager, key, 4096) for key in "abcd"}
    for key in "cadb":
        layout.compact([spans[key]], manager, local_slot_size=SLOT, arena_size=arena)
    assert layout.free_bytes() == 0
    manager.evict_oldest()
    manager.evict_oldest()
    for key in manager.drain_evicted_chunk_keys():
        layout.release(key, manager)  # frees a (page 1) and b (page 3)
    assert layout.free_bytes() == 2 * 4096

    run = [allocate(manager, key, 4096) for key in "ef"]
    run[1]["source_offset"] = 4096
    assigned = layout.compact(run, manager, local_slot_size=SLOT, arena_size=arena)
    assert [span["file_offset"] for span in assigned] == [4096, 3 * 4096]

    extra = allocate(manager, "g", 2 * 4096)
    with pytest.raises(PackedArenaFull, match="packed arena exhausted") as exc:
        layout.compact([extra], manager, local_slot_size=SLOT, arena_size=arena)
    assert exc.value.deficit_bytes == 2 * 4096
    # The failed call reserved nothing; a retried run keeps its placements.
    retry = layout.compact(run, manager, local_slot_size=SLOT, arena_size=arena)
    assert retry == assigned


def test_release_frees_only_stale_generations_and_failures_reserve_nothing() -> None:
    """Release keeps current owners; a failed batch leaves free bytes intact."""
    manager = ChunkManager(8, MetadataStore(8))
    layout = OnlinePackedLayout()
    arena = 4 * 4096
    spans = [allocate(manager, key, 4096) for key in "ab"]
    spans[1]["source_offset"] = 4096
    layout.compact(spans, manager, local_slot_size=SLOT, arena_size=arena)
    layout.release("a", manager)  # still current: nothing is freed
    assert layout.free_bytes() == 2 * 4096

    run = [allocate(manager, key, 4096) for key in "cd"]
    run[1]["source_offset"] = 4096
    big = {**allocate(manager, "e", 2 * 4096), "source_offset": 3 * 4096}
    with pytest.raises(PackedArenaFull):
        layout.compact([*run, big], manager, local_slot_size=SLOT, arena_size=arena)
    assert layout.free_bytes() == 2 * 4096

    manager.evict_oldest()
    for key in manager.drain_evicted_chunk_keys():
        layout.release(key, manager)
    assert layout.free_bytes() == 3 * 4096
    assigned = layout.compact(run, manager, local_slot_size=SLOT, arena_size=arena)
    # The freed page 0 is too small for the run; the rolled-back tail is used.
    assert [span["file_offset"] for span in assigned] == [2 * 4096, 3 * 4096]


def test_packed_arena_bridges_separate_allocation_calls() -> None:
    """Independent stores can consume adjacent packed bytes without a tail."""
    manager = ChunkManager(4, MetadataStore(4))
    layout = OnlinePackedLayout()
    first = allocate(manager, "first", 3 * 4096)
    second = allocate(manager, "second", 2 * 4096)

    first_assigned = layout.compact([first], manager, local_slot_size=SLOT)
    second_assigned = layout.compact([second], manager, local_slot_size=SLOT)

    assert second_assigned[0]["file_offset"] == (
        first_assigned[0]["file_offset"] + first_assigned[0]["nbytes"]
    )


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
            assert start >= rank_base
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
        0,
        4096,
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
    assert first[-1]["file_offset"] + 4096 == 2 * 4096
    assert second[0]["file_offset"] == 2 * 4096
    assert layout.compact(spans, manager, local_slot_size=SLOT) == first + second

    gapped = OnlinePackedLayout().compact(
        [spans[0], {**spans[1], "source_offset": SLOT}],
        manager,
        local_slot_size=SLOT,
    )
    assert [span["file_offset"] for span in gapped] == [0, 4096]


def test_write_hold_keeps_evicted_bytes_until_the_write_ends() -> None:
    """A late write's bytes are not handed to a new record while it is held."""
    manager = ChunkManager(1, MetadataStore(1))
    layout = OnlinePackedLayout()
    old = allocate(manager, "old", 3 * 4096)
    placed = layout.compact([old], manager, local_slot_size=SLOT)
    hold = layout.begin_writes(placed)

    new = allocate(manager, "new", 2 * 4096)
    layout.release("old", manager)
    with pytest.raises(PackedArenaFull):
        layout.compact([new], manager, local_slot_size=SLOT)
    assert layout.free_bytes() == SLOT - 3 * 4096

    layout.end_writes(hold)
    assert layout.free_bytes() == SLOT
    assert layout.compact([new], manager, local_slot_size=SLOT)[0]["file_offset"] == 0


def test_write_hold_of_a_live_record_frees_nothing() -> None:
    """Ending a hold on a record that is still live keeps its placement."""
    manager = ChunkManager(2, MetadataStore(2))
    layout = OnlinePackedLayout()
    record = allocate(manager, "live", 4096)
    placed = layout.compact([record], manager, local_slot_size=SLOT)
    layout.end_writes(layout.begin_writes(placed))
    assert layout.free_bytes() == 2 * SLOT - 4096
    retry = layout.compact([record], manager, local_slot_size=SLOT)
    assert retry[0]["file_offset"] == placed[0]["file_offset"]
    with pytest.raises(ValueError, match="current packed placement"):
        layout.begin_writes([{**placed[0], "file_offset": SLOT}])
