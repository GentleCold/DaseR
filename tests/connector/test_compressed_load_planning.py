# SPDX-License-Identifier: Apache-2.0

from daser.compression.format import SlotMode
from daser.connector.metadata import CompressedLoadSlot, ReqLoadSpec, StoreWriteSpan
from daser.connector.worker.load import (
    build_load_read_batches,
    build_load_read_plan,
)
from daser.connector.worker.store import (
    _packed_store_spans,
)
from daser.ops.compressed_kv import OnlinePackedSlot


def _spec() -> ReqLoadSpec:
    return ReqLoadSpec(
        chunk_key="chunk",
        start_slot=10,
        num_slots=3,
        block_ids=[20, 21, 22],
        file_offset=10 * 16384,
        token_count=384,
        compressed_slots=[
            CompressedLoadSlot(10, "compressed", 10 * 16384, 4096),
            CompressedLoadSlot(11, "compressed", 11 * 16384, 8192),
            CompressedLoadSlot(12, "raw", 12 * 16384, 16384),
        ],
    )


def test_compressed_plan_reads_only_indexed_lengths() -> None:
    spec = _spec()

    total, spans, ranges = build_load_read_plan({"req": spec}, 16384)

    assert total == 28672
    assert spans == [
        {"target_offset": 0, "nbytes": 4096, "file_offset": 10 * 16384},
        {"target_offset": 4096, "nbytes": 8192, "file_offset": 11 * 16384},
        {"target_offset": 12288, "nbytes": 16384, "file_offset": 12 * 16384},
    ]
    assert ranges == [(0, total, spec)]


def test_compressed_batches_split_by_stored_bytes() -> None:
    batches = build_load_read_batches(
        {"req": _spec()},
        slot_size=16384,
        max_batch_bytes=16384,
        include_req_ids=True,
    )

    assert [batch[0] for batch in batches] == [12288, 16384]
    first_spec = batches[0][2][0][3]
    second_spec = batches[1][2][0][3]
    assert first_spec.block_ids == [20, 21]
    assert [slot.slot_id for slot in first_spec.compressed_slots] == [10, 11]
    assert second_spec.block_ids == [22]
    assert second_spec.compressed_slots[0].mode == "raw"


def test_duplicate_compressed_source_is_read_once() -> None:
    first = _spec()
    second = ReqLoadSpec(
        **{
            **first.__dict__,
            "block_ids": [30, 31, 32],
        }
    )

    total, spans, ranges = build_load_read_plan(
        {"first": first, "second": second}, 16384
    )

    assert total == 28672
    assert len(spans) == 3
    assert ranges[0][:2] == ranges[1][:2] == (0, total)


def test_duplicate_compressed_sources_share_one_capacity_bounded_batch() -> None:
    """Destination fan-out does not charge immutable source bytes repeatedly."""
    first = _spec()
    requests = {
        f"req-{index}": ReqLoadSpec(
            **{
                **first.__dict__,
                "block_ids": [index * 3 + offset for offset in range(3)],
            }
        )
        for index in range(4)
    }

    batches = build_load_read_batches(
        requests,
        slot_size=16384,
        max_batch_bytes=28672,
        include_req_ids=True,
    )

    assert len(batches) == 1
    total, spans, ranges = batches[0]
    assert total == 28672
    assert len(spans) == 3
    assert len(ranges) == 4
    assert {item[2] for item in ranges} == set(requests)
    assert {item[0:2] for item in ranges} == {(0, total)}


def test_packed_store_spans_compact_each_allocation() -> None:
    """Adjacent records in one allocation use one compact file range."""
    raw_slot = 16_384
    source = StoreWriteSpan(
        source_offset=0,
        nbytes=3 * raw_slot,
        file_offset=10 * raw_slot,
        chunk_key="chunk",
        start_slot=10,
        num_slots=3,
        logical_slot_start=10,
        logical_slot_count=3,
    )
    packed = [
        OnlinePackedSlot(10, SlotMode.COMPRESSED, 0, 4_096),
        OnlinePackedSlot(11, SlotMode.COMPRESSED, 4_096, 8_192),
        OnlinePackedSlot(12, SlotMode.RAW, 12_288, raw_slot),
    ]

    result = _packed_store_spans([source], packed, raw_slot)

    assert [span.source_offset for span in result] == [0, 4_096, 12_288]
    assert [span.file_offset for span in result] == [
        10 * raw_slot,
        10 * raw_slot + 4_096,
        10 * raw_slot + 12_288,
    ]
    assert [span.nbytes for span in result] == [4_096, 8_192, raw_slot]


def test_packed_store_spans_fall_back_for_over_capacity_record() -> None:
    """Invalid record lengths retain the raw-stride file layout."""
    raw_slot = 16_384
    source = StoreWriteSpan(
        source_offset=0,
        nbytes=2 * raw_slot,
        file_offset=20 * raw_slot,
        chunk_key="chunk",
        start_slot=20,
        num_slots=2,
        logical_slot_start=20,
        logical_slot_count=2,
    )
    packed = [
        OnlinePackedSlot(20, SlotMode.COMPRESSED, 0, raw_slot + 4_096),
        OnlinePackedSlot(21, SlotMode.COMPRESSED, raw_slot, 4_096),
    ]

    result = _packed_store_spans([source], packed, raw_slot)

    assert [span.file_offset for span in result] == [20 * raw_slot, 21 * raw_slot]


def test_packed_store_spans_keep_allocation_boundaries() -> None:
    """Records from different allocations are never compacted across chunks."""
    raw_slot = 16_384
    spans = [
        StoreWriteSpan(0, raw_slot, 30 * raw_slot, "a", 30, 1, 30, 1),
        StoreWriteSpan(raw_slot, raw_slot, 31 * raw_slot, "b", 31, 1, 31, 1),
    ]
    packed = [
        OnlinePackedSlot(30, SlotMode.COMPRESSED, 0, 4_096),
        OnlinePackedSlot(31, SlotMode.COMPRESSED, raw_slot, 4_096),
    ]

    result = _packed_store_spans(spans, packed, raw_slot)

    assert [span.file_offset for span in result] == [30 * raw_slot, 31 * raw_slot]
