# SPDX-License-Identifier: Apache-2.0

from daser.connector.metadata import CompressedLoadSlot, ReqLoadSpec
from daser.connector.worker.load import (
    build_load_read_batches,
    build_load_read_plan,
)


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
