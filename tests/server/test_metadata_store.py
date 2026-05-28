# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import asdict
from pathlib import Path
import time

import msgpack
import pytest

from daser.server.metadata_store import ChunkMeta, MetadataStore


def make_meta(key: str, start: int, num: int, tokens: int = 16) -> ChunkMeta:
    return ChunkMeta(
        chunk_key=key,
        start_slot=start,
        num_slots=num,
        token_count=tokens,
        pos_offset=0,
        model_id="test-model",
        created_at=time.time(),
    )


def test_insert_and_get() -> None:
    store = MetadataStore(total_slots=8)
    meta = make_meta("abc", start=0, num=3)
    store.insert(meta)
    assert store.get("abc") == meta
    assert len(store) == 1


def test_slot_map_after_insert() -> None:
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=2, num=3))
    entry = store.get_slot_entry(2)
    assert entry.kind == "chunk"
    assert entry.chunk_key == "abc"
    assert entry.num_slots == 3
    assert store.get_slot_entry(3).kind == "cont"
    assert store.get_slot_entry(4).kind == "cont"


def test_remove() -> None:
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=2))
    store.remove("abc")
    assert store.get("abc") is None
    assert len(store) == 0


def test_remove_nonexistent_raises() -> None:
    store = MetadataStore(total_slots=8)
    with pytest.raises(KeyError):
        store.remove("nonexistent")


def test_insert_duplicate_raises() -> None:
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=2))
    with pytest.raises(ValueError):
        store.insert(make_meta("abc", start=2, num=2))


def test_insert_skip() -> None:
    store = MetadataStore(total_slots=8)
    store.insert_skip(start_slot=6, num_slots=2)
    entry = store.get_slot_entry(6)
    assert entry.kind == "skip"
    assert entry.num_slots == 2
    assert store.get_slot_entry(7).kind == "cont"


def test_save_and_load(tmp_path: Path) -> None:
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=3))
    store.insert(make_meta("def", start=3, num=2))
    path = str(tmp_path / "daser.index")
    store.save(path)

    store2 = MetadataStore(total_slots=8)
    store2.load(path)
    assert store2.get("abc") is not None
    assert store2.get("def") is not None
    assert len(store2) == 2
    assert store2.get_slot_entry(0).kind == "chunk"
    assert store2.get_slot_entry(1).kind == "cont"


def test_chunk_meta_defaults_initialize_access_stats() -> None:
    meta = make_meta("abc", start=0, num=2)
    assert meta.access_count == 0
    assert meta.last_access_time == meta.created_at


def test_touch_increments_access_count_and_time() -> None:
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=2))

    store.touch("abc", now=100.0)
    meta = store.get("abc")
    assert meta is not None
    assert meta.access_count == 1
    assert meta.last_access_time == 100.0

    store.touch("abc", now=101.5)
    meta_again = store.get("abc")
    assert meta_again is not None
    assert meta_again.access_count == 2
    assert meta_again.last_access_time == 101.5


def test_touch_unknown_key_is_noop() -> None:
    store = MetadataStore(total_slots=8)
    store.touch("missing")
    assert store.get("missing") is None


def test_load_backward_compatible_with_legacy_payload(tmp_path: Path) -> None:
    """Records written before access_count was introduced must still load."""
    legacy_payload = {
        "total_slots": 8,
        "chunk_index": {
            "legacy": {
                "chunk_key": "legacy",
                "start_slot": 0,
                "num_slots": 2,
                "token_count": 16,
                "pos_offset": 0,
                "model_id": "test-model",
                "created_at": 1234.0,
                "doc_ids": [],
            }
        },
        "slot_map": [
            {"kind": "chunk", "chunk_key": "legacy", "num_slots": 2},
            {"kind": "cont", "chunk_key": None, "num_slots": 0},
        ]
        + [
            {"kind": "cont", "chunk_key": None, "num_slots": 0} for _ in range(6)
        ],
    }
    path = str(tmp_path / "legacy.index")
    with open(path, "wb") as f:
        f.write(msgpack.packb(legacy_payload, use_bin_type=True))

    store = MetadataStore(total_slots=8)
    store.load(path)
    meta = store.get("legacy")
    assert meta is not None
    assert meta.access_count == 0
    assert meta.last_access_time == meta.created_at == 1234.0


def test_load_forward_compatible_with_future_fields(tmp_path: Path) -> None:
    """Records carrying unknown future fields must load without errors."""
    base = asdict(make_meta("future", start=0, num=2))
    base["self_contained_score"] = 0.91  # not yet in ChunkMeta
    base["future_extension_flag"] = True
    payload = {
        "total_slots": 8,
        "chunk_index": {"future": base},
        "slot_map": [
            {"kind": "chunk", "chunk_key": "future", "num_slots": 2},
            {"kind": "cont", "chunk_key": None, "num_slots": 0},
        ]
        + [
            {"kind": "cont", "chunk_key": None, "num_slots": 0} for _ in range(6)
        ],
    }
    path = str(tmp_path / "future.index")
    with open(path, "wb") as f:
        f.write(msgpack.packb(payload, use_bin_type=True))

    store = MetadataStore(total_slots=8)
    store.load(path)
    meta = store.get("future")
    assert meta is not None
    assert meta.chunk_key == "future"
    assert not hasattr(meta, "self_contained_score")


def test_save_then_load_preserves_access_stats(tmp_path: Path) -> None:
    store = MetadataStore(total_slots=8)
    store.insert(make_meta("abc", start=0, num=2))
    store.touch("abc", now=200.0)
    store.touch("abc", now=300.0)

    path = str(tmp_path / "daser.index")
    store.save(path)

    store2 = MetadataStore(total_slots=8)
    store2.load(path)
    meta = store2.get("abc")
    assert meta is not None
    assert meta.access_count == 2
    assert meta.last_access_time == 300.0
