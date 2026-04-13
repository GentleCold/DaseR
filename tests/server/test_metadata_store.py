# SPDX-License-Identifier: Apache-2.0

# Standard
import time
from pathlib import Path

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
