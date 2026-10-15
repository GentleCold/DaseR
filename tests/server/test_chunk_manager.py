# SPDX-License-Identifier: Apache-2.0

# Standard
import os
from pathlib import Path

# Third Party
import msgpack
import pytest

# First Party
from daser.server.chunk_manager import ChunkManager
from daser.server.doc_registry import DocEntry, DocRegistry
from daser.server.metadata_store import MetadataStore


def make_manager(total_slots: int = 8) -> ChunkManager:
    store = MetadataStore(total_slots=total_slots)
    return ChunkManager(total_slots=total_slots, metadata_store=store)


def test_alloc_returns_correct_start_slot():
    mgr = make_manager(8)
    slot = mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert slot == 0


def test_alloc_advances_head():
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    slot = mgr.alloc("key2", num_slots=2, token_count=32, model_id="m", pos_offset=48)
    assert slot == 3


def test_free_slots_decreases():
    mgr = make_manager(8)
    assert mgr.free_slots == 8
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert mgr.free_slots == 5


def test_evicts_oldest_when_full():
    mgr = make_manager(6)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    mgr.alloc("key3", num_slots=3, token_count=48, model_id="m", pos_offset=96)
    assert mgr.store.get("key1") is None
    assert mgr.store.get("key2") is not None
    assert mgr.store.get("key3") is not None


def test_auto_eviction_keys_can_be_drained():
    mgr = make_manager(6)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    assert mgr.drain_evicted_chunk_keys() == []

    mgr.alloc("key3", num_slots=3, token_count=48, model_id="m", pos_offset=96)

    assert mgr.drain_evicted_chunk_keys() == ["key1"]
    assert mgr.drain_evicted_chunk_keys() == []


def test_evict_oldest_restores_capacity():
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert mgr.free_slots == 5
    mgr.evict_oldest()
    assert mgr.free_slots == 8
    assert mgr.store.get("key1") is None


def test_tail_skips_chunk_removed_before_eviction():
    mgr = make_manager(6)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    mgr.store.remove("key2")

    mgr.alloc("key3", num_slots=3, token_count=48, model_id="m", pos_offset=96)
    mgr.alloc("key4", num_slots=3, token_count=48, model_id="m", pos_offset=144)

    assert mgr.store.get("key3") is not None
    assert mgr.store.get("key4") is not None
    assert mgr.drain_evicted_chunk_keys() == ["key1"]


def test_tail_keeps_reallocated_chunk_with_removed_key():
    mgr = make_manager(9)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    mgr.store.remove("key1")
    assert (
        mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0) == 6
    )

    # The tail passes key1's old slots; the re-allocated key1 must survive.
    mgr.alloc("key3", num_slots=3, token_count=48, model_id="m", pos_offset=96)

    assert mgr.store.get("key1") is not None
    assert mgr.store.get("key2") is not None
    assert mgr.drain_evicted_chunk_keys() == []


def test_wrap_around():
    # total=8; key1(3) at [0..2], key2(3) at [3..5] → head=6
    # key3(4) needs 4 slots; only 2 remain at tail → SKIP [6,7], head wraps to 0
    # must evict key1 then key2 to make room; key3 lands at slot 0
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    slot = mgr.alloc("key3", num_slots=4, token_count=64, model_id="m", pos_offset=96)
    assert slot == 0
    assert mgr.store.get("key3") is not None


def test_wrap_skip_advances_tail():
    # After wrap: key3 at [0..3], SKIP at [6,7], tail at 6
    # alloc key4 (2 slots): tail must consume SKIP [6,7] → tail wraps to 0
    # then key4 sits at slot 4
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    mgr.alloc("key3", num_slots=4, token_count=64, model_id="m", pos_offset=96)
    mgr.alloc("key4", num_slots=2, token_count=32, model_id="m", pos_offset=160)
    assert mgr.store.get("key4") is not None
    assert mgr.store.get("key4").start_slot == 4


def test_save_and_load_state(tmp_path: Path) -> None:
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=2, token_count=32, model_id="m", pos_offset=48)
    path = str(tmp_path / "daser.index")
    mgr.save(path, committed_keys={"key1", "key2"})

    store2 = MetadataStore(total_slots=8)
    mgr2 = ChunkManager(total_slots=8, metadata_store=store2)
    mgr2.load(path)
    assert mgr2.store.get("key1") is not None
    assert mgr2.store.get("key2") is not None
    assert mgr2.free_slots == 3
    slot = mgr2.alloc("key3", num_slots=2, token_count=32, model_id="m", pos_offset=80)
    assert slot == 5


def test_load_rejects_different_cache_reuse_mode(tmp_path: Path) -> None:
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    path = str(tmp_path / "daser.index")
    mgr.save(path, cache_reuse_mode="chunk", committed_keys={"key1"})

    restored = make_manager(8)
    with pytest.raises(ValueError, match="cache reuse mode"):
        restored.load(path, expected_cache_reuse_mode="prefix")


def test_load_rejects_missing_cache_reuse_mode_when_expected(tmp_path: Path) -> None:
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    path = str(tmp_path / "daser.index")
    mgr.save(path, committed_keys={"key1"})

    restored = make_manager(8)
    with pytest.raises(ValueError, match="cache reuse mode"):
        restored.load(path, expected_cache_reuse_mode="chunk")


def test_save_and_load_state_includes_doc_registry(tmp_path: Path) -> None:
    registry = DocRegistry()
    store = MetadataStore(total_slots=8)
    mgr = ChunkManager(total_slots=8, metadata_store=store, doc_registry=registry)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    registry.insert(
        DocEntry(
            doc_id="doc1",
            title="Doc 1",
            token_count=48,
            chunk_keys=["key1"],
            cached_mask=[True],
            tokens=[1, 2, 3],
        )
    )
    meta = mgr.store.get("key1")
    assert meta is not None
    meta.doc_ids.append("doc1")
    path = str(tmp_path / "daser.index")

    mgr.save(path, committed_keys={"key1"})

    registry2 = DocRegistry()
    store2 = MetadataStore(total_slots=8)
    mgr2 = ChunkManager(
        total_slots=8,
        metadata_store=store2,
        doc_registry=registry2,
    )
    mgr2.load(path)

    restored_doc = registry2.get("doc1")
    restored_meta = mgr2.store.get("key1")
    assert restored_doc is not None
    assert restored_doc.title == "Doc 1"
    assert restored_doc.tokens == [1, 2, 3]
    assert restored_meta is not None
    assert restored_meta.doc_ids == ["doc1"]


def test_save_failure_keeps_existing_index_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = str(tmp_path / "daser.index")
    mgr = make_manager(8)
    mgr.alloc("old", num_slots=2, token_count=32, model_id="m", pos_offset=0)
    mgr.save(path, committed_keys={"old"})

    mgr.alloc("new", num_slots=2, token_count=32, model_id="m", pos_offset=32)
    original_replace = os.replace

    def fail_replace(src: str, dst: str) -> None:
        if dst == path:
            raise OSError("replace failed")
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        mgr.save(path, committed_keys={"old", "new"})

    restored = make_manager(8)
    restored.load(path)
    assert restored.store.get("old") is not None
    assert restored.store.get("new") is None


def test_snapshot_recovery_reclaims_pending_holes_after_wrap(tmp_path: Path) -> None:
    """Retain committed offsets while reclaiming pending spans in FIFO order.

    Args:
        tmp_path: Path for the serialized wrapped ring.

    Returns:
        None. Assertions verify public allocation and eviction behavior.

    Async/thread-safety:
        Uses a single synchronous caller without IO or CUDA workers.
    """
    original = make_manager(12)
    original.alloc("old", 3, 48, "m", 0)
    original.alloc("keep", 3, 48, "m", 48)
    original.alloc("pending", 2, 32, "m", 96)
    original.alloc("keep-tail", 3, 48, "m", 128)
    # One padding slot remains at the end. Wrapping evicts old and places
    # the second pending allocation before both committed survivors.
    assert original.alloc("pending-wrap", 2, 32, "m", 176) == 0
    path = str(tmp_path / "daser.index")
    original.save(path, committed_keys={"keep", "keep-tail"})

    restored = make_manager(12)
    restored.load(path)
    assert restored.store.get("pending") is None
    assert restored.store.get("pending-wrap") is None
    for key, start in (("keep", 3), ("keep-tail", 8)):
        meta = restored.store.get(key)
        assert meta is not None
        assert meta.start_slot == start
    assert restored.store.get_slot_entry(6).kind == "skip"
    assert restored.store.get_slot_entry(0).kind == "skip"

    assert restored.alloc("fill", 1, 16, "m", 208) == 2
    assert restored.alloc("replacement", 3, 48, "m", 224) == 3
    assert restored.drain_evicted_chunk_keys() == ["keep"]
    # The middle pending span is reclaimed without evicting keep-tail.
    assert restored.alloc("reuse-hole", 2, 32, "m", 272) == 6
    assert restored.store.get("keep-tail") is not None
    assert restored.drain_evicted_chunk_keys() == []
    restored.evict_oldest()
    assert restored.drain_evicted_chunk_keys() == ["keep-tail"]
    restored.evict_oldest()  # Original one-slot wrap padding.
    restored.evict_oldest()  # Recovered pending-wrap allocation.
    assert restored.tail_slot == 2
    assert restored.store.get("fill") is not None
    assert restored.drain_evicted_chunk_keys() == []


def test_snapshot_with_only_pending_chunks_restores_empty_ring(tmp_path: Path) -> None:
    """Recover full capacity when no allocation reached publication.

    Args:
        tmp_path: Path for the all-pending snapshot.

    Returns:
        None. A whole-ring allocation verifies both capacity and reset position.

    Async/thread-safety:
        Runs synchronous manager calls on one test thread.
    """
    original = make_manager(8)
    original.alloc("pending-a", 2, 32, "m", 0)
    original.alloc("pending-b", 3, 48, "m", 32)
    path = str(tmp_path / "daser.index")
    original.save(path, committed_keys=set())

    restored = make_manager(8)
    restored.load(path)
    assert list(restored.store.iter_chunks()) == []
    assert restored.free_slots == 8
    assert restored.tail_slot == 0
    assert restored.alloc("whole-ring", 8, 128, "m", 0) == 0


@pytest.mark.parametrize("keep_committed", [False, True])
def test_snapshot_recovery_updates_document_cache_state(
    tmp_path: Path, keep_committed: bool
) -> None:
    """Keep registered document content consistent with recovered cache entries.

    Args:
        tmp_path: Path for the snapshot containing a registered document.
        keep_committed: bool selecting a partial or fully missing cache.

    Returns:
        None. Assertions check document content, cache masks and back references.

    Async/thread-safety:
        Registry and allocator are used sequentially on the test thread.
    """
    registry = DocRegistry()
    original = ChunkManager(8, MetadataStore(8), doc_registry=registry)
    for index, key in enumerate(("keep", "pending")):
        original.alloc(key, 2, 32, "m", index * 32)
        meta = original.store.get(key)
        assert meta is not None
        meta.doc_ids.append("doc")
    registry.insert(
        DocEntry(
            doc_id="doc",
            title="Registered document",
            token_count=64,
            chunk_keys=["keep", "pending"],
            cached_mask=[True, True],
            tokens=[1, 2, 3, 4],
            text="Saved document content",
        )
    )
    path = str(tmp_path / "daser.index")
    original.save(path, committed_keys={"keep"} if keep_committed else set())
    # Filtering a snapshot must not change the running document view.
    source_doc = registry.get("doc")
    assert source_doc is not None
    assert source_doc.cached_mask == [True, True]

    restored_registry = DocRegistry()
    restored = ChunkManager(8, MetadataStore(8), doc_registry=restored_registry)
    restored.load(path)
    doc = restored_registry.get("doc")
    assert doc is not None
    assert doc.title == "Registered document"
    assert doc.tokens == [1, 2, 3, 4]
    assert doc.text == "Saved document content"
    assert doc.chunk_keys == ["keep", "pending"]
    assert doc.cached_mask == [keep_committed, False]
    assert restored.store.get("pending") is None
    if keep_committed:
        meta = restored.store.get("keep")
        assert meta is not None
        assert meta.doc_ids == ["doc"]
        restored.evict_oldest()
        assert doc.cached_mask == [False, False]
    assert doc.status == "evicted"


@pytest.mark.parametrize(
    ("field", "value", "remove_field"),
    [
        pytest.param("snapshot_version", None, True, id="legacy-version-missing"),
        pytest.param("committed_keys", None, True, id="manifest-missing"),
        pytest.param("snapshot_version", 2, False, id="unsupported-version"),
        pytest.param("snapshot_version", True, False, id="boolean-version"),
        pytest.param("committed_keys", None, False, id="null-manifest"),
        pytest.param("committed_keys", "source", False, id="string-manifest"),
        pytest.param("committed_keys", [1], False, id="non-string-key"),
    ],
)
def test_invalid_snapshot_leaves_current_state_unchanged(
    tmp_path: Path, field: str, value: object, remove_field: bool
) -> None:
    """Reject unverifiable snapshots before changing ring or document state.

    Args:
        tmp_path: Path for the deliberately malformed snapshot.
        field: str naming the persisted field to corrupt.
        value: object replacing the field when it is not removed.
        remove_field: bool selecting omission instead of replacement.

    Returns:
        None. Assertions cover failure atomicity through public APIs.

    Async/thread-safety:
        Serializes and restores on one test thread with no active writers.
    """
    source = make_manager(8)
    source.alloc("source", 5, 80, "m", 0)
    path = tmp_path / "daser.index"
    source.save(str(path), committed_keys={"source"})
    payload = msgpack.unpackb(path.read_bytes(), raw=False)
    if remove_field:
        del payload[field]
    else:
        payload[field] = value
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))

    registry = DocRegistry()
    current = ChunkManager(8, MetadataStore(8), doc_registry=registry)
    current.alloc("existing", 2, 32, "m", 0)
    registry.insert(
        DocEntry(doc_id="existing-doc", title="Unchanged", chunk_keys=["existing"])
    )
    chunks_before = list(current.store.iter_chunks())
    docs_before = registry.to_dict()
    with pytest.raises(ValueError):
        current.load(str(path))

    assert list(current.store.iter_chunks()) == chunks_before
    assert registry.to_dict() == docs_before
    assert current.tail_slot == 0
    assert current.free_slots == 6
    assert current.alloc("next", 1, 16, "m", 32) == 2
