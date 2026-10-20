# SPDX-License-Identifier: Apache-2.0

# Standard
import dataclasses
import os
from pathlib import Path

# Third Party
import msgpack
import pytest

# First Party
from daser.server.chunk_manager import ChunkManager, IndexSnapshot
from daser.server.doc_registry import DocEntry, DocRegistry
from daser.server.metadata_store import MetadataStore
from daser.server.tiers import L2Placement


def make_manager(registry: DocRegistry | None = None) -> ChunkManager:
    return ChunkManager(MetadataStore(), registry)


def alloc(mgr: ChunkManager, key: str, num_slots: int, parent_key: str = "") -> int:
    meta = mgr.alloc(
        key,
        num_slots=num_slots,
        token_count=16 * num_slots,
        model_id="m",
        pos_offset=0,
        parent_key=parent_key,
    )
    return meta.start_slot


def snapshot_of(mgr: ChunkManager, mode: str | None = None) -> IndexSnapshot:
    """Return a snapshot of every chunk, each with a one-grain L2 copy."""
    metas = list(mgr.store.iter_chunks())
    registry = mgr.doc_registry
    return IndexSnapshot(
        cache_reuse_mode=mode,
        next_slot=mgr.next_slot,
        tier_grain=4096,
        l2_lane_bytes=64 * 4096,
        tensor_parallel_size=1,
        metas=metas,
        l2=[
            L2Placement(meta.chunk_key, index * 4096, (4096,) * meta.num_slots)
            for index, meta in enumerate(metas)
        ],
        docs=registry.to_dict() if registry is not None else {},
    )


def test_alloc_assigns_consecutive_ids_that_are_never_reused() -> None:
    mgr = make_manager()
    assert alloc(mgr, "a", 3) == 0
    assert alloc(mgr, "b", 2, parent_key="a") == 3
    assert mgr.store.get("b").parent_key == "a"
    mgr.remove("a")
    assert alloc(mgr, "a", 1) == 5
    assert mgr.next_slot == 6


def test_alloc_rejects_invalid_requests() -> None:
    mgr = make_manager()
    with pytest.raises(ValueError):
        alloc(mgr, "zero", 0)
    alloc(mgr, "a", 1)
    with pytest.raises(ValueError):
        alloc(mgr, "a", 1)
    with pytest.raises(KeyError):
        mgr.remove("missing")


def test_resolve_slot_maps_ids_to_live_chunks() -> None:
    mgr = make_manager()
    alloc(mgr, "a", 2)
    alloc(mgr, "b", 3)
    meta, index = mgr.resolve_slot(3)
    assert (meta.chunk_key, index) == ("b", 1)
    mgr.remove("b")
    assert mgr.resolve_slot(3) is None
    assert mgr.resolve_slot(99) is None
    assert mgr.resolve_slot(1)[0].chunk_key == "a"


def test_existing_store_entries_seed_ids() -> None:
    first = make_manager()
    alloc(first, "a", 4)
    second = ChunkManager(first.store)
    assert second.next_slot == 4
    assert second.resolve_slot(2)[0].chunk_key == "a"


def test_snapshot_round_trip(tmp_path: Path) -> None:
    mgr = make_manager()
    alloc(mgr, "a", 3)
    alloc(mgr, "b", 2, parent_key="a")
    path = str(tmp_path / "daser.index")
    mgr.save(path, snapshot_of(mgr, "prefix"))

    snapshot = ChunkManager.read_snapshot(path, expected_cache_reuse_mode="prefix")
    assert snapshot == snapshot_of(mgr, "prefix")
    restored = make_manager()
    restored.restore(snapshot)
    assert restored.store.get("b").parent_key == "a"
    assert restored.resolve_slot(4)[0].chunk_key == "b"
    assert alloc(restored, "c", 1) == 5


def test_read_snapshot_rejects_other_formats_and_modes(tmp_path: Path) -> None:
    mgr = make_manager()
    alloc(mgr, "a", 1)
    path = str(tmp_path / "daser.index")
    mgr.save(path, snapshot_of(mgr, "chunk"))
    with pytest.raises(ValueError, match="cache reuse mode"):
        ChunkManager.read_snapshot(path, expected_cache_reuse_mode="prefix")

    mgr.save(path, snapshot_of(mgr, None))
    ChunkManager.read_snapshot(path)
    with pytest.raises(ValueError, match="cache reuse mode"):
        ChunkManager.read_snapshot(path, expected_cache_reuse_mode="chunk")

    old = tmp_path / "old.index"
    old.write_bytes(msgpack.packb({"index": b"", "total_slots": 8}))
    with pytest.raises(ValueError, match="format"):
        ChunkManager.read_snapshot(str(old))


def test_validate_snapshot_rejects_bad_ranges() -> None:
    mgr = make_manager()
    alloc(mgr, "a", 3)
    alloc(mgr, "b", 3)
    good = snapshot_of(mgr)
    a, b = good.metas
    cases = [
        dataclasses.replace(good, metas=[a, dataclasses.replace(b, start_slot=1)]),
        dataclasses.replace(good, metas=[a, dataclasses.replace(b, chunk_key="a")]),
        dataclasses.replace(good, next_slot=5),
        dataclasses.replace(good, metas=[dataclasses.replace(a, start_slot=-1)]),
        dataclasses.replace(good, metas=[dataclasses.replace(a, num_slots=0)]),
        dataclasses.replace(good, next_slot=-1),
        dataclasses.replace(good, next_slot=6.0),
        dataclasses.replace(good, metas=[dataclasses.replace(a, start_slot=0.0)]),
        dataclasses.replace(good, metas=[dataclasses.replace(a, num_slots=3.0)]),
        dataclasses.replace(good, metas=[dataclasses.replace(a, num_slots=True)]),
    ]
    for snapshot in cases:
        with pytest.raises(ValueError):
            make_manager().validate_snapshot(snapshot)
    with pytest.raises(ValueError, match="empty"):
        mgr.validate_snapshot(good)
    make_manager().validate_snapshot(good)


@pytest.mark.parametrize("next_slot", [-1, 1.5, True])
def test_decoded_snapshot_rejects_invalid_identity_without_publication(
    tmp_path: Path, next_slot: object
) -> None:
    """Decoding cannot coerce malformed slot identities into an accepted index."""
    source = make_manager()
    alloc(source, "a", 1)
    path = tmp_path / "bad-identity.index"
    source.save(str(path), snapshot_of(source))
    payload = msgpack.unpackb(path.read_bytes(), raw=False)
    payload["next_slot"] = next_slot
    path.write_bytes(msgpack.packb(payload, use_bin_type=True))
    target = make_manager()
    with pytest.raises(ValueError, match="slot identity"):
        target.restore(ChunkManager.read_snapshot(str(path)))
    assert target.next_slot == 0 and len(target.store) == 0


def test_restore_rejection_leaves_the_manager_empty() -> None:
    mgr = make_manager()
    alloc(mgr, "a", 2)
    bad = dataclasses.replace(snapshot_of(mgr), next_slot=1)
    target = make_manager()
    with pytest.raises(ValueError):
        target.restore(bad)
    assert target.next_slot == 0 and len(target.store) == 0


@pytest.mark.parametrize("existing_document", [False, True])
def test_invalid_registry_cannot_partially_restore_chunks(
    existing_document: bool,
) -> None:
    """Registry validation precedes publication and preserves existing documents."""
    source = make_manager()
    alloc(source, "restored", 1)
    bad = dataclasses.replace(
        snapshot_of(source), docs={"broken": {"unknown_field": True}}
    )
    registry = DocRegistry()
    if existing_document:
        registry.insert(DocEntry(doc_id="existing", title="keep"))
    before = registry.to_dict()
    target = make_manager(registry)
    with pytest.raises(ValueError, match="document registry"):
        target.restore(bad)
    assert target.next_slot == 0
    assert len(target.store) == 0
    assert target.resolve_slot(0) is None
    assert registry.to_dict() == before
    # A rejected snapshot must not poison a later valid restore.
    target.restore(snapshot_of(source))
    assert target.store.get("restored") is not None


def test_restore_marks_documents_of_unsaved_chunks_evicted() -> None:
    registry = DocRegistry()
    mgr = make_manager(registry)
    alloc(mgr, "kept", 1)
    alloc(mgr, "lost", 1)
    registry.insert(
        DocEntry(
            doc_id="doc1",
            title="Doc 1",
            token_count=32,
            chunk_keys=["kept", "lost"],
            cached_mask=[True, True],
            tokens=[1, 2, 3],
        )
    )
    mgr.store.get("kept").doc_ids.append("doc1")
    snapshot = snapshot_of(mgr)
    snapshot = dataclasses.replace(
        snapshot,
        metas=[m for m in snapshot.metas if m.chunk_key == "kept"],
        l2=[p for p in snapshot.l2 if p.chunk_key == "kept"],
    )

    registry2 = DocRegistry()
    restored = make_manager(registry2)
    restored.restore(snapshot)
    doc = registry2.get("doc1")
    assert doc is not None
    assert doc.title == "Doc 1" and doc.tokens == [1, 2, 3]
    assert doc.cached_mask == [True, False]
    assert restored.store.get("kept").doc_ids == ["doc1"]
    assert restored.next_slot == 2


def test_save_failure_keeps_existing_index_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = str(tmp_path / "daser.index")
    mgr = make_manager()
    alloc(mgr, "old", 2)
    mgr.save(path, snapshot_of(mgr))

    alloc(mgr, "new", 2)
    original_replace = os.replace

    def fail_replace(src: str, dst: str) -> None:
        if dst == path:
            raise OSError("replace failed")
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        mgr.save(path, snapshot_of(mgr))
    assert not os.path.exists(path + ".tmp")

    keys = [meta.chunk_key for meta in ChunkManager.read_snapshot(path).metas]
    assert keys == ["old"]
