# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.server.doc_registry import DocEntry, DocRegistry


def test_doc_registry_insert_and_cached_mask() -> None:
    registry = DocRegistry()
    entry = DocEntry(
        doc_id="doc1",
        title="t",
        chunk_keys=["k1", "k2", "k3"],
    )

    registry.insert(entry)
    assert registry.get("doc1").cached_mask == [True, True, True]

    registry.mark_chunk_evicted("doc1", "k2")
    assert registry.get("doc1").cached_mask == [True, False, True]
    assert registry.get("doc1").status == "ready"

    registry.mark_chunk_evicted("doc1", "k1")
    registry.mark_chunk_evicted("doc1", "k3")
    assert registry.get("doc1").status == "evicted"


def test_doc_registry_serialization_roundtrip() -> None:
    registry = DocRegistry()
    registry.insert(
        DocEntry(
            doc_id="doc1",
            title="t",
            token_count=4,
            chunk_keys=["k1"],
            cached_mask=[False],
            status="evicted",
            tokens=[1, 2, 3, 4],
        )
    )

    restored = DocRegistry()
    restored.load_dict(registry.to_dict())

    entry = restored.get("doc1")
    assert entry is not None
    assert entry.title == "t"
    assert entry.cached_mask == [False]
    assert entry.tokens == [1, 2, 3, 4]
