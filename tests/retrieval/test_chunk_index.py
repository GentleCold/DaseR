# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import time

# First Party
from daser.connector.helpers import hash_tokens
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.server.metadata_store import ChunkMeta


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_meta(tokens: list[int], start: int = 0) -> ChunkMeta:
    """Build ChunkMeta for a token chunk."""
    return ChunkMeta(
        chunk_key=hash_tokens(tokens),
        start_slot=start,
        num_slots=max(1, len(tokens) // 4),
        token_count=len(tokens),
        pos_offset=0,
        model_id="m",
        created_at=time.time(),
    )


def test_lookup_combined_prompt_returns_doc_chunks_with_targets() -> None:
    idx = ChunkReuseIndex(block_tokens=4)
    doc_a = [1, 2, 3, 4]
    sep = [90, 91, 92, 93]
    doc_b = [5, 6, 7, 8]
    task = [100, 101, 102, 103]

    meta_a = make_meta(doc_a, start=0)
    meta_b = make_meta(doc_b, start=1)
    _run(idx.insert(meta_a))
    _run(idx.insert(meta_b))

    result = _run(idx.lookup(doc_a + sep + doc_b + task, "m"))

    assert [match.meta.chunk_key for match in result] == [
        meta_a.chunk_key,
        meta_b.chunk_key,
    ]
    assert [match.target_token_start for match in result] == [0, 8]


def test_lookup_returns_repeated_chunk_at_each_target_position() -> None:
    idx = ChunkReuseIndex(block_tokens=4)
    doc_a = [1, 2, 3, 4]
    sep = [90, 91, 92, 93]
    doc_b = [5, 6, 7, 8]
    doc_c = [9, 10, 11, 12]
    task = [100, 101, 102, 103]

    metas = [
        make_meta(doc_a, start=0),
        make_meta(sep, start=1),
        make_meta(doc_b, start=2),
        make_meta(doc_c, start=3),
    ]
    for meta in metas:
        _run(idx.insert(meta))

    result = _run(idx.lookup(doc_a + sep + doc_b + sep + doc_c + task, "m"))

    assert [match.meta.chunk_key for match in result] == [
        metas[0].chunk_key,
        metas[1].chunk_key,
        metas[2].chunk_key,
        metas[1].chunk_key,
        metas[3].chunk_key,
    ]
    assert [match.target_token_start for match in result] == [0, 4, 8, 12, 16]


def test_lookup_skips_non_block_aligned_chunk_start() -> None:
    idx = ChunkReuseIndex(block_tokens=4)
    doc = [1, 2, 3, 4]
    meta = make_meta(doc)
    _run(idx.insert(meta))

    result = _run(idx.lookup([99, 99] + doc, "m"))

    assert result == []


def test_model_id_isolation() -> None:
    idx = ChunkReuseIndex(block_tokens=4)
    doc = [1, 2, 3, 4]
    meta = make_meta(doc)
    _run(idx.insert(meta))

    result = _run(idx.lookup(doc, "other-model"))

    assert result == []


def test_lookup_hashes_only_indexed_block_windows(monkeypatch) -> None:
    """Chunk lookup should not try every cached token length at every offset."""
    from daser.retrieval import chunk_reuse

    idx = ChunkReuseIndex(block_tokens=4)
    small = list(range(4))
    large = list(range(100, 164))
    for meta in (
        make_meta(small, start=0),
        make_meta(large, start=1),
    ):
        _run(idx.insert(meta))

    calls = 0
    real_hash_tokens = chunk_reuse.hash_tokens

    def counted_hash(tokens: list[int]) -> str:
        nonlocal calls
        calls += 1
        return real_hash_tokens(tokens)

    monkeypatch.setattr(chunk_reuse, "hash_tokens", counted_hash)

    result = _run(idx.lookup(large, "m"))

    assert [match.meta.chunk_key for match in result] == [hash_tokens(large)]
    assert calls == 1
