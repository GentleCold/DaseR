# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import time

# First Party
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.metadata_store import ChunkMeta


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_meta(tokens: list[int], start: int = 0, num: int = 1) -> ChunkMeta:
    key = _hash_tokens(tokens)
    return ChunkMeta(
        chunk_key=key,
        start_slot=start,
        num_slots=num,
        token_count=len(tokens),
        pos_offset=0,
        model_id="m",
        created_at=time.time(),
    )


def test_insert_and_exact_lookup():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = [1, 2, 3, 4]
    meta = make_meta(tokens)
    _run(idx.insert(meta))
    result = _run(idx.lookup(tokens, "m"))
    assert len(result) == 1
    assert result[0].chunk_key == meta.chunk_key


def test_lookup_longer_query_finds_prefix():
    idx = PrefixHashIndex(block_tokens=4)
    tokens4 = [1, 2, 3, 4]
    meta = make_meta(tokens4)
    _run(idx.insert(meta))
    result = _run(idx.lookup([1, 2, 3, 4, 5, 6, 7, 8], "m"))
    assert len(result) == 1
    assert result[0].chunk_key == meta.chunk_key


def test_lookup_miss_returns_empty():
    idx = PrefixHashIndex(block_tokens=4)
    result = _run(idx.lookup([1, 2, 3, 4], "m"))
    assert result == []


def test_remove():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = [1, 2, 3, 4]
    meta = make_meta(tokens)
    _run(idx.insert(meta))
    _run(idx.remove(meta.chunk_key))
    result = _run(idx.lookup(tokens, "m"))
    assert result == []


def test_model_id_isolation():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = [1, 2, 3, 4]
    meta = make_meta(tokens)
    _run(idx.insert(meta))
    result = _run(idx.lookup(tokens, "other-model"))
    assert result == []
