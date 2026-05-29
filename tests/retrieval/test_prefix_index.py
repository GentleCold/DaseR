# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import time

# First Party
from daser.connector.helpers import (
    ROLLING_PREFIX_SEED,
    hash_tokens,
    rolling_prefix_key,
)
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.metadata_store import ChunkMeta


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def rolling_keys(tokens: list[int], block_tokens: int) -> list[str]:
    """Return expected rolling-prefix keys for test assertions."""
    keys: list[str] = []
    key = ROLLING_PREFIX_SEED
    aligned = (len(tokens) // block_tokens) * block_tokens
    for start in range(0, aligned, block_tokens):
        key = rolling_prefix_key(key, tokens[start : start + block_tokens])
        keys.append(key)
    return keys


def make_meta(
    tokens: list[int],
    start: int = 0,
    num: int = 1,
    chunk_key: str | None = None,
) -> ChunkMeta:
    key = chunk_key or hash_tokens(tokens)
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
    key = rolling_keys(tokens, block_tokens=4)[0]
    meta = make_meta(tokens, chunk_key=key)
    _run(idx.insert(meta))
    result = _run(idx.lookup(tokens, "m"))
    assert len(result) == 1
    assert result[0].meta.chunk_key == meta.chunk_key
    assert result[0].target_token_start == 0


def test_lookup_longer_query_finds_prefix():
    idx = PrefixHashIndex(block_tokens=4)
    tokens4 = [1, 2, 3, 4]
    key = rolling_keys(tokens4, block_tokens=4)[0]
    meta = make_meta(tokens4, chunk_key=key)
    _run(idx.insert(meta))
    result = _run(idx.lookup([1, 2, 3, 4, 5, 6, 7, 8], "m"))
    assert len(result) == 1
    assert result[0].meta.chunk_key == meta.chunk_key
    assert result[0].target_token_start == 0


def test_rolling_prefix_key_chains_previous_key_with_block_tokens():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]

    keys = rolling_keys(tokens, block_tokens=4)
    first_key = rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:4])
    second_key_from_suffix_only = rolling_prefix_key(
        ROLLING_PREFIX_SEED,
        tokens[4:8],
    )

    assert len(keys) == 2
    assert keys[0] == first_key
    assert keys[1] != second_key_from_suffix_only


def test_lookup_returns_contiguous_rolling_prefix_slots():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    keys = rolling_keys(tokens, block_tokens=4)
    first = make_meta(tokens[:4], start=10, chunk_key=keys[0])
    second = make_meta(tokens[4:8], start=11, chunk_key=keys[1])
    _run(idx.insert(first))
    _run(idx.insert(second))

    result = _run(idx.lookup(tokens, "m"))

    assert [match.meta.chunk_key for match in result] == keys
    assert [match.target_token_start for match in result] == [0, 4]


def test_lookup_stops_at_first_missing_rolling_prefix_slot():
    idx = PrefixHashIndex(block_tokens=4)
    tokens = list(range(12))
    keys = rolling_keys(tokens, block_tokens=4)
    first = make_meta(tokens[:4], start=10, chunk_key=keys[0])
    third = make_meta(tokens[8:12], start=12, chunk_key=keys[2])
    _run(idx.insert(first))
    _run(idx.insert(third))

    result = _run(idx.lookup(tokens, "m"))

    assert [match.meta.chunk_key for match in result] == [keys[0]]


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
