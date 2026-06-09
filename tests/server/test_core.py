# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

from daser.connector.helpers import ROLLING_PREFIX_SEED, hash_tokens, rolling_prefix_key

# First Party
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


def make_core(total_slots: int = 64) -> ServerCore:
    """Create a ServerCore for tests."""
    store = MetadataStore(total_slots=total_slots)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=total_slots,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    return ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )


def make_chunk_core(total_slots: int = 64) -> ServerCore:
    """Create a chunk-reuse ServerCore for tests."""
    store = MetadataStore(total_slots=total_slots)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=total_slots,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    return ServerCore(
        chunk_manager=cm,
        retrieval_index=ChunkReuseIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=ChunkPositionEncoder(initial_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )


@pytest.mark.asyncio
async def test_alloc_commit_lookup() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    assert alloc.file_offset == alloc.start_slot * SLOT_SIZE

    assert await core.lookup(tokens, "m") == []
    await core.commit_chunk(key)
    chunks = await core.lookup(tokens, "m")

    assert len(chunks) == 1
    assert chunks[0].chunk_key == key
    assert core.is_chunk_committed(key) is True


@pytest.mark.asyncio
async def test_wait_for_committed_chunks_completes_after_commit() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    waiter = asyncio.create_task(core.wait_for_committed_chunks([key], timeout_s=1.0))
    await asyncio.sleep(0)

    assert waiter.done() is False
    await core.commit_chunk(key)
    await waiter


@pytest.mark.asyncio
async def test_wait_for_committed_chunks_times_out() -> None:
    core = make_core()

    with pytest.raises(TimeoutError):
        await core.wait_for_committed_chunks(["missing"], timeout_s=0.001)


@pytest.mark.asyncio
async def test_lookup_hit_updates_chunk_access_stats() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)

    store = core.chunk_manager.store
    meta_before = store.get(key)
    assert meta_before is not None
    assert meta_before.access_count == 0

    await core.lookup(tokens, "m")
    await core.lookup(tokens, "m")

    meta_after = store.get(key)
    assert meta_after is not None
    assert meta_after.access_count == 2
    assert meta_after.last_access_time >= meta_after.created_at


@pytest.mark.asyncio
async def test_lookup_can_skip_access_stats_for_diagnostics() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)

    hits = await core.lookup(tokens, "m", record_access=False)

    meta = core.chunk_manager.store.get(key)
    assert hits
    assert meta is not None
    assert meta.access_count == 0


@pytest.mark.asyncio
async def test_lookup_miss_leaves_access_stats_untouched() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)

    store = core.chunk_manager.store
    baseline = store.get(key)
    assert baseline is not None
    baseline_count = baseline.access_count

    await core.lookup([99, 98, 97, 96], "m")

    after = store.get(key)
    assert after is not None
    assert after.access_count == baseline_count


@pytest.mark.asyncio
async def test_restored_orphan_committed_chunk_can_be_reused(tmp_path) -> None:
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    original = make_core()
    await original.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await original.commit_chunk(key)
    original.chunk_manager.save(str(tmp_path / "daser.index"))

    store = MetadataStore(total_slots=64)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=64,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    cm.load(str(tmp_path / "daser.index"))
    restored = ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )

    await restored.rebuild_retrieval_index()

    chunks = await restored.lookup(tokens, "m")
    assert len(chunks) == 1
    assert chunks[0].chunk_key == key
    assert await restored.list_documents() == []


@pytest.mark.asyncio
async def test_match_and_alloc_is_idempotent_before_commit() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4, 5, 6]
    key = first_rolling_key(tokens)

    first = await core.match_and_alloc(tokens, key, "m")
    second = await core.match_and_alloc(tokens, key, "m")

    assert first.chunks == []
    assert first.alloc == second.alloc
    assert first.alloc is not None
    assert first.alloc.chunk_key == key


@pytest.mark.asyncio
async def test_chunk_mode_lookup_returns_multiple_targeted_chunks() -> None:
    core = make_chunk_core()
    doc_a = [1, 2, 3, 4]
    sep = [90, 91, 92, 93]
    doc_b = [5, 6, 7, 8]
    task = [100, 101, 102, 103]
    key_a = hash_tokens(doc_a)
    key_b = hash_tokens(doc_b)

    for tokens in (doc_a, doc_b):
        key = hash_tokens(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)

    chunks = await core.lookup(doc_a + sep + doc_b + task, "m")

    assert [chunk.chunk_key for chunk in chunks] == [key_a, key_b]
    assert [chunk.target_token_start for chunk in chunks] == [0, 8]
    assert [chunk.pos_offset for chunk in chunks] == [0, 8]


@pytest.mark.asyncio
async def test_register_list_get_delete_document() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)

    registration = await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[key],
        token_count=len(tokens),
        tokens=tokens,
    )
    assert registration.chunk_count_cached == 1

    docs = await core.list_documents()
    assert len(docs) == 1
    assert docs[0].doc_id == "doc-1"
    assert docs[0].chunk_count_cached == 1

    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.chunk_keys == [key]
    assert doc.tokens == tokens

    result = await core.delete_document("doc-1")
    assert result.chunks_evicted == 1
    assert await core.get_document("doc-1") is None


@pytest.mark.asyncio
async def test_evict_chunk_flips_document_cached_mask() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)
    await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[key],
        token_count=len(tokens),
    )

    await core.evict_chunk(key)

    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.cached_mask == [False]
    assert doc.status == "evicted"


@pytest.mark.asyncio
async def test_auto_eviction_removes_lookup_and_updates_doc() -> None:
    core = make_core(total_slots=2)
    first = [1, 2, 3, 4]
    second = [5, 6, 7, 8]
    third = [9, 10, 11, 12]
    first_key = hash_tokens(first)

    for tokens in (first, second):
        key = hash_tokens(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)
    await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[first_key],
        token_count=len(first),
    )

    third_key = hash_tokens(third)
    await core.alloc_chunk(third_key, token_count=len(third), model_id="m")
    await core.commit_chunk(third_key)

    assert await core.lookup(first, "m") == []
    assert core.is_chunk_committed(first_key) is False
    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.cached_mask == [False]


@pytest.mark.asyncio
async def test_late_commit_after_auto_eviction_is_ignored() -> None:
    core = make_core(total_slots=2)
    first = [1, 2, 3, 4]
    second = [5, 6, 7, 8]
    third = [9, 10, 11, 12]
    first_key = hash_tokens(first)

    await core.alloc_chunk(first_key, token_count=len(first), model_id="m")
    for tokens in (second, third):
        key = hash_tokens(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)

    await core.commit_chunk(first_key)

    assert await core.lookup(first, "m") == []
    stats = await core.commit_stats()
    assert stats["commit_requests"] == 3
    assert stats["late_evicted_commits"] == 1


@pytest.mark.asyncio
async def test_is_current_allocation_rejects_evicted_or_reused_slot() -> None:
    """Delayed transfer writes are valid only for the current live chunk."""
    core = make_core(total_slots=2)
    first = [1, 2, 3, 4]
    second = [5, 6, 7, 8]
    third = [9, 10, 11, 12]
    first_key = hash_tokens(first)

    first_alloc = await core.alloc_chunk(
        first_key,
        token_count=len(first),
        model_id="m",
    )
    assert core.is_current_allocation(
        first_key,
        first_alloc.start_slot,
        first_alloc.num_slots,
    )

    for tokens in (second, third):
        key = hash_tokens(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)

    assert not core.is_current_allocation(
        first_key,
        first_alloc.start_slot,
        first_alloc.num_slots,
    )
