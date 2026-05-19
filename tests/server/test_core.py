# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


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
    key = _hash_tokens(tokens)

    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    assert alloc.file_offset == alloc.start_slot * SLOT_SIZE

    assert await core.lookup(tokens, "m") == []
    await core.commit_chunk(key)
    chunks = await core.lookup(tokens, "m", pin=True)

    assert len(chunks) == 1
    assert chunks[0].chunk_key == key
    assert chunks[0].residency == "l2_only"
    assert chunks[0].l2_durable is True
    await core.release_chunks([key])


@pytest.mark.asyncio
async def test_commit_l1_publishes_non_durable_chunk_and_commit_l2_durable() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = _hash_tokens(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_l1(key)

    chunks = await core.lookup(tokens, "m", pin=True)
    assert chunks == []

    meta = core.chunk_manager.store.get(key)
    assert meta is not None
    assert meta.residency == "l1_only"
    assert meta.l2_durable is False
    assert meta.pin_count == 1

    await core.evict_l1(key)
    assert meta.residency == "allocated"

    await core.commit_l2(key)
    assert meta.residency == "l2_only"
    assert meta.l2_durable is True
    assert meta.pin_count == 0
    chunks = await core.lookup(tokens, "m", pin=True)
    assert len(chunks) == 1
    assert chunks[0].residency == "l2_only"
    await core.release_chunks([key])

    await core.evict_l1(key)
    assert meta.residency == "l2_only"


@pytest.mark.asyncio
async def test_allocated_chunk_is_pinned_until_first_commit() -> None:
    """Ring eviction cannot remove a chunk before its first data-plane commit."""
    core = make_core(total_slots=1)
    first = [1, 2, 3, 4]
    first_key = _hash_tokens(first)

    await core.alloc_chunk(first_key, token_count=len(first), model_id="m")
    meta = core.chunk_manager.store.get(first_key)
    assert meta is not None
    assert meta.pin_count == 1

    await core.commit_l1(first_key)
    assert meta.pin_count == 1

    await core.commit_l2(first_key)
    assert meta.pin_count == 0


@pytest.mark.asyncio
async def test_lookup_only_pins_when_requested() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = _hash_tokens(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_chunk(key)
    meta = core.chunk_manager.store.get(key)
    assert meta is not None

    chunks = await core.lookup(tokens, "m")
    assert len(chunks) == 1
    assert meta.pin_count == 0

    chunks = await core.lookup(tokens, "m", pin=True)
    assert len(chunks) == 1
    assert meta.pin_count == 1


@pytest.mark.asyncio
async def test_l1_only_chunk_cannot_be_evicted_even_after_release() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = _hash_tokens(tokens)

    await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    await core.commit_l1(key)
    await core.release_chunks([key])

    await core.evict_l1(key)

    meta = core.chunk_manager.store.get(key)
    assert meta is not None
    assert meta.residency == "allocated"


@pytest.mark.asyncio
async def test_chunk_mode_lookup_returns_multiple_targeted_chunks() -> None:
    core = make_chunk_core()
    doc_a = [1, 2, 3, 4]
    sep = [90, 91, 92, 93]
    doc_b = [5, 6, 7, 8]
    task = [100, 101, 102, 103]
    key_a = _hash_tokens(doc_a)
    key_b = _hash_tokens(doc_b)

    for tokens in (doc_a, doc_b):
        key = _hash_tokens(tokens)
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
    key = _hash_tokens(tokens)
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
    key = _hash_tokens(tokens)
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
    first_key = _hash_tokens(first)

    for tokens in (first, second):
        key = _hash_tokens(tokens)
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)
    await core.register_document(
        doc_id="doc-1",
        title="first",
        chunk_keys=[first_key],
        token_count=len(first),
    )

    third_key = _hash_tokens(third)
    await core.alloc_chunk(third_key, token_count=len(third), model_id="m")
    await core.commit_chunk(third_key)

    assert await core.lookup(first, "m") == []
    doc = await core.get_document("doc-1")
    assert doc is not None
    assert doc.cached_mask == [False]


@pytest.mark.asyncio
async def test_ring_allocation_does_not_evict_lookup_pinned_chunk() -> None:
    core = make_core(total_slots=2)
    first = [1, 2, 3, 4]
    second = [5, 6, 7, 8]
    third = [9, 10, 11, 12]
    first_key = _hash_tokens(first)
    second_key = _hash_tokens(second)
    third_key = _hash_tokens(third)

    for tokens, key in ((first, first_key), (second, second_key)):
        await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
        await core.commit_chunk(key)

    pinned = await core.lookup(first, "m", pin=True)
    assert [chunk.chunk_key for chunk in pinned] == [first_key]

    with pytest.raises(MemoryError, match="pinned"):
        await core.alloc_chunk(third_key, token_count=len(third), model_id="m")

    assert [chunk.chunk_key for chunk in await core.lookup(first, "m")] == [first_key]
    assert [chunk.chunk_key for chunk in await core.lookup(second, "m")] == [second_key]
    await core.release_chunks([first_key])
