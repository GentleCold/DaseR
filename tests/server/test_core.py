# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
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


@pytest.mark.asyncio
async def test_alloc_commit_lookup() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4]
    key = _hash_tokens(tokens)

    alloc = await core.alloc_chunk(key, token_count=len(tokens), model_id="m")
    assert alloc.file_offset == alloc.start_slot * SLOT_SIZE

    assert await core.lookup(tokens, "m") == []
    await core.commit_chunk(key)
    chunks = await core.lookup(tokens, "m")

    assert len(chunks) == 1
    assert chunks[0].chunk_key == key


@pytest.mark.asyncio
async def test_match_and_alloc_is_idempotent_before_commit() -> None:
    core = make_core()
    tokens = [1, 2, 3, 4, 5, 6]
    key = _hash_tokens(tokens[:4])

    first = await core.match_and_alloc(tokens, key, "m")
    second = await core.match_and_alloc(tokens, key, "m")

    assert first.chunks == []
    assert first.alloc == second.alloc
    assert first.alloc is not None
    assert first.alloc.chunk_key == key


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
