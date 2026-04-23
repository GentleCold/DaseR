# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import msgpack
import pytest

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.chunk_manager import ChunkManager
from daser.server.doc_registry import DocEntry, DocRegistry
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def _make_server(tmp_path):
    socket_path = str(tmp_path / "test.sock")
    store = MetadataStore(total_slots=64)
    doc_registry = DocRegistry()
    cm = ChunkManager(total_slots=64, metadata_store=store, doc_registry=doc_registry)
    ri = PrefixHashIndex(block_tokens=BLOCK_TOKENS)
    pe = FixedOffsetEncoder(fixed_offset=0)
    server = IPCServer(
        socket_path=socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
        doc_registry=doc_registry,
    )
    return server, cm, ri, doc_registry, socket_path


async def _send_recv(socket_path: str, payload: dict) -> dict:
    reader, writer = await asyncio.open_unix_connection(socket_path)
    data = msgpack.packb(payload, use_bin_type=True)
    header = len(data).to_bytes(4, "big")
    writer.write(header + data)
    await writer.drain()
    resp_header = await reader.readexactly(4)
    resp_len = int.from_bytes(resp_header, "big")
    resp_data = await reader.readexactly(resp_len)
    writer.close()
    return msgpack.unpackb(resp_data, raw=False)


def test_doc_registry_insert_and_cached_mask():
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


@pytest.mark.asyncio
async def test_register_list_get_doc(tmp_path):
    server, cm, ri, doc_registry, socket_path = _make_server(tmp_path)
    await server.start()
    try:
        tokens = list(range(4))
        key = _hash_tokens(tokens)
        await _send_recv(
            socket_path,
            {
                "op": "alloc_chunk",
                "chunk_key": key,
                "token_count": 4,
                "model_id": "m",
            },
        )
        await _send_recv(socket_path, {"op": "commit_chunk", "chunk_key": key})

        resp = await _send_recv(
            socket_path,
            {
                "op": "register_doc",
                "doc_id": "doc-1",
                "title": "first",
                "chunk_keys": [key],
                "token_count": 4,
                "tokens": tokens,
            },
        )
        assert resp.get("ok") is True
        assert resp.get("chunk_count_cached") == 1

        resp = await _send_recv(socket_path, {"op": "list_docs"})
        assert len(resp["docs"]) == 1
        assert resp["docs"][0]["doc_id"] == "doc-1"
        assert resp["docs"][0]["chunk_count_cached"] == 1

        resp = await _send_recv(socket_path, {"op": "get_doc", "doc_id": "doc-1"})
        assert resp["doc"]["chunk_keys"] == [key]
        assert resp["doc"]["tokens"] == tokens

        meta = cm.store.get(key)
        assert meta is not None
        assert "doc-1" in meta.doc_ids
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_evict_doc_releases_chunks(tmp_path):
    server, cm, ri, doc_registry, socket_path = _make_server(tmp_path)
    await server.start()
    try:
        tokens = list(range(4))
        key = _hash_tokens(tokens)
        await _send_recv(
            socket_path,
            {
                "op": "alloc_chunk",
                "chunk_key": key,
                "token_count": 4,
                "model_id": "m",
            },
        )
        await _send_recv(socket_path, {"op": "commit_chunk", "chunk_key": key})
        await _send_recv(
            socket_path,
            {
                "op": "register_doc",
                "doc_id": "doc-1",
                "title": "d",
                "chunk_keys": [key],
                "token_count": 4,
            },
        )

        resp = await _send_recv(socket_path, {"op": "evict_doc", "doc_id": "doc-1"})
        assert resp.get("ok") is True
        assert resp.get("chunks_evicted") == 1
        assert cm.store.get(key) is None
        assert doc_registry.get("doc-1") is None
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_cascade_eviction_flips_cached_mask(tmp_path):
    server, cm, ri, doc_registry, socket_path = _make_server(tmp_path)
    await server.start()
    try:
        tokens = list(range(4))
        key = _hash_tokens(tokens)
        await _send_recv(
            socket_path,
            {
                "op": "alloc_chunk",
                "chunk_key": key,
                "token_count": 4,
                "model_id": "m",
            },
        )
        await _send_recv(socket_path, {"op": "commit_chunk", "chunk_key": key})
        await _send_recv(
            socket_path,
            {
                "op": "register_doc",
                "doc_id": "doc-1",
                "title": "d",
                "chunk_keys": [key],
                "token_count": 4,
            },
        )

        # Evict the chunk directly — DocRegistry should be updated via
        # the cascade inside _handle_evict_chunk.
        await _send_recv(socket_path, {"op": "evict_chunk", "chunk_key": key})

        resp = await _send_recv(socket_path, {"op": "get_doc", "doc_id": "doc-1"})
        assert resp["doc"]["cached_mask"] == [False]
        assert resp["doc"]["status"] == "evicted"
    finally:
        await server.stop()
