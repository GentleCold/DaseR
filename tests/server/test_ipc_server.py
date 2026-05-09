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
from daser.server.doc_registry import DocRegistry
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def make_server(tmp_path, total_slots: int = 64) -> IPCServer:
    socket_path = str(tmp_path / "test.sock")
    store = MetadataStore(total_slots=total_slots)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=total_slots,
        metadata_store=store,
        doc_registry=doc_registry,
    )
    ri = PrefixHashIndex(block_tokens=BLOCK_TOKENS)
    pe = FixedOffsetEncoder(fixed_offset=0)
    return IPCServer(
        socket_path=socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
        doc_registry=doc_registry,
    )


async def _send_recv(socket_path: str, payload: dict) -> dict:
    """Send one msgpack frame and receive one msgpack frame."""
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


@pytest.mark.asyncio
async def test_alloc_chunk(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    tokens = [1, 2, 3, 4]
    chunk_key = _hash_tokens(tokens)
    resp = await _send_recv(
        str(tmp_path / "test.sock"),
        {
            "op": "alloc_chunk",
            "chunk_key": chunk_key,
            "token_count": 4,
            "model_id": "m",
        },
    )
    assert "start_slot" in resp
    assert resp["file_offset"] == resp["start_slot"] * SLOT_SIZE
    await server.stop()


@pytest.mark.asyncio
async def test_alloc_chunk_is_idempotent_before_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    tokens = [1, 2, 3, 4]
    chunk_key = _hash_tokens(tokens)
    payload = {
        "op": "alloc_chunk",
        "chunk_key": chunk_key,
        "token_count": 4,
        "model_id": "m",
    }

    first = await _send_recv(sock, payload)
    second = await _send_recv(sock, payload)

    assert first == second
    assert "error" not in second
    await server.stop()


@pytest.mark.asyncio
async def test_commit_and_lookup(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    tokens = [1, 2, 3, 4]
    chunk_key = _hash_tokens(tokens)

    await _send_recv(
        sock,
        {
            "op": "alloc_chunk",
            "chunk_key": chunk_key,
            "token_count": 4,
            "model_id": "m",
        },
    )
    await _send_recv(sock, {"op": "commit_chunk", "chunk_key": chunk_key})

    resp = await _send_recv(sock, {"op": "lookup", "tokens": tokens, "model_id": "m"})
    assert len(resp["chunks"]) == 1
    assert resp["chunks"][0]["chunk_key"] == chunk_key
    await server.stop()


@pytest.mark.asyncio
async def test_lookup_miss(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    resp = await _send_recv(
        str(tmp_path / "test.sock"),
        {"op": "lookup", "tokens": [9, 8, 7, 6], "model_id": "m"},
    )
    assert resp["chunks"] == []
    await server.stop()


@pytest.mark.asyncio
async def test_match_and_alloc_is_idempotent_before_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    tokens = [1, 2, 3, 4, 5, 6]
    aligned_tokens = tokens[:4]
    chunk_key = _hash_tokens(aligned_tokens)
    payload = {
        "op": "match_and_alloc",
        "tokens": tokens,
        "chunk_key": chunk_key,
        "model_id": "m",
    }

    first = await _send_recv(sock, payload)
    second = await _send_recv(sock, payload)

    assert first == second
    assert first["chunks"] == []
    assert first["alloc"]["chunk_key"] == chunk_key
    assert "error" not in second
    await server.stop()


@pytest.mark.asyncio
async def test_evict_chunk(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    tokens = [1, 2, 3, 4]
    chunk_key = _hash_tokens(tokens)

    await _send_recv(
        sock,
        {
            "op": "alloc_chunk",
            "chunk_key": chunk_key,
            "token_count": 4,
            "model_id": "m",
        },
    )
    await _send_recv(sock, {"op": "commit_chunk", "chunk_key": chunk_key})
    await _send_recv(sock, {"op": "evict_chunk", "chunk_key": chunk_key})

    resp = await _send_recv(sock, {"op": "lookup", "tokens": tokens, "model_id": "m"})
    assert resp["chunks"] == []
    await server.stop()


@pytest.mark.asyncio
async def test_auto_eviction_removes_retrieval_index_entry(tmp_path):
    server = make_server(tmp_path, total_slots=2)
    await server.start()
    sock = str(tmp_path / "test.sock")
    tokens1 = [1, 2, 3, 4]
    tokens2 = [5, 6, 7, 8]
    tokens3 = [9, 10, 11, 12]

    for tokens in (tokens1, tokens2, tokens3):
        chunk_key = _hash_tokens(tokens)
        await _send_recv(
            sock,
            {
                "op": "alloc_chunk",
                "chunk_key": chunk_key,
                "token_count": len(tokens),
                "model_id": "m",
            },
        )
        await _send_recv(sock, {"op": "commit_chunk", "chunk_key": chunk_key})

    resp = await _send_recv(sock, {"op": "lookup", "tokens": tokens1, "model_id": "m"})
    assert resp["chunks"] == []
    await server.stop()


@pytest.mark.asyncio
async def test_lookup_doc_chunks_returns_offsets_for_registered_doc(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    chunk_keys = [_hash_tokens([1, 2, 3, 4]), _hash_tokens([5, 6, 7, 8])]

    for key in chunk_keys:
        await _send_recv(
            sock,
            {
                "op": "alloc_chunk",
                "chunk_key": key,
                "token_count": BLOCK_TOKENS,
                "model_id": "m",
            },
        )
        await _send_recv(sock, {"op": "commit_chunk", "chunk_key": key})

    await _send_recv(
        sock,
        {
            "op": "register_doc",
            "doc_id": "doc-a",
            "title": "Doc A",
            "chunk_keys": chunk_keys,
            "token_count": BLOCK_TOKENS * 2,
            "tokens": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    )

    resp = await _send_recv(
        sock,
        {
            "op": "lookup_doc_chunks",
            "doc_ids": ["doc-a"],
            "doc_start_offsets": [12],
            "model_id": "m",
        },
    )

    assert resp["missing"] == []
    assert [c["chunk_key"] for c in resp["chunks"]] == chunk_keys
    assert [c["source_pos_offset"] for c in resp["chunks"]] == [0, 0]
    assert [c["target_pos_offset"] for c in resp["chunks"]] == [12, 16]
    assert (
        resp["chunks"][0]["file_offset"] == resp["chunks"][0]["start_slot"] * SLOT_SIZE
    )
    await server.stop()


@pytest.mark.asyncio
async def test_lookup_doc_chunks_reports_missing_chunks(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "test.sock")
    present_key = _hash_tokens([1, 2, 3, 4])
    missing_key = _hash_tokens([5, 6, 7, 8])

    await _send_recv(
        sock,
        {
            "op": "alloc_chunk",
            "chunk_key": present_key,
            "token_count": BLOCK_TOKENS,
            "model_id": "m",
        },
    )
    await _send_recv(sock, {"op": "commit_chunk", "chunk_key": present_key})
    await _send_recv(
        sock,
        {
            "op": "register_doc",
            "doc_id": "doc-a",
            "title": "Doc A",
            "chunk_keys": [present_key, missing_key],
            "token_count": BLOCK_TOKENS * 2,
            "tokens": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    )

    resp = await _send_recv(
        sock,
        {
            "op": "lookup_doc_chunks",
            "doc_ids": ["doc-a"],
            "doc_start_offsets": [20],
            "model_id": "m",
        },
    )

    assert [c["chunk_key"] for c in resp["chunks"]] == [present_key]
    assert resp["missing"] == [
        {
            "doc_id": "doc-a",
            "chunk_index": 1,
            "chunk_key": missing_key,
            "reason": "evicted",
        }
    ]
    await server.stop()


@pytest.mark.asyncio
async def test_unknown_op(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    resp = await _send_recv(str(tmp_path / "test.sock"), {"op": "bad_op"})
    assert "error" in resp
    await server.stop()
