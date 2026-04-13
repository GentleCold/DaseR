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
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def make_server(tmp_path) -> IPCServer:
    socket_path = str(tmp_path / "test.sock")
    store = MetadataStore(total_slots=64)
    cm = ChunkManager(total_slots=64, metadata_store=store)
    ri = PrefixHashIndex(block_tokens=BLOCK_TOKENS)
    pe = FixedOffsetEncoder(fixed_offset=0)
    return IPCServer(
        socket_path=socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
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
async def test_unknown_op(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    resp = await _send_recv(str(tmp_path / "test.sock"), {"op": "bad_op"})
    assert "error" in resp
    await server.stop()
