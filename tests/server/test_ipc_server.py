# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from typing import Any

# Third Party
import msgpack
import pytest

from daser.position.fixed_offset import FixedOffsetEncoder

# First Party
from daser.retrieval.prefix import PrefixHashIndex, _hash_tokens
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


RUNTIME_CONFIG = {
    "store_path": "/tmp/daser.store",
    "slot_size": SLOT_SIZE,
    "block_tokens": BLOCK_TOKENS,
    "model_id": "m",
}


def make_core(total_slots: int = 64) -> ServerCore:
    """Create a ServerCore for connector API tests."""
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


async def _send_recv(socket_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one request and receive one response."""
    reader, writer = await asyncio.open_unix_connection(socket_path)
    data = msgpack.packb(payload, use_bin_type=True)
    writer.write(len(data).to_bytes(4, "big") + data)
    await writer.drain()
    header = await reader.readexactly(4)
    length = int.from_bytes(header, "big")
    body = await reader.readexactly(length)
    writer.close()
    await writer.wait_closed()
    return msgpack.unpackb(body, raw=False)


async def _send_recv_persistent(
    socket_path: str, payloads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Send multiple requests on one connection."""
    reader, writer = await asyncio.open_unix_connection(socket_path)
    responses = []
    try:
        for payload in payloads:
            data = msgpack.packb(payload, use_bin_type=True)
            writer.write(len(data).to_bytes(4, "big") + data)
            await writer.drain()
            header = await reader.readexactly(4)
            length = int.from_bytes(header, "big")
            body = await reader.readexactly(length)
            responses.append(msgpack.unpackb(body, raw=False))
    finally:
        writer.close()
        await writer.wait_closed()
    return responses


@pytest.mark.asyncio
async def test_alloc_commit_lookup(tmp_path) -> None:
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, RUNTIME_CONFIG)
    await server.start()
    try:
        tokens = [1, 2, 3, 4]
        key = _hash_tokens(tokens)
        alloc = await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "alloc_chunk",
                "chunk_key": key,
                "token_count": len(tokens),
                "model_id": "m",
            },
        )
        assert alloc["file_offset"] == alloc["start_slot"] * SLOT_SIZE
        await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "commit_chunk", "chunk_key": key},
        )
        lookup = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "lookup", "tokens": tokens, "model_id": "m"},
        )
        assert lookup["chunks"][0]["chunk_key"] == key
        assert lookup["chunks"][0]["l2_durable"] is True
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_l1_l2_ipc_lifecycle(tmp_path) -> None:
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, RUNTIME_CONFIG)
    await server.start()
    try:
        tokens = [1, 2, 3, 4]
        key = _hash_tokens(tokens)
        await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "alloc_chunk",
                "chunk_key": key,
                "token_count": len(tokens),
                "model_id": "m",
            },
        )
        await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "commit_l1", "chunk_key": key},
        )
        lookup = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "lookup", "tokens": tokens, "model_id": "m"},
        )
        assert lookup["chunks"] == []

        await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "release_chunks", "chunk_keys": [key]},
        )
        await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "commit_l2", "chunk_key": key},
        )
        lookup = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "lookup", "tokens": tokens, "model_id": "m"},
        )
        assert lookup["chunks"][0]["residency"] == "l1_l2"
        assert lookup["chunks"][0]["l2_durable"] is True
        await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "release_chunks", "chunk_keys": [key]},
        )
        await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "evict_l1", "chunk_key": key},
        )
        lookup = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "lookup", "tokens": tokens, "model_id": "m"},
        )
        assert lookup["chunks"][0]["residency"] == "l2_only"
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_persistent_connection_match_and_alloc(tmp_path) -> None:
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, RUNTIME_CONFIG)
    await server.start()
    try:
        tokens = [1, 2, 3, 4, 5]
        key = _hash_tokens(tokens[:BLOCK_TOKENS])
        responses = await _send_recv_persistent(
            str(tmp_path / "test.sock"),
            [
                {
                    "op": "match_and_alloc",
                    "tokens": tokens,
                    "chunk_key": key,
                    "model_id": "m",
                },
                {
                    "op": "match_and_alloc",
                    "tokens": tokens,
                    "chunk_key": key,
                    "model_id": "m",
                },
            ],
        )
        assert responses[0] == responses[1]
        assert responses[0]["alloc"]["chunk_key"] == key
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_document_ops_are_not_ipc_server(tmp_path) -> None:
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, RUNTIME_CONFIG)
    await server.start()
    try:
        for op in ("register_doc", "list_docs", "get_doc", "evict_doc"):
            resp = await _send_recv(str(tmp_path / "test.sock"), {"op": op})
            assert resp == {"error": f"unknown op: {op}"}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_get_runtime_config(tmp_path) -> None:
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, RUNTIME_CONFIG)
    await server.start()
    try:
        resp = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "get_runtime_config"},
        )
        assert resp == {"runtime_config": RUNTIME_CONFIG}
    finally:
        await server.stop()
