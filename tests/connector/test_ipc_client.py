# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

# First Party
from daser.connector.helpers import hash_tokens
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

SLOT_SIZE = 1024
BLOCK_TOKENS = 4


def make_core() -> ServerCore:
    """Create a ServerCore for IPC client tests."""
    store = MetadataStore(total_slots=64)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=64,
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


def make_server(tmp_path) -> IPCServer:
    """Create an IPC server for client tests."""
    socket_path = str(tmp_path / "ipc.sock")
    return IPCServer(
        socket_path=socket_path,
        core=make_core(),
    )


@pytest.mark.asyncio
async def test_sync_client_lookup(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    client = IPCClientSync(str(tmp_path / "ipc.sock"))
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, client.lookup, [1, 2, 3, 4], "m")
    assert result == []
    await server.stop()


@pytest.mark.asyncio
async def test_sync_client_alloc_and_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    loop = asyncio.get_running_loop()
    alloc = await loop.run_in_executor(
        None, lambda: client.alloc_chunk(key, token_count=4, model_id="m")
    )
    assert "start_slot" in alloc
    await loop.run_in_executor(None, client.commit_chunk, key)
    chunks = await loop.run_in_executor(None, client.lookup, tokens, "m")
    assert len(chunks) == 1
    assert chunks[0]["chunk_key"] == key
    await server.stop()


@pytest.mark.asyncio
async def test_async_client_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    sync_client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: sync_client.alloc_chunk(key, token_count=4, model_id="m")
    )

    async_client = IPCClientAsync(sock)
    await async_client.commit_chunk(key)
    chunks = await loop.run_in_executor(None, sync_client.lookup, tokens, "m")
    assert len(chunks) == 1
    await server.stop()
