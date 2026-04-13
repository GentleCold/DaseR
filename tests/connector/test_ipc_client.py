# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

# First Party
from daser.connector.daser_connector import hash_tokens
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore


def make_server(tmp_path, slot_size: int = 1024, block_tokens: int = 4) -> IPCServer:
    socket_path = str(tmp_path / "ipc.sock")
    store = MetadataStore(total_slots=64)
    cm = ChunkManager(total_slots=64, metadata_store=store)
    return IPCServer(
        socket_path=socket_path,
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=block_tokens),
        position_encoder=FixedOffsetEncoder(),
        slot_size=slot_size,
        block_tokens=block_tokens,
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
