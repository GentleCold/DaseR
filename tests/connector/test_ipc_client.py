# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

# First Party
from daser.connector.helpers import ROLLING_PREFIX_SEED, hash_tokens, rolling_prefix_key
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


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


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
        runtime_config={
            "store_path": str(tmp_path / "daser.store"),
            "slot_size": SLOT_SIZE,
            "block_tokens": BLOCK_TOKENS,
            "model_id": "m",
            "transfer_mode": "iouring",
            "l1_size_bytes": 4096,
            "l2_size_bytes": 8192,
        },
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
async def test_sync_client_get_runtime_config(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    client = IPCClientSync(str(tmp_path / "ipc.sock"))
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, client.get_runtime_config)
    assert result["store_path"] == str(tmp_path / "daser.store")
    assert result["slot_size"] == SLOT_SIZE
    assert result["block_tokens"] == BLOCK_TOKENS
    assert result["model_id"] == "m"
    await server.stop()


@pytest.mark.asyncio
async def test_sync_client_reconnects_after_server_restart(tmp_path):
    server = make_server(tmp_path)
    sock = str(tmp_path / "ipc.sock")
    await server.start()
    client = IPCClientSync(sock)
    loop = asyncio.get_running_loop()
    try:
        first = await loop.run_in_executor(None, client.get_runtime_config)
        await server.stop()

        restarted = make_server(tmp_path)
        await restarted.start()
        try:
            second = await loop.run_in_executor(None, client.get_runtime_config)
        finally:
            await restarted.stop()
    finally:
        client.close()

    assert first["model_id"] == "m"
    assert second["model_id"] == "m"


@pytest.mark.asyncio
async def test_sync_client_alloc_and_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
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
async def test_sync_client_alloc_chunks(tmp_path):
    """Sync client can batch chunk allocations in one RPC."""
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    client = IPCClientSync(sock)
    tokens_a = [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8]
    key_a = first_rolling_key(tokens_a)
    key_b = first_rolling_key(tokens_b)
    loop = asyncio.get_running_loop()

    allocs = await loop.run_in_executor(
        None,
        lambda: client.alloc_chunks(
            [
                {"chunk_key": key_a, "token_count": 4},
                {"chunk_key": key_b, "token_count": 4},
            ],
            model_id="m",
        ),
    )

    assert [alloc["chunk_key"] for alloc in allocs] == [key_a, key_b]
    assert [alloc["start_slot"] for alloc in allocs] == [0, 1]
    await loop.run_in_executor(None, client.commit_chunk, key_a)
    await loop.run_in_executor(None, client.commit_chunk, key_b)
    assert len(await loop.run_in_executor(None, client.lookup, tokens_a, "m")) == 1
    assert len(await loop.run_in_executor(None, client.lookup, tokens_b, "m")) == 1
    await server.stop()


@pytest.mark.asyncio
async def test_async_client_commit(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    sync_client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: sync_client.alloc_chunk(key, token_count=4, model_id="m")
    )

    async_client = IPCClientAsync(sock)
    await async_client.commit_chunk(key)
    chunks = await loop.run_in_executor(None, sync_client.lookup, tokens, "m")
    assert len(chunks) == 1
    await server.stop()


@pytest.mark.asyncio
async def test_async_client_reconnects_after_server_restart(tmp_path):
    server = make_server(tmp_path)
    sock = str(tmp_path / "ipc.sock")
    await server.start()
    client = IPCClientAsync(sock)
    try:
        first = await client.call({"op": "get_runtime_config"})
        await server.stop()

        restarted = make_server(tmp_path)
        await restarted.start()
        try:
            second = await client.call({"op": "get_runtime_config"})
        finally:
            await restarted.stop()
    finally:
        await client.close()

    assert first["runtime_config"]["model_id"] == "m"
    assert second["runtime_config"]["model_id"] == "m"


@pytest.mark.asyncio
async def test_async_client_retries_when_stale_socket_close_breaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale async socket can fail both drain and wait_closed before retry."""

    class StaleWriter:
        closed = False
        waited = False

        def is_closing(self) -> bool:
            return False

        def write(self, _data: bytes) -> None:
            return

        async def drain(self) -> None:
            raise ConnectionResetError("connection lost")

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            self.waited = True
            raise BrokenPipeError("broken pipe during close")

    class FreshWriter:
        writes = 0

        def is_closing(self) -> bool:
            return False

        def write(self, _data: bytes) -> None:
            self.writes += 1

        async def drain(self) -> None:
            return

        def close(self) -> None:
            return

        async def wait_closed(self) -> None:
            return

    stale_writer = StaleWriter()
    fresh_writer = FreshWriter()
    writers = [stale_writer, fresh_writer]

    async def fake_open_unix_connection(_path: str):
        return object(), writers.pop(0)

    async def fake_read_frame(_reader):
        return {"ok": True}

    monkeypatch.setattr(
        "daser.connector.ipc_client.asyncio.open_unix_connection",
        fake_open_unix_connection,
    )
    monkeypatch.setattr("daser.connector.ipc_client.read_frame", fake_read_frame)

    client = IPCClientAsync("/tmp/daser.sock")

    result = await client.call({"op": "get_runtime_config"})

    assert result == {"ok": True}
    assert stale_writer.closed is True
    assert stale_writer.waited is True
    assert fresh_writer.writes == 1


@pytest.mark.asyncio
async def test_async_client_commit_chunks(tmp_path):
    """Async client can batch chunk commits in one RPC."""
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    sync_client = IPCClientSync(sock)
    tokens_a = [1, 2, 3, 4]
    tokens_b = [5, 6, 7, 8]
    key_a = first_rolling_key(tokens_a)
    key_b = first_rolling_key(tokens_b)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: sync_client.alloc_chunk(key_a, token_count=4, model_id="m")
    )
    await loop.run_in_executor(
        None, lambda: sync_client.alloc_chunk(key_b, token_count=4, model_id="m")
    )

    async_client = IPCClientAsync(sock)
    await async_client.commit_chunks([key_a, key_b])

    chunks_a = await loop.run_in_executor(None, sync_client.lookup, tokens_a, "m")
    chunks_b = await loop.run_in_executor(None, sync_client.lookup, tokens_b, "m")
    assert len(chunks_a) == 1
    assert len(chunks_b) == 1
    await async_client.close()
    await server.stop()


@pytest.mark.asyncio
async def test_async_client_transfer_store_and_load_bytes(tmp_path):
    """Async client exposes server-owned transfer store/load operations."""
    server = make_server(tmp_path)
    await server.start()
    client = IPCClientAsync(str(tmp_path / "ipc.sock"))
    data = b"abcdefgh" * 512
    try:
        await client.transfer_store_bytes(
            data=data,
            spans=[{"source_offset": 0, "nbytes": len(data), "file_offset": 0}],
        )
        payload = await client.transfer_load_bytes(
            spans=[{"target_offset": 0, "nbytes": len(data), "file_offset": 0}],
        )
        assert payload == data
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_async_client_transfer_cuda_payload_includes_allocation_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA transfer payloads include allocation base and tensor-view offset."""

    recorded: list[dict] = []

    async def fake_call(self, payload: dict) -> dict:
        del self
        recorded.append(payload)
        return {"chunk_keys": [], "ok": True}

    monkeypatch.setattr(IPCClientAsync, "call", fake_call)
    client = IPCClientAsync("/tmp/daser.sock")

    await client.transfer_store_cuda(
        cuda_ipc_handle=b"h" * 64,
        nbytes=1024,
        device_id=0,
        device_ptr=123456,
        allocation_base_ptr=122880,
        allocation_offset=576,
        producer_pid=42,
        spans=[{"source_offset": 0, "nbytes": 1024, "file_offset": 0}],
    )
    await client.transfer_load_cuda(
        cuda_ipc_handle=b"h" * 64,
        nbytes=2048,
        device_id=0,
        device_ptr=223456,
        allocation_base_ptr=221184,
        allocation_offset=2272,
        producer_pid=43,
        spans=[{"target_offset": 0, "nbytes": 2048, "file_offset": 0}],
    )

    assert recorded[0]["payload"]["allocation_base_ptr"] == 122880
    assert recorded[0]["payload"]["allocation_offset"] == 576
    assert recorded[1]["payload"]["allocation_base_ptr"] == 221184
    assert recorded[1]["payload"]["allocation_offset"] == 2272


@pytest.mark.asyncio
async def test_async_client_registers_and_loads_registered_staging_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async client can register fixed load staging and load by buffer index."""

    recorded: list[dict] = []

    async def fake_call(self, payload: dict) -> dict:
        del self
        recorded.append(payload)
        return {"ok": True, "bytes": 16}

    monkeypatch.setattr(IPCClientAsync, "call", fake_call)
    client = IPCClientAsync("/tmp/daser.sock")

    await client.register_load_staging_cuda(
        buffer_index=1,
        cuda_ipc_handle=b"h" * 64,
        allocation_bytes=4096,
        device_id=0,
        device_ptr=223456,
        allocation_base_ptr=221184,
        allocation_offset=2272,
        producer_pid=43,
    )
    response = await client.transfer_load_registered_cuda(
        buffer_index=1,
        producer_pid=43,
        nbytes=16,
        spans=[{"target_offset": 0, "nbytes": 16, "file_offset": 0}],
    )

    assert response == {"ok": True, "bytes": 16}
    assert recorded == [
        {
            "op": "register_load_staging",
            "payload": {
                "buffer_index": 1,
                "cuda_ipc_handle": b"h" * 64,
                "allocation_bytes": 4096,
                "device_id": 0,
                "device_ptr": 223456,
                "allocation_base_ptr": 221184,
                "allocation_offset": 2272,
                "producer_pid": 43,
            },
        },
        {
            "op": "transfer_load",
            "payload": {
                "load_staging_buffer_index": 1,
                "producer_pid": 43,
                "nbytes": 16,
            },
            "spans": [{"target_offset": 0, "nbytes": 16, "file_offset": 0}],
        },
    ]


@pytest.mark.asyncio
async def test_clients_transfer_drain(tmp_path):
    """Sync and async clients can drain server-owned transfer work."""
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    sync_client = IPCClientSync(sock)
    async_client = IPCClientAsync(sock)
    loop = asyncio.get_running_loop()
    data = b"abcdefgh" * 512
    try:
        await async_client.transfer_store_bytes(
            data=data,
            spans=[{"source_offset": 0, "nbytes": len(data), "file_offset": 0}],
        )
        await async_client.transfer_drain()
        await loop.run_in_executor(None, sync_client.transfer_drain)
    finally:
        sync_client.close()
        await async_client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_sync_client_commit_stats(tmp_path):
    """Sync client can read server-side commit counters."""
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    sync_client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None, lambda: sync_client.alloc_chunk(key, token_count=4, model_id="m")
        )
        await loop.run_in_executor(None, sync_client.commit_chunk, key)
        stats = await loop.run_in_executor(None, sync_client.commit_stats)
        assert stats["commit_requests"] == 1
        assert stats["late_evicted_commits"] == 0
    finally:
        sync_client.close()
        await server.stop()


@pytest.mark.asyncio
async def test_sync_client_live_allocations(tmp_path):
    """Sync client can batch-check live server allocations."""
    server = make_server(tmp_path)
    await server.start()
    sock = str(tmp_path / "ipc.sock")
    sync_client = IPCClientSync(sock)
    tokens = [1, 2, 3, 4]
    key = hash_tokens(tokens)
    loop = asyncio.get_running_loop()
    try:
        alloc = await loop.run_in_executor(
            None, lambda: sync_client.alloc_chunk(key, token_count=4, model_id="m")
        )
        live = await loop.run_in_executor(
            None,
            lambda: sync_client.live_allocations(
                [
                    {
                        "chunk_key": key,
                        "start_slot": alloc["start_slot"],
                        "num_slots": 1,
                    },
                    {
                        "chunk_key": "missing",
                        "start_slot": 0,
                        "num_slots": 1,
                    },
                ]
            ),
        )
        assert live == {key}
    finally:
        sync_client.close()
        await server.stop()
