# SPDX-License-Identifier: Apache-2.0

# Standard
import array
import asyncio
import contextlib
import socket
import threading
import time

# Third Party
import pytest

# First Party
from daser.connector.helpers import ROLLING_PREFIX_SEED, hash_tokens, rolling_prefix_key
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.ipc_protocol import pack_frame, recv_frame
from daser.server.ipc import IPCServer
from tests.server.core_helpers import (
    BLOCK_TOKENS,
    RECORD,
    load_span,
    make_ipc_server,
    raw_span,
)


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


def make_server(tmp_path) -> IPCServer:
    """Create an IPC server with both tiers for client tests."""
    return make_ipc_server(tmp_path)


async def store_chunk(
    client: IPCClientAsync, key: str, alloc: dict, data: bytes | None = None
) -> bytes:
    """Store one record per slot of ``key``'s allocation and publish it."""
    nbytes = int(alloc["num_slots"]) * RECORD
    data = data if data is not None else bytes(range(256)) * (nbytes // 256)
    await client.transfer_store_bytes(
        data=data,
        spans=[
            raw_span(
                key,
                int(alloc["start_slot"]),
                int(alloc["num_slots"]),
            )
        ],
    )
    await client.transfer_drain()
    return data


@pytest.mark.asyncio
async def test_sync_client_lookup(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    client = IPCClientSync(str(tmp_path / "ipc.sock"))
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, client.lookup, [1, 2, 3, 4], "m")
    assert result == []
    await server.stop()


def test_sync_client_validates_prefetch_lookup_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scheduler lookup decodes exact spans and the host-tier state once."""

    def fake_call(self, payload: dict) -> dict:
        del self
        assert payload["lease_id"] == "request-1"
        assert payload["token_bytes"] == bytes(array.array("i", [1, 2, 3, 4]))
        assert "tokens" not in payload
        return {
            "chunks": [{"chunk_key": "cached"}],
            "spans": [{"start_slot": 3, "num_slots": 2, "nbytes": 8192, "extra": 1}],
            "tier": "mixed",
        }

    monkeypatch.setattr(IPCClientSync, "call", fake_call)
    client = IPCClientSync("unused.sock")

    result = client.lookup_with_prefetch(
        "request-1",
        [1, 2, 3, 4],
        "m",
        external_prefix_queries=4,
    )

    assert result.tier == "mixed"
    assert result.spans == [{"start_slot": 3, "num_slots": 2, "nbytes": 8192}]


def test_sync_client_rejects_prefetch_spans_without_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lookup_prefetch span must name its slot range."""

    def fake_call(self, payload: dict) -> dict:
        del self, payload
        return {
            "chunks": [{"chunk_key": "cached"}],
            "spans": [{"nbytes": 8192}],
            "tier": "l2",
        }

    monkeypatch.setattr(IPCClientSync, "call", fake_call)
    client = IPCClientSync("unused.sock")

    with pytest.raises(RuntimeError, match="spans"):
        client.lookup_with_prefetch("request-1", [1, 2, 3, 4], "m", 4)


@pytest.mark.asyncio
async def test_sync_client_get_runtime_config(tmp_path):
    server = make_server(tmp_path)
    await server.start()
    client = IPCClientSync(str(tmp_path / "ipc.sock"))
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, client.get_runtime_config)
    assert result["store_path"] == str(tmp_path / "daser.store")
    assert result["local_slot_size"] == RECORD
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
    writer = IPCClientAsync(sock)
    try:
        await store_chunk(writer, key, alloc)
        chunks = await loop.run_in_executor(None, client.lookup, tokens, "m")
    finally:
        await writer.close()
        client.close()
        await server.stop()
    assert len(chunks) == 1
    assert chunks[0]["chunk_key"] == key


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
    writer = IPCClientAsync(sock)
    try:
        for alloc in allocs:
            await store_chunk(writer, alloc["chunk_key"], alloc)
        hits_a = await loop.run_in_executor(None, client.lookup, tokens_a, "m")
        hits_b = await loop.run_in_executor(None, client.lookup, tokens_b, "m")
    finally:
        await writer.close()
        client.close()
        await server.stop()
    assert len(hits_a) == 1
    assert len(hits_b) == 1


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
        reader = asyncio.StreamReader()
        reader.feed_data(pack_frame({"ok": True}))
        return reader, writers.pop(0)

    monkeypatch.setattr(
        "daser.connector.ipc_client.asyncio.open_unix_connection",
        fake_open_unix_connection,
    )

    client = IPCClientAsync("/tmp/daser.sock")

    result = await client.call({"op": "get_runtime_config"})

    assert result == {"ok": True}
    assert stale_writer.closed is True
    assert stale_writer.waited is True
    assert fresh_writer.writes == 1


@pytest.mark.asyncio
async def test_async_client_transfer_store_and_load_bytes(tmp_path):
    """Async client stores a chunk's records and loads them by slot range."""
    server = make_server(tmp_path)
    await server.start()
    client = IPCClientAsync(str(tmp_path / "ipc.sock"))
    try:
        key = hash_tokens([1] * 8)
        alloc = await client.call(
            {
                "op": "alloc_chunk",
                "chunk_key": key,
                "token_count": 2 * BLOCK_TOKENS,
                "model_id": "m",
                "parent_key": "",
            }
        )
        data = await store_chunk(client, key, alloc)
        start = int(alloc["start_slot"])
        whole = await client.transfer_load_bytes(spans=[load_span(start, 2)])
        tail = await client.transfer_load_bytes(spans=[load_span(start + 1, 1)])
    finally:
        await client.close()
        await server.stop()
    assert whole == data
    assert tail == data[RECORD:]


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
        device_pci_bus_id="0000:38:00.0",
        device_ptr=123456,
        allocation_base_ptr=122880,
        allocation_offset=576,
        producer_pid=42,
        spans=[raw_span("k", 0)],
    )
    await client.transfer_load_cuda(
        cuda_ipc_handle=b"h" * 64,
        nbytes=2048,
        device_pci_bus_id="0000:38:00.0",
        device_ptr=223456,
        allocation_base_ptr=221184,
        allocation_offset=2272,
        producer_pid=43,
        spans=[load_span(0)],
        tp_rank=1,
    )

    assert recorded[0]["payload"]["allocation_base_ptr"] == 122880
    assert recorded[0]["payload"]["allocation_offset"] == 576
    assert recorded[1]["payload"]["allocation_base_ptr"] == 221184
    assert recorded[1]["payload"]["allocation_offset"] == 2272
    assert recorded[1]["tp_rank"] == 1
    assert "lease_id" not in recorded[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["load", "store"])
async def test_async_client_registers_and_transfers_registered_staging_buffer(
    monkeypatch: pytest.MonkeyPatch,
    direction: str,
) -> None:
    """Fixed staging registration and transfer share the same ownership key."""

    recorded: list[dict] = []

    async def fake_call(self, payload: dict) -> dict:
        del self
        recorded.append(payload)
        return {"ok": True, "bytes": 16, "chunk_keys": ["accepted"]}

    monkeypatch.setattr(IPCClientAsync, "call", fake_call)
    client = IPCClientAsync("/tmp/daser.sock")

    await client.register_staging_cuda(
        direction=direction,
        buffer_index=1,
        cuda_ipc_handle=b"h" * 64,
        allocation_bytes=4096,
        device_pci_bus_id="0000:38:00.0",
        device_ptr=223456,
        allocation_base_ptr=221184,
        allocation_offset=2272,
        producer_pid=43,
    )
    if direction == "load":
        response = await client.transfer_load_registered_cuda(
            buffer_index=1,
            producer_pid=43,
            nbytes=16,
            spans=[load_span(0)],
            tp_rank=1,
        )
        assert response == {"ok": True, "bytes": 16, "chunk_keys": ["accepted"]}
    else:
        keys = await client.transfer_store_registered_cuda(
            buffer_index=1,
            producer_pid=43,
            spans=[raw_span("k", 0, source_offset=2048)],
            tp_rank=1,
            tp_size=2,
        )
        assert keys == ["accepted"]
    expected = [
        {
            "op": f"register_{direction}_staging",
            "payload": {
                "buffer_index": 1,
                "cuda_ipc_handle": b"h" * 64,
                "allocation_bytes": 4096,
                "device_pci_bus_id": "0000:38:00.0",
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
            "spans": [load_span(0)],
            "tp_rank": 1,
        },
    ]
    if direction == "store":
        expected[1] = {
            "op": "transfer_store",
            "payload": {"store_staging_buffer_index": 1, "producer_pid": 43},
            "spans": [raw_span("k", 0, source_offset=2048)],
            "tp_rank": 1,
            "tp_size": 2,
        }
    assert recorded == expected


@pytest.mark.asyncio
async def test_async_client_packs_leased_runs_as_one_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leased runs travel as one (4, n) int64 table, not per-span dicts."""

    recorded: list[dict] = []

    async def fake_call(self, payload: dict) -> dict:
        del self
        recorded.append(payload)
        return {"ok": True, "bytes": 48}

    monkeypatch.setattr(IPCClientAsync, "call", fake_call)
    client = IPCClientAsync("/tmp/daser.sock")

    response = await client.transfer_load_runs(
        buffer_index=1,
        producer_pid=43,
        start_slots=[3, 7],
        intra_offsets=[0, 64],
        target_offsets=[0, 32],
        nbytes=[16, 32],
        lease_ids=["request-runs"],
        tp_rank=1,
    )

    assert response == {"ok": True, "bytes": 48}
    assert recorded == [
        {
            "op": "transfer_load_runs",
            "payload": {"load_staging_buffer_index": 1, "producer_pid": 43},
            "lease_ids": ["request-runs"],
            "tp_rank": 1,
            "runs": array.array("q", [3, 7, 0, 64, 0, 32, 16, 32]).tobytes(),
        }
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
    key = hash_tokens([1, 2, 3, 4])
    try:
        alloc = await loop.run_in_executor(
            None, lambda: sync_client.alloc_chunk(key, token_count=4, model_id="m")
        )
        await async_client.transfer_store_bytes(
            data=bytes(RECORD),
            spans=[raw_span(key, int(alloc["start_slot"]))],
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
    async_client = IPCClientAsync(sock)
    try:
        alloc = await loop.run_in_executor(
            None, lambda: sync_client.alloc_chunk(key, token_count=4, model_id="m")
        )
        await store_chunk(async_client, key, alloc)
        stats = await loop.run_in_executor(None, sync_client.commit_stats)
        assert stats["commit_requests"] == 1
        assert stats["late_evicted_commits"] == 0
    finally:
        await async_client.close()
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


def _serve_slowly(
    listener: socket.socket, delay_s: float, received: list[dict]
) -> None:
    """Answer every framed request on ``listener`` after ``delay_s`` seconds."""
    listener.settimeout(2.0)
    while True:
        try:
            conn, _addr = listener.accept()
        except OSError:
            return
        with conn:
            try:
                msg = recv_frame(conn)
            except (ConnectionError, OSError):
                continue
            received.append(msg)
            time.sleep(delay_s)
            with contextlib.suppress(OSError):
                conn.sendall(pack_frame({"ok": True, "op": msg["op"]}))


@pytest.mark.parametrize("timeout", [None, 0.05])
def test_sync_client_timeout_bounds_slow_rpcs(tmp_path, timeout) -> None:
    """A client without a timeout waits for a slow RPC; a bounded one gives up.

    Giving up does not stop the server: the bounded client retries, so the
    server receives the request twice. Callers whose RPCs wait on server-side
    resources (such as prefetch leases) must therefore not set a timeout.
    """
    sock_path = str(tmp_path / "slow.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(sock_path)
    listener.listen()
    received: list[dict] = []
    thread = threading.Thread(
        target=_serve_slowly, args=(listener, 0.3, received), daemon=True
    )
    thread.start()
    client = IPCClientSync(sock_path, timeout=timeout)
    try:
        if timeout is None:
            assert client.call({"op": "transfer_prefetch"})["ok"] is True
            assert len(received) == 1
        else:
            with pytest.raises(RuntimeError, match="transport failure"):
                client.call({"op": "transfer_prefetch"})
            thread.join(timeout=0.5)
            assert len(received) == 2
    finally:
        client.close()
        listener.close()
        thread.join(timeout=3.0)
