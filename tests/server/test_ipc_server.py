# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
from types import SimpleNamespace
from typing import Any

# Third Party
import msgpack
import pytest

# First Party
from daser.connector.helpers import ROLLING_PREFIX_SEED, rolling_prefix_key
from daser.metrics import MetricsRegistry
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore
from daser.transfer.base import TransferLayer, TransferStats

SLOT_SIZE = 4096
BLOCK_TOKENS = 4


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


RUNTIME_CONFIG = {
    "slot_size": SLOT_SIZE,
    "block_tokens": BLOCK_TOKENS,
    "model_id": "m",
    "transfer_mode": "iouring",
    "l1_size_bytes": 8192,
    "l2_size_bytes": 8192,
}


def make_runtime_config(tmp_path: Any) -> dict[str, Any]:
    """Create runtime config with a per-test store path."""
    runtime_config = dict(RUNTIME_CONFIG)
    runtime_config["store_path"] = str(tmp_path / "daser.store")
    return runtime_config


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
    server = IPCServer(str(tmp_path / "test.sock"), core, make_runtime_config(tmp_path))
    await server.start()
    try:
        tokens = [1, 2, 3, 4]
        key = first_rolling_key(tokens)
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
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_persistent_connection_match_and_alloc(tmp_path) -> None:
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, make_runtime_config(tmp_path))
    await server.start()
    try:
        tokens = [1, 2, 3, 4, 5]
        key = first_rolling_key(tokens)
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
        assert responses[0]["alloc"]["chunk_key"] == key
        assert responses[1]["alloc"]["chunk_key"] == key
        assert (
            responses[1]["alloc"]["start_slot"] == responses[0]["alloc"]["start_slot"]
        )
        assert responses[0]["alloc"]["skipped"] is False
        assert responses[1]["alloc"]["skipped"] is True
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_document_ops_are_not_ipc_server(tmp_path) -> None:
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, make_runtime_config(tmp_path))
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
    runtime_config = make_runtime_config(tmp_path)
    server = IPCServer(str(tmp_path / "test.sock"), core, runtime_config)
    await server.start()
    try:
        resp = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "get_runtime_config"},
        )
        assert resp == {"runtime_config": runtime_config}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_ipc_server_records_operation_metrics(tmp_path) -> None:
    """IPC requests should record op counts and latencies."""
    registry = MetricsRegistry()
    core = make_core()
    server = IPCServer(
        str(tmp_path / "test.sock"),
        core,
        make_runtime_config(tmp_path),
        metrics_registry=registry,
    )
    await server.start()
    try:
        resp = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "get_runtime_config"},
        )
        assert "runtime_config" in resp
    finally:
        await server.stop()

    rendered = registry.render_prometheus()
    assert (
        'daser_ipc_requests_total{op="get_runtime_config",status="ok"} 1.0' in rendered
    )
    assert (
        'daser_ipc_request_duration_seconds_count{op="get_runtime_config"} 1'
        in rendered
    )


def test_ipc_server_records_tier_counter_deltas(tmp_path) -> None:
    """Tier metrics publish monotonic L1 and L2 deltas exactly once."""
    registry = MetricsRegistry()
    server = IPCServer(
        str(tmp_path / "test.sock"),
        make_core(),
        make_runtime_config(tmp_path),
        metrics_registry=registry,
    )
    transfer = SimpleNamespace(
        stats=TransferStats(l1_hits=2, l1_misses=3, l2_reads=4),
        l1_bytes_used=1024,
    )
    server._transfer = transfer  # type: ignore[assignment]  # noqa: SLF001

    server._record_tier_metrics()  # noqa: SLF001
    server._record_tier_metrics()  # noqa: SLF001

    rendered = registry.render_prometheus()
    assert "daser_l1_hits_total 2.0" in rendered
    assert "daser_l1_misses_total 3.0" in rendered
    assert "daser_l2_reads_total 4.0" in rendered


@pytest.mark.asyncio
async def test_ipc_server_records_external_prefix_cache_metrics(tmp_path) -> None:
    """IPC can publish vLLM-equivalent external prefix cache counters."""
    registry = MetricsRegistry()
    core = make_core()
    core._metrics = registry  # noqa: SLF001
    server = IPCServer(
        str(tmp_path / "test.sock"),
        core,
        make_runtime_config(tmp_path),
        metrics_registry=registry,
    )
    await server.start()
    try:
        resp = await _send_recv(
            str(tmp_path / "test.sock"),
            {"op": "record_external_prefix_cache", "queries": 12, "hits": 8},
        )
        assert resp == {"ok": True}
    finally:
        await server.stop()

    rendered = registry.render_prometheus()
    assert "daser_external_prefix_cache_queries_total 12.0" in rendered
    assert "daser_external_prefix_cache_hits_total 8.0" in rendered


@pytest.mark.asyncio
async def test_ipc_lookup_records_external_prefix_cache_metrics(tmp_path) -> None:
    """Lookup RPC records vLLM-equivalent external prefix query and hit counters."""
    registry = MetricsRegistry()
    core = make_core()
    core._metrics = registry  # noqa: SLF001
    server = IPCServer(
        str(tmp_path / "test.sock"),
        core,
        make_runtime_config(tmp_path),
        metrics_registry=registry,
    )
    await server.start()
    try:
        tokens = [1, 2, 3, 4]
        key = first_rolling_key(tokens)
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
            {"op": "commit_chunk", "chunk_key": key},
        )
        resp = await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "lookup",
                "tokens": tokens,
                "model_id": "m",
                "external_prefix_queries": 8,
                "num_computed_tokens": 0,
            },
        )
        assert len(resp["chunks"]) == 1
    finally:
        await server.stop()

    rendered = registry.render_prometheus()
    assert "daser_external_prefix_cache_queries_total 8.0" in rendered
    assert "daser_external_prefix_cache_hits_total 4.0" in rendered


@pytest.mark.asyncio
async def test_ipc_lookup_caps_full_external_prefix_hit_like_vllm(tmp_path) -> None:
    """Lookup metrics cap full-prompt external hits the same way as vLLM."""
    registry = MetricsRegistry()
    core = make_core()
    core._metrics = registry  # noqa: SLF001
    server = IPCServer(
        str(tmp_path / "test.sock"),
        core,
        make_runtime_config(tmp_path),
        metrics_registry=registry,
    )
    await server.start()
    try:
        tokens = [1, 2, 3, 4]
        key = first_rolling_key(tokens)
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
            {"op": "commit_chunk", "chunk_key": key},
        )
        await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "lookup",
                "tokens": tokens,
                "model_id": "m",
                "external_prefix_queries": 4,
                "num_computed_tokens": 0,
            },
        )
    finally:
        await server.stop()

    rendered = registry.render_prometheus()
    assert "daser_external_prefix_cache_queries_total 4.0" in rendered
    assert "daser_external_prefix_cache_hits_total 3.0" in rendered


@pytest.mark.asyncio
async def test_ipc_server_records_transfer_metrics_without_info_summary(
    tmp_path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transfer ops record metrics without hot-path INFO throughput logs."""
    registry = MetricsRegistry()
    core = make_core()
    server = IPCServer(
        str(tmp_path / "test.sock"),
        core,
        make_runtime_config(tmp_path),
        metrics_registry=registry,
    )
    await server.start()
    try:
        store = await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "transfer_store",
                "payload": {"data": b"a" * SLOT_SIZE},
                "spans": [{"source_offset": 0, "nbytes": SLOT_SIZE, "file_offset": 0}],
            },
        )
        assert store["bytes"] == SLOT_SIZE
    finally:
        await server.stop()

    rendered = registry.render_prometheus()
    assert 'daser_transfer_operations_total{op="store",status="ok"} 1.0' in rendered
    assert f'daser_transfer_bytes_total{{op="store"}} {float(SLOT_SIZE)}' in rendered
    assert 'daser_transfer_duration_seconds_count{op="store"} 1' in rendered
    assert "throughput_gbps=" not in caplog.text


@pytest.mark.asyncio
async def test_transfer_store_and_load_with_bytes_payload(tmp_path) -> None:
    """IPC transfer ops call the server-owned transfer layer."""
    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, make_runtime_config(tmp_path))
    await server.start()
    try:
        store = await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "transfer_store",
                "payload": {"data": b"a" * SLOT_SIZE + b"b" * SLOT_SIZE},
                "spans": [
                    {"source_offset": 0, "nbytes": SLOT_SIZE, "file_offset": 0},
                    {
                        "source_offset": SLOT_SIZE,
                        "nbytes": SLOT_SIZE,
                        "file_offset": SLOT_SIZE,
                    },
                ],
            },
        )
        load = await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "transfer_load",
                "payload": {"return_data": True},
                "spans": [
                    {"target_offset": 0, "nbytes": SLOT_SIZE, "file_offset": 0},
                    {
                        "target_offset": SLOT_SIZE,
                        "nbytes": SLOT_SIZE,
                        "file_offset": SLOT_SIZE,
                    },
                ],
            },
        )

        assert store == {"ok": True, "bytes": SLOT_SIZE * 2, "chunk_keys": []}
        assert load == {
            "ok": True,
            "bytes": SLOT_SIZE * 2,
            "data": b"a" * SLOT_SIZE + b"b" * SLOT_SIZE,
        }
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_transfer_store_skips_stale_chunk_span(tmp_path) -> None:
    """IPC store ignores delayed spans whose chunk allocation was evicted."""
    core = make_core()
    runtime_config = make_runtime_config(tmp_path)
    server = IPCServer(str(tmp_path / "test.sock"), core, runtime_config)
    await server.start()
    try:
        store = await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "transfer_store",
                "payload": {"data": b"a" * SLOT_SIZE},
                "spans": [
                    {
                        "source_offset": 0,
                        "nbytes": SLOT_SIZE,
                        "file_offset": 0,
                        "chunk_key": "evicted",
                        "start_slot": 0,
                        "num_slots": 1,
                    }
                ],
            },
        )
        load = await _send_recv(
            str(tmp_path / "test.sock"),
            {
                "op": "transfer_load",
                "payload": {"return_data": True},
                "spans": [{"target_offset": 0, "nbytes": SLOT_SIZE, "file_offset": 0}],
            },
        )

        assert store == {"ok": True, "bytes": 0, "chunk_keys": []}
        assert load == {"ok": True, "bytes": SLOT_SIZE, "data": b"\0" * SLOT_SIZE}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_cuda_ipc_payload_buffer_reuses_open_handle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated CUDA IPC payloads for the same staging allocation reuse a handle."""

    class FakeOpened:
        def __init__(self) -> None:
            self.array = bytearray(1024)
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    opened_buffers: list[FakeOpened] = []
    open_calls: list[dict[str, Any]] = []

    def fake_open_cuda_ipc_buffer(**kwargs: Any) -> FakeOpened:
        open_calls.append(kwargs)
        opened = FakeOpened()
        opened_buffers.append(opened)
        return opened

    class FakeTransfer(TransferLayer):
        async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
            return 0

        async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
            return 0

        async def load_bytes_grouped(
            self,
            _dst: Any,
            _spans: list[dict[str, int]],
        ) -> int:
            return 0

        def close(self) -> None:
            pass

    def fake_ensure_transfer(_server: IPCServer) -> FakeTransfer:
        return FakeTransfer()

    monkeypatch.setattr(
        "daser.server.ipc.server.open_cuda_ipc_buffer",
        fake_open_cuda_ipc_buffer,
    )
    monkeypatch.setattr(
        "daser.server.ipc.server._CachedCudaArray.synchronize",
        lambda _self: None,
    )
    monkeypatch.setattr(IPCServer, "_ensure_transfer", fake_ensure_transfer)

    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, make_runtime_config(tmp_path))
    payload = {
        "cuda_ipc_handle": b"h" * 64,
        "nbytes": 1024,
        "device_id": 0,
        "device_ptr": 123456,
        "allocation_base_ptr": 122880,
        "allocation_offset": 576,
        "producer_pid": 42,
    }

    await server.start()
    try:
        request = {
            "op": "transfer_load",
            "payload": payload,
            "spans": [],
        }
        first = await _send_recv(str(tmp_path / "test.sock"), request)
        second = await _send_recv(str(tmp_path / "test.sock"), dict(request))
    finally:
        await server.stop()

    assert first["ok"] is True
    assert second["ok"] is True
    assert len(opened_buffers) == 1
    assert open_calls == [
        {
            "handle": b"h" * 64,
            "nbytes": 1024,
            "device_id": 0,
            "local_ptr": None,
            "allocation_offset": 576,
        }
    ]
    assert opened_buffers[0].closed == 1


@pytest.mark.asyncio
async def test_registered_load_staging_is_scoped_by_producer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equal worker-local indexes resolve to each producer's CUDA mapping."""

    class FakeOpened:
        def __init__(self, marker: int) -> None:
            self.array = bytearray([marker]) * 1024
            self.closed = 0

        def close(self) -> None:
            self.closed += 1

    opened_buffers: list[FakeOpened] = []
    open_calls: list[dict[str, Any]] = []
    loaded_buffers: list[Any] = []

    def fake_open_cuda_ipc_buffer(**kwargs: Any) -> FakeOpened:
        open_calls.append(kwargs)
        opened = FakeOpened(len(opened_buffers) + 1)
        opened_buffers.append(opened)
        return opened

    class FakeTransfer(TransferLayer):
        async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
            return 0

        async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
            return 0

        async def load_bytes_grouped(
            self,
            dst: Any,
            spans: list[dict[str, int]],
        ) -> int:
            loaded_buffers.append(dst)
            return sum(int(span["nbytes"]) for span in spans)

        def close(self) -> None:
            pass

    def fake_ensure_transfer(_server: IPCServer) -> FakeTransfer:
        return FakeTransfer()

    monkeypatch.setattr(
        "daser.server.ipc.server.open_cuda_ipc_buffer",
        fake_open_cuda_ipc_buffer,
    )
    monkeypatch.setattr(
        "daser.server.ipc.server._CachedCudaArray.synchronize",
        lambda _self: None,
    )
    monkeypatch.setattr(IPCServer, "_ensure_transfer", fake_ensure_transfer)

    core = make_core()
    server = IPCServer(str(tmp_path / "test.sock"), core, make_runtime_config(tmp_path))

    await server.start()
    try:
        registered = []
        loaded = []
        for producer_pid in (42, 43):
            registered.append(
                await _send_recv(
                    str(tmp_path / "test.sock"),
                    {
                        "op": "register_load_staging",
                        "payload": {
                            "buffer_index": 1,
                            "cuda_ipc_handle": b"h" * 64,
                            "allocation_bytes": 1024,
                            "device_id": 0,
                            "device_ptr": 123456,
                            "allocation_base_ptr": 122880,
                            "allocation_offset": 576,
                            "producer_pid": producer_pid,
                        },
                    },
                )
            )
        for producer_pid in (42, 43):
            loaded.append(
                await _send_recv(
                    str(tmp_path / "test.sock"),
                    {
                        "op": "transfer_load",
                        "payload": {
                            "load_staging_buffer_index": 1,
                            "producer_pid": producer_pid,
                            "nbytes": 128,
                        },
                        "spans": [
                            {"target_offset": 0, "nbytes": 128, "file_offset": 0}
                        ],
                    },
                )
            )
    finally:
        await server.stop()

    assert registered == [{"ok": True}, {"ok": True}]
    assert [response["bytes"] for response in loaded] == [128, 128]
    assert [bytes(buffer[0:1]) for buffer in loaded_buffers] == [b"\x01", b"\x02"]
    assert open_calls == 2 * [
        {
            "handle": b"h" * 64,
            "nbytes": 1024,
            "device_id": 0,
            "local_ptr": None,
            "allocation_offset": 576,
        }
    ]
    assert [opened.closed for opened in opened_buffers] == [1, 1]


@pytest.mark.asyncio
async def test_stop_accepting_closes_listener_before_transfer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class FakeTransfer(TransferLayer):
        def __init__(self, **_kwargs: Any) -> None:
            events.append("ensure_transfer")

        async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
            return nbytes

        async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
            return nbytes

        async def drain(self) -> None:
            events.append("drain")

        def close(self) -> None:
            events.append("transfer_close")

    monkeypatch.setattr(
        "daser.server.ipc.server.TieredIOUringTransferLayer",
        FakeTransfer,
    )

    core = make_core()
    socket_path = str(tmp_path / "test.sock")
    server = IPCServer(socket_path, core, make_runtime_config(tmp_path))
    await server.start()

    init = await _send_recv(socket_path, {"op": "init_transfer"})
    assert init == {"ok": True}
    assert events == ["ensure_transfer"]

    await server.stop_accepting()

    assert not os.path.exists(socket_path)
    assert events == ["ensure_transfer"]

    await server.close()

    assert events == ["ensure_transfer", "drain", "transfer_close"]


@pytest.mark.asyncio
async def test_eager_transfer_initialization_avoids_lazy_init_on_first_request(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eagerly initializing the transfer layer creates it before any request."""
    init_events: list[str] = []

    class FakeTransfer(TransferLayer):
        def __init__(self, **_kwargs: Any) -> None:
            init_events.append("transfer_created")

        async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
            return nbytes

        async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
            return nbytes

        async def drain(self) -> None:
            pass

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "daser.server.ipc.server.TieredIOUringTransferLayer",
        FakeTransfer,
    )

    core = make_core()
    socket_path = str(tmp_path / "test.sock")
    server = IPCServer(socket_path, core, make_runtime_config(tmp_path))
    await server.start()

    # Eagerly initialize — must create the transfer before any request
    await server.initialize_transfer()
    assert init_events == ["transfer_created"]

    # First transfer_load must NOT trigger a second construction
    load = await _send_recv(
        socket_path,
        {
            "op": "transfer_load",
            "payload": {"return_data": True},
            "spans": [{"target_offset": 0, "nbytes": 8, "file_offset": 0}],
        },
    )
    assert load["ok"] is True
    assert init_events == ["transfer_created"]  # no second construction

    await server.stop()


@pytest.mark.asyncio
async def test_skip_l2_selects_iouring_transfer_without_store_path(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """skip_l2 should wire IPC transfer ops to iouring with its L2 tier disabled."""
    init_kwargs: list[dict[str, Any]] = []

    class FakeTieredIOUringTransfer(TransferLayer):
        coalesce_store_spans = True

        def __init__(self, **kwargs: Any) -> None:
            init_kwargs.append(kwargs)

        async def load_bytes(self, dst: Any, file_offset: int, nbytes: int) -> int:
            return await self.load_bytes_grouped(
                dst,
                [{"target_offset": 0, "file_offset": file_offset, "nbytes": nbytes}],
            )

        async def store_bytes(self, src: Any, file_offset: int, nbytes: int) -> int:
            return await self.store_bytes_grouped(
                src,
                [{"source_offset": 0, "file_offset": file_offset, "nbytes": nbytes}],
            )

        async def store_bytes_grouped(
            self,
            src: Any,
            spans: list[dict[str, Any]],
        ) -> int:
            total = 0
            for span in spans:
                source_offset = int(span.get("source_offset", 0))
                nbytes = int(span["nbytes"])
                assert int(span["file_offset"]) == 0
                assert bytes(src[source_offset : source_offset + nbytes]) == (
                    b"a" * nbytes
                )
                total += nbytes
            return total

        async def load_bytes_grouped(
            self,
            dst: Any,
            spans: list[dict[str, Any]],
        ) -> int:
            total = 0
            for span in spans:
                target_offset = int(span.get("target_offset", 0))
                nbytes = int(span["nbytes"])
                assert int(span["file_offset"]) == 0
                memoryview(dst).cast("B")[target_offset : target_offset + nbytes] = (
                    b"a" * nbytes
                )
                total += nbytes
            return total

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        "daser.server.ipc.server.TieredIOUringTransferLayer",
        FakeTieredIOUringTransfer,
    )

    runtime_config = make_runtime_config(tmp_path)
    runtime_config["skip_l2"] = True
    runtime_config["store_path"] = ""
    core = make_core()
    socket_path = str(tmp_path / "test.sock")
    server = IPCServer(socket_path, core, runtime_config)
    await server.start()
    try:
        store = await _send_recv(
            socket_path,
            {
                "op": "transfer_store",
                "payload": {"data": b"a" * SLOT_SIZE},
                "spans": [{"source_offset": 0, "nbytes": SLOT_SIZE, "file_offset": 0}],
            },
        )
        load = await _send_recv(
            socket_path,
            {
                "op": "transfer_load",
                "payload": {"return_data": True},
                "spans": [{"target_offset": 0, "nbytes": SLOT_SIZE, "file_offset": 0}],
            },
        )
    finally:
        await server.stop()

    assert init_kwargs == [
        {"path": "", "l1_bytes": 8192, "l2_bytes": 8192, "skip_l2": True}
    ]
    assert store == {"ok": True, "bytes": SLOT_SIZE, "chunk_keys": []}
    assert load == {"ok": True, "bytes": SLOT_SIZE, "data": b"a" * SLOT_SIZE}
