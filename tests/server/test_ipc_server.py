# SPDX-License-Identifier: Apache-2.0
"""IPC server contract: lookup, server-placed stores and loads, leases."""

# Standard
import asyncio
from collections.abc import Sequence
import os
from pathlib import Path
import threading
from typing import Any

# Third Party
import msgpack
import numpy as np
import pytest

# First Party
from daser.connector.helpers import (
    ROLLING_PREFIX_SEED,
    rolling_prefix_key,
    rolling_prefix_keys,
)
from daser.metrics import MetricsRegistry
from daser.server.core import ChunkInfo, ServerCore
from daser.server.ipc import IPCServer
from daser.server.ipc import server as ipc_server_module
from daser.server.ipc.server import _CachedCudaArray
from daser.server.tiers import L1, L2
from daser.transfer import Move, StoreHandle, TransferLayer, TransferStats
from tests.server.core_helpers import (
    BLOCK_TOKENS,
    RECORD,
    alloc_and_store,
    ipc_runtime_config,
    load_span,
    make_core,
    make_ipc_server,
    make_layout,
    raw_span,
)

PRODUCER = 42


def first_rolling_key(tokens: list[int]) -> str:
    """Return the first rolling-prefix key for one test block."""
    return rolling_prefix_key(ROLLING_PREFIX_SEED, tokens[:BLOCK_TOKENS])


def pattern(seed: int, nbytes: int = RECORD) -> bytes:
    """Return ``nbytes`` of a seed-dependent byte pattern."""
    return bytes((seed * 31 + i) % 251 for i in range(nbytes))


async def _send_recv(socket_path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Send one request and receive one response."""
    (response,) = await _send_recv_persistent(socket_path, [payload])
    return response


async def _send_recv_persistent(
    socket_path: str, payloads: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Send several requests on one connection and return the responses."""
    reader, writer = await asyncio.open_unix_connection(socket_path)
    responses = []
    try:
        for payload in payloads:
            data = msgpack.packb(payload, use_bin_type=True)
            writer.write(len(data).to_bytes(4, "big") + data)
            await writer.drain()
            length = int.from_bytes(await reader.readexactly(4), "big")
            body = await reader.readexactly(length)
            responses.append(msgpack.unpackb(body, raw=False))
    finally:
        writer.close()
        await writer.wait_closed()
    return responses


def sock(server_path: Path) -> str:
    """Return the socket path ``make_ipc_server`` uses under ``tmp_path``."""
    return str(server_path / "ipc.sock")


async def alloc(
    socket_path: str, key: str, token_count: int = BLOCK_TOKENS, parent: str = ""
) -> dict[str, Any]:
    """Allocate one chunk over IPC."""
    return await _send_recv(
        socket_path,
        {
            "op": "alloc_chunk",
            "chunk_key": key,
            "token_count": token_count,
            "model_id": "m",
            "parent_key": parent,
        },
    )


async def store(
    socket_path: str, spans: list[dict[str, Any]], data: bytes, **extra: Any
) -> dict[str, Any]:
    """Store spans from a bytes payload (rank 0 of 1)."""
    return await _send_recv(
        socket_path,
        {
            "op": "transfer_store",
            "payload": {"data": data},
            "spans": spans,
            "tp_rank": 0,
            "tp_size": 1,
            **extra,
        },
    )


async def load(socket_path: str, spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Load spans into a returned bytes buffer (rank 0)."""
    return await _send_recv(
        socket_path,
        {
            "op": "transfer_load",
            "payload": {"return_data": True},
            "spans": spans,
            "tp_rank": 0,
        },
    )


async def drain(socket_path: str) -> None:
    """Wait for background store reports and transfer IO."""
    assert await _send_recv(socket_path, {"op": "transfer_drain"}) == {"ok": True}


async def store_raw_chunk(
    socket_path: str, key: str, data: bytes, parent: str = ""
) -> dict[str, Any]:
    """Allocate and store one raw chunk of ``len(data)`` bytes, then drain."""
    num_slots = len(data) // RECORD
    allocation = await alloc(socket_path, key, num_slots * BLOCK_TOKENS, parent)
    reply = await store(
        socket_path, [raw_span(key, allocation["start_slot"], num_slots)], data
    )
    assert reply["chunk_keys"] == [key]
    await drain(socket_path)
    return allocation


class FakeTransfer(TransferLayer):
    """Transfer double recording moves; stores complete at once."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.events: list[str] = []
        self.stored: list[tuple[Move, bytes]] = []
        self.loads: list[tuple[Any, list[Move], str]] = []
        self.runs: list[tuple[Any, list[int], list[int], list[int]]] = []
        self.persist_error: BaseException | None = None
        self._stats = TransferStats()

    @property
    def stats(self) -> TransferStats:
        """Return the (static) counters."""
        return self._stats

    async def store(self, buffer: Any, moves: Sequence[Move]) -> StoreHandle:
        """Record each move's source bytes and resolve the handle."""
        for move in moves:
            source = buffer[move.buffer_offset : move.buffer_offset + move.nbytes]
            self.stored.append((move, bytes(source)))
        loop = asyncio.get_running_loop()
        staged: asyncio.Future[None] = loop.create_future()
        persisted: asyncio.Future[None] = loop.create_future()
        staged.set_result(None)
        if self.persist_error is None:
            persisted.set_result(None)
        else:
            persisted.set_exception(self.persist_error)
        return StoreHandle(staged, persisted)

    async def load(self, buffer: Any, moves: Sequence[Move], source: str) -> None:
        """Record the load."""
        self.loads.append((buffer, list(moves), source))

    async def promote(self, moves: Sequence[Move]) -> None:
        """Accept promotions."""

    async def load_runs(
        self, buffer: Any, l1_offsets: Any, targets: Any, lengths: Any
    ) -> None:
        """Record one run table."""
        self.runs.append((buffer, list(l1_offsets), list(targets), list(lengths)))

    async def drain(self) -> None:
        """Record the drain."""
        self.events.append("drain")

    async def close(self) -> None:
        """Record the close."""
        self.events.append("close")


class FakeOpened:
    """Stand-in for an opened CUDA IPC mapping."""

    def __init__(self, nbytes: int, marker: int = 0) -> None:
        self.array = (np.arange(nbytes, dtype=np.uint32) ^ marker).astype(np.uint8)
        self.closed = 0

    def close(self) -> None:
        """Count closes."""
        self.closed += 1


def staging_payload(index: int, nbytes: int, producer: int = PRODUCER) -> dict:
    """Return a ``register_*_staging`` payload."""
    return {
        "buffer_index": index,
        "producer_pid": producer,
        "cuda_ipc_handle": b"h" * 64,
        "allocation_bytes": nbytes,
        "device_pci_bus_id": "0000:38:00.0",
        "device_ptr": 4096,
        "allocation_base_ptr": 4096,
        "allocation_offset": 0,
    }


# ----------------------------------------------------------------------
# Lookup, allocation and config
# ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("storage_format", "expected"),
    [("raw", False), ("compressed-online", True)],
)
async def test_lookup_waits_only_for_online_packed_store(
    tmp_path: Path, storage_format: str, expected: bool
) -> None:
    """Online packed lookups wait for pending writers; raw lookups do not."""

    class LookupCore:
        tier_layout = make_layout(l2_records=4)
        index_epoch = 0
        exclusive_tiers = False

        def __init__(self) -> None:
            self.waits: list[bool] = []

        async def lookup(
            self, _tokens: Any, _model_id: str, *, wait_for_pending: bool = False
        ) -> list[ChunkInfo]:
            self.waits.append(wait_for_pending)
            return []

    core = LookupCore()
    server = IPCServer(
        sock(tmp_path),
        core,
        {"storage_format": storage_format},  # type: ignore[arg-type]
    )
    await server.start()
    try:
        reply = await _send_recv(
            sock(tmp_path), {"op": "lookup", "tokens": [1, 2, 3, 4], "model_id": "m"}
        )
    finally:
        await server.stop()
    assert reply == {"chunks": [], "index_epoch": 0}
    assert core.waits == [expected]


@pytest.mark.asyncio
async def test_lookup_defer_pending_answers_without_waiting(tmp_path: Path) -> None:
    """defer_pending replies pending at once; False resolves immediately."""
    server = make_ipc_server(tmp_path, storage_format="compressed-online")
    socket_path = sock(tmp_path)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    msg = {"op": "lookup", "tokens": tokens, "model_id": "m"}
    await server.start()
    try:
        allocation = await alloc(socket_path, key)
        loop = asyncio.get_running_loop()
        started = loop.time()
        deferred = await _send_recv(socket_path, {**msg, "defer_pending": True})
        immediate = await _send_recv(socket_path, {**msg, "defer_pending": False})
        assert loop.time() - started < 0.5
        assert deferred == {"chunks": [], "pending": True}
        assert immediate == {"chunks": [], "index_epoch": 0}

        await store(
            socket_path, [raw_span(key, allocation["start_slot"])], b"a" * RECORD
        )
        await drain(socket_path)
        resolved = await _send_recv(socket_path, {**msg, "defer_pending": True})
    finally:
        await server.stop()
    assert [chunk["chunk_key"] for chunk in resolved["chunks"]] == [key]
    assert "pending" not in resolved


@pytest.mark.asyncio
async def test_lookup_reply_carries_index_epoch_until_a_removal(
    tmp_path: Path,
) -> None:
    core = make_core(l2_records=64, l1_records=16)
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    tokens = [1, 2, 3, 4]
    key = first_rolling_key(tokens)
    await server.start()
    try:
        await store_raw_chunk(socket_path, key, pattern(1))
        lookup = await _send_recv(
            socket_path, {"op": "lookup", "tokens": tokens, "model_id": "m"}
        )
        epoch = lookup["index_epoch"]
        assert lookup["chunks"][0]["chunk_key"] == key
        assert "file_offset" not in lookup["chunks"][0]
        # An insertion leaves earlier results valid; a removal does not.
        await alloc(socket_path, first_rolling_key([9, 9, 9, 9]))
        assert (await _send_recv(socket_path, {"op": "index_epoch"}))[
            "index_epoch"
        ] == epoch
        await _send_recv(socket_path, {"op": "evict_chunk", "chunk_key": key})
        assert (await _send_recv(socket_path, {"op": "index_epoch"}))[
            "index_epoch"
        ] > epoch
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_persistent_connection_match_and_alloc(tmp_path: Path) -> None:
    server = make_ipc_server(tmp_path)
    tokens = [1, 2, 3, 4, 5]
    key = first_rolling_key(tokens)
    request = {"op": "match_and_alloc", "tokens": tokens, "chunk_key": key}
    await server.start()
    try:
        first, second = await _send_recv_persistent(
            sock(tmp_path), [{**request, "model_id": "m"}, {**request, "model_id": "m"}]
        )
    finally:
        await server.stop()
    assert first["alloc"]["chunk_key"] == second["alloc"]["chunk_key"] == key
    assert first["alloc"]["start_slot"] == second["alloc"]["start_slot"]
    assert "file_offset" not in first["alloc"]
    assert (first["alloc"]["skipped"], second["alloc"]["skipped"]) == (False, True)


@pytest.mark.asyncio
async def test_document_ops_are_not_ipc_ops(tmp_path: Path) -> None:
    server = make_ipc_server(tmp_path)
    await server.start()
    try:
        for op in ("register_doc", "list_docs", "get_doc", "evict_doc"):
            reply = await _send_recv(sock(tmp_path), {"op": op})
            assert reply == {"error": f"unknown op: {op}"}
    finally:
        await server.stop()


@pytest.mark.asyncio
async def test_get_runtime_config_and_operation_metrics(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    server = make_ipc_server(tmp_path, registry=registry)
    expected = ipc_runtime_config(tmp_path, make_layout(64, 16))
    await server.start()
    try:
        reply = await _send_recv(sock(tmp_path), {"op": "get_runtime_config"})
    finally:
        await server.stop()
    assert reply == {"runtime_config": expected}
    rendered = registry.render_prometheus()
    assert (
        'daser_ipc_requests_total{op="get_runtime_config",status="ok"} 1.0' in rendered
    )
    assert (
        'daser_ipc_request_duration_seconds_count{op="get_runtime_config"} 1'
        in rendered
    )


@pytest.mark.asyncio
async def test_external_prefix_cache_metrics(tmp_path: Path) -> None:
    """Lookups count vLLM-equivalent external prefix queries and hits."""
    registry = MetricsRegistry()
    server = make_ipc_server(tmp_path, registry=registry)
    socket_path = sock(tmp_path)
    tokens = [1, 2, 3, 4]
    await server.start()
    try:
        await store_raw_chunk(socket_path, first_rolling_key(tokens), pattern(1))
        lookup = {"op": "lookup", "tokens": tokens, "model_id": "m"}
        partial = await _send_recv(
            socket_path,
            {**lookup, "external_prefix_queries": 8, "num_computed_tokens": 0},
        )
        # A full-prompt hit is capped at queries - 1, as vLLM does.
        await _send_recv(
            socket_path,
            {**lookup, "external_prefix_queries": 4, "num_computed_tokens": 0},
        )
        standalone = await _send_recv(
            socket_path, {"op": "record_external_prefix_cache", "queries": 2, "hits": 1}
        )
    finally:
        await server.stop()
    assert len(partial["chunks"]) == 1
    assert standalone == {"ok": True}
    rendered = registry.render_prometheus()
    assert "daser_external_prefix_cache_queries_total 14.0" in rendered
    assert "daser_external_prefix_cache_hits_total 8.0" in rendered


# ----------------------------------------------------------------------
# Stores and loads (real io_uring transfer)
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_and_load_roundtrip(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A stored chunk is published, loadable from L1 and counted."""
    registry = MetricsRegistry()
    server = make_ipc_server(tmp_path, registry=registry)
    socket_path = sock(tmp_path)
    tokens = list(range(8))
    key = first_rolling_key(tokens)
    data = pattern(1) + pattern(2)
    await server.start()
    try:
        allocation = await alloc(socket_path, key, 8)
        reply = await store(
            socket_path, [raw_span(key, allocation["start_slot"], 2)], data
        )
        assert reply == {"ok": True, "bytes": 2 * RECORD, "chunk_keys": [key]}
        await drain(socket_path)
        lookup = await _send_recv(
            socket_path, {"op": "lookup", "tokens": tokens, "model_id": "m"}
        )
        assert [c["chunk_key"] for c in lookup["chunks"]] == [key]
        assert "compressed_slots" not in lookup["chunks"][0]
        loaded = await load(socket_path, [load_span(allocation["start_slot"], 2)])
        stats = await _send_recv(socket_path, {"op": "commit_stats"})
    finally:
        await server.stop()
    assert loaded == {
        "ok": True,
        "bytes": 2 * RECORD,
        "l1_segments": 1,
        "l2_segments": 0,
        "data": data,
    }
    assert stats["commit_stats"]["commit_requests"] == 1
    rendered = registry.render_prometheus()
    assert 'daser_transfer_operations_total{op="store",status="ok"} 1.0' in rendered
    assert f'daser_transfer_bytes_total{{op="store"}} {float(2 * RECORD)}' in rendered
    assert 'daser_transfer_duration_seconds_count{op="load"} 1' in rendered
    assert "daser_l1_hits_total 1.0" in rendered
    assert "throughput_gbps=" not in caplog.text


@pytest.mark.asyncio
async def test_chunk_split_across_store_requests_publishes_once_complete(
    tmp_path: Path,
) -> None:
    """Raw slots of one chunk may arrive in several requests (slot_index)."""
    core = make_core(64, 16)
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    key = "split"
    await server.start()
    try:
        start = (await alloc(socket_path, key, 3 * BLOCK_TOKENS))["start_slot"]
        first = await store(
            socket_path,
            [raw_span(key, start, 3, slot_index=2, count=1)],
            pattern(3),
        )
        await drain(socket_path)
        assert first["chunk_keys"] == [key]
        assert not core.is_chunk_committed(key)
        second = await store(
            socket_path,
            [raw_span(key, start, 3, slot_index=0, count=2)],
            pattern(1) + pattern(2),
        )
        await drain(socket_path)
        assert second["chunk_keys"] == [key]
        assert core.is_chunk_committed(key)
        loaded = await load(socket_path, [load_span(start, 3)])
    finally:
        await server.stop()
    assert loaded["data"] == pattern(1) + pattern(2) + pattern(3)


@pytest.mark.asyncio
async def test_released_partial_store_is_skipped_as_stale(tmp_path: Path) -> None:
    """After release_chunk_writer the rest of a split store is dropped."""
    core = make_core(64, 16)
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    key = "partial"
    await server.start()
    try:
        start = (await alloc(socket_path, key, 2 * BLOCK_TOKENS))["start_slot"]
        await store(
            socket_path, [raw_span(key, start, 2, slot_index=0, count=1)], pattern(1)
        )
        await drain(socket_path)
        released = await _send_recv(
            socket_path,
            {
                "op": "release_chunk_writer",
                "chunk_key": key,
                "start_slot": start,
                "num_slots": 2,
            },
        )
        late = await store(
            socket_path, [raw_span(key, start, 2, slot_index=1, count=1)], pattern(2)
        )
        await drain(socket_path)
    finally:
        await server.stop()
    assert released == {"released": True}
    assert late == {"ok": True, "bytes": 0, "chunk_keys": []}
    assert not core.is_chunk_committed(key)


@pytest.mark.asyncio
async def test_stale_store_span_is_skipped(tmp_path: Path) -> None:
    server = make_ipc_server(tmp_path)
    socket_path = sock(tmp_path)
    key = first_rolling_key([1, 2, 3, 4])
    await server.start()
    try:
        start = (await alloc(socket_path, key))["start_slot"]
        wrong_slot = await store(socket_path, [raw_span(key, start + 7)], pattern(1))
        await _send_recv(socket_path, {"op": "evict_chunk", "chunk_key": key})
        evicted = await store(socket_path, [raw_span(key, start)], pattern(1))
    finally:
        await server.stop()
    assert wrong_slot == {"ok": True, "bytes": 0, "chunk_keys": []}
    assert evicted == {"ok": True, "bytes": 0, "chunk_keys": []}


@pytest.mark.asyncio
async def test_load_spans_mid_chunk_and_across_chunks(tmp_path: Path) -> None:
    """A load span may start inside a chunk and cover the next one."""
    server = make_ipc_server(tmp_path)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        a = await store_raw_chunk(socket_path, "a", pattern(1) + pattern(2))
        b = await store_raw_chunk(socket_path, "b", pattern(3))
        assert b["start_slot"] == a["start_slot"] + 2
        loaded = await load(
            socket_path,
            [
                load_span(a["start_slot"] + 1, 2, target_offset=0),
                load_span(a["start_slot"], 1, target_offset=2 * RECORD),
            ],
        )
        short = await load(
            socket_path,
            [{**load_span(a["start_slot"], 2), "nbytes": RECORD}],
        )
        unknown = await load(socket_path, [load_span(1000)])
    finally:
        await server.stop()
    assert loaded["data"] == pattern(2) + pattern(3) + pattern(1)
    assert loaded["l1_segments"] == 3
    assert "error" in short
    assert "error" in unknown


@pytest.mark.asyncio
async def test_l2_read_promotes_the_chunk_into_l1(tmp_path: Path) -> None:
    """A whole L2-only chunk read once is served from L1 next time."""
    core = make_core(64, 1)
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        a = await store_raw_chunk(socket_path, "a", pattern(1))
        await store_raw_chunk(socket_path, "b", pattern(2))
        assert core.full_tiers("a") == (L2,)
        first = await load(socket_path, [load_span(a["start_slot"])])
        assert L1 in core.full_tiers("a")
        second = await load(socket_path, [load_span(a["start_slot"])])
    finally:
        await server.stop()
    assert (first["l1_segments"], first["l2_segments"]) == (0, 1)
    assert (second["l1_segments"], second["l2_segments"]) == (1, 0)
    assert first["data"] == second["data"] == pattern(1)


@pytest.mark.asyncio
async def test_load_copies_ready_chunks_while_another_promotes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An L1 chunk is copied before a sibling's in-flight promotion ends."""
    fake = FakeTransfer()
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    core = make_core(64, 2)
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        b = await store_raw_chunk(socket_path, "b", pattern(1))
        await evict_from_l1(socket_path, core, ["b"])
        a = await store_raw_chunk(socket_path, "a", pattern(2))
        assert L1 in core.full_tiers("a")
        (promotion,) = await core.begin_promotions(["b"])
        assert promotion is not None and core.promotion_pending("b")
        loading = asyncio.ensure_future(
            load(
                socket_path,
                [
                    load_span(b["start_slot"]),
                    load_span(a["start_slot"], target_offset=RECORD),
                ],
            )
        )
        for _ in range(200):
            if fake.loads or loading.done():
                break
            await asyncio.sleep(0.01)
        early = list(fake.loads)
        still_waiting = not loading.done()
        core.finish_promotion(promotion, True)
        reply = await asyncio.wait_for(loading, 5)
    finally:
        await server.stop()
    assert still_waiting
    assert [
        (source, [m.buffer_offset for m in moves]) for _b, moves, source in early
    ] == [(L1, [RECORD])]
    assert (reply["l1_segments"], reply["l2_segments"]) == (2, 0)
    assert [m.buffer_offset for _b, moves, _s in fake.loads for m in moves] == [
        RECORD,
        0,
    ]


@pytest.mark.asyncio
async def test_load_span_with_wrong_nbytes_reads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A span whose nbytes disagrees with its records fails before any IO."""
    fake = FakeTransfer()
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    server = make_ipc_server(tmp_path)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        a = await store_raw_chunk(socket_path, "a", pattern(1))
        b = await store_raw_chunk(socket_path, "b", pattern(2))
        reply = await load(
            socket_path,
            [
                load_span(a["start_slot"]),
                {**load_span(b["start_slot"], target_offset=RECORD), "nbytes": 1},
            ],
        )
    finally:
        await server.stop()
    assert "record bytes" in reply["error"]
    assert fake.loads == []


@pytest.mark.asyncio
async def test_packed_store_publishes_record_modes(tmp_path: Path) -> None:
    """A packed chunk carries per-slot modes and stored lengths in lookups."""
    registry = MetricsRegistry()
    server = make_ipc_server(
        tmp_path,
        registry=registry,
        storage_format="compressed-online",
        local_slot_size=2 * RECORD,
    )
    socket_path = sock(tmp_path)
    tokens = list(range(8))
    key = rolling_prefix_keys(tokens, BLOCK_TOKENS)[0]
    data = pattern(1, 2 * RECORD) + pattern(2)
    await server.start()
    try:
        start = (await alloc(socket_path, key, 8))["start_slot"]
        packed = {"chunk_key": key, "start_slot": start, "num_slots": 2}
        spans = [
            {
                **packed,
                "source_offset": 0,
                "nbytes": 2 * RECORD,
                "slot_index": 0,
                "packed": True,
                "mode": "raw",
            },
            {
                **packed,
                "source_offset": 2 * RECORD,
                "nbytes": RECORD,
                "slot_index": 1,
                "packed": True,
                "mode": "compressed",
            },
        ]
        missing = await store(socket_path, spans[:1], data)
        unknown = await store(socket_path, [spans[0], {**spans[1], "mode": "x"}], data)
        reply = await store(socket_path, spans, data)
        await drain(socket_path)
        lookup = await _send_recv(
            socket_path, {"op": "lookup", "tokens": tokens, "model_id": "m"}
        )
        loaded = await load(
            socket_path,
            [
                {
                    "target_offset": 0,
                    "start_slot": start,
                    "num_slots": 2,
                    "nbytes": 3 * RECORD,
                }
            ],
        )
    finally:
        await server.stop()
    assert "error" in missing and "error" in unknown
    assert reply["chunk_keys"] == [key]
    assert lookup["chunks"][0]["compressed_slots"] == [
        {"slot_id": start, "mode": "raw", "stored_length": 2 * RECORD},
        {"slot_id": start + 1, "mode": "compressed", "stored_length": RECORD},
    ]
    assert loaded["data"] == data
    assert 'daser_packed_record_ratio_count{mode="compressed"} 1' in (
        registry.render_prometheus()
    )


@pytest.mark.asyncio
async def test_packed_lookup_run_lists_every_chunk_slot(tmp_path: Path) -> None:
    """A lookup run merging adjacent packed chunks lists all of their slots."""
    server = make_ipc_server(tmp_path, storage_format="compressed-online")
    socket_path = sock(tmp_path)
    tokens = list(range(8))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    sizes = (RECORD, 2 * RECORD)
    await server.start()
    try:
        starts = []
        parent = ""
        for key, nbytes in zip(keys, sizes, strict=True):
            start = (await alloc(socket_path, key, parent=parent))["start_slot"]
            starts.append(start)
            span = {
                "chunk_key": key,
                "start_slot": start,
                "num_slots": 1,
                "source_offset": 0,
                "nbytes": nbytes,
                "slot_index": 0,
                "packed": True,
                "mode": "compressed",
            }
            await store(socket_path, [span], pattern(len(starts), nbytes))
            parent = key
        await drain(socket_path)
        lookup = await _send_recv(
            socket_path, {"op": "lookup", "tokens": tokens, "model_id": "m"}
        )
    finally:
        await server.stop()
    (run,) = lookup["chunks"]
    assert (run["start_slot"], run["num_slots"]) == (starts[0], 2)
    assert [
        (slot["slot_id"], slot["stored_length"]) for slot in run["compressed_slots"]
    ] == [(starts[0], RECORD), (starts[1], 2 * RECORD)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"tp_size": 2},
        {"tp_rank": 1},
        {"spans": [{**raw_span("k", 0), "chunk_key": ""}]},
        {"spans": [{**raw_span("k", 0), "nbytes": RECORD // 2}]},
        {"spans": [{**raw_span("k", 0), "slot_index": 1}]},
        {"spans": [{**raw_span("k", 0), "source_offset": -1}]},
        {"spans": [raw_span("k", 0), {**raw_span("k", 0), "packed": True}]},
    ],
)
async def test_malformed_store_requests_fail(
    tmp_path: Path, bad: dict[str, Any]
) -> None:
    core = make_core(64, 16)
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        start = (await alloc(socket_path, "k"))["start_slot"]
        assert start == 0
        msg = {
            "op": "transfer_store",
            "payload": {"data": pattern(1)},
            "spans": [raw_span("k", 0)],
            "tp_rank": 0,
            "tp_size": 1,
            **bad,
        }
        reply = await _send_recv(socket_path, msg)
    finally:
        await server.stop()
    assert "error" in reply
    assert not core.is_chunk_committed("k")


@pytest.mark.asyncio
async def test_raw_store_without_local_slot_size_fails(tmp_path: Path) -> None:
    """A server configured without a raw record size rejects raw stores."""
    core = make_core(64, 16)
    server = make_ipc_server(tmp_path, core=core, local_slot_size=0)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        start = (await alloc(socket_path, "k"))["start_slot"]
        reply = await store(socket_path, [raw_span("k", start)], pattern(1))
    finally:
        await server.stop()
    assert "local_slot_size" in reply["error"]
    assert not core.is_chunk_committed("k")


@pytest.mark.asyncio
async def test_failed_l2_write_of_l2_only_chunk_fails_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store without an L1 copy replies after L2 and reports its failure."""
    fake = FakeTransfer()
    fake.persist_error = OSError("disk gone")
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    core = make_core(64, 1)
    server = make_ipc_server(tmp_path, l1_records=1, core=core)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        kept = await alloc_and_store(core, "kept")
        lease = core.pin_lease(kept.chunk_key)  # L1 stays full of a lease
        start = (await alloc(socket_path, "k"))["start_slot"]
        reply = await store(socket_path, [raw_span("k", start)], pattern(1))
        assert lease is not None
        core.unpin_lease(lease)
        retry = await alloc(socket_path, "k")
    finally:
        await server.stop()
    assert "disk gone" in reply["error"]
    assert not core.is_chunk_committed("k")
    assert retry["skipped"] is False


@pytest.mark.asyncio
async def test_failed_l2_write_keeps_the_l1_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With an L1 copy the store succeeds; a failed L2 write only drops L2."""
    fake = FakeTransfer()
    fake.persist_error = OSError("disk gone")
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    core = make_core(64, 16)
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        start = (await alloc(socket_path, "k"))["start_slot"]
        reply = await store(socket_path, [raw_span("k", start)], pattern(1))
        drained = await _send_recv(socket_path, {"op": "transfer_drain"})
    finally:
        await server.stop()
    assert reply["chunk_keys"] == ["k"]
    assert drained == {"ok": True}
    assert core.is_chunk_committed("k")
    assert core.full_tiers("k") == (L1,)
    assert "L2 write of" in caplog.text


# ----------------------------------------------------------------------
# Readahead and prefetch leases
# ----------------------------------------------------------------------


async def evict_from_l1(
    socket_path: str, core: ServerCore, keys: list[str], seed: int = 100
) -> None:
    """Store filler chunks until every key in ``keys`` is L2-only."""
    while any(core.full_tiers(key) != (L2,) for key in keys):
        seed += 1
        await store_raw_chunk(socket_path, f"filler-{seed}", pattern(seed))


async def store_chain(socket_path: str, tokens: list[int]) -> list[dict[str, Any]]:
    """Store one single-block chunk per block of ``tokens``."""
    allocations = []
    parent = ""
    for index, key in enumerate(rolling_prefix_keys(tokens, BLOCK_TOKENS)):
        allocations.append(
            await store_raw_chunk(socket_path, key, pattern(index + 1), parent)
        )
        parent = key
    return allocations


@pytest.mark.asyncio
async def test_exclusive_tiers_demote_cold_l1_chunks_to_l2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full L1 demotes its coldest chunk; its bytes then load from L2."""
    monkeypatch.setenv("DASER_EXCLUSIVE_TIERS", "1")
    monkeypatch.setenv("DASER_EXCLUSIVE_CLEAN", str(RECORD))
    registry = MetricsRegistry()
    core = make_core(64, 2, registry=registry)
    server = make_ipc_server(tmp_path, l1_records=2, core=core, registry=registry)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        a = await store_raw_chunk(socket_path, "a", pattern(1))
        assert core.full_tiers("a") == (L1,)
        await store_raw_chunk(socket_path, "b", pattern(2))
        # L1 is full of dirty copies: the coldest one was written to L2.
        assert core.full_tiers("a") == (L1, L2)
        assert core.full_tiers("b") == (L1,)
        await store_raw_chunk(socket_path, "c", pattern(3))
        tiers = {key: core.full_tiers(key) for key in ("a", "b", "c")}
        reply = await load(socket_path, [load_span(a["start_slot"])])
        await drain(socket_path)
        loaded = {key: core.full_tiers(key) for key in ("a", "b", "c")}
    finally:
        await server.stop()
    # c took a's clean L1 extent; b was demoted next.
    assert tiers == {"a": (L2,), "b": (L1, L2), "c": (L1,)}
    assert (reply["l1_segments"], reply["l2_segments"]) == (0, 1)
    assert reply["data"] == pattern(1)
    # The read promoted a over b's clean copy and kept a's L2 copy; c was
    # demoted while the promotion was still in flight.
    assert loaded == {"a": (L1, L2), "b": (L2,), "c": (L1, L2)}
    rendered = registry.render_prometheus()
    assert 'daser_tier_demotions_total{outcome="ok"} 3.0' in rendered


@pytest.mark.asyncio
async def test_lookup_readahead_promotes_the_external_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lookup hits promote L2-only chunks of the window in pieces."""
    monkeypatch.setenv("DASER_LOAD_READAHEAD", str(RECORD))
    monkeypatch.setenv("DASER_LOAD_READAHEAD_WINDOW", "4")
    registry = MetricsRegistry()
    core = make_core(64, 2, registry=registry)
    server = make_ipc_server(tmp_path, l1_records=2, core=core, registry=registry)
    socket_path = sock(tmp_path)
    tokens = list(range(8))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    await server.start()
    try:
        await store_chain(socket_path, tokens)
        await evict_from_l1(socket_path, core, keys)
        reply = await _send_recv(
            socket_path,
            {
                "op": "lookup",
                "tokens": tokens + [99],
                "model_id": "m",
                "external_prefix_queries": 9,
                "num_computed_tokens": 0,
            },
        )
        await drain(socket_path)
        tiers = [core.full_tiers(key) for key in keys]
    finally:
        await server.stop()
    assert len(reply["chunks"]) == 1  # one run of two chunks
    assert tiers == [(L1, L2), (L1, L2)]
    rendered = registry.render_prometheus()
    assert (
        f'daser_readahead_bytes_total{{outcome="read"}} {float(2 * RECORD)}' in rendered
    )


@pytest.mark.asyncio
async def test_prefetch_lease_promotes_and_pins_its_window(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """lookup_prefetch pins every chunk; transfer_prefetch brings it to L1."""
    registry = MetricsRegistry()
    core = make_core(64, 2, registry=registry)
    server = make_ipc_server(tmp_path, l1_records=2, core=core, registry=registry)
    socket_path = sock(tmp_path)
    tokens = list(range(8))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    query = {
        "op": "lookup_prefetch",
        "tokens": tokens + [99],
        "model_id": "m",
        "external_prefix_queries": 9,
        "num_computed_tokens": 0,
    }
    await server.start()
    try:
        allocations = await store_chain(socket_path, tokens)
        await evict_from_l1(socket_path, core, keys[:1])
        start = allocations[0]["start_slot"]
        assert allocations[1]["start_slot"] == start + 1
        if core.full_tiers(keys[1]) == (L2,):
            await load(socket_path, [load_span(start + 1)])  # promote block 1
        assert L1 in core.full_tiers(keys[1])

        leased = await _send_recv(socket_path, {**query, "lease_id": "req"})
        prefetch = await _send_recv(
            socket_path, {"op": "transfer_prefetch", "lease_id": "req"}
        )
        # The leased chunks survive L1 pressure until the release.
        await store_raw_chunk(socket_path, "pressure", pattern(50))
        tiers = [core.full_tiers(key) for key in keys]
        loaded = await load(socket_path, [load_span(start, 2)])
        released = await _send_recv(
            socket_path, {"op": "release_transfer_lease", "lease_id": "req"}
        )
        gone = await _send_recv(
            socket_path, {"op": "transfer_prefetch", "lease_id": "req"}
        )
    finally:
        await server.stop()
    assert leased["tier"] == "mixed"
    assert leased["spans"] == [
        {"start_slot": start, "num_slots": 2, "nbytes": 2 * RECORD}
    ]
    assert prefetch == {
        "ok": True,
        "requested_bytes": 2 * RECORD,
        "l1_bytes": RECORD,
        "l2_bytes": RECORD,
    }
    assert all(L1 in tier for tier in tiers)
    assert (loaded["l1_segments"], loaded["data"]) == (2, pattern(1) + pattern(2))
    assert released == {"ok": True}
    assert gone == {"error": "transfer lease 'req' is not active"}
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    rendered = registry.render_prometheus()
    assert 'daser_prefetch_operations_total{status="ok"} 1.0' in rendered
    assert 'daser_prefetch_operations_total{status="incomplete"} 1.0' in rendered


@pytest.mark.asyncio
async def test_exclusive_prefetch_demotes_dirty_l1_chunks_to_make_room(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An L1 full of dirty chunks is demoted so a lease can promote."""
    monkeypatch.setenv("DASER_EXCLUSIVE_TIERS", "1")
    monkeypatch.setenv("DASER_EXCLUSIVE_CLEAN", "0")
    registry = MetricsRegistry()
    core = make_core(64, 2, registry=registry)
    server = make_ipc_server(tmp_path, l1_records=2, core=core, registry=registry)
    socket_path = sock(tmp_path)
    tokens = list(range(8))
    keys = rolling_prefix_keys(tokens, BLOCK_TOKENS)
    await server.start()
    try:
        for index, key in enumerate(("p", "q")):
            await store_raw_chunk(socket_path, key, pattern(40 + index))
        # L1 holds two dirty chunks and keeps no clean room: the chain is
        # stored to L2 only.
        allocations = await store_chain(socket_path, tokens)
        before = [core.full_tiers(key) for key in (*keys, "p", "q")]
        leased = await _send_recv(
            socket_path,
            {
                "op": "lookup_prefetch",
                "lease_id": "req",
                "tokens": tokens + [99],
                "model_id": "m",
                "external_prefix_queries": 9,
                "num_computed_tokens": 0,
            },
        )
        prefetch = await _send_recv(
            socket_path, {"op": "transfer_prefetch", "lease_id": "req"}
        )
        loaded = await load(socket_path, [load_span(allocations[0]["start_slot"], 2)])
        await _send_recv(
            socket_path, {"op": "release_transfer_lease", "lease_id": "req"}
        )
        after = [core.full_tiers(key) for key in (*keys, "p", "q")]
    finally:
        await server.stop()
    assert before == [(L2,), (L2,), (L1,), (L1,)]
    assert leased["tier"] == "l2"
    assert prefetch["ok"] is True and prefetch["l2_bytes"] == 2 * RECORD
    assert (loaded["l1_segments"], loaded["data"]) == (2, pattern(1) + pattern(2))
    # p and q were demoted, then their clean L1 copies made room; the
    # promoted chain keeps its L2 copies.
    assert after == [(L1, L2), (L1, L2), (L2,), (L2,)]
    assert core.demotion_deficit() == 0
    rendered = registry.render_prometheus()
    assert 'daser_prefetch_operations_total{status="ok"} 1.0' in rendered
    assert "daser_prefetch_room_timeouts_total" not in rendered


@pytest.mark.asyncio
async def test_exclusive_prefetch_without_l1_room_times_out_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A lease larger than L1 gives up after the room wait; loads still work."""
    monkeypatch.setenv("DASER_EXCLUSIVE_TIERS", "1")
    monkeypatch.setenv("DASER_EXCLUSIVE_CLEAN", "0")
    monkeypatch.setattr(ipc_server_module, "_LEASE_ROOM_WAIT_S", 0.3)
    registry = MetricsRegistry()
    core = make_core(64, 1, registry=registry)
    server = make_ipc_server(tmp_path, l1_records=1, core=core, registry=registry)
    socket_path = sock(tmp_path)
    tokens = list(range(8))
    await server.start()
    try:
        await store_raw_chunk(socket_path, "p", pattern(40))
        allocations = await store_chain(socket_path, tokens)
        leased = await _send_recv(
            socket_path,
            {
                "op": "lookup_prefetch",
                "lease_id": "req",
                "tokens": tokens + [99],
                "model_id": "m",
                "external_prefix_queries": 9,
                "num_computed_tokens": 0,
            },
        )
        prefetch = await _send_recv(
            socket_path, {"op": "transfer_prefetch", "lease_id": "req"}
        )
        await _send_recv(
            socket_path, {"op": "release_transfer_lease", "lease_id": "req"}
        )
        loaded = await load(socket_path, [load_span(allocations[0]["start_slot"], 2)])
    finally:
        await server.stop()
    assert leased["tier"] == "l2"
    assert prefetch.get("ok") is not True
    assert loaded["data"] == pattern(1) + pattern(2)
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    assert "daser_prefetch_room_timeouts_total 1.0" in registry.render_prometheus()


@pytest.mark.asyncio
async def test_prefetch_lease_waits_for_the_lease_budget(tmp_path: Path) -> None:
    """A lease that does not fit the budget waits for an earlier release."""
    core = make_core(64, 1)
    server = make_ipc_server(
        tmp_path, l1_records=1, core=core, prefetch_lease_budget_bytes=RECORD
    )
    socket_path = sock(tmp_path)
    prompts = [[1, 2, 3, 4], [5, 6, 7, 8]]
    keys = [first_rolling_key(tokens) for tokens in prompts]
    await server.start()
    try:
        for index, key in enumerate(keys):
            await store_raw_chunk(socket_path, key, pattern(index + 1))
        await evict_from_l1(socket_path, core, keys)
        for lease_id, tokens in zip(("a", "b"), prompts, strict=True):
            leased = await _send_recv(
                socket_path,
                {
                    "op": "lookup_prefetch",
                    "lease_id": lease_id,
                    "tokens": tokens + [99],
                    "model_id": "m",
                    "external_prefix_queries": 5,
                    "num_computed_tokens": 0,
                },
            )
            assert leased["tier"] == "l2"
        first = await _send_recv(
            socket_path, {"op": "transfer_prefetch", "lease_id": "a"}
        )
        waiting = asyncio.create_task(
            _send_recv(socket_path, {"op": "transfer_prefetch", "lease_id": "b"})
        )
        await asyncio.sleep(0.1)
        assert not waiting.done()
        await _send_recv(socket_path, {"op": "release_transfer_lease", "lease_id": "a"})
        second = await asyncio.wait_for(waiting, 5)
        await _send_recv(socket_path, {"op": "release_transfer_lease", "lease_id": "b"})
    finally:
        await server.stop()
    assert first["l2_bytes"] == RECORD
    assert second["l2_bytes"] == RECORD
    assert L1 in core.full_tiers(keys[1])


@pytest.mark.asyncio
async def test_prefetch_without_lookup_lease_is_a_recoverable_error(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    registry = MetricsRegistry()
    server = make_ipc_server(tmp_path, registry=registry)
    await server.start()
    try:
        reply = await _send_recv(
            sock(tmp_path), {"op": "transfer_prefetch", "lease_id": "req-1"}
        )
    finally:
        await server.stop()
    assert reply == {"error": "transfer lease 'req-1' is not active"}
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    rendered = registry.render_prometheus()
    assert 'daser_prefetch_operations_total{status="incomplete"} 1.0' in rendered


async def leased_chain_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[IPCServer, ServerCore, FakeTransfer, list[int], FakeOpened]:
    """Start a server with a fake transfer, a two-block chain leased as ``req``.

    Returns:
        ``(server, core, transfer, chunk start slots, registered buffer)``.
    """
    fake = FakeTransfer()
    opened = FakeOpened(4 * RECORD, marker=7)
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    monkeypatch.setattr(
        "daser.server.ipc.server.open_cuda_ipc_buffer", lambda **_kwargs: opened
    )
    core = make_core(64, 16)
    server = make_ipc_server(tmp_path, core=core)
    tokens = list(range(8))
    starts = []
    parent = ""
    for key in rolling_prefix_keys(tokens, BLOCK_TOKENS):
        starts.append((await alloc_and_store(core, key, parent_key=parent)).start_slot)
        parent = key
    await server.start()
    registered = await _send_recv(
        sock(tmp_path),
        {"op": "register_load_staging", "payload": staging_payload(1, 4 * RECORD)},
    )
    assert registered == {"ok": True}
    leased = await _send_recv(
        sock(tmp_path),
        {
            "op": "lookup_prefetch",
            "lease_id": "req",
            "tokens": tokens + [99],
            "model_id": "m",
            "external_prefix_queries": 9,
            "num_computed_tokens": 0,
        },
    )
    assert leased["tier"] == "l1"
    return server, core, fake, starts, opened


def runs_bytes(
    slots: list[int], intra: list[int], targets: list[int], lengths: list[int]
) -> bytes:
    """Pack a ``(4, n)`` int64 run table."""
    return np.array([slots, intra, targets, lengths], dtype=np.int64).tobytes()


@pytest.mark.asyncio
async def test_load_runs_maps_slots_to_leased_l1_offsets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run table becomes one load_runs call at the chunks' L1 offsets."""
    server, core, fake, starts, opened = await leased_chain_server(
        tmp_path, monkeypatch
    )
    try:
        planned = await core.plan_load(starts[0], 2)
        lanes = [segment.lane_offset for segment in planned]
        assert all(segment.tier == L1 for segment in planned)
        core.release_load(planned)
        reply = await _send_recv(
            sock(tmp_path),
            {
                "op": "transfer_load_runs",
                "payload": {"load_staging_buffer_index": 1, "producer_pid": PRODUCER},
                "lease_ids": ["req"],
                "tp_rank": 0,
                "runs": runs_bytes(
                    [starts[1], starts[0]], [64, 0], [0, 512], [256, 512]
                ),
            },
        )
    finally:
        await server.stop()
    assert reply["ok"] is True and reply["bytes"] == 768
    ((buffer, offsets, targets, lengths),) = fake.runs
    layout = core.tier_layout
    assert offsets == [
        layout.absolute_offset(L1, 0, lanes[1] + 64),
        layout.absolute_offset(L1, 0, lanes[0]),
    ]
    assert (targets, lengths) == ([0, 512], [256, 512])
    assert bytes(buffer[:4]) == bytes(opened.array[:4])
    assert opened.closed == 1


@pytest.mark.asyncio
async def test_load_runs_reads_records_of_several_leases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One run table may name slots of every lease of the step."""
    server, core, fake, starts, _opened = await leased_chain_server(
        tmp_path, monkeypatch
    )
    tokens = [50, 51, 52, 53]
    other = (await alloc_and_store(core, first_rolling_key(tokens))).start_slot
    try:
        leased = await _send_recv(
            sock(tmp_path),
            {
                "op": "lookup_prefetch",
                "lease_id": "req-2",
                "tokens": tokens + [99],
                "model_id": "m",
                "external_prefix_queries": 5,
                "num_computed_tokens": 0,
            },
        )
        assert leased["tier"] == "l1"
        planned = await core.plan_load(other, 1)
        lane = planned[0].lane_offset
        core.release_load(planned)
        reply = await _send_recv(
            sock(tmp_path),
            {
                "op": "transfer_load_runs",
                "payload": {"load_staging_buffer_index": 1, "producer_pid": PRODUCER},
                "lease_ids": ["req", "req-2"],
                "tp_rank": 0,
                "runs": runs_bytes([starts[0], other], [0, 0], [0, 512], [512, 512]),
            },
        )
    finally:
        await server.stop()
    assert reply["ok"] is True and reply["bytes"] == 1024
    ((_buffer, offsets, _targets, _lengths),) = fake.runs
    assert offsets[1] == core.tier_layout.absolute_offset(L1, 0, lane)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        {"payload": {}},
        {"runs": b""},
        {"runs": b"\0" * 24},
        {"lease_ids": ["other"]},
        {"lease_ids": []},
        {"lease_ids": ["req", "other"]},
        {"slots": [999]},
        {"intra": [RECORD - 8]},
        {"intra": [-8]},
    ],
)
async def test_load_runs_rejects_malformed_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad: dict[str, Any]
) -> None:
    """Missing buffers or leases and runs outside their records fail."""
    server, _core, fake, starts, _opened = await leased_chain_server(
        tmp_path, monkeypatch
    )
    slots = bad.pop("slots", [starts[0]])
    intra = bad.pop("intra", [0])
    msg = {
        "op": "transfer_load_runs",
        "payload": {"load_staging_buffer_index": 1, "producer_pid": PRODUCER},
        "lease_ids": ["req"],
        "tp_rank": 0,
        "runs": runs_bytes(slots, intra, [0], [16]),
        **bad,
    }
    try:
        reply = await _send_recv(sock(tmp_path), msg)
    finally:
        await server.stop()
    assert "error" in reply
    assert fake.runs == []


# ----------------------------------------------------------------------
# Transfer backend selection and lifecycle
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skip_l2_stores_and_loads_through_l1_only(tmp_path: Path) -> None:
    """Without L2 the io_uring layer keeps chunks in L1 only."""
    core = make_core(layout=make_layout(l2_records=0, l1_records=8))
    server = make_ipc_server(tmp_path, core=core)
    socket_path = sock(tmp_path)
    await server.start()
    try:
        a = await store_raw_chunk(socket_path, "a", pattern(1))
        loaded = await load(socket_path, [load_span(a["start_slot"])])
    finally:
        await server.stop()
    assert core.full_tiers("a") == (L1,)
    assert loaded["data"] == pattern(1)
    assert not (tmp_path / "daser.store").exists()


@pytest.mark.asyncio
async def test_iouring_transfer_is_sized_from_the_tier_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    class Recorder(FakeTransfer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            created.append((args, kwargs))

    monkeypatch.setattr("daser.server.ipc.server.TieredIOUringTransferLayer", Recorder)
    monkeypatch.setenv("DASER_L2_WRITE_GATE", "0")
    server = make_ipc_server(tmp_path, 64, 16, l2_read_stripe_bytes=8192)
    await server.initialize_transfer()
    await server.close()
    assert created == [
        (
            (str(tmp_path / "daser.store"),),
            {
                "l1_bytes": 16 * RECORD,
                "staging_bytes": 4 * RECORD,
                "l2_bytes": 64 * RECORD,
                "skip_l2": False,
                "l2_read_stripe_bytes": 8192,
                "l2_write_gate": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_transfer_mode_must_match_the_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """gds needs L2 only; iouring needs an L1 tier."""
    created: list[tuple[Any, ...]] = []

    class Recorder(FakeTransfer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            created.append((*args, kwargs))

    monkeypatch.setattr("daser.transfer.gds.GDSTransferLayer", Recorder)
    gds_with_l1 = make_ipc_server(tmp_path / "a", transfer_mode="gds")
    gds = make_ipc_server(tmp_path / "b", l1_records=0, transfer_mode="gds")
    iouring_without_l1 = make_ipc_server(tmp_path / "c", l1_records=0)
    with pytest.raises(ValueError, match="gds"):
        await gds_with_l1.initialize_transfer()
    with pytest.raises(ValueError, match="L1"):
        await iouring_without_l1.initialize_transfer()
    await gds.initialize_transfer()
    await gds.close()
    assert created == [(str(tmp_path / "b" / "daser.store"), {"l2_bytes": 64 * RECORD})]


@pytest.mark.asyncio
async def test_stop_accepting_closes_listener_before_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeTransfer()
    monkeypatch.setattr(
        "daser.server.ipc.server.TieredIOUringTransferLayer", lambda *a, **k: fake
    )
    server = make_ipc_server(tmp_path)
    await server.start()
    assert await _send_recv(sock(tmp_path), {"op": "init_transfer"}) == {"ok": True}
    await server.stop_accepting()
    assert not os.path.exists(sock(tmp_path))
    assert fake.events == []
    await server.close()
    assert fake.events == ["drain", "close"]


@pytest.mark.asyncio
async def test_stop_closes_idle_persistent_clients(tmp_path: Path) -> None:
    """Persistent connector streams cannot keep the listener alive on stop."""
    from daser.ipc_protocol import read_frame, write_frame

    server = make_ipc_server(tmp_path)
    await server.start()
    reader, writer = await asyncio.open_unix_connection(sock(tmp_path))
    try:
        await write_frame(writer, {"op": "get_runtime_config"})
        assert "runtime_config" in await read_frame(reader)
        await asyncio.wait_for(server.stop(), 2)
        assert await asyncio.wait_for(reader.read(), 1) == b""
        await asyncio.wait_for(server.stop(), 2)
    finally:
        writer.close()
        await writer.wait_closed()
        await server.stop()


@pytest.mark.asyncio
async def test_init_transfer_request_does_not_block_other_clients(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow transfer initialization keeps serving other requests."""
    release = threading.Event()

    class SlowTransfer(FakeTransfer):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            assert release.wait(timeout=10.0)
            super().__init__()

    monkeypatch.setattr(
        "daser.server.ipc.server.TieredIOUringTransferLayer", SlowTransfer
    )
    server = make_ipc_server(tmp_path)
    await server.start()
    try:
        eager = asyncio.create_task(server.initialize_transfer())
        await asyncio.sleep(0.05)
        init = asyncio.create_task(_send_recv(sock(tmp_path), {"op": "init_transfer"}))
        await asyncio.sleep(0.05)
        config = await asyncio.wait_for(
            _send_recv(sock(tmp_path), {"op": "get_runtime_config"}), timeout=2.0
        )
        assert "error" not in config
        assert not init.done()
        release.set()
        assert await asyncio.wait_for(init, timeout=5.0) == {"ok": True}
        await asyncio.wait_for(eager, timeout=5.0)
    finally:
        release.set()
        await server.stop()


@pytest.mark.asyncio
async def test_eager_transfer_initialization_creates_one_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    created: list[FakeTransfer] = []

    def factory(*args: Any, **kwargs: Any) -> FakeTransfer:
        created.append(FakeTransfer())
        return created[-1]

    monkeypatch.setattr("daser.server.ipc.server.TieredIOUringTransferLayer", factory)
    server = make_ipc_server(tmp_path)
    await server.start()
    try:
        await server.initialize_transfer()
        reply = await load(sock(tmp_path), [])
    finally:
        await server.stop()
    assert reply["ok"] is True
    assert len(created) == 1


# ----------------------------------------------------------------------
# CUDA IPC payloads and registered staging buffers
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cuda_ipc_load_payload_reuses_open_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[FakeOpened] = []
    calls: list[dict[str, Any]] = []

    def open_buffer(**kwargs: Any) -> FakeOpened:
        calls.append(kwargs)
        opened.append(FakeOpened(1024))
        return opened[-1]

    fake = FakeTransfer()
    monkeypatch.setattr("daser.server.ipc.server.open_cuda_ipc_buffer", open_buffer)
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    server = make_ipc_server(tmp_path)
    request = {
        "op": "transfer_load",
        "payload": {
            "cuda_ipc_handle": b"h" * 64,
            "nbytes": 1024,
            "device_pci_bus_id": "0000:38:00.0",
            "device_ptr": 123456,
            "allocation_base_ptr": 122880,
            "allocation_offset": 576,
            "producer_pid": PRODUCER,
        },
        "spans": [],
        "tp_rank": 0,
    }
    await server.start()
    try:
        first = await _send_recv(sock(tmp_path), request)
        second = await _send_recv(sock(tmp_path), request)
    finally:
        await server.stop()
    assert first["ok"] is True and second["ok"] is True
    assert first["transfer_stats_delta"]["l2_reads"] == 0
    assert calls == [
        {
            "handle": b"h" * 64,
            "nbytes": 1024,
            "pci_bus_id": "0000:38:00.0",
            "local_ptr": None,
            "allocation_offset": 576,
        }
    ]
    assert opened[0].closed == 1


@pytest.mark.asyncio
async def test_registered_load_staging_is_scoped_by_producer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Equal worker-local indexes resolve to each producer's mapping."""
    opened: list[FakeOpened] = []

    def open_buffer(**kwargs: Any) -> FakeOpened:
        opened.append(FakeOpened(kwargs["nbytes"], marker=len(opened) + 1))
        return opened[-1]

    fake = FakeTransfer()
    monkeypatch.setattr("daser.server.ipc.server.open_cuda_ipc_buffer", open_buffer)
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    core = make_core(64, 16)
    server = make_ipc_server(tmp_path, core=core)
    chunk = await alloc_and_store(core, "a")
    await server.start()
    try:
        for producer in (42, 43):
            assert await _send_recv(
                sock(tmp_path),
                {
                    "op": "register_load_staging",
                    "payload": staging_payload(1, RECORD, producer),
                },
            ) == {"ok": True}
        replies = [
            await _send_recv(
                sock(tmp_path),
                {
                    "op": "transfer_load",
                    "payload": {
                        "load_staging_buffer_index": 1,
                        "producer_pid": producer,
                    },
                    "spans": [load_span(chunk.start_slot)],
                    "tp_rank": 0,
                },
            )
            for producer in (42, 43)
        ]
        unknown = await _send_recv(
            sock(tmp_path),
            {
                "op": "transfer_load",
                "payload": {"load_staging_buffer_index": 1, "producer_pid": 44},
                "spans": [load_span(chunk.start_slot)],
                "tp_rank": 0,
            },
        )
    finally:
        await server.stop()
    assert [reply["bytes"] for reply in replies] == [RECORD, RECORD]
    assert [(source, len(moves)) for _buf, moves, source in fake.loads] == [
        ("l1", 1),
        ("l1", 1),
    ]
    assert [bytes(buf[0:1]) for buf, _moves, _source in fake.loads] == [
        bytes(opened[0].array[0:1]),
        bytes(opened[1].array[0:1]),
    ]
    assert "error" in unknown
    assert [mapping.closed for mapping in opened] == [1, 1]


@pytest.mark.asyncio
async def test_registered_store_staging_reuses_regions_until_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Store mappings isolate workers, validate bounds and outlive the IO."""
    opened: list[FakeOpened] = []

    def open_buffer(**kwargs: Any) -> FakeOpened:
        opened.append(FakeOpened(kwargs["nbytes"], marker=len(opened)))
        return opened[-1]

    fake = FakeTransfer()
    monkeypatch.setattr("daser.server.ipc.server.open_cuda_ipc_buffer", open_buffer)
    monkeypatch.setattr(IPCServer, "_ensure_transfer", lambda _self: fake)
    server = make_ipc_server(tmp_path)
    socket_path = sock(tmp_path)

    def staged_store(key: str, start: int, producer: int, offset: int) -> dict:
        return {
            "op": "transfer_store",
            "payload": {"store_staging_buffer_index": 0, "producer_pid": producer},
            "spans": [raw_span(key, start, source_offset=offset)],
            "tp_rank": 0,
            "tp_size": 1,
        }

    await server.start()
    try:
        for producer in (42, 43):
            assert await _send_recv(
                socket_path,
                {
                    "op": "register_store_staging",
                    "payload": staging_payload(0, 2 * RECORD, producer),
                },
            ) == {"ok": True}
        starts = [(await alloc(socket_path, key))["start_slot"] for key in "abc"]
        good = [
            await _send_recv(socket_path, staged_store("a", starts[0], 42, RECORD)),
            await _send_recv(socket_path, staged_store("b", starts[1], 43, 0)),
        ]
        bad = [
            await _send_recv(socket_path, staged_store("c", starts[2], 42, RECORD + 1)),
            await _send_recv(socket_path, staged_store("c", starts[2], 44, 0)),
        ]
    finally:
        await server.stop()
    assert [reply["chunk_keys"] for reply in good] == [["a"], ["b"]]
    assert all("error" in reply for reply in bad)
    assert [data for _move, data in fake.stored] == [
        bytes(opened[0].array[RECORD : 2 * RECORD]),
        bytes(opened[1].array[:RECORD]),
    ]
    assert [mapping.closed for mapping in opened] == [1, 1]


def test_cached_cuda_array_uses_private_copy_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load mappings copy on a private stream, not the default stream."""
    # CPU CI installs an import-only cupy stub without the CUDA namespace.
    pytest.importorskip("cupy.cuda")
    import cupy

    class FakeArray:
        data = type("Data", (), {"ptr": 123})()
        device = type("Device", (), {"id": 0})()

    class Opened:
        array = FakeArray()

        def close(self) -> None:
            pass

    class DeviceContext:
        def __enter__(self) -> "DeviceContext":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(
        cupy.cuda, "Stream", lambda **_kwargs: type("S", (), {"ptr": 456})()
    )
    monkeypatch.setattr(cupy.cuda, "Device", lambda _device: DeviceContext())
    assert _CachedCudaArray(Opened()).copy_stream_ptr == 456
