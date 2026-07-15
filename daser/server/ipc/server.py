# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
import contextlib
from dataclasses import asdict
import os
import threading
import time
from typing import Any

from daser.ipc_protocol import read_frame, write_frame

# First Party
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.server.core import ChunkInfo, ServerCore
from daser.transfer import TransferLayer
from daser.transfer.cuda_ipc import open_cuda_ipc_buffer
from daser.transfer.iouring import TieredIOUringTransferLayer

logger = init_logger(__name__)


_CUDA_IPC_CACHE_LIMIT = 16


def _external_prefix_hits(
    chunks: list[ChunkInfo], num_computed_tokens: int, queries: int
) -> int:
    """Return tokens vLLM will count as external prefix cache hits.

    Args:
        chunks: DaseR lookup chunks.
        num_computed_tokens: tokens vLLM already computed locally.
        queries: vLLM external prefix query token count.

    Returns:
        Contiguous external-prefix hit tokens using vLLM connector semantics.
    """
    covered_until = num_computed_tokens
    for chunk in sorted(chunks, key=lambda item: int(item.target_token_start)):
        target_start = int(chunk.target_token_start)
        target_end = target_start + int(chunk.token_count)
        if target_end <= covered_until:
            continue
        if target_start > covered_until:
            break
        covered_until = target_end
    hits = covered_until - num_computed_tokens
    if hits >= queries:
        hits = queries - 1
    return max(0, min(hits, queries))


def _coalesce_transfer_spans(spans: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Merge adjacent transfer spans without changing byte contents.

    Args:
        spans: transfer spans with source_offset, file_offset, and nbytes.

    Returns:
        Coalesced spans sorted by source and file offset.
    """
    normalized = [
        {
            "source_offset": int(span.get("source_offset", 0)),
            "file_offset": int(span["file_offset"]),
            "nbytes": int(span["nbytes"]),
        }
        for span in spans
        if int(span["nbytes"]) > 0
    ]
    normalized.sort(key=lambda span: (span["source_offset"], span["file_offset"]))
    if not normalized:
        return []

    merged = [normalized[0]]
    for span in normalized[1:]:
        prev = merged[-1]
        prev_source_end = prev["source_offset"] + prev["nbytes"]
        prev_file_end = prev["file_offset"] + prev["nbytes"]
        if (
            span["source_offset"] == prev_source_end
            and span["file_offset"] == prev_file_end
        ):
            prev["nbytes"] += span["nbytes"]
        else:
            merged.append(span)
    return merged


class IPCServer:
    """IPC server over Unix socket + msgpack.

    This server is the internal IPC interface for vLLM DaserConnector. It only
    exposes connector cache operations and delegates all business logic to
    ServerCore.

    Args:
        socket_path: Unix socket path.
        core: shared DaseR server core.
        runtime_config: connector runtime values returned by
            ``get_runtime_config``.

    Async/thread-safety:
        Must be started and stopped from the server asyncio event loop.
    """

    def __init__(
        self,
        socket_path: str,
        core: ServerCore,
        runtime_config: dict[str, Any] | None = None,
        metrics_registry: MetricsRegistry | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._core = core
        self._runtime_config = runtime_config or {}
        self._metrics = metrics_registry or REGISTRY
        self._server: asyncio.AbstractServer | None = None
        self._transfer: TransferLayer | None = None
        self._transfer_lock = threading.Lock()
        self._cuda_ipc_cache: OrderedDict[
            tuple[int, int, int, int | None], "_CachedCudaArray"
        ] = OrderedDict()
        self._load_staging_buffers: dict[tuple[int, int], _CachedCudaArray] = {}
        self._op_handlers: dict[
            str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
        ] = {
            "lookup": self._op_lookup,
            "record_external_prefix_cache": self._op_record_external_prefix_cache,
            "get_runtime_config": self._op_get_runtime_config,
            "alloc_chunk": self._op_alloc_chunk,
            "alloc_chunks": self._op_alloc_chunks,
            "match_and_alloc": self._op_match_and_alloc,
            "commit_chunk": self._op_commit_chunk,
            "commit_chunks": self._op_commit_chunks,
            "commit_stats": self._op_commit_stats,
            "live_allocations": self._op_live_allocations,
            "transfer_drain": self._op_transfer_drain,
            "transfer_prefetch": self._transfer_prefetch,
            "init_transfer": self._op_init_transfer,
            "transfer_store": self._transfer_store,
            "transfer_load": self._transfer_load,
            "register_load_staging": self._register_load_staging,
            "evict_chunk": self._op_evict_chunk,
            "release_chunk_writer": self._op_release_chunk_writer,
        }

    async def start(self) -> None:
        """Start listening on the Unix socket.

        Async/thread-safety:
            Removes a stale socket path and starts an asyncio Unix server.
        """
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        logger.info("[IPC] listening on %s", self._socket_path)

    async def initialize_transfer(self) -> None:
        """Eagerly create the transfer layer off the event loop.

        Offloads the blocking ``_ensure_transfer`` call (which may allocate
        pinned memory pools, open io_uring rings, etc.) to a thread so the
        server is fully provisioned before the first inference request.

        Async/thread-safety:
            Must be called from the server asyncio event loop during startup.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_transfer)

    async def drain_transfer(self) -> None:
        """Wait for server-owned transfer-layer background work.

        Async/thread-safety:
            Runs on the server asyncio event loop. The transfer layer is
            initialized on demand, then its async ``drain`` method is awaited
            when present.
        """
        transfer = self._ensure_transfer()
        await transfer.drain()

    async def stop(self) -> None:
        """Stop the server and remove the socket file.

        Async/thread-safety:
            Closes the asyncio server on the current event loop.
        """
        await self.stop_accepting()
        await self.close()
        logger.info("[IPC] server stopped")

    async def stop_accepting(self) -> None:
        """Stop accepting new IPC connections and remove the socket path.

        Async/thread-safety:
            Closes the asyncio listener on the current event loop. Existing
            transfer resources remain open until ``close`` is called.
        """
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        logger.info("[IPC] server stopped accepting")

    async def close(self) -> None:
        """Drain and close transfer resources owned by the IPC server.

        Async/thread-safety:
            Runs on the server asyncio event loop after new IPC work has been
            rejected.
        """
        if self._transfer is not None:
            await self._transfer.drain()
            self._transfer.close()
            self._transfer = None
        self._close_cuda_ipc_cache()
        logger.info("[IPC] resources closed")

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle one connector connection with multiple frames.

        Args:
            reader: request stream reader.
            writer: response stream writer.

        Async/thread-safety:
            Runs one coroutine per client connection.
        """
        try:
            while True:
                try:
                    msg = await read_frame(reader)
                except asyncio.IncompleteReadError:
                    return
                response = await self._dispatch(msg)
                await write_frame(writer, response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] error handling request: %s", exc)
            try:
                await write_frame(writer, {"error": str(exc)})
            except Exception:
                pass
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _dispatch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one decoded connector request.

        Args:
            msg: request dict containing an ``op`` key.

        Returns:
            Response dict suitable for msgpack encoding.

        Async/thread-safety:
            Calls ServerCore on the same asyncio event loop.
        """
        op = str(msg.get("op", "unknown"))
        ipc_labels = {"op": op}
        started = time.perf_counter()
        status = "error"
        try:
            handler = self._op_handlers.get(op)
            if handler is None:
                return {"error": f"unknown op: {op}"}
            response = await handler(msg)
            ok = response.get("ok", True) is not False and "error" not in response
            status = "ok" if ok else "error"
            return response
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] request failed: %s", exc)
            return {"error": str(exc)}
        finally:
            elapsed = time.perf_counter() - started
            self._metrics.counter(
                "daser_ipc_requests_total",
                "IPC requests by operation and status.",
            ).inc(labels={**ipc_labels, "status": status})
            self._metrics.histogram(
                "daser_ipc_request_duration_seconds",
                "IPC request latency by operation.",
                buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0),
            ).observe(elapsed, labels=ipc_labels)

    async def _op_lookup(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``lookup`` request, recording external prefix counters."""
        chunks = await self._core.lookup(msg["tokens"], msg["model_id"])
        if "external_prefix_queries" in msg:
            queries = int(msg.get("external_prefix_queries", 0))
            await self._core.record_external_prefix_cache(
                queries=queries,
                hits=_external_prefix_hits(
                    chunks,
                    num_computed_tokens=int(msg.get("num_computed_tokens", 0)),
                    queries=queries,
                ),
            )
        return {"chunks": [chunk.to_dict() for chunk in chunks]}

    async def _op_record_external_prefix_cache(
        self, msg: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle a standalone ``record_external_prefix_cache`` request."""
        await self._core.record_external_prefix_cache(
            queries=int(msg.get("queries", 0)),
            hits=int(msg.get("hits", 0)),
        )
        return {"ok": True}

    async def _op_get_runtime_config(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``get_runtime_config`` request."""
        return {"runtime_config": dict(self._runtime_config)}

    async def _op_alloc_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a single ``alloc_chunk`` request."""
        alloc = await self._core.alloc_chunk(
            msg["chunk_key"], int(msg["token_count"]), msg["model_id"]
        )
        return alloc.to_dict(include_chunk_key=False)

    async def _op_alloc_chunks(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a batched ``alloc_chunks`` request."""
        allocs = await self._core.alloc_chunks(
            list(msg.get("chunks", [])), msg["model_id"]
        )
        return {
            "allocations": [alloc.to_dict(include_chunk_key=True) for alloc in allocs]
        }

    async def _op_match_and_alloc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``match_and_alloc`` request."""
        result = await self._core.match_and_alloc(
            msg["tokens"], msg.get("chunk_key", ""), msg["model_id"]
        )
        return result.to_dict()

    async def _op_commit_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a single ``commit_chunk`` request."""
        await self._core.commit_chunk(
            msg["chunk_key"],
            tp_rank=int(msg.get("tp_rank", 0)),
            tp_size=int(msg.get("tp_size", 1)),
        )
        return {"ok": True}

    async def _op_commit_chunks(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a batched ``commit_chunks`` request."""
        for chunk_key in msg.get("chunk_keys", []):
            await self._core.commit_chunk(
                chunk_key,
                tp_rank=int(msg.get("tp_rank", 0)),
                tp_size=int(msg.get("tp_size", 1)),
            )
        return {"ok": True}

    async def _op_commit_stats(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``commit_stats`` request."""
        return {"commit_stats": await self._core.commit_stats()}

    async def _op_live_allocations(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``live_allocations`` request."""
        live = await self._core.live_allocations(list(msg.get("allocations", [])))
        return {"chunk_keys": live}

    async def _op_transfer_drain(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``transfer_drain`` request."""
        transfer = self._transfer
        if transfer is not None:
            await transfer.drain()
        return {"ok": True}

    async def _transfer_prefetch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Promote storage spans into the server-owned host-memory tier.

        Args:
            msg: IPC request containing ``spans`` with file offsets and sizes.

        Returns:
            Requested, L1-resident, and L2-read byte counts.

        Async/thread-safety:
            Runs on the IPC event loop and delegates to the transfer layer's
            asynchronous prefetch capability.
        """
        transfer = self._ensure_transfer()
        result = await transfer.prefetch_bytes_grouped(list(msg.get("spans", [])))
        self._metrics.counter(
            "daser_prefetch_operations_total",
            "Host-tier prefetch operations by result.",
        ).inc(labels={"status": "ok"})
        bytes_counter = self._metrics.counter(
            "daser_prefetch_bytes_total",
            "Host-tier prefetch bytes by tier.",
        )
        bytes_counter.inc(result.requested_bytes, labels={"tier": "requested"})
        bytes_counter.inc(result.l1_bytes, labels={"tier": "l1"})
        bytes_counter.inc(result.l2_bytes, labels={"tier": "l2"})
        return {
            "ok": True,
            "requested_bytes": result.requested_bytes,
            "l1_bytes": result.l1_bytes,
            "l2_bytes": result.l2_bytes,
        }

    async def _op_init_transfer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``init_transfer`` request."""
        self._ensure_transfer()
        return {"ok": True}

    async def _op_evict_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``evict_chunk`` request."""
        await self._core.evict_chunk(msg["chunk_key"])
        return {"ok": True}

    async def _op_release_chunk_writer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``release_chunk_writer`` request."""
        released = await self._core.release_chunk_writer(
            chunk_key=str(msg["chunk_key"]),
            start_slot=int(msg["start_slot"]),
            num_slots=int(msg["num_slots"]),
        )
        return {"released": released}

    async def _transfer_store(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Store one or more spans through the server-owned transfer layer.

        Args:
            msg: IPC request with ``payload`` and ``spans``.

        Returns:
            Response dict with total bytes stored.

        Async/thread-safety:
            Runs on the IPC event loop and awaits transfer-layer operations.
        """
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        started = time.perf_counter()
        backend = str(self._runtime_config.get("transfer_mode", "gds"))
        transfer = self._ensure_transfer()
        total = 0
        stored_chunk_keys: list[str] = []
        buffer = self._payload_buffer(payload)
        try:
            live_spans: list[dict[str, Any]] = []
            for span in spans:
                nbytes = int(span["nbytes"])
                file_offset = int(span["file_offset"])
                chunk_key = str(span.get("chunk_key", ""))
                if chunk_key:
                    live = self._core.is_current_allocation(
                        chunk_key=chunk_key,
                        start_slot=int(span.get("start_slot", -1)),
                        num_slots=int(span.get("num_slots", 0)),
                    )
                    if not live:
                        logger.debug(
                            "[IPC] skip stale transfer_store key=%s offset=%d bytes=%d",
                            chunk_key[:8],
                            file_offset,
                            nbytes,
                        )
                        continue
                    stored_chunk_keys.append(chunk_key)
                live_spans.append(span)

            store_spans = (
                _coalesce_transfer_spans(live_spans)
                if transfer.coalesce_store_spans
                else live_spans
            )
            total = await transfer.store_bytes_grouped(buffer, store_spans)
        finally:
            if isinstance(buffer, _UncachedCudaArray):
                buffer.close()
        self._record_transfer_metrics(
            op="store",
            backend=backend,
            status="ok",
            nbytes=total,
            elapsed_s=time.perf_counter() - started,
        )
        return {"ok": True, "bytes": total, "chunk_keys": stored_chunk_keys}

    async def _transfer_load(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Load one or more spans through the server-owned transfer layer.

        Args:
            msg: IPC request with ``payload`` and ``spans``.

        Returns:
            Response dict with total bytes loaded and optional bytes data.

        Async/thread-safety:
            Runs on the IPC event loop and awaits transfer-layer operations.
        """
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        started_total = time.perf_counter()
        backend = str(self._runtime_config.get("transfer_mode", "gds"))
        transfer = self._ensure_transfer()
        total_size = self._payload_size(payload, spans)
        open_ms = 0.0
        if payload.get("return_data"):
            buffer: Any = bytearray(total_size)
        else:
            open_start = time.perf_counter()
            buffer = self._payload_buffer(payload)
            open_ms = (time.perf_counter() - open_start) * 1000

        total = 0
        load_ms = 0.0
        sync_ms = 0.0
        close_ms = 0.0
        close_one_shot_buffer = isinstance(buffer, _UncachedCudaArray) and (
            "load_staging_buffer_index" not in payload
        )
        try:
            before = asdict(transfer.stats)
            started = time.perf_counter()
            load_start = time.perf_counter()
            total = await transfer.load_bytes_grouped(buffer, spans)
            load_ms = (time.perf_counter() - load_start) * 1000
            synchronize = getattr(buffer, "synchronize", None)
            if synchronize is not None:
                sync_start = time.perf_counter()
                synchronize()
                sync_ms = (time.perf_counter() - sync_start) * 1000
            elapsed_ms = (time.perf_counter() - started) * 1000
            after = asdict(transfer.stats)
            stats_delta = {
                key: int(after.get(key, 0)) - int(before.get(key, 0))
                for key in set(before) | set(after)
            }
            logger.debug(
                "[IPC] transfer_load timing: spans=%d bytes=%d total_size=%d "
                "open_ms=%.3f load_ms=%.3f sync_ms=%.3f elapsed_ms=%.3f "
                "stats_delta=%s",
                len(spans),
                total,
                total_size,
                open_ms,
                load_ms,
                sync_ms,
                elapsed_ms,
                stats_delta,
            )
            response: dict[str, Any] = {"ok": True, "bytes": total}
            if payload.get("return_data"):
                response["data"] = bytes(buffer)
            else:
                response["transfer_ms"] = elapsed_ms
                response["transfer_open_ms"] = open_ms
                response["transfer_load_ms"] = load_ms
                response["transfer_sync_ms"] = sync_ms
                response["transfer_stats_delta"] = stats_delta
            self._record_transfer_metrics(
                op="load",
                backend=backend,
                status="ok",
                nbytes=total,
                elapsed_s=time.perf_counter() - started_total,
            )
            return response
        finally:
            if close_one_shot_buffer:
                close = getattr(buffer, "close", None)
                close_start = time.perf_counter()
                close()
                close_ms = (time.perf_counter() - close_start) * 1000
                logger.info(
                    "[IPC] transfer_load close timing: bytes=%d close_ms=%.3f",
                    total,
                    close_ms,
                )

    async def _register_load_staging(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Register one fixed CUDA load staging buffer for indexed reuse.

        Args:
            msg: IPC request whose payload describes a worker-owned fixed CUDA
                staging allocation.

        Returns:
            ``{"ok": True}`` after the CUDA IPC mapping is open and cached.

        Async/thread-safety:
            Runs on the IPC event loop during worker initialization. Replacing
            an existing index closes the old mapping only after the new mapping
            is available.
        """
        payload = msg.get("payload", {})
        buffer_index = int(payload["buffer_index"])
        producer_pid = int(payload["producer_pid"])
        buffer_key = (producer_pid, buffer_index)
        opened = self._open_cuda_ipc_payload(
            payload=payload,
            nbytes_key="allocation_bytes",
            cache_mapping=False,
        )
        previous = self._load_staging_buffers.get(buffer_key)
        self._load_staging_buffers[buffer_key] = opened
        if previous is not None:
            previous.close()
        return {"ok": True}

    def _record_transfer_metrics(
        self,
        op: str,
        backend: str,
        status: str,
        nbytes: int,
        elapsed_s: float,
    ) -> None:
        """Record transfer operation metrics and GB/s log output.

        Args:
            op: Transfer operation, ``load`` or ``store``.
            backend: Configured transfer backend (used for logging only).
            status: Operation status.
            nbytes: Bytes transferred.
            elapsed_s: Operation latency in seconds.
        """
        labels = {"op": op}
        self._metrics.counter(
            "daser_transfer_operations_total",
            "Transfer operations by operation and status.",
        ).inc(labels={**labels, "status": status})
        self._metrics.counter(
            "daser_transfer_bytes_total",
            "Transfer bytes by operation.",
        ).inc(nbytes, labels=labels)
        self._metrics.histogram(
            "daser_transfer_duration_seconds",
            "Transfer operation latency by operation.",
            buckets=(0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
        ).observe(elapsed_s, labels=labels)
        self._metrics.histogram(
            "daser_transfer_chunk_size_bytes",
            "Transfer size per operation in bytes.",
            buckets=(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456),
        ).observe(nbytes, labels=labels)
        self._record_tier_metrics()
        throughput_gbps = (nbytes / elapsed_s / 1_000_000_000) if elapsed_s > 0 else 0.0
        logger.debug(
            "[IPC] transfer_%s summary backend=%s status=%s bytes=%d "
            "elapsed_ms=%.3f throughput_gbps=%.3f",
            op,
            backend,
            status,
            nbytes,
            elapsed_s * 1000,
            throughput_gbps,
        )

    def _record_tier_metrics(self) -> None:
        """Publish L1 cache metrics and the cumulative L2 read counter."""
        transfer = self._transfer
        if transfer is None:
            return
        stats = transfer.stats
        current_hits = stats.l1_hits
        current_misses = stats.l1_misses
        current_l2_reads = stats.l2_reads
        prev_hits = getattr(self, "_prev_l1_hits", 0)
        prev_misses = getattr(self, "_prev_l1_misses", 0)
        prev_l2_reads = getattr(self, "_prev_l2_reads", 0)
        delta_hits = current_hits - prev_hits
        delta_misses = current_misses - prev_misses
        delta_l2_reads = current_l2_reads - prev_l2_reads
        if delta_hits > 0:
            self._metrics.counter("daser_l1_hits_total", "L1 memory cache hits.").inc(
                delta_hits
            )
        if delta_misses > 0:
            self._metrics.counter(
                "daser_l1_misses_total", "L1 memory cache misses."
            ).inc(delta_misses)
        if delta_l2_reads > 0:
            self._metrics.counter(
                "daser_l2_reads_total", "Reads served from the L2 storage tier."
            ).inc(delta_l2_reads)
        self._prev_l1_hits = current_hits
        self._prev_l1_misses = current_misses
        self._prev_l2_reads = current_l2_reads
        l1_used = transfer.l1_bytes_used
        l1_capacity = int(self._runtime_config.get("l1_size_bytes", 0))
        self._metrics.gauge("daser_l1_bytes_used", "L1 memory cache bytes in use.").set(
            l1_used
        )
        self._metrics.gauge(
            "daser_l1_bytes_capacity", "L1 memory cache total capacity."
        ).set(l1_capacity)

    def _ensure_transfer(self) -> TransferLayer:
        """Return the server-owned transfer layer, creating it on first use.

        Protected by a threading lock so that concurrent calls from the
        eager ``initialize_transfer`` thread-pool path and the event-loop
        IPC request path cannot create duplicate transfer layer instances.
        """
        if self._transfer is not None:
            return self._transfer
        with self._transfer_lock:
            if self._transfer is not None:
                return self._transfer
            mode = str(self._runtime_config.get("transfer_mode", "gds"))
            path = str(self._runtime_config.get("store_path", ""))
            skip_l2 = bool(self._runtime_config.get("skip_l2", False))
            if mode == "gds":
                if skip_l2:
                    raise ValueError("skip_l2 is incompatible with gds transfer")
                from daser.transfer.gds import GDSTransferLayer

                self._transfer = GDSTransferLayer(path)
            elif mode == "iouring":
                l2_bytes = int(
                    self._runtime_config.get(
                        "l2_size_bytes",
                        self._runtime_config.get("total_store_bytes", 0),
                    )
                )
                if l2_bytes <= 0:
                    slot_size = int(self._runtime_config.get("slot_size", 0))
                    total_slots = int(self._runtime_config.get("total_slots", 0))
                    l2_bytes = slot_size * total_slots
                self._transfer = TieredIOUringTransferLayer(
                    path=path,
                    l1_bytes=int(self._runtime_config.get("l1_size_bytes", l2_bytes)),
                    l2_bytes=l2_bytes,
                    skip_l2=skip_l2,
                )
            else:
                raise ValueError(f"unknown transfer_mode: {mode}")
        return self._transfer

    def _payload_buffer(self, payload: dict[str, Any]) -> Any:
        """Return a byte-addressable buffer for an IPC transfer payload."""
        if "data" in payload:
            return bytearray(payload["data"])
        if "load_staging_buffer_index" in payload:
            buffer_index = int(payload["load_staging_buffer_index"])
            producer_pid = int(payload["producer_pid"])
            buffer_key = (producer_pid, buffer_index)
            try:
                return self._load_staging_buffers[buffer_key]
            except KeyError as exc:
                raise ValueError(
                    "unknown load staging buffer: "
                    f"producer_pid={producer_pid} index={buffer_index}"
                ) from exc
        if "cuda_ipc_handle" in payload:
            return self._open_cuda_ipc_payload(payload=payload, nbytes_key="nbytes")
        raise ValueError("transfer payload requires data or cuda_ipc_handle")

    def _open_cuda_ipc_payload(
        self,
        *,
        payload: dict[str, Any],
        nbytes_key: str,
        cache_mapping: bool = True,
    ) -> "_CachedCudaArray":
        """Open or reuse a CUDA IPC payload mapping.

        Args:
            payload: CUDA IPC payload with handle, device, allocation base, and
                offset fields.
            nbytes_key: Payload field that identifies the mapping size.
            cache_mapping: If True, reuse the general CUDA IPC LRU cache. Fixed
                registered staging buffers pass False because their lifetime is
                owned by ``_load_staging_buffers`` and must not be LRU-evicted.

        Returns:
            Cached or one-shot CUDA array wrapper for the mapped allocation.
        """
        local_ptr = None
        producer_pid = int(payload.get("producer_pid", -1))
        device_ptr = int(payload["device_ptr"])
        nbytes = int(payload[nbytes_key])
        device_id = int(payload["device_id"]) if "device_id" in payload else None
        allocation_offset = int(payload.get("allocation_offset", 0))
        allocation_base_ptr = int(
            payload.get("allocation_base_ptr", device_ptr - allocation_offset)
        )
        if producer_pid == os.getpid():
            local_ptr = allocation_base_ptr
        if local_ptr is None and cache_mapping:
            key = (
                producer_pid,
                allocation_base_ptr,
                nbytes + allocation_offset,
                device_id,
            )
            cached = self._cuda_ipc_cache.get(key)
            if cached is None:
                self._evict_cuda_ipc_cache_if_needed()
                opened = open_cuda_ipc_buffer(
                    handle=payload["cuda_ipc_handle"],
                    nbytes=nbytes,
                    device_id=device_id,
                    local_ptr=None,
                    allocation_offset=allocation_offset,
                )
                cached = _CachedCudaArray(opened)
                self._cuda_ipc_cache[key] = cached
            else:
                self._cuda_ipc_cache.move_to_end(key)
            return cached
        opened = open_cuda_ipc_buffer(
            handle=payload["cuda_ipc_handle"],
            nbytes=nbytes,
            device_id=device_id,
            local_ptr=local_ptr,
            allocation_offset=allocation_offset,
        )
        return _UncachedCudaArray(opened)

    def _evict_cuda_ipc_cache_if_needed(self) -> None:
        """Evict one cached CUDA IPC mapping when the cache is full."""
        if len(self._cuda_ipc_cache) < _CUDA_IPC_CACHE_LIMIT:
            return
        _key, cached = self._cuda_ipc_cache.popitem(last=False)
        cached.close()

    def _close_cuda_ipc_cache(self) -> None:
        """Close all cached CUDA IPC mappings."""
        for cached in self._load_staging_buffers.values():
            cached.close()
        self._load_staging_buffers.clear()
        for cached in self._cuda_ipc_cache.values():
            cached.close()
        self._cuda_ipc_cache.clear()

    def _payload_size(
        self, payload: dict[str, Any], spans: list[dict[str, Any]]
    ) -> int:
        """Return destination payload size for transfer_load."""
        if "nbytes" in payload:
            return int(payload["nbytes"])
        max_end = 0
        for span in spans:
            max_end = max(
                max_end,
                int(span.get("target_offset", 0)) + int(span["nbytes"]),
            )
        return max_end


class _CachedCudaArray:
    """Sliceable wrapper for a cached CUDA IPC buffer."""

    def __init__(self, opened: Any) -> None:
        self._opened = opened

    def __getitem__(self, item: Any) -> Any:
        """Return a CuPy array slice."""
        return self._opened.array[item]

    def synchronize(self) -> None:
        """Synchronize CUDA writes issued through the opened array."""
        import cupy
        from cupy.cuda import runtime

        with cupy.cuda.Device(int(self._opened.array.device.id)):
            runtime.streamSynchronize(0)

    def close(self) -> None:
        """Close the CUDA IPC handle."""
        self._opened.close()


class _UncachedCudaArray(_CachedCudaArray):
    """Sliceable wrapper that owns a one-shot CUDA IPC buffer."""
