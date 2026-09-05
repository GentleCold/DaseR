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

from daser.compression import CompressedStoreIndex
from daser.config import (
    STORAGE_FORMAT_COMPRESSED_ONLINE,
    STORAGE_FORMAT_COMPRESSED_READ_ONLY,
)
from daser.ipc_protocol import read_frame, write_frame

# First Party
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.server.core import ChunkInfo, ServerCore
from daser.transfer import TransferLayer
from daser.transfer.cuda_ipc import open_cuda_ipc_buffer
from daser.transfer.iouring import TieredIOUringTransferLayer

logger = init_logger(__name__)

# Keep one online-packed L1 admission bounded.  The worker may batch many
# records to amortize CUDA metadata launches, but retaining a GiB-scale pinned
# allocation until every child io_uring write completes stalls later loads and
# makes the server's asynchronous store path visible in TTFT.  Raw spans keep
# the existing unbounded adjacency coalescing contract.
_PACKED_STORE_COALESCE_BYTES = 128 * 1024 * 1024


_CUDA_IPC_CACHE_LIMIT = 16
_PACKED_IO_ALIGNMENT = 4096


def _packed_physical_slot(span: dict[str, Any]) -> int:
    """Return the DaseR ring slot represented by one packed span.

    Prefix reuse allocates one physical slot per synthetic ``:store:<index>``
    request, while chunk reuse allocates a multi-slot physical range whose
    prompt-relative offset is carried by ``logical_slot_start``.  The packed
    metadata index must use the physical ring slot in both cases.
    """
    logical_slot = int(span.get("logical_slot_start", -1))
    start_slot = int(span.get("start_slot", -1))
    num_slots = int(span.get("num_slots", 0))
    if logical_slot < 0:
        raise ValueError("packed span is missing logical slot metadata")
    if start_slot < 0 or num_slots <= 0:
        # Keep the allocator usable for minimal unit-test spans that predate
        # allocation metadata; production transfers always provide both.
        return logical_slot
    offset = 0 if num_slots == 1 else logical_slot
    if offset < 0 or offset >= num_slots:
        raise ValueError("packed span logical slot is outside its allocation")
    return start_slot + offset


class _OnlinePackedExtentAllocator:
    """Validate online packed extents inside their raw allocation envelope.

    Online records are variable-length, but their source allocation is still
    a fixed raw-stride range owned by the ring allocator.  The worker compacts
    records only inside that allocation before sending the spans here.  Keep
    those offsets instead of moving records into a global append arena: a
    global variable-length arena fragments as ring slots are rewritten and
    can reject a valid exact-capacity workload despite enough total free
    bytes.  Reusing the allocation envelope also preserves the worker's
    adjacent-span coalescing and makes slot reuse overwrite-safe.
    """

    def assign(
        self,
        spans: list[dict[str, Any]],
        capacity: int,
        *,
        local_slot_size: int,
        rank_base: int = 0,
    ) -> list[dict[str, Any]]:
        """Validate and preserve physical ranges for packed spans.

        Args:
            spans: Live transfer spans in source-buffer order.
            capacity: Exclusive byte limit of the L2 store.
            local_slot_size: Raw byte envelope for one rank-local slot.
            rank_base: Byte offset of the current tensor-parallel rank lane.

        Returns:
            Copies of ``spans`` with validated packed ``file_offset`` values.

        Raises:
            ValueError: If packed span metadata or capacity is invalid.
            MemoryError: If a packed span exceeds the L2 capacity or its raw
                allocation envelope.

        Async/thread-safety:
            Pure event-loop bookkeeping with no blocking or suspension.
        """
        if capacity <= 0 or local_slot_size <= 0 or rank_base < 0:
            raise ValueError("invalid packed extent geometry")
        assigned: list[dict[str, Any]] = []
        seen_slots: set[int] = set()
        ranges: list[tuple[int, int]] = []
        for span in spans:
            updated = dict(span)
            if not bool(span.get("packed", False)):
                assigned.append(updated)
                continue
            nbytes = int(span["nbytes"])
            logical_count = int(span.get("logical_slot_count", 0))
            logical_slot = int(span.get("logical_slot_start", -1))
            if nbytes <= 0 or nbytes % _PACKED_IO_ALIGNMENT:
                raise ValueError("online packed span must be positive and aligned")
            if logical_count != 1 or logical_slot < 0:
                raise ValueError("packed span must describe one logical slot")
            physical_slot = _packed_physical_slot(span)
            if physical_slot in seen_slots:
                raise ValueError("packed transfer repeats a logical slot")
            seen_slots.add(physical_slot)

            offset = int(span.get("file_offset", -1))
            allocation_start = rank_base + int(span.get("start_slot", -1)) * (
                local_slot_size
            )
            allocation_slots = int(span.get("num_slots", 0))
            allocation_end = allocation_start + allocation_slots * local_slot_size
            if offset < 0 or offset % _PACKED_IO_ALIGNMENT:
                raise ValueError("online packed span file offset is invalid")
            if allocation_slots <= 0 or allocation_start < rank_base:
                raise ValueError("packed span allocation metadata is invalid")
            if offset + nbytes > capacity:
                raise MemoryError(
                    "online packed store capacity exhausted: "
                    f"need [{offset}, {offset + nbytes}), capacity={capacity}"
                )
            if offset < allocation_start or offset + nbytes > allocation_end:
                raise MemoryError(
                    "online packed span exceeds its raw allocation envelope: "
                    f"range=[{offset}, {offset + nbytes}), "
                    f"allocation=[{allocation_start}, {allocation_end})"
                )
            ranges.append((offset, offset + nbytes))
            updated["file_offset"] = offset
            assigned.append(updated)
        ordered_ranges = sorted(ranges)
        for index, (start, _end) in enumerate(ordered_ranges):
            if index and start < ordered_ranges[index - 1][1]:
                raise ValueError("online packed transfer contains overlapping ranges")
        return assigned


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


def _prefetch_spans_from_chunks(
    chunks: list[ChunkInfo],
    *,
    external_start: int,
    external_tokens: int,
    block_tokens: int,
    slot_size: int,
    tensor_parallel_size: int,
    rank_stride_bytes: int,
) -> list[dict[str, int]]:
    """Translate an admitted external KV window into physical TP-lane spans.

    Args:
        chunks: Server lookup chunks covering the prompt prefix.
        external_start: Token offset where external cache loading begins.
        external_tokens: Number of externally admitted tokens.
        block_tokens: Tokens stored in one logical cache slot.
        slot_size: Aggregate bytes per logical slot across TP ranks.
        tensor_parallel_size: Number of physical rank lanes.
        rank_stride_bytes: Byte distance between adjacent rank lanes.

    Returns:
        Sorted block-aligned physical storage ranges for host-tier admission.
    """
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive for prefetch lookup")
    if external_tokens <= 0 or external_start % block_tokens != 0:
        return []
    if slot_size <= 0 or tensor_parallel_size <= 0:
        raise ValueError("invalid transfer geometry for prefetch lookup")
    if slot_size % tensor_parallel_size:
        raise ValueError("slot_size must divide evenly across tensor-parallel ranks")
    local_slot_size = slot_size // tensor_parallel_size
    external_end = external_start + external_tokens
    spans: list[dict[str, int]] = []
    for chunk in sorted(chunks, key=lambda item: int(item.target_token_start)):
        target_start = int(chunk.target_token_start)
        target_end = target_start + int(chunk.token_count)
        load_start = max(target_start, external_start)
        load_end = min(target_end, external_end)
        load_start = ((load_start + block_tokens - 1) // block_tokens) * block_tokens
        load_end = (load_end // block_tokens) * block_tokens
        if load_end <= load_start:
            continue
        start_slot = int(chunk.start_slot) + (
            (load_start - target_start) // block_tokens
        )
        nbytes = ((load_end - load_start) // block_tokens) * local_slot_size
        for rank in range(tensor_parallel_size):
            spans.append(
                {
                    "file_offset": rank * rank_stride_bytes
                    + start_slot * local_slot_size,
                    "nbytes": nbytes,
                }
            )
    return spans


def _coalesce_transfer_spans(
    spans: list[dict[str, Any]],
    *,
    max_packed_bytes: int | None = None,
) -> list[dict[str, int]]:
    """Merge adjacent transfer spans without changing byte contents.

    Args:
        spans: transfer spans with source_offset, file_offset, and nbytes.
        max_packed_bytes: Optional upper bound for one coalesced packed span.
            Raw spans and individual packed spans larger than the bound are
            unchanged.

    Returns:
        Coalesced spans sorted by source and file offset.
    """
    if max_packed_bytes is not None and max_packed_bytes <= 0:
        raise ValueError("max_packed_bytes must be positive when provided")
    normalized = [
        {
            "source_offset": int(span.get("source_offset", 0)),
            "file_offset": int(span["file_offset"]),
            "nbytes": int(span["nbytes"]),
            "packed": bool(span.get("packed", False)),
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
            and span["packed"] == prev["packed"]
            and (
                not span["packed"]
                or max_packed_bytes is None
                or prev["nbytes"] + span["nbytes"] <= max_packed_bytes
            )
        ):
            prev["nbytes"] += span["nbytes"]
        else:
            merged.append(span)
    return merged


def _assign_online_packed_offsets(
    spans: list[dict[str, Any]],
    next_offset: int,
    capacity: int,
) -> tuple[list[dict[str, Any]], int]:
    """Assign compact physical extents to online packed records.

    Online compression discovers each record length on the worker, so the
    server cannot reserve a raw-stride destination during allocation.  The
    control plane therefore assigns an append-only packed extent immediately
    before transfer.  Logical slot metadata remains unchanged and is still
    used for publication; only the physical file offset is relocated.

    Args:
        spans: Live transfer spans in source-buffer order.
        next_offset: Current append cursor in the packed store file.
        capacity: L2 store capacity in bytes.

    Returns:
        A copy of ``spans`` with packed offsets assigned and the next cursor.

    Raises:
        ValueError: If the cursor or capacity is invalid, or a packed span is
            not alignment-sized.
        MemoryError: If the packed append arena is full.

    Async/thread-safety:
        Pure event-loop CPU bookkeeping.  The IPC server invokes it on its
        single asyncio event loop, so the append cursor is serialized with
        transfer requests without an additional lock.
    """
    if next_offset < 0 or capacity <= 0 or next_offset > capacity:
        raise ValueError("invalid online packed append geometry")
    cursor = next_offset
    assigned: list[dict[str, Any]] = []
    for span in spans:
        updated = dict(span)
        if not bool(span.get("packed", False)):
            assigned.append(updated)
            continue
        nbytes = int(span["nbytes"])
        if nbytes <= 0 or nbytes % _PACKED_IO_ALIGNMENT:
            raise ValueError("online packed span must be positive and aligned")
        cursor = (
            (cursor + _PACKED_IO_ALIGNMENT - 1) // _PACKED_IO_ALIGNMENT
        ) * _PACKED_IO_ALIGNMENT
        if cursor + nbytes > capacity:
            raise MemoryError(
                "online packed store capacity exhausted: "
                f"need [{cursor}, {cursor + nbytes}), capacity={capacity}"
            )
        updated["file_offset"] = cursor
        assigned.append(updated)
        cursor += nbytes
    return assigned, cursor


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
        compressed_store_index: Optional immutable physical slot index for the
            experimental read-only storage format.

    Async/thread-safety:
        Must be started and stopped from the server asyncio event loop.
    """

    def __init__(
        self,
        socket_path: str,
        core: ServerCore,
        runtime_config: dict[str, Any] | None = None,
        metrics_registry: MetricsRegistry | None = None,
        compressed_store_index: CompressedStoreIndex | None = None,
    ) -> None:
        self._socket_path = socket_path
        self._core = core
        self._runtime_config = runtime_config or {}
        self._compressed_store_index = compressed_store_index
        self._metrics = metrics_registry or REGISTRY
        self._server: asyncio.AbstractServer | None = None
        self._transfer: TransferLayer | None = None
        self._transfer_lock = threading.Lock()
        # Compressed-online records are assigned physical extents after the
        # worker has discovered their actual lengths.  The allocator is owned
        # by this server event loop and recycles extents when ring logical
        # slots are overwritten; it is intentionally independent of logical
        # ring-slot allocation.
        self._packed_extent_allocator = _OnlinePackedExtentAllocator()
        self._cuda_ipc_cache: OrderedDict[
            tuple[int, int, int, int | None], "_CachedCudaArray"
        ] = OrderedDict()
        self._load_staging_buffers: dict[tuple[int, int], _CachedCudaArray] = {}
        self._op_handlers: dict[
            str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
        ] = {
            "lookup": self._op_lookup,
            "lookup_prefetch": self._op_lookup_prefetch,
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
            "release_transfer_lease": self._op_release_transfer_lease,
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
        chunks = await self._lookup_core(msg["tokens"], msg["model_id"])
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
        return {"chunks": self._chunk_payloads(chunks)}

    async def _op_lookup_prefetch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Lookup, classify exact external spans, and lease an all-L1 result."""
        lease_id = str(msg.get("lease_id", ""))
        if not lease_id:
            raise ValueError("lookup_prefetch requires lease_id")
        chunks = await self._lookup_core(msg["tokens"], msg["model_id"])
        queries = int(msg.get("external_prefix_queries", 0))
        num_computed_tokens = int(msg.get("num_computed_tokens", 0))
        hits = _external_prefix_hits(chunks, num_computed_tokens, queries)
        await self._core.record_external_prefix_cache(queries=queries, hits=hits)
        if not chunks or hits <= 0:
            return {"chunks": self._chunk_payloads(chunks), "spans": []}

        block_tokens = int(self._runtime_config.get("block_tokens", 0))
        storage_format = self._runtime_config.get("storage_format")
        if storage_format == STORAGE_FORMAT_COMPRESSED_ONLINE:
            spans = self._online_compressed_prefetch_spans(
                chunks,
                external_start=num_computed_tokens,
                external_tokens=hits,
                block_tokens=block_tokens,
            )
        elif self._compressed_store_index is None:
            spans = _prefetch_spans_from_chunks(
                chunks,
                external_start=num_computed_tokens,
                external_tokens=hits,
                block_tokens=block_tokens,
                slot_size=int(self._runtime_config.get("slot_size", 0)),
                tensor_parallel_size=int(
                    self._runtime_config.get("tensor_parallel_size", 1)
                ),
                rank_stride_bytes=int(self._runtime_config.get("rank_stride_bytes", 0)),
            )
        else:
            spans = self._compressed_prefetch_spans(
                chunks,
                external_start=num_computed_tokens,
                external_tokens=hits,
                block_tokens=block_tokens,
            )
        if not spans:
            return {"chunks": self._chunk_payloads(chunks), "spans": []}
        transfer = self._ensure_transfer()
        tier = await transfer.classify_and_acquire_lease(lease_id, spans)
        return {
            "chunks": self._chunk_payloads(chunks),
            "spans": spans,
            "tier": tier,
        }

    async def _lookup_core(self, tokens: list[int], model_id: str) -> list[ChunkInfo]:
        """Run an immediate lookup against the committed retrieval index.

        Args:
            tokens: Prompt token IDs.
            model_id: Model identifier used for cache isolation.

        Returns:
            Retrieval chunks returned by the server core.

        Async/thread-safety:
            Runs on the IPC server event loop. Online stores are published to
            the retrieval index only after their transfer commits; an
            uncommitted record is therefore a safe miss and must not make the
            request wait for a background writer. The compatibility fallback
            is limited to test doubles or older public core implementations
            that do not expose the keyword.
        """
        try:
            return await self._core.lookup(
                tokens,
                model_id,
                wait_for_pending=False,
            )
        except TypeError as exc:
            if "wait_for_pending" not in str(exc):
                raise
            return await self._core.lookup(tokens, model_id)

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
        self._require_writable_storage()
        alloc = await self._core.alloc_chunk(
            msg["chunk_key"], int(msg["token_count"]), msg["model_id"]
        )
        return alloc.to_dict(include_chunk_key=False)

    async def _op_alloc_chunks(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a batched ``alloc_chunks`` request."""
        self._require_writable_storage()
        allocs = await self._core.alloc_chunks(
            list(msg.get("chunks", [])), msg["model_id"]
        )
        return {
            "allocations": [alloc.to_dict(include_chunk_key=True) for alloc in allocs]
        }

    async def _op_match_and_alloc(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``match_and_alloc`` request."""
        self._require_writable_storage()
        result = await self._core.match_and_alloc(
            msg["tokens"], msg.get("chunk_key", ""), msg["model_id"]
        )
        return result.to_dict()

    async def _op_commit_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a single ``commit_chunk`` request."""
        self._require_writable_storage()
        await self._core.commit_chunk(
            msg["chunk_key"],
            tp_rank=int(msg.get("tp_rank", 0)),
            tp_size=int(msg.get("tp_size", 1)),
        )
        return {"ok": True}

    async def _op_commit_chunks(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a batched ``commit_chunks`` request."""
        self._require_writable_storage()
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
        lease_id = str(msg.get("lease_id", "")) or None
        spans = list(msg.get("spans", []))
        if lease_id is None:
            result = await transfer.prefetch_bytes_grouped(spans)
        else:
            result = await transfer.prefetch_bytes_grouped(spans, lease_id=lease_id)
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

    async def _op_release_transfer_lease(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Idempotently release remaining host-tier bytes for one request."""
        lease_id = str(msg.get("lease_id", ""))
        if not lease_id:
            raise ValueError("release_transfer_lease requires lease_id")
        transfer = self._transfer
        if transfer is not None:
            await transfer.release_lease(lease_id)
        return {"ok": True}

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
        self._require_writable_storage()
        transfer = self._ensure_transfer()
        total = 0
        stored_chunk_keys: list[str] = []
        accepted_spans: list[dict[str, Any]] = []
        tp_rank = int(msg.get("tp_rank", 0))
        tp_size = int(msg.get("tp_size", 1))
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

            if self._runtime_config.get(
                "storage_format"
            ) == STORAGE_FORMAT_COMPRESSED_ONLINE and any(
                bool(span.get("packed", False)) for span in live_spans
            ):
                capacity = int(
                    self._runtime_config.get(
                        "l2_size_bytes",
                        self._runtime_config.get("total_store_bytes", 0),
                    )
                )
                if capacity <= 0:
                    capacity = int(self._runtime_config.get("slot_size", 0)) * int(
                        self._runtime_config.get("total_slots", 0)
                    )
                local_slot_size = int(
                    self._runtime_config.get(
                        "local_slot_size",
                        int(self._runtime_config.get("slot_size", 0))
                        // max(1, tp_size),
                    )
                )
                rank_stride_bytes = int(
                    self._runtime_config.get("rank_stride_bytes", 0)
                )
                live_spans = self._packed_extent_allocator.assign(
                    live_spans,
                    capacity,
                    local_slot_size=local_slot_size,
                    rank_base=tp_rank * rank_stride_bytes,
                )

            for span in live_spans:
                chunk_key = str(span.get("chunk_key", ""))
                if not chunk_key:
                    continue
                accepted_spans.append(
                    {
                        "chunk_key": chunk_key,
                        "file_offset": int(span["file_offset"]),
                        "nbytes": int(span["nbytes"]),
                        "start_slot": int(span.get("start_slot", -1)),
                        "num_slots": int(span.get("num_slots", 0)),
                        "logical_slot_start": int(span.get("logical_slot_start", -1)),
                        "logical_slot_count": int(span.get("logical_slot_count", 0)),
                        "packed": bool(span.get("packed", False)),
                        "mode": str(span.get("mode", "compressed")),
                    }
                )

            store_spans = (
                _coalesce_transfer_spans(
                    live_spans,
                    max_packed_bytes=_PACKED_STORE_COALESCE_BYTES,
                )
                if transfer.coalesce_store_spans
                else live_spans
            )
            logger.info(
                "[IPC] transfer_store spans=%d packed=%d dispatched=%d bytes=%d",
                len(live_spans),
                sum(bool(span.get("packed", False)) for span in live_spans),
                len(store_spans),
                sum(int(span["nbytes"]) for span in store_spans),
            )
            total = await transfer.store_bytes_grouped(buffer, store_spans)
        finally:
            if isinstance(buffer, _UncachedCudaArray):
                buffer.close()
        if accepted_spans:
            configured_tp_size = int(
                self._runtime_config.get("tensor_parallel_size", tp_size)
            )
            if configured_tp_size != tp_size:
                raise ValueError(
                    f"transfer TP size {tp_size} != configured {configured_tp_size}"
                )
            slot_size = int(self._runtime_config.get("slot_size", 0))
            local_slot_size = int(
                self._runtime_config.get(
                    "local_slot_size", slot_size // max(1, tp_size)
                )
            )
            rank_stride_bytes = int(self._runtime_config.get("rank_stride_bytes", 0))
            await self._core.record_store_ranges(
                accepted_spans,
                tp_rank=tp_rank,
                tp_size=tp_size,
                local_slot_size=local_slot_size,
                rank_stride_bytes=rank_stride_bytes,
            )
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
        lease_id = str(msg.get("lease_id", "")) or None
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
        leased_load_started = False
        try:
            before = asdict(transfer.stats)
            started = time.perf_counter()
            load_start = time.perf_counter()
            if lease_id is None:
                total = await transfer.load_bytes_grouped(buffer, spans)
            else:
                total = await transfer.load_leased_bytes_grouped(
                    buffer,
                    spans,
                    lease_id,
                )
                leased_load_started = True
            load_ms = (time.perf_counter() - load_start) * 1000
            synchronize = getattr(buffer, "synchronize", None)
            if synchronize is not None:
                sync_start = time.perf_counter()
                # CUDA stream synchronization is a blocking runtime call.  It
                # must finish before releasing a leased L1 range, but waiting
                # on the IPC event loop would serialize unrelated load/store
                # requests behind the slowest destination copy.
                await asyncio.to_thread(synchronize)
                sync_ms = (time.perf_counter() - sync_start) * 1000
            if lease_id is not None:
                await transfer.release_lease_ranges(lease_id, spans)
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
        except BaseException:
            if lease_id is not None:
                if leased_load_started:
                    await transfer.release_lease_ranges(lease_id, [])
                await transfer.release_lease(lease_id)
            raise
        finally:
            if close_one_shot_buffer:
                close = getattr(buffer, "close", None)
                close_start = time.perf_counter()
                if close is not None:
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
                storage_format = self._runtime_config.get("storage_format")
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
                    read_only=(
                        self._compressed_store_index is not None
                        or storage_format == STORAGE_FORMAT_COMPRESSED_READ_ONLY
                    ),
                    coalesce_load_misses=(
                        self._compressed_store_index is not None
                        or storage_format
                        in (
                            STORAGE_FORMAT_COMPRESSED_ONLINE,
                            STORAGE_FORMAT_COMPRESSED_READ_ONLY,
                        )
                    ),
                )
            else:
                raise ValueError(f"unknown transfer_mode: {mode}")
        return self._transfer

    def _chunk_payloads(self, chunks: list[ChunkInfo]) -> list[dict[str, Any]]:
        """Serialize lookup chunks and attach immutable compressed slot refs."""
        payloads: list[dict[str, Any]] = []
        for chunk in chunks:
            payload = chunk.to_dict()
            if self._compressed_store_index is not None:
                refs = self._compressed_store_index.resolve_slots(
                    chunk.start_slot, chunk.num_slots
                )
                payload["compressed_slots"] = [
                    {
                        "slot_id": ref.slot_id,
                        "mode": ref.mode.name.lower(),
                        "file_offset": ref.file_offset,
                        "stored_length": ref.stored_length,
                    }
                    for ref in refs
                ]
            elif (
                self._runtime_config.get("storage_format")
                == STORAGE_FORMAT_COMPRESSED_ONLINE
            ):
                refs = self._core.packed_slot_refs(
                    chunk.start_slot,
                    chunk.num_slots,
                )
                if len(refs) == chunk.num_slots:
                    payload["compressed_slots"] = refs
            payloads.append(payload)
        return payloads

    def _compressed_prefetch_spans(
        self,
        chunks: list[ChunkInfo],
        *,
        external_start: int,
        external_tokens: int,
        block_tokens: int,
    ) -> list[dict[str, int]]:
        """Return indexed compressed spans for an admitted external window."""
        index = self._compressed_store_index
        if index is None:
            raise RuntimeError("compressed prefetch requires a side index")
        external_end = external_start + external_tokens
        spans: list[dict[str, int]] = []
        for chunk in sorted(chunks, key=lambda item: item.target_token_start):
            target_start = chunk.target_token_start
            target_end = target_start + chunk.token_count
            load_start = max(target_start, external_start)
            load_end = min(target_end, external_end)
            load_start = (
                (load_start + block_tokens - 1) // block_tokens
            ) * block_tokens
            load_end = (load_end // block_tokens) * block_tokens
            if load_end <= load_start:
                continue
            start_slot = chunk.start_slot + (
                (load_start - target_start) // block_tokens
            )
            num_slots = (load_end - load_start) // block_tokens
            spans.extend(
                {
                    "file_offset": ref.file_offset,
                    "nbytes": ref.stored_length,
                }
                for ref in index.resolve_slots(start_slot, num_slots)
            )
        return spans

    def _online_compressed_prefetch_spans(
        self,
        chunks: list[ChunkInfo],
        *,
        external_start: int,
        external_tokens: int,
        block_tokens: int,
    ) -> list[dict[str, int]]:
        """Translate an online packed window into exact physical read spans.

        Args:
            chunks: Committed lookup chunks covering the prompt prefix.
            external_start: Token offset where external cache loading begins.
            external_tokens: Number of externally admitted tokens.
            block_tokens: Tokens stored in one logical cache slot.

        Returns:
            Packed file offsets and lengths for the aligned external window.
            An empty list is returned when a committed chunk has not published
            every packed slot yet; callers then skip lease-based prefetch until
            the immutable online metadata is complete.

        Async/thread-safety:
            Runs on the server asyncio event loop and only reads control-plane
            metadata. It performs no I/O or blocking operations.
        """
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive for prefetch lookup")
        if external_tokens <= 0 or external_start % block_tokens != 0:
            return []
        external_end = external_start + external_tokens
        spans: list[dict[str, int]] = []
        for chunk in sorted(chunks, key=lambda item: item.target_token_start):
            target_start = int(chunk.target_token_start)
            target_end = target_start + int(chunk.token_count)
            load_start = max(target_start, external_start)
            load_end = min(target_end, external_end)
            load_start = (
                (load_start + block_tokens - 1) // block_tokens
            ) * block_tokens
            load_end = (load_end // block_tokens) * block_tokens
            if load_end <= load_start:
                continue
            start_slot = int(chunk.start_slot) + (
                (load_start - target_start) // block_tokens
            )
            num_slots = (load_end - load_start) // block_tokens
            refs = self._core.packed_slot_refs(start_slot, num_slots)
            if len(refs) != num_slots:
                return []
            spans.extend(
                {
                    "file_offset": int(ref["file_offset"]),
                    "nbytes": int(ref["stored_length"]),
                }
                for ref in refs
            )
        return spans

    def _require_writable_storage(self) -> None:
        """Reject mutation operations in immutable compressed-read-only mode."""
        if (
            self._runtime_config.get("storage_format")
            == STORAGE_FORMAT_COMPRESSED_READ_ONLY
        ):
            raise ValueError("compressed-read-only storage rejects all writes")

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
        self._copy_stream: Any | None = None

    def __getitem__(self, item: Any) -> Any:
        """Return a CuPy array slice."""
        return self._opened.array[item]

    def synchronize(self) -> None:
        """Synchronize CUDA writes issued through the opened array."""
        import cupy
        from cupy.cuda import runtime

        with cupy.cuda.Device(int(self._opened.array.device.id)):
            stream_ptr = (
                int(self._copy_stream.ptr) if self._copy_stream is not None else 0
            )
            runtime.streamSynchronize(stream_ptr)

    @property
    def copy_stream_ptr(self) -> int:
        """Return the private stream used for asynchronous H2D copies.

        Returns:
            CUDA stream pointer, or zero for non-CUDA test doubles.

        Async/thread-safety:
            Called on the IPC server event loop while a destination mapping is
            being prepared. Stream creation is lazy and each mapping is used
            by one staging lease at a time, so no cross-request locking is
            needed here.
        """
        array = self._opened.array
        if getattr(getattr(array, "data", None), "ptr", None) is None:
            return 0
        if self._copy_stream is None:
            import cupy

            with cupy.cuda.Device(int(array.device.id)):
                self._copy_stream = cupy.cuda.Stream(non_blocking=True)
        return int(self._copy_stream.ptr)

    def close(self) -> None:
        """Close the CUDA IPC handle."""
        self._opened.close()


class _UncachedCudaArray(_CachedCudaArray):
    """Sliceable wrapper that owns a one-shot CUDA IPC buffer."""
