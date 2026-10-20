# SPDX-License-Identifier: Apache-2.0

"""Connector IPC server: cache control plane and server-placed transfers.

The connector names chunks and slot ids only. Every byte location is chosen
here from ``ServerCore`` tier placements and handed to the transfer layer as
absolute ``Move`` offsets:

- ``transfer_store``: ``place_store`` -> ``TransferLayer.store`` ->
  ``store_staged`` / ``store_persisted`` (or ``store_failed``). The reply is
  sent once the worker buffer was consumed; a chunk without an L1 copy is
  replied after its L2 write, so the chunk is published by then.
- ``transfer_load``: ``plan_load`` per chunk -> one ``TransferLayer.load`` per
  source tier and ready batch (a batch is cut before a chunk whose promotion
  is in flight); whole L2-only chunks are promoted into L1 on the same read.
- Lookup readahead promotes the external window's L2-only chunks in the
  background (bounded in flight).
- Prefetch leases pin the L1 copies of a request's window; layer-wise runs
  read them through ``load_runs``.
"""

# Standard
import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Coroutine, Sequence
import contextlib
from dataclasses import asdict, dataclass, field
import os
import threading
import time
from typing import Any

# Third Party
import numpy

# First Party
from daser import stalldump
from daser.config import (
    DEFAULT_LOAD_READAHEAD_BYTES,
    L2_WRITE_GATE_ENV,
    LOAD_READAHEAD_ENV,
    LOAD_READAHEAD_WINDOW_ENV,
    STORAGE_FORMAT_COMPRESSED_ONLINE,
)
from daser.connector.helpers import TokenSequence
from daser.ipc_protocol import read_frame, write_frame
from daser.logging import init_logger
from daser.metrics import REGISTRY, MetricsRegistry
from daser.server.core import ChunkInfo, LoadSegment, ServerCore, StoreRequest
from daser.server.tiers import (
    L1,
    L2,
    ChunkUnavailable,
    Demotion,
    Lease,
    Promotion,
    StorePlacement,
    TierLayout,
    record_offsets,
)
from daser.transfer import Move, TransferLayer
from daser.transfer.cuda_ipc import open_cuda_ipc_buffer
from daser.transfer.iouring import TieredIOUringTransferLayer
from daser.transfer.tasks import wait_shared

logger = init_logger(__name__)

_CUDA_IPC_CACHE_LIMIT = 16
# Exclusive tiers: how long a lease waits for demotions to make L1 room before
# giving up (the request then loads without the lease), and the poll period
# for room freed by anything other than a demotion batch.
_LEASE_ROOM_WAIT_S = 2.0
_LEASE_ROOM_POLL_S = 0.05
_RECORD_MODES = ("raw", "compressed")


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


def _external_load_window(
    target_start: int,
    target_end: int,
    external_start: int,
    external_end: int,
    block_tokens: int,
) -> tuple[int, int]:
    """Return the block-aligned token window the worker loads from one chunk.

    Must match the connector's load planning: the start rounds up, and the end
    rounds up to cover a partially admitted final block (vLLM admits at most
    ``prompt_len - 1`` external tokens) but never past the chunk.

    Args:
        target_start: First prompt token the chunk covers.
        target_end: Token after the last prompt token the chunk covers.
        external_start: Token offset where external loading begins.
        external_end: Token after the last externally admitted token.
        block_tokens: Tokens per KV block.

    Returns:
        ``(load_start, load_end)``; empty when ``load_end <= load_start``.
    """
    load_start = max(target_start, external_start)
    load_end = min(target_end, external_end)
    load_start = ((load_start + block_tokens - 1) // block_tokens) * block_tokens
    load_end = ((load_end + block_tokens - 1) // block_tokens) * block_tokens
    return load_start, min(load_end, target_end)


def _window_slots(
    chunks: list[ChunkInfo],
    external_start: int,
    external_tokens: int,
    block_tokens: int,
) -> list[tuple[ChunkInfo, int, int]]:
    """Return the slot range of each chunk a worker load of a window reads.

    Args:
        chunks: Lookup chunks covering the prompt prefix.
        external_start: Token offset where external loading begins.
        external_tokens: Externally admitted tokens.
        block_tokens: Tokens per slot.

    Returns:
        ``(chunk, first slot index in the chunk, slot count)`` in prompt
        order; empty when the window is empty or not block aligned.

    Raises:
        ValueError: If ``block_tokens`` is not positive.
    """
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive for external windows")
    if external_tokens <= 0 or external_start % block_tokens:
        return []
    external_end = external_start + external_tokens
    windows = []
    for chunk in sorted(chunks, key=lambda item: int(item.target_token_start)):
        target_start = int(chunk.target_token_start)
        load_start, load_end = _external_load_window(
            target_start,
            target_start + int(chunk.token_count),
            external_start,
            external_end,
            block_tokens,
        )
        if load_end > load_start:
            windows.append(
                (
                    chunk,
                    (load_start - target_start) // block_tokens,
                    (load_end - load_start) // block_tokens,
                )
            )
    return windows


def _first_error(futures: Sequence[asyncio.Future[Any]]) -> BaseException | None:
    """Return the first failure among completed futures, or None.

    Args:
        futures: Completed futures.
    """
    for future in futures:
        if future.cancelled():
            return asyncio.CancelledError()
        error = future.exception()
        if error is not None:
            return error
    return None


async def _outcome(futures: Sequence[asyncio.Future[Any]]) -> BaseException | None:
    """Wait for every future without cancelling any; return the first failure.

    Args:
        futures: Futures to wait for; shared futures may appear in several
            callers' lists.

    Returns:
        The first failure in ``futures`` order, or None when all succeeded.

    Async/thread-safety:
        Shields the futures, so cancelling the caller never cancels them; a
        cancellation of the caller is re-raised once they all ended.
    """
    pending = [future for future in futures if not future.done()]
    cancelled: asyncio.CancelledError | None = None
    while pending:
        try:
            await asyncio.shield(asyncio.wait(pending))
        except asyncio.CancelledError as exc:
            cancelled = exc
        pending = [future for future in pending if not future.done()]
    if cancelled is not None:
        raise cancelled
    return _first_error(futures)


def _lease_tables(
    states: Sequence["_LeaseState"],
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    """Merge ready leases' slot tables into one sorted lookup table.

    Args:
        states: Leases whose ``_lease_ready`` task completed.

    Returns:
        ``(slot ids, L1 lane offsets, record sizes)`` sorted by slot id. A
        slot pinned by several leases (a shared prefix chunk) maps to the
        same L1 record, so the first entry is kept.
    """
    tables = []
    for state in states:
        assert state.slot_ids is not None and state.lane_offsets is not None
        assert state.record_sizes is not None
        tables.append((state.slot_ids, state.lane_offsets, state.record_sizes))
    if len(tables) == 1:
        return tables[0]
    slots, lanes, sizes = (
        numpy.concatenate(column) for column in zip(*tables, strict=True)
    )
    slot_ids, first = numpy.unique(slots, return_index=True)
    return slot_ids, lanes[first], sizes[first]


@dataclass(frozen=True)
class _LoadPart:
    """The slots one load span reads from one chunk.

    Attributes:
        chunk_key: Chunk read.
        start_slot: First slot id read.
        num_slots: Slots read.
        target_offset: Destination buffer offset of the first record.
        nbytes: Record bytes read on one rank.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    target_offset: int
    nbytes: int


@dataclass(eq=False)
class _ChunkStore:
    """Store progress of one chunk on one rank across store requests.

    A raw chunk larger than one worker staging batch arrives in several
    requests. Every request places the chunk (placement is idempotent) and
    moves its slots; the request that completes the slot set reports the
    chunk's outcome once every request's IO ended.

    Attributes:
        num_slots: Slots of the allocation.
        slots: Slot indices received so far.
        staged: ``StoreHandle.staged`` of every request carrying the chunk.
        persisted: ``StoreHandle.persisted`` of those requests.
    """

    num_slots: int
    slots: set[int] = field(default_factory=set)
    staged: list[asyncio.Future[None]] = field(default_factory=list)
    persisted: list[asyncio.Future[None]] = field(default_factory=list)


@dataclass(eq=False)
class _StorePart:
    """One chunk carried by one store request.

    Attributes:
        key: ``(chunk_key, start_slot, tp_rank)`` of the chunk store.
        num_slots: Slots of the allocation.
        slots: Slot indices this request writes.
        placement: This request's placement (its own io pins).
        tracker: Progress shared by the chunk's requests; set when the
            request's IO was submitted.
        reporter: Whether this request completed the slot set and reports
            the chunk's outcome; other requests only release their pins.
    """

    key: tuple[str, int, int]
    num_slots: int
    slots: frozenset[int]
    placement: StorePlacement
    tracker: _ChunkStore | None = None
    reporter: bool = False


@dataclass(eq=False)
class _LeaseState:
    """Server state of one request prefetch lease.

    Attributes:
        lease_id: Scheduler request id.
        leases: Tier lease pins, one per leased chunk.
        windows: ``(start_slot, num_slots)`` the request loads, per chunk.
        promote_keys: Chunks that had no L1 copy at lookup.
        nbytes: L1 extent bytes of the leased chunks over all ranks,
            counted against the prefetch lease budget.
        admitted: Whether ``nbytes`` is counted in the budget.
        released: Set by ``release_transfer_lease``.
        task: Promotes ``promote_keys`` and pins ``segments``; shared by
            ``transfer_prefetch`` and the first ``transfer_load_runs``.
        segments: L1 segments of every window, io pinned until release.
        slot_ids: Sorted slot ids of ``segments``.
        lane_offsets: L1 lane offset of each slot id's record.
        record_sizes: Record bytes of each slot id.
        active_loads: Device copies borrowing these segments; release waits
            until every such call has completed its destination synchronization.
    """

    lease_id: str
    leases: list[Lease]
    windows: list[tuple[int, int]]
    promote_keys: list[str]
    nbytes: int
    admitted: bool = False
    released: bool = False
    task: asyncio.Task[None] | None = None
    segments: list[LoadSegment] = field(default_factory=list)
    slot_ids: numpy.ndarray | None = None
    lane_offsets: numpy.ndarray | None = None
    record_sizes: numpy.ndarray | None = None
    active_loads: int = 0


class IPCServer:
    """IPC server over Unix socket + msgpack.

    This server is the internal IPC interface for vLLM DaserConnector. It
    exposes connector cache operations, places every transfer through
    ``ServerCore`` and drives the server-owned transfer layer.

    Args:
        socket_path: Unix socket path.
        core: shared DaseR server core.
        runtime_config: connector runtime values returned by
            ``get_runtime_config``.
        metrics_registry: registry for IPC and transfer metrics.

    Async/thread-safety:
        Must be started, used and stopped from the server asyncio event loop.
        Only ``initialize_transfer`` runs transfer construction on an
        executor thread, guarded by a lock.
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
        self._layout: TierLayout = core.tier_layout
        self._runtime_config = runtime_config or {}
        self._metrics = metrics_registry or REGISTRY
        self._server: asyncio.AbstractServer | None = None
        self._connections: dict[asyncio.StreamWriter, asyncio.Task[Any]] = {}
        self._stopping = False
        self._stop_task: asyncio.Task[None] | None = None
        self._transfer: TransferLayer | None = None
        self._transfer_lock = threading.Lock()
        self._local_slot_size = int(self._runtime_config.get("local_slot_size", 0))
        self._block_tokens = int(self._runtime_config.get("block_tokens", 0))
        # Online packed stores publish asynchronously after the worker has
        # released its source snapshot, so a successor lookup can observe the
        # writer key during the short transfer/commit interval. Retry for this
        # format; raw keeps immediate-miss semantics.
        self._packed_format = (
            self._runtime_config.get("storage_format")
            == STORAGE_FORMAT_COMPRESSED_ONLINE
        )
        self._lookup_wait_pending = self._packed_format
        logger.info("lookup wait_for_pending=%s", self._lookup_wait_pending)
        # Lookup readahead: promotion group size (0 disables) and groups in
        # flight. It needs both tiers.
        promotes = self._layout.enabled(L1) and self._layout.enabled(L2)
        self._readahead_piece = (
            int(os.environ.get(LOAD_READAHEAD_ENV) or DEFAULT_LOAD_READAHEAD_BYTES)
            if promotes
            else 0
        )
        self._readahead_budget = self._readahead_piece * int(
            os.environ.get(LOAD_READAHEAD_WINDOW_ENV) or 4
        )
        self._readahead_inflight = 0
        # Exclusive tiers: demotion batches of about one readahead piece (all
        # rank lanes), at most four in flight.
        self._demote_piece = (
            self._readahead_piece or DEFAULT_LOAD_READAHEAD_BYTES
        ) // self._layout.tensor_parallel_size
        self._demote_budget = 4 * self._demote_piece
        self._demote_inflight = 0
        self._demote_scheduled = False
        # Set when a demotion batch finishes; leases waiting for L1 room.
        self._room_changed = asyncio.Event()
        if core.exclusive_tiers:
            core.set_demotion_listener(self._kick_demotions)
        self._lease_budget = int(
            self._runtime_config.get("prefetch_lease_budget_bytes", 0)
        )
        self._leased_bytes = 0
        self._lease_waiters: list[asyncio.Future[None]] = []
        self._leases: dict[str, _LeaseState] = {}
        self._chunk_stores: dict[tuple[str, int, int], _ChunkStore] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._prev_l2_write_bytes = 0
        self._prev_l2_reads = 0
        self._prev_l2_timing_us: dict[str, int] = {}
        self._prev_l2_read_contention: dict[tuple[int, bool], tuple[int, int, int]] = {}
        self._prev_dst_copy: dict[str, int] = {}
        self._cuda_ipc_cache: OrderedDict[
            tuple[int, int, int, str | None], "_CachedCudaArray"
        ] = OrderedDict()
        self._load_staging_buffers: dict[tuple[int, int], _CachedCudaArray] = {}
        self._store_staging_buffers: dict[tuple[int, int], _CachedCudaArray] = {}
        self._op_handlers: dict[
            str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
        ] = {
            "lookup": self._op_lookup,
            "index_epoch": self._op_index_epoch,
            "lookup_prefetch": self._op_lookup_prefetch,
            "record_external_prefix_cache": self._op_record_external_prefix_cache,
            "get_runtime_config": self._op_get_runtime_config,
            "alloc_chunk": self._op_alloc_chunk,
            "alloc_chunks": self._op_alloc_chunks,
            "match_and_alloc": self._op_match_and_alloc,
            "commit_stats": self._op_commit_stats,
            "live_allocations": self._op_live_allocations,
            "transfer_drain": self._op_transfer_drain,
            "transfer_prefetch": self._op_transfer_prefetch,
            "init_transfer": self._op_init_transfer,
            "transfer_store": self._op_transfer_store,
            "transfer_load": self._op_transfer_load,
            "transfer_load_runs": self._op_transfer_load_runs,
            "register_load_staging": self._op_register_load_staging,
            "register_store_staging": self._op_register_store_staging,
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
        stalldump.register_loop("server", asyncio.get_running_loop())
        stalldump.register_state(
            "server-transfer",
            lambda: {} if self._transfer is None else self._transfer.debug_state(),
        )
        stalldump.register_state(
            "server-ipc",
            lambda: {
                "tasks": len(self._tasks),
                "chunk_stores": len(self._chunk_stores),
                "leases": len(self._leases),
                "lease_waiters": len(self._lease_waiters),
                "readahead_inflight": self._readahead_inflight,
                "demote_inflight": self._demote_inflight,
            },
        )
        self._server = await asyncio.start_unix_server(
            self._handle_connection, path=self._socket_path
        )
        logger.info("[IPC] listening on %s", self._socket_path)

    async def initialize_transfer(self) -> None:
        """Eagerly create the transfer layer off the event loop.

        Offloads the blocking ``_ensure_transfer`` call (pinned memory
        allocation, io_uring rings, cuFile handles) to a thread so the server
        is fully provisioned before the first inference request.

        Async/thread-safety:
            Must be called from the server asyncio event loop.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ensure_transfer)

    async def drain_transfer(self) -> None:
        """Wait for background store reports, promotions and transfer IO.

        Raises:
            RuntimeError: For the first transfer-layer store failure not yet
                reported by an earlier drain.

        Async/thread-safety:
            Runs on the server asyncio event loop. Work started while
            draining is waited for as well.
        """
        # Done callbacks remove tracked tasks on a later loop turn. gather
        # may return synchronously for already completed tasks, so waiting on
        # the whole set can spin forever and starve those cleanup callbacks.
        # Completion itself is sufficient: every task's bookkeeping is done.
        while pending := [task for task in self._tasks if not task.done()]:
            await asyncio.gather(*pending, return_exceptions=True)
        if self._transfer is not None:
            await self._transfer.drain()

    async def stop(self) -> None:
        """Stop accepting connections and close transfer resources.

        Async/thread-safety:
            Runs on the server asyncio event loop.
        """
        await self.stop_accepting()
        await self.close()
        logger.info("[IPC] server stopped")

    async def stop_accepting(self) -> None:
        """Close client streams and finish accepted operations before shutdown.

        Async/thread-safety:
            Runs idempotently on the owning loop. Concurrent callers share
            one stop task; cancelling a waiter does not cancel that task.
            Existing transfer resources remain open until ``close``.
        """
        self._stopping = True
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_accepting())
        await asyncio.shield(self._stop_task)

    async def close(self) -> None:
        """Release leases, drain transfer work and close transfer resources.

        Async/thread-safety:
            Runs on the server asyncio event loop after new IPC work has been
            rejected. Store failures found while draining are logged.
        """
        for lease_id in list(self._leases):
            self._release_lease(lease_id)
        if self._transfer is not None:
            try:
                await self.drain_transfer()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[IPC] transfer drain at close failed: %r", exc)
            await self._transfer.close()
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
        task = asyncio.current_task()
        assert task is not None
        self._connections[writer] = task
        try:
            while not self._stopping:
                try:
                    msg = await read_frame(reader)
                except asyncio.IncompleteReadError:
                    return
                if self._stopping:
                    return
                response = await self._dispatch(msg)
                if self._stopping:
                    return
                await write_frame(writer, response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[IPC] error handling request: %s", exc)
            with contextlib.suppress(Exception):
                await write_frame(writer, {"error": str(exc)})
        finally:
            self._connections.pop(writer, None)
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
        # Requests still running: a stalled op stays visible on /metrics.
        inflight = self._metrics.gauge(
            "daser_ipc_requests_inflight",
            "IPC requests currently being handled, by operation.",
        )
        inflight.inc(labels=ipc_labels)
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
            inflight.inc(-1.0, labels=ipc_labels)
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

    # ------------------------------------------------------------------
    # Lookup and allocation
    # ------------------------------------------------------------------

    async def _op_lookup(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``lookup`` request, recording external prefix counters.

        ``defer_pending`` selects how a lookup that a pending writer may
        extend is answered when pending waits are enabled: absent keeps the
        server-side wait, True replies ``{"pending": True}`` at once so a
        synchronous caller can retry later, and False resolves immediately.
        With ``external_prefix_queries`` the external window's L2-only chunks
        are promoted into L1 in the background (readahead).
        """
        tokens = self._lookup_tokens(msg)
        # Read before the lookup awaits: a removal during it advances the
        # epoch past the reply's, so a caller never reuses a stale result.
        epoch = self._core.index_epoch
        defer_pending = msg.get("defer_pending")
        if self._lookup_wait_pending and defer_pending is not None:
            chunks = await self._lookup_now(
                tokens,
                msg["model_id"],
                defer=bool(defer_pending),
                num_computed_tokens=int(msg.get("num_computed_tokens", 0)),
            )
            if chunks is None:
                return {"chunks": [], "pending": True}
        else:
            chunks = await self._lookup_core(tokens, msg["model_id"])
        if "external_prefix_queries" in msg:
            queries = int(msg.get("external_prefix_queries", 0))
            num_computed_tokens = int(msg.get("num_computed_tokens", 0))
            hits = _external_prefix_hits(chunks, num_computed_tokens, queries)
            await self._core.record_external_prefix_cache(queries=queries, hits=hits)
            if self._readahead_piece and chunks and hits > 0:
                await self._start_readahead(
                    _window_slots(chunks, num_computed_tokens, hits, self._block_tokens)
                )
        return {"chunks": self._chunk_payloads(chunks), "index_epoch": epoch}

    async def _op_index_epoch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``index_epoch`` request.

        Returns:
            ``{"index_epoch": int}``; a lookup reply carrying the same value
            still describes chunks that are all present in the index.
        """
        return {"index_epoch": self._core.index_epoch}

    async def _op_lookup_prefetch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Look up a prompt and lease the L1 copies of its external window.

        Every chunk of the window is lease pinned. ``tier`` is ``l1`` when
        all of them already have an L1 copy, else ``mixed`` (some) or ``l2``
        (none); ``transfer_prefetch`` then promotes the rest. Without an L1
        tier, or for an empty window, no lease is taken and ``tier`` is None.

        Returns:
            ``{"chunks", "spans", "tier"}``; ``spans`` are the window's
            ``{start_slot, num_slots, nbytes}`` (``nbytes`` per rank) and are
            empty exactly when ``tier`` is None.
        """
        lease_id = str(msg.get("lease_id", ""))
        if not lease_id:
            raise ValueError("lookup_prefetch requires lease_id")
        self._release_lease(lease_id)
        tokens = self._lookup_tokens(msg)
        chunks = await self._lookup_core(tokens, msg["model_id"])
        queries = int(msg.get("external_prefix_queries", 0))
        num_computed_tokens = int(msg.get("num_computed_tokens", 0))
        hits = _external_prefix_hits(chunks, num_computed_tokens, queries)
        await self._core.record_external_prefix_cache(queries=queries, hits=hits)
        payloads = self._chunk_payloads(chunks)
        if self._stopping or not chunks or hits <= 0 or not self._layout.enabled(L1):
            return {"chunks": payloads, "spans": [], "tier": None}
        windows = _window_slots(chunks, num_computed_tokens, hits, self._block_tokens)
        leases: list[Lease] = []
        spans: list[dict[str, int]] = []
        promote_keys: list[str] = []
        nbytes = 0
        tp = self._layout.tensor_parallel_size
        for chunk, first, count in windows:
            start_slot = int(chunk.start_slot) + first
            parts = self._core.resolve_slots(start_slot, count)
            span_bytes = 0
            for key, index, run in parts or []:
                records = self._core.chunk_records(key)
                lease = self._core.pin_lease(key) if records is not None else None
                if records is None or lease is None:
                    parts = None
                    break
                leases.append(lease)
                span_bytes += sum(records.record_sizes[index : index + run])
                if L1 not in self._core.full_tiers(key):
                    promote_keys.append(key)
                nbytes += self._layout.extent_length(L1, records.record_sizes) * tp
            if parts is None:
                for taken in leases:
                    self._core.unpin_lease(taken)
                return {"chunks": payloads, "spans": [], "tier": None}
            spans.append(
                {"start_slot": start_slot, "num_slots": count, "nbytes": span_bytes}
            )
        if not spans:
            return {"chunks": payloads, "spans": [], "tier": None}
        if not promote_keys:
            tier = "l1"
        elif len(promote_keys) == len(leases):
            tier = "l2"
        else:
            tier = "mixed"
        self._leases[lease_id] = _LeaseState(
            lease_id=lease_id,
            leases=leases,
            windows=[(span["start_slot"], span["num_slots"]) for span in spans],
            promote_keys=promote_keys,
            nbytes=nbytes,
        )
        return {"chunks": payloads, "spans": spans, "tier": tier}

    @staticmethod
    def _lookup_tokens(msg: dict[str, Any]) -> TokenSequence:
        """Decode the compact token representation used by scheduler lookup.

        Args:
            msg: IPC request containing ``token_bytes`` or ``tokens``.

        Returns:
            Token IDs as packed native-int bytes or a list.

        Raises:
            ValueError: if the packed representation is malformed.
        """
        packed = msg.get("token_bytes")
        if packed is not None:
            if not isinstance(packed, bytes) or len(packed) % 4:
                raise ValueError("lookup token_bytes must be int32 aligned bytes")
            return packed
        return msg["tokens"]

    async def _lookup_core(
        self, tokens: TokenSequence, model_id: str
    ) -> list[ChunkInfo]:
        """Run a lookup against the committed retrieval index.

        Args:
            tokens: Prompt token IDs.
            model_id: Model identifier used for cache isolation.

        Returns:
            Retrieval chunks returned by the server core; every one has a
            tier copy valid on every rank.

        Async/thread-safety:
            Runs on the IPC server event loop; waits for pending writers only
            in compressed-online mode.
        """
        return await self._core.lookup(
            tokens, model_id, wait_for_pending=self._lookup_wait_pending
        )

    async def _lookup_now(
        self,
        tokens: TokenSequence,
        model_id: str,
        *,
        defer: bool,
        num_computed_tokens: int = 0,
    ) -> list[ChunkInfo] | None:
        """Run a lookup that never waits for pending writers.

        Args:
            tokens: Prompt token IDs.
            model_id: Model identifier used for cache isolation.
            defer: Return None instead of chunks when a pending writer may
                extend the match.
            num_computed_tokens: Tokens the caller's GPU prefix cache already
                holds; writers ending within them do not defer.

        Returns:
            Retrieval chunks, or None for a deferred lookup.

        Async/thread-safety:
            Runs on the IPC server event loop without sleeping, so a
            synchronous scheduler RPC never waits for a writer commit.
        """
        if defer:
            return await self._core.lookup_unless_pending(
                tokens, model_id, num_computed_tokens
            )
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
        alloc = await self._core.alloc_chunk(
            msg["chunk_key"],
            int(msg["token_count"]),
            msg["model_id"],
            parent_key=str(msg.get("parent_key", "")),
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

    async def _op_commit_stats(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``commit_stats`` request."""
        return {"commit_stats": await self._core.commit_stats()}

    async def _op_live_allocations(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``live_allocations`` request."""
        live = await self._core.live_allocations(list(msg.get("allocations", [])))
        return {"chunk_keys": live}

    async def _op_transfer_drain(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``transfer_drain`` request (no-op before transfer init)."""
        if self._transfer is not None:
            await self.drain_transfer()
        return {"ok": True}

    async def _op_init_transfer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``init_transfer`` request.

        Waits off the event loop: during startup the eager initializer may
        hold the transfer lock while pinning L1 for minutes, and blocking the
        loop on it would stall every other client's request past its timeout.
        """
        await self.initialize_transfer()
        return {"ok": True}

    async def _op_evict_chunk(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle an ``evict_chunk`` request."""
        await self._core.evict_chunk(msg["chunk_key"])
        return {"ok": True}

    async def _op_release_chunk_writer(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Handle a ``release_chunk_writer`` request.

        Also forgets partial multi-request stores of the allocation; their
        in-flight IO still releases its own pins.
        """
        chunk_key = str(msg["chunk_key"])
        start_slot = int(msg["start_slot"])
        released = await self._core.release_chunk_writer(
            chunk_key=chunk_key,
            start_slot=start_slot,
            num_slots=int(msg["num_slots"]),
        )
        for key in [
            key
            for key in self._chunk_stores
            if key[0] == chunk_key and key[1] == start_slot
        ]:
            del self._chunk_stores[key]
        return {"released": released}

    # ------------------------------------------------------------------
    # Prefetch leases
    # ------------------------------------------------------------------

    async def _op_transfer_prefetch(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Promote a lease's chunks into L1 and pin their L1 segments.

        Args:
            msg: IPC request with the ``lease_id`` of a ``lookup_prefetch``.

        Returns:
            ``{"ok", "requested_bytes", "l1_bytes", "l2_bytes"}`` (bytes over
            all ranks; ``l2_bytes`` were promoted), or ``{"error"}`` when the
            lease is gone or a chunk could not be brought into L1 (the
            scheduler releases the lease and loads without it).

        Async/thread-safety:
            Waits for earlier leases while the lease budget is exhausted.
        """
        lease_id = str(msg.get("lease_id", ""))
        operations = self._metrics.counter(
            "daser_prefetch_operations_total",
            "Host-tier prefetch operations by result.",
        )
        state = self._leases.get(lease_id)
        if state is None or state.released:
            operations.inc(labels={"status": "incomplete"})
            return {"error": f"transfer lease {lease_id!r} is not active"}
        try:
            await wait_shared([self._lease_ready(state)])
        except (ChunkUnavailable, OSError, RuntimeError) as exc:
            operations.inc(labels={"status": "incomplete"})
            logger.warning("[IPC] transfer_prefetch lease=%s: %s", lease_id[:8], exc)
            return {"error": str(exc)}
        operations.inc(labels={"status": "ok"})
        tp = self._layout.tensor_parallel_size
        requested = tp * sum(seg.nbytes for seg in state.segments)
        promoted = tp * sum(
            seg.nbytes for seg in state.segments if seg.chunk_key in state.promote_keys
        )
        bytes_counter = self._metrics.counter(
            "daser_prefetch_bytes_total",
            "Host-tier prefetch bytes by tier.",
        )
        bytes_counter.inc(requested, labels={"tier": "requested"})
        bytes_counter.inc(requested - promoted, labels={"tier": "l1"})
        bytes_counter.inc(promoted, labels={"tier": "l2"})
        return {
            "ok": True,
            "requested_bytes": requested,
            "l1_bytes": requested - promoted,
            "l2_bytes": promoted,
        }

    async def _op_release_transfer_lease(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Idempotently release the pins of one request lease."""
        lease_id = str(msg.get("lease_id", ""))
        if not lease_id:
            raise ValueError("release_transfer_lease requires lease_id")
        self._release_lease(lease_id)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Transfers
    # ------------------------------------------------------------------

    async def _op_transfer_store(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Place and store whole or partial chunks from a worker buffer.

        Args:
            msg: IPC request with ``payload``, ``tp_rank``, ``tp_size`` and
                ``spans`` of ``{source_offset, nbytes, chunk_key, start_slot,
                num_slots, slot_index, packed, mode}``. A raw span covers
                ``nbytes / local_slot_size`` slots from ``slot_index``; a
                packed span is one record, and a request carries every
                record of its packed chunks.

        Returns:
            ``{"ok", "bytes", "chunk_keys"}`` once the buffer was consumed
            (and, for chunks without an L1 copy, written to L2); stale
            allocations are skipped and not listed.

        Raises:
            ValueError: For a malformed span or a TP size mismatch.
            TierFull: If a tier cannot hold a chunk.

        Async/thread-safety:
            Runs on the IPC event loop. IO completion is reported to the
            core by a background task that outlives the request.
        """
        started = time.perf_counter()
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        tp_rank = int(msg.get("tp_rank", 0))
        tp_size = int(msg.get("tp_size", 1))
        if tp_size != self._layout.tensor_parallel_size or not 0 <= tp_rank < tp_size:
            raise ValueError(
                f"transfer rank {tp_rank}/{tp_size} does not match the server "
                f"TP size {self._layout.tensor_parallel_size}"
            )
        transfer = self._ensure_transfer()
        groups = self._store_groups(spans)
        buffer, one_shot = self._store_buffer(payload, spans)
        parts: list[_StorePart] = []
        try:
            placements = await self._core.place_store(
                [request for request, _spans, _slots in groups]
            )
            moves: list[Move] = []
            stored_keys: list[str] = []
            nbytes = 0
            for (request, group_spans, slots), placement in zip(
                groups, placements, strict=True
            ):
                if placement is None:
                    logger.debug("[IPC] skip stale store key=%s", request.chunk_key[:8])
                    continue
                parts.append(
                    _StorePart(
                        key=(request.chunk_key, request.start_slot, tp_rank),
                        num_slots=request.num_slots,
                        slots=slots,
                        placement=placement,
                    )
                )
                stored_keys.append(request.chunk_key)
                group_moves = self._store_moves(placement, group_spans, tp_rank)
                moves.extend(group_moves)
                nbytes += sum(move.nbytes for move in group_moves)
            handle = await transfer.store(buffer, moves) if parts else None
        except BaseException:
            await self._abandon_store_parts(parts, tp_rank)
            if one_shot:
                buffer.close()
            raise
        if handle is None:
            if one_shot:
                buffer.close()
            return {"ok": True, "bytes": 0, "chunk_keys": []}
        reply: asyncio.Future[BaseException | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._register_store_parts(parts, handle.staged, handle.persisted)
        self._spawn(
            self._finish_store(
                parts,
                handle.staged,
                handle.persisted,
                buffer if one_shot else None,
                reply,
                tp_rank,
            )
        )
        error = await asyncio.shield(reply)
        if error is not None:
            raise error
        self._record_packed_ratios(spans)
        self._record_transfer_metrics("store", nbytes, time.perf_counter() - started)
        return {"ok": True, "bytes": nbytes, "chunk_keys": stored_keys}

    async def _op_transfer_load(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Read slot ranges from their tiers into a worker buffer.

        Args:
            msg: IPC request with ``payload``, ``tp_rank`` and ``spans`` of
                ``{target_offset, start_slot, num_slots, nbytes}``; a span
                may start inside a chunk and cover adjacent chunks, and its
                records land back to back from ``target_offset``.

        Returns:
            ``{"ok", "bytes", "l1_segments", "l2_segments", "transfer_ms",
            "transfer_open_ms", "transfer_load_ms", "transfer_stats_delta"}``,
            or ``{"ok", "bytes", "data"}`` for ``return_data`` payloads.

        Raises:
            ValueError: If a span's ``nbytes`` differs from its records.
            ChunkUnavailable: If a slot has no committed chunk.

        Async/thread-safety:
            Runs on the IPC event loop. Whole L2-only chunks read here are
            promoted into L1 by the same read (all ranks' lanes). Segments
            are copied in batches, so ready chunks are not held back by a
            later chunk's in-flight promotion.
        """
        started = time.perf_counter()
        payload = msg.get("payload", {})
        spans = list(msg.get("spans", []))
        tp_rank = int(msg.get("tp_rank", 0))
        if not 0 <= tp_rank < self._layout.tensor_parallel_size:
            raise ValueError(f"invalid TP rank {tp_rank}")
        transfer = self._ensure_transfer()
        open_ms = 0.0
        if payload.get("return_data"):
            buffer: Any = bytearray(self._payload_size(payload, spans))
            one_shot = False
        else:
            open_started = time.perf_counter()
            buffer = self._payload_buffer(payload)
            open_ms = (time.perf_counter() - open_started) * 1000
            # Registered staging mappings live until shutdown.
            one_shot = isinstance(buffer, _UncachedCudaArray) and (
                "load_staging_buffer_index" not in payload
            )
        segments: list[LoadSegment] = []
        before = asdict(transfer.stats)
        planning_started = time.perf_counter()
        try:
            wait_s = await self._stream_load(transfer, buffer, spans, tp_rank, segments)
        finally:
            self._core.release_load(segments)
            if one_shot:
                buffer.close()
        finished = time.perf_counter()
        after = asdict(transfer.stats)
        total = sum(segment.nbytes for segment in segments)
        l1_segments = sum(segment.tier == L1 for segment in segments)
        io_s = finished - planning_started - wait_s
        self._record_load_metrics(
            l1_segments,
            len(segments) - l1_segments,
            resolve_s=planning_started - started + wait_s,
            io_s=io_s,
        )
        self._record_transfer_metrics("load", total, finished - started)
        response: dict[str, Any] = {
            "ok": True,
            "bytes": total,
            "l1_segments": l1_segments,
            "l2_segments": len(segments) - l1_segments,
        }
        if payload.get("return_data"):
            response["data"] = bytes(buffer)
            return response
        response["transfer_ms"] = (finished - started) * 1000
        response["transfer_open_ms"] = open_ms
        response["transfer_load_ms"] = io_s * 1000
        response["transfer_stats_delta"] = {
            key: int(after[key]) - int(before[key]) for key in after
        }
        return response

    async def _op_transfer_load_runs(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Copy byte runs of a lease's L1 records into a registered buffer.

        Layer-wise delivery copies one plane group of every record of a
        step's requests per call. ``runs`` packs ``(start_slots, intra_offsets,
        targets, lengths)`` as one ``(4, n)`` int64 array: each run reads
        ``lengths[i]`` bytes at ``intra_offsets[i]`` inside the record of slot
        ``start_slots[i]``.

        Args:
            msg: IPC request with ``payload`` (registered staging buffer),
                ``lease_ids`` (one per request of the step), ``tp_rank`` and
                ``runs`` bytes.

        Returns:
            ``{"ok", "bytes", "transfer_ms"}`` once the copies completed on
            the device.

        Raises:
            ValueError: If the run table is malformed, a lease is not active,
                a slot is in none of the leases
                or a run leaves its record.
            ChunkUnavailable: If the lease could not bring a chunk into L1.

        Async/thread-safety:
            Runs on the IPC event loop. The first call of a lease pins its L1
            segments (shared with ``transfer_prefetch``); the lease stays
            until ``release_transfer_lease``.
        """
        started = time.perf_counter()
        lease_ids = [str(lease_id) for lease_id in msg.get("lease_ids", [])]
        packed = msg.get("runs")
        payload = msg.get("payload", {})
        tp_rank = int(msg.get("tp_rank", 0))
        if "load_staging_buffer_index" not in payload:
            raise ValueError("transfer_load_runs requires a registered buffer")
        if not isinstance(packed, bytes) or not packed or len(packed) % 32:
            raise ValueError("transfer_load_runs runs must be (4, n) int64 bytes")
        if not lease_ids:
            raise ValueError("transfer_load_runs requires lease_ids")
        states = []
        for lease_id in lease_ids:
            state = self._leases.get(lease_id)
            if state is None:
                raise ValueError(f"transfer lease {lease_id!r} is not active")
            states.append(state)
        await wait_shared([self._lease_ready(state) for state in states])
        if any(state.released for state in states):
            raise ChunkUnavailable("transfer lease was released during preparation")
        slot_ids, lane_offsets, record_sizes = _lease_tables(states)
        slots, intra, targets, lengths = numpy.frombuffer(
            packed, dtype=numpy.int64
        ).reshape(4, -1)
        index = numpy.searchsorted(slot_ids, slots)
        if int(index.max()) >= len(slot_ids) or not numpy.array_equal(
            slot_ids[index], slots
        ):
            raise ValueError("transfer_load_runs names a slot outside its leases")
        if int(intra.min()) < 0 or bool(
            numpy.any(intra + lengths > record_sizes[index])
        ):
            raise ValueError("transfer_load_runs run leaves its record")
        offsets = (
            self._layout.absolute_offset(L1, tp_rank, 0) + lane_offsets[index] + intra
        )
        for state in states:
            state.active_loads += 1
        try:
            await self._ensure_transfer().load_runs(
                self._payload_buffer(payload), offsets, targets, lengths
            )
        finally:
            # The transfer API returns only after destination DMA completion,
            # also on cancellation. Idle leases may go immediately at stop;
            # these borrowed source pages must survive until this point.
            for state in states:
                state.active_loads -= 1
                if state.released and not state.active_loads:
                    self._finish_lease_release(state)
        total = int(lengths.sum())
        elapsed_s = time.perf_counter() - started
        self._record_transfer_metrics("load", total, elapsed_s)
        return {"ok": True, "bytes": total, "transfer_ms": elapsed_s * 1000}

    async def _op_register_load_staging(self, msg: dict[str, Any]) -> dict[str, Any]:
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
        return self._register_staging(msg, self._load_staging_buffers)

    async def _op_register_store_staging(self, msg: dict[str, Any]) -> dict[str, Any]:
        """Map a fixed worker store buffer during startup, outside timed traffic."""
        return self._register_staging(msg, self._store_staging_buffers)

    # ------------------------------------------------------------------
    # Store helpers
    # ------------------------------------------------------------------

    async def _stop_accepting(self) -> None:
        """Join accepted handlers before waiting for the listener's transports."""
        if self._server is not None:
            self._server.close()
        connections = list(self._connections.items())
        for writer, _task in connections:
            writer.close()
        # Disconnected clients cannot release leases. Wake budget/capacity
        # waiters before joining handlers, while active device copies retain
        # their source pages independently until load_runs returns.
        for lease_id in list(self._leases):
            self._release_lease(lease_id)
        # Python 3.12's wait_closed also waits for accepted connections. A
        # connector keeps them open, so close streams first, then allow every
        # already dispatched operation to finish before freeing mappings.
        if connections:
            await asyncio.gather(
                *(task for _writer, task in connections), return_exceptions=True
            )
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None
        if os.path.exists(self._socket_path):
            os.unlink(self._socket_path)
        logger.info("[IPC] server stopped accepting")

    def _store_groups(
        self, spans: list[dict[str, Any]]
    ) -> list[tuple[StoreRequest, list[dict[str, int]], frozenset[int]]]:
        """Validate store spans and group them into per-chunk store requests.

        Args:
            spans: Request spans (see ``_op_transfer_store``).

        Returns:
            One ``(StoreRequest, spans, slots)`` per chunk allocation, with
            spans as ``{source_offset, nbytes, slot_index}`` (one per record
            for packed chunks, in slot order) and the slot indices they
            write. Stale allocations are kept; the core skips them.

        Raises:
            ValueError: For a malformed span, a packed chunk whose records do
                not cover every slot once, a raw span not made of whole
                slots inside its allocation, or a raw span when the runtime
                config has no ``local_slot_size``.
        """
        slot_size = self._local_slot_size
        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for span in spans:
            key = str(span.get("chunk_key", ""))
            if not key:
                raise ValueError("store span requires chunk_key")
            group = (key, int(span["start_slot"]), int(span["num_slots"]))
            grouped.setdefault(group, []).append(span)
        result: list[tuple[StoreRequest, list[dict[str, int]], frozenset[int]]] = []
        for (key, start_slot, num_slots), members in grouped.items():
            parsed = [
                {
                    "source_offset": int(span.get("source_offset", 0)),
                    "nbytes": int(span["nbytes"]),
                    "slot_index": int(span["slot_index"]),
                }
                for span in members
            ]
            if any(s["source_offset"] < 0 or s["nbytes"] <= 0 for s in parsed):
                raise ValueError(f"store span of {key} has an empty or negative range")
            packed = {bool(span.get("packed", False)) for span in members}
            if len(packed) != 1:
                raise ValueError(f"store spans of {key} mix packed and raw records")
            if packed.pop():
                parsed.sort(key=lambda s: s["slot_index"])
                if [s["slot_index"] for s in parsed] != list(range(num_slots)):
                    raise ValueError(
                        f"packed store of {key} must carry each of its "
                        f"{num_slots} records once"
                    )
                by_slot = sorted(members, key=lambda span: int(span["slot_index"]))
                modes = tuple(str(span.get("mode", "")) for span in by_slot)
                if any(mode not in _RECORD_MODES for mode in modes):
                    raise ValueError(f"packed store of {key} has an unknown mode")
                request = StoreRequest(
                    key,
                    start_slot,
                    num_slots,
                    tuple(s["nbytes"] for s in parsed),
                    modes,
                )
                slots = frozenset(range(num_slots))
            else:
                if slot_size <= 0:
                    raise ValueError(
                        "raw store spans need runtime_config local_slot_size"
                    )
                covered: set[int] = set()
                for s in parsed:
                    count, rest = divmod(s["nbytes"], slot_size)
                    if rest or not 0 <= s["slot_index"] <= num_slots - count:
                        raise ValueError(
                            f"raw store span of {key} is not whole slots of "
                            "its allocation"
                        )
                    covered.update(range(s["slot_index"], s["slot_index"] + count))
                request = StoreRequest(
                    key, start_slot, num_slots, (slot_size,) * num_slots
                )
                slots = frozenset(covered)
            result.append((request, parsed, slots))
        return result

    def _store_buffer(
        self, payload: dict[str, Any], spans: list[dict[str, Any]]
    ) -> tuple[Any, bool]:
        """Open the source buffer of a store request.

        Args:
            payload: ``data``, a registered store buffer index, or a one-shot
                CUDA IPC handle.
            spans: Request spans, used to bound-check registered buffers and
                size one-shot mappings.

        Returns:
            ``(buffer, one_shot)``; a one-shot mapping is closed by the
            caller once the store consumed it.

        Raises:
            ValueError: For an unknown registered buffer or a span outside it.
        """
        if "store_staging_buffer_index" in payload:
            buffer_key = (
                int(payload["producer_pid"]),
                int(payload["store_staging_buffer_index"]),
            )
            buffer = self._store_staging_buffers.get(buffer_key)
            if buffer is None:
                raise ValueError(
                    f"store staging buffer is not registered: {buffer_key}"
                )
            for span in spans:
                offset, nbytes = int(span.get("source_offset", 0)), int(span["nbytes"])
                if offset < 0 or nbytes < 0 or offset + nbytes > buffer.nbytes:
                    raise ValueError(
                        "store source span exceeds registered staging buffer"
                    )
            return buffer, False
        if "cuda_ipc_handle" in payload:
            # The worker may export a compact view ending at its last record;
            # map through the furthest submitted byte. Store mappings are
            # one-shot because the source extent varies per batch.
            payload = dict(payload)
            payload["nbytes"] = max(
                [int(payload.get("nbytes", 0))]
                + [
                    int(span.get("source_offset", 0)) + int(span["nbytes"])
                    for span in spans
                ]
            )
            return (
                self._open_cuda_ipc_payload(
                    payload=payload, nbytes_key="nbytes", cache_mapping=False
                ),
                True,
            )
        return self._payload_buffer(payload), False

    def _store_moves(
        self,
        placement: StorePlacement,
        spans: list[dict[str, int]],
        tp_rank: int,
    ) -> list[Move]:
        """Return the moves writing one chunk's spans into its placement.

        Args:
            placement: Chunk placement.
            spans: ``{source_offset, nbytes, slot_index}`` spans of the chunk.
            tp_rank: Storing rank (selects the lane).
        """
        offsets = record_offsets(placement.record_sizes)
        layout = self._layout
        moves = []
        for span in spans:
            within = offsets[span["slot_index"]]
            moves.append(
                Move(
                    buffer_offset=span["source_offset"],
                    nbytes=span["nbytes"],
                    l1_offset=(
                        None
                        if placement.l1 is None
                        else layout.absolute_offset(
                            L1, tp_rank, placement.l1.offset + within
                        )
                    ),
                    l2_offset=(
                        None
                        if placement.l2 is None
                        else layout.absolute_offset(
                            L2, tp_rank, placement.l2.offset + within
                        )
                    ),
                )
            )
        return moves

    def _register_store_parts(
        self,
        parts: list[_StorePart],
        staged: asyncio.Future[None],
        persisted: asyncio.Future[None],
    ) -> None:
        """Add one request's IO to its chunks' trackers and pick reporters.

        Args:
            parts: The request's chunk parts.
            staged: The request's ``StoreHandle.staged``.
            persisted: The request's ``StoreHandle.persisted``.

        Async/thread-safety:
            Synchronous on the event loop right after ``TransferLayer.store``
            returned, so the request completing a chunk's slot set sees the
            IO of every earlier request of the chunk.
        """
        for part in parts:
            tracker = self._chunk_stores.setdefault(
                part.key, _ChunkStore(part.num_slots)
            )
            part.tracker = tracker
            tracker.staged.append(staged)
            tracker.persisted.append(persisted)
            tracker.slots.update(part.slots)
            if len(tracker.slots) == tracker.num_slots:
                part.reporter = True
                del self._chunk_stores[part.key]

    async def _abandon_store_parts(self, parts: list[_StorePart], tp_rank: int) -> None:
        """Undo placements of a store request whose IO was never submitted.

        A chunk with earlier requests in flight only drops this request's
        pins; otherwise the store is reported failed, so a placement nobody
        writes does not linger.

        Args:
            parts: Placed parts of the request.
            tp_rank: Storing rank.
        """
        tp = self._layout.tensor_parallel_size
        for part in parts:
            if part.key in self._chunk_stores:
                self._core.release_placement(part.placement)
            else:
                await self._core.store_failed(part.placement, tp_rank, tp)

    async def _finish_store(
        self,
        parts: list[_StorePart],
        staged: asyncio.Future[None],
        persisted: asyncio.Future[None],
        one_shot: Any,
        reply: asyncio.Future[BaseException | None],
        tp_rank: int,
    ) -> None:
        """Report one store request's IO to the core and resolve its reply.

        A reporter part reports the chunk's outcome over every request of
        the chunk (``store_staged`` / ``store_persisted``, or
        ``store_failed`` on any failure); other parts only release their
        pins once this request's IO ended.

        Args:
            parts: The request's chunk parts.
            staged: The request's ``StoreHandle.staged``.
            persisted: The request's ``StoreHandle.persisted``.
            one_shot: One-shot source mapping to close once staged, or None.
            reply: Resolved with the request's error (or None) once the
                buffer was consumed and chunks without an L1 copy persisted.
            tp_rank: Storing rank.

        Async/thread-safety:
            Background task on the event loop; never cancelled by the request.
        """
        tp = self._layout.tensor_parallel_size
        try:
            staged_error = await _outcome([staged])
            if one_shot is not None:
                one_shot.close()
            if staged_error is not None and not reply.done():
                reply.set_result(staged_error)
            waiting: list[_StorePart] = []
            for part in parts:
                placement = part.placement
                if not part.reporter or placement.l1 is None:
                    waiting.append(part)
                    continue
                assert part.tracker is not None
                error = await _outcome(part.tracker.staged)
                if error is not None:
                    await self._core.store_failed(placement, tp_rank, tp)
                    continue
                await self._core.store_staged(placement, tp_rank, tp)
                if placement.l2 is not None:
                    waiting.append(part)
            unstaged = any(p.reporter and p.placement.l1 is None for p in waiting)
            if not unstaged and not reply.done():
                reply.set_result(None)
            await _outcome([persisted])
            for part in waiting:
                error = await self._finish_persisted(part, tp_rank)
                if error is not None and part.placement.l1 is None:
                    if not reply.done():
                        reply.set_result(error)
            if not reply.done():
                reply.set_result(None)
        except BaseException as exc:
            if not reply.done():
                reply.set_result(exc)
            raise

    async def _finish_persisted(
        self, part: _StorePart, tp_rank: int
    ) -> BaseException | None:
        """End one chunk part once its request's L2 IO ended.

        Args:
            part: Chunk part of a request whose IO ended.
            tp_rank: Storing rank.

        Returns:
            The chunk's store failure reported by a reporter, else None.
        """
        tp = self._layout.tensor_parallel_size
        placement = part.placement
        tracker = part.tracker
        assert tracker is not None
        if not part.reporter:
            # The reporter (a later request) sees this request's failure
            # through the shared futures; a stale allocation never gets one.
            self._core.release_placement(placement)
            if self._chunk_stores.get(
                part.key
            ) is tracker and not self._core.is_current_allocation(
                part.key[0], part.key[1], part.num_slots
            ):
                del self._chunk_stores[part.key]
            return None
        futures = tracker.staged + tracker.persisted
        error = await _outcome(futures)
        if error is not None:
            await self._core.store_failed(placement, tp_rank, tp)
            if placement.l1 is not None:
                logger.warning(
                    "[IPC] L2 write of %s failed: %r", part.key[0][:8], error
                )
            return error
        await self._core.store_persisted(placement, tp_rank, tp)
        return None

    # ------------------------------------------------------------------
    # Load helpers
    # ------------------------------------------------------------------

    async def _stream_load(
        self,
        transfer: TransferLayer,
        buffer: Any,
        spans: list[dict[str, Any]],
        tp_rank: int,
        segments: list[LoadSegment],
    ) -> float:
        """Plan load spans chunk by chunk and copy them in ready batches.

        Every chunk without an in-flight promotion is pinned and copied in a
        first batch; each later batch waits for one promoting chunk and takes
        every deferred chunk ready by then, so copies overlap the waits.

        Args:
            transfer: Transfer layer.
            buffer: Destination buffer.
            spans: ``{target_offset, start_slot, num_slots, nbytes}`` spans.
            tp_rank: Loading rank.
            segments: Filled with every pinned segment, also on failure; the
                caller releases them with ``ServerCore.release_load`` once
                this returns.

        Returns:
            Seconds blocked planning (waiting for promotions).

        Raises:
            ValueError: If a span's ``nbytes`` differs from its records.
            ChunkUnavailable: If a slot has no committed chunk.
            BaseException: The first failure, after every batch ended.

        Async/thread-safety:
            Runs on the IPC event loop; batches run as tasks that never
            outlive this call.
        """
        parts = self._load_parts(spans)
        batches: list[asyncio.Future[None]] = []
        wait_s = 0.0

        async def plan(part: _LoadPart) -> tuple[int, LoadSegment]:
            planned = await self._core.plan_load(part.start_slot, part.num_slots)
            segments.extend(planned)
            if len(planned) != 1 or planned[0].nbytes != part.nbytes:
                raise ChunkUnavailable(f"chunk {part.chunk_key} changed meanwhile")
            return part.target_offset, planned[0]

        def flush(batch: list[tuple[int, LoadSegment]]) -> None:
            if batch:
                batches.append(
                    asyncio.ensure_future(
                        self._load_batch(transfer, buffer, batch, tp_rank)
                    )
                )

        try:
            deferred = [p for p in parts if self._core.promotion_pending(p.chunk_key)]
            # No await between the checks and these pins (plan_load waits
            # only on promotions), so the first batch never blocks.
            flush(
                [
                    await plan(part)
                    for part in parts
                    if not self._core.promotion_pending(part.chunk_key)
                ]
            )
            while deferred:
                waited = time.perf_counter()
                batch = [await plan(deferred.pop(0))]
                wait_s += time.perf_counter() - waited
                waiting = []
                for part in deferred:
                    if self._core.promotion_pending(part.chunk_key):
                        waiting.append(part)
                    else:
                        batch.append(await plan(part))
                deferred = waiting
                flush(batch)
        finally:
            error = await _outcome(batches)
        if error is not None:
            raise error
        return wait_s

    def _load_parts(self, spans: list[dict[str, Any]]) -> list[_LoadPart]:
        """Split load spans into per-chunk parts with their buffer offsets.

        Args:
            spans: ``{target_offset, start_slot, num_slots, nbytes}`` spans.

        Returns:
            Parts in span order; nothing is pinned.

        Raises:
            ValueError: If a span is malformed or its ``nbytes`` differs from
                its records.
            ChunkUnavailable: If a slot has no committed chunk.
        """
        parts: list[_LoadPart] = []
        for span in spans:
            target = int(span["target_offset"])
            start_slot = int(span["start_slot"])
            num_slots = int(span["num_slots"])
            if target < 0 or num_slots < 1:
                raise ValueError(f"invalid load span at slot {start_slot}")
            resolved = self._core.resolve_slots(start_slot, num_slots)
            if resolved is None:
                raise ChunkUnavailable(f"slot {start_slot} has no committed chunk")
            slot = start_slot
            nbytes = 0
            for key, index, count in resolved:
                records = self._core.chunk_records(key)
                if records is None:
                    raise ChunkUnavailable(f"chunk {key} has no tier copy")
                size = sum(records.record_sizes[index : index + count])
                parts.append(_LoadPart(key, slot, count, target + nbytes, size))
                nbytes += size
                slot += count
            if nbytes != int(span["nbytes"]):
                raise ValueError(
                    f"load span at slot {start_slot} has {nbytes} "
                    f"record bytes, not {span['nbytes']}"
                )
        return parts

    async def _promote_on_read(self, segments: list[LoadSegment]) -> list[Promotion]:
        """Admit the whole L2-only chunks of a load into L1.

        Args:
            segments: Planned load segments.

        Returns:
            Promotions begun (finish them after the read).
        """
        if not (self._layout.enabled(L1) and self._layout.enabled(L2)):
            return []
        keys = []
        for segment in segments:
            if segment.tier != L2 or segment.slot_index != 0:
                continue
            records = self._core.chunk_records(segment.chunk_key)
            if records is not None and len(records.record_sizes) == len(
                segment.record_sizes
            ):
                keys.append(segment.chunk_key)
        if not keys:
            return []
        promotions = await self._core.begin_promotions(list(dict.fromkeys(keys)))
        return [promotion for promotion in promotions if promotion is not None]

    async def _load_batch(
        self,
        transfer: TransferLayer,
        buffer: Any,
        planned: list[tuple[int, LoadSegment]],
        tp_rank: int,
    ) -> None:
        """Copy one batch of pinned segments, promoting its whole L2 chunks.

        Args:
            transfer: Transfer layer.
            buffer: Destination buffer.
            planned: ``(buffer offset, segment)`` pairs.
            tp_rank: Loading rank.

        Raises:
            BaseException: The first transfer failure, after every call
                ended; the batch's promotions are finished either way.
        """
        promotions = await self._promote_on_read(
            [segment for _target, segment in planned]
        )
        ok = False
        try:
            await self._run_load(transfer, buffer, planned, promotions, tp_rank)
            ok = True
        finally:
            for promotion in promotions:
                self._core.finish_promotion(promotion, ok)

    async def _run_load(
        self,
        transfer: TransferLayer,
        buffer: Any,
        planned: list[tuple[int, LoadSegment]],
        promotions: list[Promotion],
        tp_rank: int,
    ) -> None:
        """Issue one transfer load per source tier (and peer promotions).

        Args:
            transfer: Transfer layer.
            buffer: Destination buffer.
            planned: ``(buffer offset, segment)`` pairs.
            promotions: Promotions of whole chunks read here; this rank's
                lane is filled by the L2 read, other ranks' lanes by
                ``promote``.
            tp_rank: Loading rank.

        Raises:
            BaseException: The first transfer failure, after every call
                ended.
        """
        layout = self._layout
        targets = {promotion.chunk_key: promotion.target for promotion in promotions}
        by_tier: dict[str, list[Move]] = {L1: [], L2: []}
        for buffer_offset, segment in planned:
            lane = segment.lane_offset
            promoted = targets.get(segment.chunk_key) if segment.tier == L2 else None
            by_tier[segment.tier].append(
                Move(
                    buffer_offset=buffer_offset,
                    nbytes=segment.nbytes,
                    l1_offset=(
                        layout.absolute_offset(L1, tp_rank, lane)
                        if segment.tier == L1
                        else None
                        if promoted is None
                        else layout.absolute_offset(L1, tp_rank, promoted.offset)
                    ),
                    l2_offset=(
                        layout.absolute_offset(L2, tp_rank, lane)
                        if segment.tier == L2
                        else None
                    ),
                )
            )
        calls: list[Coroutine[Any, Any, None]] = [
            transfer.load(buffer, moves, tier)  # type: ignore[arg-type]
            for tier, moves in by_tier.items()
            if moves
        ]
        peer_ranks = [
            rank for rank in range(layout.tensor_parallel_size) if rank != tp_rank
        ]
        peers = [
            move
            for promotion in promotions
            for move in self._promotion_moves(promotion, peer_ranks)
        ]
        if peers:
            calls.append(transfer.promote(peers))
        tasks = [asyncio.ensure_future(call) for call in calls]
        error = await _outcome(tasks)
        if error is not None:
            raise error

    def _promotion_moves(
        self, promotion: Promotion, ranks: Sequence[int] | None = None
    ) -> list[Move]:
        """Return the L2 -> L1 moves of a promotion on rank lanes.

        Args:
            promotion: Admitted promotion.
            ranks: Lanes to fill; None means every rank.
        """
        nbytes = sum(promotion.record_sizes)
        layout = self._layout
        if ranks is None:
            ranks = range(layout.tensor_parallel_size)
        return [
            Move(
                buffer_offset=0,
                nbytes=nbytes,
                l1_offset=layout.absolute_offset(L1, rank, promotion.target.offset),
                l2_offset=layout.absolute_offset(L2, rank, promotion.source.offset),
            )
            for rank in ranks
        ]

    async def _start_readahead(self, windows: list[tuple[ChunkInfo, int, int]]) -> None:
        """Promote the L2-only chunks of a lookup's external window.

        Bounded by ``readahead piece * window`` bytes in flight; chunks are
        admitted in prompt order and promoted in groups of about one piece.

        Args:
            windows: ``_window_slots`` of the lookup.

        Async/thread-safety:
            Runs on the event loop; the promotions run as background tasks.
            Loads of a promoting chunk wait for it.
        """
        tp = self._layout.tensor_parallel_size
        keys: list[str] = []
        cost: dict[str, int] = {}
        budget = self._readahead_budget - self._readahead_inflight
        candidates = [
            key
            for chunk, first, count in windows
            for key, _index, _count in (
                self._core.resolve_slots(int(chunk.start_slot) + first, count) or []
            )
        ]
        for key in candidates:
            records = self._core.chunk_records(key)
            if records is None or self._core.full_tiers(key) != (L2,):
                continue
            nbytes = sum(records.record_sizes) * tp
            if nbytes > budget:
                break
            budget -= nbytes
            keys.append(key)
            cost[key] = nbytes
        if not keys:
            return
        group: list[Promotion] = []
        group_bytes = 0
        for promotion in await self._core.begin_promotions(keys):
            if promotion is None:
                continue
            nbytes = cost[promotion.chunk_key]
            if group and group_bytes + nbytes > self._readahead_piece:
                self._spawn_readahead(group, group_bytes)
                group, group_bytes = [], 0
            group.append(promotion)
            group_bytes += nbytes
        if group:
            self._spawn_readahead(group, group_bytes)

    def _spawn_readahead(self, promotions: list[Promotion], nbytes: int) -> None:
        """Start one readahead promotion group.

        Args:
            promotions: Admitted promotions.
            nbytes: Their bytes over all ranks.
        """
        self._readahead_inflight += nbytes
        self._spawn(self._run_readahead(promotions, nbytes))

    async def _run_readahead(self, promotions: list[Promotion], nbytes: int) -> None:
        """Read one readahead group into L1 and finish its promotions.

        Args:
            promotions: Admitted promotions.
            nbytes: Their bytes over all ranks.
        """
        ok = False
        try:
            moves = [m for p in promotions for m in self._promotion_moves(p)]
            await self._ensure_transfer().promote(moves)
            ok = True
        finally:
            for promotion in promotions:
                self._core.finish_promotion(promotion, ok)
            self._readahead_inflight -= nbytes
            self._metrics.counter(
                "daser_readahead_bytes_total",
                "Lookup readahead bytes by outcome.",
            ).inc(nbytes, labels={"outcome": "read" if ok else "dropped"})

    # ------------------------------------------------------------------
    # Demotion helpers (exclusive tiers)
    # ------------------------------------------------------------------

    def _kick_demotions(self) -> None:
        """Start demotions when L1 lacks free or clean bytes.

        Registered as the core's demotion listener; also re-run when a
        demotion batch ends. At most one filler task runs at a time.
        """
        if (
            self._demote_scheduled
            or self._transfer is None
            or self._demote_inflight >= self._demote_budget
            or not self._core.demotion_deficit()
        ):
            return
        self._demote_scheduled = True
        self._spawn(self._fill_demotions())

    async def _fill_demotions(self) -> None:
        """Admit demotion batches until the deficit or the window is covered."""
        try:
            while (
                self._demote_inflight < self._demote_budget
                and self._core.demotion_deficit()
            ):
                demotions = await self._core.begin_demotions(self._demote_piece)
                if not demotions:
                    break
                lane_bytes = sum(d.source.length for d in demotions)
                self._demote_inflight += lane_bytes
                self._spawn(self._run_demotions(demotions, lane_bytes))
        finally:
            self._demote_scheduled = False

    async def _run_demotions(self, demotions: list[Demotion], lane_bytes: int) -> None:
        """Write one demotion batch to L2 and finish its demotions.

        Args:
            demotions: Admitted demotions.
            lane_bytes: Their L1 lane bytes.
        """
        ok = False
        try:
            moves = [m for d in demotions for m in self._demotion_moves(d)]
            await self._ensure_transfer().demote(moves)
            ok = True
        finally:
            for demotion in demotions:
                self._core.finish_demotion(demotion, ok)
            self._demote_inflight -= lane_bytes
            self._room_changed.set()
            self._kick_demotions()

    def _demotion_moves(self, demotion: Demotion) -> list[Move]:
        """Return the L1 -> L2 moves of a demotion on every rank lane.

        Args:
            demotion: Admitted demotion.
        """
        nbytes = sum(demotion.record_sizes)
        layout = self._layout
        return [
            Move(
                buffer_offset=0,
                nbytes=nbytes,
                l1_offset=layout.absolute_offset(L1, rank, demotion.source.offset),
                l2_offset=layout.absolute_offset(L2, rank, demotion.target.offset),
            )
            for rank in range(layout.tensor_parallel_size)
        ]

    # ------------------------------------------------------------------
    # Lease helpers
    # ------------------------------------------------------------------

    def _lease_ready(self, state: _LeaseState) -> asyncio.Task[None]:
        """Return the task bringing a lease's chunks into pinned L1 segments.

        Args:
            state: Active lease.
        """
        if state.task is None:
            state.task = asyncio.ensure_future(self._prepare_lease(state))
            self._tasks.add(state.task)

            def done(finished: asyncio.Task[None]) -> None:
                self._tasks.discard(finished)
                # Waiters report the failure; a shielded task whose waiter
                # left must not log "exception was never retrieved".
                if not finished.cancelled():
                    finished.exception()

            state.task.add_done_callback(done)
        return state.task

    async def _prepare_lease(self, state: _LeaseState) -> None:
        """Promote a lease's L2-only chunks and pin its L1 segments.

        Args:
            state: Active lease.

        Raises:
            ChunkUnavailable: If the lease was released, a chunk left the
                index, or a chunk could not be admitted into L1.
        """
        if self._stopping or state.released:
            raise ChunkUnavailable(f"lease {state.lease_id} was released")
        if state.promote_keys:
            await self._admit_lease(state)
            if self._core.exclusive_tiers:
                await self._wait_l1_room(state)
            promotions = [
                p
                for p in await self._core.begin_promotions(state.promote_keys)
                if p is not None
            ]
            ok = False
            try:
                if promotions:
                    await self._ensure_transfer().promote(
                        [m for p in promotions for m in self._promotion_moves(p)]
                    )
                ok = True
            finally:
                for promotion in promotions:
                    self._core.finish_promotion(promotion, ok)
        segments: list[LoadSegment] = []
        slot_ids: list[int] = []
        try:
            for start_slot, num_slots in state.windows:
                planned = await self._core.plan_load(start_slot, num_slots)
                segments.extend(planned)
                slot_ids.extend(range(start_slot, start_slot + num_slots))
            if state.released:
                raise ChunkUnavailable(f"lease {state.lease_id} was released")
            if any(segment.tier != L1 for segment in segments):
                raise ChunkUnavailable(
                    f"lease {state.lease_id} could not bring every chunk into L1"
                )
        except BaseException:
            self._core.release_load(segments)
            raise
        lanes = []
        sizes = []
        for segment in segments:
            offsets = record_offsets(segment.record_sizes)
            lanes.extend(segment.lane_offset + offset for offset in offsets)
            sizes.extend(segment.record_sizes)
        order = numpy.argsort(numpy.asarray(slot_ids, dtype=numpy.int64))
        state.slot_ids = numpy.asarray(slot_ids, dtype=numpy.int64)[order]
        state.lane_offsets = numpy.asarray(lanes, dtype=numpy.int64)[order]
        state.record_sizes = numpy.asarray(sizes, dtype=numpy.int64)[order]
        state.segments = segments

    async def _admit_lease(self, state: _LeaseState) -> None:
        """Wait until a lease fits the prefetch lease budget, then count it.

        A lease is admitted alone when no other lease is counted, so an
        oversized request still makes progress.

        Args:
            state: Lease whose chunks need promotion.

        Raises:
            ChunkUnavailable: If the lease is released while waiting.
        """
        while (
            self._lease_budget
            and self._leased_bytes
            and self._leased_bytes + state.nbytes > self._lease_budget
        ):
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._lease_waiters.append(waiter)
            try:
                await waiter
            finally:
                if waiter in self._lease_waiters:
                    self._lease_waiters.remove(waiter)
            if state.released:
                raise ChunkUnavailable(f"lease {state.lease_id} was released")
        state.admitted = True
        self._leased_bytes += state.nbytes

    async def _wait_l1_room(self, state: _LeaseState) -> None:
        """Wait until L1 can take a lease's promotions without a demotion.

        With exclusive tiers L1 admits a promotion only into free or clean
        room; dirty chunks leave through demotions. A lease promoting more
        than that room would promote part of its chunks (removing their L2
        copies), fail, and demote them again. The lease raises the demotion
        target by the bytes it needs and waits for the room instead.

        Args:
            state: Admitted lease with chunks to promote.

        Raises:
            ChunkUnavailable: If the lease is released, a chunk left the
                index, or the room does not appear within
                ``_LEASE_ROOM_WAIT_S`` (nothing was promoted; the request
                loads without the lease).
        """
        need = 0
        for key in state.promote_keys:
            records = self._core.chunk_records(key)
            if records is None:
                raise ChunkUnavailable(f"chunk {key} has no tier copy")
            need += self._layout.extent_length(L1, records.record_sizes)
        if self._core.l1_room() >= need:
            return
        self._core.add_l1_demand(need)
        deadline = time.monotonic() + _LEASE_ROOM_WAIT_S
        try:
            while self._core.l1_room() < need:
                if state.released:
                    raise ChunkUnavailable(f"lease {state.lease_id} was released")
                left = deadline - time.monotonic()
                if left <= 0:
                    self._metrics.counter(
                        "daser_prefetch_room_timeouts_total",
                        "Leases that gave up waiting for L1 room.",
                    ).inc()
                    raise ChunkUnavailable(
                        f"lease {state.lease_id} found no L1 room for {need} bytes"
                    )
                self._room_changed.clear()
                # asyncio.TimeoutError is not the builtin before Python 3.11.
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(
                        self._room_changed.wait(), min(left, _LEASE_ROOM_POLL_S)
                    )
        finally:
            self._core.add_l1_demand(-need)

    def _release_lease(self, lease_id: str) -> None:
        """Retire a lease, releasing its pins after any active copies; idempotent.

        Args:
            lease_id: Lease to release.

        Async/thread-safety:
            Synchronous on the event loop. A preparation still running sees
            ``released`` and releases what it pinned itself.
        """
        state = self._leases.pop(lease_id, None)
        if state is None:
            return
        state.released = True
        if not state.active_loads:
            self._finish_lease_release(state)

    def _finish_lease_release(self, state: _LeaseState) -> None:
        """Release pins once neither admission nor an active load needs them."""
        for lease in state.leases:
            self._core.unpin_lease(lease)
        self._core.release_load(state.segments)
        state.segments = []
        if state.admitted:
            self._leased_bytes -= state.nbytes
            state.admitted = False
        for waiter in self._lease_waiters:
            if not waiter.done():
                waiter.set_result(None)
        self._lease_waiters.clear()
        self._room_changed.set()

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _spawn(self, body: Coroutine[Any, Any, Any]) -> asyncio.Task[Any]:
        """Run a background task tracked for ``drain_transfer``.

        Args:
            body: Coroutine to run.

        Returns:
            The task; its failure is logged.
        """
        task = asyncio.get_running_loop().create_task(body)
        self._tasks.add(task)

        def done(finished: asyncio.Task[Any]) -> None:
            self._tasks.discard(finished)
            if not finished.cancelled() and finished.exception() is not None:
                logger.warning(
                    "[IPC] background transfer task failed: %r", finished.exception()
                )

        task.add_done_callback(done)
        return task

    def _register_staging(
        self,
        msg: dict[str, Any],
        buffers: dict[tuple[int, int], "_CachedCudaArray"],
    ) -> dict[str, Any]:
        """Open a worker pool mapping; its owner retains it until server shutdown."""
        payload = msg.get("payload", {})
        buffer_index = int(payload["buffer_index"])
        producer_pid = int(payload["producer_pid"])
        if buffer_index < 0 or int(payload["allocation_bytes"]) <= 0:
            raise ValueError("staging index must be non-negative and capacity positive")
        buffer_key = (producer_pid, buffer_index)
        opened = self._open_cuda_ipc_payload(
            payload=payload,
            nbytes_key="allocation_bytes",
            cache_mapping=False,
        )
        previous = buffers.get(buffer_key)
        buffers[buffer_key] = opened
        if previous is not None:
            previous.close()
        return {"ok": True}

    def _record_transfer_metrics(self, op: str, nbytes: int, elapsed_s: float) -> None:
        """Record one transfer operation and the backend's counters.

        Args:
            op: Transfer operation, ``load`` or ``store``.
            nbytes: Bytes transferred (one rank).
            elapsed_s: Operation latency in seconds.
        """
        labels = {"op": op}
        self._metrics.counter(
            "daser_transfer_operations_total",
            "Transfer operations by operation and status.",
        ).inc(labels={**labels, "status": "ok"})
        self._metrics.counter(
            "daser_transfer_bytes_total",
            "Transfer bytes by operation.",
        ).inc(nbytes, labels=labels)
        self._metrics.histogram(
            "daser_transfer_duration_seconds",
            "Transfer operation latency by operation.",
            buckets=(
                0.0005,
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
            ),
        ).observe(elapsed_s, labels=labels)
        self._metrics.histogram(
            "daser_transfer_chunk_size_bytes",
            "Transfer size per operation in bytes.",
            buckets=(65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456),
        ).observe(nbytes, labels=labels)
        self._record_backend_metrics()

    def _record_load_metrics(
        self, l1_segments: int, l2_segments: int, *, resolve_s: float, io_s: float
    ) -> None:
        """Count a load's tier hits and phase times.

        Args:
            l1_segments: Segments read from L1.
            l2_segments: Segments read from L2.
            resolve_s: Seconds resolving and pinning segments (includes
                waits for in-flight promotions).
            io_s: Seconds in transfer calls.
        """
        if l1_segments:
            self._metrics.counter(
                "daser_l1_hits_total", "Load segments read from L1."
            ).inc(l1_segments)
        if l2_segments:
            self._metrics.counter(
                "daser_l1_misses_total", "Load segments read from L2."
            ).inc(l2_segments)
        phases = self._metrics.counter(
            "daser_load_phase_seconds_total", "Load wall time by phase."
        )
        phases.inc(resolve_s, labels={"phase": "resolve"})
        phases.inc(io_s, labels={"phase": "io"})

    def _record_packed_ratios(self, spans: list[dict[str, Any]]) -> None:
        """Observe each stored packed record's size relative to a raw slot.

        Args:
            spans: Spans of one store request.

        Async/thread-safety:
            Call on the server event loop. A retried store observes its
            records again.
        """
        packed = [span for span in spans if span.get("packed", False)]
        if not packed or self._local_slot_size <= 0:
            return
        histogram = self._metrics.histogram(
            "daser_packed_record_ratio",
            "Stored bytes of each packed record over one raw slot, by mode.",
            buckets=tuple(i / 100 for i in range(50, 101, 2)),
        )
        for span in packed:
            histogram.observe(
                int(span["nbytes"]) / self._local_slot_size,
                labels={"mode": str(span.get("mode", "compressed"))},
            )

    def _record_backend_metrics(self) -> None:
        """Publish deltas of the transfer backend's cumulative counters."""
        transfer = self._transfer
        if transfer is None:
            return
        stats = transfer.stats
        delta = stats.l2_write_bytes - self._prev_l2_write_bytes
        if delta > 0:
            self._metrics.counter(
                "daser_l2_write_bytes_total", "Bytes written to the L2 storage tier."
            ).inc(delta)
        self._prev_l2_write_bytes = stats.l2_write_bytes
        delta = stats.l2_reads - self._prev_l2_reads
        if delta > 0:
            self._metrics.counter(
                "daser_l2_reads_total", "Reads served from the L2 storage tier."
            ).inc(delta)
        self._prev_l2_reads = stats.l2_reads
        timing = dict(transfer.l2_timing_us)
        for key, current in timing.items():
            delta = current - self._prev_l2_timing_us.get(key, 0)
            if delta > 0:
                kind, phase = key.split("_", 1)
                self._metrics.counter(
                    "daser_l2_executor_seconds_total",
                    "L2 executor queue wait and native IO time.",
                ).inc(delta / 1_000_000, labels={"kind": kind, "phase": phase})
        self._prev_l2_timing_us = timing
        contention = dict(transfer.l2_read_contention)
        for (bucket, writing), current in contention.items():
            labels = {"inflight": str(bucket), "write": str(writing).lower()}
            prev = self._prev_l2_read_contention.get((bucket, writing), (0, 0, 0))
            for name, description, value in (
                (
                    "daser_l2_read_jobs_total",
                    "L2 read jobs by read concurrency and write activity at start.",
                    current[0] - prev[0],
                ),
                (
                    "daser_l2_read_io_seconds_total",
                    "L2 read native IO time by read concurrency and write activity.",
                    (current[1] - prev[1]) / 1_000_000,
                ),
                (
                    "daser_l2_read_bytes_total",
                    "L2 bytes read by read concurrency and write activity.",
                    current[2] - prev[2],
                ),
            ):
                if value > 0:
                    self._metrics.counter(name, description).inc(value, labels=labels)
        self._prev_l2_read_contention = contention
        for unit, current in (
            ("copies", stats.dst_copies),
            ("runs", stats.dst_copy_runs),
            ("bytes", stats.dst_copy_bytes),
        ):
            delta = current - self._prev_dst_copy.get(unit, 0)
            if delta > 0:
                self._metrics.counter(
                    "daser_dst_copy_total",
                    "Host-to-destination copy submissions, runs and bytes.",
                ).inc(delta, labels={"unit": unit})
            self._prev_dst_copy[unit] = current

    def _ensure_transfer(self) -> TransferLayer:
        """Return the server-owned transfer layer, creating it on first use.

        The backend is sized from the core's tier layout: the L1 arena holds
        every rank's L1 lane, L2 covers every rank's L2 lane.

        Raises:
            ValueError: For an unknown mode or a mode the tiers cannot use.

        Async/thread-safety:
            Protected by a threading lock so the eager ``initialize_transfer``
            thread and the event loop never create two instances.
        """
        if self._transfer is not None:
            return self._transfer
        with self._transfer_lock:
            if self._transfer is not None:
                return self._transfer
            layout = self._layout
            tp = layout.tensor_parallel_size
            mode = str(self._runtime_config.get("transfer_mode", "gds"))
            path = str(self._runtime_config.get("store_path", ""))
            if mode == "gds":
                if layout.enabled(L1) or not layout.enabled(L2):
                    raise ValueError("gds transfer needs L2 only (no L1 tier)")
                from daser.transfer.gds import GDSTransferLayer

                self._transfer = GDSTransferLayer(
                    path, l2_bytes=layout.l2_lane_bytes * tp
                )
            elif mode == "iouring":
                if not layout.enabled(L1):
                    raise ValueError("iouring transfer needs an L1 tier")
                skip_l2 = not layout.enabled(L2)
                self._transfer = TieredIOUringTransferLayer(
                    path,
                    l1_bytes=layout.l1_lane_bytes * tp,
                    staging_bytes=int(self._runtime_config.get("staging_bytes", 0)),
                    l2_bytes=layout.l2_lane_bytes * tp,
                    skip_l2=skip_l2,
                    l2_read_stripe_bytes=int(
                        self._runtime_config.get("l2_read_stripe_bytes", 0)
                    ),
                    l2_write_gate=os.environ.get(L2_WRITE_GATE_ENV) != "0",
                )
            else:
                raise ValueError(f"unknown transfer_mode: {mode}")
        return self._transfer

    def _chunk_payloads(self, chunks: list[ChunkInfo]) -> list[dict[str, Any]]:
        """Serialize lookup chunks; packed chunks carry ``compressed_slots``.

        Args:
            chunks: Lookup chunks.

        Returns:
            ``ChunkInfo.to_dict`` per chunk, plus ``compressed_slots`` of
            ``{slot_id, mode, stored_length}`` (one per slot) for results
            whose chunks all carry codec modes. A result may be a run of
            several adjacent chunks.
        """
        payloads: list[dict[str, Any]] = []
        for chunk in chunks:
            payload = chunk.to_dict()
            payloads.append(payload)
            if not self._packed_format:
                continue
            start = int(chunk.start_slot)
            slots: list[dict[str, Any]] = []
            for key, first, count in (
                self._core.resolve_slots(start, int(chunk.num_slots)) or []
            ):
                records = self._core.chunk_records(key)
                if records is None or not records.record_modes:
                    slots = []
                    break
                for index in range(first, first + count):
                    slots.append(
                        {
                            "slot_id": start + len(slots),
                            "mode": records.record_modes[index],
                            "stored_length": records.record_sizes[index],
                        }
                    )
            if slots:
                payload["compressed_slots"] = slots
        return payloads

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
                owned by the staging registries and must not be LRU-evicted.

        Returns:
            Cached or one-shot CUDA array wrapper for the mapped allocation.
        """
        local_ptr = None
        producer_pid = int(payload.get("producer_pid", -1))
        device_ptr = int(payload["device_ptr"])
        nbytes = int(payload[nbytes_key])
        pci_bus_id = (
            str(payload["device_pci_bus_id"])
            if "device_pci_bus_id" in payload
            else None
        )
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
                pci_bus_id,
            )
            cached = self._cuda_ipc_cache.get(key)
            if cached is None:
                self._evict_cuda_ipc_cache_if_needed()
                opened = open_cuda_ipc_buffer(
                    handle=payload["cuda_ipc_handle"],
                    nbytes=nbytes,
                    pci_bus_id=pci_bus_id,
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
            pci_bus_id=pci_bus_id,
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
        for registry in (self._load_staging_buffers, self._store_staging_buffers):
            for cached in registry.values():
                cached.close()
            registry.clear()
        for cached in self._cuda_ipc_cache.values():
            cached.close()
        self._cuda_ipc_cache.clear()

    def _payload_size(
        self, payload: dict[str, Any], spans: list[dict[str, Any]]
    ) -> int:
        """Return the destination size of a ``return_data`` load."""
        if "nbytes" in payload:
            return int(payload["nbytes"])
        return max(
            [0]
            + [
                int(span.get("target_offset", 0)) + int(span["nbytes"])
                for span in spans
            ]
        )


class _CachedCudaArray:
    """Sliceable wrapper for a cached CUDA IPC buffer."""

    def __init__(self, opened: Any) -> None:
        self._opened = opened
        self._copy_stream: Any | None = None

    def __getitem__(self, item: Any) -> Any:
        """Return a CuPy array slice."""
        return self._opened.array[item]

    @property
    def nbytes(self) -> int:
        """Return the byte length of the mapped CUDA view."""
        return int(self._opened.array.nbytes)

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
