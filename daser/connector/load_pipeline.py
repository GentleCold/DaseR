# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import threading
from typing import Any

from daser.connector.ipc_client import IPCClientAsync
from daser.connector.metadata import ReqLoadSpec
from daser.connector.worker_memory import FixedCudaStagingPool


class LoadPipeline:
    """Own load-side asyncio, IPC, staging, queue, and completion state.

    Args:
        socket_path: DaseR server Unix socket path.
        client_count: Number of independent async IPC connections.

    Async/thread-safety:
        Construction and public methods run on the vLLM worker thread. Async
        IPC runs exclusively on the private ``daser-load-io`` thread.
    """

    def __init__(self, socket_path: str, client_count: int) -> None:
        self._clients = [IPCClientAsync(socket_path) for _ in range(client_count)]
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="daser-load-io",
        )
        self._queue: asyncio.Queue[Any] | None = None
        self._queue_lock = threading.Lock()
        self._dispatcher_future: Any | None = None
        self._pending: dict[str, Any] = {}
        self._invalid_block_ids: set[int] = set()
        self._staging_pool: FixedCudaStagingPool | None = None
        self._staging_registered = False
        self._thread.start()

    @property
    def pending(self) -> dict[str, Any]:
        """Return worker-thread-owned request completion records."""
        return self._pending

    @property
    def invalid_block_ids(self) -> set[int]:
        """Return load-error block IDs accumulated by completion polling."""
        return self._invalid_block_ids

    @property
    def staging_pool(self) -> FixedCudaStagingPool | None:
        """Return the configured fixed staging pool, if registered."""
        return self._staging_pool

    @property
    def staging_registered(self) -> bool:
        """Return whether every fixed staging buffer has CUDA IPC registration."""
        return self._staging_registered

    @staging_registered.setter
    def staging_registered(self, value: bool) -> None:
        self._staging_registered = value

    def set_staging_pool(self, pool: FixedCudaStagingPool) -> None:
        """Install the fixed load staging pool before request traffic."""
        self._staging_pool = pool
        self._staging_registered = False

    def submit(self, coro: Any) -> Any:
        """Submit a coroutine to the private load event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def client(self, buffer_index: int | None = None) -> IPCClientAsync:
        """Return the IPC client assigned to a staging buffer index."""
        index = 0 if buffer_index is None else int(buffer_index)
        return self._clients[index % len(self._clients)]

    def clients(self) -> tuple[IPCClientAsync, ...]:
        """Return all load IPC clients for startup and shutdown orchestration."""
        return tuple(self._clients)

    def ensure_queue(self) -> asyncio.Queue[Any]:
        """Create the load request queue on its owning event loop once."""
        with self._queue_lock:
            if self._queue is None:
                future = self.submit(self._create_queue())
                self._queue = future.result(timeout=5.0)
            return self._queue

    def enqueue(self, item: Any) -> None:
        """Append one request from the worker thread without blocking."""
        queue = self.ensure_queue()
        self._loop.call_soon_threadsafe(queue.put_nowait, item)

    def ensure_dispatcher(self, coro: Any) -> None:
        """Start the single queue dispatcher when none is active."""
        with self._queue_lock:
            if (
                self._dispatcher_future is not None
                and not self._dispatcher_future.done()
            ):
                coro.close()
                return
            self._dispatcher_future = self.submit(coro)

    def shutdown(self) -> None:
        """Drain queue ownership, close IPC clients, and stop the load loop."""
        if self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        if self._dispatcher_future is not None:
            self._dispatcher_future.result(timeout=120.0)
        for client in self._clients:
            self.submit(client.close()).result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    async def _create_queue(self) -> asyncio.Queue[Any]:
        return asyncio.Queue()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


@dataclass(frozen=True)
class LoadCopyRun:
    """Describe one contiguous staging range with a shared KV transform."""

    start: int
    end: int
    block_ids: list[int]
    pos_offset: int


def build_load_read_plan(
    reqs_to_load: dict[str, ReqLoadSpec],
    slot_size: int,
    include_req_ids: bool = False,
) -> tuple[int, list[dict[str, int]], list[Any]]:
    """Build one combined server read and staging restore plan.

    Args:
        reqs_to_load: Request IDs mapped to scheduler load specifications.
        slot_size: Bytes stored for one rank-local KV slot.
        include_req_ids: Include request IDs in restore ranges when true.

    Returns:
        Total bytes, server read spans, and ranges mapping staging back to
        request specifications.

    Async/thread-safety:
        Pure CPU planning; safe to call from worker or load-loop threads.
    """
    total_bytes = 0
    spans: list[dict[str, int]] = []
    per_req_ranges: list[Any] = []
    source_ranges: dict[tuple[str, int, int, int, int], tuple[int, int]] = {}
    for req_id, spec in reqs_to_load.items():
        num_slots = len(spec.block_ids)
        if num_slots == 0:
            continue
        nbytes = num_slots * slot_size
        source_key = (
            spec.chunk_key,
            spec.start_slot,
            spec.num_slots,
            spec.file_offset,
            nbytes,
        )
        existing = source_ranges.get(source_key)
        if existing is None:
            start = total_bytes
            end = start + nbytes
            spans.append(
                {
                    "target_offset": start,
                    "nbytes": nbytes,
                    "file_offset": spec.file_offset,
                }
            )
            source_ranges[source_key] = (start, end)
            total_bytes = end
        else:
            start, end = existing
        if include_req_ids:
            per_req_ranges.append((start, end, req_id, spec))
        else:
            per_req_ranges.append((start, end, spec))
    return total_bytes, spans, per_req_ranges


def build_load_read_batches(
    reqs_to_load: dict[str, ReqLoadSpec],
    slot_size: int,
    max_batch_bytes: int,
    include_req_ids: bool = False,
) -> list[tuple[int, list[dict[str, int]], list[Any]]]:
    """Split load work into staging-capacity-bounded read plans.

    Args:
        reqs_to_load: Request IDs mapped to scheduler load specifications.
        slot_size: Bytes stored for one rank-local KV slot.
        max_batch_bytes: Maximum staging bytes in one transfer.
        include_req_ids: Include request IDs in restore ranges when true.

    Returns:
        Ordered read plans; requests larger than the cap are split on slot
        boundaries.

    Async/thread-safety:
        Pure CPU planning; safe to call from worker or load-loop threads.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive")
    max_slots = max(1, max_batch_bytes // slot_size)
    batches: list[tuple[int, list[dict[str, int]], list[Any]]] = []
    current: dict[str, ReqLoadSpec] = {}
    current_slots = 0
    synthetic_id = 0

    def flush() -> None:
        nonlocal current, current_slots
        if current:
            batches.append(
                build_load_read_plan(
                    current,
                    slot_size,
                    include_req_ids=include_req_ids,
                )
            )
        current = {}
        current_slots = 0

    for req_id, spec in reqs_to_load.items():
        cursor = 0
        while cursor < len(spec.block_ids):
            if current_slots >= max_slots:
                flush()
            available = max_slots - current_slots
            take = min(available, len(spec.block_ids) - cursor)
            if take <= 0:
                flush()
                continue
            part = spec.block_ids[cursor : cursor + take]
            batch_spec = replace(
                spec,
                start_slot=spec.start_slot + cursor,
                num_slots=take,
                block_ids=part,
                file_offset=spec.file_offset + cursor * slot_size,
            )
            key = (
                req_id
                if cursor == 0 and take == len(spec.block_ids)
                else f"{req_id}#{synthetic_id}"
            )
            synthetic_id += 1
            current[key] = batch_spec
            current_slots += take
            cursor += take
    flush()
    return batches


def build_load_copy_runs(
    per_req_ranges: list[tuple[int, int, ReqLoadSpec]],
) -> list[LoadCopyRun]:
    """Merge adjacent restore ranges with the same position transform.

    Args:
        per_req_ranges: Per-request staging ranges from a read plan.

    Returns:
        Ordered contiguous copy runs.

    Async/thread-safety:
        Pure CPU planning; safe to call from worker or load-loop threads.
    """
    runs: list[LoadCopyRun] = []
    run_start = -1
    run_end = -1
    run_pos_offset = 0
    run_block_ids: list[int] = []

    def flush() -> None:
        nonlocal run_start, run_end, run_pos_offset, run_block_ids
        if run_start >= 0 and run_block_ids:
            runs.append(
                LoadCopyRun(
                    start=run_start,
                    end=run_end,
                    block_ids=run_block_ids,
                    pos_offset=run_pos_offset,
                )
            )
        run_start = -1
        run_end = -1
        run_pos_offset = 0
        run_block_ids = []

    for item in per_req_ranges:
        if len(item) == 3:
            start, end, spec = item
        else:
            start, end, _req_id, spec = item
        if not spec.block_ids:
            continue
        if run_start >= 0 and start == run_end and spec.pos_offset == run_pos_offset:
            run_end = end
            run_block_ids.extend(spec.block_ids)
            continue
        flush()
        run_start = start
        run_end = end
        run_pos_offset = spec.pos_offset
        run_block_ids = list(spec.block_ids)
    flush()
    return runs
