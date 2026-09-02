# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, replace
import os
import threading
import time
from typing import Any

# Third Party
import cupy
import torch

from daser.config import (
    STORAGE_FORMAT_COMPRESSED_ONLINE,
    STORAGE_FORMAT_COMPRESSED_READ_ONLY,
)
from daser.connector.helpers import base_req_id
from daser.connector.ipc_client import IPCClientAsync
from daser.connector.metadata import ReqLoadSpec
from daser.connector.worker.memory import (
    CudaStagingLease,
    FixedCudaStagingPool,
)
from daser.connector.worker.staging import copy_staging_to_kv_cache
from daser.logging import init_logger
from daser.ops.compressed_kv import (
    FusedCompressedKVDecoder,
    compressed_slot_metadata,
)
from daser.transfer.cuda_ipc import (
    cuda_allocation_base_and_offset,
    cuda_array_device_id,
    cuda_array_pointer,
    export_cuda_ipc_handle,
)

logger = init_logger(__name__)

_LOAD_DISPATCH_WAIT_TIMEOUT_S = 0.001
# CUDA event queries are cheap and non-blocking.  Keep their polling cadence
# separate from IPC completion dispatch: a millisecond dispatcher timeout is
# useful for batching transfer futures, but it adds avoidable quantization to
# the decoder completion that gates vLLM prefill and staging-ring reuse.
_LOAD_EVENT_POLL_INTERVAL_S = 0.0001
_LoadBatch = tuple[int, list[dict[str, int]], list[Any]]


@dataclass
class _LoadRequest:
    """Own one base request's specs and completion future."""

    req_id: str
    specs: dict[str, ReqLoadSpec]
    future: Future[None]

    @property
    def block_ids(self) -> list[int]:
        """Return all vLLM blocks affected by this request."""
        return [block_id for spec in self.specs.values() for block_id in spec.block_ids]

    @property
    def lease_id(self) -> str | None:
        """Return the single request lease shared by all grouped load specs."""
        lease_ids = {spec.lease_id for spec in self.specs.values() if spec.lease_id}
        if len(lease_ids) > 1:
            raise ValueError(f"grouped load has conflicting lease IDs: {lease_ids}")
        return next(iter(lease_ids), None)


@dataclass
class _InflightLoadBatch:
    """Hold one submitted load batch and its fixed staging lease."""

    total_bytes: int
    per_req_ranges: list[Any]
    staging_lease: CudaStagingLease
    future: Any
    submitted_at: float
    buffer_index: int
    response: dict[str, Any] | None = None
    wait_ms: float = 0.0
    ipc_ms: float = 0.0
    copy_ms: float = 0.0
    copies: int = 0
    copy_runs: int = 0
    restore_future: Any | None = None


@dataclass(frozen=True)
class _LoadBatchTiming:
    """Record transfer and restore accounting for one load batch."""

    bytes: int
    copies: int
    copy_runs: int
    ipc_ms: float
    wait_ms: float
    copy_ms: float
    worker_sync_ms: float
    transfer_open_ms: float
    transfer_load_ms: float
    transfer_sync_ms: float
    l1_hits: int
    l1_misses: int
    l2_reads: int


@dataclass
class _InflightRequestLoad:
    """Track active and remaining load batches for one request."""

    request: _LoadRequest
    buffer_index: int
    batches: deque[_LoadBatch]
    active: _InflightLoadBatch | None
    completed: list[_LoadBatchTiming]


class LoadPipeline:
    """Own the complete worker load state machine.

    Args:
        socket_path: DaseR server Unix socket path.
        client_count: Independent load IPC lanes and maximum inflight requests.

    Async/thread-safety:
        Public methods are called on the vLLM worker thread. Queue dispatch,
        IPC, and CUDA restore execute on the private load thread.
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
        self._pending: dict[str, _LoadRequest] = {}
        self._invalid_block_ids: set[int] = set()
        self._staging_pool: FixedCudaStagingPool | None = None
        self._staging_registered = False
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._local_slot_size = 0
        self._rank_stride_bytes = 0
        self._tp_rank = 0
        self._load_key_scale = 1.0
        self._load_value_scale = 1.0
        self._rope_delta_scale = 1.0
        self._rope_base = 10000.0
        self._rope_rotary_dim = 0
        self._rope_is_neox_style = True
        self._cuda_stream: torch.cuda.Stream | None = None
        self._compressed_decoder: FusedCompressedKVDecoder | None = None
        self._thread.start()

    def configure(
        self,
        *,
        kv_caches: dict[str, torch.Tensor],
        layer_names: list[str],
        local_slot_size: int,
        rank_stride_bytes: int,
        tp_rank: int,
        staging_pool: FixedCudaStagingPool,
        load_key_scale: float,
        load_value_scale: float,
        rope_delta_scale: float,
        rope_base: float,
        rope_rotary_dim: int,
        rope_is_neox_style: bool,
    ) -> None:
        """Configure immutable KV layout and transform state.

        Args:
            kv_caches: Registered vLLM KV tensors.
            layer_names: Stable storage layer order.
            local_slot_size: Bytes stored per slot by this TP rank.
            rank_stride_bytes: Byte distance between rank lanes.
            tp_rank: Current tensor-parallel rank.
            staging_pool: Fixed load staging buffers.
            load_key_scale: Load-time key scaling factor.
            load_value_scale: Load-time value scaling factor.
            rope_delta_scale: Position-offset scaling factor.
            rope_base: RoPE theta/base.
            rope_rotary_dim: Number of dimensions covered by RoPE.
            rope_is_neox_style: Whether RoPE uses split-half rotation.

        Async/thread-safety:
            Called once on the worker thread before request traffic.
        """
        self._kv_caches = kv_caches
        self._layer_names = list(layer_names)
        self._local_slot_size = local_slot_size
        self._rank_stride_bytes = rank_stride_bytes
        self._tp_rank = tp_rank
        self._staging_pool = staging_pool
        self._staging_registered = False
        self._load_key_scale = load_key_scale
        self._load_value_scale = load_value_scale
        self._rope_delta_scale = rope_delta_scale
        self._rope_base = rope_base
        self._rope_rotary_dim = rope_rotary_dim
        self._rope_is_neox_style = rope_is_neox_style

    def initialize_transfer(self) -> None:
        """Initialize load IPC lanes and register staging buffers.

        Async/thread-safety:
            Called on the worker thread during startup. IPC runs on the load
            loop and is joined before this method returns.
        """
        for client in self._clients:
            self._submit(client.init_transfer()).result(timeout=120.0)
        self._register_staging_buffers()

    def configure_compression(
        self,
        *,
        storage_format: str,
        codebooks: bytes,
        tile_scalars: int,
    ) -> None:
        """Configure immutable fused restore state from server runtime config.

        Args:
            storage_format: ``raw`` or ``compressed-read-only``.
            codebooks: Plane-major static high-byte codebooks.
            tile_scalars: Codec tile size encoded in the side index.

        Async/thread-safety:
            Called on the worker thread before transfer initialization. Kernel
            compilation and metadata allocation complete before request timing.
        """
        if storage_format == "raw":
            self._compressed_decoder = None
            return
        if storage_format not in (
            STORAGE_FORMAT_COMPRESSED_READ_ONLY,
            STORAGE_FORMAT_COMPRESSED_ONLINE,
        ):
            raise ValueError(f"unknown storage format: {storage_format}")
        if self._staging_pool is None or len(self._kv_caches) != 1:
            raise ValueError("compressed restore requires cross-layer KV staging")
        kv_cache = next(iter(self._kv_caches.values()))
        minimum_record_bytes = max(4096, self._local_slot_size // 2)
        max_slots = max(1, self._staging_pool.buffer_bytes // minimum_record_bytes)
        self._compressed_decoder = FusedCompressedKVDecoder(
            kv_cache=kv_cache,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
            ring_depth=self._staging_pool.depth,
            max_slots_per_buffer=max_slots,
        )

    def configure_rank_geometry(self, rank_stride_bytes: int, tp_rank: int) -> None:
        """Apply server-finalized tensor-parallel lane geometry.

        Args:
            rank_stride_bytes: Byte distance between server-owned rank lanes.
            tp_rank: Current tensor-parallel rank.

        Async/thread-safety:
            Called on the worker thread after runtime-config refresh and before
            any load is submitted.
        """
        self._rank_stride_bytes = rank_stride_bytes
        self._tp_rank = tp_rank

    def start(self, reqs_to_load: dict[str, ReqLoadSpec]) -> None:
        """Queue request loads for background transfer and restore.

        Args:
            reqs_to_load: Scheduler load metadata keyed by request/spec ID.

        Async/thread-safety:
            Called from the worker thread. Queue dispatch, IPC, and restore run
            on the private load thread.
        """
        if not reqs_to_load:
            return
        if not self._layer_names or not self._kv_caches:
            self.mark_failed(reqs_to_load, "no registered KV cache layout")
            return
        queue = self._ensure_queue()
        grouped: dict[str, dict[str, ReqLoadSpec]] = {}
        for spec_id, spec in reqs_to_load.items():
            grouped.setdefault(base_req_id(spec_id), {})[spec_id] = spec
        for req_id, specs in grouped.items():
            request = _LoadRequest(req_id, specs, Future())
            self._pending[req_id] = request
            self._loop.call_soon_threadsafe(
                queue.put_nowait,
                request,
            )
        self._ensure_dispatcher()

    def mark_failed(
        self,
        reqs_to_load: dict[str, ReqLoadSpec],
        reason: str,
    ) -> None:
        """Record submission failures for completion polling.

        Args:
            reqs_to_load: Load specs that could not be submitted.
            reason: Diagnostic failure reason.

        Async/thread-safety:
            Called on the worker thread before background submission.
        """
        grouped: dict[str, dict[str, ReqLoadSpec]] = {}
        for spec_id, spec in reqs_to_load.items():
            grouped.setdefault(base_req_id(spec_id), {})[spec_id] = spec
        for req_id, specs in grouped.items():
            future: Future[None] = Future()
            future.set_exception(RuntimeError(reason))
            self._pending[req_id] = _LoadRequest(req_id, specs, future)

    def collect_finished(self) -> set[str]:
        """Collect completed loads without blocking the worker thread.

        Returns:
            Base request IDs whose load lifecycle completed in this poll.

        Async/thread-safety:
            Called on the worker thread; request futures provide cross-thread
            visibility from the load thread.
        """
        finished: set[str] = set()
        collected_futures: set[int] = set()
        for req_id, load in list(self._pending.items()):
            if not load.future.done():
                continue
            future_id = id(load.future)
            try:
                if future_id not in collected_futures:
                    load.future.result()
                    collected_futures.add(future_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[CONNECTOR] async load failed req=%s blocks=%s: %s",
                    req_id,
                    load.block_ids,
                    exc,
                )
                self._invalid_block_ids.update(load.block_ids)
                collected_futures.add(future_id)
            finally:
                del self._pending[req_id]
            finished.add(req_id)
        return finished

    def take_invalid_block_ids(self) -> set[int]:
        """Return and clear block IDs targeted by failed loads.

        Returns:
            vLLM block IDs that must be invalidated.

        Async/thread-safety:
            Called on the worker thread after ``collect_finished``.
        """
        invalid = set(self._invalid_block_ids)
        self._invalid_block_ids.clear()
        return invalid

    def shutdown(self) -> None:
        """Drain queue ownership, close IPC clients, and stop the load loop."""
        for load in {id(item.future): item for item in self._pending.values()}.values():
            if not load.future.done():
                try:
                    load.future.result(timeout=120.0)
                except Exception:  # noqa: BLE001
                    pass
        self.collect_finished()
        if self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)
        if self._dispatcher_future is not None:
            self._dispatcher_future.result(timeout=120.0)
        for client in self._clients:
            self._submit(client.close()).result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def _submit(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _client(self, buffer_index: int | None = None) -> IPCClientAsync:
        index = 0 if buffer_index is None else int(buffer_index)
        return self._clients[index % len(self._clients)]

    def _ensure_queue(self) -> asyncio.Queue[Any]:
        with self._queue_lock:
            if self._queue is None:
                self._queue = self._submit(self._create_queue()).result(timeout=5.0)
            return self._queue

    def _ensure_dispatcher(self) -> None:
        with self._queue_lock:
            if (
                self._dispatcher_future is not None
                and not self._dispatcher_future.done()
            ):
                return
            self._dispatcher_future = self._submit(self._run_dispatcher())

    async def _create_queue(self) -> asyncio.Queue[Any]:
        return asyncio.Queue()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _run_dispatcher(self) -> None:
        sample_tensor = next(iter(self._kv_caches.values()))
        if sample_tensor.device.type == "cuda":
            torch.cuda.set_device(sample_tensor.device)
            if self._cuda_stream is None:
                # Keep restore on the default-priority stream.  A low-priority
                # decoder can wait behind vLLM prefill and inflate cache-hit
                # TTFT even though the transfer itself is asynchronous.
                self._cuda_stream = torch.cuda.Stream(device=sample_tensor.device)
        queue = self._ensure_queue()
        if self._staging_pool is None:
            raise RuntimeError("load staging pool is not configured")
        free_buffers = deque(
            range(max(1, min(len(self._clients), self._staging_pool.depth)))
        )
        queued: deque[_LoadRequest] = deque()
        active: list[_InflightRequestLoad] = []
        try:
            while True:
                if not queued:
                    if active:
                        await self._drain_queue(queue, queued)
                    else:
                        item = await queue.get()
                        if item is None:
                            return
                        queued.append(item)
                while queued and free_buffers:
                    request = queued.popleft()
                    buffer_index = free_buffers.popleft()
                    try:
                        state = self._submit_request(request, buffer_index)
                    except BaseException as exc:
                        if not request.future.done():
                            request.future.set_exception(exc)
                        free_buffers.append(buffer_index)
                        continue
                    if state.active is None:
                        free_buffers.append(state.buffer_index)
                    else:
                        active.append(state)
                consumed = False
                for state in list(active):
                    active_batch = state.active
                    if active_batch is None:
                        continue
                    if (
                        active_batch.restore_future is None
                        and not active_batch.future.done()
                    ):
                        continue
                    if (
                        active_batch.restore_future is not None
                        and not active_batch.restore_future.done()
                    ):
                        continue
                    try:
                        reusable_buffer, request_done = self._consume_request(state)
                    except BaseException as exc:
                        if not state.request.future.done():
                            state.request.future.set_exception(exc)
                        active.remove(state)
                        free_buffers.append(state.buffer_index)
                        consumed = True
                        continue
                    if request_done:
                        active.remove(state)
                        free_buffers.append(reusable_buffer)
                    consumed = True
                if consumed:
                    continue
                if active:
                    await self._wait_for_completion(active)
        except BaseException as exc:
            for state in active:
                if not state.request.future.done():
                    state.request.future.set_exception(exc)
                if state.active is not None:
                    state.active.staging_lease.release()
            for item in queued:
                if not item.future.done():
                    item.future.set_exception(exc)
            raise

    async def _drain_queue(
        self,
        queue: asyncio.Queue[Any],
        queued: deque[_LoadRequest],
    ) -> None:
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                await queue.put(None)
                return
            queued.append(item)

    async def _wait_for_completion(
        self,
        active: list[_InflightRequestLoad],
    ) -> None:
        # Once a transfer completes, ``_consume_request`` launches the CUDA
        # restore and keeps the IPC future in the completed state. Waiting on
        # that stale future here makes the dispatcher return immediately on
        # every iteration while decode is still running, creating a tight
        # busy-loop and starving the restore-event poller. Track the current
        # stage instead: transfer completion before restore, restore completion
        # afterwards.
        wrapped: set[asyncio.Future[Any]] = set()
        for request in active:
            batch = request.active
            if batch is None:
                continue
            completion = (
                batch.restore_future
                if batch.restore_future is not None
                else batch.future
            )
            if isinstance(completion, asyncio.Future):
                wrapped.add(completion)
            else:
                wrapped.add(asyncio.wrap_future(completion))
        if not wrapped:
            await asyncio.sleep(0)
            return
        done, _pending = await asyncio.wait(
            wrapped,
            timeout=_LOAD_DISPATCH_WAIT_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for future in done:
            future.exception()
        if not done:
            await asyncio.sleep(0)

    async def _wait_for_cuda_event(self, event: Any) -> float:
        """Poll a restore event without blocking the load event loop.

        Args:
            event: CUDA event recorded after all restore writes for a batch.

        Returns:
            Milliseconds spent waiting for the event to become visible.

        Async/thread-safety:
            Runs on the private load event loop. ``Event.query`` is a
            non-blocking CUDA runtime call, and the short asyncio yield keeps
            transfer futures for other requests progressing while the decoder
            occupies the GPU.
        """
        wait_start = time.perf_counter()
        while not event.query():
            await asyncio.sleep(_LOAD_EVENT_POLL_INTERVAL_S)
        return (time.perf_counter() - wait_start) * 1000

    def _submit_request(
        self,
        request: _LoadRequest,
        buffer_index: int,
    ) -> _InflightRequestLoad:
        if self._staging_pool is None:
            raise RuntimeError("load staging pool is not configured")
        specs = {
            spec_id: (
                spec
                if spec.compressed_slots
                else replace(
                    spec,
                    file_offset=(
                        self._tp_rank * self._rank_stride_bytes
                        + spec.start_slot * self._local_slot_size
                    ),
                )
            )
            for spec_id, spec in request.specs.items()
        }
        batches = deque(
            build_load_read_batches(
                specs,
                self._local_slot_size,
                max_batch_bytes=self._staging_pool.buffer_bytes,
                include_req_ids=True,
            )
        )
        if not batches:
            request.future.set_result(None)
            return _InflightRequestLoad(request, buffer_index, batches, None, [])
        return _InflightRequestLoad(
            request=request,
            buffer_index=buffer_index,
            batches=batches,
            active=self._submit_batch(
                request.lease_id,
                batches.popleft(),
                buffer_index,
            ),
            completed=[],
        )

    def _submit_batch(
        self,
        lease_id: str | None,
        batch: _LoadBatch,
        buffer_index: int,
    ) -> _InflightLoadBatch:
        if self._staging_pool is None:
            raise RuntimeError("load staging pool is not configured")
        total_bytes, spans, per_req_ranges = batch
        lease = self._staging_pool.acquire_index(buffer_index, total_bytes)
        staging = lease.view
        if self._staging_registered:
            transfer = self._client(buffer_index).transfer_load_registered_cuda(
                buffer_index=buffer_index,
                producer_pid=os.getpid(),
                nbytes=total_bytes,
                spans=spans,
                lease_id=lease_id,
            )
        else:
            cp_staging = cupy.asarray(staging)
            device_ptr = cuda_array_pointer(cp_staging)
            allocation_base, allocation_offset = cuda_allocation_base_and_offset(
                device_ptr
            )
            transfer = self._client(buffer_index).transfer_load_cuda(
                cuda_ipc_handle=export_cuda_ipc_handle(cp_staging),
                nbytes=total_bytes,
                device_id=cuda_array_device_id(cp_staging),
                device_ptr=device_ptr,
                allocation_base_ptr=allocation_base,
                allocation_offset=allocation_offset,
                producer_pid=os.getpid(),
                spans=spans,
                lease_id=lease_id,
            )
        submitted_at = time.perf_counter()
        return _InflightLoadBatch(
            total_bytes=total_bytes,
            per_req_ranges=per_req_ranges,
            staging_lease=lease,
            future=self._submit(transfer),
            submitted_at=submitted_at,
            buffer_index=buffer_index,
        )

    def _consume_request(
        self,
        state: _InflightRequestLoad,
    ) -> tuple[int, bool]:
        active = state.active
        if active is None:
            return state.buffer_index, True
        if active.restore_future is None:
            self._start_restore(active)
            return state.buffer_index, False
        state.completed.append(self._finish_batch(active))
        if not state.batches:
            self._log_request_timing(state)
            if not state.request.future.done():
                state.request.future.set_result(None)
            return state.buffer_index, True
        state.active = self._submit_batch(
            state.request.lease_id,
            state.batches.popleft(),
            state.buffer_index,
        )
        return state.buffer_index, False

    def _start_restore(self, state: _InflightLoadBatch) -> None:
        """Launch one restore and defer lease reuse until its CUDA event.

        Args:
            state: Transfer-complete batch whose staging bytes are ready.

        Async/thread-safety:
            Called on the private load event loop. The restore is enqueued on
            the pipeline CUDA stream; no host stream synchronization is used.
        """
        wait_start = time.perf_counter()
        response = state.future.result(timeout=120.0)
        state.wait_ms = (time.perf_counter() - wait_start) * 1000
        state.ipc_ms = (time.perf_counter() - state.submitted_at) * 1000
        copy_start = time.perf_counter()
        state.copies, state.copy_runs, event = self._restore_batch(state)
        state.copy_ms = (time.perf_counter() - copy_start) * 1000
        if event is None:
            restore_future: Future[float] = Future()
            restore_future.set_result(0.0)
            state.restore_future = restore_future
        else:
            state.restore_future = self._submit(self._wait_for_cuda_event(event))
        state.response = response if isinstance(response, dict) else {}

    def _finish_batch(self, state: _InflightLoadBatch) -> _LoadBatchTiming:
        """Finalize timing and release staging after restore completion."""
        try:
            assert state.restore_future is not None
            restore_wait_ms = float(state.restore_future.result(timeout=120.0))
            payload = state.response or {}
            stats = payload.get("transfer_stats_delta", {})
            stats = stats if isinstance(stats, dict) else {}
            return _LoadBatchTiming(
                bytes=state.total_bytes,
                copies=state.copies,
                copy_runs=state.copy_runs,
                ipc_ms=state.ipc_ms,
                wait_ms=state.wait_ms,
                copy_ms=state.copy_ms,
                worker_sync_ms=restore_wait_ms,
                transfer_open_ms=float(payload.get("transfer_open_ms", 0.0)),
                transfer_load_ms=float(payload.get("transfer_load_ms", 0.0)),
                transfer_sync_ms=float(payload.get("transfer_sync_ms", 0.0)),
                l1_hits=int(stats.get("l1_hits", 0)),
                l1_misses=int(stats.get("l1_misses", 0)),
                l2_reads=int(stats.get("l2_reads", 0)),
            )
        finally:
            state.staging_lease.release()

    def _restore_batch(
        self,
        state: _InflightLoadBatch,
    ) -> tuple[int, int, torch.cuda.Event | None]:
        staging = state.staging_lease.view
        compressed = any(
            (item[2] if len(item) == 3 else item[3]).compressed_slots
            for item in state.per_req_ranges
        )
        if compressed:
            if not all(
                (item[2] if len(item) == 3 else item[3]).compressed_slots
                for item in state.per_req_ranges
            ):
                raise ValueError("raw and indexed load specs cannot share a batch")
            decoder = self._compressed_decoder
            if decoder is None or self._cuda_stream is None:
                raise RuntimeError("compressed decoder is not configured")
            offsets, block_ids, modes = compressed_slot_metadata(state.per_req_ranges)
            restored = decoder.decode(
                staging=staging,
                staging_offsets=offsets,
                block_ids=block_ids,
                modes=modes,
                buffer_index=state.buffer_index,
                stream=self._cuda_stream,
            )
            event = torch.cuda.Event(blocking=False)
            event.record(self._cuda_stream)
            return restored, 1, event
        runs = build_load_copy_runs(state.per_req_ranges)
        copies = 0
        context = (
            torch.cuda.stream(self._cuda_stream)
            if self._cuda_stream is not None
            else nullcontext()
        )
        with context:
            for run in runs:
                copies += copy_staging_to_kv_cache(
                    staging=staging[run.start : run.end],
                    kv_caches=self._kv_caches,
                    layer_names=self._layer_names,
                    block_ids=run.block_ids,
                    slot_size=self._local_slot_size,
                    load_key_scale=self._load_key_scale,
                    load_value_scale=self._load_value_scale,
                    pos_offset=run.pos_offset,
                    rope_delta_scale=self._rope_delta_scale,
                    rope_base=self._rope_base,
                    rope_rotary_dim=self._rope_rotary_dim,
                    rope_is_neox_style=self._rope_is_neox_style,
                )
        if self._cuda_stream is None:
            return copies, len(runs), None
        event = torch.cuda.Event(blocking=False)
        event.record(self._cuda_stream)
        return copies, len(runs), event

    def _log_request_timing(self, state: _InflightRequestLoad) -> None:
        rows = state.completed
        logger.debug(
            "[CONNECTOR] load timing req=%s batches=%d bytes=%d copy_runs=%d "
            "gpu_copies=%d ipc_ms=%.3f dispatcher_wait_ms=%.3f copy_ms=%.3f "
            "worker_sync_ms=%.3f transfer_open_ms=%.3f "
            "transfer_load_ms=%.3f transfer_sync_ms=%.3f l1_hits=%d "
            "l1_misses=%d l2_reads=%d",
            state.request.req_id,
            len(rows),
            sum(row.bytes for row in rows),
            sum(row.copy_runs for row in rows),
            sum(row.copies for row in rows),
            sum(row.ipc_ms for row in rows),
            sum(row.wait_ms for row in rows),
            sum(row.copy_ms for row in rows),
            sum(row.worker_sync_ms for row in rows),
            sum(row.transfer_open_ms for row in rows),
            sum(row.transfer_load_ms for row in rows),
            sum(row.transfer_sync_ms for row in rows),
            sum(row.l1_hits for row in rows),
            sum(row.l1_misses for row in rows),
            sum(row.l2_reads for row in rows),
        )

    def _register_staging_buffers(self) -> None:
        pool = self._staging_pool
        if self._staging_registered or pool is None:
            return
        try:
            for buffer_index in range(pool.depth):
                tensor = pool.buffer(buffer_index)
                cp_tensor = cupy.asarray(tensor)
                device_ptr = cuda_array_pointer(cp_tensor)
                allocation_base, allocation_offset = cuda_allocation_base_and_offset(
                    device_ptr
                )
                self._submit(
                    self._client(buffer_index).register_load_staging_cuda(
                        buffer_index=buffer_index,
                        cuda_ipc_handle=export_cuda_ipc_handle(cp_tensor),
                        allocation_bytes=int(tensor.numel()),
                        device_id=cuda_array_device_id(cp_tensor),
                        device_ptr=device_ptr,
                        allocation_base_ptr=allocation_base,
                        allocation_offset=allocation_offset,
                        producer_pid=os.getpid(),
                    )
                ).result(timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CONNECTOR] registered load staging unavailable; falling back "
                "to per-load CUDA IPC payloads: %s",
                exc,
            )
            self._staging_registered = False
            return
        self._staging_registered = True
        logger.info("[CONNECTOR] registered %d load staging buffers", pool.depth)


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
    source_ranges: dict[tuple[Any, ...], tuple[int, int]] = {}
    for req_id, spec in reqs_to_load.items():
        num_slots = len(spec.block_ids)
        if num_slots == 0:
            continue
        source_key: tuple[Any, ...]
        if spec.compressed_slots:
            if len(spec.compressed_slots) != num_slots:
                raise ValueError("compressed slot metadata does not match block IDs")
            physical = tuple(
                (slot.file_offset, slot.stored_length, slot.mode)
                for slot in spec.compressed_slots
            )
            nbytes = sum(slot.stored_length for slot in spec.compressed_slots)
            source_key = (spec.chunk_key, spec.start_slot, physical)
        else:
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
            if spec.compressed_slots:
                target_offset = start
                for slot in spec.compressed_slots:
                    spans.append(
                        {
                            "target_offset": target_offset,
                            "nbytes": slot.stored_length,
                            "file_offset": slot.file_offset,
                        }
                    )
                    target_offset += slot.stored_length
            else:
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
    batches: list[tuple[int, list[dict[str, int]], list[Any]]] = []
    current: dict[str, ReqLoadSpec] = {}
    current_bytes = 0
    synthetic_id = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if current:
            batches.append(
                build_load_read_plan(
                    current,
                    slot_size,
                    include_req_ids=include_req_ids,
                )
            )
        current = {}
        current_bytes = 0

    for req_id, spec in reqs_to_load.items():
        cursor = 0
        while cursor < len(spec.block_ids):
            take = 0
            part_bytes = 0
            while cursor + take < len(spec.block_ids):
                slot_bytes = (
                    spec.compressed_slots[cursor + take].stored_length
                    if spec.compressed_slots
                    else slot_size
                )
                if slot_bytes > max_batch_bytes:
                    raise ValueError("one load slot exceeds staging buffer capacity")
                if current_bytes + part_bytes + slot_bytes > max_batch_bytes:
                    break
                part_bytes += slot_bytes
                take += 1
            if take == 0:
                flush()
                continue
            part = spec.block_ids[cursor : cursor + take]
            compressed_slots = spec.compressed_slots[cursor : cursor + take]
            batch_spec = replace(
                spec,
                start_slot=spec.start_slot + cursor,
                num_slots=take,
                block_ids=part,
                file_offset=(
                    compressed_slots[0].file_offset
                    if compressed_slots
                    else spec.file_offset + cursor * slot_size
                ),
                compressed_slots=compressed_slots,
            )
            key = (
                req_id
                if cursor == 0 and take == len(spec.block_ids)
                else f"{req_id}#{synthetic_id}"
            )
            synthetic_id += 1
            current[key] = batch_spec
            current_bytes += part_bytes
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
