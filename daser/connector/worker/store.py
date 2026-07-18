# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import os
import threading
from typing import Any

import cupy
import torch

from daser.connector.helpers import base_req_id
from daser.connector.ipc_client import IPCClientAsync
from daser.connector.metadata import ReqStoreSpec, StoreWriteSpan
from daser.connector.worker.memory import (
    DEFAULT_STORE_STAGING_BYTES,
    CudaStagingLease,
    FixedCudaStagingPool,
)
from daser.connector.worker.staging import (
    CROSS_LAYER_KV_CACHE_KEY,
    copy_cross_layer_kv_cache_to_staging,
    copy_kv_cache_to_staging,
    record_cuda_event,
)
from daser.logging import init_logger
from daser.transfer.cuda_ipc import (
    cuda_allocation_base_and_offset,
    cuda_array_device_id,
    cuda_array_pointer,
    export_cuda_ipc_handle,
)

logger = init_logger(__name__)


@dataclass
class _DeferredFinishedSave:
    """Hold request store work until vLLM reports it finished."""

    reqs_to_store: dict[str, ReqStoreSpec]
    finished: bool = False
    future: Any | None = None


class StorePipeline:
    """Own the complete worker store state machine.

    Args:
        socket_path: DaseR server Unix socket path.

    Async/thread-safety:
        Public methods are called on the vLLM worker thread. Snapshot, IPC, and
        commit execute on the private store thread and its fixed CUDA stream.
    """

    def __init__(self, socket_path: str) -> None:
        self._client = IPCClientAsync(socket_path)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="daser-store-io",
        )
        self._staging_bytes = 0
        self._staging_pool: FixedCudaStagingPool | None = None
        self._pending_finished_saves: dict[str, _DeferredFinishedSave] = {}
        self._store_capacity = 1
        self._store_semaphore: asyncio.Semaphore | None = None
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._layer_idx_map: dict[str, int] = {}
        self._local_slot_size = 0
        self._rank_stride_bytes = 0
        self._tp_rank = 0
        self._tp_size = 1
        self._cuda_stream: torch.cuda.Stream | None = None
        self._thread.start()

    def configure(
        self,
        *,
        kv_caches: dict[str, torch.Tensor],
        layer_names: list[str],
        layer_idx_map: dict[str, int],
        local_slot_size: int,
        rank_stride_bytes: int,
        tp_rank: int,
        tp_size: int,
        staging_bytes: int,
        staging_pool: FixedCudaStagingPool,
    ) -> None:
        """Configure immutable KV layout and staging state.

        Args:
            kv_caches: Registered vLLM KV tensors.
            layer_names: Stable storage layer order.
            layer_idx_map: Layer names mapped to storage indices.
            local_slot_size: Bytes stored per slot by this TP rank.
            rank_stride_bytes: Byte distance between rank lanes.
            tp_rank: Current tensor-parallel rank.
            tp_size: Tensor-parallel world size.
            staging_bytes: Maximum bytes per store batch.
            staging_pool: Fixed store staging buffers.

        Async/thread-safety:
            Called once on the worker thread before request traffic.
        """
        self._kv_caches = kv_caches
        self._layer_names = list(layer_names)
        self._layer_idx_map = dict(layer_idx_map)
        self._local_slot_size = local_slot_size
        self._rank_stride_bytes = rank_stride_bytes
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._staging_bytes = staging_bytes
        self._staging_pool = staging_pool
        self._store_capacity = staging_pool.depth
        self._store_semaphore = None

    def initialize_transfer(self) -> None:
        """Initialize the store IPC transfer client on its event loop.

        Async/thread-safety:
            Called on the worker thread during startup and waits only for the
            store loop's initialization future.
        """
        self._submit(self._client.init_transfer()).result(timeout=120.0)

    def configure_rank_geometry(
        self,
        rank_stride_bytes: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Apply server-finalized tensor-parallel lane geometry.

        Args:
            rank_stride_bytes: Byte distance between server-owned rank lanes.
            tp_rank: Current tensor-parallel rank.
            tp_size: Tensor-parallel world size used for commit coordination.

        Async/thread-safety:
            Called on the worker thread after runtime-config refresh and before
            any store is submitted.
        """
        self._rank_stride_bytes = rank_stride_bytes
        self._tp_rank = tp_rank
        self._tp_size = tp_size

    def queue_finished(
        self,
        reqs_to_store: dict[str, ReqStoreSpec],
    ) -> None:
        """Queue stores until vLLM reports their requests finished.

        Args:
            reqs_to_store: Store metadata for the current worker step.
        Async/thread-safety:
            Called on the worker thread to accumulate immutable store intent.
            CUDA ordering is captured later, when vLLM reports completion.
        """
        for req_id, spec in reqs_to_store.items():
            base_id = base_req_id(req_id)
            save = self._pending_finished_saves.get(base_id)
            if save is None:
                save = _DeferredFinishedSave({})
                self._pending_finished_saves[base_id] = save
            save.reqs_to_store[req_id] = spec

    def collect_finished(self, finished_req_ids: set[str]) -> set[str]:
        """Submit newly finished stores and collect completed requests.

        Args:
            finished_req_ids: Requests vLLM finished in this step.

        Returns:
            Request IDs whose store and commit lifecycle has completed.

        Async/thread-safety:
            Called on the worker thread. Store work runs on the private loop.
        """
        finished: set[str] = set()
        for req_id in finished_req_ids:
            save = self._pending_finished_saves.get(req_id)
            if save is not None:
                save.finished = True
        for req_id, save in list(self._pending_finished_saves.items()):
            if not save.finished and save.future is None:
                continue
            if save.future is not None and save.future.done():
                try:
                    save.future.result(timeout=120.0)
                    finished.add(req_id)
                finally:
                    del self._pending_finished_saves[req_id]

        # Submit the entire finished FIFO here. The private event loop applies
        # the staging-depth bound, so one completed save releases the next
        # queued save without requiring another vLLM connector step.
        for save in self._pending_finished_saves.values():
            if save.finished and save.future is None:
                self._submit_save(save)
        return finished

    def shutdown(self) -> None:
        """Finish queued stores, close IPC, and stop the store loop.

        Async/thread-safety:
            Called once on the worker thread after request traffic stops.
        """
        first_error: BaseException | None = None
        try:
            submitted_ids = [
                req_id
                for req_id, save in self._pending_finished_saves.items()
                if save.future is not None
            ]
            for req_id in submitted_ids:
                save = self._pending_finished_saves[req_id]
                try:
                    if save.future is not None:
                        save.future.result(timeout=120.0)
                except BaseException as exc:  # preserve cleanup during shutdown
                    if first_error is None:
                        first_error = exc
                finally:
                    del self._pending_finished_saves[req_id]
            for req_id in list(self._pending_finished_saves):
                save = self._pending_finished_saves[req_id]
                try:
                    self._submit_save(save)
                    assert save.future is not None
                    save.future.result(timeout=120.0)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                finally:
                    del self._pending_finished_saves[req_id]
            try:
                self._submit(self._client.close()).result(timeout=5.0)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=5.0)
        if first_error is not None:
            raise first_error

    def _submit(self, coro: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _submit_save(self, save: _DeferredFinishedSave) -> None:
        """Capture producer ordering and enqueue one save in FIFO order."""
        sample = next(iter(self._kv_caches.values()), None)
        producer_event = record_cuda_event(sample) if sample is not None else None
        save.future = self._submit(self._run_bounded_save(save, producer_event))

    async def _run_bounded_save(
        self,
        save: _DeferredFinishedSave,
        producer_event: torch.cuda.Event | None,
    ) -> None:
        """Run one save while preserving FIFO staging-buffer admission.

        Args:
            save: Finished request save to transfer.
            producer_event: CUDA event ordering the KV snapshot.

        Async/thread-safety:
            Runs on the private store event loop. The semaphore prevents more
            saves from entering synchronous staging acquisition than there are
            preallocated buffers; semaphore waiters are admitted in FIFO order.
        """
        semaphore = self._store_semaphore
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(1, self._store_capacity))
            self._store_semaphore = semaphore
        async with semaphore:
            await self._store_finished_save(save, producer_event)

    def _plan_finished_save(
        self,
        save: _DeferredFinishedSave,
    ) -> list[tuple[list[int], list[StoreWriteSpan]]]:
        if not self._kv_caches or self._staging_pool is None:
            return []
        reqs_to_store = {
            req_id: replace(
                spec,
                file_offset=(
                    self._tp_rank * self._rank_stride_bytes
                    + spec.start_slot * self._local_slot_size
                ),
            )
            for req_id, spec in save.reqs_to_store.items()
        }
        return build_staging_store_batches(
            reqs_to_store,
            self._local_slot_size,
            max_batch_bytes=self._staging_bytes,
        )

    async def _store_finished_save(
        self,
        save: _DeferredFinishedSave,
        producer_event: torch.cuda.Event | None,
    ) -> None:
        for block_ids, spans in self._plan_finished_save(save):
            staged = self._stage_batch(block_ids, spans, producer_event)
            try:
                await self._write_cuda_buffer(staged)
            finally:
                staged.lease.release()

    def _stage_batch(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
        producer_event: torch.cuda.Event | None,
    ) -> "StagedStoreBatch":
        if self._staging_pool is None:
            raise RuntimeError("store staging pool is not configured")
        sample = next(iter(self._kv_caches.values()))
        nbytes = len(block_ids) * self._local_slot_size
        lease = self._staging_pool.acquire(nbytes)
        stream = self._cuda_stream
        if sample.device.type == "cuda" and stream is None:
            torch.cuda.set_device(sample.device)
            stream = torch.cuda.Stream(device=sample.device)
            self._cuda_stream = stream
        if stream is not None:
            if producer_event is not None:
                stream.wait_event(producer_event)
            with torch.cuda.stream(stream):
                self._copy_blocks(lease.view, block_ids, sample)
            stream.synchronize()
        else:
            self._copy_blocks(lease.view, block_ids, sample)
        return StagedStoreBatch(lease.view, spans, lease)

    def _copy_blocks(
        self,
        staging: torch.Tensor,
        block_ids: list[int],
        sample: torch.Tensor,
    ) -> None:
        block_index = torch.tensor(block_ids, dtype=torch.long, device=sample.device)
        cross_layer = self._kv_caches.get(CROSS_LAYER_KV_CACHE_KEY)
        if cross_layer is not None:
            copy_cross_layer_kv_cache_to_staging(
                staging,
                cross_layer,
                block_ids,
                len(self._layer_names),
                self._local_slot_size,
                block_index,
            )
            return
        for layer_name in self._layer_names:
            copy_kv_cache_to_staging(
                staging,
                self._kv_caches[layer_name],
                self._layer_idx_map[layer_name],
                block_ids,
                len(self._layer_names),
                self._local_slot_size,
                block_index,
            )

    async def _write_cuda_buffer(self, staged: "StagedStoreBatch") -> list[str]:
        if staged.buffer.device.type == "cuda":
            torch.cuda.set_device(staged.buffer.device)
        cp_buffer = cupy.asarray(staged.buffer)
        device_ptr = cuda_array_pointer(cp_buffer)
        allocation_base, allocation_offset = cuda_allocation_base_and_offset(device_ptr)
        return await self._client.transfer_store_cuda(
            cuda_ipc_handle=export_cuda_ipc_handle(cp_buffer),
            nbytes=staged.buffer.nbytes,
            device_id=cuda_array_device_id(cp_buffer),
            device_ptr=device_ptr,
            allocation_base_ptr=allocation_base,
            allocation_offset=allocation_offset,
            producer_pid=os.getpid(),
            tp_rank=self._tp_rank,
            tp_size=self._tp_size,
            spans=[
                {
                    "source_offset": span.source_offset,
                    "nbytes": span.nbytes,
                    "file_offset": span.file_offset,
                    "chunk_key": span.chunk_key,
                    "start_slot": span.start_slot,
                    "num_slots": span.num_slots,
                }
                for span in staged.spans
            ],
        )


@dataclass(frozen=True)
class StagedStoreBatch:
    """Hold a worker CUDA snapshot until its async store completes."""

    buffer: torch.Tensor
    spans: list[StoreWriteSpan]
    lease: CudaStagingLease


def build_staging_store_batches(
    reqs_to_store: dict[str, ReqStoreSpec],
    slot_size: int,
    max_batch_bytes: int = DEFAULT_STORE_STAGING_BYTES,
) -> list[tuple[list[int], list[StoreWriteSpan]]]:
    """Split store requests into bounded slot-major staging batches.

    Args:
        reqs_to_store: Request IDs mapped to store specifications.
        slot_size: Bytes stored for one rank-local KV slot.
        max_batch_bytes: Maximum GPU staging bytes in one batch.

    Returns:
        Ordered block ID and server write-span batches.

    Async/thread-safety:
        Pure CPU planning; safe to call from worker or store-loop threads.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    max_slots = max(1, max_batch_bytes // slot_size)
    batches: list[tuple[list[int], list[StoreWriteSpan]]] = []
    batch_blocks: list[int] = []
    batch_spans: list[StoreWriteSpan] = []
    written_specs: set[tuple[str, int, int, int, int]] = set()

    def flush_batch() -> None:
        nonlocal batch_blocks, batch_spans
        if batch_blocks:
            batches.append((batch_blocks, batch_spans))
        batch_blocks = []
        batch_spans = []

    for spec in reqs_to_store.values():
        source_key = (
            spec.chunk_key,
            spec.start_slot,
            spec.num_slots,
            spec.file_offset,
            len(spec.block_ids),
        )
        if source_key in written_specs:
            continue
        written_specs.add(source_key)
        cursor = 0
        while cursor < len(spec.block_ids):
            if len(batch_blocks) >= max_slots:
                flush_batch()
            available = max_slots - len(batch_blocks)
            take = min(available, len(spec.block_ids) - cursor)
            if take <= 0:
                flush_batch()
                continue
            source_slot = len(batch_blocks)
            part = spec.block_ids[cursor : cursor + take]
            batch_blocks.extend(part)
            batch_spans.append(
                StoreWriteSpan(
                    source_offset=source_slot * slot_size,
                    nbytes=take * slot_size,
                    file_offset=spec.file_offset + cursor * slot_size,
                    chunk_key=spec.chunk_key,
                    start_slot=spec.start_slot,
                    num_slots=spec.num_slots,
                )
            )
            cursor += take
    flush_batch()
    return batches
