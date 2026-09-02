# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
import os
import threading
import time
from typing import Any

import cupy
import torch

from daser.config import (
    STORAGE_FORMAT_COMPRESSED_ONLINE,
    STORAGE_FORMAT_COMPRESSED_READ_ONLY,
    STORAGE_FORMAT_RAW,
)
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
from daser.ops.compressed_kv import (
    ONLINE_PACK_BATCH_SLOTS,
    FusedOnlineKVPacker,
    OnlinePackedSlot,
)
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
    producer_event: torch.cuda.Event | None = None


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
        self._staging_lease_semaphore: asyncio.Semaphore | None = None
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._layer_names: list[str] = []
        self._layer_idx_map: dict[str, int] = {}
        self._local_slot_size = 0
        self._rank_stride_bytes = 0
        self._tp_rank = 0
        self._tp_size = 1
        self._cuda_stream: torch.cuda.Stream | None = None
        # ``FusedOnlineKVPacker`` owns reusable host/device metadata buffers.
        # Keep staging launches serialized while allowing the asyncio store
        # loop to progress transfers for an already-completed batch.
        self._stage_lock = threading.Lock()
        self._storage_format = STORAGE_FORMAT_RAW
        self._online_packer: FusedOnlineKVPacker | None = None
        self._diagnostic_emitted = False
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
        self._staging_lease_semaphore = None

    @property
    def max_slots_per_buffer(self) -> int:
        """Return the number of raw slots admitted by one store buffer."""
        if self._staging_pool is None or self._local_slot_size <= 0:
            return 0
        return max(1, self._staging_pool.buffer_bytes // self._local_slot_size)

    def configure_compression(
        self,
        *,
        storage_format: str,
        codebooks: bytes,
        tile_scalars: int,
    ) -> None:
        """Bind the startup-warmed online packer to server codebooks.

        Args:
            storage_format: ``raw``, ``compressed-read-only`` or
                ``compressed-online``.
            codebooks: Server-owned plane-major codebook bytes.
            tile_scalars: Codec tile quantum.

        Async/thread-safety:
            Called once on the worker thread before store submissions. Kernel
            construction only reuses the registration-time warm cache.
        """
        self._storage_format = storage_format
        self._online_packer = None
        if storage_format != STORAGE_FORMAT_COMPRESSED_ONLINE:
            if storage_format not in (
                STORAGE_FORMAT_RAW,
                STORAGE_FORMAT_COMPRESSED_READ_ONLY,
            ):
                raise ValueError(f"unknown storage format: {storage_format}")
            return
        if self._staging_pool is None or len(self._kv_caches) != 1:
            raise ValueError("compressed-online requires cross-layer KV staging")
        kv_cache = next(iter(self._kv_caches.values()))
        self._online_packer = FusedOnlineKVPacker(
            kv_cache=kv_cache,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
            max_slots_per_buffer=self.max_slots_per_buffer,
        )
        logger.info(
            "[CONNECTOR] online pack configured slots_per_buffer=%d "
            "slot_size=%d tile_scalars=%d",
            self.max_slots_per_buffer,
            self._local_slot_size,
            tile_scalars,
        )

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
            In packed mode, the event is recorded at this point so deferred
            packing waits only for the current forward pass. Raw mode keeps
            the legacy late event capture in ``_submit_save``.
        """
        producer_event: torch.cuda.Event | None = None
        if getattr(self, "_online_packer", None) is not None:
            sample = next(iter(getattr(self, "_kv_caches", {}).values()), None)
            if sample is not None:
                producer_event = record_cuda_event(sample)
        for req_id, spec in reqs_to_store.items():
            base_id = base_req_id(req_id)
            save = self._pending_finished_saves.get(base_id)
            if save is None:
                save = _DeferredFinishedSave({}, producer_event=producer_event)
                self._pending_finished_saves[base_id] = save
            elif producer_event is not None:
                # A request may publish multiple chunks over several worker
                # steps. The newest event orders every earlier publication on
                # the same producer stream while covering the latest chunk.
                save.producer_event = producer_event
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
        producer_event = save.producer_event
        if producer_event is None:
            # Preserve raw/read-only behavior and lightweight test probes that
            # do not expose a packed producer event.
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
        max_batch_bytes = self._staging_bytes
        if getattr(self, "_online_packer", None) is not None:
            # A large first-turn prefix can contain 16+ blocks. Compressing
            # the whole prefix in one CUDA launch monopolizes memory
            # bandwidth while other conversations are prefilling. Small
            # online batches let vLLM work between codec launches; each batch
            # still follows the same server-owned span and publication path.
            max_batch_bytes = min(
                max_batch_bytes,
                ONLINE_PACK_BATCH_SLOTS * self._local_slot_size,
            )
        return build_staging_store_batches(
            reqs_to_store,
            self._local_slot_size,
            max_batch_bytes=max_batch_bytes,
        )

    async def _store_finished_save(
        self,
        save: _DeferredFinishedSave,
        producer_event: torch.cuda.Event | None,
    ) -> None:
        batches = self._plan_finished_save(save)
        if not batches:
            return

        # Keep raw mode byte-for-byte and scheduling equivalent to master. For
        # packed stores, the staging pool has two independent leases in the
        # production configuration.  Start the next (single producer-locked)
        # pack as soon as the current staging snapshot is ready so its CUDA
        # work overlaps the server's asynchronous L1/L2 admission.  The
        # current lease remains live until transfer completion, and the next
        # stage task is drained on errors before its lease can be reclaimed.
        overlap_batches = (
            getattr(self, "_online_packer", None) is not None
            and len(batches) > 1
            and int(getattr(self, "_store_capacity", 1)) > 1
        )
        if not overlap_batches:
            for block_ids, spans in batches:
                staged = await self._stage_batch_acquired(
                    block_ids,
                    spans,
                    producer_event,
                )
                try:
                    await self._write_cuda_buffer(staged)
                finally:
                    self._release_staged(staged)
            return

        pending_stage: asyncio.Task[StagedStoreBatch] | None = None
        try:
            pending_stage = asyncio.create_task(
                self._stage_batch_acquired(
                    batches[0][0],
                    batches[0][1],
                    producer_event,
                )
            )
            for index, (block_ids, spans) in enumerate(batches):
                del block_ids, spans
                staged = await pending_stage
                pending_stage = None
                if index + 1 < len(batches):
                    next_block_ids, next_spans = batches[index + 1]
                    pending_stage = asyncio.create_task(
                        self._stage_batch_acquired(
                            next_block_ids,
                            next_spans,
                            producer_event,
                        )
                    )
                try:
                    await self._write_cuda_buffer(staged)
                finally:
                    self._release_staged(staged)
        except BaseException:
            if pending_stage is not None:
                await self._drain_staged_task(pending_stage)
            raise

    async def _stage_batch_acquired(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
        producer_event: torch.cuda.Event | None,
    ) -> "StagedStoreBatch":
        """Acquire a bounded staging lease and snapshot one store batch.

        Args:
            block_ids: Physical vLLM blocks in logical order.
            spans: Server-owned write spans for this batch.
            producer_event: CUDA event ordering the live KV snapshot.

        Returns:
            A staged batch that owns one lease-semaphore permit until transfer
            completion.

        Async/thread-safety:
            Runs on the private store event loop.  CUDA staging itself is
            delegated to an executor thread and remains serialized by the
            packer lock; the permit bounds all in-flight leases globally.
        """
        lease_semaphore = getattr(self, "_staging_lease_semaphore", None)
        if lease_semaphore is None:
            pool = self._staging_pool
            depth = pool.depth if pool is not None else self._store_capacity
            lease_semaphore = asyncio.Semaphore(max(1, depth))
            self._staging_lease_semaphore = lease_semaphore
        await lease_semaphore.acquire()
        try:
            staged = await asyncio.to_thread(
                self._stage_batch_serialized,
                block_ids,
                spans,
                producer_event,
            )
        except BaseException:
            lease_semaphore.release()
            raise
        return StagedStoreBatch(
            buffer=staged.buffer,
            spans=staged.spans,
            lease=staged.lease,
            lease_permit=lease_semaphore,
        )

    async def _drain_staged_task(
        self,
        task: asyncio.Task["StagedStoreBatch"],
    ) -> None:
        """Finish and release a prefetched staging task during error cleanup."""
        try:
            staged = await asyncio.shield(task)
        except BaseException:
            return
        self._release_staged(staged)

    def _release_staged(self, staged: "StagedStoreBatch") -> None:
        """Release a staged buffer and its global lease permit exactly once."""
        stage_lock = getattr(self, "_stage_lock", None)
        if stage_lock is None:
            staged.lease.release()
        else:
            with stage_lock:
                staged.lease.release()
        if staged.lease_permit is not None:
            staged.lease_permit.release()

    def _stage_batch_serialized(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
        producer_event: torch.cuda.Event | None,
    ) -> "StagedStoreBatch":
        """Stage one batch off the store event-loop thread.

        Args:
            block_ids: Physical vLLM blocks in logical order.
            spans: Server-owned write spans for this batch.
            producer_event: CUDA event ordering the live KV snapshot.

        Returns:
            A leased staging batch whose bytes are ready for IPC transfer.

        Async/thread-safety:
            Runs in an executor thread.  The lock protects reusable packer
            metadata buffers; CUDA stream ordering still protects the staging
            bytes before ``_write_cuda_buffer`` is called.
        """
        stage_lock = getattr(self, "_stage_lock", None)
        if stage_lock is None:
            # Lightweight tests may construct the pipeline with ``__new__``
            # and replace staging entirely; preserve that public helper
            # contract without requiring the full runtime initializer.
            return self._stage_batch(block_ids, spans, producer_event)
        with stage_lock:
            return self._stage_batch(block_ids, spans, producer_event)

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
            # Keep the fixed store stream at CUDA's default priority.  The
            # producer event and final synchronization already order the
            # deferred snapshot before IPC; lowering stream priority can make
            # the codec wait behind vLLM prefill and increase TTFT.
            stream = torch.cuda.Stream(device=sample.device)
            self._cuda_stream = stream
        stage_started = time.perf_counter()
        packed_mode = self._online_packer is not None
        pack_ms = 0.0
        if stream is not None:
            if producer_event is not None:
                stream.wait_event(producer_event)
            with torch.cuda.stream(stream):
                if self._online_packer is not None:
                    logical_slots = _logical_slots_for_batch(block_ids, spans)
                    pack_started = time.perf_counter()
                    packed = self._online_packer.pack_into(
                        staging=lease.view,
                        block_ids=block_ids,
                        logical_slots=logical_slots,
                        slot_stride=self._local_slot_size,
                        stream=stream,
                    )
                    pack_ms = (time.perf_counter() - pack_started) * 1000
                    spans = _packed_store_spans(spans, packed, self._local_slot_size)
                else:
                    self._copy_blocks(lease.view, block_ids, sample)
            sync_started = time.perf_counter()
            stream.synchronize()
            sync_ms = (time.perf_counter() - sync_started) * 1000
        else:
            self._copy_blocks(lease.view, block_ids, sample)
            sync_ms = 0.0
        logger.debug(
            "[CONNECTOR] store stage: slots=%d bytes=%d packed=%s total_ms=%.3f "
            "pack_ms=%.3f sync_ms=%.3f",
            len(block_ids),
            nbytes,
            packed_mode,
            (time.perf_counter() - stage_started) * 1000,
            pack_ms,
            sync_ms,
        )
        if packed_mode and not self._diagnostic_emitted:
            logger.info(
                "[CONNECTOR] online pack first store slots=%d spans=%s",
                len(block_ids),
                [
                    {
                        "mode": span.packed_mode or "raw",
                        "source_offset": span.source_offset,
                        "nbytes": span.nbytes,
                        "file_offset": span.file_offset,
                    }
                    for span in spans
                ],
            )
            self._diagnostic_emitted = True
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
            nbytes=_store_payload_nbytes(staged.buffer.nbytes, staged.spans),
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
                    "logical_slot_start": span.logical_slot_start,
                    "logical_slot_count": span.logical_slot_count,
                    "packed": span.packed,
                    "mode": span.packed_mode or "raw",
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
    lease_permit: asyncio.Semaphore | None = None


def _store_payload_nbytes(
    buffer_nbytes: int,
    spans: list[StoreWriteSpan],
) -> int:
    """Return the CUDA IPC mapping length for one store batch.

    Packed records are compacted inside a raw-sized staging lease. Mapping only
    through the final packed source byte avoids exposing and registering the
    unused tail while preserving the raw path's existing allocation size.

    Args:
        buffer_nbytes: Logical byte capacity of the leased staging view.
        spans: Store spans describing source-buffer ranges.

    Returns:
        The logical mapping length sent to the server-owned transfer layer.

    Raises:
        ValueError: If a span falls outside the leased staging view.

    Async/thread-safety:
        Pure CPU validation; safe to call from the store event-loop thread.
    """
    if buffer_nbytes < 0:
        raise ValueError("store staging buffer size must be non-negative")
    max_end = 0
    has_packed_span = False
    for span in spans:
        source_offset = int(span.source_offset)
        nbytes = int(span.nbytes)
        if source_offset < 0 or nbytes < 0:
            raise ValueError("store span source range must be non-negative")
        end = source_offset + nbytes
        if end > buffer_nbytes:
            raise ValueError("store span exceeds staging buffer")
        max_end = max(max_end, end)
        has_packed_span = has_packed_span or bool(span.packed)
    if not has_packed_span:
        return buffer_nbytes
    return max_end


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
                    logical_slot_start=spec.start_slot + cursor,
                    logical_slot_count=take,
                )
            )
            cursor += take
    flush_batch()
    return batches


def _logical_slots_for_batch(
    block_ids: list[int], spans: list[StoreWriteSpan]
) -> list[int]:
    """Expand bounded write spans into one logical slot ID per source block."""
    slots: list[int] = []
    for span in spans:
        count = span.logical_slot_count
        start = span.logical_slot_start
        if start < 0 or count <= 0:
            raise ValueError("store span is missing logical slot metadata")
        slots.extend(range(start, start + count))
    if len(slots) != len(block_ids):
        raise ValueError("store span slot count does not match source blocks")
    return slots


def _packed_store_spans(
    spans: list[StoreWriteSpan],
    packed: list[OnlinePackedSlot],
    slot_size: int,
) -> list[StoreWriteSpan]:
    """Build transfer spans for online records, compacting each allocation.

    Packed records are emitted back-to-back inside the raw allocation that
    owns them.  This makes the source and file ranges coalesce into one DMA
    span when a batch covers adjacent logical slots, while preserving the
    original raw-stride layout for malformed metadata or records that exceed
    their reserved capacity.

    Args:
        spans: Original raw-stride spans grouped by allocation and batch.
        packed: One packed record description per source slot, in span order.
        slot_size: Raw bytes reserved for one rank-local slot.

    Returns:
        One packed transfer span per logical slot.

    Raises:
        ValueError: If span metadata, slot lengths, or record ordering is
            inconsistent.

    Async/thread-safety:
        Pure CPU planning; safe to call from the store worker thread.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")

    result: list[StoreWriteSpan] = []
    packed_cursor = 0
    for source in spans:
        count = source.logical_slot_count
        if count <= 0 or source.logical_slot_start < 0:
            raise ValueError("store span is missing logical slot metadata")
        source_records = packed[packed_cursor : packed_cursor + count]
        if len(source_records) != count:
            raise ValueError("packed slot count does not match store spans")

        # Each packed record is aligned and must fit the raw slot envelope.  A
        # malformed record falls back to the old slot-stride file offset so it
        # cannot write into a neighbouring allocation.
        compact_bytes = sum(record.stored_length for record in source_records)
        compact_file_layout = (
            source.file_offset >= 0
            and compact_bytes <= count * slot_size
            and all(0 < record.stored_length <= slot_size for record in source_records)
        )
        file_cursor = source.file_offset
        for relative, record in enumerate(source_records):
            if record.logical_slot != source.logical_slot_start + relative:
                raise ValueError("packed slots are not ordered within store span")
            file_offset = (
                file_cursor
                if compact_file_layout
                else source.file_offset + relative * slot_size
            )
            result.append(
                StoreWriteSpan(
                    source_offset=record.source_offset,
                    nbytes=record.stored_length,
                    file_offset=file_offset,
                    chunk_key=source.chunk_key,
                    start_slot=source.start_slot,
                    num_slots=source.num_slots,
                    logical_slot_start=record.logical_slot,
                    logical_slot_count=1,
                    packed=True,
                    packed_mode=record.mode.name.lower(),
                )
            )
            if compact_file_layout:
                file_cursor += record.stored_length
        packed_cursor += count

    if packed_cursor != len(packed):
        raise ValueError("packed slot count does not match store spans")
    return result
