# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Standard
import asyncio
import os
from typing import TYPE_CHECKING, Any

# Third Party
import cupy
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

if TYPE_CHECKING:
    # Third Party
    from vllm.attention import AttentionMetadata
    from vllm.forward_context import ForwardContext

# First Party
from daser.connector.metadata import (
    DaserConnectorMeta,
    StoreWriteSpan,
)
from daser.connector.staging import (
    DEFAULT_PENDING_STORE_STAGING_BYTES,
    DEFAULT_STORE_STAGING_BYTES,
    CudaStagingLease,
    CudaStagingPool,
    StagedStoreBatch,
)
from daser.connector.staging import (
    build_load_copy_runs as _build_load_copy_runs,
)
from daser.connector.staging import (
    build_load_read_batches as _build_load_read_batches,
)
from daser.connector.staging import (
    build_staging_store_batches as _build_staging_store_batches,
)
from daser.connector.staging import (
    copy_kv_cache_to_staging as _copy_kv_cache_to_staging,
)
from daser.connector.staging import (
    copy_staging_to_kv_cache as _copy_staging_to_kv_cache,
)
from daser.connector.staging import (
    derive_store_staging_limits as _derive_store_staging_limits,
)
from daser.connector.staging import (
    record_cuda_event as _record_cuda_event,
)
from daser.connector.staging import (
    synchronize_cuda_tensor as _synchronize_cuda_tensor,
)
from daser.logging import init_logger
from daser.transfer.cuda_ipc import (
    cuda_array_device_id,
    cuda_array_pointer,
    export_cuda_ipc_handle,
)

logger = init_logger(__name__)


class WorkerConnectorMixin:
    """Worker-role vLLM connector behavior.

    Async/thread-safety:
        Public methods are called on vLLM worker threads. Blocking NVMe work is
        submitted to the connector's background asyncio loop.
    """

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register the per-layer KV cache tensors.

        Args:
            kv_caches: dict mapping layer_name -> KV tensor.
        """
        self._kv_caches = kv_caches
        self._layer_names = list(kv_caches.keys())
        self._layer_idx_map = {name: idx for idx, name in enumerate(self._layer_names)}
        sample = next(iter(kv_caches.values()), None)
        if kv_caches:
            (
                self._store_staging_bytes,
                self._pending_store_staging_limit_bytes,
            ) = _derive_store_staging_limits(sample.device)
            logger.info(
                "[CONNECTOR] register_kv_caches: %d layers, first shape=%s dtype=%s",
                len(kv_caches),
                sample.shape,
                sample.dtype,
            )
            logger.info(
                "[CONNECTOR] transient store staging caps: batch=%d pending=%d",
                self._store_staging_bytes,
                self._pending_store_staging_limit_bytes,
            )

        if self._slot_size == 0 and self._layer_names and sample is not None:
            num_blocks = sample.shape[1] if sample.dim() >= 2 else 1
            layer_size = sample.nbytes // num_blocks
            self._slot_size = layer_size * len(self._layer_names)
            logger.info(
                "[CONNECTOR] computed slot_size=%d from %d layers",
                self._slot_size,
                len(self._layer_names),
            )

        if sample is not None:
            self._store_staging_bytes = max(
                self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES,
                self._slot_size,
            )
            self._staging_pool = CudaStagingPool(
                device=sample.device,
                initial_bytes=self._store_staging_bytes,
                max_buffer_bytes=self._store_staging_bytes,
            )
            logger.info(
                "[CONNECTOR] preallocated staging buffer=%d cap=%d pending=%d",
                self._store_staging_bytes,
                self._store_staging_bytes,
                self._pending_store_staging_limit_bytes,
            )

        if self._ensure_transfer_ready():
            asyncio.run_coroutine_threadsafe(
                self._ipc_async.init_transfer(),
                self._bg_loop,
            ).result(timeout=120.0)

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        super().bind_connector_metadata(connector_metadata)
        self._meta = connector_metadata
        self._reap_save_futures(block=False)
        self._pending_commits = set()
        self._clear_save_state()
        for spec in connector_metadata.reqs_to_store.values():
            if spec.block_ids:
                self._pending_commits.add(spec.chunk_key)

    def clear_connector_metadata(self) -> None:
        """Clear metadata after forward pass completes."""
        super().clear_connector_metadata()
        self._meta = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Load all KV cache blocks for cache-hit requests.

        Args:
            forward_context: vLLM ForwardContext for this forward pass.
        """
        if self._meta is None or not self._meta.reqs_to_load:
            return
        logger.debug(
            "[CONNECTOR] start_load_kv: %d reqs to load",
            len(self._meta.reqs_to_load),
        )
        if not self._ensure_transfer_ready():
            return

        num_layers = len(self._layer_names)
        if num_layers == 0:
            return

        sample_tensor = next(iter(self._kv_caches.values()), None)
        if sample_tensor is None:
            return

        load_batches = _build_load_read_batches(
            self._meta.reqs_to_load,
            self._slot_size,
            max_batch_bytes=(self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES),
        )
        if not load_batches:
            return

        total_copies = 0
        total_copy_runs = 0
        for total_bytes, spans, per_req_ranges in load_batches:
            staging_lease = self._acquire_staging(total_bytes, sample_tensor.device)
            staging = staging_lease.view
            try:
                cp_staging = cupy.asarray(staging)
                cuda_handle = export_cuda_ipc_handle(cp_staging)
                device_id = cuda_array_device_id(cp_staging)
                device_ptr = cuda_array_pointer(cp_staging)

                asyncio.run_coroutine_threadsafe(
                    self._ipc_async.transfer_load_cuda(
                        cuda_ipc_handle=cuda_handle,
                        nbytes=total_bytes,
                        device_id=device_id,
                        device_ptr=device_ptr,
                        producer_pid=os.getpid(),
                        spans=spans,
                    ),
                    self._bg_loop,
                ).result(timeout=120.0)

                copy_runs = _build_load_copy_runs(per_req_ranges)
                total_copy_runs += len(copy_runs)
                for run in copy_runs:
                    total_copies += _copy_staging_to_kv_cache(
                        staging=staging[run.start : run.end],
                        kv_caches=self._kv_caches,
                        layer_names=self._layer_names,
                        block_ids=run.block_ids,
                        slot_size=self._slot_size,
                        load_key_scale=self._load_key_scale,
                        load_value_scale=self._load_value_scale,
                        pos_offset=run.pos_offset,
                        rope_delta_scale=self._rope_delta_scale,
                        rope_base=self._rope_base,
                        rope_rotary_dim=self._rope_rotary_dim,
                        rope_is_neox_style=self._rope_is_neox_style,
                    )
                _synchronize_cuda_tensor(sample_tensor)
            finally:
                staging_lease.release()

        logger.debug(
            "[CONNECTOR] start_load_kv: %d reqs, %d batches, %d copy runs, "
            "%d GPU copies",
            len(self._meta.reqs_to_load),
            len(load_batches),
            total_copy_runs,
            total_copies,
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op because all KV loading is done eagerly in start_load_kv.

        Args:
            layer_name: ignored.
        """
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Submit this layer's KV blocks for server-owned transfer.

        Args:
            layer_name: name of the current attention layer.
            kv_layer: full KV cache tensor for this layer.
            attn_metadata: attention metadata (not directly used).
        """
        if self._meta is None or not self._meta.reqs_to_store:
            return
        if not self._ensure_transfer_ready():
            return

        if layer_name not in self._layer_idx_map:
            logger.warning(
                "[CONNECTOR] save_kv_layer: unknown layer %s, skipping", layer_name
            )

    def wait_for_save(self) -> None:
        """Wait for submitted layer stores and commit visible chunks."""
        if self._meta is None:
            return

        commit_keys = list(self._pending_commits)
        reqs_to_store = dict(self._meta.reqs_to_store)
        if commit_keys and reqs_to_store:
            batch_futures = []
            batches = _build_staging_store_batches(
                reqs_to_store,
                self._slot_size,
                max_batch_bytes=(
                    self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES
                ),
            )
            for block_ids, spans in batches:
                staged = self._stage_store_batch(block_ids, spans)
                if staged is None:
                    continue
                future = asyncio.run_coroutine_threadsafe(
                    self._write_cuda_buffer(
                        buffer=staged.buffer,
                        ready_event=staged.ready_event,
                        spans=staged.spans,
                    ),
                    self._bg_loop,
                )
                self._track_save_future(future, staged.buffer.nbytes, staged.lease)
                batch_futures.append(future)
            if batch_futures:
                commit_future = asyncio.run_coroutine_threadsafe(
                    self._commit_after_store_futures(batch_futures, commit_keys),
                    self._bg_loop,
                )
                self._track_save_future(commit_future, 0, None)
        self._clear_save_state()
        self._pending_commits.clear()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Collect completed background saves after a worker step.

        Args:
            finished_req_ids: Request IDs that vLLM finished in this step.

        Returns:
            ``(None, None)`` because DaseR does not take ownership of request
            blocks beyond the current vLLM lifecycle.
        """
        self._reap_save_futures(block=False)
        return None, None

    def shutdown(self) -> None:
        """Stop the background IO loop."""
        if self._role != KVConnectorRole.WORKER:
            return
        self._reap_save_futures(block=True)
        asyncio.run_coroutine_threadsafe(
            self._ipc_async.close(),
            self._bg_loop,
        ).result(timeout=10.0)
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)

    def _run_bg_loop(self) -> None:
        """Run the background asyncio IO loop."""
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

    def _ensure_transfer_ready(self) -> bool:
        """Refresh server transfer config and mark worker data plane ready."""
        if getattr(self, "_transfer_ready", False):
            return True

        self._refresh_runtime_config()
        if not self._store_path or not self._slot_size:
            logger.warning(
                "[CONNECTOR] server transfer config is not ready; start DaseR server "
                "before sending requests",
            )
            return False

        self._transfer_ready = True
        logger.info("[CONNECTOR] server transfer mode=%s", self._transfer_mode)
        return True

    def _clear_save_state(self) -> None:
        """Clear worker-side per-forward save state."""
        return

    def _reap_save_futures(self, block: bool) -> None:
        """Collect completed background save tasks.

        Args:
            block: If True, wait for every pending save. If False, collect only
                tasks that are already complete.
        """
        remaining = []
        pending_bytes = getattr(self, "_pending_save_staging_bytes", 0)
        for record in self._save_futures:
            if isinstance(record, tuple):
                future = record[0]
                staging_bytes = int(record[1]) if len(record) > 1 else 0
                staging_lease = record[2] if len(record) > 2 else None
            else:
                future = record
                staging_bytes = 0
                staging_lease = None
            if block or future.done():
                try:
                    future.result(timeout=120.0)
                finally:
                    pending_bytes = max(0, pending_bytes - staging_bytes)
                    if staging_lease is not None:
                        staging_lease.release()
            else:
                remaining.append((future, staging_bytes, staging_lease))
        self._save_futures = remaining
        self._pending_save_staging_bytes = pending_bytes

    def _track_save_future(
        self,
        future: Any,
        staging_bytes: int,
        staging_lease: CudaStagingLease | None,
    ) -> None:
        """Track one background save future and its live staging bytes.

        Args:
            future: Future returned by ``asyncio.run_coroutine_threadsafe``.
            staging_bytes: GPU staging bytes kept alive by the future.
            staging_lease: Optional reusable staging lease to release after
                ``future`` completes.

        Async/thread-safety:
            Called on the worker thread. Completion is collected by
            ``_reap_save_futures``.
        """
        self._pending_save_staging_bytes = (
            getattr(self, "_pending_save_staging_bytes", 0) + staging_bytes
        )
        self._save_futures.append((future, staging_bytes, staging_lease))

    def _wait_for_save_staging_capacity(self, nbytes: int) -> None:
        """Apply backpressure before allocating another store staging buffer.

        Args:
            nbytes: Size of the next staging tensor.

        Async/thread-safety:
            Called by vLLM's worker thread. It may wait for already-submitted
            background stores when live staging would exceed the configured
            cap.
        """
        limit = max(
            self._pending_store_staging_limit_bytes
            or DEFAULT_PENDING_STORE_STAGING_BYTES,
            nbytes,
        )
        while (
            getattr(self, "_pending_save_staging_bytes", 0) + nbytes > limit
            and self._save_futures
        ):
            record = self._save_futures.pop(0)
            if isinstance(record, tuple):
                future = record[0]
                staging_bytes = int(record[1]) if len(record) > 1 else 0
                staging_lease = record[2] if len(record) > 2 else None
            else:
                future = record
                staging_bytes = 0
                staging_lease = None
            try:
                future.result(timeout=120.0)
            finally:
                self._pending_save_staging_bytes = max(
                    0,
                    getattr(self, "_pending_save_staging_bytes", 0) - staging_bytes,
                )
                if staging_lease is not None:
                    staging_lease.release()
            self._reap_save_futures(block=False)

    def _acquire_staging(
        self,
        nbytes: int,
        device: torch.device,
    ) -> CudaStagingLease:
        """Acquire a reusable staging buffer for a CUDA IPC transfer.

        Args:
            nbytes: Logical byte count needed for the transfer.
            device: Device used when the pool has not been initialized yet.

        Returns:
            A staging lease whose ``view`` is safe to export through CUDA IPC.

        Async/thread-safety:
            Called from the worker thread. Store-path callers must retain the
            lease until the background server transfer completes.
        """
        pool = getattr(self, "_staging_pool", None)
        if pool is None:
            max_bytes = max(
                nbytes,
                self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES,
            )
            pool = CudaStagingPool(
                device=device,
                initial_bytes=0,
                max_buffer_bytes=max_bytes,
            )
            self._staging_pool = pool
        return pool.acquire(nbytes)

    def _stage_store_batch(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
    ) -> StagedStoreBatch | None:
        """Snapshot one bounded batch of KV blocks into CUDA staging.

        Args:
            block_ids: vLLM KV block IDs to snapshot.
            spans: Server store spans targeting this staging batch.

        Returns:
            A staged batch ready for CUDA IPC transfer, or ``None`` when the
            connector has no layer state.

        Async/thread-safety:
            Runs on the vLLM worker thread so KV cache reads are launched before
            vLLM can recycle the source blocks. The returned tensor is kept
            alive by the background transfer future.
        """
        num_layers = len(self._layer_names)
        if num_layers == 0:
            return None
        sample_tensor = next(iter(self._kv_caches.values()), None)
        if sample_tensor is None:
            return None
        if not block_ids or not spans:
            return None
        nbytes = len(block_ids) * self._slot_size
        self._wait_for_save_staging_capacity(nbytes)
        staging_lease = self._acquire_staging(nbytes, sample_tensor.device)
        staging = staging_lease.view
        block_index = torch.tensor(
            block_ids,
            dtype=torch.long,
            device=sample_tensor.device,
        )
        for layer_name in self._layer_names:
            _copy_kv_cache_to_staging(
                staging=staging,
                kv_layer=self._kv_caches[layer_name],
                layer_idx=self._layer_idx_map[layer_name],
                block_ids=block_ids,
                num_layers=num_layers,
                slot_size=self._slot_size,
                block_index=block_index,
            )
        return StagedStoreBatch(
            buffer=staging,
            ready_event=_record_cuda_event(staging),
            spans=spans,
            lease=staging_lease,
        )

    async def _commit_after_store_futures(
        self,
        batch_futures: list[Any],
        commit_keys: list[str],
    ) -> None:
        """Commit chunks after all staged transfer batches finish.

        Args:
            batch_futures: Futures for each staged store batch.
            commit_keys: Chunk keys to publish after stores complete.

        Async/thread-safety:
            Runs on the connector background event loop and does not read vLLM
            KV cache tensors.
        """
        stored_keys: list[str] = []
        for future in batch_futures:
            stored_keys.extend(await asyncio.wrap_future(future))
        await self._commit_stored_keys(stored_keys, commit_keys)

    async def _write_cuda_buffer(
        self,
        buffer: torch.Tensor,
        ready_event: torch.cuda.Event | None,
        spans: list[StoreWriteSpan],
    ) -> list[str]:
        """Write selected spans from one contiguous CUDA buffer.

        Args:
            buffer: CUDA tensor exported over CUDA IPC.
            ready_event: Producer-stream event for ``buffer``.
            spans: Source/destination write spans.

        Returns:
            Chunk keys accepted by the server for this buffer.
        """
        if ready_event is not None:
            ready_event.synchronize()
        else:
            _synchronize_cuda_tensor(buffer)
        cp_buffer = cupy.asarray(buffer)
        cuda_ipc_handle = export_cuda_ipc_handle(cp_buffer)
        device_id = cuda_array_device_id(cp_buffer)
        device_ptr = cuda_array_pointer(cp_buffer)
        stored_keys = await self._ipc_async.transfer_store_cuda(
            cuda_ipc_handle=cuda_ipc_handle,
            nbytes=buffer.nbytes,
            device_id=device_id,
            device_ptr=device_ptr,
            producer_pid=os.getpid(),
            spans=[
                {
                    "source_offset": span.source_offset,
                    "nbytes": span.nbytes,
                    "file_offset": span.file_offset,
                    "chunk_key": span.chunk_key,
                    "start_slot": span.start_slot,
                    "num_slots": span.num_slots,
                }
                for span in spans
            ],
        )
        return stored_keys

    async def _commit_stored_keys(
        self,
        stored_keys: list[str],
        commit_keys: list[str],
    ) -> None:
        """Commit requested chunks whose store spans were accepted."""
        requested = set(commit_keys)
        candidate_keys = (
            commit_keys
            if not stored_keys
            else [key for key in stored_keys if key in requested]
        )
        keys_to_commit = list(dict.fromkeys(candidate_keys))
        await self._ipc_async.commit_chunks(keys_to_commit)
