# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Standard
import asyncio
from dataclasses import dataclass
import os
import time
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
from daser.connector.helpers import base_req_id
from daser.connector.metadata import (
    DaserConnectorMeta,
    ReqStoreSpec,
    StoreWriteSpan,
)
from daser.connector.staging import (
    CROSS_LAYER_KV_CACHE_KEY,
    DEFAULT_PENDING_STORE_STAGING_BYTES,
    DEFAULT_STORE_STAGING_BYTES,
    FUSED_RESTORE_MIN_SLOTS,
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
    copy_cross_layer_kv_cache_to_staging as _copy_cross_layer_kv_cache_to_staging,
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
from daser.ops.rope_apply import (
    apply_rope_delta_to_key_block as _apply_rope_delta_to_key_block,
)
from daser.ops.rope_apply import (
    apply_rope_delta_to_kv_key_block_table,
    restore_cross_layer_kv_cache_table,
)
from daser.transfer.cuda_ipc import (
    cuda_array_device_id,
    cuda_array_pointer,
    export_cuda_ipc_handle,
)

logger = init_logger(__name__)

_ROPE_WARMUP_BLOCKS = 1


@dataclass
class _DeferredFinishedSave:
    """Store work held until vLLM reports a request as finished."""

    commit_keys: set[str]
    reqs_to_store: dict[str, ReqStoreSpec]
    submitted: bool = False
    future: Any | None = None


@dataclass
class _SaveFuture:
    """One background save future and the staging it keeps alive.

    Attributes:
        future: future returned by ``asyncio.run_coroutine_threadsafe``.
        staging_bytes: GPU staging bytes held alive until completion.
        lease: optional reusable staging lease released after completion.
    """

    future: Any
    staging_bytes: int
    lease: CudaStagingLease | None

    def release(self) -> None:
        """Release the reusable staging lease, if any."""
        if self.lease is not None:
            self.lease.release()
            self.lease = None


def _cuda_allocation_base_and_offset(device_ptr: int) -> tuple[int, int]:
    """Return CUDA allocation base pointer and byte offset for ``device_ptr``.

    Args:
        device_ptr: CUDA device pointer exported through IPC.

    Returns:
        Tuple of ``(allocation_base_ptr, byte_offset)``.
    """
    try:
        from cuda.bindings import driver as cuda_driver

        result, base_ptr, _allocation_size = cuda_driver.cuMemGetAddressRange(
            device_ptr
        )
        if result == cuda_driver.CUresult.CUDA_SUCCESS:
            base = int(base_ptr)
            return base, int(device_ptr) - base
    except Exception as exc:  # noqa: BLE001
        logger.debug("[CONNECTOR] cuMemGetAddressRange failed: %s", exc)
    return int(device_ptr), 0


def _warm_rope_apply_backends(
    device: torch.device,
    dtype: torch.dtype,
    block_tokens: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    rope_base: float,
    is_neox_style: bool,
) -> None:
    """Warm dynamic-shape RoPE apply operators.

    Args:
        device: device that owns the worker KV cache.
        dtype: KV cache dtype.
        block_tokens: tokens per cache block.
        heads: number of KV heads.
        head_dim: per-head dimension.
        rotary_dim: number of dimensions covered by RoPE.
        rope_base: RoPE theta/base.
        is_neox_style: True for split-half rotation, False for interleaved.

    Async/thread-safety:
        Runs synchronously during worker KV cache registration, before request
        traffic starts. It launches CUDA work on the current stream. TileLang
        failures are surfaced to avoid silently entering a slow restore path.
    """
    if device.type != "cuda" or rotary_dim <= 0 or head_dim < rotary_dim:
        return
    sample = torch.empty(
        (_ROPE_WARMUP_BLOCKS, block_tokens, heads, head_dim),
        dtype=dtype,
        device=device,
    )
    _apply_rope_delta_to_key_block(
        sample,
        delta=1,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )
    torch.cuda.synchronize(device)


def _warm_cross_layer_restore_backends(
    device: torch.device,
    dtype: torch.dtype,
    layers: int,
    block_tokens: int,
    heads: int,
    head_dim: int,
    rotary_dim: int,
    rope_base: float,
    is_neox_style: bool,
) -> None:
    """Warm cross-layer staging restore TileLang kernels.

    Args:
        device: device that owns the worker KV cache.
        dtype: KV cache dtype.
        layers: number of model KV layers.
        block_tokens: tokens per cache block.
        heads: number of KV heads.
        head_dim: per-head dimension.
        rotary_dim: number of dimensions covered by RoPE.
        rope_base: RoPE theta/base.
        is_neox_style: True for split-half rotation, False for interleaved.

    Async/thread-safety:
        Runs synchronously during worker KV cache registration, before request
        traffic starts. TileLang import/compile failures are surfaced to avoid
        silently entering a slow restore path.
    """
    if device.type != "cuda" or rotary_dim <= 0 or head_dim < rotary_dim:
        return
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    freqs = inv_freq
    cos_table = freqs.cos().contiguous()
    sin_table = freqs.sin().contiguous()
    for blocks, use_fused_restore in (
        (_ROPE_WARMUP_BLOCKS, False),
        (FUSED_RESTORE_MIN_SLOTS, True),
    ):
        sample = torch.empty(
            blocks,
            layers,
            2,
            block_tokens,
            heads,
            head_dim,
            dtype=dtype,
            device=device,
        )
        if use_fused_restore:
            dst = torch.empty_like(sample)
            restore_cross_layer_kv_cache_table(
                sample,
                dst,
                cos_table=cos_table,
                sin_table=sin_table,
                rotary_dim=rotary_dim,
                is_neox_style=is_neox_style,
            )
        else:
            apply_rope_delta_to_kv_key_block_table(
                sample,
                cos_table=cos_table,
                sin_table=sin_table,
                rotary_dim=rotary_dim,
                is_neox_style=is_neox_style,
            )
    torch.cuda.synchronize(device)


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
            if sample.dim() >= 5:
                _warm_rope_apply_backends(
                    device=sample.device,
                    dtype=sample.dtype,
                    block_tokens=int(sample.shape[-3]),
                    heads=int(sample.shape[-2]),
                    head_dim=int(sample.shape[-1]),
                    rotary_dim=int(getattr(self, "_rope_rotary_dim", 0)),
                    rope_base=float(getattr(self, "_rope_base", 10000.0)),
                    is_neox_style=bool(getattr(self, "_rope_is_neox_style", True)),
                )

        self._init_server_transfer()

    def register_cross_layers_kv_cache(
        self,
        kv_cache: torch.Tensor,
        attn_backend: type[Any],
    ) -> None:
        """Register vLLM's cross-layer KV cache tensor.

        Args:
            kv_cache: vLLM tensor whose logical layout starts with
                ``[blocks, layers, 2, block_tokens, heads, head_dim]`` for the
                NHD layout DaseR requests.
            attn_backend: Attention backend that created ``kv_cache``.

        Async/thread-safety:
            Called once during worker initialization before request traffic.
        """
        kv_cache_config = getattr(self, "_kv_cache_config", None)
        layer_names: list[str] = []
        if kv_cache_config is not None:
            for group in getattr(kv_cache_config, "kv_cache_groups", []):
                layer_names.extend(list(getattr(group, "layer_names", [])))
        if not layer_names:
            layer_count = int(kv_cache.shape[1]) if kv_cache.dim() >= 2 else 0
            layer_names = [f"layer.{idx}" for idx in range(layer_count)]
        self._cross_layers_attn_backend = attn_backend
        self._kv_caches = {CROSS_LAYER_KV_CACHE_KEY: kv_cache}
        self._layer_names = layer_names
        self._layer_idx_map = {name: idx for idx, name in enumerate(self._layer_names)}
        if kv_cache.dim() < 6:
            logger.warning(
                "[CONNECTOR] cross-layer KV cache has unsupported shape=%s",
                tuple(kv_cache.shape),
            )
            return
        (
            self._store_staging_bytes,
            self._pending_store_staging_limit_bytes,
        ) = _derive_store_staging_limits(kv_cache.device)
        layer_size = kv_cache[0, 0].nbytes
        if self._slot_size == 0:
            self._slot_size = layer_size * len(self._layer_names)
            logger.info(
                "[CONNECTOR] computed cross-layer slot_size=%d from %d layers",
                self._slot_size,
                len(self._layer_names),
            )
        self._store_staging_bytes = max(
            self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES,
            self._slot_size,
        )
        self._staging_pool = CudaStagingPool(
            device=kv_cache.device,
            initial_bytes=self._store_staging_bytes,
            max_buffer_bytes=self._store_staging_bytes,
        )
        logger.info(
            "[CONNECTOR] register_cross_layers_kv_cache: layers=%d shape=%s dtype=%s",
            len(self._layer_names),
            tuple(kv_cache.shape),
            kv_cache.dtype,
        )
        _warm_rope_apply_backends(
            device=kv_cache.device,
            dtype=kv_cache.dtype,
            block_tokens=int(kv_cache.shape[-3]),
            heads=int(kv_cache.shape[-2]),
            head_dim=int(kv_cache.shape[-1]),
            rotary_dim=int(getattr(self, "_rope_rotary_dim", 0)),
            rope_base=float(getattr(self, "_rope_base", 10000.0)),
            is_neox_style=bool(getattr(self, "_rope_is_neox_style", True)),
        )
        _warm_cross_layer_restore_backends(
            device=kv_cache.device,
            dtype=kv_cache.dtype,
            layers=int(kv_cache.shape[1]),
            block_tokens=int(kv_cache.shape[-3]),
            heads=int(kv_cache.shape[-2]),
            head_dim=int(kv_cache.shape[-1]),
            rotary_dim=int(getattr(self, "_rope_rotary_dim", 0)),
            rope_base=float(getattr(self, "_rope_base", 10000.0)),
            is_neox_style=bool(getattr(self, "_rope_is_neox_style", True)),
        )
        self._init_server_transfer()

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        super().bind_connector_metadata(connector_metadata)
        self._meta = connector_metadata
        self._reap_save_futures(block=False)
        self._pending_commits = set()
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
        total_bytes_loaded = 0
        total_ipc_ms = 0.0
        total_copy_ms = 0.0
        total_sync_ms = 0.0
        total_l1_hits = 0
        total_l1_misses = 0
        total_l2_reads = 0
        total_transfer_open_ms = 0.0
        total_transfer_load_ms = 0.0
        total_transfer_sync_ms = 0.0
        for total_bytes, spans, per_req_ranges in load_batches:
            total_bytes_loaded += total_bytes
            staging_lease = self._acquire_staging(total_bytes, sample_tensor.device)
            staging = staging_lease.view
            try:
                cp_staging = cupy.asarray(staging)
                cuda_handle = export_cuda_ipc_handle(cp_staging)
                device_id = cuda_array_device_id(cp_staging)
                device_ptr = cuda_array_pointer(cp_staging)
                ipc_base_ptr, ipc_offset = _cuda_allocation_base_and_offset(device_ptr)

                ipc_start = time.perf_counter()
                load_response = self._submit_load_coroutine(
                    self._transfer_load_cuda(
                        cuda_ipc_handle=cuda_handle,
                        nbytes=total_bytes,
                        device_id=device_id,
                        device_ptr=device_ptr,
                        allocation_base_ptr=ipc_base_ptr,
                        allocation_offset=ipc_offset,
                        producer_pid=os.getpid(),
                        spans=spans,
                    )
                ).result(timeout=120.0)
                total_ipc_ms += (time.perf_counter() - ipc_start) * 1000
                total_transfer_open_ms += float(
                    load_response.get("transfer_open_ms", 0.0)
                )
                total_transfer_load_ms += float(
                    load_response.get("transfer_load_ms", 0.0)
                )
                total_transfer_sync_ms += float(
                    load_response.get("transfer_sync_ms", 0.0)
                )
                stats = load_response.get("transfer_stats_delta", {})
                if isinstance(stats, dict):
                    total_l1_hits += int(stats.get("l1_hits", 0))
                    total_l1_misses += int(stats.get("l1_misses", 0))
                    total_l2_reads += int(stats.get("l2_reads", 0))

                copy_runs = _build_load_copy_runs(per_req_ranges)
                total_copy_runs += len(copy_runs)
                copy_start = time.perf_counter()
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
                total_copy_ms += (time.perf_counter() - copy_start) * 1000
                sync_start = time.perf_counter()
                _synchronize_cuda_tensor(sample_tensor)
                total_sync_ms += (time.perf_counter() - sync_start) * 1000
            finally:
                staging_lease.release()

        logger.debug(
            "[CONNECTOR] start_load_kv timing: reqs=%d batches=%d bytes=%d "
            "copy_runs=%d gpu_copies=%d ipc_ms=%.3f copy_ms=%.3f "
            "sync_ms=%.3f transfer_open_ms=%.3f transfer_load_ms=%.3f "
            "transfer_sync_ms=%.3f l1_hits=%d l1_misses=%d l2_reads=%d",
            len(self._meta.reqs_to_load),
            len(load_batches),
            total_bytes_loaded,
            total_copy_runs,
            total_copies,
            total_ipc_ms,
            total_copy_ms,
            total_sync_ms,
            total_transfer_open_ms,
            total_transfer_load_ms,
            total_transfer_sync_ms,
            total_l1_hits,
            total_l1_misses,
            total_l2_reads,
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
        """Queue stores until vLLM reports request completion."""
        if self._meta is None:
            return

        commit_keys = list(self._pending_commits)
        reqs_to_store = dict(self._meta.reqs_to_store)
        if commit_keys and reqs_to_store:
            pending_finished = getattr(self, "_pending_finished_saves", None)
            if pending_finished is None:
                pending_finished = {}
                self._pending_finished_saves = pending_finished
            for req_id, spec in reqs_to_store.items():
                base_id = base_req_id(req_id)
                save = pending_finished.get(base_id)
                if save is None:
                    save = _DeferredFinishedSave(commit_keys=set(), reqs_to_store={})
                    pending_finished[base_id] = save
                save.reqs_to_store[req_id] = spec
                if spec.chunk_key in commit_keys:
                    save.commit_keys.add(spec.chunk_key)
        self._pending_commits.clear()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Collect completed background saves after a worker step.

        Args:
            finished_req_ids: Request IDs that vLLM finished in this step.

        Returns:
            Finished-saving request IDs and no async receiving IDs.
        """
        self._reap_save_futures(block=False)
        pending_finished = getattr(self, "_pending_finished_saves", {})
        if not pending_finished:
            return None, None

        finished_sending: set[str] = set()
        candidates = set(finished_req_ids)
        candidates.update(
            req_id for req_id, save in pending_finished.items() if save.submitted
        )
        for req_id in list(candidates):
            save = pending_finished.get(req_id)
            if save is None:
                continue
            if not save.submitted:
                save.future = self._submit_finished_save(save)
                save.submitted = True
            future = save.future
            if future is None:
                finished_sending.add(req_id)
                del pending_finished[req_id]
            elif future.done():
                future.result(timeout=120.0)
                finished_sending.add(req_id)
                del pending_finished[req_id]
        return finished_sending or None, None

    def shutdown(self) -> None:
        """Stop the background IO loop."""
        if self._role != KVConnectorRole.WORKER:
            return
        for req_id in list(getattr(self, "_pending_finished_saves", {})):
            self.get_finished({req_id})
        self._reap_save_futures(block=True)
        load_client = getattr(self, "_ipc_load_async", None)
        store_client = getattr(self, "_ipc_store_async", None)
        if load_client is not None:
            self._submit_load_coroutine(load_client.close()).result(timeout=10.0)
        if store_client is not None:
            self._submit_store_coroutine(store_client.close()).result(timeout=10.0)
        self._load_loop.call_soon_threadsafe(self._load_loop.stop)
        self._store_loop.call_soon_threadsafe(self._store_loop.stop)
        self._load_thread.join(timeout=5)
        self._store_thread.join(timeout=5)

    def _run_load_loop(self) -> None:
        """Run the foreground load asyncio IO loop."""
        asyncio.set_event_loop(self._load_loop)
        self._load_loop.run_forever()

    def _run_store_loop(self) -> None:
        """Run the background store asyncio IO loop."""
        asyncio.set_event_loop(self._store_loop)
        self._store_loop.run_forever()

    def _submit_load_coroutine(self, coro: Any) -> Any:
        """Submit foreground load work to the dedicated load event loop.

        Args:
            coro: Coroutine object to schedule.

        Returns:
            Future returned by ``asyncio.run_coroutine_threadsafe``.

        Async/thread-safety:
            Called from vLLM worker threads. A load-only loop prevents cache-hit
            reads from queueing behind background store coroutines.
        """
        loop = self._load_loop
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _submit_store_coroutine(self, coro: Any) -> Any:
        """Submit background store and commit work to the store event loop.

        Args:
            coro: Coroutine object to schedule.

        Returns:
            Future returned by ``asyncio.run_coroutine_threadsafe``.

        Async/thread-safety:
            Called from vLLM worker threads. Store work is serialized on the
            store loop and does not occupy the foreground load loop.
        """
        loop = self._store_loop
        return asyncio.run_coroutine_threadsafe(coro, loop)

    def _ensure_transfer_ready(self) -> bool:
        """Refresh server transfer config and mark worker data plane ready."""
        if getattr(self, "_transfer_ready", False):
            return True

        self._refresh_runtime_config()
        if not self._slot_size or (
            not getattr(self, "_skip_l2", False) and not self._store_path
        ):
            logger.warning(
                "[CONNECTOR] server transfer config is not ready; start DaseR server "
                "before sending requests",
            )
            return False

        self._transfer_ready = True
        logger.info("[CONNECTOR] server transfer mode=%s", self._transfer_mode)
        return True

    def _init_server_transfer(self) -> None:
        """Initialize the server-owned transfer layer on both IO loops.

        Async/thread-safety:
            Called from a vLLM worker thread after KV-cache registration. Does
            nothing until the server transfer config and both async IO loops
            are ready.
        """
        if not (
            self._ensure_transfer_ready()
            and getattr(self, "_ipc_load_async", None) is not None
            and getattr(self, "_ipc_store_async", None) is not None
            and getattr(self, "_load_loop", None) is not None
            and getattr(self, "_store_loop", None) is not None
        ):
            return
        self._submit_load_coroutine(self._ipc_load_async.init_transfer()).result(
            timeout=120.0
        )
        self._submit_store_coroutine(self._ipc_store_async.init_transfer()).result(
            timeout=120.0
        )

    def _reap_save_futures(self, block: bool) -> None:
        """Collect completed background save tasks.

        Args:
            block: If True, wait for every pending save. If False, collect only
                tasks that are already complete.
        """
        remaining: list[_SaveFuture] = []
        pending_bytes = self._pending_save_staging_bytes
        for record in self._save_futures:
            if block or record.future.done():
                try:
                    record.future.result(timeout=120.0)
                finally:
                    pending_bytes = max(0, pending_bytes - record.staging_bytes)
                    record.release()
            else:
                remaining.append(record)
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
        self._pending_save_staging_bytes += staging_bytes
        self._save_futures.append(
            _SaveFuture(future=future, staging_bytes=staging_bytes, lease=staging_lease)
        )

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
        while self._pending_save_staging_bytes + nbytes > limit and self._save_futures:
            record = self._save_futures.pop(0)
            try:
                record.future.result(timeout=120.0)
            finally:
                self._pending_save_staging_bytes = max(
                    0,
                    self._pending_save_staging_bytes - record.staging_bytes,
                )
                record.release()
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
        cross_layer_kv_cache = self._kv_caches.get(CROSS_LAYER_KV_CACHE_KEY)
        if cross_layer_kv_cache is not None:
            _copy_cross_layer_kv_cache_to_staging(
                staging=staging,
                kv_cache=cross_layer_kv_cache,
                block_ids=block_ids,
                num_layers=num_layers,
                slot_size=self._slot_size,
                block_index=block_index,
            )
        else:
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

    def _submit_finished_save(self, save: _DeferredFinishedSave) -> Any | None:
        """Submit one request's deferred KV store after request completion.

        Args:
            save: Deferred store plan built during ``wait_for_save``.

        Returns:
            Future that completes after all store batches are committed, or
            ``None`` when no store batch could be staged.

        Async/thread-safety:
            Called by vLLM's worker thread from ``get_finished`` while vLLM is
            still holding the finished request's KV blocks.
        """
        batch_futures = []
        batches = _build_staging_store_batches(
            save.reqs_to_store,
            self._slot_size,
            max_batch_bytes=(self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES),
        )
        for block_ids, spans in batches:
            staged = self._stage_store_batch(block_ids, spans)
            if staged is None:
                continue
            future = self._submit_store_coroutine(
                self._write_cuda_buffer(
                    buffer=staged.buffer,
                    ready_event=staged.ready_event,
                    spans=staged.spans,
                )
            )
            self._track_save_future(future, staged.buffer.nbytes, staged.lease)
            batch_futures.append(future)
        if not batch_futures:
            return None
        commit_future = self._submit_store_coroutine(
            self._commit_after_store_futures(batch_futures, sorted(save.commit_keys)),
        )
        self._track_save_future(commit_future, 0, None)
        return commit_future

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
        ipc_base_ptr, ipc_offset = _cuda_allocation_base_and_offset(device_ptr)
        stored_keys = await self._transfer_store_cuda(
            cuda_ipc_handle=cuda_ipc_handle,
            nbytes=buffer.nbytes,
            device_id=device_id,
            device_ptr=device_ptr,
            allocation_base_ptr=ipc_base_ptr,
            allocation_offset=ipc_offset,
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
        candidate_keys = [key for key in stored_keys if key in requested]
        keys_to_commit = list(dict.fromkeys(candidate_keys))
        await self._ipc_store_async.commit_chunks(keys_to_commit)

    async def _transfer_load_cuda(self, **kwargs: Any) -> dict[str, Any]:
        """Load through the dedicated worker load IPC client.

        Args:
            **kwargs: forwarded CUDA transfer payload fields.

        Returns:
            Server load response with timing counters.

        Async/thread-safety:
            Runs on the worker load event loop. A dedicated client keeps
            cache-hit loads from queueing behind store RPCs.
        """
        return await self._ipc_load_async.transfer_load_cuda(**kwargs)

    async def _transfer_store_cuda(self, **kwargs: Any) -> list[str]:
        """Store through the dedicated worker store IPC client.

        Args:
            **kwargs: forwarded CUDA transfer payload fields.

        Returns:
            Chunk keys accepted by the server.

        Async/thread-safety:
            Runs on the worker store event loop and serializes only with other
            store/commit traffic.
        """
        return await self._ipc_store_async.transfer_store_cuda(**kwargs)
