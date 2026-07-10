# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Standard
import asyncio
from collections import deque
from dataclasses import dataclass
import os
import queue
import threading
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
    FixedCudaStagingPool,
    StagedStoreBatch,
    StoreCudaStagingPool,
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
_LOAD_REQUEST_MAX_INFLIGHT = 8
_LOAD_DISPATCH_WAIT_TIMEOUT_S = 0.001
_LOAD_STAGING_RESERVE_BYTES = 1 << 30
_MIN_STORE_STAGING_POOL_DEPTH = 1
_LoadBatch = tuple[int, list[dict[str, int]], list[Any]]


def _store_staging_pool_depth(buffer_bytes: int, pending_limit_bytes: int) -> int:
    """Return fixed store staging pool depth for the configured byte budget.

    Args:
        buffer_bytes: Capacity of one fixed staging buffer.
        pending_limit_bytes: Total pending store staging byte budget.

    Returns:
        Number of fixed store staging buffers to preallocate.

    Async/thread-safety:
        Pure helper used during worker-side pool initialization.
    """
    if buffer_bytes <= 0:
        raise ValueError("buffer_bytes must be positive")
    if pending_limit_bytes <= 0:
        return _MIN_STORE_STAGING_POOL_DEPTH
    return max(_MIN_STORE_STAGING_POOL_DEPTH, pending_limit_bytes // buffer_bytes)


def _load_staging_pool_depth(
    buffer_bytes: int,
    pending_limit_bytes: int,
    device: torch.device,
) -> int:
    """Return fixed load staging depth under memory and inflight constraints.

    Args:
        buffer_bytes: Capacity of one fixed staging buffer.
        pending_limit_bytes: Existing staging byte budget from store limits.
        device: CUDA device used for staging allocation.

    Returns:
        Number of fixed load staging buffers to preallocate.

    Async/thread-safety:
        Pure helper except for querying CUDA free memory. Called during worker
        initialization before request traffic.
    """
    depth = min(
        _LOAD_REQUEST_MAX_INFLIGHT,
        _store_staging_pool_depth(buffer_bytes, pending_limit_bytes),
    )
    if device.type != "cuda":
        return max(1, depth)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except Exception:  # noqa: BLE001
        return max(1, depth)
    usable_bytes = max(0, int(free_bytes) - _LOAD_STAGING_RESERVE_BYTES)
    memory_depth = max(1, usable_bytes // buffer_bytes)
    return max(1, min(depth, memory_depth))


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


@dataclass
class _PendingLoad:
    """One background load future tracked until vLLM can resume the request.

    Attributes:
        future: Future running the cross-layer load work.
        block_ids: vLLM KV block IDs targeted by the load.
        lease: optional staging lease held until the load completes.
    """

    future: Any
    block_ids: list[int]
    lease: CudaStagingLease | None

    def release(self) -> None:
        """Release the reusable staging lease, if any."""
        if self.lease is not None:
            self.lease.release()
            self.lease = None


@dataclass
class _InflightLoadBatch:
    """One submitted load batch and its fixed staging lease.

    Attributes:
        buffer_index: Fixed load staging buffer index.
        total_bytes: Logical bytes in this transfer batch.
        per_req_ranges: Ranges used to restore staging bytes into vLLM KV cache.
        staging_lease: Fixed staging lease retained until restore completes.
        future: Future returned by the load event loop.
        submitted_at: Wall-clock timestamp used for wait timing.
    """

    buffer_index: int
    total_bytes: int
    per_req_ranges: list[Any]
    staging_lease: CudaStagingLease
    future: Any
    submitted_at: float


@dataclass
class _InflightRequestLoad:
    """One request-level load submitted by the dispatcher.

    Attributes:
        item: Original queued request item.
        buffer_index: Fixed staging buffer owned by this request state.
        batches: Bounded read batches for this request.
        next_batch: Index of the next unsubmitted read batch.
        remaining_batches: Number of read batches not yet consumed.
        active: Current in-flight read batch for this request, if any.
        completed: Consumed batch timing rows for logging.
    """

    item: "_QueuedLoadRequest"
    buffer_index: int
    batches: list[_LoadBatch]
    next_batch: int
    remaining_batches: int
    active: _InflightLoadBatch | None
    completed: list["_ConsumedLoadBatch"]


@dataclass
class _ConsumedLoadBatch:
    """Timing and accounting data for one restored load batch."""

    buffer_index: int
    bytes: int
    copies: int
    copy_runs: int
    ipc_ms: float
    wait_ms: float
    copy_ms: float
    transfer_open_ms: float
    transfer_load_ms: float
    transfer_sync_ms: float
    l1_hits: int
    l1_misses: int
    l2_reads: int


class _ImmediateLoadError:
    """Completed future used when a load cannot be submitted.

    Args:
        message: Error message raised when the future is collected.

    Async/thread-safety:
        Immutable testable stand-in for a failed background future. It does not
        spawn threads or perform IO.
    """

    def __init__(self, message: str) -> None:
        self._message = message

    def done(self) -> bool:
        """Return True because this failed future is already complete."""
        return True

    def result(self, timeout: float | None = None) -> None:
        """Raise the seeded load-start failure.

        Args:
            timeout: Ignored timeout for ``Future`` API compatibility.
        """
        del timeout
        raise RuntimeError(self._message)


class _RequestLoadFuture:
    """Small future used to release request loads independently.

    Async/thread-safety:
        The connector load executor marks completion and the vLLM worker thread
        polls ``done``/``result`` from ``get_finished``. ``threading.Event``
        provides cross-thread visibility for the result state.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._error: BaseException | None = None

    def done(self) -> bool:
        """Return whether this request load has completed."""
        return self._event.is_set()

    def result(self, timeout: float | None = None) -> None:
        """Wait for completion and raise a stored load error, if any.

        Args:
            timeout: Optional timeout in seconds.
        """
        if not self._event.wait(timeout):
            raise TimeoutError("request load did not complete before timeout")
        if self._error is not None:
            raise self._error

    def set_result(self) -> None:
        """Mark this request load as successful."""
        self._event.set()

    def set_exception(self, error: BaseException) -> None:
        """Mark this request load as failed.

        Args:
            error: Exception to re-raise from ``result``.
        """
        self._error = error
        self._event.set()


@dataclass
class _QueuedLoadRequest:
    """One request waiting in the worker-side load queue.

    Attributes:
        req_id: Base vLLM request ID used for completion.
        spec_id: Scheduler metadata ID for this load spec.
        spec: Load spec to restore into the vLLM KV cache.
        future: Per-request completion future observed by ``get_finished``.
    """

    req_id: str
    spec_id: str
    spec: Any
    future: _RequestLoadFuture


class LoadRequestDispatcher:
    """Bound request-level load concurrency by max in-flight and staging depth.

    Args:
        max_inflight: Maximum request loads allowed in flight.
        staging_depth: Number of fixed staging buffers available to request
            loads.

    Async/thread-safety:
        The dispatcher object is owned by the connector load asyncio loop. Test
        helpers may call its pure synchronous scheduling helpers directly.
    """

    def __init__(
        self,
        max_inflight: int,
        staging_depth: int,
    ) -> None:
        self._effective_inflight = max(1, min(max_inflight, staging_depth))
        self._free_buffers: deque[int] = deque(range(self._effective_inflight))

    @property
    def effective_inflight(self) -> int:
        """Return the active request limit after applying staging depth."""
        return self._effective_inflight

    def submit_ready(
        self,
        connector: Any,
        queued: list[_QueuedLoadRequest],
        sample_tensor: torch.Tensor,
    ) -> list[_InflightRequestLoad]:
        """Submit queued requests while in-flight slots and buffers are free.

        Args:
            connector: Worker connector that owns submit helpers.
            queued: Mutable FIFO list of queued request work.
            sample_tensor: Representative KV cache tensor.

        Returns:
            Newly submitted request states.
        """
        submitted: list[_InflightRequestLoad] = []
        while queued and self._free_buffers:
            buffer_index = self._free_buffers.popleft()
            item = queued.pop(0)
            state = connector._submit_request_load_for_dispatcher(  # noqa: SLF001
                item,
                buffer_index,
                sample_tensor,
            )
            if state.active is None:
                self._free_buffers.append(state.buffer_index)
                continue
            submitted.append(state)
        return submitted

    def consume_ready(
        self,
        connector: Any,
        active: list[_InflightRequestLoad],
        sample_tensor: torch.Tensor,
    ) -> list[_InflightRequestLoad]:
        """Consume completed request loads and release their buffers.

        Args:
            connector: Worker connector that owns consume helpers.
            active: Mutable list of active request load states.
            sample_tensor: Representative KV cache tensor.

        Returns:
            States consumed during this call.
        """
        consumed: list[_InflightRequestLoad] = []
        for state in list(active):
            active_batch = state.active
            if active_batch is None:
                active.remove(state)
                self._free_buffers.append(state.buffer_index)
                consumed.append(state)
                continue
            if not active_batch.future.done():
                continue
            reusable_buffer, request_done = connector._consume_dispatcher_load(  # noqa: SLF001
                state,
                sample_tensor,
            )
            if request_done:
                active.remove(state)
                self._free_buffers.append(reusable_buffer)
            consumed.append(state)
        return consumed


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
            self._store_staging_pool = StoreCudaStagingPool(
                device=sample.device,
                buffer_bytes=self._store_staging_bytes,
                depth=_store_staging_pool_depth(
                    self._store_staging_bytes,
                    self._pending_store_staging_limit_bytes,
                ),
            )
            self._load_staging_pool = FixedCudaStagingPool(
                device=sample.device,
                buffer_bytes=self._store_staging_bytes,
                depth=_load_staging_pool_depth(
                    self._store_staging_bytes,
                    self._pending_store_staging_limit_bytes,
                    sample.device,
                ),
            )
            self._load_staging_registered = False
            logger.info(
                "[CONNECTOR] preallocated staging buffer=%d cap=%d pending=%d "
                "load_request_max_inflight=%d load_staging_depth=%d",
                self._store_staging_bytes,
                self._store_staging_bytes,
                self._pending_store_staging_limit_bytes,
                _LOAD_REQUEST_MAX_INFLIGHT,
                self._load_staging_pool.depth,
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
        self._store_staging_pool = StoreCudaStagingPool(
            device=kv_cache.device,
            buffer_bytes=self._store_staging_bytes,
            depth=_store_staging_pool_depth(
                self._store_staging_bytes,
                self._pending_store_staging_limit_bytes,
            ),
        )
        self._load_staging_pool = FixedCudaStagingPool(
            device=kv_cache.device,
            buffer_bytes=self._store_staging_bytes,
            depth=_load_staging_pool_depth(
                self._store_staging_bytes,
                self._pending_store_staging_limit_bytes,
                kv_cache.device,
            ),
        )
        self._load_staging_registered = False
        logger.info(
            "[CONNECTOR] register_cross_layers_kv_cache: layers=%d shape=%s "
            "dtype=%s load_request_max_inflight=%d load_staging_depth=%d",
            len(self._layer_names),
            tuple(kv_cache.shape),
            kv_cache.dtype,
            _LOAD_REQUEST_MAX_INFLIGHT,
            self._load_staging_pool.depth,
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
        """Submit async KV cache loads for cache-hit requests.

        Args:
            forward_context: vLLM ForwardContext for this forward pass.
        """
        del forward_context, kwargs
        if self._meta is None or not self._meta.reqs_to_load:
            return
        logger.debug(
            "[CONNECTOR] start_load_kv: %d reqs to load",
            len(self._meta.reqs_to_load),
        )
        reqs_to_load = dict(self._meta.reqs_to_load)
        if not self._ensure_transfer_ready():
            self._mark_load_start_failed(
                reqs_to_load,
                "server transfer config is not ready",
            )
            return

        num_layers = len(self._layer_names)
        if num_layers == 0:
            self._mark_load_start_failed(reqs_to_load, "no registered KV cache layers")
            return

        sample_tensor = next(iter(self._kv_caches.values()), None)
        if sample_tensor is None:
            self._mark_load_start_failed(reqs_to_load, "no registered KV cache tensor")
            return

        pending_loads = getattr(self, "_pending_loads", None)
        if pending_loads is None:
            pending_loads = {}
            self._pending_loads = pending_loads
        load_queue = self._ensure_load_request_queue(sample_tensor)
        for spec_id, spec in reqs_to_load.items():
            req_id = base_req_id(spec_id)
            request_future = _RequestLoadFuture()
            pending_loads[req_id] = _PendingLoad(
                future=request_future,
                block_ids=list(spec.block_ids),
                lease=None,
            )
            self._enqueue_load_request(
                load_queue,
                _QueuedLoadRequest(
                    req_id=req_id,
                    spec_id=spec_id,
                    spec=spec,
                    future=request_future,
                ),
            )
        self._ensure_load_request_dispatcher(sample_tensor)

    def _mark_load_start_failed(
        self,
        reqs_to_load: dict[str, Any],
        reason: str,
    ) -> None:
        """Record failed load submission so vLLM can release waiting requests.

        Args:
            reqs_to_load: Load metadata that could not be submitted.
            reason: Human-readable failure reason used in diagnostics.

        Async/thread-safety:
            Called on the vLLM worker thread before any background load is
            started. Completion is later reported through ``get_finished``.
        """
        if not reqs_to_load:
            return
        block_ids = [
            block_id for spec in reqs_to_load.values() for block_id in spec.block_ids
        ]
        failed_future = _ImmediateLoadError(reason)
        pending_loads = getattr(self, "_pending_loads", None)
        if pending_loads is None:
            pending_loads = {}
            self._pending_loads = pending_loads
        for req_id in {base_req_id(req_id) for req_id in reqs_to_load}:
            pending_loads[req_id] = _PendingLoad(
                future=failed_future,
                block_ids=list(block_ids),
                lease=None,
            )

    def _ensure_load_request_queue(
        self, sample_tensor: torch.Tensor | None = None
    ) -> Any:
        """Return the worker-side request load queue, creating it if needed.

        Returns:
            Queue used to group request loads across scheduler windows.

        Async/thread-safety:
            Called on the vLLM worker thread. Attribute installation is guarded
            by a small lock because multiple model-runner calls may race during
            startup.
        """
        load_queue = getattr(self, "_load_request_queue", None)
        if load_queue is not None:
            return load_queue
        lock = getattr(self, "_load_request_queue_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._load_request_queue_lock = lock
        with lock:
            load_queue = getattr(self, "_load_request_queue", None)
            if load_queue is None:
                if sample_tensor is not None and hasattr(self, "_load_loop"):
                    future = asyncio.run_coroutine_threadsafe(
                        self._create_load_request_queue(),
                        self._load_loop,
                    )
                    load_queue = future.result(timeout=10.0)
                else:
                    load_queue = queue.Queue()
                self._load_request_queue = load_queue
        return load_queue

    async def _create_load_request_queue(self) -> asyncio.Queue[Any]:
        """Create an asyncio request queue on the load event loop."""
        return asyncio.Queue()

    def _enqueue_load_request(
        self,
        load_queue: Any,
        request: _QueuedLoadRequest,
    ) -> None:
        """Enqueue one request from a worker thread into the load queue.

        Args:
            load_queue: Queue created by ``_ensure_load_request_queue``.
            request: Request-level load work item.

        Async/thread-safety:
            Called on vLLM worker threads. Production queues are asyncio queues
            owned by ``_load_loop``; test queues may be synchronous ``queue.Queue``
            instances.
        """
        if isinstance(load_queue, asyncio.Queue):
            self._load_loop.call_soon_threadsafe(load_queue.put_nowait, request)
        else:
            load_queue.put(request)

    def _ensure_load_request_dispatcher(self, sample_tensor: torch.Tensor) -> None:
        """Start the persistent request load dispatcher if it is not running.

        Args:
            sample_tensor: Representative KV cache tensor used by load workers.

        Async/thread-safety:
            Called on the vLLM worker thread. The dispatcher itself runs on the
            connector load asyncio loop and keeps request-level loads in flight
            up to the fixed max-inflight/staging-pool limit.
        """
        dispatcher_future = getattr(self, "_load_request_dispatcher_future", None)
        if dispatcher_future is not None and not dispatcher_future.done():
            return
        lock = getattr(self, "_load_request_queue_lock", None)
        if lock is None:
            lock = threading.Lock()
            self._load_request_queue_lock = lock
        with lock:
            dispatcher_future = getattr(self, "_load_request_dispatcher_future", None)
            if dispatcher_future is not None and not dispatcher_future.done():
                return
            self._load_request_dispatcher_future = asyncio.run_coroutine_threadsafe(
                self._run_load_request_dispatcher(sample_tensor),
                self._load_loop,
            )

    async def _run_load_request_dispatcher(self, sample_tensor: torch.Tensor) -> None:
        """Continuously dispatch request loads as in-flight slots become free.

        Args:
            sample_tensor: Representative KV cache tensor used for load staging.

        Async/thread-safety:
            Runs on the connector load asyncio loop. The dispatcher never waits
            to coalesce requests; it submits each queued request immediately when
            both an in-flight slot and a fixed staging buffer are available.
        """
        load_queue = self._ensure_load_request_queue(sample_tensor)
        load_staging_pool = self._ensure_load_staging_pool(sample_tensor)
        dispatcher = LoadRequestDispatcher(
            max_inflight=_LOAD_REQUEST_MAX_INFLIGHT,
            staging_depth=load_staging_pool.depth,
        )
        queued: list[_QueuedLoadRequest] = []
        active: list[_InflightRequestLoad] = []
        try:
            while True:
                if not queued:
                    if active:
                        await self._drain_load_queue(load_queue, queued)
                    else:
                        item = await load_queue.get()
                        if item is None:
                            return
                        queued.append(item)
                active.extend(dispatcher.submit_ready(self, queued, sample_tensor))
                consumed = dispatcher.consume_ready(self, active, sample_tensor)
                if consumed:
                    continue
                if not active:
                    continue
                await self._wait_for_dispatcher_completion(active)
        except BaseException as exc:
            for state in active:
                if not state.item.future.done():
                    state.item.future.set_exception(exc)
                active_batch = state.active
                if active_batch is not None:
                    active_batch.staging_lease.release()
            for item in queued:
                if not item.future.done():
                    item.future.set_exception(exc)
            raise

    async def _drain_load_queue(
        self,
        load_queue: asyncio.Queue[Any],
        queued: list[_QueuedLoadRequest],
    ) -> None:
        """Move immediately available queue items into a local FIFO list."""
        while True:
            try:
                item = load_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if item is None:
                await load_queue.put(None)
                return
            queued.append(item)

    async def _wait_for_dispatcher_completion(
        self,
        active: list[_InflightRequestLoad],
    ) -> None:
        """Wait until any request-level active read future completes."""
        wrapped: dict[asyncio.Future[Any], _InflightRequestLoad] = {}
        for state in active:
            active_batch = state.active
            if active_batch is None:
                continue
            wrapped[asyncio.wrap_future(active_batch.future)] = state
        if not wrapped:
            await asyncio.sleep(0)
            return
        done, _pending = await asyncio.wait(
            wrapped.keys(),
            timeout=_LOAD_DISPATCH_WAIT_TIMEOUT_S,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            await asyncio.sleep(0)

    def _ensure_load_staging_pool(
        self,
        sample_tensor: torch.Tensor,
    ) -> FixedCudaStagingPool:
        """Return the fixed load staging pool, creating it before traffic if needed.

        Args:
            sample_tensor: Representative KV cache tensor used for device
                placement when the pool was not initialized during registration.

        Returns:
            Fixed load staging pool with two preallocated buffers.

        Async/thread-safety:
            Called from the worker load executor. Normal production flow creates
            the pool during KV-cache registration; this fallback keeps tests and
            deferred initialization paths explicit while still allocating only
            once before batch submission.
        """
        pool = getattr(self, "_load_staging_pool", None)
        if pool is not None:
            return pool
        buffer_bytes = max(
            self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES,
            self._slot_size,
        )
        pool = FixedCudaStagingPool(
            device=sample_tensor.device,
            buffer_bytes=buffer_bytes,
            depth=_load_staging_pool_depth(
                buffer_bytes,
                getattr(self, "_pending_store_staging_limit_bytes", 0),
                sample_tensor.device,
            ),
        )
        self._load_staging_pool = pool
        return pool

    def _submit_load_batch(
        self,
        batch: _LoadBatch,
        buffer_index: int,
        sample_tensor: torch.Tensor,
    ) -> _InflightLoadBatch:
        """Submit one load batch into a fixed staging buffer.

        Args:
            batch: Tuple from ``build_load_read_batches``.
            buffer_index: Fixed load staging buffer to use.
            sample_tensor: Representative KV cache tensor for device context.

        Returns:
            In-flight batch state consumed by ``_consume_loaded_batch``.

        Async/thread-safety:
            Runs on the connector load executor. The returned state owns the
            fixed staging lease until consumption finishes.
        """
        del sample_tensor
        total_bytes, spans, per_req_ranges = batch
        pool = self._ensure_load_staging_pool(next(iter(self._kv_caches.values())))
        staging_lease = pool.acquire_index(buffer_index, total_bytes)
        staging = staging_lease.view

        submitted_at = time.perf_counter()
        if getattr(self, "_load_staging_registered", False):
            transfer_coro = self._transfer_load_registered_cuda(
                buffer_index=buffer_index,
                nbytes=total_bytes,
                spans=spans,
            )
        else:
            cp_staging = cupy.asarray(staging)
            cuda_handle = export_cuda_ipc_handle(cp_staging)
            device_id = cuda_array_device_id(cp_staging)
            device_ptr = cuda_array_pointer(cp_staging)
            ipc_base_ptr, ipc_offset = _cuda_allocation_base_and_offset(device_ptr)
            transfer_coro = self._transfer_load_cuda(
                buffer_index=buffer_index,
                cuda_ipc_handle=cuda_handle,
                nbytes=total_bytes,
                device_id=device_id,
                device_ptr=device_ptr,
                allocation_base_ptr=ipc_base_ptr,
                allocation_offset=ipc_offset,
                producer_pid=os.getpid(),
                spans=spans,
            )
        future = self._submit_load_coroutine(transfer_coro)
        return _InflightLoadBatch(
            buffer_index=buffer_index,
            total_bytes=total_bytes,
            per_req_ranges=per_req_ranges,
            staging_lease=staging_lease,
            future=future,
            submitted_at=submitted_at,
        )

    def _submit_request_load_for_dispatcher(
        self,
        item: _QueuedLoadRequest,
        buffer_index: int,
        sample_tensor: torch.Tensor,
    ) -> _InflightRequestLoad:
        """Submit the first read batch for one queued request load.

        Args:
            item: Request-level queued load.
            buffer_index: Fixed staging buffer index reserved by the dispatcher.
            sample_tensor: Representative KV cache tensor.

        Returns:
            Request state tracked by ``LoadRequestDispatcher``.

        Async/thread-safety:
            Runs on the connector load asyncio loop. Each request owns at most one
            active staging buffer at a time; multi-segment requests submit the
            next segment only after the previous segment has been restored.
        """
        load_staging_pool = self._ensure_load_staging_pool(sample_tensor)
        load_batches = _build_load_read_batches(
            {item.spec_id: item.spec},
            self._slot_size,
            max_batch_bytes=load_staging_pool.buffer_bytes,
            include_req_ids=True,
        )
        if not load_batches:
            item.future.set_result()
            return _InflightRequestLoad(
                item=item,
                buffer_index=buffer_index,
                batches=[],
                next_batch=0,
                remaining_batches=0,
                active=None,
                completed=[],
            )
        active_batch = self._submit_load_batch(
            load_batches[0],
            buffer_index,
            sample_tensor,
        )
        return _InflightRequestLoad(
            item=item,
            buffer_index=buffer_index,
            batches=load_batches,
            next_batch=1,
            remaining_batches=len(load_batches),
            active=active_batch,
            completed=[],
        )

    def _consume_dispatcher_load(
        self,
        state: _InflightRequestLoad,
        sample_tensor: torch.Tensor,
    ) -> tuple[int, bool]:
        """Consume one completed request read and finish or advance the request.

        Args:
            state: Request-level load state with a completed active batch.
            sample_tensor: Representative KV cache tensor.

        Returns:
            Tuple of released staging buffer index and whether the request is
            fully complete.

        Async/thread-safety:
            Runs on the connector load asyncio loop after the associated transfer
            future is complete. The staging buffer is not reused until restore
            kernels have synchronized and the lease has been released.
        """
        active_batch = state.active
        if active_batch is None:
            return state.buffer_index, True
        consumed = self._consume_loaded_batch(active_batch, sample_tensor)
        state.completed.append(consumed)
        state.remaining_batches = max(0, state.remaining_batches - 1)
        reusable_buffer = consumed.buffer_index
        state.buffer_index = reusable_buffer
        if state.remaining_batches == 0:
            if not state.item.future.done():
                state.item.future.set_result()
            return reusable_buffer, True
        state.active = self._submit_load_batch(
            state.batches[state.next_batch],
            reusable_buffer,
            sample_tensor,
        )
        state.next_batch += 1
        return reusable_buffer, False

    def _consume_loaded_batch(
        self,
        state: _InflightLoadBatch,
        sample_tensor: torch.Tensor,
    ) -> _ConsumedLoadBatch:
        """Wait for one submitted load batch and restore it into vLLM KV cache.

        Args:
            state: In-flight load batch returned by ``_submit_load_batch``.
            sample_tensor: Representative KV cache tensor for synchronization.

        Returns:
            Timing and accounting data for the restored batch.

        Async/thread-safety:
            Runs on the connector load executor. It releases the fixed staging
            lease only after restore kernels that read the staging view have
            been synchronized.
        """
        try:
            wait_start = time.perf_counter()
            load_response = state.future.result(timeout=120.0)
            wait_ms = (time.perf_counter() - wait_start) * 1000
            ipc_ms = (time.perf_counter() - state.submitted_at) * 1000
            transfer_open_ms = float(load_response.get("transfer_open_ms", 0.0))
            transfer_load_ms = float(load_response.get("transfer_load_ms", 0.0))
            transfer_sync_ms = float(load_response.get("transfer_sync_ms", 0.0))
            stats = load_response.get("transfer_stats_delta", {})
            l1_hits = 0
            l1_misses = 0
            l2_reads = 0
            if isinstance(stats, dict):
                l1_hits = int(stats.get("l1_hits", 0))
                l1_misses = int(stats.get("l1_misses", 0))
                l2_reads = int(stats.get("l2_reads", 0))

            copy_runs = _build_load_copy_runs(state.per_req_ranges)
            copy_start = time.perf_counter()
            copies = 0
            staging = state.staging_lease.view
            for run in copy_runs:
                copies += _copy_staging_to_kv_cache(
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
            copy_ms = (time.perf_counter() - copy_start) * 1000
            _synchronize_cuda_tensor(sample_tensor)
            return _ConsumedLoadBatch(
                buffer_index=state.buffer_index,
                bytes=state.total_bytes,
                copies=copies,
                copy_runs=len(copy_runs),
                ipc_ms=ipc_ms,
                wait_ms=wait_ms,
                copy_ms=copy_ms,
                transfer_open_ms=transfer_open_ms,
                transfer_load_ms=transfer_load_ms,
                transfer_sync_ms=transfer_sync_ms,
                l1_hits=l1_hits,
                l1_misses=l1_misses,
                l2_reads=l2_reads,
            )
        finally:
            state.staging_lease.release()

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op because async loads complete before vLLM resumes requests.

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
        """Collect completed background transfers after a worker step.

        Args:
            finished_req_ids: Request IDs that vLLM finished in this step.

        Returns:
            Finished-saving request IDs and finished async-loading request IDs.
        """
        self._reap_save_futures(block=False)
        finished_recving = self._collect_finished_loads()
        pending_finished = getattr(self, "_pending_finished_saves", {})
        if not pending_finished:
            return None, finished_recving or None

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
        return finished_sending or None, finished_recving or None

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return and clear block IDs whose async load failed.

        Returns:
            vLLM block IDs that should be treated as invalid.
        """
        invalid_blocks = getattr(self, "_invalid_load_block_ids", None)
        if invalid_blocks is None:
            self._invalid_load_block_ids = set()
            return set()
        invalid = set(invalid_blocks)
        invalid_blocks.clear()
        return invalid

    def _collect_finished_loads(self) -> set[str]:
        """Poll async load futures without blocking.

        Returns:
            Base request IDs whose async load future completed in this poll.
        """
        pending_loads = getattr(self, "_pending_loads", {})
        if not pending_loads:
            return set()

        finished_recving: set[str] = set()
        collected_futures: set[int] = set()
        for req_id, load in list(pending_loads.items()):
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
                self._invalid_load_block_ids.update(load.block_ids)
                collected_futures.add(future_id)
            finally:
                load.release()
                del pending_loads[req_id]
            finished_recving.add(req_id)
        return finished_recving

    def shutdown(self) -> None:
        """Stop the background IO loop."""
        if self._role != KVConnectorRole.WORKER:
            return
        pending_loads = getattr(self, "_pending_loads", {})
        for load in {id(load.future): load for load in pending_loads.values()}.values():
            if not load.future.done():
                try:
                    load.future.result(timeout=120.0)
                except Exception:  # noqa: BLE001
                    pass
        self._collect_finished_loads()
        for req_id in list(getattr(self, "_pending_finished_saves", {})):
            self.get_finished({req_id})
        self._reap_save_futures(block=True)
        load_queue = getattr(self, "_load_request_queue", None)
        if load_queue is not None:
            if isinstance(load_queue, asyncio.Queue):
                self._load_loop.call_soon_threadsafe(load_queue.put_nowait, None)
            else:
                load_queue.put(None)
        dispatcher_future = getattr(self, "_load_request_dispatcher_future", None)
        if dispatcher_future is not None:
            try:
                dispatcher_future.result(timeout=120.0)
            except Exception:  # noqa: BLE001
                pass
        load_clients = list(
            dict.fromkeys(
                getattr(self, "_ipc_load_async_pool", [])
                or [getattr(self, "_ipc_load_async", None)]
            )
        )
        store_client = getattr(self, "_ipc_store_async", None)
        for load_client in load_clients:
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
        for load_client in self._load_ipc_clients():
            self._submit_load_coroutine(load_client.init_transfer()).result(
                timeout=120.0
            )
        self._submit_store_coroutine(self._ipc_store_async.init_transfer()).result(
            timeout=120.0
        )
        self._register_load_staging_buffers()

    def _load_ipc_clients(self) -> list[Any]:
        """Return fixed load IPC clients used for parallel load RPCs.

        Returns:
            Load IPC clients. The list falls back to the legacy single client
            when the connector was constructed by tests or older harnesses.

        Async/thread-safety:
            Called on the vLLM worker thread during initialization and from the
            load event loop when selecting the client for a submitted batch.
        """
        clients = getattr(self, "_ipc_load_async_pool", None)
        if clients:
            return list(clients)
        client = getattr(self, "_ipc_load_async", None)
        return [client] if client is not None else []

    def _load_ipc_client_for_buffer(self, buffer_index: int | None = None) -> Any:
        """Return the load IPC client assigned to a staging buffer.

        Args:
            buffer_index: Fixed staging buffer index for the transfer.

        Returns:
            Async IPC client for the selected load lane.

        Async/thread-safety:
            Pure selection helper. Each returned client owns its own IPC socket,
            so separate fixed staging buffers can have concurrent server RPCs
            instead of serializing on one client lock.
        """
        clients = self._load_ipc_clients()
        if not clients:
            raise RuntimeError("load IPC client is not initialized")
        if buffer_index is None:
            return clients[0]
        return clients[int(buffer_index) % len(clients)]

    def _register_load_staging_buffers(self) -> None:
        """Register fixed load staging buffers with the server.

        Async/thread-safety:
            Called after server transfer initialization and before request
            traffic. Registration failures are logged and leave the worker on
            the compatible per-load CUDA IPC payload path.
        """
        if getattr(self, "_load_staging_registered", False):
            return
        pool = getattr(self, "_load_staging_pool", None)
        if pool is None or not self._load_ipc_clients():
            return
        try:
            for buffer_index in range(pool.depth):
                tensor = pool.buffer(buffer_index)
                cp_tensor = cupy.asarray(tensor)
                device_ptr = cuda_array_pointer(cp_tensor)
                ipc_base_ptr, ipc_offset = _cuda_allocation_base_and_offset(device_ptr)
                load_client = self._load_ipc_client_for_buffer(buffer_index)
                self._submit_load_coroutine(
                    load_client.register_load_staging_cuda(
                        buffer_index=buffer_index,
                        cuda_ipc_handle=export_cuda_ipc_handle(cp_tensor),
                        allocation_bytes=int(tensor.numel()),
                        device_id=cuda_array_device_id(cp_tensor),
                        device_ptr=device_ptr,
                        allocation_base_ptr=ipc_base_ptr,
                        allocation_offset=ipc_offset,
                        producer_pid=os.getpid(),
                    )
                ).result(timeout=120.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[CONNECTOR] registered load staging unavailable; falling back "
                "to per-load CUDA IPC payloads: %s",
                exc,
            )
            self._load_staging_registered = False
            return
        self._load_staging_registered = True
        logger.info(
            "[CONNECTOR] registered %d fixed load staging buffers",
            pool.depth,
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

    def _wait_for_store_staging_release(self, nbytes: int) -> None:
        """Wait until one store staging lease can return to the fixed pool.

        Args:
            nbytes: Size of the staging lease requested by the caller.

        Async/thread-safety:
            Called from the worker thread when the fixed store staging pool is
            exhausted. It first applies byte-budget backpressure, then waits
            for one oldest store future if no lease was released yet.
        """
        pool = getattr(self, "_store_staging_pool", None)
        before = pool.available if pool is not None else 0
        self._wait_for_save_staging_capacity(nbytes)
        if pool is None or pool.available > before or not self._save_futures:
            return
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
        pool = getattr(self, "_store_staging_pool", None)
        if pool is None:
            max_bytes = max(
                nbytes,
                self._store_staging_bytes or DEFAULT_STORE_STAGING_BYTES,
            )
            pool = StoreCudaStagingPool(
                device=device,
                buffer_bytes=max_bytes,
                depth=_store_staging_pool_depth(
                    max_bytes,
                    self._pending_store_staging_limit_bytes
                    or DEFAULT_PENDING_STORE_STAGING_BYTES,
                ),
            )
            self._store_staging_pool = pool
        return pool.acquire(
            nbytes,
            wait_for_release=lambda: self._wait_for_store_staging_release(nbytes),
        )

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
            ``None`` once request KV has been copied into staging and background
            store/commit futures have been tracked.

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
        return None

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
        buffer_index = kwargs.pop("buffer_index", None)
        client = self._load_ipc_client_for_buffer(buffer_index)
        return await client.transfer_load_cuda(**kwargs)

    async def _transfer_load_registered_cuda(self, **kwargs: Any) -> dict[str, Any]:
        """Load through a pre-registered fixed CUDA staging buffer.

        Args:
            **kwargs: forwarded registered-buffer transfer fields.

        Returns:
            Server load response with timing counters.

        Async/thread-safety:
            Runs on the worker load event loop. The server has already opened
            the CUDA IPC mapping during initialization, so this hot-path call
            only identifies the staging buffer index and logical byte range.
        """
        buffer_index = int(kwargs.get("buffer_index", 0))
        client = self._load_ipc_client_for_buffer(buffer_index)
        return await client.transfer_load_registered_cuda(**kwargs)

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
