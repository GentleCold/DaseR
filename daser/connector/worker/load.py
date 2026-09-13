# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
import math
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
    PreparedKVRestore,
    compressed_slot_metadata,
)
from daser.ops.green_context import (
    GreenContextStream,
    create_green_context,
    parse_green_context_sm_count,
)
from daser.ops.stream_priority import cuda_stream_priority
from daser.transfer.cuda_ipc import (
    cuda_allocation_base_and_offset,
    cuda_array_device_id,
    cuda_array_pointer,
    export_cuda_ipc_handle,
)

logger = init_logger(__name__)

# CUDA event queries are cheap and non-blocking. Keep their polling cadence
# separate from IPC completion dispatch. The dispatcher default is an immediate
# yield; a positive timeout remains available as an explicit tuning knob for
# deployments that prefer batching transfer futures over first-token latency.
_LOAD_EVENT_POLL_INTERVAL_S = 0.0001
_EVENT_POLL_US_ENV = "DASER_LOAD_EVENT_POLL_US"
_GREEN_CONTEXT_ENV = "DASER_COMPRESSED_LOAD_GREEN_SM_COUNT"
_COMPRESSED_STREAM_ENV = "DASER_COMPRESSED_LOAD_STREAM"
_SEGMENT_SOURCES_ENV = "DASER_COMPRESSED_LOAD_SEGMENT_SOURCES"
_SEGMENT_DEPTH_ENV = "DASER_COMPRESSED_LOAD_SEGMENT_DEPTH"
_LAYER_PIPELINE_ENV = "DASER_LAYER_PIPELINE_LAYERS"
_LAYER_PIPELINE_DEPTH_ENV = "DASER_LAYER_PIPELINE_DEPTH"
_LAYER_PIPELINE_BURST_ENV = "DASER_LAYER_PIPELINE_BURST"
_EARLY_PACKED_FUTURE_ENV = "DASER_EARLY_PACKED_FUTURE"
_COALESCE_LIMIT_ENV = "DASER_COMPRESSED_COALESCE_MAX_REQUESTS"
_COALESCE_WINDOW_US_ENV = "DASER_COMPRESSED_COALESCE_WINDOW_US"
_BATCH_LOOKAHEAD_ENV = "DASER_LOAD_BATCH_LOOKAHEAD"
_DISPATCH_WAIT_US_ENV = "DASER_LOAD_DISPATCH_WAIT_US"
_LoadBatch = tuple[int, list[dict[str, int]], list[Any]]
_LoadSourceKey = tuple[Any, ...]


def _compressed_request_identity(
    specs: dict[str, ReqLoadSpec],
) -> tuple[Any, ...] | None:
    if not specs or any(
        not spec.compressed_slots or spec.lease_id for spec in specs.values()
    ):
        return None
    return tuple(
        (
            spec.chunk_key,
            spec.start_slot,
            spec.num_slots,
            spec.file_offset,
            spec.token_count,
            spec.target_token_start,
            spec.pos_offset,
            tuple(
                (slot.slot_id, slot.mode, slot.file_offset, slot.stored_length)
                for slot in spec.compressed_slots
            ),
        )
        for spec in specs.values()
    )


def _coalesce_compressed_requests(
    requests: list[_LoadRequest],
) -> list[_LoadRequest]:
    """Merge identical packed sources while bounding destination fanout.

    Args:
        requests: Request load plans produced for one scheduler step.

    Returns:
        Plans grouped by identical immutable packed sources.  When
        ``DASER_COMPRESSED_COALESCE_MAX_REQUESTS`` is a positive integer, a
        group is capped at that many request destinations.  A bounded group
        keeps one physical read shared while avoiding a single restore launch
        that changes the model's prefill batch geometry.

    Raises:
        ValueError: If the optional bound is not a non-negative integer.

    Async/thread-safety:
        Pure worker-side planning; it reads startup configuration and does
        not mutate request or cache state.
    """
    raw_limit = os.environ.get(_COALESCE_LIMIT_ENV, "0").strip()
    try:
        limit = int(raw_limit or "0")
    except ValueError as exc:
        raise ValueError(
            f"{_COALESCE_LIMIT_ENV} must be a non-negative integer"
        ) from exc
    if limit < 0:
        raise ValueError(f"{_COALESCE_LIMIT_ENV} must be a non-negative integer")
    coalesced: list[_LoadRequest] = []
    identities: list[tuple[Any, ...] | None] = []
    destination_blocks: list[set[int]] = []
    for request in requests:
        identity = _compressed_request_identity(request.specs)
        request_blocks = set(request.block_ids)
        matched_index: int | None = None
        if identity is not None:
            for index, existing_identity in enumerate(identities):
                if (
                    identity == existing_identity
                    and (limit == 0 or len(coalesced[index].req_ids) < limit)
                    and request_blocks.isdisjoint(destination_blocks[index])
                    and request.specs.keys().isdisjoint(coalesced[index].specs)
                ):
                    matched_index = index
                    break
        if matched_index is None:
            coalesced.append(request)
            identities.append(identity)
            destination_blocks.append(request_blocks)
            continue

        existing = coalesced[matched_index]
        merged_specs = dict(existing.specs)
        merged_specs.update(request.specs)
        coalesced[matched_index] = _LoadRequest(
            req_ids=(*existing.req_ids, *request.req_ids),
            specs=merged_specs,
            future=existing.future,
            layer_events=existing.layer_events,
        )
        destination_blocks[matched_index].update(request_blocks)
    return coalesced


def _load_source_descriptor(
    spec: ReqLoadSpec,
    slot_size: int,
) -> tuple[_LoadSourceKey, int]:
    num_slots = len(spec.block_ids)
    if spec.compressed_slots:
        if len(spec.compressed_slots) != num_slots:
            raise ValueError("compressed slot metadata does not match block IDs")
        physical = tuple(
            (slot.slot_id, slot.file_offset, slot.stored_length, slot.mode)
            for slot in spec.compressed_slots
        )
        return (
            (spec.chunk_key, spec.start_slot, spec.num_slots, physical),
            sum(slot.stored_length for slot in spec.compressed_slots),
        )
    nbytes = num_slots * slot_size
    return (
        (
            spec.chunk_key,
            spec.start_slot,
            spec.num_slots,
            spec.file_offset,
            nbytes,
        ),
        nbytes,
    )


@dataclass
class _LoadRequest:
    """Own one or more coalesced base requests and their completion future."""

    req_ids: tuple[str, ...]
    specs: dict[str, ReqLoadSpec]
    future: Future[None]
    layer_events: list[Future[Any]] | None = None

    @property
    def req_id(self) -> str:
        """Return a compact identity for diagnostics."""
        return ",".join(self.req_ids)

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
    future: asyncio.Task[dict[str, Any]] = field(init=False)
    submitted_at: float
    buffer_index: int
    response: dict[str, Any] | None = None
    wait_ms: float = 0.0
    ipc_ms: float = 0.0
    copy_ms: float = 0.0
    copies: int = 0
    copy_runs: int = 0
    restore_future: asyncio.Future[float] | None = None
    prepared_restore: PreparedKVRestore | None = None


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
    lookahead: _InflightLoadBatch | None = None


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
        self._published_layer_events: dict[str, list[Future[Any]]] = {}
        self._layer_events_lock = threading.Lock()
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
        self._compressed_cuda_stream: torch.cuda.Stream | None = None
        self._green_context: GreenContextStream | None = None
        self._green_context_sm_count = parse_green_context_sm_count(
            os.getenv(_GREEN_CONTEXT_ENV), _GREEN_CONTEXT_ENV
        )
        self._compressed_segment_sources = self._parse_positive_env(
            _SEGMENT_SOURCES_ENV
        )
        self._compressed_segment_depth = self._parse_positive_env(_SEGMENT_DEPTH_ENV)
        # Lookahead owns only one next batch; it is a switch, not a depth.
        self._batch_lookahead = self._parse_bool_env(_BATCH_LOOKAHEAD_ENV, default=True)
        self._dispatch_wait_timeout_s = self._parse_nonnegative_float_env(
            _DISPATCH_WAIT_US_ENV, scale=1e-6, default=0.0
        )
        self._event_poll_interval_s = self._parse_nonnegative_float_env(
            _EVENT_POLL_US_ENV, scale=1e-6, default=100.0
        )
        self._layer_pipeline_layers = self._parse_positive_env(_LAYER_PIPELINE_ENV)
        self._layer_pipeline_depth = self._parse_positive_env(_LAYER_PIPELINE_DEPTH_ENV)
        self._layer_pipeline_burst = self._parse_bool_env(_LAYER_PIPELINE_BURST_ENV)
        self._early_packed_future = self._parse_bool_env(_EARLY_PACKED_FUTURE_ENV)
        # The ordinary packed path completes one restore future before vLLM
        # enters the model. In that mode no per-layer event exists, so avoid
        # taking a cross-thread lock for every transformer layer. Experimental
        # layer-pipeline modes explicitly re-enable this hook.
        self._layer_wait_enabled = bool(
            self._layer_pipeline_layers or self._early_packed_future
        )
        self._coalesce_window_s = self._parse_nonnegative_float_env(
            _COALESCE_WINDOW_US_ENV, scale=1e-6
        )
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

    @property
    def layer_wait_enabled(self) -> bool:
        """Return whether the per-layer CUDA event hook is active.

        Returns:
            ``True`` only for opt-in layer-pipeline or early-future modes.

        Async/thread-safety:
            Immutable after construction and safe to read on the worker
            thread without synchronization.
        """
        return self._layer_wait_enabled

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
            storage_format: Raw, compressed-read-only or compressed-online.
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
        # One immutable source record can fan out to multiple destinations in a
        # scheduler step. Metadata capacity therefore follows the destination
        # KV cache, not the number of source records that fit in staging. The
        # extra typed rows are tiny and do not enlarge the CUDA staging pool.
        max_destination_slots = int(kv_cache.shape[0])
        self._compressed_decoder = FusedCompressedKVDecoder(
            kv_cache=kv_cache,
            codebooks=codebooks,
            tile_scalars=tile_scalars,
            ring_depth=self._staging_pool.depth,
            max_slots_per_buffer=max_destination_slots,
            online=storage_format == STORAGE_FORMAT_COMPRESSED_ONLINE,
        )
        if self._green_context_sm_count:
            self._green_context = create_green_context(
                device=kv_cache.device,
                sm_count=self._green_context_sm_count,
            )
            self._compressed_cuda_stream = self._green_context.stream
            logger.warning(
                "[GREEN_CONTEXT] compressed load enabled requested_sms=%d "
                "provisioned_sms=%d; codec work is isolated from the primary "
                "prefill stream but may have lower standalone throughput",
                self._green_context_sm_count,
                self._green_context.provisioned_sms,
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
        requests = [
            _LoadRequest(
                (req_id,),
                specs,
                Future(),
                [Future() for _ in self._layer_names]
                if self._layer_pipeline_layers or self._early_packed_future
                else None,
            )
            for req_id, specs in grouped.items()
        ]
        queued_requests = requests
        if os.environ.get("DASER_DISABLE_COMPRESSED_COALESCE", "0") == "0":
            queued_requests = _coalesce_compressed_requests(requests)
        for request in queued_requests:
            for req_id in request.req_ids:
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
            self._pending[req_id] = _LoadRequest((req_id,), specs, future)

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
                if load.layer_events is not None:
                    for event_future in load.layer_events:
                        if not event_future.done():
                            event_future.set_exception(exc)
                collected_futures.add(future_id)
            finally:
                if load.layer_events is not None:
                    for request_id in load.req_ids:
                        self._publish_layer_events(request_id, load.layer_events)
                del self._pending[req_id]
            finished.add(req_id)
        return finished

    def wait_for_layer_load(self, layer_name: str, request_ids: set[str]) -> None:
        """Enqueue model-stream waits for staged packed layer readiness.

        Args:
            layer_name: Current vLLM attention layer name.
            request_ids: Base request IDs scheduled in the current step.

        Async/thread-safety:
            Called synchronously on the vLLM worker thread. It may wait for
            the private load thread to publish a CUDA event, then only enqueues
            a dependency on the worker's current CUDA stream.
        """
        if not self._layer_wait_enabled:
            return
        try:
            layer_index = self._layer_names.index(layer_name)
        except ValueError:
            return
        for request_id in request_ids:
            with self._layer_events_lock:
                events = self._published_layer_events.get(request_id)
            if events is None or layer_index >= len(events):
                continue
            event = events[layer_index].result()
            if event is not None:
                torch.cuda.current_stream().wait_event(event)
            if layer_index == len(events) - 1:
                with self._layer_events_lock:
                    self._published_layer_events.pop(request_id, None)

    def release_layer_events(self, request_ids: set[str]) -> None:
        """Release published event references after a model step consumes them."""
        with self._layer_events_lock:
            for request_id in request_ids:
                self._published_layer_events.pop(request_id, None)

    def _publish_layer_events(self, request_id: str, events: list[Future[Any]]) -> None:
        """Publish layer CUDA events before releasing the request future.

        Args:
            request_id: Base request ID whose KV destination is being restored.
            events: Per-layer futures containing CUDA event dependencies.

        Async/thread-safety:
            Called by the load event-loop thread and read by the vLLM worker
            thread. The lock makes publication visible before the worker can
            enter the layer wait hook; event objects themselves are immutable
            after publication.
        """
        with self._layer_events_lock:
            self._published_layer_events[request_id] = events

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
        green_context = self._green_context
        self._green_context = None
        self._compressed_cuda_stream = None
        if green_context is not None:
            green_context.close()

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

    @staticmethod
    def _parse_positive_env(name: str) -> int:
        """Parse an optional positive integer used by startup-only scheduling.

        Args:
            name: Environment variable to read.

        Returns:
            The configured positive integer, or ``0`` when disabled.

        Raises:
            ValueError: If the value is not an integer greater than zero.

        Async/thread-safety:
            Pure construction-time parsing; no shared state is accessed.
        """
        value = os.environ.get(name, "").strip()
        if not value:
            return 0
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if parsed <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return parsed

    @staticmethod
    def _parse_bool_env(name: str, *, default: bool = False) -> bool:
        """Parse a boolean used by startup-only load scheduling.

        Args:
            name: Environment variable containing ``0`` or ``1``.
            default: Value to use when the variable is unset.

        Returns:
            The default when unset; otherwise ``True`` only for ``1``.
            An empty value disables the setting.

        Raises:
            ValueError: If a non-empty value is neither ``0`` nor ``1``.

        Async/thread-safety:
            Startup-only parsing; the result is immutable for the pipeline.
        """
        value = os.environ.get(name, "1" if default else "0").strip()
        if not value:
            return False
        if value not in {"0", "1"}:
            raise ValueError(f"{name} must be 0 or 1")
        return value == "1"

    @staticmethod
    def _parse_nonnegative_float_env(
        name: str, *, scale: float = 1.0, default: float = 0.0
    ) -> float:
        """Parse an optional non-negative numeric startup setting.

        Args:
            name: Environment variable containing a decimal number.
            scale: Multiplier converting the external unit to seconds.

        Returns:
            The scaled value, or the scaled default when unset.

        Raises:
            ValueError: If the value is not finite or non-negative.

        Async/thread-safety:
            Startup-only parsing; no event-loop state is accessed.
        """
        value = os.environ.get(name, str(default)).strip()
        if not value:
            return 0.0
        try:
            parsed = float(value)
        except ValueError as exc:
            raise ValueError(f"{name} must be a non-negative number") from exc
        if parsed < 0 or not math.isfinite(parsed):
            raise ValueError(f"{name} must be a non-negative number")
        return parsed * scale

    @staticmethod
    def _compressed_stream_mode() -> str:
        """Return the packed restore stream policy selected at startup.

        Returns:
            ``"high"`` when unset. Explicit ``"default"`` uses the model
            stream, while ``"separate"`` uses an independent non-blocking
            stream. ``"low"`` and ``"high"`` use
            the least-urgent and most-urgent supported priorities respectively.
            High priority lets short restores avoid waiting behind queued
            model work without changing the completion barrier.

        Raises:
            ValueError: If the environment value is unsupported.

        Async/thread-safety:
            Pure startup configuration; the value is read before requests
            are dispatched and is immutable for the pipeline lifetime.
        """
        mode = os.environ.get(_COMPRESSED_STREAM_ENV, "high").strip().lower()
        if mode not in {"default", "separate", "low", "high"}:
            raise ValueError(
                f"{_COMPRESSED_STREAM_ENV} must be 'default', 'separate', "
                "'low', or 'high'"
            )
        return mode

    @staticmethod
    def _compressed_stream_priority(mode: str) -> int | None:
        """Return the CUDA priority for a packed restore stream.

        Args:
            mode: Startup stream mode returned by :meth:`_compressed_stream_mode`.

        Returns:
            ``None`` for the default CUDA priority, otherwise the requested
            endpoint of the device's supported priority range.

        Async/thread-safety:
            Pure CUDA device query performed on the load event-loop thread
            before the stream is created.
        """
        if mode not in {"low", "high"}:
            return None
        return cuda_stream_priority(mode)

    async def _run_dispatcher(self) -> None:
        sample_tensor = next(iter(self._kv_caches.values()))
        if sample_tensor.device.type == "cuda":
            torch.cuda.set_device(sample_tensor.device)
            if self._cuda_stream is None:
                # CUDA current-stream state is thread-local.  Reuse the stream
                # captured by the worker during KV registration when
                # available; discovering it here would otherwise select the
                # private load thread's stream and introduce implicit
                # default-stream synchronization with model kernels.
                self._cuda_stream = torch.cuda.current_stream(sample_tensor.device)
            if (
                self._compressed_decoder is not None
                and self._compressed_cuda_stream is None
                and (
                    self._compressed_stream_mode() in {"separate", "low", "high"}
                    or self._layer_pipeline_layers > 0
                )
            ):
                # The request completion remains gated by the event recorded
                # on this stream, so vLLM never consumes a partially restored
                # KV block.  A separate stream lets independent requests make
                # progress while packed decode is queued behind another
                # request's model work. Explicit stream mode "default" keeps
                # the historical single-stream ordering.
                stream_mode = self._compressed_stream_mode()
                priority = self._compressed_stream_priority(stream_mode)
                stream_kwargs: dict[str, Any] = {"device": sample_tensor.device}
                if priority is not None:
                    stream_kwargs["priority"] = priority
                self._compressed_cuda_stream = torch.cuda.Stream(**stream_kwargs)
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
                if (
                    queued
                    and free_buffers
                    and len(queued) == 1
                    and self._coalesce_window_s > 0.0
                ):
                    # A short admission window gives requests submitted by
                    # adjacent scheduler steps a chance to join the same
                    # immutable packed-source fanout.  It is opt-in because
                    # waiting here trades one host scheduling delay for fewer
                    # physical reads and decoder submissions.
                    await asyncio.sleep(self._coalesce_window_s)
                    await self._drain_queue(queue, queued)
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
                        reusable_buffer, request_done = self._consume_request(
                            state, free_buffers
                        )
                    except BaseException as exc:
                        # Consumption can promote lookahead before a later
                        # submission fails. Drain the state's current owners,
                        # not the stale active_batch from before promotion.
                        # Publishing failure lets vLLM invalidate/reuse KV
                        # blocks, so it must follow every sibling's drain.
                        released_indices = []
                        for batch in (state.active, state.lookahead):
                            if batch is not None:
                                await self._drain_batch_on_failure(batch)
                                batch.staging_lease.release()
                                released_indices.append(batch.buffer_index)
                        if not state.request.future.done():
                            state.request.future.set_exception(exc)
                        active.remove(state)
                        for released_index in released_indices:
                            if released_index not in free_buffers:
                                free_buffers.append(released_index)
                        consumed = True
                        continue
                    if request_done:
                        active.remove(state)
                        if reusable_buffer is not None:
                            free_buffers.append(reusable_buffer)
                    elif reusable_buffer is not None:
                        free_buffers.append(reusable_buffer)
                    if not request_done:
                        self._fill_lookahead(active, free_buffers)
                    consumed = True
                if consumed:
                    continue
                if active:
                    await self._wait_for_completion(active)
        except BaseException as exc:
            for state in active:
                for batch in (state.active, state.lookahead):
                    if batch is not None:
                        await self._drain_batch_on_failure(batch)
                        batch.staging_lease.release()
                if not state.request.future.done():
                    state.request.future.set_exception(exc)
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
        completions: set[asyncio.Future[Any]] = set()
        for request in active:
            batch = request.active
            if batch is None:
                continue
            completion = (
                batch.restore_future
                if batch.restore_future is not None
                else batch.future
            )
            completions.add(completion)
        if not completions:
            await asyncio.sleep(0)
            return
        done, _pending = await asyncio.wait(
            completions,
            timeout=getattr(self, "_dispatch_wait_timeout_s", 0.0),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for future in done:
            future.exception()
        if not done:
            await asyncio.sleep(0)

    async def _wait_for_cuda_event(
        self,
        event: Any,
        buffer_index: int = 0,
        deferred_token: str | None = None,
    ) -> float:
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
            await asyncio.sleep(
                getattr(self, "_event_poll_interval_s", _LOAD_EVENT_POLL_INTERVAL_S)
            )
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
        state = _InflightLoadBatch(
            total_bytes=total_bytes,
            per_req_ranges=per_req_ranges,
            staging_lease=lease,
            submitted_at=time.perf_counter(),
            buffer_index=buffer_index,
        )
        # The dispatcher must know the lease before metadata preparation or
        # CUDA IPC export can fail. Owning those operations in the batch Task
        # gives their partial GPU work the same failure drain as the transfer.
        state.future = asyncio.create_task(self._transfer_batch(state, spans, lease_id))
        return state

    async def _transfer_batch(
        self,
        state: _InflightLoadBatch,
        spans: list[dict[str, int]],
        lease_id: str | None,
    ) -> dict[str, Any]:
        """Prepare metadata and await IO under an already-owned staging lease."""
        staging = state.staging_lease.view
        per_req_ranges = state.per_req_ranges
        buffer_index = state.buffer_index
        if (
            self._compressed_decoder is not None
            and self._compressed_cuda_stream is not None
            and per_req_ranges
            and all(
                (item[3] if len(item) == 4 else item[2]).compressed_slots
                for item in per_req_ranges
            )
        ):
            offsets, block_ids, modes = compressed_slot_metadata(per_req_ranges)
            state.prepared_restore = self._compressed_decoder.prepare(
                staging=staging,
                staging_offsets=offsets,
                block_ids=block_ids,
                modes=modes,
                buffer_index=buffer_index,
                stream=self._compressed_cuda_stream,
            )
        if self._staging_registered:
            transfer = self._client(buffer_index).transfer_load_registered_cuda(
                buffer_index=buffer_index,
                producer_pid=os.getpid(),
                nbytes=state.total_bytes,
                spans=spans,
                lease_id=lease_id,
                defer_copy=(
                    os.environ.get("DASER_DEFER_L1_COPY", "0") != "0"
                    and lease_id is None
                    and bool(per_req_ranges)
                    and all(
                        (item[3] if len(item) == 4 else item[2]).compressed_slots
                        for item in per_req_ranges
                    )
                ),
            )
        else:
            if staging.device.type == "cuda":
                # The dispatcher owns its event-loop thread; CUDA current
                # device is thread-local and must match the staging lease
                # before exporting a per-load IPC handle.
                torch.cuda.set_device(staging.device)
            cp_staging = cupy.asarray(staging)
            device_ptr = cuda_array_pointer(cp_staging)
            allocation_base, allocation_offset = cuda_allocation_base_and_offset(
                device_ptr
            )
            transfer = self._client(buffer_index).transfer_load_cuda(
                cuda_ipc_handle=export_cuda_ipc_handle(cp_staging),
                nbytes=state.total_bytes,
                device_id=cuda_array_device_id(cp_staging),
                device_ptr=device_ptr,
                allocation_base_ptr=allocation_base,
                allocation_offset=allocation_offset,
                producer_pid=os.getpid(),
                spans=spans,
                lease_id=lease_id,
            )
        return await transfer

    def _consume_request(
        self,
        state: _InflightRequestLoad,
        free_buffers: deque[int],
    ) -> tuple[int | None, bool]:
        active = state.active
        if active is None:
            return state.buffer_index, True
        if active.restore_future is None:
            layer_request = (
                state.request
                if (
                    not state.batches
                    and state.request.layer_events is not None
                    and (self._layer_pipeline_layers > 0 or self._early_packed_future)
                )
                else None
            )
            self._start_restore(active, layer_request)
            self._fill_request_lookahead(state, free_buffers)
            return None, False
        state.completed.append(self._finish_batch(active))
        released_buffer = state.buffer_index
        if state.lookahead is not None:
            state.active = state.lookahead
            state.buffer_index = state.lookahead.buffer_index
            state.lookahead = None
            # The completed batch's staging buffer is now safe to reuse for a
            # second lookahead transfer.  The restore event has already been
            # observed by the dispatcher, so no CUDA user can still read it.
            if state.batches:
                batch = state.batches.popleft()
                try:
                    state.lookahead = self._submit_batch(
                        state.request.lease_id,
                        batch,
                        released_buffer,
                    )
                except BaseException:
                    state.batches.appendleft(batch)
                    free_buffers.appendleft(released_buffer)
                    raise
                return None, False
            return released_buffer, False
        if not state.batches:
            self._log_request_timing(state)
            if not state.request.future.done():
                state.request.future.set_result(None)
            return released_buffer, True
        state.active = self._submit_batch(
            state.request.lease_id,
            state.batches.popleft(),
            released_buffer,
        )
        state.buffer_index = released_buffer
        return None, False

    def _fill_lookahead(
        self,
        active: list[_InflightRequestLoad],
        free_buffers: deque[int],
    ) -> None:
        """Start one transfer lookahead for each eligible request.

        Args:
            active: Requests currently owned by the dispatcher.
            free_buffers: Staging indices not held by an active transfer or
                restore. Indices are removed when a lookahead lease is made.

        Async/thread-safety:
            Called only on the load event-loop thread. A lookahead owns its
            staging lease until its own restore event completes, so it cannot
            overwrite bytes consumed by the current batch.
        """
        if self._batch_lookahead <= 0:
            return
        for state in active:
            if not free_buffers:
                return
            self._fill_request_lookahead(state, free_buffers)

    def _fill_request_lookahead(
        self,
        state: _InflightRequestLoad,
        free_buffers: deque[int],
    ) -> None:
        """Submit one next-batch transfer while the current restore runs."""
        if (
            self._batch_lookahead <= 0
            or state.lookahead is not None
            or not state.batches
            or not free_buffers
        ):
            return
        lookahead_buffer = free_buffers.popleft()
        batch = state.batches.popleft()
        try:
            state.lookahead = self._submit_batch(
                state.request.lease_id,
                batch,
                lookahead_buffer,
            )
        except BaseException:
            state.batches.appendleft(batch)
            free_buffers.appendleft(lookahead_buffer)
            raise

    async def _drain_batch_on_failure(self, state: _InflightLoadBatch) -> None:
        """Drain one speculative batch before returning its staging lease.

        Args:
            state: Batch that may still have a server copy or CUDA restore in
                flight after a sibling batch failed.

        Async/thread-safety:
            Runs on the load event-loop thread. Awaiting the transfer and
            restore futures preserves the IPC/CUDA lifetime contract before a
            failed request releases its speculative staging buffer.
        """
        pending: list[asyncio.Future[Any]] = [state.future]
        if state.restore_future is not None:
            pending.append(state.restore_future)
        drain = asyncio.gather(*pending, return_exceptions=True)
        while not drain.done():
            try:
                await asyncio.shield(drain)
            except asyncio.CancelledError:
                # Cancellation cannot return a buffer still written by the
                # server. Keep draining even after repeated cancellation.
                continue
        stream = self._compressed_cuda_stream or self._cuda_stream
        if stream is not None:
            # Prepare/restore may have submitted GPU work before raising and
            # recording its normal event. Only this error path needs a full
            # stream barrier, offloaded so sibling async transfers can finish.
            gpu_drain = asyncio.create_task(asyncio.to_thread(stream.synchronize))
            while not gpu_drain.done():
                try:
                    await asyncio.shield(gpu_drain)
                except asyncio.CancelledError:
                    continue
            gpu_drain.result()

    def _start_restore(
        self,
        state: _InflightLoadBatch,
        layer_request: _LoadRequest | None = None,
    ) -> None:
        """Launch one restore and defer lease reuse until its CUDA event.

        Args:
            state: Transfer-complete batch whose staging bytes are ready.

        Async/thread-safety:
            Called on the private load event loop. The restore is enqueued on
            the pipeline CUDA stream; no host stream synchronization is used.
        """
        wait_start = time.perf_counter()
        response = state.future.result()
        state.wait_ms = (time.perf_counter() - wait_start) * 1000
        state.ipc_ms = (time.perf_counter() - state.submitted_at) * 1000
        self._wait_for_remote_copy(state, response)
        copy_start = time.perf_counter()
        layered = (
            self._start_layer_pipeline_restore(state, layer_request)
            if layer_request is not None
            else None
        )
        segmented = (
            None if layered is not None else self._start_segmented_restore(state)
        )
        event: torch.cuda.Event | None
        if layered is not None:
            state.copies, state.copy_runs, event, restore_task = layered
        elif segmented is None:
            state.copies, state.copy_runs, event = self._restore_batch(state)
        else:
            state.copies, state.copy_runs, _restore_task = segmented
            event = None
        state.copy_ms = (time.perf_counter() - copy_start) * 1000
        if layered is not None:
            state.restore_future = restore_task
        elif segmented is not None:
            state.restore_future = segmented[2]
        elif event is None:
            restore_future: asyncio.Future[float] = self._loop.create_future()
            restore_future.set_result(0.0)
            state.restore_future = restore_future
        else:
            state.restore_future = asyncio.create_task(
                self._wait_for_cuda_event(
                    event,
                    state.buffer_index,
                    None,
                )
            )
            if (
                self._early_packed_future
                and layer_request is not None
                and self._is_compressed_batch(state)
            ):
                # Publish one event for every layer as soon as the complete
                # decode launch is queued.  The request future can then join
                # the scheduler's next batch while the model stream inserts
                # an explicit wait before its first attention layer.  Staging
                # ownership remains with ``restore_future`` until the CUDA
                # event is observed, so early publication cannot expose
                # partially restored bytes or recycle the ring early.
                for event_future in layer_request.layer_events or ():
                    if not event_future.done():
                        event_future.set_result(event)
                for request_id in layer_request.req_ids:
                    self._publish_layer_events(
                        request_id, layer_request.layer_events or []
                    )
                if not layer_request.future.done():
                    layer_request.future.set_result(None)

        state.response = response if isinstance(response, dict) else {}

    def _start_layer_pipeline_restore(
        self,
        state: _InflightLoadBatch,
        request: _LoadRequest,
    ) -> tuple[int, int, torch.cuda.Event, asyncio.Task[float]] | None:
        """Decode packed KV by layer groups and publish CUDA readiness events."""
        if not self._is_compressed_batch(state):
            return None
        decoder = self._compressed_decoder
        stream = self._compressed_cuda_stream
        if decoder is None or stream is None or request.layer_events is None:
            return None
        plan = state.prepared_restore
        if plan is None:
            offsets, block_ids, modes = compressed_slot_metadata(state.per_req_ranges)
            plan = decoder.prepare(
                staging=state.staging_lease.view,
                staging_offsets=offsets,
                block_ids=block_ids,
                modes=modes,
                buffer_index=state.buffer_index,
                stream=stream,
            )
        if plan is None:
            return None
        plane_count = len(self._layer_names) * 2
        group = max(2, self._layer_pipeline_layers * 2)
        # A zero depth preserves the original layer prototype: all layer
        # groups are submitted immediately.  A positive depth is a bounded
        # lookahead window.  It limits GPU queueing without changing the
        # physical read span or staging allocation.
        depth = self._layer_pipeline_depth or (plane_count + group - 1) // group
        pending: list[tuple[torch.cuda.Event, int, int]] = []
        begin = 0
        while begin < plane_count and len(pending) < depth:
            end = min(begin + group, plane_count)
            plan.submit_plane_range(begin, end)
            event = torch.cuda.Event(blocking=False)
            event.record(stream)
            pending.append((event, begin, end))
            self._publish_layer_group_events(request.layer_events, event, begin, end)
            begin = end
        initial_group_count = len(pending)
        if begin == plane_count:
            plan.finish_plane_ranges()
        final_event = pending[-1][0]
        task = asyncio.create_task(
            self._finish_layer_pipeline_restore(
                plan,
                pending,
                begin,
                plane_count,
                group,
                stream,
                request.layer_events,
            )
        )
        for request_id in request.req_ids:
            self._publish_layer_events(request_id, request.layer_events)
        if not request.future.done():
            # The worker may schedule the request as soon as every layer has a
            # CUDA event dependency. The final task retains staging ownership.
            request.future.set_result(None)
        return plan.destination_count, initial_group_count, final_event, task

    @staticmethod
    def _publish_layer_group_events(
        layer_events: list[Future[Any]],
        event: torch.cuda.Event,
        plane_begin: int,
        plane_end: int,
    ) -> None:
        """Resolve layer wait futures for one submitted K/V group.

        Args:
            layer_events: Per-layer futures consumed by the vLLM wait hook.
            event: CUDA event recorded after the group's decode launches.
            plane_begin: Inclusive K/V plane index.
            plane_end: Exclusive K/V plane index.

        Async/thread-safety:
            Called by the private load thread.  Futures are published before
            the request can be scheduled; the worker only enqueues
            ``wait_event`` and never synchronizes the host for a resolved
            event.
        """
        for layer_index in range(
            plane_begin // 2, min(plane_end // 2, len(layer_events))
        ):
            if not layer_events[layer_index].done():
                layer_events[layer_index].set_result(event)

    async def _finish_layer_pipeline_restore(
        self,
        plan: PreparedKVRestore,
        pending: list[tuple[torch.cuda.Event, int, int]],
        next_begin: int,
        plane_count: int,
        group: int,
        stream: torch.cuda.Stream,
        layer_events: list[Future[Any]],
    ) -> float:
        """Drain layer groups and submit the next group after each CUDA event.

        Args:
            plan: Prepared immutable restore metadata.
            pending: Initially submitted groups and their completion events.
            next_begin: First plane not submitted by the initial lookahead.
            plane_count: Total K/V planes in the cross-layer cache.
            group: Number of planes per layer group.
            stream: Stream receiving subsequent decode launches.
            layer_events: Futures used by the per-layer vLLM wait hook.

        Returns:
            Milliseconds spent polling group completion events.

        Async/thread-safety:
            Runs on the private load loop.  CUDA queries are non-blocking;
            the staging lease remains owned by the parent batch until this
            coroutine has submitted and observed every group.
        """
        wait_ms = 0.0
        try:
            while pending:
                event, _begin, _end = pending.pop(0)
                wait_ms += await self._wait_for_cuda_event(event)
                if next_begin >= plane_count:
                    continue
                if self._layer_pipeline_burst:
                    # The first group is the dependency that gates model
                    # entry. Once it is ready, queue the remaining groups in
                    # one burst so later layer waits overlap model execution
                    # instead of serializing one launch per completion poll.
                    while next_begin < plane_count:
                        next_end = min(next_begin + group, plane_count)
                        plan.submit_plane_range(next_begin, next_end)
                        next_event = torch.cuda.Event(blocking=False)
                        next_event.record(stream)
                        pending.append((next_event, next_begin, next_end))
                        self._publish_layer_group_events(
                            layer_events, next_event, next_begin, next_end
                        )
                        next_begin = next_end
                    continue
                next_end = min(next_begin + group, plane_count)
                plan.submit_plane_range(next_begin, next_end)
                next_event = torch.cuda.Event(blocking=False)
                next_event.record(stream)
                pending.append((next_event, next_begin, next_end))
                self._publish_layer_group_events(
                    layer_events, next_event, next_begin, next_end
                )
                next_begin = next_end
            plan.finish_plane_ranges()
            return wait_ms
        except BaseException as exc:
            for event_future in layer_events:
                if not event_future.done():
                    event_future.set_exception(exc)
            raise

    def _start_segmented_restore(
        self,
        state: _InflightLoadBatch,
    ) -> tuple[int, int, asyncio.Task[float]] | None:
        """Submit a bounded prefix of compressed decode work.

        The normal path enqueues one decoder launch containing every source in
        a completed transfer.  When both startup-only segment knobs are set,
        this method submits only ``depth`` segments and hands the remaining
        launches to a coroutine that advances after CUDA completion events.
        The staging lease remains owned by the parent batch for the entire
        coroutine, so limiting GPU queue depth never weakens lifetime safety.

        Args:
            state: Transfer-complete load batch with a prepared decoder plan.

        Returns:
            ``(destination_count, segment_count, task)`` for the experiment,
            or ``None`` when disabled or when the batch is raw.

        Async/thread-safety:
            Called on the private load event-loop thread. CUDA launches are
            asynchronous; the returned task performs non-blocking event polls.
        """
        segment = self._compressed_segment_sources
        depth = self._compressed_segment_depth
        if segment <= 0 or depth <= 0 or not self._is_compressed_batch(state):
            return None
        decoder = self._compressed_decoder
        restore_stream = self._compressed_cuda_stream or self._cuda_stream
        if decoder is None or restore_stream is None:
            raise RuntimeError("compressed decoder is not configured")
        plan = state.prepared_restore
        if plan is None:
            offsets, block_ids, modes = compressed_slot_metadata(state.per_req_ranges)
            plan = decoder.prepare(
                staging=state.staging_lease.view,
                staging_offsets=offsets,
                block_ids=block_ids,
                modes=modes,
                buffer_index=state.buffer_index,
                stream=restore_stream,
            )
        if plan is None:
            return None
        events: list[Any] = []
        launches = 0
        while launches < depth and plan.remaining_sources:
            plan.submit_next(segment)
            launches += 1
            event = torch.cuda.Event(blocking=False)
            event.record(restore_stream)
            events.append(event)
        task = asyncio.create_task(
            self._finish_segmented_restore(plan, events, segment, depth, restore_stream)
        )
        return plan.destination_count, len(events), task

    async def _finish_segmented_restore(
        self,
        plan: PreparedKVRestore,
        events: list[Any],
        segment: int,
        depth: int,
        stream: torch.cuda.Stream,
    ) -> float:
        """Advance a bounded decoder plan after each CUDA segment event.

        Args:
            plan: Prepared immutable restore metadata.
            events: Completion events for currently queued segments.
            segment: Maximum source records per launch.
            depth: Maximum number of launches kept in flight.
            stream: Stream receiving subsequent decoder launches.

        Returns:
            Total host wait time spent observing segment completion.

        Async/thread-safety:
            Runs on the load event loop and uses only non-blocking CUDA event
            queries. The caller retains the staging lease until this task ends.
        """
        del depth  # The event list itself is the bounded queue state.
        wait_ms = 0.0
        while events:
            event = events.pop(0)
            wait_ms += await self._wait_for_cuda_event(event)
            if plan.remaining_sources:
                plan.submit_next(segment)
                next_event = torch.cuda.Event(blocking=False)
                next_event.record(stream)
                events.append(next_event)
        return wait_ms

    @staticmethod
    def _is_compressed_batch(state: _InflightLoadBatch) -> bool:
        """Return whether every range in a batch uses packed metadata."""
        return bool(state.per_req_ranges) and all(
            (item[2] if len(item) == 3 else item[3]).compressed_slots
            for item in state.per_req_ranges
        )

    def _wait_for_remote_copy(
        self,
        state: _InflightLoadBatch,
        response: Any,
    ) -> None:
        """Join a server-side H2D copy through a CUDA IPC event.

        Args:
            state: Transfer batch whose staging destination was filled by the
                server process.
            response: IPC response carrying an optional event handle.

        Async/thread-safety:
            Runs on the private load thread. ``wait_event`` only enqueues a
            dependency on the restore stream and never synchronizes the host.
        """
        if not isinstance(response, dict):
            return
        raw_handle = response.get("cuda_event_ipc_handle")
        if raw_handle is None:
            return
        restore_stream = self._compressed_cuda_stream or self._cuda_stream
        if restore_stream is None or self._staging_pool is None:
            raise RuntimeError("CUDA event returned without a restore stream")
        handle = bytes(raw_handle)
        device = state.staging_lease.view.device
        event = torch.cuda.Event.from_ipc_handle(device, handle)
        restore_stream.wait_event(event)

    def _finish_batch(self, state: _InflightLoadBatch) -> _LoadBatchTiming:
        """Finalize timing and release staging after restore completion."""
        try:
            assert state.restore_future is not None
            restore_wait_ms = float(state.restore_future.result())
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
            restore_stream = self._compressed_cuda_stream or self._cuda_stream
            if decoder is None or restore_stream is None:
                raise RuntimeError("compressed decoder is not configured")
            plan = state.prepared_restore
            if plan is None:
                offsets, block_ids, modes = compressed_slot_metadata(
                    state.per_req_ranges
                )
                plan = decoder.prepare(
                    staging=staging,
                    staging_offsets=offsets,
                    block_ids=block_ids,
                    modes=modes,
                    buffer_index=state.buffer_index,
                    stream=restore_stream,
                )
            restored = 0 if plan is None else plan.submit_next(plan.remaining_sources)
            event = torch.cuda.Event(blocking=False)
            event.record(restore_stream)
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
                    self._client(buffer_index).register_staging_cuda(
                        direction="load",
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
        source_key, nbytes = _load_source_descriptor(spec, slot_size)
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
        Read plans in first-source order. Sources larger than the cap are
        split on slot boundaries, with every alias of a fragment kept in the
        same plan so reducing staging capacity does not duplicate payload IO.

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
    source_groups: dict[_LoadSourceKey, list[tuple[str, ReqLoadSpec]]] = {}
    for req_id, spec in reqs_to_load.items():
        if spec.block_ids:
            source_key, _source_bytes = _load_source_descriptor(spec, slot_size)
            source_groups.setdefault(source_key, []).append((req_id, spec))

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

    for aliases in source_groups.values():
        _first_id, source = aliases[0]
        cursor = 0
        while cursor < len(source.block_ids):
            take = 0
            part_bytes = 0
            while cursor + take < len(source.block_ids):
                slot_bytes = (
                    source.compressed_slots[cursor + take].stored_length
                    if source.compressed_slots
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
            # Place all destinations beside their shared source fragment.
            # Walking an entire alias first would flush its earlier fragments
            # before the next alias arrives, forcing identical bytes through
            # SSD/H2D repeatedly whenever a source exceeds one staging lease.
            for req_id, spec in aliases:
                compressed_slots = spec.compressed_slots[cursor : cursor + take]
                batch_spec = replace(
                    spec,
                    start_slot=spec.start_slot + cursor,
                    num_slots=take,
                    block_ids=spec.block_ids[cursor : cursor + take],
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
