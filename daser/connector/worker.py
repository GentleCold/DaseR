# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

# Third Party
import cupy
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    tensorboard_trace_handler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

if TYPE_CHECKING:
    # Third Party
    from vllm.attention import AttentionMetadata
    from vllm.forward_context import ForwardContext

# First Party
from daser.connector.gds_transfer import GDSTransferLayer
from daser.connector.metadata import (
    DaserConnectorMeta,
    ReqLoadSpec,
    ReqStoreSpec,
    StoreFuture,
    StoreWriteSpan,
)
from daser.logging import init_logger

logger = init_logger(__name__)

DEFAULT_ROPE_DELTA_SCALE = 1.0

_DASER_LOAD_PROFILE_TAGS = (
    "daser::alloc_staging",
    "daser::gds_read",
    "daser::staging_to_kv",
    "daser::index_copy",
    "daser::kv_scale",
    "daser::rope_delta",
)


def _event_cuda_us(event: Any) -> float:
    """Return CUDA/device time in microseconds for a profiler average event."""
    return float(
        getattr(event, "cuda_time_total", 0)
        or getattr(event, "device_time_total", 0)
        or getattr(event, "self_cuda_time_total", 0)
        or getattr(event, "self_device_time_total", 0)
    )


def _mark_wall_segment(
    segments: dict[str, float] | None, name: str, start: float
) -> float:
    """Record a wall-clock segment ending now and return a new start time.

    Args:
        segments: optional dict to populate; ignored when ``None``.
        name: segment label.
        start: ``time.perf_counter()`` at segment start.

    Returns:
        Fresh ``time.perf_counter()`` for chaining.
    """
    if segments is None:
        return time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    segments[name] = (time.perf_counter() - start) * 1000.0
    return time.perf_counter()


def _tensorboard_trace_handler(
    log_dir: str, wall_segments: dict[str, float]
) -> Any:
    """Build a TensorBoard trace handler that skips empty load profiles.

    Args:
        log_dir: directory passed to ``tensorboard_trace_handler``.
        wall_segments: wall-clock segment map populated during load.

    Returns:
        Callable suitable for ``profile(on_trace_ready=...)``.
    """
    base_handler = tensorboard_trace_handler(log_dir, worker_name="daser_load_kv")

    def _handler(prof: torch.profiler.profile) -> None:
        if wall_segments.get("daser::gds_read", 0.0) <= 0.0:
            return
        base_handler(prof)

    return _handler


def _format_load_profile_report(
    prof: torch.profiler.profile,
    wall_segments: dict[str, float] | None = None,
) -> str:
    """Build a segment timing table from a PyTorch profiler run.

    Args:
        prof: profiler instance after ``start_load_kv`` completes.
        wall_segments: optional wall-clock milliseconds collected during load.

    Returns:
        Human-readable report with per-segment CUDA/CPU microseconds.
    """
    events = prof.key_averages()
    lines = [
        "PyTorch profiler (device_time + cpu_time; gds_read is often CPU-only)",
        "segment          cuda_ms  cpu_ms  count",
        "-------------------------------------------",
    ]
    total_cuda_us = 0.0
    for tag in _DASER_LOAD_PROFILE_TAGS:
        matched = [event for event in events if event.key == tag]
        if not matched:
            continue
        cuda_us = sum(_event_cuda_us(event) for event in matched)
        cpu_us = sum(float(event.cpu_time_total) for event in matched)
        count = sum(event.count for event in matched)
        total_cuda_us += cuda_us
        lines.append(
            f"{tag:16} {cuda_us / 1000.0:8.3f} {cpu_us / 1000.0:7.3f} {count:5d}"
        )
    lines.append("-------------------------------------------")
    lines.append(f"{'TOTAL (segments)':16} {total_cuda_us / 1000.0:8.3f}")
    if wall_segments:
        lines.append("")
        lines.append("Wall clock (cuda synchronized where applicable):")
        lines.append("segment          wall_ms")
        lines.append("-------------------------------------------")
        wall_total = 0.0
        for tag in _DASER_LOAD_PROFILE_TAGS:
            if tag in wall_segments:
                lines.append(f"{tag:16} {wall_segments[tag]:8.3f}")
                wall_total += wall_segments[tag]
        if "load_total_wall_ms" in wall_segments:
            lines.append("-------------------------------------------")
            lines.append(
                f"{'load_total':16} {wall_segments['load_total_wall_ms']:8.3f}"
            )
        elif wall_total > 0:
            lines.append("-------------------------------------------")
            lines.append(f"{'wall_sum':16} {wall_total:8.3f}")
    lines.append("")
    lines.append("Top CUDA kernels (all ops):")
    lines.append(
        prof.key_averages().table(sort_by="device_time_total", row_limit=12)
    )
    return "\n".join(lines)


def _apply_rope_delta_to_key_block(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Rotate an already-RoPE'd K block by a relative position delta.

    Args:
        key_block: K cache block with shape [block_tokens, heads, head_dim].
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.
    """
    if delta == 0 or rotary_dim <= 0:
        return
    if key_block.shape[-1] < rotary_dim:
        return

    rot = key_block[..., :rotary_dim]
    compute = rot.float()
    device = key_block.device
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    freqs = delta * inv_freq
    cos = freqs.cos().view(*([1] * (compute.dim() - 1)), -1)
    sin = freqs.sin().view(*([1] * (compute.dim() - 1)), -1)

    if is_neox_style:
        x1, x2 = torch.chunk(compute, 2, dim=-1)
        rotated = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    else:
        x1 = compute[..., ::2]
        x2 = compute[..., 1::2]
        rotated = torch.stack((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
        rotated = rotated.flatten(-2)

    rot.copy_(rotated.to(key_block.dtype))


def _copy_staging_to_kv_cache(
    staging: torch.Tensor,
    kv_caches: dict[str, torch.Tensor],
    layer_names: list[str],
    block_ids: list[int],
    slot_size: int,
    load_key_scale: float = 1.0,
    load_value_scale: float = 1.0,
    pos_offset: int = 0,
    rope_delta_scale: float = DEFAULT_ROPE_DELTA_SCALE,
    rope_base: float = 10000.0,
    rope_rotary_dim: int = 0,
    rope_is_neox_style: bool = True,
) -> int:
    """Copy slot-major staging bytes into vLLM KV cache blocks.

    Args:
        staging: Contiguous uint8 tensor containing whole request KV bytes.
        kv_caches: Per-layer vLLM KV cache tensors.
        layer_names: Layer iteration order matching on-disk layout.
        block_ids: vLLM KV block IDs corresponding to staging slots.
        slot_size: Total bytes for all layers in one slot.
        load_key_scale: Optional multiplier for loaded K tensors.
        load_value_scale: Optional multiplier for loaded V tensors.
        pos_offset: Position delta for loaded chunk reuse.
        rope_delta_scale: Multiplier applied to pos_offset before RoPE update.
        rope_base: RoPE theta/base.
        rope_rotary_dim: Number of head dimensions covered by RoPE.
        rope_is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        Number of layer-level copy operations issued.

    Async/thread-safety:
        Synchronous GPU tensor copies on the vLLM worker thread.
    """
    if not block_ids or not layer_names:
        return 0
    num_layers = len(layer_names)
    layer_size = slot_size // num_layers
    num_slots = len(block_ids)
    staging_by_layer = staging.view(num_slots, num_layers, layer_size)
    block_index = torch.tensor(block_ids, dtype=torch.long, device=staging.device)

    copies = 0
    for layer_idx, layer_name in enumerate(layer_names):
        kv_tensor = kv_caches.get(layer_name)
        if kv_tensor is None:
            continue
        if kv_tensor.dim() >= 2:
            sample = kv_tensor[:, block_ids[0]]
            src = (
                staging_by_layer[:, layer_idx, :]
                .view(kv_tensor.dtype)
                .view(num_slots, *sample.shape)
            )
            with record_function("daser::index_copy"):
                kv_tensor.index_copy_(1, block_index, src.movedim(0, 1))
        else:
            sample = kv_tensor[block_ids[0]]
            src = (
                staging_by_layer[:, layer_idx, :]
                .view(kv_tensor.dtype)
                .view(num_slots, *sample.shape)
            )
            with record_function("daser::index_copy"):
                kv_tensor.index_copy_(0, block_index, src)
        if (
            load_key_scale != 1.0
            or load_value_scale != 1.0
            or (pos_offset and rope_rotary_dim > 0)
        ):
            for block_id in block_ids:
                dst = (
                    kv_tensor[:, block_id]
                    if kv_tensor.dim() >= 2
                    else kv_tensor[block_id]
                )
                if dst.dim() > 0 and dst.shape[0] >= 2:
                    if load_key_scale != 1.0 or load_value_scale != 1.0:
                        with record_function("daser::kv_scale"):
                            if load_key_scale != 1.0:
                                dst[0].mul_(load_key_scale)
                            if load_value_scale != 1.0:
                                dst[1].mul_(load_value_scale)
                if (
                    pos_offset
                    and kv_tensor.dim() >= 5
                    and dst.dim() == 4
                    and dst.shape[0] >= 2
                    and rope_rotary_dim > 0
                ):
                    with record_function("daser::rope_delta"):
                        _apply_rope_delta_to_key_block(
                            dst[0],
                            delta=round(pos_offset * rope_delta_scale),
                            rope_base=rope_base,
                            rotary_dim=rope_rotary_dim,
                            is_neox_style=rope_is_neox_style,
                        )
        copies += 1
    return copies


def _copy_kv_cache_to_staging(
    staging: torch.Tensor,
    kv_layer: torch.Tensor,
    layer_idx: int,
    block_ids: list[int],
    num_layers: int,
    slot_size: int,
) -> None:
    """Copy one vLLM KV layer for all requested blocks into staging.

    Args:
        staging: Contiguous uint8 tensor with slot-major DaseR layout.
        kv_layer: vLLM KV cache tensor for one attention layer.
        layer_idx: Index of ``kv_layer`` in the DaseR on-disk layer order.
        block_ids: vLLM KV block IDs to persist.
        num_layers: Total number of KV layers in the model.
        slot_size: Total bytes for all layers in one slot.

    Async/thread-safety:
        Synchronous GPU tensor copies on the vLLM worker thread.
    """
    if not block_ids:
        return
    layer_size = slot_size // num_layers
    num_slots = len(block_ids)
    staging_by_layer = staging.view(num_slots, num_layers, layer_size)
    block_index = torch.tensor(block_ids, dtype=torch.long, device=kv_layer.device)
    if kv_layer.dim() >= 2:
        src = kv_layer.index_select(1, block_index).movedim(1, 0)
    else:
        src = kv_layer.index_select(0, block_index)
    dst = (
        staging_by_layer[:, layer_idx, :]
        .view(kv_layer.dtype)
        .view(num_slots, *src.shape[1:])
    )
    dst.copy_(src)


def _build_store_write_spans(
    reqs_to_store: dict[str, ReqStoreSpec],
    req_slot_ranges: dict[str, tuple[int, int]],
    slot_size: int,
) -> list[StoreWriteSpan]:
    """Build coalesced pwrite spans for a step staging tensor.

    Args:
        reqs_to_store: Store specs keyed by request ID.
        req_slot_ranges: Request ID to ``(start_slot_index, end_slot_index)``.
        slot_size: DaseR bytes per KV slot.

    Returns:
        Coalesced write spans.

    Async/thread-safety:
        Pure CPU helper. It does not mutate connector state.
    """
    spans: list[StoreWriteSpan] = []
    for req_id, spec in reqs_to_store.items():
        slot_range = req_slot_ranges.get(req_id)
        if slot_range is None:
            continue
        start, end = slot_range
        if end <= start:
            continue
        spans.append(
            StoreWriteSpan(
                source_offset=start * slot_size,
                nbytes=(end - start) * slot_size,
                file_offset=spec.start_slot * slot_size,
            )
        )

    spans.sort(key=lambda span: (span.source_offset, span.file_offset))
    if not spans:
        return []

    merged: list[StoreWriteSpan] = [spans[0]]
    for span in spans[1:]:
        prev = merged[-1]
        prev_source_end = prev.source_offset + prev.nbytes
        prev_file_end = prev.file_offset + prev.nbytes
        if span.source_offset == prev_source_end and span.file_offset == prev_file_end:
            merged[-1] = StoreWriteSpan(
                source_offset=prev.source_offset,
                nbytes=prev.nbytes + span.nbytes,
                file_offset=prev.file_offset,
            )
        else:
            merged.append(span)
    return merged


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
        if kv_caches:
            sample = next(iter(kv_caches.values()))
            logger.info(
                "[CONNECTOR] register_kv_caches: %d layers, first shape=%s dtype=%s",
                len(kv_caches),
                sample.shape,
                sample.dtype,
            )

        if self._slot_size == 0 and self._layer_names:
            sample = next(iter(kv_caches.values()))
            num_blocks = sample.shape[1] if sample.dim() >= 2 else 1
            layer_size = sample.nbytes // num_blocks
            self._slot_size = layer_size * len(self._layer_names)
            logger.info(
                "[CONNECTOR] computed slot_size=%d from %d layers",
                self._slot_size,
                len(self._layer_names),
            )

        self._ensure_gds_ready()

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        super().bind_connector_metadata(connector_metadata)
        self._meta = connector_metadata
        self._reap_store_futures(block=False)
        self._pending_commits = set()
        self._clear_save_staging()
        slot_cursor = 0
        for req_id, spec in connector_metadata.reqs_to_store.items():
            num_slots = len(spec.block_ids)
            if num_slots == 0:
                continue
            self._pending_commits.add(spec.chunk_key)
            self._save_req_slot_ranges[req_id] = (slot_cursor, slot_cursor + num_slots)
            self._save_all_block_ids.extend(spec.block_ids)
            slot_cursor += num_slots
        self._throttle_store_futures(slot_cursor * self._slot_size)

    def clear_connector_metadata(self) -> None:
        """Clear metadata after forward pass completes."""
        super().clear_connector_metadata()
        self._meta = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Load all KV cache blocks for cache-hit requests.

        When ``load_profile`` is enabled in ``kv_connector_extra_config``, emits
        a PyTorch profiler segment report after the load completes.

        Args:
            forward_context: vLLM ForwardContext for this forward pass.
        """
        if getattr(self, "_load_profile", False):
            if self._meta is None or not self._meta.reqs_to_load:
                self._run_load_kv(forward_context)
                return
            trace_dir = getattr(self, "_load_profile_tensorboard_dir", "")
            wall_segments: dict[str, float] = {}
            load_start = time.perf_counter()
            on_trace_ready = None
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
                on_trace_ready = _tensorboard_trace_handler(trace_dir, wall_segments)
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                on_trace_ready=on_trace_ready,
            ) as prof:
                self._run_load_kv(forward_context, wall_segments=wall_segments)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall_segments["load_total_wall_ms"] = (
                time.perf_counter() - load_start
            ) * 1000.0
            logger.info(
                "[CONNECTOR] load profile report:\n%s",
                _format_load_profile_report(prof, wall_segments),
            )
            if trace_dir and wall_segments.get("daser::gds_read", 0.0) > 0.0:
                logger.info(
                    "[CONNECTOR] load profile tensorboard logdir: %s "
                    "(tensorboard --logdir=%s)",
                    trace_dir,
                    trace_dir,
                )
            return
        self._run_load_kv(forward_context)

    def _run_load_kv(
        self,
        forward_context: "ForwardContext",
        wall_segments: dict[str, float] | None = None,
    ) -> None:
        """Execute the synchronous KV load path for pending cache hits.

        Args:
            forward_context: vLLM ForwardContext for this forward pass.
        """
        if self._meta is None or not self._meta.reqs_to_load:
            return
        logger.debug(
            "[CONNECTOR] start_load_kv: %d reqs to load",
            len(self._meta.reqs_to_load),
        )
        if not self._ensure_gds_ready():
            return

        num_layers = len(self._layer_names)
        if num_layers == 0:
            return

        per_req: list[tuple[cupy.ndarray, torch.Tensor, ReqLoadSpec]] = []
        coros: list = []
        sample_tensor = next(iter(self._kv_caches.values()), None)
        if sample_tensor is None:
            return

        alloc_start = time.perf_counter()
        for spec in self._meta.reqs_to_load.values():
            num_slots = len(spec.block_ids)
            if num_slots == 0:
                continue
            total_bytes = num_slots * self._slot_size
            with record_function("daser::alloc_staging"):
                staging = torch.empty(
                    total_bytes, dtype=torch.uint8, device=sample_tensor.device
                )
                cp_staging = cupy.asarray(staging)

            async def _read(
                cp: cupy.ndarray = cp_staging,
                off: int = spec.start_slot * self._slot_size,
                nb: int = total_bytes,
            ) -> int:
                return await self._gds.read_into_async(cp, off, nb)

            coros.append(_read())
            per_req.append((cp_staging, staging, spec))
        alloc_start = _mark_wall_segment(
            wall_segments, "daser::alloc_staging", alloc_start
        )

        if not coros:
            return

        async def _run_all(cs: list) -> list:
            return await asyncio.gather(*cs)

        gds_start = time.perf_counter()
        with record_function("daser::gds_read"):
            asyncio.run_coroutine_threadsafe(_run_all(coros), self._bg_loop).result(
                timeout=120.0
            )
        gds_start = _mark_wall_segment(wall_segments, "daser::gds_read", gds_start)

        total_copies = 0
        copy_start = time.perf_counter()
        for _, staging, spec in per_req:
            with record_function("daser::staging_to_kv"):
                total_copies += _copy_staging_to_kv_cache(
                    staging=staging,
                    kv_caches=self._kv_caches,
                    layer_names=self._layer_names,
                    block_ids=spec.block_ids,
                    slot_size=self._slot_size,
                    load_key_scale=self._load_key_scale,
                    load_value_scale=self._load_value_scale,
                    pos_offset=spec.pos_offset,
                    rope_delta_scale=self._rope_delta_scale,
                    rope_base=self._rope_base,
                    rope_rotary_dim=self._rope_rotary_dim,
                    rope_is_neox_style=self._rope_is_neox_style,
                )
        _mark_wall_segment(wall_segments, "daser::staging_to_kv", copy_start)
        logger.debug(
            "[CONNECTOR] start_load_kv: %d reqs, %d GPU copies, %d GDS reads",
            len(per_req),
            total_copies,
            len(coros),
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
        """Aggregate this layer's KV into the step save staging buffer.

        Args:
            layer_name: name of the current attention layer.
            kv_layer: full KV cache tensor for this layer.
            attn_metadata: attention metadata (not directly used).
        """
        if self._meta is None or not self._meta.reqs_to_store:
            return
        if not self._ensure_gds_ready():
            return

        num_layers = len(self._layer_names)
        if num_layers == 0:
            return
        if layer_name not in self._layer_idx_map:
            logger.warning(
                "[CONNECTOR] save_kv_layer: unknown layer %s, skipping", layer_name
            )
            return
        if not self._save_all_block_ids:
            return

        if self._save_step_staging is None:
            self._save_step_staging = torch.empty(
                len(self._save_all_block_ids) * self._slot_size,
                dtype=torch.uint8,
                device=kv_layer.device,
            )

        _copy_kv_cache_to_staging(
            staging=self._save_step_staging,
            kv_layer=kv_layer,
            layer_idx=self._layer_idx_map[layer_name],
            block_ids=self._save_all_block_ids,
            num_layers=num_layers,
            slot_size=self._slot_size,
        )

    def wait_for_save(self) -> None:
        """Submit save staging writes and commit them after IO completes."""
        if self._meta is None or self._save_step_staging is None:
            self._reap_store_futures(block=False)
            return

        staging = self._save_step_staging
        cp_staging = cupy.asarray(staging)
        spans = _build_store_write_spans(
            reqs_to_store=self._meta.reqs_to_store,
            req_slot_ranges=self._save_req_slot_ranges,
            slot_size=self._slot_size,
        )
        commit_keys = list(self._pending_commits)

        if spans and commit_keys:
            future = asyncio.run_coroutine_threadsafe(
                self._write_and_commit(cp_staging, spans, commit_keys),
                self._bg_loop,
            )
            self._store_futures.append(
                StoreFuture(future=future, staging=staging, nbytes=staging.nbytes)
            )
            self._inflight_store_bytes += staging.nbytes

        self._clear_save_staging()
        self._pending_commits.clear()
        self._reap_store_futures(block=False)

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
        self._reap_store_futures(block=False)
        return None, None

    def shutdown(self) -> None:
        """Close GDS file handle and stop the background IO loop."""
        if self._role != KVConnectorRole.WORKER:
            return
        self._reap_store_futures(block=True)
        if self._gds is not None:
            self._gds.close()
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)

    def _run_bg_loop(self) -> None:
        """Run the background asyncio IO loop."""
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

    def _ensure_gds_ready(self) -> bool:
        """Initialize the worker GDS backend once the server store is ready."""
        if self._gds is not None:
            return True

        self._refresh_runtime_config()
        if not self._store_path or not os.path.exists(self._store_path):
            logger.warning(
                "[CONNECTOR] store file %s is not ready; start DaseR server "
                "before sending requests",
                self._store_path,
            )
            return False

        self._gds = GDSTransferLayer(self._store_path)
        logger.info("[CONNECTOR] GDS backend=%s", self._gds.backend.value)
        return True

    def _clear_save_staging(self) -> None:
        """Clear worker-side per-forward save staging state."""
        self._save_all_block_ids = []
        self._save_req_slot_ranges = {}
        self._save_step_staging = None

    async def _write_and_commit(
        self,
        cp_staging: cupy.ndarray,
        spans: list[StoreWriteSpan],
        commit_keys: list[str],
    ) -> None:
        """Write staged KV spans and publish chunks after IO completes.

        Args:
            cp_staging: CuPy view of the step staging tensor.
            spans: Coalesced source/destination write spans.
            commit_keys: Chunk keys to publish after all writes complete.
        """

        async def _write(span: StoreWriteSpan) -> int:
            cp_slice = cp_staging[span.source_offset : span.source_offset + span.nbytes]
            return await self._gds.write_async(cp_slice, span.file_offset, span.nbytes)

        await asyncio.gather(*(_write(span) for span in spans))
        await asyncio.gather(
            *(self._ipc_async.commit_chunk(key) for key in commit_keys)
        )

    def _reap_store_futures(self, block: bool) -> None:
        """Collect completed background store tasks.

        Args:
            block: If True, wait for every pending task. If False, collect only
                tasks that are already complete.
        """
        remaining: list[StoreFuture] = []
        for store_future in self._store_futures:
            if block or store_future.future.done():
                store_future.future.result(timeout=120.0)
                self._inflight_store_bytes -= store_future.nbytes
            else:
                remaining.append(store_future)
        self._store_futures = remaining
        if not self._store_futures:
            self._inflight_store_bytes = 0

    def _throttle_store_futures(self, next_nbytes: int) -> None:
        """Bound GPU memory held by deferred store staging buffers.

        Args:
            next_nbytes: Bytes needed by the next step staging allocation.
        """
        if next_nbytes <= 0:
            return
        self._reap_store_futures(block=False)
        while (
            self._store_futures
            and self._inflight_store_bytes + next_nbytes
            > self._max_inflight_store_bytes
        ):
            store_future = self._store_futures.pop(0)
            store_future.future.result(timeout=120.0)
            self._inflight_store_bytes -= store_future.nbytes
