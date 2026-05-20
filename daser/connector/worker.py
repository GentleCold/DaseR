# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Standard
import asyncio
from dataclasses import dataclass
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
    ReqLoadSpec,
    ReqStoreSpec,
    StoreWriteSpan,
)
from daser.logging import init_logger
from daser.transfer.cuda_ipc import (
    cuda_array_device_id,
    cuda_array_pointer,
    export_cuda_ipc_handle,
)

logger = init_logger(__name__)

DEFAULT_ROPE_DELTA_SCALE = 1.0
DEFAULT_STORE_STAGING_BYTES = 1 << 30
DEFAULT_PENDING_STORE_STAGING_BYTES = 2 << 30
MIN_STORE_STAGING_BYTES = 256 << 20


@dataclass(frozen=True)
class _StagedStoreBatch:
    """Worker-owned CUDA staging batch ready for server-side transfer."""

    buffer: torch.Tensor
    ready_event: torch.cuda.Event | None
    spans: list[StoreWriteSpan]


@dataclass(frozen=True)
class _LoadCopyRun:
    """One contiguous staging range that can be copied with one layer loop."""

    start: int
    end: int
    block_ids: list[int]
    pos_offset: int


def _synchronize_cuda_tensor(tensor: torch.Tensor) -> None:
    """Synchronize pending CUDA work for a tensor before cross-process handoff.

    Args:
        tensor: Tensor whose device stream must be visible across CUDA IPC.

    Async/thread-safety:
        Synchronous barrier on the current worker thread. It is intentionally
        conservative until CUDA IPC event handoff is added.
    """
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()


def _record_cuda_event(tensor: torch.Tensor) -> torch.cuda.Event | None:
    """Record the tensor's current CUDA stream for deferred synchronization.

    Args:
        tensor: Tensor whose producer stream should be observed.

    Returns:
        A CUDA event recorded on the current stream, or ``None`` for CPU
        tensors.

    Async/thread-safety:
        Must be called on the producer thread before handing ``tensor`` to a
        background task.
    """
    if not tensor.is_cuda:
        return None
    event = torch.cuda.Event(blocking=False)
    event.record(torch.cuda.current_stream(tensor.device))
    return event


def _contiguous_block_range(block_ids: list[int]) -> tuple[int, int] | None:
    """Return ``(start, stop)`` when block IDs are a contiguous range."""
    if not block_ids:
        return None
    start = block_ids[0]
    for idx, block_id in enumerate(block_ids):
        if block_id != start + idx:
            return None
    return start, start + len(block_ids)


def _derive_store_staging_limits(device: torch.device) -> tuple[int, int]:
    """Return transient GPU staging caps for a CUDA device.

    Args:
        device: Device that will own worker-side staging tensors.

    Returns:
        ``(single_batch_bytes, pending_bytes)``. The cap is based on both total
        and currently free VRAM. This keeps staging transient and bounded after
        vLLM has allocated its KV cache instead of reserving a fixed fraction of
        the device unconditionally.

    Async/thread-safety:
        Reads CUDA device properties only; safe during worker initialization.
    """
    if device.type != "cuda":
        return DEFAULT_STORE_STAGING_BYTES, DEFAULT_PENDING_STORE_STAGING_BYTES
    props = torch.cuda.get_device_properties(device)
    total = int(props.total_memory)
    try:
        free, _ = torch.cuda.mem_get_info(device)
        free = int(free)
    except (RuntimeError, TypeError, ValueError):
        free = total
    batch_ceiling = min(
        DEFAULT_STORE_STAGING_BYTES,
        max(MIN_STORE_STAGING_BYTES, free // 16),
    )
    pending_ceiling = min(
        DEFAULT_PENDING_STORE_STAGING_BYTES,
        max(batch_ceiling, free // 8),
    )
    batch = min(
        batch_ceiling,
        max(MIN_STORE_STAGING_BYTES, total // 80),
    )
    pending = min(
        pending_ceiling,
        max(batch, total // 26),
    )
    return batch, pending


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
    block_range = _contiguous_block_range(block_ids)
    block_index = (
        None
        if block_range is not None
        else torch.tensor(block_ids, dtype=torch.long, device=staging.device)
    )

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
            if block_range is None:
                if block_index is None:
                    raise RuntimeError("block_index is required for non-contiguous IDs")
                kv_tensor.index_copy_(1, block_index, src.movedim(0, 1))
            else:
                start, stop = block_range
                kv_tensor[:, start:stop].copy_(src.movedim(0, 1))
        else:
            sample = kv_tensor[block_ids[0]]
            src = (
                staging_by_layer[:, layer_idx, :]
                .view(kv_tensor.dtype)
                .view(num_slots, *sample.shape)
            )
            if block_range is None:
                if block_index is None:
                    raise RuntimeError("block_index is required for non-contiguous IDs")
                kv_tensor.index_copy_(0, block_index, src)
            else:
                start, stop = block_range
                kv_tensor[start:stop].copy_(src)
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
    block_index: torch.Tensor | None = None,
) -> None:
    """Copy one vLLM KV layer for requested blocks into slot-major staging.

    Args:
        staging: Contiguous uint8 tensor with slot-major DaseR layout.
        kv_layer: vLLM KV cache tensor for one attention layer.
        layer_idx: Index of ``kv_layer`` in the DaseR on-disk layer order.
        block_ids: vLLM KV block IDs to persist.
        num_layers: Total number of KV layers in the model.
        slot_size: Total bytes for all layers in one slot.
        block_index: Optional prebuilt CUDA/CPU tensor containing block IDs.

    Async/thread-safety:
        Synchronous GPU tensor copies on the vLLM worker thread.
    """
    if not block_ids:
        return
    layer_size = slot_size // num_layers
    num_slots = len(block_ids)
    staging_by_layer = staging.view(num_slots, num_layers, layer_size)
    if block_index is None:
        block_index = torch.tensor(block_ids, dtype=torch.long, device=kv_layer.device)
    if kv_layer.dim() >= 2:
        block_range = _contiguous_block_range(block_ids)
        if block_range is None:
            src = kv_layer.index_select(1, block_index).movedim(1, 0)
        else:
            start, stop = block_range
            src = kv_layer[:, start:stop].movedim(1, 0)
    else:
        block_range = _contiguous_block_range(block_ids)
        if block_range is None:
            src = kv_layer.index_select(0, block_index)
        else:
            start, stop = block_range
            src = kv_layer[start:stop]
    dst = (
        staging_by_layer[:, layer_idx, :]
        .view(kv_layer.dtype)
        .view(num_slots, *src.shape[1:])
    )
    dst.copy_(src)


def _build_load_read_plan(
    reqs_to_load: dict[str, ReqLoadSpec],
    slot_size: int,
) -> tuple[int, list[dict[str, int]], list[tuple[int, int, ReqLoadSpec]]]:
    """Build a combined transfer-load plan for one forward step.

    Args:
        reqs_to_load: request ID to load spec from scheduler metadata.
        slot_size: bytes per vLLM KV slot.

    Returns:
        ``(total_bytes, spans, per_req_ranges)`` where spans target one
        combined staging tensor and per-request ranges map slices back to
        their original load specs.
    """
    total_bytes = 0
    spans: list[dict[str, int]] = []
    per_req_ranges: list[tuple[int, int, ReqLoadSpec]] = []
    for spec in reqs_to_load.values():
        num_slots = len(spec.block_ids)
        if num_slots == 0:
            continue
        nbytes = num_slots * slot_size
        start = total_bytes
        end = start + nbytes
        spans.append(
            {
                "target_offset": start,
                "nbytes": nbytes,
                "file_offset": spec.start_slot * slot_size,
            }
        )
        per_req_ranges.append((start, end, spec))
        total_bytes = end
    return total_bytes, spans, per_req_ranges


def _build_load_copy_runs(
    per_req_ranges: list[tuple[int, int, ReqLoadSpec]],
) -> list[_LoadCopyRun]:
    """Merge adjacent load ranges that share the same KV transform.

    Args:
        per_req_ranges: Per-request staging ranges from ``_build_load_read_plan``.

    Returns:
        Ordered copy runs. Each run covers a contiguous staging slice and the
        matching flattened block ID list.

    Async/thread-safety:
        Pure CPU helper. It does not mutate connector state.
    """
    runs: list[_LoadCopyRun] = []
    run_start = -1
    run_end = -1
    run_pos_offset = 0
    run_block_ids: list[int] = []

    def flush() -> None:
        nonlocal run_start, run_end, run_pos_offset, run_block_ids
        if run_start >= 0 and run_block_ids:
            runs.append(
                _LoadCopyRun(
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

    for start, end, spec in per_req_ranges:
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


def _build_staging_store_batches(
    reqs_to_store: dict[str, ReqStoreSpec],
    slot_size: int,
    max_batch_bytes: int = DEFAULT_STORE_STAGING_BYTES,
) -> list[tuple[list[int], list[StoreWriteSpan]]]:
    """Split store requests into bounded slot-major staging batches.

    Args:
        reqs_to_store: Store specs keyed by request ID.
        slot_size: DaseR bytes per KV slot.
        max_batch_bytes: Maximum GPU staging bytes per batch.

    Returns:
        List of ``(block_ids, spans)`` batches. Span source offsets are relative
        to that batch's staging tensor.

    Async/thread-safety:
        Pure CPU helper. It does not mutate connector state.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    max_slots = max(1, max_batch_bytes // slot_size)
    batches: list[tuple[list[int], list[StoreWriteSpan]]] = []
    batch_blocks: list[int] = []
    batch_spans: list[StoreWriteSpan] = []

    def flush_batch() -> None:
        nonlocal batch_blocks, batch_spans
        if batch_blocks:
            batches.append((batch_blocks, batch_spans))
        batch_blocks = []
        batch_spans = []

    for spec in reqs_to_store.values():
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
                    file_offset=(spec.start_slot + cursor) * slot_size,
                    chunk_key=spec.chunk_key,
                    start_slot=spec.start_slot,
                    num_slots=spec.num_slots,
                )
            )
            cursor += take
    flush_batch()
    return batches


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

        total_bytes, spans, per_req_ranges = _build_load_read_plan(
            self._meta.reqs_to_load,
            self._slot_size,
        )
        if not spans:
            return

        staging = torch.empty(
            total_bytes, dtype=torch.uint8, device=sample_tensor.device
        )
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

        total_copies = 0
        copy_runs = _build_load_copy_runs(per_req_ranges)
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

        logger.debug(
            "[CONNECTOR] start_load_kv: %d reqs, %d copy runs, %d GPU copies, "
            "1 transfer read",
            len(per_req_ranges),
            len(copy_runs),
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
                self._track_save_future(future, staged.buffer.nbytes)
                batch_futures.append(future)
            if batch_futures:
                commit_future = asyncio.run_coroutine_threadsafe(
                    self._commit_after_store_futures(batch_futures, commit_keys),
                    self._bg_loop,
                )
                self._track_save_future(commit_future, 0)
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
                future, staging_bytes = record
            else:
                future = record
                staging_bytes = 0
            if block or future.done():
                try:
                    future.result(timeout=120.0)
                finally:
                    pending_bytes = max(0, pending_bytes - staging_bytes)
            else:
                remaining.append((future, staging_bytes))
        self._save_futures = remaining
        self._pending_save_staging_bytes = pending_bytes

    def _track_save_future(self, future: Any, staging_bytes: int) -> None:
        """Track one background save future and its live staging bytes.

        Args:
            future: Future returned by ``asyncio.run_coroutine_threadsafe``.
            staging_bytes: GPU staging bytes kept alive by the future.

        Async/thread-safety:
            Called on the worker thread. Completion is collected by
            ``_reap_save_futures``.
        """
        self._pending_save_staging_bytes = (
            getattr(self, "_pending_save_staging_bytes", 0) + staging_bytes
        )
        self._save_futures.append((future, staging_bytes))

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
                future, staging_bytes = record
            else:
                future = record
                staging_bytes = 0
            try:
                future.result(timeout=120.0)
            finally:
                self._pending_save_staging_bytes = max(
                    0,
                    getattr(self, "_pending_save_staging_bytes", 0) - staging_bytes,
                )
            self._reap_save_futures(block=False)

    def _stage_store_batch(
        self,
        block_ids: list[int],
        spans: list[StoreWriteSpan],
    ) -> _StagedStoreBatch | None:
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
        staging = torch.empty(nbytes, dtype=torch.uint8, device=sample_tensor.device)
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
        return _StagedStoreBatch(
            buffer=staging,
            ready_event=_record_cuda_event(staging),
            spans=spans,
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
