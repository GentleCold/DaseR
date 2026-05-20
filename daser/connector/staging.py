# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Standard
from dataclasses import dataclass, replace

# Third Party
import torch

# First Party
from daser.connector.metadata import ReqLoadSpec, ReqStoreSpec, StoreWriteSpan

DEFAULT_ROPE_DELTA_SCALE = 1.0
DEFAULT_STORE_STAGING_BYTES = 512 << 20
DEFAULT_PENDING_STORE_STAGING_BYTES = 1 << 30
MIN_STORE_STAGING_BYTES = 64 << 20


@dataclass
class CudaStagingLease:
    """One logical staging allocation leased from ``CudaStagingPool``.

    Args:
        pool: Owning pool that will receive the allocation on release.
        tensor: Backing tensor, possibly larger than ``nbytes``.
        nbytes: Logical byte count used by the current transfer.

    Async/thread-safety:
        The lease is released on the vLLM worker thread after transfer
        completion. It must not be reused while an async store future owns it.
    """

    pool: "CudaStagingPool"
    tensor: torch.Tensor
    nbytes: int
    _released: bool = False

    @property
    def view(self) -> torch.Tensor:
        """Return the logical byte view for the active transfer.

        Returns:
            A 1-D uint8 tensor slice with ``nbytes`` elements.

        Async/thread-safety:
            The returned tensor remains valid until ``release`` is called.
        """
        return self.tensor[: self.nbytes]

    def release(self) -> None:
        """Return the lease to the owning pool.

        Async/thread-safety:
            Call once no CUDA IPC transfer can access ``view``.
        """
        if self._released:
            return
        self._released = True
        self.pool.release(self)


class CudaStagingPool:
    """Reusable worker-side GPU staging buffers for CUDA IPC transfer.

    Args:
        device: Device on which staging tensors are allocated.
        initial_bytes: Size of the buffer allocated during initialization.
        max_buffer_bytes: Maximum size of one staging buffer.

    Async/thread-safety:
        The pool is owned by one vLLM worker thread. It does not use locks; the
        worker tracks async store futures and releases leases only after those
        futures complete.
    """

    def __init__(
        self,
        device: torch.device,
        initial_bytes: int,
        max_buffer_bytes: int,
    ) -> None:
        if initial_bytes < 0:
            raise ValueError("initial_bytes must be non-negative")
        if max_buffer_bytes <= 0:
            raise ValueError("max_buffer_bytes must be positive")
        self._device = device
        self._max_buffer_bytes = max_buffer_bytes
        self._free: list[torch.Tensor] = []
        if initial_bytes:
            self._free.append(
                torch.empty(initial_bytes, dtype=torch.uint8, device=device)
            )

    @property
    def max_buffer_bytes(self) -> int:
        """Return the maximum bytes allowed for a single staging buffer."""
        return self._max_buffer_bytes

    def acquire(self, nbytes: int) -> CudaStagingLease:
        """Lease a staging tensor with at least ``nbytes`` capacity.

        Args:
            nbytes: Logical transfer byte count.

        Returns:
            Lease whose ``view`` is sized to ``nbytes``.

        Raises:
            ValueError: If ``nbytes`` exceeds the configured single-buffer cap.

        Async/thread-safety:
            Synchronous worker-thread allocation path. Reuses preallocated
            buffers first; only grows when the current workload exceeds the
            initialization size.
        """
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if nbytes > self._max_buffer_bytes:
            raise ValueError(
                f"staging request {nbytes} exceeds cap {self._max_buffer_bytes}"
            )
        selected_idx = -1
        selected_size = 0
        for idx, candidate in enumerate(self._free):
            capacity = int(candidate.numel())
            if capacity >= nbytes and (selected_idx < 0 or capacity < selected_size):
                selected_idx = idx
                selected_size = capacity
        if selected_idx >= 0:
            tensor = self._free.pop(selected_idx)
        else:
            capacity = max(nbytes, min(self._max_buffer_bytes, nbytes))
            tensor = torch.empty(capacity, dtype=torch.uint8, device=self._device)
        return CudaStagingLease(pool=self, tensor=tensor, nbytes=nbytes)

    def release(self, lease: CudaStagingLease) -> None:
        """Return a completed staging lease to the free list.

        Args:
            lease: Lease previously returned by ``acquire``.

        Async/thread-safety:
            Caller must ensure no CUDA operation or server CUDA IPC mapping is
            still using the tensor.
        """
        self._free.append(lease.tensor)


@dataclass(frozen=True)
class StagedStoreBatch:
    """Worker-owned CUDA staging batch ready for server-side transfer.

    Args:
        buffer: Logical uint8 staging view exported through CUDA IPC.
        ready_event: CUDA event recorded after KV -> staging copies.
        spans: Server write spans targeting ``buffer``.
        lease: Optional reusable staging lease backing ``buffer``.

    Async/thread-safety:
        The batch crosses to the connector background loop; ``lease`` is
        released only after the async server transfer future completes.
    """

    buffer: torch.Tensor
    ready_event: torch.cuda.Event | None
    spans: list[StoreWriteSpan]
    lease: CudaStagingLease | None


@dataclass(frozen=True)
class LoadCopyRun:
    """One contiguous staging range that can be copied with one layer loop."""

    start: int
    end: int
    block_ids: list[int]
    pos_offset: int


def synchronize_cuda_tensor(tensor: torch.Tensor) -> None:
    """Synchronize pending CUDA work for a tensor before cross-process handoff.

    Args:
        tensor: Tensor whose device stream must be visible across CUDA IPC.

    Async/thread-safety:
        Synchronous barrier on the current worker thread. It is intentionally
        conservative until CUDA IPC event handoff is added.
    """
    if tensor.is_cuda:
        torch.cuda.current_stream(tensor.device).synchronize()


def record_cuda_event(tensor: torch.Tensor) -> torch.cuda.Event | None:
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


def contiguous_block_range(block_ids: list[int]) -> tuple[int, int] | None:
    """Return ``(start, stop)`` when block IDs are a contiguous range."""
    if not block_ids:
        return None
    start = block_ids[0]
    for idx, block_id in enumerate(block_ids):
        if block_id != start + idx:
            return None
    return start, start + len(block_ids)


def derive_store_staging_limits(device: torch.device) -> tuple[int, int]:
    """Return bounded GPU staging caps for a CUDA device.

    Args:
        device: Device that will own worker-side staging tensors.

    Returns:
        ``(single_batch_bytes, pending_bytes)``. The cap is based on both total
        and currently free VRAM after vLLM has allocated KV cache. Defaults are
        intentionally modest because staging is an IPC transport buffer, not a
        persistent cache tier.

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
    batch = min(
        DEFAULT_STORE_STAGING_BYTES,
        max(MIN_STORE_STAGING_BYTES, min(total // 160, free // 32)),
    )
    pending = min(
        DEFAULT_PENDING_STORE_STAGING_BYTES,
        max(batch, min(total // 80, free // 16)),
    )
    return batch, pending


def apply_rope_delta_to_key_block(
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


def copy_staging_to_kv_cache(
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
    block_range = contiguous_block_range(block_ids)
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
                    apply_rope_delta_to_key_block(
                        dst[0],
                        delta=round(pos_offset * rope_delta_scale),
                        rope_base=rope_base,
                        rotary_dim=rope_rotary_dim,
                        is_neox_style=rope_is_neox_style,
                    )
        copies += 1
    return copies


def copy_kv_cache_to_staging(
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
        block_range = contiguous_block_range(block_ids)
        if block_range is None:
            src = kv_layer.index_select(1, block_index).movedim(1, 0)
        else:
            start, stop = block_range
            src = kv_layer[:, start:stop].movedim(1, 0)
    else:
        block_range = contiguous_block_range(block_ids)
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


def build_load_read_plan(
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


def build_load_read_batches(
    reqs_to_load: dict[str, ReqLoadSpec],
    slot_size: int,
    max_batch_bytes: int,
) -> list[tuple[int, list[dict[str, int]], list[tuple[int, int, ReqLoadSpec]]]]:
    """Build bounded load staging plans for one forward step.

    Args:
        reqs_to_load: request ID to load spec from scheduler metadata.
        slot_size: bytes per vLLM KV slot.
        max_batch_bytes: Maximum staging bytes for one transfer batch.

    Returns:
        List of ``build_load_read_plan``-style tuples. Individual requests are
        split at block boundaries when one request exceeds the staging cap.

    Async/thread-safety:
        Pure CPU helper. It does not mutate connector state.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    if max_batch_bytes <= 0:
        raise ValueError("max_batch_bytes must be positive")
    max_slots = max(1, max_batch_bytes // slot_size)
    batches: list[
        tuple[int, list[dict[str, int]], list[tuple[int, int, ReqLoadSpec]]]
    ] = []
    current: dict[str, ReqLoadSpec] = {}
    current_slots = 0
    synthetic_id = 0

    def flush() -> None:
        nonlocal current, current_slots
        if current:
            batches.append(build_load_read_plan(current, slot_size))
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
                file_offset=(spec.start_slot + cursor) * slot_size,
            )
            key = (
                req_id
                if cursor == 0 and take == len(spec.block_ids)
                else (f"{req_id}#{synthetic_id}")
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
    """Merge adjacent load ranges that share the same KV transform.

    Args:
        per_req_ranges: Per-request staging ranges from ``build_load_read_plan``.

    Returns:
        Ordered copy runs. Each run covers a contiguous staging slice and the
        matching flattened block ID list.

    Async/thread-safety:
        Pure CPU helper. It does not mutate connector state.
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


def build_staging_store_batches(
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
