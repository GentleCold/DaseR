# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# Third Party
import torch

# First Party
from daser.logging import init_logger
from daser.ops.rope_apply import (
    apply_rope_delta_to_key_block as _apply_rope_delta_to_key_block,
)
from daser.ops.rope_apply import (
    apply_rope_delta_to_kv_key_block as _apply_rope_delta_to_kv_key_block,
)
from daser.ops.rope_apply import (
    apply_rope_delta_to_kv_key_block_table,
    restore_cross_layer_kv_cache_table,
)

DEFAULT_ROPE_DELTA_SCALE = 1.0
CROSS_LAYER_KV_CACHE_KEY = "__cross_layers__"
FUSED_RESTORE_MIN_SLOTS = 32

logger = init_logger(__name__)
_rope_table_cache: dict[
    tuple[torch.device, int, float, int],
    tuple[torch.Tensor, torch.Tensor],
] = {}


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


def apply_rope_delta_to_key_block(
    key_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Rotate an already-RoPE'd K block by a relative position delta.

    Args:
        key_block: K cache block with shape [..., block_tokens, heads, head_dim].
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. ``key_block`` is modified in place.

    Async/thread-safety:
        Performs tensor work on the current PyTorch stream.
    """
    _apply_rope_delta_to_key_block(
        key_block,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def apply_rope_delta_to_kv_key_block(
    kv_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Rotate K entries inside a full KV staging block by a RoPE delta.

    Args:
        kv_block: KV staging tensor with shape
            ``[blocks, layers, 2, block_tokens, heads, head_dim]``.
        delta: relative RoPE position delta to apply in place.
        rope_base: RoPE theta/base.
        rotary_dim: number of head dimensions covered by RoPE.
        is_neox_style: True for split-half rotation, False for interleaved.

    Returns:
        None. Only the key slice is modified in place.

    Async/thread-safety:
        Performs tensor work on the current PyTorch stream.
    """
    _apply_rope_delta_to_kv_key_block(
        kv_block,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def _transform_loaded_staging_batch(
    staging_by_layer: torch.Tensor,
    layer_sample: torch.Tensor,
    load_key_scale: float,
    load_value_scale: float,
    pos_offset: int,
    rope_delta_scale: float,
    rope_base: float,
    rope_rotary_dim: int,
    rope_is_neox_style: bool,
) -> None:
    """Apply load-time transforms once over all staging layers in a copy run."""
    if staging_by_layer.numel() == 0 or layer_sample.dim() < 4:
        return
    num_slots = int(staging_by_layer.shape[0])
    num_layers = int(staging_by_layer.shape[1])
    kv_batch = staging_by_layer.view(layer_sample.dtype).view(
        num_slots,
        num_layers,
        *layer_sample.shape,
    )
    if load_key_scale != 1.0:
        kv_batch[:, :, 0].mul_(load_key_scale)
    if load_value_scale != 1.0:
        kv_batch[:, :, 1].mul_(load_value_scale)
    if (
        not pos_offset
        or rope_rotary_dim <= 0
        or layer_sample.shape[-1] < rope_rotary_dim
    ):
        return
    if kv_batch.dim() == 6 and kv_batch.is_contiguous():
        apply_rope_delta_to_kv_key_block(
            kv_batch,
            delta=round(pos_offset * rope_delta_scale),
            rope_base=rope_base,
            rotary_dim=rope_rotary_dim,
            is_neox_style=rope_is_neox_style,
        )
        return
    if layer_sample.dim() != 4:
        return
    apply_rope_delta_to_key_block(
        kv_batch[:, :, 0],
        delta=round(pos_offset * rope_delta_scale),
        rope_base=rope_base,
        rotary_dim=rope_rotary_dim,
        is_neox_style=rope_is_neox_style,
    )


def _copy_staging_to_cross_layer_kv_cache(
    staging_by_layer: torch.Tensor,
    cross_layer_kv_cache: torch.Tensor,
    block_ids: list[int],
    load_key_scale: float,
    load_value_scale: float,
    pos_offset: int,
    rope_delta_scale: float,
    rope_base: float,
    rope_rotary_dim: int,
    rope_is_neox_style: bool,
) -> int:
    """Copy staging bytes into a vLLM cross-layer KV cache in one bulk write."""
    num_slots = len(block_ids)
    layer_sample = cross_layer_kv_cache[block_ids[0], 0]
    src = staging_by_layer.view(cross_layer_kv_cache.dtype).view(
        num_slots,
        cross_layer_kv_cache.shape[1],
        *layer_sample.shape,
    )
    block_range = contiguous_block_range(block_ids)
    dst_contiguous = False
    start = 0
    stop = 0
    if block_range is not None:
        start, stop = block_range
        dst_contiguous = cross_layer_kv_cache[start:stop].is_contiguous()
    can_rotate_target = (
        block_range is not None
        and load_key_scale == 1.0
        and load_value_scale == 1.0
        and pos_offset
        and rope_rotary_dim > 0
        and layer_sample.shape[-1] >= rope_rotary_dim
        and src.is_contiguous()
        and dst_contiguous
    )
    if can_rotate_target:
        dst = cross_layer_kv_cache[start:stop]
        delta = round(pos_offset * rope_delta_scale)
        if num_slots >= FUSED_RESTORE_MIN_SLOTS:
            _restore_cross_layer_with_tables(
                src,
                dst,
                delta=delta,
                rope_base=rope_base,
                rotary_dim=rope_rotary_dim,
                is_neox_style=rope_is_neox_style,
            )
            return 1
        dst.copy_(src)
        _apply_rope_delta_with_tables(
            dst,
            delta=delta,
            rope_base=rope_base,
            rotary_dim=rope_rotary_dim,
            is_neox_style=rope_is_neox_style,
        )
        return 1
    _transform_loaded_staging_batch(
        staging_by_layer,
        layer_sample=layer_sample,
        load_key_scale=load_key_scale,
        load_value_scale=load_value_scale,
        pos_offset=pos_offset,
        rope_delta_scale=rope_delta_scale,
        rope_base=rope_base,
        rope_rotary_dim=rope_rotary_dim,
        rope_is_neox_style=rope_is_neox_style,
    )
    if block_range is None:
        block_index = torch.tensor(
            block_ids,
            dtype=torch.long,
            device=staging_by_layer.device,
        )
        cross_layer_kv_cache.index_copy_(0, block_index, src)
    else:
        start, stop = block_range
        cross_layer_kv_cache[start:stop].copy_(src)
    return 1


def _apply_rope_delta_with_tables(
    kv_block: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Apply RoPE using cached trig tables."""
    if kv_block.device.type != "cuda" or kv_block.dtype not in (
        torch.bfloat16,
        torch.float16,
        torch.float32,
    ):
        raise ValueError("TileLang RoPE restore requires CUDA fp16/bf16/fp32 KV")
    cos_table, sin_table = _get_rope_delta_tables(
        kv_block.device,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
    )
    apply_rope_delta_to_kv_key_block_table(
        kv_block,
        cos_table=cos_table,
        sin_table=sin_table,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def _restore_cross_layer_with_tables(
    src_kv: torch.Tensor,
    dst_kv: torch.Tensor,
    delta: int,
    rope_base: float,
    rotary_dim: int,
    is_neox_style: bool,
) -> None:
    """Restore cross-layer KV using cached trig tables."""
    cos_table, sin_table = _get_rope_delta_tables(
        src_kv.device,
        delta=delta,
        rope_base=rope_base,
        rotary_dim=rotary_dim,
    )
    restore_cross_layer_kv_cache_table(
        src_kv,
        dst_kv,
        cos_table=cos_table,
        sin_table=sin_table,
        rotary_dim=rotary_dim,
        is_neox_style=is_neox_style,
    )


def _get_rope_delta_tables(
    device: torch.device,
    delta: int,
    rope_base: float,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached fp32 RoPE delta cosine/sine tables."""
    key = (device, int(delta), float(rope_base), int(rotary_dim))
    cached = _rope_table_cache.get(key)
    if cached is not None:
        return cached
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    freqs = int(delta) * inv_freq
    tables = (freqs.cos().contiguous(), freqs.sin().contiguous())
    _rope_table_cache[key] = tables
    return tables


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
    cross_layer_kv_cache = kv_caches.get(CROSS_LAYER_KV_CACHE_KEY)
    if (
        cross_layer_kv_cache is not None
        and cross_layer_kv_cache.dim() >= 6
        and cross_layer_kv_cache.shape[1] == num_layers
    ):
        return _copy_staging_to_cross_layer_kv_cache(
            staging_by_layer=staging_by_layer,
            cross_layer_kv_cache=cross_layer_kv_cache,
            block_ids=block_ids,
            load_key_scale=load_key_scale,
            load_value_scale=load_value_scale,
            pos_offset=pos_offset,
            rope_delta_scale=rope_delta_scale,
            rope_base=rope_base,
            rope_rotary_dim=rope_rotary_dim,
            rope_is_neox_style=rope_is_neox_style,
        )
    first_kv = next(
        (kv_caches[name] for name in layer_names if kv_caches.get(name) is not None),
        None,
    )
    if first_kv is not None:
        layer_sample = (
            first_kv[:, block_ids[0]] if first_kv.dim() >= 2 else first_kv[block_ids[0]]
        )
        _transform_loaded_staging_batch(
            staging_by_layer,
            layer_sample=layer_sample,
            load_key_scale=load_key_scale,
            load_value_scale=load_value_scale,
            pos_offset=pos_offset,
            rope_delta_scale=rope_delta_scale,
            rope_base=rope_base,
            rope_rotary_dim=rope_rotary_dim,
            rope_is_neox_style=rope_is_neox_style,
        )
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
        # KV tensors are either block-major ([blocks, ...]) or kv-major
        # ([2, blocks, ...]); block_dim points at the block axis.
        block_dim = 1 if kv_tensor.dim() >= 2 else 0
        sample = (
            kv_tensor[:, block_ids[0]] if block_dim == 1 else kv_tensor[block_ids[0]]
        )
        src = (
            staging_by_layer[:, layer_idx, :]
            .view(kv_tensor.dtype)
            .view(num_slots, *sample.shape)
        )
        # staging is slot-major (slots first); align it to the block axis.
        src = src.movedim(0, block_dim)
        if block_range is None:
            if block_index is None:
                raise RuntimeError("block_index is required for non-contiguous IDs")
            kv_tensor.index_copy_(block_dim, block_index, src)
        else:
            start, stop = block_range
            if block_dim == 1:
                kv_tensor[:, start:stop].copy_(src)
            else:
                kv_tensor[start:stop].copy_(src)
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


def copy_cross_layer_kv_cache_to_staging(
    staging: torch.Tensor,
    kv_cache: torch.Tensor,
    block_ids: list[int],
    num_layers: int,
    slot_size: int,
    block_index: torch.Tensor | None = None,
) -> None:
    """Copy vLLM cross-layer KV blocks into slot-major staging bytes.

    Args:
        staging: Contiguous uint8 tensor with slot-major DaseR layout.
        kv_cache: vLLM cross-layer KV cache tensor with blocks as dim 0 and
            layers as dim 1.
        block_ids: vLLM KV block IDs to persist.
        num_layers: Total number of KV layers in the model.
        slot_size: Total bytes for all layers in one slot.
        block_index: Optional prebuilt tensor containing block IDs.

    Async/thread-safety:
        Synchronous GPU tensor copy on the vLLM worker thread.
    """
    if not block_ids:
        return
    layer_size = slot_size // num_layers
    num_slots = len(block_ids)
    staging_by_layer = staging.view(num_slots, num_layers, layer_size)
    block_range = contiguous_block_range(block_ids)
    if block_range is None:
        if block_index is None:
            block_index = torch.tensor(
                block_ids,
                dtype=torch.long,
                device=kv_cache.device,
            )
        src = kv_cache.index_select(0, block_index)
    else:
        start, stop = block_range
        src = kv_cache[start:stop]
    dst = staging_by_layer.view(kv_cache.dtype).view(
        num_slots,
        num_layers,
        *src.shape[2:],
    )
    dst.copy_(src)
