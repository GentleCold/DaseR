# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
import time
from typing import TYPE_CHECKING, Any

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
    StoreChunkWrite,
    StoreFuture,
    StoreWriteSpan,
)
from daser.logging import init_logger

logger = init_logger(__name__)

DEFAULT_ROPE_DELTA_SCALE = 1.0


def _profile_enabled() -> bool:
    """Return True when connector micro-profiling is enabled.

    Returns:
        True if DASER_PROFILE_CONNECTOR is set to a truthy value.
    """
    return os.environ.get("DASER_PROFILE_CONNECTOR", "").lower() in {
        "1",
        "true",
        "yes",
    }


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
            kv_tensor.index_copy_(1, block_index, src.movedim(0, 1))
        else:
            sample = kv_tensor[block_ids[0]]
            src = (
                staging_by_layer[:, layer_idx, :]
                .view(kv_tensor.dtype)
                .view(num_slots, *sample.shape)
            )
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


def _copy_host_staging_to_contiguous_kv_cache(
    staging: torch.Tensor,
    kv_caches: dict[str, torch.Tensor],
    layer_names: list[str],
    block_ids: list[int],
    slot_size: int,
) -> int:
    """Copy host slot-major staging into contiguous KV cache block ranges.

    This fast path avoids an intermediate GPU staging tensor when the target
    vLLM block IDs are contiguous. It is intended for pinned host L1 hits.

    Args:
        staging: Contiguous CPU uint8 tensor containing whole request KV bytes.
        kv_caches: Per-layer vLLM KV cache tensors.
        layer_names: Layer iteration order matching storage layout.
        block_ids: contiguous vLLM KV block IDs to write.
        slot_size: Total bytes for all layers in one slot.

    Returns:
        Number of layer-level copy operations issued, or 0 if block IDs are not
        contiguous.

    Async/thread-safety:
        Synchronous host-to-device copies on the vLLM worker thread.
    """
    if not block_ids or not layer_names:
        return 0
    first_block = block_ids[0]
    if block_ids != list(range(first_block, first_block + len(block_ids))):
        return 0
    num_layers = len(layer_names)
    layer_size = slot_size // num_layers
    num_slots = len(block_ids)
    staging_by_layer = staging.view(num_slots, num_layers, layer_size)

    copies = 0
    for layer_idx, layer_name in enumerate(layer_names):
        kv_tensor = kv_caches.get(layer_name)
        if kv_tensor is None:
            continue
        if kv_tensor.dim() >= 2:
            dst = kv_tensor[:, first_block : first_block + num_slots]
            src = (
                staging_by_layer[:, layer_idx, :]
                .view(kv_tensor.dtype)
                .view(num_slots, *dst[:, 0].shape)
                .movedim(0, 1)
            )
        else:
            dst = kv_tensor[first_block : first_block + num_slots]
            src = (
                staging_by_layer[:, layer_idx, :]
                .view(kv_tensor.dtype)
                .view(num_slots, *dst[0].shape)
            )
        dst.copy_(src, non_blocking=True)
        copies += 1
    return copies


def _merge_load_staging(
    staging_and_blocks: list[tuple[torch.Tensor, list[int]]],
    slot_size: int,
) -> tuple[torch.Tensor, list[int]]:
    """Merge per-request staging tensors into one block-ordered tensor.

    Args:
        staging_and_blocks: pairs of slot-major staging tensor and block IDs.
        slot_size: bytes per slot.

    Returns:
        Merged staging tensor and matching block IDs sorted by block ID.

    Async/thread-safety:
        Synchronous tensor copies on the caller's thread.
    """
    items: list[tuple[int, torch.Tensor]] = []
    for staging, block_ids in staging_and_blocks:
        for slot_idx, block_id in enumerate(block_ids):
            start = slot_idx * slot_size
            end = start + slot_size
            items.append((block_id, staging[start:end]))
    items.sort(key=lambda item: item[0])
    if not items:
        raise ValueError("cannot merge empty load staging")
    merged = torch.empty(
        len(items) * slot_size,
        dtype=torch.uint8,
        device=items[0][1].device,
    )
    block_ids = []
    for idx, (block_id, slot) in enumerate(items):
        start = idx * slot_size
        merged[start : start + slot_size].copy_(slot)
        block_ids.append(block_id)
    return merged, block_ids


def _copy_kv_cache_to_staging(
    staging: torch.Tensor,
    kv_layer: torch.Tensor,
    layer_idx: int,
    block_ids: list[int],
    num_layers: int,
    slot_size: int,
    block_index: torch.Tensor | None = None,
) -> None:
    """Copy one vLLM KV layer for all requested blocks into staging.

    Args:
        staging: Contiguous uint8 tensor with slot-major DaseR layout.
        kv_layer: vLLM KV cache tensor for one attention layer.
        layer_idx: Index of ``kv_layer`` in the DaseR on-disk layer order.
        block_ids: vLLM KV block IDs to persist.
        num_layers: Total number of KV layers in the model.
        slot_size: Total bytes for all layers in one slot.
        block_index: Optional precomputed index tensor for ``block_ids``.

    Async/thread-safety:
        Synchronous GPU tensor copies on the vLLM worker thread.
    """
    if not block_ids:
        return
    layer_size = slot_size // num_layers
    num_slots = len(block_ids)
    staging_by_layer = staging.view(num_slots, num_layers, layer_size)
    first_block = block_ids[0]
    contiguous_blocks = block_ids == list(range(first_block, first_block + num_slots))
    if kv_layer.dim() >= 2:
        if contiguous_blocks:
            src = kv_layer[:, first_block : first_block + num_slots].movedim(1, 0)
        else:
            index = block_index
            if index is None:
                index = torch.tensor(
                    block_ids,
                    dtype=torch.long,
                    device=kv_layer.device,
                )
            src = kv_layer.index_select(1, index).movedim(1, 0)
    else:
        if contiguous_blocks:
            src = kv_layer[first_block : first_block + num_slots]
        else:
            index = block_index
            if index is None:
                index = torch.tensor(
                    block_ids,
                    dtype=torch.long,
                    device=kv_layer.device,
                )
            src = kv_layer.index_select(0, index)
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


def _build_store_chunk_writes(
    reqs_to_store: dict[str, ReqStoreSpec],
    req_slot_ranges: dict[str, tuple[int, int]],
    slot_size: int,
) -> list[StoreChunkWrite]:
    """Build immutable chunk-write descriptors for asynchronous stores.

    Args:
        reqs_to_store: Store specs keyed by request ID.
        req_slot_ranges: Request ID to ``(start_slot_index, end_slot_index)``.
        slot_size: DaseR bytes per KV slot.

    Returns:
        Chunk write descriptors captured before worker metadata is cleared.

    Async/thread-safety:
        Pure CPU helper. It does not mutate connector state.
    """
    writes: list[StoreChunkWrite] = []
    for req_id, spec in reqs_to_store.items():
        slot_range = req_slot_ranges.get(req_id)
        if slot_range is None:
            continue
        start, end = slot_range
        if end <= start:
            continue
        writes.append(
            StoreChunkWrite(
                chunk_key=spec.chunk_key,
                source_offset=start * slot_size,
                nbytes=(end - start) * slot_size,
                file_offset=spec.file_offset,
            )
        )
    return writes


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

        self._ensure_transfer_ready()

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        super().bind_connector_metadata(connector_metadata)
        self._meta = connector_metadata
        self._pin_local_lookup_chunks(connector_metadata)
        self._reap_store_futures(block=False)
        self._clear_save_staging()
        slot_cursor = 0
        for req_id, spec in connector_metadata.reqs_to_store.items():
            num_slots = len(spec.block_ids)
            if num_slots == 0:
                continue
            self._save_req_slot_ranges[req_id] = (slot_cursor, slot_cursor + num_slots)
            self._save_all_block_ids.extend(spec.block_ids)
            slot_cursor += num_slots
        self._throttle_store_futures(slot_cursor * self._slot_size)

    def clear_connector_metadata(self) -> None:
        """Clear metadata after forward pass completes."""
        super().clear_connector_metadata()
        self._meta = None

    def _pin_local_lookup_chunks(self, connector_metadata: DaserConnectorMeta) -> None:
        """Protect lookup-hit L1 entries before store throttling can evict them.

        Args:
            connector_metadata: metadata containing load specs for this step.
        """
        if not connector_metadata.reqs_to_load:
            return
        transfer = self._transfer
        if transfer is None:
            return
        transfer.pin_chunks_for_lookup(
            [spec.chunk_key for spec in connector_metadata.reqs_to_load.values()]
        )

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

        per_req: list[tuple[torch.Tensor, ReqLoadSpec]] = []
        coros: list = []
        sample_tensor = next(iter(self._kv_caches.values()), None)
        if sample_tensor is None:
            return
        transfer = self._transfer
        if transfer is None:
            raise RuntimeError("transfer layer is not initialized")

        for spec in self._meta.reqs_to_load.values():
            num_slots = len(spec.block_ids)
            if num_slots == 0:
                continue
            total_bytes = num_slots * self._slot_size
            staging = torch.empty(
                total_bytes, dtype=torch.uint8, device=sample_tensor.device
            )

            async def _read(
                off: int = spec.start_slot * self._slot_size,
                nb: int = total_bytes,
                load_spec: ReqLoadSpec = spec,
                torch_staging: torch.Tensor = staging,
            ) -> tuple[str, torch.Tensor, ReqLoadSpec]:
                await transfer.read_chunk_into_async(
                    chunk_key=load_spec.chunk_key,
                    buf=torch_staging,
                    file_offset=off,
                    nbytes=nb,
                    l2_durable=load_spec.l2_durable,
                    protect_lookup=False,
                )
                await self._ipc_async.release_chunks([load_spec.chunk_key])
                transfer.release_lookup_pins([load_spec.chunk_key])
                return ("gpu", torch_staging, load_spec)

            coros.append(_read())

        if not coros:
            return

        async def _run_all(cs: list) -> list:
            return await asyncio.gather(*cs)

        prof = _profile_enabled()
        t_read0 = time.perf_counter() if prof else 0.0
        read_results = asyncio.run_coroutine_threadsafe(
            _run_all(coros), self._bg_loop
        ).result(timeout=120.0)
        t_read1 = time.perf_counter() if prof else 0.0
        for _, staging, spec in read_results:
            per_req.append((staging, spec))

        total_copies = 0
        t_direct0 = time.perf_counter() if prof else 0.0
        t_direct1 = time.perf_counter() if prof else 0.0
        mergeable: list[tuple[torch.Tensor, list[int]]] = []
        fallback_reqs: list[tuple[torch.Tensor, ReqLoadSpec]] = []
        for staging, spec in per_req:
            if self._can_merge_load_copy(spec):
                mergeable.append((staging, spec.block_ids))
            else:
                fallback_reqs.append((staging, spec))
        if mergeable:
            t_merge0 = time.perf_counter() if prof else 0.0
            merged_staging, merged_block_ids = _merge_load_staging(
                mergeable,
                slot_size=self._slot_size,
            )
            total_copies += _copy_staging_to_kv_cache(
                staging=merged_staging,
                kv_caches=self._kv_caches,
                layer_names=self._layer_names,
                block_ids=merged_block_ids,
                slot_size=self._slot_size,
            )
            t_merge1 = time.perf_counter() if prof else 0.0
        else:
            t_merge0 = t_merge1 = time.perf_counter() if prof else 0.0
        t_fallback0 = time.perf_counter() if prof else 0.0
        for staging, spec in fallback_reqs:
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
        t_fallback1 = time.perf_counter() if prof else 0.0
        if prof:
            logger.info(
                "[PROFILE] load reqs=%d host=%d gpu=%d read=%.6fs "
                "direct=%.6fs merge=%.6fs fallback=%.6fs copies=%d",
                len(read_results),
                0,
                len(per_req),
                t_read1 - t_read0,
                t_direct1 - t_direct0,
                t_merge1 - t_merge0,
                t_fallback1 - t_fallback0,
                total_copies,
            )

        logger.debug(
            "[CONNECTOR] start_load_kv: %d reqs, %d GPU copies, %d GDS reads",
            len(per_req),
            total_copies,
            len(coros),
        )

    def _can_direct_host_load(self, spec: ReqLoadSpec) -> bool:
        """Return True when host L1 bytes can copy directly into KV cache.

        Args:
            spec: load specification.

        Returns:
            True for contiguous block IDs without load-time transforms.
        """
        block_ids = spec.block_ids
        if not block_ids:
            return False
        return (
            block_ids == list(range(block_ids[0], block_ids[0] + len(block_ids)))
            and self._load_key_scale == 1.0
            and self._load_value_scale == 1.0
            and not (spec.pos_offset and self._rope_rotary_dim > 0)
        )

    def _can_merge_load_copy(self, spec: ReqLoadSpec) -> bool:
        """Return True when a load can join the batched copy-out path.

        Args:
            spec: load specification.

        Returns:
            True when no load-time transforms are required.
        """
        return (
            self._load_key_scale == 1.0
            and self._load_value_scale == 1.0
            and not (spec.pos_offset and self._rope_rotary_dim > 0)
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
        if not self._ensure_transfer_ready():
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
            first_block = self._save_all_block_ids[0]
            contiguous_blocks = self._save_all_block_ids == list(
                range(first_block, first_block + len(self._save_all_block_ids))
            )
            if not contiguous_blocks:
                self._save_block_index = torch.tensor(
                    self._save_all_block_ids,
                    dtype=torch.long,
                    device=kv_layer.device,
                )

        prof = _profile_enabled()
        t_copy0 = time.perf_counter() if prof else 0.0
        _copy_kv_cache_to_staging(
            staging=self._save_step_staging,
            kv_layer=kv_layer,
            layer_idx=self._layer_idx_map[layer_name],
            block_ids=self._save_all_block_ids,
            num_layers=num_layers,
            slot_size=self._slot_size,
            block_index=self._save_block_index,
        )
        if prof:
            logger.info(
                "[PROFILE] save layer=%s blocks=%d pack=%.6fs",
                layer_name,
                len(self._save_all_block_ids),
                time.perf_counter() - t_copy0,
            )

    def wait_for_save(self) -> None:
        """Submit save staging writes and commit them after IO completes."""
        if self._meta is None or self._save_step_staging is None:
            self._reap_store_futures(block=False)
            return

        staging = self._save_step_staging
        chunk_writes = _build_store_chunk_writes(
            reqs_to_store=self._meta.reqs_to_store,
            req_slot_ranges=self._save_req_slot_ranges,
            slot_size=self._slot_size,
        )

        if chunk_writes:
            prof = _profile_enabled()
            t_submit0 = time.perf_counter() if prof else 0.0
            future = asyncio.run_coroutine_threadsafe(
                self._write_and_commit(staging, chunk_writes),
                self._bg_loop,
            )
            if prof:
                logger.info(
                    "[PROFILE] save submit chunks=%d submit=%.6fs",
                    len(chunk_writes),
                    time.perf_counter() - t_submit0,
                )
            self._store_futures.append(
                StoreFuture(future=future, staging=staging, nbytes=staging.nbytes)
            )
            self._inflight_store_bytes += staging.nbytes

        self._clear_save_staging()
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
        if self._transfer is not None:
            self._transfer.close()
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)

    def _run_bg_loop(self) -> None:
        """Run the background asyncio IO loop."""
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()

    def _ensure_transfer_ready(self) -> bool:
        """Initialize the worker transfer backend once the server store is ready."""
        if self._transfer is not None:
            return True

        self._refresh_runtime_config()
        if not self._store_path or not os.path.exists(self._store_path):
            logger.warning(
                "[CONNECTOR] store file %s is not ready; start DaseR server "
                "before sending requests",
                self._store_path,
            )
            return False

        self._transfer = self._build_transfer_layer()
        logger.info(
            "[CONNECTOR] transfer backend=%s", self._transfer.backend_name.value
        )
        return True

    def _clear_save_staging(self) -> None:
        """Clear worker-side per-forward save staging state."""
        self._save_all_block_ids = []
        self._save_block_index = None
        self._save_req_slot_ranges = {}
        self._save_step_staging = None

    async def _write_and_commit(
        self,
        save_staging: torch.Tensor,
        chunk_writes: list[StoreChunkWrite],
    ) -> None:
        """Write staged KV spans and publish chunks after IO completes.

        Args:
            chunk_writes: Chunk write descriptors captured before metadata
                cleanup.
            save_staging: Step staging tensor accepted by the transfer layer.
        """
        transfer = self._transfer
        if transfer is None:
            raise RuntimeError("transfer layer is not initialized")

        prof = _profile_enabled()
        t_write0 = time.perf_counter() if prof else 0.0
        for write in chunk_writes:
            await transfer.write_chunk_async(
                chunk_key=write.chunk_key,
                buf=save_staging[
                    write.source_offset : write.source_offset + write.nbytes
                ],
                file_offset=write.file_offset,
                nbytes=write.nbytes,
            )
        if prof:
            logger.info(
                "[PROFILE] save write_chunk chunks=%d bytes=%d elapsed=%.6fs",
                len(chunk_writes),
                sum(write.nbytes for write in chunk_writes),
                time.perf_counter() - t_write0,
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
