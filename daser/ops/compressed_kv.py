# SPDX-License-Identifier: Apache-2.0

"""Fused strict-lossless decode from GPU staging into cross-layer KV cache."""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray
import torch

from daser.compression.format import (
    CODEBOOK_ENTRIES,
    IO_ALIGNMENT,
    CompressedStoreGeometry,
    SlotMode,
    digest_bytes,
)
from daser.logging import init_logger

_SLOT_HEADER_BYTES = IO_ALIGNMENT
# Bound each online codec launch so a large first-turn prefix yields to vLLM
# between batches.  The raw staging lease remains unchanged; this only limits
# the amount of GPU work submitted by one pack launch.
# Keep enough slots per codec launch to amortize fixed metadata and kernel
# overhead while staying below the staging lease size used by the worker.
# The batch is bounded at the connector layer, so increasing this value does
# not change the persisted slot geometry or transfer ownership.
_DEFAULT_ONLINE_PACK_BATCH_SLOTS = 85


def _online_pack_batch_slots_from_env() -> int:
    """Resolve the bounded online codec batch size for this worker process.

    ``DASER_ONLINE_PACK_BATCH_SLOTS`` is intentionally capped by the tested
    default so a benchmark can screen smaller launch quanta without changing
    the staging lease geometry. Invalid values keep the production default.
    """
    raw_value = os.environ.get("DASER_ONLINE_PACK_BATCH_SLOTS")
    if raw_value is None:
        return _DEFAULT_ONLINE_PACK_BATCH_SLOTS
    try:
        value = int(raw_value)
    except ValueError:
        return _DEFAULT_ONLINE_PACK_BATCH_SLOTS
    return min(max(1, value), _DEFAULT_ONLINE_PACK_BATCH_SLOTS)


ONLINE_PACK_BATCH_SLOTS = _online_pack_batch_slots_from_env()
logger = init_logger(__name__)


def _fixed_envelope_geometry(
    *, slot_stride: int, num_planes: int, plane_scalars: int, max_tiles: int
) -> tuple[int, int, int] | None:
    """Return the online 3-bit plane envelope and its escape capacity.

    Online packing uses the seven representable entries of the static
    codebook and emits all remaining high bytes through the escape stream.
    Sizing the fixed envelope with the 3-bit symbol stream leaves the full
    capacity available to escapes instead of retaining the historical 4-bit
    admission threshold. Plane records are byte-addressed inside the slot and
    are never submitted as independent IO requests, so the envelope does not
    round each plane to 4 KiB. This recovers the alignment remainder for
    escape bytes and avoids an unnecessary raw fallback near the capacity
    boundary.
    """
    payload_base = plane_scalars + (plane_scalars * 3 + 7) // 8 + 8 * (max_tiles + 1)
    available = slot_stride - IO_ALIGNMENT
    if available <= 0 or num_planes <= 0:
        return None
    # Only the complete slot is aligned. The persisted format and TileLang
    # scratch both address plane records by byte offset, so per-plane alignment
    # would strand up to ``num_planes * (IO_ALIGNMENT - 1)`` usable bytes.
    plane_bytes = available // num_planes
    fixed_escape = plane_bytes - payload_base
    if fixed_escape <= 0:
        return None
    return plane_bytes, fixed_escape, IO_ALIGNMENT + num_planes * plane_bytes


def _fixed_single_read_geometry(
    *, slot_stride: int, num_planes: int, plane_scalars: int, max_tiles: int
) -> tuple[int, int, int] | None:
    """Return scratch parameters for contiguous per-plane escape prefixes.

    Payload emission is fused into the final staging write, so intermediate
    storage contains only the two uint32 prefix tables. The raw fallback
    envelope remains independent of this compact internal scratch layout.
    """
    envelope = _fixed_envelope_geometry(
        slot_stride=slot_stride,
        num_planes=num_planes,
        plane_scalars=plane_scalars,
        max_tiles=max_tiles,
    )
    if envelope is None or max_tiles <= 0:
        return None
    _plane_bytes, fixed_escape, _stored_length = envelope
    tile_capacity = (fixed_escape + max_tiles - 1) // max_tiles
    # Scratch retains token and raw-escape prefixes for the packed 3-bit
    # stream. The persisted 4-bit fallback still copies only its first table.
    scratch_plane_bytes = 8 * (max_tiles + 1)
    scratch_slot_bytes = num_planes * scratch_plane_bytes
    return tile_capacity, scratch_plane_bytes, scratch_slot_bytes


def warm_fused_online_kv_packer(
    kv_cache: torch.Tensor, max_slots_per_buffer: int, tile_scalars: int = 1024
) -> None:
    """Compile and launch the online packer before serving traffic.

    Args:
        kv_cache: Worker-owned contiguous BF16 KV cache.
        max_slots_per_buffer: Maximum packed records in one store staging lease.
        tile_scalars: Dynamic codec tile size.

    Raises:
        ValueError: If the cache geometry is unsupported.
        RuntimeError: If the CUDA online pack kernel cannot compile or launch
            the representative kernel.
    """
    _warm_tilelang_codec(kv_cache, max_slots_per_buffer, tile_scalars)
    return


def _warm_tilelang_codec(
    kv_cache: torch.Tensor, max_slots_per_buffer: int, tile_scalars: int
) -> None:
    """Compile and launch representative TileLang store/load kernels.

    This runs during vLLM KV-cache registration.  The representative launch
    uses one slot, while the compiled kernels keep model geometry static and
    accept request slot counts and offsets as runtime tensors.  No request
    path can therefore trigger TileLang JIT compilation.
    """
    from daser.ops.tilelang_compressed_kv import (
        compile_load_kernel,
        compile_store_kernels,
    )

    if (
        kv_cache.device.type != "cuda"
        or kv_cache.dim() != 6
        or not kv_cache.is_contiguous()
        or kv_cache.dtype is not torch.bfloat16
    ):
        raise ValueError("online packer requires contiguous CUDA BF16 KV cache")
    if max_slots_per_buffer <= 0 or tile_scalars <= 0:
        raise ValueError("online packer geometry must be positive")
    num_planes = int(kv_cache.shape[1]) * 2
    plane_scalars = int(np.prod(kv_cache.shape[3:]))
    block_tokens = int(kv_cache.shape[3])
    num_blocks = int(kv_cache.shape[0])
    store = compile_store_kernels(
        num_blocks=num_blocks,
        num_planes=num_planes,
        plane_scalars=plane_scalars,
        tile_scalars=tile_scalars,
    )
    loads = tuple(
        compile_load_kernel(
            num_blocks=num_blocks,
            num_planes=num_planes,
            plane_scalars=plane_scalars,
            tile_scalars=tile_scalars,
            staging_bytes=int(kv_cache[0].nbytes),
            fanout=fanout,
            online=True,
        )
        for fanout in (False, True)
    )
    max_tiles = (plane_scalars + tile_scalars - 1) // tile_scalars
    envelope = _fixed_envelope_geometry(
        slot_stride=int(kv_cache[0].nbytes),
        num_planes=num_planes,
        plane_scalars=plane_scalars,
        max_tiles=max_tiles,
    )
    single_read = _fixed_single_read_geometry(
        slot_stride=int(kv_cache[0].nbytes),
        num_planes=num_planes,
        plane_scalars=plane_scalars,
        max_tiles=max_tiles,
    )
    if envelope is None or single_read is None:
        raise ValueError("online packer cannot fit a fixed TileLang envelope")
    plane_record_bytes, fixed_escape_bytes, _ = envelope
    tile_escape_capacity, scratch_plane_bytes, scratch_slot_stride = single_read
    device = kv_cache.device
    with torch.cuda.device(device), torch.inference_mode(False):
        ids = torch.zeros(1, dtype=torch.int32, device=device)
        logical_ids = torch.zeros(1, dtype=torch.int64, device=device)
        lookup = torch.zeros(num_planes * 256, dtype=torch.uint8, device=device)
        codebook_hash = torch.zeros(32, dtype=torch.uint8, device=device)
        bits = kv_cache.view(torch.uint16).reshape(
            num_blocks, num_planes, plane_scalars
        )
        scratch = torch.empty(scratch_slot_stride, dtype=torch.uint8, device=device)
        max_tiles = (plane_scalars + tile_scalars - 1) // tile_scalars
        totals = torch.zeros(num_planes, dtype=torch.uint32, device=device)
        raw_totals = torch.zeros(num_planes, dtype=torch.uint32, device=device)
        overflow = torch.zeros(1, dtype=torch.uint32, device=device)
        source_offsets = torch.zeros(num_planes, dtype=torch.int64, device=device)
        destination_offsets = torch.zeros(num_planes, dtype=torch.int64, device=device)
        payload_bytes = torch.zeros(num_planes, dtype=torch.int64, device=device)
        slot_offsets = torch.zeros(1, dtype=torch.int64, device=device)
        symbol_bits = torch.zeros(1, dtype=torch.int32, device=device)
        output = torch.empty(int(kv_cache[0].nbytes), dtype=torch.uint8, device=device)
        max_escape_words = (plane_scalars * 3 + 31) // 32
        escape_words = torch.zeros(
            num_planes * max_escape_words, dtype=torch.uint32, device=device
        )
        # The representative load launch targets block zero because the
        # TileLang destination shape is the production KV cache. Preserve the
        # live cache value across warmup so registration cannot alter the first
        # request's keys/values.
        saved_block = kv_cache[0].clone()
        stream = torch.cuda.current_stream(device)
        store.encode(
            bits,
            ids,
            lookup,
            totals,
            raw_totals,
            overflow,
            scratch,
            block_tokens,
            block_tokens,
            fixed_escape_bytes,
            scratch_plane_bytes,
        )
        store.layout(
            totals,
            raw_totals,
            overflow,
            symbol_bits,
            source_offsets,
            destination_offsets,
            payload_bytes,
            slot_offsets,
            int(kv_cache[0].nbytes),
            scratch_plane_bytes,
            scratch_slot_stride,
        )
        store.compact(
            bits,
            ids,
            logical_ids,
            overflow,
            output,
            escape_words,
            destination_offsets,
            payload_bytes,
            totals,
            raw_totals,
            slot_offsets,
            symbol_bits,
            lookup,
            codebook_hash,
            scratch,
            tile_escape_capacity,
            scratch_plane_bytes,
            block_tokens,
            block_tokens,
            int(kv_cache[0].nbytes),
        )
        for load in loads:
            load(
                output,
                slot_offsets,
                ids,
                torch.arange(2, dtype=torch.int32, device=device),
                # Exercise both precompiled variants before traffic. Actual
                # source/destination counts remain runtime inputs. The live
                # cache block is restored after synchronized warmup.
                torch.ones(1, dtype=torch.int32, device=device),
                torch.zeros(
                    num_planes * CODEBOOK_ENTRIES, dtype=torch.uint8, device=device
                ),
                bits,
            )
        stream.synchronize()
        kv_cache[0].copy_(saved_block)


@dataclass(frozen=True)
class OnlinePackedSlot:
    """Metadata for one packed payload written into a staging buffer."""

    logical_slot: int
    mode: SlotMode
    source_offset: int
    stored_length: int


class FusedOnlineKVPacker:
    """Encode live KV blocks directly into a CUDA store staging buffer."""

    def __init__(
        self,
        *,
        kv_cache: torch.Tensor,
        codebooks: bytes,
        tile_scalars: int,
        max_slots_per_buffer: int,
    ) -> None:
        if (
            kv_cache.device.type != "cuda"
            or kv_cache.dim() != 6
            or not kv_cache.is_contiguous()
            or kv_cache.dtype is not torch.bfloat16
        ):
            raise ValueError("online packer requires contiguous CUDA BF16 KV cache")
        if tile_scalars <= 0 or max_slots_per_buffer <= 0:
            raise ValueError("online packer geometry must be positive")
        self._device = kv_cache.device
        self._num_blocks = int(kv_cache.shape[0])
        self._num_planes = int(kv_cache.shape[1]) * 2
        self._plane_scalars = int(np.prod(kv_cache.shape[3:]))
        self._tile_scalars = int(tile_scalars)
        self._max_slots = int(max_slots_per_buffer)
        self._geometry = CompressedStoreGeometry(
            num_slots=self._num_blocks,
            slot_size=int(kv_cache[0].nbytes),
            block_tokens=int(kv_cache.shape[3]),
            num_layers=int(kv_cache.shape[1]),
            num_kv_heads=int(kv_cache.shape[4]),
            head_dim=int(kv_cache.shape[5]),
            tile_scalars=self._tile_scalars,
        )
        if len(codebooks) != self._num_planes * CODEBOOK_ENTRIES:
            raise ValueError("online codebooks do not match KV geometry")
        self._codebook_hash = digest_bytes(codebooks)
        self._kv_bits = kv_cache.view(torch.uint16).reshape(
            self._num_blocks, self._num_planes, self._plane_scalars
        )
        lookup: NDArray[np.uint8] = np.full(
            (self._num_planes, 256), CODEBOOK_ENTRIES, dtype=np.uint8
        )
        tables = np.frombuffer(codebooks, dtype=np.uint8).reshape(
            self._num_planes, CODEBOOK_ENTRIES
        )
        # The main online wire stream represents entries 0..6. Entries 7..13
        # use three-bit secondary escape tokens; entry 14 and values outside
        # the immutable table use the raw-byte sentinel in that stream.
        lookup[np.arange(self._num_planes)[:, None], tables] = np.arange(
            CODEBOOK_ENTRIES, dtype=np.uint8
        )
        self._lookup = torch.from_numpy(lookup.reshape(-1).copy()).to(self._device)
        from daser.ops.tilelang_compressed_kv import compile_store_kernels

        self._tilelang_store = compile_store_kernels(
            num_blocks=self._num_blocks,
            num_planes=self._num_planes,
            plane_scalars=self._plane_scalars,
            tile_scalars=self._tile_scalars,
        )
        # Store uses the TileLang bundle exclusively. Raw fallback records are
        # emitted by the compact stage when a packed record overflows.
        self._diagnostic_emitted = False
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        # Online records use only three-bit escape symbols. The ceiling also
        # includes the last word of a symbol split across a word boundary.
        self._max_escape_words = (self._plane_scalars * 3 + 31) // 32
        envelope = _fixed_envelope_geometry(
            slot_stride=self._geometry.slot_size,
            num_planes=self._num_planes,
            plane_scalars=self._plane_scalars,
            max_tiles=max_tiles,
        )
        if envelope is None or max_tiles > 1024:
            raise ValueError(
                "compressed-online requires a fixed TileLang envelope for "
                f"slot_size={self._geometry.slot_size}, planes={self._num_planes}, "
                f"plane_scalars={self._plane_scalars}, "
                f"tile_scalars={self._tile_scalars}"
            )
        (
            self._fixed_plane_record_bytes,
            self._fixed_escape_bytes,
            self._fixed_stored_length,
        ) = envelope
        single_read_geometry = _fixed_single_read_geometry(
            slot_stride=self._geometry.slot_size,
            num_planes=self._num_planes,
            plane_scalars=self._plane_scalars,
            max_tiles=max_tiles,
        )
        if single_read_geometry is None:
            raise ValueError(
                "compressed-online requires single-read TileLang scratch geometry"
            )
        (
            self._fixed_tile_escape_capacity,
            self._fixed_scratch_plane_record_bytes,
            self._fixed_scratch_slot_stride,
        ) = single_read_geometry
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        total_count = self._max_slots * self._num_planes
        with torch.inference_mode(False):
            self._host_block_ids = torch.empty(
                self._max_slots, dtype=torch.int32, pin_memory=True
            )
            self._host_logical_slots = torch.empty(
                self._max_slots, dtype=torch.int64, pin_memory=True
            )
            # Planning stays device-resident until the compact per-plane
            # totals are known. The host only needs those totals to return
            # variable-length spans; the fused compact stage serializes
            # descriptors directly into each staging header.
            self._host_totals = torch.empty(
                total_count, dtype=torch.uint32, pin_memory=True
            )
            self._host_raw_totals = torch.empty(
                total_count, dtype=torch.uint32, pin_memory=True
            )
            self._host_overflow = torch.empty(
                self._max_slots, dtype=torch.uint32, pin_memory=True
            )
            self._device_block_ids = torch.empty(
                self._max_slots, dtype=torch.int32, device=self._device
            )
            self._device_logical_slots = torch.empty(
                self._max_slots, dtype=torch.int64, device=self._device
            )
            self._device_codebook_hash = torch.tensor(
                list(self._codebook_hash), dtype=torch.uint8, device=self._device
            )
            self._device_totals = torch.empty(
                total_count, dtype=torch.uint32, device=self._device
            )
            self._device_raw_totals = torch.empty(
                total_count, dtype=torch.uint32, device=self._device
            )
            self._device_overflow = torch.empty(
                self._max_slots, dtype=torch.uint32, device=self._device
            )
            self._device_slot_offsets = torch.empty(
                self._max_slots, dtype=torch.int64, device=self._device
            )
            self._host_symbol_bits = torch.empty(
                self._max_slots, dtype=torch.int32, pin_memory=True
            )
            self._device_symbol_bits = torch.empty(
                self._max_slots, dtype=torch.int32, device=self._device
            )
            # The production store path caps a codec launch at
            # ``ONLINE_PACK_BATCH_SLOTS``. Allocate its compact prefix scratch
            # before request traffic. Payload bytes go directly to staging;
            # reserving their old envelope here would leave large unused holes
            # between prefix tables and consume inference memory needlessly.
            fixed_scratch_slots = min(self._max_slots, ONLINE_PACK_BATCH_SLOTS)
            self._fixed_scratch: torch.Tensor | None
            if self._fixed_scratch_slot_stride > 0 and fixed_scratch_slots > 0:
                self._fixed_scratch = torch.empty(
                    fixed_scratch_slots * self._fixed_scratch_slot_stride,
                    dtype=torch.uint8,
                    device=self._device,
                )
                self._fixed_scratch_slots = fixed_scratch_slots
            else:
                self._fixed_scratch = None
                self._fixed_scratch_slots = 0
            compact_count = self._max_slots * self._num_planes
            self._device_compact_source = torch.empty(
                compact_count, dtype=torch.int64, device=self._device
            )
            self._device_compact_destination = torch.empty(
                compact_count, dtype=torch.int64, device=self._device
            )
            self._device_compact_bytes = torch.empty(
                compact_count, dtype=torch.int64, device=self._device
            )
            self._device_escape_words = torch.empty(
                self._max_slots * self._num_planes * self._max_escape_words,
                dtype=torch.uint32,
                device=self._device,
            )

    def pack_into(
        self,
        *,
        staging: torch.Tensor,
        block_ids: list[int],
        logical_slots: list[int],
        slot_stride: int,
        stream: torch.cuda.Stream,
        valid_token_count: int | None = None,
    ) -> list[OnlinePackedSlot]:
        """Pack selected live blocks into fixed-stride staging regions.

        Args:
            staging: Worker-owned CUDA byte staging buffer.
            block_ids: Physical vLLM blocks in logical order.
            logical_slots: DaseR slot IDs corresponding to ``block_ids``.
            slot_stride: Raw rank-local slot bytes and staging region stride.
            stream: Store CUDA stream ordering the pack and metadata copies.
            valid_token_count: Optional valid token extent for a partial tail.

        Returns:
            Actual mode, source offset, and stored length for every slot.
        """
        if not block_ids or len(block_ids) != len(logical_slots):
            raise ValueError("online packer block metadata is invalid")
        if len(block_ids) > self._max_slots or slot_stride != self._geometry.slot_size:
            raise ValueError("online packer staging geometry is invalid")
        if staging.device != self._device or staging.dtype is not torch.uint8:
            raise ValueError("online packer staging must be a CUDA byte tensor")
        if staging.numel() < len(block_ids) * slot_stride:
            raise ValueError("online packer staging capacity is insufficient")
        if any(block < 0 or block >= self._num_blocks for block in block_ids):
            raise ValueError("online packer block ID is invalid")
        slot_count = len(block_ids)
        effective_valid_tokens = (
            slot_count * self._geometry.block_tokens
            if valid_token_count is None
            else int(valid_token_count)
        )
        if not 0 < effective_valid_tokens <= slot_count * self._geometry.block_tokens:
            raise ValueError("online packer valid token extent is invalid")
        # The fixed writer performs count, prefix construction, and payload
        # emission on the TileLang stream, then compacts only the live payload
        # bytes. The fixed envelope is internal scratch; transfer bytes and
        # on-disk records remain variable-length.
        return self._pack_fixed_into(
            staging=staging,
            block_ids=block_ids,
            logical_slots=logical_slots,
            slot_stride=slot_stride,
            stream=stream,
            valid_token_count=effective_valid_tokens,
        )

    def _pack_fixed_into(
        self,
        *,
        staging: torch.Tensor,
        block_ids: list[int],
        logical_slots: list[int],
        slot_stride: int,
        stream: torch.cuda.Stream,
        valid_token_count: int,
    ) -> list[OnlinePackedSlot]:
        """Pack records through a fixed scratch envelope and compact them.

        Args:
            staging: CUDA byte buffer receiving slot records.
            block_ids: Physical source blocks in logical order.
            logical_slots: DaseR slot IDs written into record headers.
            slot_stride: Raw bytes reserved for one source slot.
            stream: CUDA stream ordering source reads and staging writes.
            valid_token_count: Number of valid prompt tokens in this batch.

        Returns:
            Packed slot metadata with raw mode for overflow records.  Compressed
            records retain their actual variable-length payloads in ``staging``.

        Async/thread-safety:
            Called under the store pipeline staging lock. The stream is
            synchronized once after this method returns by the caller, after
            the host has planned variable record spans.  Header serialization
            is fused into the final TileLang compact launch.
        """
        slot_count = len(block_ids)
        max_tiles = (self._plane_scalars + self._tile_scalars - 1) // self._tile_scalars
        row_count = slot_count * self._num_planes
        with torch.cuda.device(self._device):
            # The fixed writer is intentionally isolated from the exported
            # staging allocation.  Allocate only for the active batch because
            # ``max_slots_per_buffer`` can describe a much larger pool.
            scratch_slot_stride = self._fixed_scratch_slot_stride
            scratch_plane_record_bytes = self._fixed_scratch_plane_record_bytes
            tile_escape_capacity = self._fixed_tile_escape_capacity
            if (
                scratch_slot_stride <= 0
                or scratch_plane_record_bytes <= 0
                or tile_escape_capacity <= 0
            ):
                raise RuntimeError("single-read fixed scratch geometry is invalid")
            if (
                self._fixed_scratch is None
                or self._fixed_scratch_slots < slot_count
                or self._fixed_scratch.numel() < slot_count * scratch_slot_stride
            ):
                self._fixed_scratch = torch.empty(
                    slot_count * scratch_slot_stride,
                    dtype=torch.uint8,
                    device=self._device,
                )
                self._fixed_scratch_slots = slot_count
            scratch = self._fixed_scratch[: slot_count * scratch_slot_stride]

            host_ids = self._host_block_ids[:slot_count]
            host_ids.copy_(torch.as_tensor(block_ids, dtype=torch.int32))
            host_logical_slots = self._host_logical_slots[:slot_count]
            host_logical_slots.copy_(torch.as_tensor(logical_slots, dtype=torch.int64))
            device_ids = self._device_block_ids[:slot_count]
            device_logical_slots = self._device_logical_slots[:slot_count]
            device_totals = self._device_totals[:row_count]
            device_raw_totals = self._device_raw_totals[:row_count]
            device_overflow = self._device_overflow[:slot_count]
            device_sources = self._device_compact_source[:row_count]
            device_destinations = self._device_compact_destination[:row_count]
            device_bytes = self._device_compact_bytes[:row_count]
            device_escape_words = self._device_escape_words[
                : row_count * self._max_escape_words
            ]
            device_slot_offsets = self._device_slot_offsets[:slot_count]
            device_symbol_bits = self._device_symbol_bits[:slot_count]
            with torch.cuda.stream(stream):
                device_ids.copy_(host_ids, non_blocking=True)
                device_logical_slots.copy_(host_logical_slots, non_blocking=True)
                # Keep initialization on the same stream as the writer.  A
                # zero launched on the worker's current stream could race the
                # fixed kernel when the store stream is independent.
                device_overflow.zero_()
                self._tilelang_store.encode(
                    self._kv_bits,
                    device_ids,
                    self._lookup,
                    device_totals,
                    device_raw_totals,
                    device_overflow,
                    scratch,
                    self._geometry.block_tokens,
                    valid_token_count,
                    self._fixed_escape_bytes,
                    scratch_plane_record_bytes,
                )
                # Keep variable-length layout planning on the same CUDA
                # stream as the writer.  This lets compaction start
                # immediately after the single-read kernel instead of waiting
                # for Python to consume totals and copy descriptor arrays back
                # to the device.
                self._tilelang_store.layout(
                    device_totals,
                    device_raw_totals,
                    device_overflow,
                    device_symbol_bits,
                    device_sources,
                    device_destinations,
                    device_bytes,
                    device_slot_offsets,
                    slot_stride,
                    scratch_plane_record_bytes,
                    scratch_slot_stride,
                )
                self._host_totals[:row_count].copy_(device_totals, non_blocking=True)
                self._host_raw_totals[:row_count].copy_(
                    device_raw_totals, non_blocking=True
                )
                self._host_overflow[:slot_count].copy_(
                    device_overflow, non_blocking=True
                )
                self._host_symbol_bits[:slot_count].copy_(
                    device_symbol_bits, non_blocking=True
                )
                self._tilelang_store.compact(
                    self._kv_bits,
                    device_ids,
                    device_logical_slots,
                    device_overflow,
                    staging,
                    device_escape_words,
                    device_destinations,
                    device_bytes,
                    device_totals,
                    device_raw_totals,
                    device_slot_offsets,
                    device_symbol_bits,
                    self._lookup,
                    self._device_codebook_hash,
                    scratch,
                    tile_escape_capacity,
                    scratch_plane_record_bytes,
                    self._geometry.block_tokens,
                    valid_token_count,
                    slot_stride,
                )

            # Totals and fallback flags are the only metadata needed by the
            # host to return source spans.  The device has already laid out,
            # compacted, and serialized all compressed headers by this point.
            stream.synchronize()
            overflow_host = self._host_overflow[:slot_count].numpy().copy()
            totals_host = self._host_totals[:row_count].numpy()
            raw_totals_host = self._host_raw_totals[:row_count].numpy()
            symbol_bits_host = self._host_symbol_bits[:slot_count].numpy().copy()
            compressed = overflow_host == 0
            if not np.isin(symbol_bits_host[compressed], (3, 4, 5)).all():
                raise RuntimeError("device returned invalid compressed symbol width")
            # Rebuild the device layout in bulk from the completed pinned
            # metadata. A Python loop over every slot/plane holds the GIL
            # after each pack launch; array arithmetic leaves only the small
            # per-slot IPC object construction in Python. Widen before any
            # multiplication or reduction, since wire counts are uint32 but
            # batch lengths and offsets are byte-addressed int64 values.
            plane_totals = totals_host.reshape(slot_count, self._num_planes).astype(
                np.int64
            )
            plane_raw_totals = raw_totals_host.reshape(
                slot_count, self._num_planes
            ).astype(np.int64)
            main_bits = np.where(symbol_bits_host == 4, 4, 3)
            secondary_bits = np.where(symbol_bits_host == 5, 3, 4)
            symbol_lengths = (self._plane_scalars * main_bits + 7) // 8
            prefix_lengths = np.where(symbol_bits_host == 4, 4, 8) * (max_tiles + 1)
            escape_lengths = np.where(
                symbol_bits_host[:, None] == 4,
                plane_raw_totals,
                (plane_totals * secondary_bits[:, None] + 7) // 8 + plane_raw_totals,
            )
            lengths = (
                IO_ALIGNMENT
                + self._num_planes
                * (self._plane_scalars + symbol_lengths + prefix_lengths)
                + escape_lengths.sum(axis=1)
            )
            lengths = (lengths + IO_ALIGNMENT - 1) // IO_ALIGNMENT * IO_ALIGNMENT
            # Overflow rows have no compressed selector contract and must
            # retain their complete raw stride regardless of count contents.
            lengths = np.where(compressed, lengths, slot_stride)
            if np.any(lengths > slot_stride):
                raise RuntimeError("device and host fixed layout planners disagree")
            plans = [
                (SlotMode.COMPRESSED if is_compressed else SlotMode.RAW, int(length))
                for is_compressed, length in zip(compressed, lengths, strict=True)
            ]

            # Recompute the compact slot bases for the returned IPC spans. The
            # device planner uses the same slot-major scan; checking capacity
            # here keeps malformed geometry from exposing an out-of-bounds
            # mapping even though payload compaction has already completed.
            slot_bases: NDArray[np.int64] = np.empty(slot_count, dtype=np.int64)
            slot_bases[0] = 0
            np.cumsum(lengths[:-1], out=slot_bases[1:])
            staging_cursor = int(lengths.sum())
            if staging_cursor > slot_count * slot_stride:
                raise RuntimeError("online packed staging layout exceeds capacity")
        compressed_count = sum(mode is SlotMode.COMPRESSED for mode, _ in plans)
        packed_bytes = sum(
            length for mode, length in plans if mode is SlotMode.COMPRESSED
        )
        raw_fallback_count = slot_count - compressed_count
        raw_bytes = slot_count * slot_stride
        stored_bytes = packed_bytes + raw_fallback_count * slot_stride
        savings_pct = (1.0 - stored_bytes / raw_bytes) * 100.0
        escape_bytes = int(plane_totals[compressed].sum())
        logger.debug(
            "[PACK] fixed scratch slots=%d compressed=%d packed_bytes=%d "
            "raw_fallback=%d stored_bytes=%d raw_bytes=%d savings_pct=%.2f "
            "escape_bytes=%d scratch_bytes=%d escape_capacity=%d overflow=%d",
            slot_count,
            compressed_count,
            packed_bytes,
            raw_fallback_count,
            stored_bytes,
            raw_bytes,
            savings_pct,
            escape_bytes,
            self._fixed_scratch_slot_stride * slot_count,
            self._fixed_escape_bytes,
            raw_fallback_count,
        )
        return [
            OnlinePackedSlot(
                logical_slot=int(logical_slots[index]),
                mode=mode,
                source_offset=int(slot_bases[index]),
                stored_length=length,
            )
            for index, (mode, length) in enumerate(plans)
        ]


@dataclass
class _MetadataRing:
    """Persistent pinned/device launch metadata for one staging buffer."""

    host_offsets: torch.Tensor
    host_blocks: torch.Tensor
    host_modes: torch.Tensor
    host_destination_offsets: torch.Tensor
    device_offsets: torch.Tensor
    device_blocks: torch.Tensor
    device_modes: torch.Tensor
    device_destination_offsets: torch.Tensor


class PreparedKVRestore:
    """Submit portions of one immutable restore plan without reuploading metadata.

    Args:
        kernel: Already warmed singleton or fanout TileLang kernel.
        staging: Leased CUDA payload to read until the final completion event.
        offsets: Device offsets for the deduplicated source records.
        blocks: Device destination block IDs, in source-major CSR order.
        destination_offsets: Absolute CSR indices into ``blocks``.
        modes: Device raw/packed tags, one per source.
        codebooks: Fixed device codebooks.
        destination: Registered KV tensor in the kernel's bitwise view.
        fanout: Whether CSR offsets are needed for shared sources.
        stream: Stream on which metadata was uploaded and kernels will launch.

    Async/thread-safety:
        The load thread exclusively owns this plan and its metadata ring. The
        caller must retain the ring and staging lease until a CUDA event after
        the last submission completes, including when abandoning a partial plan.
        Holding tensor references alone does not prevent ring or staging reuse.
    """

    def __init__(
        self,
        *,
        kernel: Any,
        staging: torch.Tensor,
        offsets: torch.Tensor,
        blocks: torch.Tensor,
        destination_offsets: torch.Tensor,
        modes: torch.Tensor,
        codebooks: torch.Tensor,
        destination: torch.Tensor,
        fanout: bool,
        stream: torch.cuda.Stream,
    ) -> None:
        self._kernel = kernel
        self._staging = staging
        self._offsets = offsets
        self._blocks = blocks
        self._destination_offsets = destination_offsets
        self._modes = modes
        self._codebooks = codebooks
        self._destination = destination
        self._fanout = fanout
        self._stream = stream
        self._cursor = 0

    @property
    def remaining_sources(self) -> int:
        """Return unsubmitted sources on the exclusive load owner thread."""
        return self._offsets.numel() - self._cursor

    @property
    def destination_count(self) -> int:
        """Return the total destination slots covered by this immutable plan."""
        return self._blocks.numel()

    def submit_next(self, max_sources: int) -> int:
        """Enqueue at most ``max_sources`` sources on the prepared stream.

        Args:
            max_sources: Positive source count; independent of the IO batch.
        Returns:
            Number of sources submitted, or zero when all were submitted.
        Raises:
            ValueError: If max_sources is not positive.
        Async/thread-safety:
            Called on the exclusive load thread. Does not wait for the GPU or
            enforce queue depth; the caller must observe completion events to
            bound submissions and to keep the parent lease alive on errors.
        """
        if max_sources <= 0:
            raise ValueError("restore segment size must be positive")
        begin = self._cursor
        end = min(begin + max_sources, self._offsets.numel())
        if begin == end:
            return 0
        # CSR indices remain absolute when its row array is sliced, so fanout
        # must retain the full destination vector. Singleton indices are local
        # to the launch and therefore slice their block vector as well.
        blocks = self._blocks if self._fanout else self._blocks[begin:end]
        with torch.cuda.stream(self._stream):
            self._kernel(
                self._staging,
                self._offsets[begin:end],
                blocks,
                self._destination_offsets[begin : end + 1],
                self._modes[begin:end],
                self._codebooks,
                self._destination,
            )
        self._cursor = end
        return end - begin


class FusedCompressedKVDecoder:
    """Own a compiled decoder and persistent metadata for fixed staging slots.

    Args:
        kv_cache: Contiguous cross-layer tensor with layout
            ``[blocks, layers, 2, tokens, heads, dim]``.
        codebooks: Plane-major static 15-entry high-byte tables.
        tile_scalars: Codec tile size encoded in the side index.
        ring_depth: Number of independently leased load staging buffers.
        max_slots_per_buffer: Maximum slot records described by one launch.
        online: Whether all compressed records originate from the online
            three-bit encoder; selected at startup, never from request shape.

    Async/thread-safety:
        Constructed before request traffic. ``decode`` is called only on the
        LoadPipeline thread; metadata rings are independently indexed by the
        staging buffer lease.
    """

    def __init__(
        self,
        *,
        kv_cache: torch.Tensor,
        codebooks: bytes,
        tile_scalars: int,
        ring_depth: int,
        max_slots_per_buffer: int,
        online: bool = False,
    ) -> None:
        if kv_cache.device.type != "cuda" or kv_cache.dim() != 6:
            raise ValueError("compressed restore requires a 6D CUDA KV cache")
        if not kv_cache.is_contiguous() or kv_cache.dtype is not torch.bfloat16:
            raise ValueError("compressed restore requires contiguous BF16 KV cache")
        if ring_depth <= 0 or max_slots_per_buffer <= 0 or tile_scalars <= 0:
            raise ValueError("compressed decoder ring geometry must be positive")
        self._kv_cache = kv_cache
        self._num_blocks = int(kv_cache.shape[0])
        self._num_layers = int(kv_cache.shape[1])
        self._num_planes = self._num_layers * 2
        self._plane_scalars = int(np.prod(kv_cache.shape[3:]))
        self._tile_scalars = tile_scalars
        self._tiles_per_plane = (self._plane_scalars + tile_scalars - 1) // tile_scalars
        expected_codebooks = self._num_planes * CODEBOOK_ENTRIES
        if len(codebooks) != expected_codebooks:
            raise ValueError("compressed codebooks do not match KV layer geometry")
        from daser.ops.tilelang_compressed_kv import compile_load_kernel

        self._kv_bits = kv_cache.view(torch.uint16).reshape(
            self._num_blocks, self._num_planes, self._plane_scalars
        )
        self._kernels = tuple(
            compile_load_kernel(
                num_blocks=self._num_blocks,
                num_planes=self._num_planes,
                plane_scalars=self._plane_scalars,
                tile_scalars=self._tile_scalars,
                staging_bytes=max_slots_per_buffer * int(kv_cache[0].nbytes),
                fanout=fanout,
                online=online,
            )
            for fanout in (False, True)
        )
        self._codebooks = torch.from_numpy(
            np.frombuffer(codebooks, dtype=np.uint8).copy()
        ).to(kv_cache.device)
        self._rings = tuple(
            self._allocate_metadata(kv_cache.device, max_slots_per_buffer)
            for _ in range(ring_depth)
        )

    def decode(
        self,
        *,
        staging: torch.Tensor,
        staging_offsets: list[int],
        block_ids: list[int],
        modes: list[int],
        buffer_index: int,
        stream: torch.cuda.Stream,
    ) -> int:
        """Decode shared sources once directly into all registered KV targets.

        Args:
            staging: GPU byte tensor filled by the server transfer layer.
            staging_offsets: Start of each indexed slot record in staging.
            block_ids: Matching, distinct destination physical vLLM blocks.
            modes: Zero for raw slots and one for compressed slots.
            buffer_index: Fixed staging/metadata ring index.
            stream: LoadPipeline CUDA stream ordered after transfer completion.

        Returns:
            Number of logical slots restored by this launch.

        Raises:
            ValueError: If metadata lengths, capacity, or block IDs are invalid.

        Async/thread-safety:
            Called on one load thread. The caller retains the staging lease and
            synchronizes ``stream`` before reusing its ring index.
        """
        plan = self.prepare(
            staging=staging,
            staging_offsets=staging_offsets,
            block_ids=block_ids,
            modes=modes,
            buffer_index=buffer_index,
            stream=stream,
        )
        if plan is None:
            return 0
        plan.submit_next(plan.remaining_sources)
        return plan.destination_count

    def prepare(
        self,
        *,
        staging: torch.Tensor,
        staging_offsets: list[int],
        block_ids: list[int],
        modes: list[int],
        buffer_index: int,
        stream: torch.cuda.Stream,
    ) -> PreparedKVRestore | None:
        """Upload metadata once and prepare independently submitted source ranges.

        Args:
            staging: Completed server transfer's leased GPU byte tensor.
            staging_offsets: One source record offset per destination.
            block_ids: Matching distinct destination physical KV blocks.
            modes: Matching zero/raw or one/packed record modes.
            buffer_index: Exclusively leased staging/metadata ring index.
            stream: Stream ordered after transfer completion.
        Returns:
            Restore plan sharing the original buffers, or None for empty input.
        Raises:
            ValueError: On invalid lengths, capacity, IDs or conflicting modes.
        Async/thread-safety:
            Call only on the load thread. The ring's pinned and device metadata
            must remain immutable until an event after the plan's last kernel
            completes. This call uploads metadata but does not launch decode.
        """
        slot_count = len(staging_offsets)
        if not (slot_count == len(block_ids) == len(modes)):
            raise ValueError("compressed restore metadata lengths do not match")
        if slot_count == 0:
            return None
        if not 0 <= buffer_index < len(self._rings):
            raise ValueError("compressed metadata ring index is invalid")
        ring = self._rings[buffer_index]
        if slot_count > ring.host_offsets.numel():
            raise ValueError("compressed restore exceeds metadata ring capacity")
        if staging.device != self._kv_cache.device or staging.dtype is not torch.uint8:
            raise ValueError("compressed staging must be a CUDA byte tensor")
        if any(offset < 0 or offset >= staging.numel() for offset in staging_offsets):
            raise ValueError("compressed restore staging offset is out of bounds")
        if any(block < 0 or block >= self._num_blocks for block in block_ids):
            raise ValueError("compressed restore destination block is invalid")
        if len(set(block_ids)) != slot_count:
            raise ValueError("compressed restore destination blocks must be distinct")
        if any(mode not in (0, 1) for mode in modes):
            raise ValueError("compressed restore slot mode is invalid")
        if any(
            mode == 1 and offset + _SLOT_HEADER_BYTES > staging.numel()
            for offset, mode in zip(staging_offsets, modes, strict=True)
        ):
            raise ValueError("compressed record header exceeds staging capacity")
        raw_slot_bytes = int(self._kv_cache[0].nbytes)
        if any(
            mode == 0 and offset + raw_slot_bytes > staging.numel()
            for offset, mode in zip(staging_offsets, modes, strict=True)
        ):
            raise ValueError("raw restore record exceeds staging capacity")
        # Equal staging offsets identify the same immutable source record in
        # this lease. Group only within the current completed transfer: there
        # is no cross-request wait or persistent GPU cache. Destination lists
        # form a CSR map so arbitrary fanout remains a runtime dimension and
        # every source scalar is decoded once before its final KV writes.
        sources: dict[int, tuple[int, list[int]]] = {}
        for offset, mode, block in zip(staging_offsets, modes, block_ids, strict=True):
            existing = sources.get(offset)
            if existing is None:
                sources[offset] = (mode, [block])
            else:
                if existing[0] != mode:
                    raise ValueError("compressed restore source has conflicting modes")
                existing[1].append(block)
        source_count = len(sources)
        destination_offsets = [0]
        destinations: list[int] = []
        source_modes: list[int] = []
        for mode, targets in sources.values():
            destinations.extend(targets)
            destination_offsets.append(len(destinations))
            source_modes.append(mode)

        # The decoder metadata is small, but this method runs once for every
        # cache-hit batch.  Per-element tensor assignment takes the Python
        # interpreter lock for every slot and creates a host-side dispatch
        # point before the asynchronous H2D copies. Convert each list
        # once into its declared dtype and copy through the contiguous NumPy
        # view of the persistent pinned tensor instead.
        np.copyto(
            ring.host_offsets[:source_count].numpy(),
            np.asarray(list(sources), dtype=np.int64),
        )
        np.copyto(
            ring.host_blocks[:slot_count].numpy(),
            np.asarray(destinations, dtype=np.int32),
        )
        np.copyto(
            ring.host_modes[:source_count].numpy(),
            np.asarray(source_modes, dtype=np.int32),
        )
        np.copyto(
            ring.host_destination_offsets[: source_count + 1].numpy(),
            np.asarray(destination_offsets, dtype=np.int32),
        )
        with torch.cuda.stream(stream):
            ring.device_offsets[:source_count].copy_(
                ring.host_offsets[:source_count], non_blocking=True
            )
            ring.device_blocks[:slot_count].copy_(
                ring.host_blocks[:slot_count], non_blocking=True
            )
            ring.device_modes[:source_count].copy_(
                ring.host_modes[:source_count], non_blocking=True
            )
            fanout = source_count < slot_count
            if fanout:
                ring.device_destination_offsets[: source_count + 1].copy_(
                    ring.host_destination_offsets[: source_count + 1], non_blocking=True
                )
        return PreparedKVRestore(
            kernel=self._kernels[fanout],
            staging=staging,
            offsets=ring.device_offsets[:source_count],
            blocks=ring.device_blocks[:slot_count],
            destination_offsets=ring.device_destination_offsets[: source_count + 1],
            modes=ring.device_modes[:source_count],
            codebooks=self._codebooks,
            destination=self._kv_bits,
            fanout=fanout,
            stream=stream,
        )

    @staticmethod
    def _allocate_metadata(device: torch.device, capacity: int) -> _MetadataRing:
        # vLLM registers KV caches under InferenceMode, while this metadata is
        # updated later on the load thread. Explicitly create normal tensors so
        # thread-local InferenceMode state at construction cannot make the ring
        # immutable outside that context.
        with torch.inference_mode(False):
            host_offsets = torch.empty(capacity, dtype=torch.int64, pin_memory=True)
            host_blocks = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
            host_modes = torch.empty(capacity, dtype=torch.int32, pin_memory=True)
            return _MetadataRing(
                host_offsets=host_offsets,
                host_blocks=host_blocks,
                host_modes=host_modes,
                host_destination_offsets=torch.empty(
                    capacity + 1, dtype=torch.int32, pin_memory=True
                ),
                device_offsets=torch.empty(capacity, dtype=torch.int64, device=device),
                device_blocks=torch.empty(capacity, dtype=torch.int32, device=device),
                device_modes=torch.empty(capacity, dtype=torch.int32, device=device),
                device_destination_offsets=torch.empty(
                    capacity + 1, dtype=torch.int32, device=device
                ),
            )


def compressed_slot_metadata(
    per_req_ranges: list[Any],
) -> tuple[list[int], list[int], list[int]]:
    """Flatten batch restore ranges into kernel slot metadata.

    Args:
        per_req_ranges: Read-plan ranges containing ReqLoadSpec values.

    Returns:
        Staging offsets, destination block IDs, and integer slot modes.

    Async/thread-safety:
        Pure CPU planning safe on the load pipeline thread.
    """
    offsets: list[int] = []
    block_ids: list[int] = []
    modes: list[int] = []
    for item in per_req_ranges:
        start, _end, spec = item if len(item) == 3 else (item[0], item[1], item[3])
        cursor = int(start)
        if len(spec.compressed_slots) != len(spec.block_ids):
            raise ValueError("compressed slot metadata does not match block IDs")
        for slot, block_id in zip(spec.compressed_slots, spec.block_ids, strict=True):
            offsets.append(cursor)
            block_ids.append(int(block_id))
            modes.append(0 if slot.mode == "raw" else 1)
            cursor += slot.stored_length
    return offsets, block_ids, modes


__all__ = [
    "FusedCompressedKVDecoder",
    "FusedOnlineKVPacker",
    "OnlinePackedSlot",
    "compressed_slot_metadata",
    "warm_fused_online_kv_packer",
]
