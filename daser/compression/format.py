# SPDX-License-Identifier: Apache-2.0

"""Versioned binary contracts for strict-lossless compressed KV slots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import struct

# Version 1 stores one escape and one raw-escape prefix per tile.
FORMAT_VERSION = 1
IO_ALIGNMENT = 4096
CODEBOOK_ENTRIES = 15
KV_PLANES = 2
# Online producers and worker startup warmup must agree before the server's
# runtime configuration is available.
ONLINE_TILE_SCALARS = 256
SLOT_MAGIC = b"DKVSLOT1"

_SLOT_HEADER = struct.Struct("<8sIIIIQIIQQ32s32s")
_PLANE_DESCRIPTOR = struct.Struct("<HH11I")
# Online records use a 3-bit main stream whose all-ones symbol escapes to a
# packed 3-bit secondary stream (seven secondary entries plus a raw sentinel).
# The byte after the descriptor table records this layout as flags.
SYMBOL_BITS = 3
ESCAPE_SYMBOL_BITS = 3
_ONLINE_LAYOUT_FLAGS = SYMBOL_BITS | 0x80 | 0x40


def slot_header_bytes(plane_count: int) -> int:
    """Return the aligned header bytes that precede a slot's plane records.

    The fixed header, one descriptor per plane, and the layout flags byte
    occupy as many ``IO_ALIGNMENT`` pages as they need: one page up to 41
    layers (82 planes), more for deeper models. The length is persisted in
    the header's ``header_bytes`` field, so single-page records are unchanged.

    Args:
        plane_count: Layer/K-or-V planes described by the header.

    Returns:
        ``IO_ALIGNMENT``-aligned header length in bytes.

    Raises:
        ValueError: If ``plane_count`` is not positive.

    Async/thread-safety:
        Pure; safe from any thread.
    """
    if plane_count <= 0:
        raise ValueError("compressed slot headers need at least one plane")
    return align_up(_SLOT_HEADER.size + plane_count * _PLANE_DESCRIPTOR.size + 1)


def prefix_table_bytes(tile_count: int) -> int:
    """Return the bytes of one plane's escape and raw-escape prefix tables.

    Args:
        tile_count: Tiles in the plane.

    Returns:
        Two little-endian ``uint32`` tables of ``tiles + 1``
        cumulative counts each, stored back-to-back.

    Async/thread-safety:
        Pure and safe to call from any thread.
    """
    return 8 * (tile_count + 1)


def align_up(value: int, alignment: int = IO_ALIGNMENT) -> int:
    """Round a non-negative byte count up to an alignment.

    Args:
        value: Byte count to align.
        alignment: Positive power-of-two alignment.

    Returns:
        The smallest aligned integer greater than or equal to ``value``.

    Raises:
        ValueError: If either argument is invalid.

    Async/thread-safety:
        Pure and safe to call from any thread.
    """
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("value must be non-negative and alignment a power of two")
    return (value + alignment - 1) & -alignment


def online_fixed_envelope_geometry(
    *,
    slot_stride: int,
    num_planes: int,
    plane_scalars: int,
    max_tiles: int,
) -> tuple[int, int, int] | None:
    """Return the online packed envelope geometry used by the codec.

    Args:
        slot_stride: Raw bytes available for one local KV slot.
        num_planes: Number of K/V planes in the cross-layer record.
        plane_scalars: BF16 scalar count in one plane.
        max_tiles: Maximum codec tiles in one plane.

    Returns:
        A tuple of per-plane bytes, fixed escape bytes, and aligned stored
        length, or ``None`` when the raw slot cannot hold the envelope.

    Async/thread-safety:
        Pure arithmetic with no I/O or shared state.
    """
    payload_base = (
        plane_scalars + (plane_scalars * 3 + 7) // 8 + prefix_table_bytes(max_tiles)
    )
    if num_planes <= 0:
        return None
    header_bytes = slot_header_bytes(num_planes)
    available = slot_stride - header_bytes
    if available <= 0:
        return None
    plane_bytes = available // num_planes
    fixed_escape = plane_bytes - payload_base
    if fixed_escape <= 0:
        return None
    return plane_bytes, fixed_escape, header_bytes + num_planes * plane_bytes


def digest_bytes(payload: bytes | bytearray | memoryview) -> bytes:
    """Return the SHA-256 digest of a byte payload.

    Args:
        payload: Bytes included in the digest.

    Returns:
        Raw 32-byte SHA-256 digest.

    Async/thread-safety:
        Pure and safe to call from any thread.
    """
    return hashlib.sha256(payload).digest()


class SlotMode(IntEnum):
    """Physical encoding selected for one fixed slot envelope."""

    RAW = 0
    COMPRESSED = 1


@dataclass(frozen=True)
class CompressedStoreGeometry:
    """Immutable geometry shared by a compressed store and its model.

    Attributes:
        slot_size: Raw bytes reserved for each logical slot.
        block_tokens: Tokens represented by one slot.
        num_layers: Number of model KV layers.
        num_kv_heads: KV heads stored by the TP=1 worker.
        head_dim: Scalars in each KV head.
        dtype_bytes: Bytes per scalar; version one requires BF16 (two bytes).
        tile_scalars: Independently decodable scalar count used by the kernel.
        num_slots: Number of slot envelopes a fixed-size buffer holds, or
            None when the geometry describes records of an unbounded store
            (the server tier store has no slot count).
    """

    slot_size: int
    block_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2
    tile_scalars: int = ONLINE_TILE_SCALARS
    num_slots: int | None = None

    def __post_init__(self) -> None:
        expected = (
            self.num_layers
            * KV_PLANES
            * self.block_tokens
            * self.num_kv_heads
            * self.head_dim
            * self.dtype_bytes
        )
        values = (
            self.slot_size,
            self.block_tokens,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.tile_scalars,
        )
        if any(value <= 0 for value in values) or (
            self.num_slots is not None and self.num_slots <= 0
        ):
            raise ValueError("compressed store geometry values must be positive")
        if self.dtype_bytes != 2:
            raise ValueError("compressed format version 1 supports BF16 only")
        if self.slot_size != expected:
            raise ValueError(
                f"slot_size {self.slot_size} does not match geometry {expected}"
            )
        if self.slot_size % IO_ALIGNMENT:
            raise ValueError("slot_size must be 4 KiB aligned for O_DIRECT")

    @property
    def plane_count(self) -> int:
        """Return the number of independently coded layer/K-or-V planes."""
        return self.num_layers * KV_PLANES

    @property
    def header_bytes(self) -> int:
        """Return the aligned header bytes before the first plane record."""
        return slot_header_bytes(self.plane_count)

    @property
    def plane_scalars(self) -> int:
        """Return BF16 scalar count in one layer/K-or-V plane."""
        return self.block_tokens * self.num_kv_heads * self.head_dim

    @property
    def plane_bytes(self) -> int:
        """Return uncompressed bytes in one layer/K-or-V plane."""
        return self.plane_scalars * self.dtype_bytes

    @property
    def online_fixed_stored_length(self) -> int:
        """Return the maximum aligned online packed length for one slot.

        Returns:
            The fixed TileLang envelope size used to derive a conservative
            logical slot capacity for compressed-online storage.

        Raises:
            ValueError: If this geometry cannot represent a fixed online
                envelope.

        Async/thread-safety:
            Pure arithmetic with no I/O or shared state.
        """
        max_tiles = (self.plane_scalars + self.tile_scalars - 1) // self.tile_scalars
        envelope = online_fixed_envelope_geometry(
            slot_stride=self.slot_size,
            num_planes=self.plane_count,
            plane_scalars=self.plane_scalars,
            max_tiles=max_tiles,
        )
        if envelope is None:
            raise ValueError("compressed-online geometry has no fixed envelope")
        return envelope[2]


@dataclass(frozen=True)
class PlaneDescriptor:
    """Offsets required to decode one layer/K-or-V plane inside a slot."""

    layer: int
    kv: int
    scalar_count: int
    tile_count: int
    record_offset: int
    record_length: int
    low_offset: int
    symbol_offset: int
    prefix_offset: int
    escape_offset: int
    escape_count: int

    def pack(self) -> bytes:
        """Serialize this descriptor into the version-one fixed struct."""
        return _PLANE_DESCRIPTOR.pack(
            self.layer,
            self.kv,
            self.scalar_count,
            self.tile_count,
            self.record_offset,
            self.record_length,
            self.low_offset,
            self.symbol_offset,
            self.prefix_offset,
            self.escape_offset,
            self.escape_count,
            0,
            0,
        )

    @classmethod
    def unpack_from(cls, payload: bytes | memoryview, offset: int) -> "PlaneDescriptor":
        """Parse one descriptor from a validated slot header buffer."""
        fields = _PLANE_DESCRIPTOR.unpack_from(payload, offset)
        return cls(*fields[:11])


@dataclass(frozen=True)
class SlotHeader:
    """Validated compressed-slot header and its ordered plane descriptors."""

    slot_id: int
    raw_length: int
    stored_length: int
    tile_scalars: int
    num_layers: int
    codebook_hash: bytes
    raw_hash: bytes
    descriptors: tuple[PlaneDescriptor, ...]
    format_version: int = FORMAT_VERSION

    def pack(self) -> bytes:
        """Serialize the header and descriptor table into aligned pages.

        Returns:
            ``slot_header_bytes(len(descriptors))`` bytes.

        Raises:
            ValueError: If a hash has the wrong length or the format version
                is unknown.
        """
        if len(self.codebook_hash) != 32 or len(self.raw_hash) != 32:
            raise ValueError("slot hashes must contain 32 bytes")
        if self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported compressed slot format version")
        header_bytes = slot_header_bytes(len(self.descriptors))
        page = bytearray(header_bytes)
        _SLOT_HEADER.pack_into(
            page,
            0,
            SLOT_MAGIC,
            self.format_version,
            header_bytes,
            IO_ALIGNMENT,
            self.tile_scalars,
            self.slot_id,
            self.num_layers,
            len(self.descriptors),
            self.raw_length,
            self.stored_length,
            self.codebook_hash,
            self.raw_hash,
        )
        offset = _SLOT_HEADER.size
        for descriptor in self.descriptors:
            page[offset : offset + _PLANE_DESCRIPTOR.size] = descriptor.pack()
            offset += _PLANE_DESCRIPTOR.size
        page[offset] = _ONLINE_LAYOUT_FLAGS
        return bytes(page)

    @classmethod
    def parse(
        cls,
        payload: bytes | memoryview,
        *,
        expected_geometry: CompressedStoreGeometry | None = None,
        expected_slot_id: int | None = None,
        expected_codebook_hash: bytes | None = None,
    ) -> "SlotHeader":
        """Parse and fully validate a compressed slot header.

        Args:
            payload: Complete aligned compressed slot payload.
            expected_geometry: Optional store geometry contract.
            expected_slot_id: Optional logical slot expected at this envelope.
            expected_codebook_hash: Optional immutable codebook identity.

        Returns:
            Validated SlotHeader with ordered descriptors.

        Raises:
            ValueError: On magic, version, geometry, bounds, or layout mismatch.

        Async/thread-safety:
            Pure parsing; safe to call concurrently.
        """
        if len(payload) < IO_ALIGNMENT:
            raise ValueError("compressed slot is shorter than its header page")
        fields = _SLOT_HEADER.unpack_from(payload, 0)
        (
            magic,
            version,
            header_bytes,
            alignment,
            tile_scalars,
            slot_id,
            num_layers,
            plane_count,
            raw_length,
            stored_length,
            codebook_hash,
            raw_hash,
        ) = fields
        if magic != SLOT_MAGIC:
            raise ValueError("unsupported compressed slot magic")
        if version != FORMAT_VERSION:
            raise ValueError("unsupported compressed slot format version")
        if alignment != IO_ALIGNMENT or plane_count <= 0:
            raise ValueError("compressed slot uses unsupported alignment")
        if header_bytes != slot_header_bytes(plane_count):
            raise ValueError("compressed slot header length is inconsistent")
        if len(payload) < header_bytes:
            raise ValueError("compressed slot is shorter than its header")
        if not header_bytes <= stored_length <= len(payload) or (
            stored_length % IO_ALIGNMENT
        ):
            raise ValueError("compressed slot stored length is invalid")
        if tile_scalars <= 0 or raw_length <= 0 or raw_length % (plane_count * 2):
            raise ValueError("compressed slot geometry is invalid")
        if expected_slot_id is not None and slot_id != expected_slot_id:
            raise ValueError("compressed slot ID does not match its envelope")
        if (
            expected_codebook_hash is not None
            and codebook_hash != expected_codebook_hash
        ):
            raise ValueError("compressed slot codebook hash mismatch")
        if plane_count != num_layers * KV_PLANES:
            raise ValueError("compressed slot plane count is inconsistent")
        flags_offset = _SLOT_HEADER.size + plane_count * _PLANE_DESCRIPTOR.size
        if int(payload[flags_offset]) != _ONLINE_LAYOUT_FLAGS:
            raise ValueError("compressed slot symbol layout is unsupported")
        if expected_geometry is not None:
            if (
                raw_length != expected_geometry.slot_size
                or num_layers != expected_geometry.num_layers
                or tile_scalars != expected_geometry.tile_scalars
                or plane_count != expected_geometry.plane_count
            ):
                raise ValueError("compressed slot geometry mismatch")

        descriptors = tuple(
            PlaneDescriptor.unpack_from(
                payload,
                _SLOT_HEADER.size + index * _PLANE_DESCRIPTOR.size,
            )
            for index in range(plane_count)
        )
        plane_scalars = (
            expected_geometry.plane_scalars
            if expected_geometry is not None
            else raw_length // (plane_count * 2)
        )
        previous_end = header_bytes
        for index, descriptor in enumerate(descriptors):
            expected_layer, expected_kv = divmod(index, KV_PLANES)
            if (descriptor.layer, descriptor.kv) != (expected_layer, expected_kv):
                raise ValueError("compressed slot descriptors are out of order")
            if descriptor.scalar_count != plane_scalars:
                raise ValueError("compressed plane scalar count mismatch")
            if (
                descriptor.tile_count
                != (descriptor.scalar_count + tile_scalars - 1) // tile_scalars
            ):
                raise ValueError("compressed plane tile count mismatch")
            record_end = descriptor.record_offset + descriptor.record_length
            # Escape and raw-escape prefix tables are stored back-to-back.
            prefix_end = descriptor.prefix_offset + prefix_table_bytes(
                descriptor.tile_count
            )
            symbol_bytes = (descriptor.scalar_count * SYMBOL_BITS + 7) // 8
            symbol_end = descriptor.symbol_offset + symbol_bytes
            low_end = descriptor.low_offset + descriptor.scalar_count
            escape_code_bytes = (descriptor.escape_count * ESCAPE_SYMBOL_BITS + 7) // 8
            escape_code_end = descriptor.escape_offset + escape_code_bytes
            # Validate every stream against the actual payload before scanning
            # escape codes. A corrupt descriptor must not cause indexing beyond
            # the sliced stream or use an unbounded claimed escape count.
            if (
                descriptor.record_offset != previous_end
                or descriptor.record_length <= 0
                or descriptor.escape_count > descriptor.scalar_count
                or not (
                    descriptor.record_offset
                    <= descriptor.low_offset
                    <= low_end
                    <= descriptor.symbol_offset
                    <= symbol_end
                    <= descriptor.prefix_offset
                    <= prefix_end
                    <= descriptor.escape_offset
                    <= escape_code_end
                    <= record_end
                    <= stored_length
                )
            ):
                raise ValueError("compressed plane offsets are invalid")
            raw_escape_count = 0
            escape_codes = payload[descriptor.escape_offset : escape_code_end]
            raw_code = (1 << ESCAPE_SYMBOL_BITS) - 1
            for token in range(descriptor.escape_count):
                bit_offset = token * ESCAPE_SYMBOL_BITS
                byte_offset = bit_offset // 8
                shift = bit_offset & 7
                value = escape_codes[byte_offset]
                if shift + ESCAPE_SYMBOL_BITS > 8:
                    value |= escape_codes[byte_offset + 1] << 8
                raw_escape_count += int(((value >> shift) & raw_code) == raw_code)
            if escape_code_end + raw_escape_count > record_end:
                raise ValueError("packed raw escapes exceed plane record")
            previous_end = record_end
        if previous_end > stored_length:
            raise ValueError("compressed slot records exceed stored length")
        return cls(
            slot_id=slot_id,
            raw_length=raw_length,
            stored_length=stored_length,
            tile_scalars=tile_scalars,
            num_layers=num_layers,
            codebook_hash=codebook_hash,
            raw_hash=raw_hash,
            descriptors=descriptors,
            format_version=version,
        )


__all__ = [
    "CODEBOOK_ENTRIES",
    "ESCAPE_SYMBOL_BITS",
    "FORMAT_VERSION",
    "IO_ALIGNMENT",
    "KV_PLANES",
    "SYMBOL_BITS",
    "CompressedStoreGeometry",
    "PlaneDescriptor",
    "SlotHeader",
    "SlotMode",
    "align_up",
    "digest_bytes",
    "online_fixed_envelope_geometry",
    "prefix_table_bytes",
    "slot_header_bytes",
]
