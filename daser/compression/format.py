# SPDX-License-Identifier: Apache-2.0

"""Versioned binary contracts for strict-lossless compressed KV slots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import struct
from typing import Any, Mapping

FORMAT_VERSION = 1
IO_ALIGNMENT = 4096
CODEBOOK_ENTRIES = 15
KV_PLANES = 2
CODEC_ID = "daser.lossless.bf16.3bit"
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
    *, slot_stride: int, num_planes: int, plane_scalars: int, max_tiles: int
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
    payload_base = plane_scalars + (plane_scalars * 3 + 7) // 8 + 8 * (max_tiles + 1)
    available = slot_stride - IO_ALIGNMENT
    if available <= 0 or num_planes <= 0:
        return None
    plane_bytes = available // num_planes
    fixed_escape = plane_bytes - payload_base
    if fixed_escape <= 0:
        return None
    return plane_bytes, fixed_escape, IO_ALIGNMENT + num_planes * plane_bytes


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


def codec_identity_digest(*, codebook_hash: bytes, tile_scalars: int) -> bytes:
    """Return the immutable identity digest for one online codec instance.

    Args:
        codebook_hash: SHA-256 digest of the plane-major codebook bytes.
        tile_scalars: Scalar quantum used by the independently decodable tiles.

    Returns:
        SHA-256 digest covering the codec name, wire format, symbol widths,
        tile geometry, and codebook identity.

    Raises:
        ValueError: If the codebook digest is not 32 bytes or the tile size is
            not positive.

    Async/thread-safety:
        Pure CPU hashing with no I/O or shared state.
    """
    if len(codebook_hash) != hashlib.sha256().digest_size:
        raise ValueError("codebook_hash must be a SHA-256 digest")
    if tile_scalars <= 0:
        raise ValueError("tile_scalars must be positive")
    identity = b"\0".join(
        (
            CODEC_ID.encode("ascii"),
            struct.pack(
                "<IIII",
                FORMAT_VERSION,
                tile_scalars,
                SYMBOL_BITS,
                ESCAPE_SYMBOL_BITS,
            ),
            codebook_hash,
        )
    )
    return digest_bytes(identity)


@dataclass(frozen=True)
class SlotPublication:
    """Immutable publication record for one lossless slot envelope.

    The record is the control-plane carrier for reload.  Its digests are
    deliberately separate: ``source_digest`` identifies the bytes before
    encoding, ``stored_digest`` identifies the bytes written to the store, and
    ``restored_digest`` is the post-decode acceptance check.  RAW fallback
    uses the same contract and therefore cannot silently bypass identity or
    lifecycle validation.

    Attributes:
        slot_id: Logical slot identity assigned by the server.
        mode: RAW or COMPRESSED envelope representation.
        format_version: Version of the persisted slot wire format.
        codec_id: Stable human-readable codec family identifier.
        codec_digest: Digest of all codec parameters and codebook bytes.
        raw_length: Original fixed-envelope byte length.
        stored_length: Number of bytes published for this record.
        source_digest: Digest of the source KV bytes before encoding.
        restored_digest: Digest expected after lossless decode.
        stored_digest: Digest of the exact bytes written to the store.

    Async/thread-safety:
        Immutable and safe to pass between the server event loop and worker
        threads after construction. Validation is synchronous and CPU-only.
    """

    slot_id: int
    mode: SlotMode
    format_version: int
    codec_id: str
    codec_digest: bytes
    raw_length: int
    stored_length: int
    source_digest: bytes
    restored_digest: bytes
    stored_digest: bytes

    def __post_init__(self) -> None:
        if self.slot_id < 0:
            raise ValueError("slot_id must be non-negative")
        if self.mode not in (SlotMode.RAW, SlotMode.COMPRESSED):
            raise ValueError("unknown slot publication mode")
        if self.format_version != FORMAT_VERSION:
            raise ValueError("unsupported slot publication format version")
        if self.codec_id != CODEC_ID:
            raise ValueError("unsupported slot publication codec")
        if self.raw_length <= 0 or self.stored_length <= 0:
            raise ValueError("slot publication lengths must be positive")
        if self.stored_length > self.raw_length:
            raise ValueError("stored slot exceeds its fixed raw envelope")
        if self.mode is SlotMode.RAW and self.stored_length != self.raw_length:
            raise ValueError("raw slot publication must use the full envelope")
        for name, digest in (
            ("codec_digest", self.codec_digest),
            ("source_digest", self.source_digest),
            ("restored_digest", self.restored_digest),
            ("stored_digest", self.stored_digest),
        ):
            if len(digest) != hashlib.sha256().digest_size:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if self.source_digest != self.restored_digest:
            raise ValueError("lossless publication source and restored digests differ")

    @classmethod
    def from_encoded(
        cls,
        *,
        slot_id: int,
        mode: SlotMode,
        raw_length: int,
        stored_payload: bytes | bytearray | memoryview,
        source_digest: bytes,
        codebook_hash: bytes,
        tile_scalars: int,
    ) -> "SlotPublication":
        """Build a complete publication record from encoded bytes.

        Args:
            slot_id: Logical server-assigned slot ID.
            mode: Representation used for ``stored_payload``.
            raw_length: Fixed raw slot byte length.
            stored_payload: Exact bytes that will be written to the store.
            source_digest: SHA-256 digest of the source raw slot.
            codebook_hash: SHA-256 digest of the codebook used for encoding.
            tile_scalars: Codec tile quantum used by the encoder.

        Returns:
            A complete immutable publication contract.

        Raises:
            ValueError: If payload lengths or digests are inconsistent.

        Async/thread-safety:
            Pure CPU metadata construction; safe to call from any thread.
        """
        payload = bytes(stored_payload)
        if raw_length <= 0 or len(payload) <= 0:
            raise ValueError("publication lengths must be positive")
        if mode is SlotMode.RAW and len(payload) != raw_length:
            raise ValueError("raw publication payload must fill the envelope")
        return cls(
            slot_id=slot_id,
            mode=mode,
            format_version=FORMAT_VERSION,
            codec_id=CODEC_ID,
            codec_digest=codec_identity_digest(
                codebook_hash=codebook_hash,
                tile_scalars=tile_scalars,
            ),
            raw_length=raw_length,
            stored_length=len(payload),
            source_digest=source_digest,
            restored_digest=source_digest,
            stored_digest=digest_bytes(payload),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SlotPublication":
        """Validate and deserialize a server-to-worker publication payload.

        Args:
            payload: Msgpack-safe mapping containing every contract field.

        Returns:
            An immutable publication record.

        Raises:
            ValueError: If a field is missing, malformed, or inconsistent.

        Async/thread-safety:
            Pure validation with no I/O; safe on the worker load thread.
        """
        try:
            mode = (
                SlotMode.COMPRESSED
                if payload["mode"] == "compressed"
                else (SlotMode.RAW if payload["mode"] == "raw" else None)
            )
            if mode is None:
                raise ValueError("unknown slot publication mode")
            return cls(
                slot_id=int(payload["slot_id"]),
                mode=mode,
                format_version=int(payload["format_version"]),
                codec_id=str(payload["codec_id"]),
                codec_digest=bytes(payload["codec_digest"]),
                raw_length=int(payload["raw_length"]),
                stored_length=int(payload["stored_length"]),
                source_digest=bytes(payload["source_digest"]),
                restored_digest=bytes(payload["restored_digest"]),
                stored_digest=bytes(payload["stored_digest"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith(
                ("unknown slot publication", "slot_id", "unsupported", "raw slot")
            ):
                raise
            raise ValueError("invalid slot publication payload") from exc

    def to_payload(self) -> dict[str, object]:
        """Return a msgpack-safe immutable publication payload.

        Returns:
            A new mapping whose byte values are safe for msgpack binary mode.

        Async/thread-safety:
            Pure allocation with no I/O; safe to call from any thread.
        """
        return {
            "slot_id": self.slot_id,
            "mode": self.mode.name.lower(),
            "format_version": self.format_version,
            "codec_id": self.codec_id,
            "codec_digest": self.codec_digest,
            "raw_length": self.raw_length,
            "stored_length": self.stored_length,
            "source_digest": self.source_digest,
            "restored_digest": self.restored_digest,
            "stored_digest": self.stored_digest,
        }

    def validate_reload(
        self,
        stored_payload: bytes | bytearray | memoryview,
        *,
        expected_slot_id: int,
        expected_codec_digest: bytes,
    ) -> None:
        """Validate store bytes and identity before handing them to decode.

        Args:
            stored_payload: Bytes read from the physical store.
            expected_slot_id: Logical slot ID requested by the load plan.
            expected_codec_digest: Codec identity expected by the worker.

        Raises:
            ValueError: If the record is stale, truncated, or belongs to a
                different codec instance.

        Async/thread-safety:
            Synchronous digest validation; callers should run it off the event
            loop when validating large payloads.
        """
        payload = bytes(stored_payload)
        if self.slot_id != expected_slot_id:
            raise ValueError("slot publication slot identity mismatch")
        if self.codec_digest != expected_codec_digest:
            raise ValueError("slot publication codec identity mismatch")
        if len(payload) != self.stored_length:
            raise ValueError("slot publication stored length mismatch")
        if digest_bytes(payload) != self.stored_digest:
            raise ValueError("slot publication stored digest mismatch")

    def validate_restored(
        self, restored_payload: bytes | bytearray | memoryview
    ) -> None:
        """Validate the byte-exact output of a lossless decoder.

        Args:
            restored_payload: Raw BF16 bytes produced by the worker decoder.

        Raises:
            ValueError: If the decoder output has the wrong length or digest.

        Async/thread-safety:
            Synchronous digest validation; safe to call after GPU-to-host
            completion on the worker thread.
        """
        restored = bytes(restored_payload)
        if len(restored) != self.raw_length:
            raise ValueError("slot publication restored length mismatch")
        if digest_bytes(restored) != self.restored_digest:
            raise ValueError("slot publication restored digest mismatch")


@dataclass(frozen=True)
class CompressedStoreGeometry:
    """Immutable geometry shared by a compressed store and its model.

    Attributes:
        num_slots: Number of fixed envelopes in the data file.
        slot_size: Raw bytes reserved for each logical slot.
        block_tokens: Tokens represented by one slot.
        num_layers: Number of model KV layers.
        num_kv_heads: KV heads stored by the TP=1 worker.
        head_dim: Scalars in each KV head.
        dtype_bytes: Bytes per scalar; version one requires BF16 (two bytes).
        tile_scalars: Independently decodable scalar count used by the kernel.
    """

    num_slots: int
    slot_size: int
    block_tokens: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2
    tile_scalars: int = 1024

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
            self.num_slots,
            self.slot_size,
            self.block_tokens,
            self.num_layers,
            self.num_kv_heads,
            self.head_dim,
            self.tile_scalars,
        )
        if any(value <= 0 for value in values):
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
            logical-ring capacity for compressed-online storage.

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

    @property
    def online_min_stored_length(self) -> int:
        """Return the aligned lower bound for one online packed record.

        Returns:
            The smallest byte range that the online format can describe for
            this geometry. The value is used only to size logical metadata;
            physical admission still uses each record's actual length.

        Async/thread-safety:
            Pure arithmetic with no I/O or shared state.
        """
        max_tiles = (self.plane_scalars + self.tile_scalars - 1) // self.tile_scalars
        payload_base = (
            self.plane_scalars + (self.plane_scalars * 3 + 7) // 8 + 8 * (max_tiles + 1)
        )
        return align_up(IO_ALIGNMENT + self.plane_count * payload_base)


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

    def pack(self) -> bytes:
        """Serialize the header and descriptor table into one 4 KiB page."""
        if len(self.codebook_hash) != 32 or len(self.raw_hash) != 32:
            raise ValueError("slot hashes must contain 32 bytes")
        page = bytearray(IO_ALIGNMENT)
        _SLOT_HEADER.pack_into(
            page,
            0,
            SLOT_MAGIC,
            FORMAT_VERSION,
            IO_ALIGNMENT,
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
        if offset + len(self.descriptors) * _PLANE_DESCRIPTOR.size > IO_ALIGNMENT:
            raise ValueError("plane descriptor table exceeds the slot header page")
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
        if magic != SLOT_MAGIC or version != FORMAT_VERSION:
            raise ValueError("unsupported compressed slot magic or version")
        if header_bytes != IO_ALIGNMENT or alignment != IO_ALIGNMENT:
            raise ValueError("compressed slot uses unsupported alignment")
        if stored_length > len(payload) or stored_length % IO_ALIGNMENT:
            raise ValueError("compressed slot stored length is invalid")
        if expected_slot_id is not None and slot_id != expected_slot_id:
            raise ValueError("compressed slot ID does not match its envelope")
        if (
            expected_codebook_hash is not None
            and codebook_hash != expected_codebook_hash
        ):
            raise ValueError("compressed slot codebook hash mismatch")
        if plane_count != num_layers * KV_PLANES:
            raise ValueError("compressed slot plane count is inconsistent")
        if _SLOT_HEADER.size + plane_count * _PLANE_DESCRIPTOR.size > header_bytes:
            raise ValueError("compressed slot descriptor table is truncated")
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
            else raw_length // max(1, plane_count * 2)
        )
        previous_end = IO_ALIGNMENT
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
            prefix_end = descriptor.prefix_offset + 8 * (descriptor.tile_count + 1)
            symbol_bytes = (descriptor.scalar_count * SYMBOL_BITS + 7) // 8
            symbol_end = descriptor.symbol_offset + symbol_bytes
            low_end = descriptor.low_offset + descriptor.scalar_count
            escape_code_bytes = (descriptor.escape_count * ESCAPE_SYMBOL_BITS + 7) // 8
            escape_code_end = descriptor.escape_offset + escape_code_bytes
            if escape_code_end > record_end:
                raise ValueError("packed escape code stream exceeds plane record")
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
            escape_end = escape_code_end + raw_escape_count
            if (
                descriptor.record_offset != previous_end
                or descriptor.record_length <= 0
                or not (
                    descriptor.record_offset
                    <= descriptor.low_offset
                    <= low_end
                    <= descriptor.symbol_offset
                    <= symbol_end
                    <= descriptor.prefix_offset
                    <= prefix_end
                    <= descriptor.escape_offset
                    <= escape_end
                    <= record_end
                    <= stored_length
                )
            ):
                raise ValueError("compressed plane offsets are invalid")
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
        )


__all__ = [
    "CODEBOOK_ENTRIES",
    "CODEC_ID",
    "ESCAPE_SYMBOL_BITS",
    "FORMAT_VERSION",
    "IO_ALIGNMENT",
    "KV_PLANES",
    "SYMBOL_BITS",
    "CompressedStoreGeometry",
    "PlaneDescriptor",
    "SlotHeader",
    "SlotMode",
    "SlotPublication",
    "align_up",
    "digest_bytes",
    "codec_identity_digest",
    "online_fixed_envelope_geometry",
]
