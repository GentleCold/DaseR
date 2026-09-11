# SPDX-License-Identifier: Apache-2.0

"""Versioned binary contracts for strict-lossless compressed KV slots."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import hashlib
import os
from pathlib import Path
import struct
import tempfile
from typing import BinaryIO, Iterable

from daser.logging import init_logger

logger = init_logger(__name__)

FORMAT_VERSION = 1
IO_ALIGNMENT = 4096
CODEBOOK_ENTRIES = 15
KV_PLANES = 2
# Online producers and worker startup warmup must agree before the server's
# runtime configuration is available. Offline stores carry their own geometry.
ONLINE_TILE_SCALARS = 256
SLOT_MAGIC = b"DKVSLOT1"
INDEX_MAGIC = b"DKVIDX01"

_INDEX_HEADER = struct.Struct("<8s11I2Q32s32s")
_INDEX_ENTRY = struct.Struct("<QII32s32s")
_SLOT_HEADER = struct.Struct("<8sIIIIQIIQQ32s32s")
_PLANE_DESCRIPTOR = struct.Struct("<HH11I")
_SUPPORTED_SYMBOL_BITS = frozenset((3, 4))
ESCAPE_PACKED_FLAG = 0x80
# A packed 3-bit main stream may use either four bits (eight secondary
# entries plus a raw sentinel) or three bits (seven secondary entries plus a
# raw sentinel).  Keep the narrower escape stream behind a separate flag so
# existing stores remain readable without a format-version bump.
ESCAPE_3BIT_FLAG = 0x40


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
    symbol_bits: int = 4
    escape_packed: bool = False
    escape_symbol_bits: int = 4

    def pack(self) -> bytes:
        """Serialize the header and descriptor table into one 4 KiB page."""
        if len(self.codebook_hash) != 32 or len(self.raw_hash) != 32:
            raise ValueError("slot hashes must contain 32 bytes")
        if self.symbol_bits not in _SUPPORTED_SYMBOL_BITS:
            raise ValueError("compressed slots support only 3-bit or 4-bit symbols")
        if self.escape_packed and self.symbol_bits != 3:
            raise ValueError("packed escapes require the 3-bit symbol stream")
        if self.escape_symbol_bits not in (3, 4):
            raise ValueError("packed escapes support only 3-bit or 4-bit tokens")
        if not self.escape_packed and self.escape_symbol_bits != 4:
            raise ValueError("unpacked escapes require 4-bit metadata")
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
        # The byte immediately after the descriptor table was reserved in
        # format version one.  Zero means the historical nibble codec; a
        # non-zero value selects the lossless bit-packed symbol width.
        page[offset] = self.symbol_bits
        if self.escape_packed:
            page[offset] |= ESCAPE_PACKED_FLAG
            if self.escape_symbol_bits == 3:
                page[offset] |= ESCAPE_3BIT_FLAG
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
        flags = int(payload[flags_offset])
        escape_packed = bool(flags & ESCAPE_PACKED_FLAG)
        escape_symbol_bits = 3 if flags & ESCAPE_3BIT_FLAG else 4
        symbol_bits = flags & ~(ESCAPE_PACKED_FLAG | ESCAPE_3BIT_FLAG)
        symbol_bits = symbol_bits or 4
        if symbol_bits not in _SUPPORTED_SYMBOL_BITS:
            raise ValueError("compressed slot symbol width is unsupported")
        if escape_packed and symbol_bits != 3:
            raise ValueError("packed escapes require the 3-bit symbol stream")
        if not escape_packed and escape_symbol_bits != 4:
            raise ValueError("unpacked escapes cannot use the 3-bit token flag")
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
            prefix_end = descriptor.prefix_offset + 4 * (descriptor.tile_count + 1) * (
                2 if escape_packed else 1
            )
            symbol_bytes = (descriptor.scalar_count * symbol_bits + 7) // 8
            symbol_end = descriptor.symbol_offset + symbol_bytes
            low_end = descriptor.low_offset + descriptor.scalar_count
            if escape_packed:
                escape_code_bytes = (
                    descriptor.escape_count * escape_symbol_bits + 7
                ) // 8
                escape_code_end = descriptor.escape_offset + escape_code_bytes
                if escape_code_end > record_end:
                    raise ValueError("packed escape code stream exceeds plane record")
                raw_escape_count = 0
                escape_codes = payload[descriptor.escape_offset : escape_code_end]
                for token in range(descriptor.escape_count):
                    bit_offset = token * escape_symbol_bits
                    byte_offset = bit_offset // 8
                    shift = bit_offset & 7
                    value = escape_codes[byte_offset]
                    if shift + escape_symbol_bits > 8:
                        value |= escape_codes[byte_offset + 1] << 8
                    code = (value >> shift) & ((1 << escape_symbol_bits) - 1)
                    raw_escape_count += int(code == (1 << escape_symbol_bits) - 1)
                escape_end = escape_code_end + raw_escape_count
            else:
                escape_end = descriptor.escape_offset + descriptor.escape_count
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
            symbol_bits=symbol_bits,
            escape_packed=escape_packed,
            escape_symbol_bits=escape_symbol_bits,
        )


@dataclass(frozen=True)
class CompressedSlotRef:
    """One server-resolved physical record for a logical DaseR slot."""

    slot_id: int
    mode: SlotMode
    file_offset: int
    stored_length: int
    raw_hash: bytes
    encoded_hash: bytes

    def to_payload(self) -> dict[str, object]:
        """Return a msgpack-safe payload for scheduler/worker metadata."""
        return {
            "slot_id": self.slot_id,
            "mode": self.mode.name.lower(),
            "file_offset": self.file_offset,
            "stored_length": self.stored_length,
            "raw_hash": self.raw_hash,
            "encoded_hash": self.encoded_hash,
        }


class CompressedStoreIndex:
    """Immutable side index for fixed-envelope compressed KV storage.

    Args:
        geometry: Store and model geometry.
        model_hash: SHA-256 identity of the model configuration.
        codebooks: Plane-major 15-byte high-byte tables.
        entries: Exactly one physical record reference per logical slot.

    Async/thread-safety:
        Instances are immutable after construction and safe for concurrent
        server lookups. ``write`` is an offline single-writer operation.
    """

    def __init__(
        self,
        geometry: CompressedStoreGeometry,
        model_hash: bytes,
        codebooks: bytes,
        entries: Iterable[CompressedSlotRef],
    ) -> None:
        expected_codebooks = geometry.plane_count * CODEBOOK_ENTRIES
        if len(model_hash) != 32:
            raise ValueError("model_hash must contain 32 bytes")
        if len(codebooks) != expected_codebooks:
            raise ValueError(
                "codebooks contain "
                f"{len(codebooks)} bytes, expected {expected_codebooks}"
            )
        refs = tuple(entries)
        if len(refs) != geometry.num_slots:
            raise ValueError("compressed index must contain one entry per slot")
        for expected_slot, ref in enumerate(refs):
            if ref.slot_id != expected_slot:
                raise ValueError("compressed index slot entries must be contiguous")
            if ref.file_offset != expected_slot * geometry.slot_size:
                raise ValueError("compressed slot file offset violates fixed envelopes")
            if (
                ref.stored_length <= 0
                or ref.stored_length > geometry.slot_size
                or ref.stored_length % IO_ALIGNMENT
            ):
                raise ValueError("compressed slot stored length is invalid")
            if len(ref.raw_hash) != 32 or len(ref.encoded_hash) != 32:
                raise ValueError("compressed slot hashes must contain 32 bytes")
            if ref.mode is SlotMode.RAW and ref.stored_length != geometry.slot_size:
                raise ValueError("raw-mode slots must store the complete raw envelope")
        self._geometry = geometry
        self._model_hash = model_hash
        self._codebooks = codebooks
        self._codebook_hash = digest_bytes(codebooks)
        self._entries = refs

    @property
    def geometry(self) -> CompressedStoreGeometry:
        """Return the validated immutable store geometry."""
        return self._geometry

    @property
    def model_hash(self) -> bytes:
        """Return the model-configuration SHA-256 identity."""
        return self._model_hash

    @property
    def codebooks(self) -> bytes:
        """Return plane-major static high-byte codebooks."""
        return self._codebooks

    @property
    def codebook_hash(self) -> bytes:
        """Return the SHA-256 identity of all static codebooks."""
        return self._codebook_hash

    def resolve_slots(
        self, start_slot: int, num_slots: int
    ) -> tuple[CompressedSlotRef, ...]:
        """Resolve an ordered logical slot range into exact physical spans.

        Args:
            start_slot: First logical DaseR slot.
            num_slots: Number of consecutive slots.

        Returns:
            Immutable ordered physical record references.

        Raises:
            ValueError: If the requested range is empty or outside the store.

        Async/thread-safety:
            Read-only and safe for concurrent server requests.
        """
        end_slot = start_slot + num_slots
        if start_slot < 0 or num_slots <= 0 or end_slot > len(self._entries):
            raise ValueError("compressed slot range is outside the store")
        return self._entries[start_slot:end_slot]

    def write(self, path: str | os.PathLike[str]) -> None:
        """Atomically write this side index.

        Args:
            path: Destination side-index path.

        Async/thread-safety:
            Synchronous offline operation. Callers must serialize writers and
            must not invoke it on an asyncio hot path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        codebooks_offset = IO_ALIGNMENT
        entries_offset = align_up(codebooks_offset + len(self._codebooks))
        header = _INDEX_HEADER.pack(
            INDEX_MAGIC,
            FORMAT_VERSION,
            IO_ALIGNMENT,
            IO_ALIGNMENT,
            self._geometry.num_slots,
            self._geometry.slot_size,
            self._geometry.block_tokens,
            self._geometry.num_layers,
            self._geometry.num_kv_heads,
            self._geometry.head_dim,
            self._geometry.dtype_bytes,
            self._geometry.tile_scalars,
            codebooks_offset,
            entries_offset,
            self._model_hash,
            self._codebook_hash,
        )
        with tempfile.NamedTemporaryFile(
            mode="w+b", dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as handle:
            temp_path = Path(handle.name)
            try:
                handle.write(header)
                handle.write(bytes(IO_ALIGNMENT - len(header)))
                handle.write(self._codebooks)
                handle.write(bytes(entries_offset - handle.tell()))
                for ref in self._entries:
                    handle.write(
                        _INDEX_ENTRY.pack(
                            ref.slot_id,
                            int(ref.mode),
                            ref.stored_length,
                            ref.raw_hash,
                            ref.encoded_hash,
                        )
                    )
                handle.flush()
                os.fsync(handle.fileno())
                os.replace(temp_path, target)
            finally:
                if temp_path.exists():
                    temp_path.unlink()
        logger.info(
            "[INDEX] wrote compressed side index with %d slots", len(self._entries)
        )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        expected_geometry: CompressedStoreGeometry | None = None,
        expected_model_hash: bytes | None = None,
    ) -> "CompressedStoreIndex":
        """Load and validate a compressed-store side index.

        Args:
            path: Existing binary side-index path.
            expected_geometry: Optional runtime geometry that must match.
            expected_model_hash: Optional model identity that must match.

        Returns:
            Immutable validated CompressedStoreIndex.

        Raises:
            ValueError: On truncation, unsupported format, hash, or geometry.

        Async/thread-safety:
            Synchronous startup operation. The returned object is immutable.
        """
        with open(path, "rb") as handle:
            return cls._load_handle(
                handle,
                expected_geometry=expected_geometry,
                expected_model_hash=expected_model_hash,
            )

    @classmethod
    def _load_handle(
        cls,
        handle: BinaryIO,
        *,
        expected_geometry: CompressedStoreGeometry | None,
        expected_model_hash: bytes | None,
    ) -> "CompressedStoreIndex":
        header_page = handle.read(IO_ALIGNMENT)
        if len(header_page) != IO_ALIGNMENT:
            raise ValueError("compressed side index header is truncated")
        fields = _INDEX_HEADER.unpack_from(header_page)
        (
            magic,
            version,
            header_bytes,
            alignment,
            num_slots,
            slot_size,
            block_tokens,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype_bytes,
            tile_scalars,
            codebooks_offset,
            entries_offset,
            model_hash,
            codebook_hash,
        ) = fields
        if magic != INDEX_MAGIC or version != FORMAT_VERSION:
            raise ValueError("unsupported compressed side index magic or version")
        if header_bytes != IO_ALIGNMENT or alignment != IO_ALIGNMENT:
            raise ValueError("compressed side index uses unsupported alignment")
        geometry = CompressedStoreGeometry(
            num_slots=num_slots,
            slot_size=slot_size,
            block_tokens=block_tokens,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
            tile_scalars=tile_scalars,
        )
        if expected_geometry is not None and geometry != expected_geometry:
            raise ValueError("compressed side index runtime geometry mismatch")
        if expected_model_hash is not None and model_hash != expected_model_hash:
            raise ValueError("compressed side index model hash mismatch")
        codebook_length = geometry.plane_count * CODEBOOK_ENTRIES
        if codebooks_offset != IO_ALIGNMENT or entries_offset < align_up(
            codebooks_offset + codebook_length
        ):
            raise ValueError("compressed side index offsets are invalid")
        handle.seek(codebooks_offset)
        codebooks = handle.read(codebook_length)
        if (
            len(codebooks) != codebook_length
            or digest_bytes(codebooks) != codebook_hash
        ):
            raise ValueError("compressed side index codebooks are truncated or corrupt")
        handle.seek(entries_offset)
        refs: list[CompressedSlotRef] = []
        for expected_slot in range(num_slots):
            packed = handle.read(_INDEX_ENTRY.size)
            if len(packed) != _INDEX_ENTRY.size:
                raise ValueError("compressed side index entries are truncated")
            slot_id, raw_mode, stored_length, raw_hash, encoded_hash = (
                _INDEX_ENTRY.unpack(packed)
            )
            if slot_id != expected_slot:
                raise ValueError("compressed side index slot IDs are not contiguous")
            try:
                mode = SlotMode(raw_mode)
            except ValueError as exc:
                raise ValueError(
                    "compressed side index contains unknown slot mode"
                ) from exc
            refs.append(
                CompressedSlotRef(
                    slot_id=slot_id,
                    mode=mode,
                    file_offset=slot_id * slot_size,
                    stored_length=stored_length,
                    raw_hash=raw_hash,
                    encoded_hash=encoded_hash,
                )
            )
        return cls(geometry, model_hash, codebooks, refs)


__all__ = [
    "CODEBOOK_ENTRIES",
    "FORMAT_VERSION",
    "IO_ALIGNMENT",
    "KV_PLANES",
    "CompressedSlotRef",
    "CompressedStoreGeometry",
    "CompressedStoreIndex",
    "PlaneDescriptor",
    "SlotHeader",
    "SlotMode",
    "align_up",
    "digest_bytes",
]
