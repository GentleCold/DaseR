# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata

from daser.compression.format import CODEC_ID, FORMAT_VERSION


@dataclass(frozen=True)
class CompressedLoadSlot:
    """One indexed physical slot record carried to the worker load path.

    Attributes:
        slot_id: Logical fixed-envelope DaseR slot.
        mode: Explicit ``raw`` or ``compressed`` record mode.
        file_offset: Physical aligned offset in ``daser.store``.
        stored_length: Aligned bytes to transfer, excluding envelope tail.
        format_version: Persisted slot format version.
        codec_id: Stable lossless codec family identifier.
        codec_digest: Digest of the codec parameters and codebook.
    """

    slot_id: int
    mode: Literal["raw", "compressed"]
    file_offset: int
    stored_length: int
    format_version: int = FORMAT_VERSION
    codec_id: str = CODEC_ID
    codec_digest: bytes = b""

    def __post_init__(self) -> None:
        if self.slot_id < 0 or self.mode not in ("raw", "compressed"):
            raise ValueError("invalid compressed load slot identity")
        if self.file_offset < 0 or self.stored_length <= 0:
            raise ValueError("invalid compressed load slot byte range")
        if self.format_version != FORMAT_VERSION or self.codec_id != CODEC_ID:
            raise ValueError("unsupported compressed load slot codec")
        if self.codec_digest and len(self.codec_digest) != 32:
            raise ValueError("compressed load slot codec digest must be SHA-256")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CompressedLoadSlot":
        """Validate a server lookup payload for scheduler/worker handoff."""
        return cls(
            slot_id=int(payload["slot_id"]),
            mode=str(payload["mode"]),  # type: ignore[arg-type]
            file_offset=int(payload["file_offset"]),
            stored_length=int(payload["stored_length"]),
            format_version=int(payload.get("format_version", FORMAT_VERSION)),
            codec_id=str(payload.get("codec_id", CODEC_ID)),
            codec_digest=bytes(payload.get("codec_digest", b"")),
        )


@dataclass
class ReqLoadSpec:
    """Load specification for one request.

    Attributes:
        chunk_key: xxh3_128 of the cached token sequence.
        start_slot: first DaseR slot for this chunk.
        num_slots: number of slots in the chunk.
        block_ids: vLLM block IDs allocated to hold the loaded KV.
        file_offset: byte offset of slot 0 in daser.store.
        token_count: number of tokens covered.
        target_token_start: token offset where this chunk starts in the
            current prompt.
        pos_offset: target-aware position offset returned by the server.
        lease_id: Base request ID retaining host-tier bytes, or empty when the
            load follows the ordinary non-prefetch path.
        compressed_slots: Ordered packed slot records in compressed-online
            mode; empty for the raw path.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int
    target_token_start: int = 0
    pos_offset: int = 0
    lease_id: str = ""
    compressed_slots: list[CompressedLoadSlot] = field(default_factory=list)


@dataclass
class ReqStoreSpec:
    """Store specification for one request.

    Attributes:
        chunk_key: xxh3_128 of this request's token sequence.
        start_slot: first DaseR slot allocated for this chunk.
        num_slots: number of slots allocated.
        block_ids: vLLM block IDs whose KV to save.
        file_offset: byte offset of slot 0 in daser.store.
        token_count: number of tokens to store.
        logical_slot_start: first prompt-relative slot represented by this
            allocation. This is distinct from ``start_slot``, the physical
            ring-buffer slot allocated by DaseR.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int
    logical_slot_start: int = 0


@dataclass(frozen=True)
class StoreWriteSpan:
    """One contiguous source-buffer slice to write to the store file.

    Attributes:
        source_offset: Byte offset in the CUDA IPC source buffer.
        nbytes: Number of bytes to write.
        file_offset: Byte offset in the DaseR store file.
        chunk_key: Optional chunk key used by the server to suppress stale
            delayed writes after ring-buffer eviction.
        start_slot: First slot allocated for chunk_key.
        num_slots: Number of slots allocated for chunk_key.
    """

    source_offset: int
    nbytes: int
    file_offset: int
    chunk_key: str = ""
    start_slot: int = -1
    num_slots: int = 0
    logical_slot_start: int = -1
    logical_slot_count: int = 0
    packed: bool = False
    packed_mode: str = ""
    format_version: int = 0
    codec_id: str = ""
    codec_digest: bytes = b""


@dataclass
class DaserConnectorMeta(KVConnectorMetadata):
    """Metadata passed from scheduler to worker each scheduling step.

    Attributes:
        reqs_to_load: req_id -> ReqLoadSpec for cache hits.
        reqs_to_store: req_id -> ReqStoreSpec for new chunks to persist.
        cancelled_store_req_ids: Preempted base IDs whose published, unsent
            worker stores must be discarded before this step's new stores.
    """

    reqs_to_load: dict[str, ReqLoadSpec] = field(default_factory=dict)
    reqs_to_store: dict[str, ReqStoreSpec] = field(default_factory=dict)
    cancelled_store_req_ids: set[str] = field(default_factory=set)
