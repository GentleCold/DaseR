# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass(frozen=True)
class CompressedLoadSlot:
    """One indexed physical slot record carried to the worker load path.

    Attributes:
        slot_id: Logical fixed-envelope DaseR slot.
        mode: Explicit ``raw`` or ``compressed`` record mode.
        file_offset: Physical aligned offset in ``daser.store``.
        stored_length: Aligned bytes to transfer, excluding envelope tail.
    """

    slot_id: int
    mode: Literal["raw", "compressed"]
    file_offset: int
    stored_length: int

    def __post_init__(self) -> None:
        if self.slot_id < 0 or self.mode not in ("raw", "compressed"):
            raise ValueError("invalid compressed load slot identity")
        if self.file_offset < 0 or self.stored_length <= 0:
            raise ValueError("invalid compressed load slot byte range")

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CompressedLoadSlot":
        """Validate a server lookup payload for scheduler/worker handoff."""
        return cls(
            slot_id=int(payload["slot_id"]),
            mode=str(payload["mode"]),  # type: ignore[arg-type]
            file_offset=int(payload["file_offset"]),
            stored_length=int(payload["stored_length"]),
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
        compressed_slots: Ordered physical slot records in read-only compressed
            mode; empty for the existing raw path.
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


@dataclass
class DaserConnectorMeta(KVConnectorMetadata):
    """Metadata passed from scheduler to worker each scheduling step.

    Attributes:
        reqs_to_load: req_id -> ReqLoadSpec for cache hits.
        reqs_to_store: req_id -> ReqStoreSpec for new chunks to persist.
    """

    reqs_to_load: dict[str, ReqLoadSpec] = field(default_factory=dict)
    reqs_to_store: dict[str, ReqStoreSpec] = field(default_factory=dict)
