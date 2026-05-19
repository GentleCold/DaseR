# SPDX-License-Identifier: Apache-2.0

# Standard
import concurrent.futures
from dataclasses import dataclass, field

# Third Party
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class ReqLoadSpec:
    """Load specification for one request.

    Attributes:
        chunk_key: SHA256 of the cached token sequence.
        start_slot: first DaseR slot for this chunk.
        num_slots: number of slots in the chunk.
        block_ids: vLLM block IDs allocated to hold the loaded KV.
        file_offset: byte offset of slot 0 in daser.store.
        token_count: number of tokens covered.
        target_token_start: token offset where this chunk starts in the
            current prompt.
        pos_offset: target-aware position offset returned by the server.
        residency: server-reported residency state.
        l2_durable: True when the chunk can fall back to SSD L2.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int
    target_token_start: int = 0
    pos_offset: int = 0
    residency: str = "l2_only"
    l2_durable: bool = True


@dataclass
class ReqStoreSpec:
    """Store specification for one request.

    Attributes:
        chunk_key: SHA256 of this request's token sequence.
        num_slots: number of slots required for this chunk.
        block_ids: vLLM block IDs whose KV to save.
        token_count: number of tokens to store.
        start_slot: first DaseR slot allocated for this chunk. Filled by the
            worker-side allocation immediately before transfer.
        file_offset: byte offset of slot 0 in daser.store. Filled by the
            worker-side allocation immediately before transfer.
        residency: server-reported residency state.
        l2_durable: True when the chunk already has durable L2 bytes.
    """

    chunk_key: str
    num_slots: int
    block_ids: list[int]
    token_count: int
    start_slot: int = 0
    file_offset: int = 0
    residency: str = "allocated"
    l2_durable: bool = False


@dataclass(frozen=True)
class StoreWriteSpan:
    """One contiguous slice of step staging to write to the store file.

    Attributes:
        source_offset: Byte offset in the step staging tensor.
        nbytes: Number of bytes to write.
        file_offset: Byte offset in the DaseR store file.
    """

    source_offset: int
    nbytes: int
    file_offset: int


@dataclass(frozen=True)
class StoreChunkWrite:
    """One chunk write captured at save submission time.

    Attributes:
        chunk_key: cache key to publish after write progress.
        source_offset: Byte offset in the step staging tensor.
        nbytes: Number of bytes in the chunk.
        token_count: Number of prompt tokens covered by this chunk.
    """

    chunk_key: str
    source_offset: int
    nbytes: int
    token_count: int


@dataclass(frozen=True)
class StoreFuture:
    """Background store task plus resources kept alive until completion.

    Attributes:
        future: Background asyncio task submitted with ``run_coroutine_threadsafe``.
        staging: Torch tensor backing the submitted transfer task.
        nbytes: Bytes held by ``staging`` for inflight memory accounting.
    """

    future: concurrent.futures.Future[None]
    staging: torch.Tensor
    nbytes: int


@dataclass
class DaserConnectorMeta(KVConnectorMetadata):
    """Metadata passed from scheduler to worker each scheduling step.

    Attributes:
        reqs_to_load: req_id -> ReqLoadSpec for cache hits.
        reqs_to_store: req_id -> ReqStoreSpec for new chunks to persist.
    """

    reqs_to_load: dict[str, ReqLoadSpec] = field(default_factory=dict)
    reqs_to_store: dict[str, ReqStoreSpec] = field(default_factory=dict)
