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
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int
    target_token_start: int = 0
    pos_offset: int = 0


@dataclass
class ReqStoreSpec:
    """Store specification for one request.

    Attributes:
        chunk_key: SHA256 of this request's token sequence.
        start_slot: first DaseR slot allocated for this chunk.
        num_slots: number of slots allocated.
        block_ids: vLLM block IDs whose KV to save.
        file_offset: byte offset of slot 0 in daser.store.
        token_count: number of tokens to store.
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int


@dataclass(frozen=True)
class StoreWriteSpan:
    """One contiguous slice of step staging to write to the store file.

    Attributes:
        source_offset: Byte offset in the step staging tensor.
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


@dataclass(frozen=True)
class StoreFuture:
    """Background store task plus resources kept alive until completion.

    Attributes:
        future: Background asyncio task submitted with ``run_coroutine_threadsafe``.
        staging: Torch tensor backing the CuPy view used by the task.
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
