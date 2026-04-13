# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import concurrent.futures
import hashlib
import math
import os
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

# Third Party
import cupy
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

if TYPE_CHECKING:
    # Third Party
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.v1.request import Request
    from vllm.attention import AttentionMetadata

# First Party
from daser.connector.gds_transfer import GDSTransferLayer
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.logging import init_logger, init_perf_logger

logger = init_logger(__name__)
perf = init_perf_logger(__name__)


def hash_tokens(tokens: list[int]) -> str:
    """Return hex SHA256 of token ID sequence.

    Args:
        tokens: list of integer token IDs.

    Returns:
        64-character hex string.
    """
    h = hashlib.sha256()
    for tok in tokens:
        h.update(tok.to_bytes(4, "little"))
    return h.hexdigest()


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
    """

    chunk_key: str
    start_slot: int
    num_slots: int
    block_ids: list[int]
    file_offset: int
    token_count: int


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


@dataclass
class DaserConnectorMeta(KVConnectorMetadata):
    """Metadata passed from scheduler to worker each scheduling step.

    Attributes:
        reqs_to_load: req_id → ReqLoadSpec for cache hits.
        reqs_to_store: req_id → ReqStoreSpec for new chunks to persist.
    """

    reqs_to_load: dict[str, ReqLoadSpec] = field(default_factory=dict)
    reqs_to_store: dict[str, ReqStoreSpec] = field(default_factory=dict)


class DaserConnector(KVConnectorBase_V1):
    """vLLM KVConnectorBase_V1 implementation backed by DaseR.

    Runs in two roles determined by the `role` constructor argument:
    - SCHEDULER: uses IPCClientSync for cache lookup and slot allocation.
    - WORKER: uses GDSTransferLayer for NVMe↔GPU DMA.

    The scheduler side communicates with the DaseR server over a Unix
    socket to resolve cache hits and allocate slots. The worker side
    performs GDS IO layer by layer, enabling overlap between NVMe DMA
    and GPU attention computation.

    Args:
        vllm_config: full VllmConfig from vLLM.
        role: KVConnectorRole.SCHEDULER or KVConnectorRole.WORKER.
        kv_cache_config: optional KV cache configuration (unused).
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Any = None,
    ) -> None:
        self._role = role

        extra: dict[str, Any] = {}
        if (
            hasattr(vllm_config, "kv_transfer_config")
            and vllm_config.kv_transfer_config is not None
        ):
            extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}

        self._socket_path: str = extra.get("socket_path", "/tmp/daser.sock")
        self._store_path: str = extra.get(
            "store_path", "/data/zwt/daser_test/daser.store"
        )
        self._slot_size: int = int(extra.get("slot_size", 0))
        self._block_tokens: int = int(extra.get("block_tokens", 16))
        self._model_id: str = extra.get("model_id", "default")

        if role == KVConnectorRole.SCHEDULER:
            self._ipc_sync = IPCClientSync(self._socket_path)
            self._pending_loads: dict[str, dict[str, Any]] = {}
            self._pending_stores: dict[str, dict[str, Any]] = {}
            self._req_tokens: dict[str, list[int]] = {}
        else:
            self._gds: Optional[GDSTransferLayer] = None
            self._ipc_async = IPCClientAsync(self._socket_path)
            self._kv_caches: dict[str, torch.Tensor] = {}
            self._layer_names: list[str] = []
            self._layer_idx_map: dict[str, int] = {}  # O(1) layer index lookup
            self._meta: Optional[DaserConnectorMeta] = None
            # Futures keyed by layer_name; each resolves when that layer's
            # GDS reads complete. Using concurrent.futures.Future so that
            # the synchronous wait_for_layer_load can call .result() without
            # risk of re-entering an already-running asyncio event loop.
            self._load_futures: dict[str, concurrent.futures.Future] = {}
            self._store_futures: list[concurrent.futures.Future] = []
            self._pending_commits: set[str] = set()
            # Dedicated background event loop for all GDS async IO in the
            # worker role. vLLM's worker may itself run inside an asyncio
            # loop; using a separate background loop avoids run_until_complete
            # re-entrancy errors.
            self._bg_loop = asyncio.new_event_loop()
            self._bg_thread = threading.Thread(
                target=self._run_bg_loop,
                daemon=True,
                name="daser-io",
            )
            self._bg_thread.start()

        logger.info("[CONNECTOR] role=%s socket=%s", role.name, self._socket_path)

    # ------------------------------------------------------------------
    # Scheduler-side methods
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> "tuple[int | None, bool]":
        """Query DaseR for cached KV matching request tokens.

        Aligns the token count to block boundaries, hashes the prefix,
        and calls the DaseR server lookup. On a hit, stores the chunk
        info for later use in update_state_after_alloc.

        Args:
            request: vLLM Request with prompt_token_ids.
            num_computed_tokens: tokens already in vLLM's KV cache.

        Returns:
            (num_external_tokens, is_async) — (0, False) on miss.
        """
        tokens = list(request.prompt_token_ids)
        self._req_tokens[request.request_id] = tokens

        start = num_computed_tokens
        available = len(tokens) - start
        if available < self._block_tokens:
            return 0, False

        aligned = (available // self._block_tokens) * self._block_tokens
        prefix = tokens[: start + aligned]

        try:
            chunks = self._ipc_sync.lookup(prefix, self._model_id)
        except Exception as exc:
            logger.warning("[CONNECTOR] lookup failed: %s", exc)
            return 0, False

        if not chunks:
            return 0, False

        best = chunks[0]
        extra_tokens = best["token_count"] - num_computed_tokens
        if extra_tokens <= 0:
            return 0, False

        self._pending_loads[request.request_id] = best
        logger.debug(
            "[CONNECTOR] cache hit req=%s tokens=%d",
            request.request_id,
            extra_tokens,
        )
        return extra_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        """Record block IDs for requests that will load or store KV.

        Args:
            request: vLLM Request.
            blocks: vLLM KV cache block allocation for this request.
            num_external_tokens: tokens from DaseR (0 if miss).
        """
        req_id = request.request_id
        block_ids: list[int] = list(blocks.block_ids.flatten().tolist())

        if num_external_tokens > 0 and req_id in self._pending_loads:
            chunk = self._pending_loads[req_id]
            num_needed = math.ceil(num_external_tokens / self._block_tokens)
            chunk["block_ids"] = block_ids[:num_needed]
            logger.debug(
                "[CONNECTOR] load blocks req=%s blocks=%s",
                req_id,
                block_ids[:num_needed],
            )
        else:
            tokens = self._req_tokens.get(req_id, [])
            if not tokens:
                return
            aligned = (len(tokens) // self._block_tokens) * self._block_tokens
            if aligned == 0:
                return
            chunk_key = hash_tokens(tokens[:aligned])
            try:
                alloc = self._ipc_sync.alloc_chunk(
                    chunk_key, token_count=aligned, model_id=self._model_id
                )
            except Exception as exc:
                logger.warning("[CONNECTOR] alloc_chunk failed: %s", exc)
                return
            alloc["chunk_key"] = chunk_key
            alloc["block_ids"] = block_ids[: math.ceil(aligned / self._block_tokens)]
            alloc["token_count"] = aligned
            self._pending_stores[req_id] = alloc
            logger.debug("[CONNECTOR] alloc store req=%s key=%s", req_id, chunk_key[:8])

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> DaserConnectorMeta:
        """Package pending load/store specs into connector metadata.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.

        Returns:
            DaserConnectorMeta with reqs_to_load and reqs_to_store.
        """
        meta = DaserConnectorMeta()

        scheduled_ids: set[str] = set()
        for attr in (
            "scheduled_new_reqs",
            "scheduled_resumed_reqs",
            "scheduled_running_reqs",
        ):
            for r in getattr(scheduler_output, attr, []):
                scheduled_ids.add(r.request_id)

        for req_id, chunk in list(self._pending_loads.items()):
            if req_id in scheduled_ids and "block_ids" in chunk:
                meta.reqs_to_load[req_id] = ReqLoadSpec(
                    chunk_key=chunk["chunk_key"],
                    start_slot=chunk["start_slot"],
                    num_slots=chunk["num_slots"],
                    block_ids=chunk["block_ids"],
                    file_offset=chunk["file_offset"],
                    token_count=chunk["token_count"],
                )
                del self._pending_loads[req_id]

        for req_id, alloc in list(self._pending_stores.items()):
            if req_id in scheduled_ids and "block_ids" in alloc:
                meta.reqs_to_store[req_id] = ReqStoreSpec(
                    chunk_key=alloc["chunk_key"],
                    start_slot=alloc["start_slot"],
                    num_slots=alloc["num_slots"],
                    block_ids=alloc["block_ids"],
                    file_offset=alloc["file_offset"],
                    token_count=alloc["token_count"],
                )
                del self._pending_stores[req_id]

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> "tuple[bool, dict[str, Any] | None]":
        """Clean up per-request state after inference completes.

        Args:
            request: finished vLLM Request.
            block_ids: block IDs being freed.

        Returns:
            (False, None) — no async cleanup needed.
        """
        self._req_tokens.pop(request.request_id, None)
        self._pending_loads.pop(request.request_id, None)
        self._pending_stores.pop(request.request_id, None)
        return False, None

    # ------------------------------------------------------------------
    # Worker-side methods
    # ------------------------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register the per-layer KV cache tensors.

        Called once after vLLM allocates the KV cache. Layer names are
        stored in insertion order for index-based IO offset computation.

        Args:
            kv_caches: dict mapping layer_name → KV tensor.
        """
        self._kv_caches = kv_caches
        self._layer_names = list(kv_caches.keys())
        self._layer_idx_map = {name: idx for idx, name in enumerate(self._layer_names)}

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

        if self._gds is None:
            if not os.path.exists(self._store_path):
                size = max(self._slot_size * 1024, 64 * 1024 * 1024)
                parent = os.path.dirname(self._store_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                with open(self._store_path, "wb") as f:
                    f.write(b"\x00" * size)
                logger.info(
                    "[CONNECTOR] pre-allocated store %s (%d bytes)",
                    self._store_path,
                    size,
                )
            self._gds = GDSTransferLayer(self._store_path)
            logger.info("[CONNECTOR] GDS backend=%s", self._gds.backend.value)

    def bind_connector_metadata(self, connector_metadata: DaserConnectorMeta) -> None:
        """Receive scheduler metadata before each forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        self._meta = connector_metadata
        self._load_futures = {}
        self._store_futures = []
        self._pending_commits = set()

    def clear_connector_metadata(self) -> None:
        """Clear metadata after forward pass completes."""
        self._meta = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Submit GDS reads for all cache-hit requests.

        Submits pread for each layer of each matching chunk upfront.
        Stores per-layer concurrent.futures.Future objects for
        wait_for_layer_load to block on.  All IO runs in the worker's
        dedicated background asyncio loop so that this synchronous method
        is safe to call from inside vLLM's own event loop context.

        Args:
            forward_context: vLLM ForwardContext for this forward pass.
        """
        if self._meta is None or not self._meta.reqs_to_load:
            return
        if self._gds is None:
            return

        num_layers = len(self._layer_names)
        if num_layers == 0:
            return
        layer_size = self._slot_size // num_layers

        for layer_idx, layer_name in enumerate(self._layer_names):
            layer_coros = []
            kv_tensor = self._kv_caches.get(layer_name)
            if kv_tensor is None:
                continue

            for spec in self._meta.reqs_to_load.values():
                for slot_i, block_id in enumerate(spec.block_ids):
                    file_offset = (
                        spec.start_slot + slot_i
                    ) * self._slot_size + layer_idx * layer_size

                    if kv_tensor.dim() >= 2:
                        buf = torch.empty(
                            kv_tensor[:, 0].shape,
                            dtype=kv_tensor.dtype,
                            device=kv_tensor.device,
                        )
                    else:
                        buf = torch.empty_like(kv_tensor[0])

                    cp_buf = cupy.asarray(buf.contiguous())
                    nbytes = cp_buf.nbytes

                    async def _load_one(
                        cp=cp_buf,
                        off=file_offset,
                        nb=nbytes,
                        dest=kv_tensor,
                        bid=block_id,
                        src=buf,
                    ) -> int:
                        read = await self._gds.read_into_async(cp, off, nb)
                        if dest.dim() >= 2:
                            dest[:, bid].copy_(src)
                        else:
                            dest[bid].copy_(src)
                        return read

                    layer_coros.append(_load_one())

            if layer_coros:
                self._load_futures[layer_name] = asyncio.run_coroutine_threadsafe(
                    asyncio.gather(*layer_coros),
                    self._bg_loop,
                )

        logger.debug(
            "[CONNECTOR] start_load_kv: %d layer futures submitted",
            len(self._load_futures),
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Block until GDS reads for layer_name complete.

        Called before computing attention for this layer, enabling
        overlap between NVMe DMA and GPU compute on earlier layers.

        Args:
            layer_name: the layer whose KV data must be resident.
        """
        fut = self._load_futures.get(layer_name)
        if fut is not None:
            fut.result()

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        """Submit GDS writes for this layer for all store requests.

        Args:
            layer_name: name of the current attention layer.
            kv_layer: full KV cache tensor for this layer.
            attn_metadata: attention metadata (not directly used).
        """
        if self._meta is None or not self._meta.reqs_to_store:
            return
        if self._gds is None:
            return

        num_layers = len(self._layer_names)
        if num_layers == 0:
            return
        if layer_name not in self._layer_idx_map:
            logger.warning(
                "[CONNECTOR] save_kv_layer: unknown layer %s, skipping", layer_name
            )
            return
        layer_idx = self._layer_idx_map[layer_name]  # O(1) lookup
        layer_size = self._slot_size // num_layers

        for spec in self._meta.reqs_to_store.values():
            self._pending_commits.add(spec.chunk_key)

            for slot_i, block_id in enumerate(spec.block_ids):
                file_offset = (
                    spec.start_slot + slot_i
                ) * self._slot_size + layer_idx * layer_size

                if kv_layer.dim() >= 2:
                    data = kv_layer[:, block_id].contiguous()
                else:
                    data = kv_layer[block_id].contiguous()

                cp_data = cupy.asarray(data)
                nbytes = cp_data.nbytes

                async def _save_one(cp=cp_data, off=file_offset, nb=nbytes) -> int:
                    return await self._gds.write_async(cp, off, nb)

                self._store_futures.append(
                    asyncio.run_coroutine_threadsafe(_save_one(), self._bg_loop)
                )

    def wait_for_save(self) -> None:
        """Block until all pending GDS writes complete, then commit.

        After all writes are confirmed, sends commit_chunk IPC messages
        to make the chunks searchable in the DaseR index. Uses
        concurrent.futures.Future.result() to avoid re-entering an
        already-running asyncio event loop in vLLM's worker context.
        """
        if not self._store_futures:
            return

        for fut in self._store_futures:
            fut.result()
        self._store_futures.clear()

        async def _commit_all() -> None:
            for key in self._pending_commits:
                await self._ipc_async.commit_chunk(key)

        asyncio.run_coroutine_threadsafe(_commit_all(), self._bg_loop).result()
        self._pending_commits.clear()

    def shutdown(self) -> None:
        """Close GDS file handle and stop the background IO loop."""
        if self._gds is not None:
            self._gds.close()
        self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
        self._bg_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_bg_loop(self) -> None:
        """Entry point for the background asyncio IO thread.

        Sets the event loop for this thread and runs it until stop() is
        called from shutdown().
        """
        asyncio.set_event_loop(self._bg_loop)
        self._bg_loop.run_forever()
