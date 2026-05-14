# SPDX-License-Identifier: Apache-2.0

# Standard
import array
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
import math
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

# Third Party
import cupy
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
import xxhash

if TYPE_CHECKING:
    # Third Party
    from vllm.attention import AttentionMetadata
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.v1.request import Request

# First Party
from daser.connector.gds_transfer import GDSTransferLayer
from daser.connector.ipc_client import IPCClientAsync, IPCClientSync
from daser.logging import init_logger, init_perf_logger

logger = init_logger(__name__)
perf = init_perf_logger(__name__)


def _get_kv_transfer_flag(request: "Request", key: str) -> Any:
    """Return ``request.kv_transfer_params[key]`` if present, else ``None``.

    vLLM exposes the per-request ``kv_transfer_params`` dict supplied by
    the OpenAI completion API. Callers may omit the attribute entirely
    on Request mocks/older vLLM versions, so the lookup is defensive on
    every step and returns ``None`` for any missing/invalid layer.

    Args:
        request: vLLM ``Request`` (or any object exposing the attribute).
        key: connector-specific flag name to extract.

    Returns:
        The value stored under ``key`` in ``kv_transfer_params``, or
        ``None`` if either the attribute or the key is absent.
    """
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    return params.get(key)


def hash_tokens(tokens: list[int]) -> str:
    """Return hex xxh3_128 of token ID sequence.

    Switched from SHA256 for speed (~8× faster for kilobyte-scale inputs
    on commodity CPUs). The hash is only used for cache-key equality in
    PrefixHashIndex, not for any cryptographic property.

    Args:
        tokens: list of integer token IDs.

    Returns:
        32-character hex string.
    """
    # Pack as a contiguous C-int array for a single hash pass; avoids
    # the per-token Python-loop overhead of repeated h.update() calls.
    buf = bytes(array.array("i", tokens))
    return xxhash.xxh3_128(buf).hexdigest()


def _block_ids_for_chunk(
    block_ids: list[int],
    target_token_start: int,
    num_slots: int,
    block_tokens: int,
    max_tokens: int | None = None,
) -> list[int]:
    """Return vLLM block IDs for a chunk's target prompt range.

    Args:
        block_ids: all block IDs allocated to the request.
        target_token_start: token offset where the chunk starts in the prompt.
        num_slots: number of blocks/slots covered by the chunk.
        block_tokens: tokens per vLLM block.
        max_tokens: optional upper bound on accepted external tokens.

    Returns:
        Slice of block_ids for the chunk, or an empty list when the range
        is not block-aligned or exceeds the allocated blocks.
    """
    if target_token_start % block_tokens != 0:
        return []
    target_block_start = target_token_start // block_tokens
    effective_slots = num_slots
    if max_tokens is not None:
        remaining_tokens = max_tokens - target_token_start
        if remaining_tokens <= 0:
            return []
        effective_slots = min(num_slots, math.ceil(remaining_tokens / block_tokens))
    target_block_end = target_block_start + effective_slots
    if target_block_start < 0 or target_block_end > len(block_ids):
        return []
    return block_ids[target_block_start:target_block_end]


def _contiguous_prefix_tokens(
    chunks: list[dict[str, Any]], num_computed_tokens: int
) -> int:
    """Return external tokens covered contiguously after computed tokens.

    Args:
        chunks: server chunk payloads with target_token_start and token_count.
        num_computed_tokens: tokens vLLM already has locally.

    Returns:
        Number of additional contiguous prefix tokens covered by chunks.
    """
    covered_until = num_computed_tokens
    for chunk in sorted(
        chunks,
        key=lambda item: int(item.get("target_token_start", 0)),
    ):
        target_start = int(chunk.get("target_token_start", 0))
        token_count = int(chunk["token_count"])
        target_end = target_start + token_count
        if target_end <= covered_until:
            continue
        if target_start > covered_until:
            break
        covered_until = target_end
    return covered_until - num_computed_tokens


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

    Returns:
        None. ``key_block`` is updated in place.
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


@dataclass
class PendingStore:
    """Scheduler-side state for a prompt KV store that may span steps.

    Attributes:
        chunk_key: xxh3_128 hex of the full block-aligned prompt prefix.
        token_count: number of aligned prompt tokens that must be computed
            before the chunk can be published.
        block_ids: vLLM block IDs covering the prompt prefix seen so far.
    """

    chunk_key: str
    token_count: int
    block_ids: list[int] = field(default_factory=list)


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
        # Initialize base class to set _connector_metadata and other base attrs.
        # This must come first so that has_connector_metadata() works correctly.
        super().__init__(vllm_config, role, kv_cache_config)

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
        self._rope_base: float = 10000.0
        self._rope_rotary_dim: int = 0
        self._rope_is_neox_style: bool = True
        self._rope_delta_scale: float = float(extra.get("rope_delta_scale", -1.0))
        self._load_key_scale: float = float(extra.get("load_key_scale", 1.0))
        self._load_value_scale: float = float(extra.get("load_value_scale", 1.0))
        self._init_rope_config(vllm_config)

        if role == KVConnectorRole.SCHEDULER:
            self._ipc_sync = IPCClientSync(self._socket_path)
            self._pending_loads: dict[str, dict[str, Any]] = {}
            self._pending_stores: dict[str, dict[str, Any]] = {}
            # Pre-allocated chunks returned by match_and_alloc on lookup
            # miss, keyed by req_id until update_state_after_alloc attaches
            # vLLM block IDs and promotes to _pending_stores.
            self._pending_alloc: dict[str, PendingStore] = {}
            self._req_tokens: dict[str, list[int]] = {}
        else:
            self._gds: Optional[GDSTransferLayer] = None
            self._ipc_async = IPCClientAsync(self._socket_path)
            self._kv_caches: dict[str, torch.Tensor] = {}
            self._layer_names: list[str] = []
            self._layer_idx_map: dict[str, int] = {}  # O(1) layer index lookup
            self._meta: Optional[DaserConnectorMeta] = None
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

        Issues a cache lookup RPC per request and records delayed store
        allocation intent on a miss. The actual alloc_chunk RPC is sent
        after vLLM block allocation, when the connector knows how many
        prompt blocks this scheduling step will compute.

        Requests carrying ``kv_transfer_params={"daser_skip_save": True}``
        opt out of the write-back: lookup still runs (so cached doc
        prefixes are loaded), but no chunk is reserved for persisting
        this request's KV.

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
        full_aligned = (len(tokens) // self._block_tokens) * self._block_tokens
        # Per-request opt-out: callers (e.g. the service layer /infer
        # path) set kv_transfer_params={"daser_skip_save": True} when the
        # prompt is single-use and not worth persisting.
        skip_save = bool(_get_kv_transfer_flag(request, "daser_skip_save"))
        if skip_save or full_aligned == 0:
            store_key = ""
        else:
            store_key = hash_tokens(tokens[:full_aligned])

        try:
            # Defer allocation until update_state_after_alloc(), where vLLM has
            # decided how many prompt blocks this scheduling step will really
            # compute. Allocating the full aligned prompt here can expose
            # unwritten KV when chunked prefill only schedules a prefix.
            resp = self._ipc_sync.match_and_alloc(prefix, "", self._model_id)
        except Exception as exc:
            logger.warning("[CONNECTOR] match_and_alloc failed: %s", exc)
            return 0, False

        chunks = resp.get("chunks", [])

        if not chunks:
            if store_key:
                self._pending_alloc[request.request_id] = PendingStore(
                    chunk_key=store_key,
                    token_count=full_aligned,
                )
            logger.info("[CONNECTOR] cache miss req=%s", request.request_id[:8])
            return 0, False

        if len(chunks) == 1:
            best = chunks[0]
            extra_tokens = best["token_count"] - num_computed_tokens
        else:
            extra_tokens = _contiguous_prefix_tokens(chunks, num_computed_tokens)
        if extra_tokens <= 0:
            return 0, False

        available = len(tokens) - num_computed_tokens
        if extra_tokens >= available:
            extra_tokens = (available // self._block_tokens) * self._block_tokens
            if extra_tokens == available:
                extra_tokens -= self._block_tokens
                if extra_tokens <= 0:
                    return 0, False

        if len(chunks) == 1:
            self._pending_loads[request.request_id] = chunks[0]
        else:
            self._pending_loads[request.request_id] = {
                str(i): chunk for i, chunk in enumerate(chunks)
            }

        logger.info(
            "[CONNECTOR] cache hit req=%s chunks=%d prefix_tokens=%d",
            request.request_id[:8],
            len(chunks),
            extra_tokens,
        )
        # Return is_async=False: load happens synchronously during the forward
        # pass via wait_for_layer_load, so the request is scheduled normally
        # in the same step.  is_async=True would place the request in
        # WAITING_FOR_REMOTE_KVS (0 scheduled tokens), which requires a
        # separate get_finished() notification that DaserConnector does not
        # implement.
        return extra_tokens, False

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
        # blocks.blocks is tuple[Sequence[KVCacheBlock], ...]; group 0 is the
        # standard attention KV group. Each KVCacheBlock has a .block_id attr.
        block_ids: list[int] = [blk.block_id for blk in blocks.blocks[0]]

        if req_id in self._pending_loads:
            chunks = self._pending_loads[req_id]
            if "chunk_key" in chunks:
                chunk = chunks
                num_needed = math.ceil(num_external_tokens / self._block_tokens)
                chunk["block_ids"] = block_ids[:num_needed]
                logger.debug(
                    "[CONNECTOR] load blocks req=%s blocks=%s",
                    req_id,
                    block_ids[:num_needed],
                )
                return
            for key, chunk in list(chunks.items()):
                target_token_start = int(chunk.get("target_token_start", 0))
                selected = _block_ids_for_chunk(
                    block_ids=block_ids,
                    target_token_start=target_token_start,
                    num_slots=int(chunk["num_slots"]),
                    block_tokens=self._block_tokens,
                    max_tokens=num_external_tokens,
                )
                if not selected:
                    logger.warning(
                        "[CONNECTOR] skip load req=%s key=%s target=%d slots=%d",
                        req_id[:8],
                        chunk.get("chunk_key", "")[:8],
                        int(chunk.get("target_token_start", 0)),
                        int(chunk["num_slots"]),
                    )
                    del chunks[key]
                    continue
                chunk["block_ids"] = selected
                logger.debug(
                    "[CONNECTOR] load blocks req=%s key=%s blocks=%s",
                    req_id,
                    chunk.get("chunk_key", "")[:8],
                    selected,
                )
        else:
            pending_store = self._pending_alloc.get(req_id)
            if pending_store is None:
                return
            requested_tokens = pending_store.token_count
            pending_store.block_ids = block_ids[
                : math.ceil(requested_tokens / self._block_tokens)
            ]
            self._maybe_allocate_pending_store(req_id, pending_store)

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

        # num_scheduled_tokens is a dict[req_id, int] that covers all
        # scheduled requests (new + cached/resumed + running) in vLLM v1.
        scheduled_ids: set[str] = set(scheduler_output.num_scheduled_tokens.keys())
        self._record_cached_store_blocks(scheduler_output)

        for req_id, chunks in list(self._pending_loads.items()):
            if req_id not in scheduled_ids:
                continue
            if "chunk_key" in chunks:
                chunk = chunks
                if "block_ids" in chunk:
                    meta.reqs_to_load[req_id] = ReqLoadSpec(
                        chunk_key=chunk["chunk_key"],
                        start_slot=chunk["start_slot"],
                        num_slots=chunk["num_slots"],
                        block_ids=chunk["block_ids"],
                        file_offset=chunk["file_offset"],
                        token_count=chunk["token_count"],
                        target_token_start=int(chunk.get("target_token_start", 0)),
                        pos_offset=int(chunk.get("pos_offset", 0)),
                    )
                    del self._pending_loads[req_id]
                continue
            ready = True
            for key, chunk in chunks.items():
                if "block_ids" not in chunk:
                    ready = False
                    continue
                load_id = req_id if len(chunks) == 1 else f"{req_id}:{key}"
                meta.reqs_to_load[load_id] = ReqLoadSpec(
                    chunk_key=chunk["chunk_key"],
                    start_slot=chunk["start_slot"],
                    num_slots=chunk["num_slots"],
                    block_ids=chunk["block_ids"],
                    file_offset=chunk["file_offset"],
                    token_count=chunk["token_count"],
                    target_token_start=int(chunk.get("target_token_start", 0)),
                    pos_offset=int(chunk.get("pos_offset", 0)),
                )
            if ready:
                del self._pending_loads[req_id]

        for req_id, alloc in list(self._pending_stores.items()):
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            should_store = (
                req_id in scheduled_ids
                and scheduled_tokens > 0
                and "block_ids" in alloc
            )
            if should_store:
                meta.reqs_to_store[req_id] = ReqStoreSpec(
                    chunk_key=alloc["chunk_key"],
                    start_slot=alloc["start_slot"],
                    num_slots=alloc["num_slots"],
                    block_ids=alloc["block_ids"],
                    file_offset=alloc["file_offset"],
                    token_count=alloc["token_count"],
                )
                del self._pending_stores[req_id]

        for req_id, spec in meta.reqs_to_load.items():
            logger.info(
                "[CONNECTOR] meta LOAD  req=%s start_slot=%d blocks=%s tokens=%d",
                req_id[:8],
                spec.start_slot,
                spec.block_ids,
                spec.token_count,
            )
        for req_id, spec in meta.reqs_to_store.items():
            logger.info(
                "[CONNECTOR] meta STORE req=%s start_slot=%d blocks=%s tokens=%d",
                req_id[:8],
                spec.start_slot,
                spec.block_ids,
                spec.token_count,
            )
        return meta

    def _record_cached_store_blocks(self, scheduler_output: "SchedulerOutput") -> None:
        """Append blocks from later chunked-prefill steps to store trackers.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.

        Returns:
            None. Mutates pending store trackers for requests that vLLM
            represents as scheduled cached/running requests. Called only on
            the scheduler thread during metadata construction.
        """
        cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
        if cached_reqs is None:
            return

        req_ids = getattr(cached_reqs, "req_ids", [])
        new_block_ids = getattr(cached_reqs, "new_block_ids", [])
        for req_id, block_groups in zip(req_ids, new_block_ids, strict=False):
            pending_store = self._pending_alloc.get(req_id)
            if pending_store is None or block_groups is None:
                continue
            if not block_groups:
                continue
            block_group = block_groups[0]
            if block_group is None:
                continue
            block_ids = list(block_group)
            if req_id in getattr(cached_reqs, "resumed_req_ids", set()):
                pending_store.block_ids = block_ids
            else:
                pending_store.block_ids.extend(block_ids)
            needed_slots = math.ceil(pending_store.token_count / self._block_tokens)
            if len(pending_store.block_ids) > needed_slots:
                pending_store.block_ids = pending_store.block_ids[:needed_slots]
            self._maybe_allocate_pending_store(req_id, pending_store)

    def _maybe_allocate_pending_store(
        self, req_id: str, pending_store: PendingStore
    ) -> None:
        """Allocate a DaseR chunk once a pending store has full KV coverage.

        Args:
            req_id: vLLM request ID being tracked.
            pending_store: store tracker for the request.

        Returns:
            None. Moves the request from ``_pending_alloc`` to
            ``_pending_stores`` when enough block IDs are present. This method
            runs only on the scheduler thread and performs no blocking hot-path
            I/O beyond the allocation IPC.
        """
        requested_tokens = pending_store.token_count
        num_slots = math.ceil(requested_tokens / self._block_tokens)
        if len(pending_store.block_ids) < num_slots:
            return
        tokens = self._req_tokens.get(req_id, [])
        if len(tokens) < requested_tokens:
            return
        chunk_key = pending_store.chunk_key
        if chunk_key != hash_tokens(tokens[:requested_tokens]):
            logger.warning("[CONNECTOR] pending store key mismatch req=%s", req_id[:8])
            self._pending_alloc.pop(req_id, None)
            return
        try:
            alloc = self._ipc_sync.alloc_chunk(
                chunk_key,
                requested_tokens,
                self._model_id,
            )
        except Exception as exc:
            logger.warning("[CONNECTOR] alloc_chunk failed: %s", exc)
            return
        alloc["chunk_key"] = chunk_key
        alloc["token_count"] = requested_tokens
        alloc["num_slots"] = num_slots
        alloc["block_ids"] = pending_store.block_ids[:num_slots]
        self._pending_stores[req_id] = alloc
        self._pending_alloc.pop(req_id, None)
        logger.debug(
            "[CONNECTOR] alloc store req=%s key=%s tokens=%d/%d",
            req_id,
            alloc["chunk_key"][:8],
            requested_tokens,
            requested_tokens,
        )

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
        self._pending_alloc.pop(request.request_id, None)
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

        Also updates the base-class _connector_metadata so that
        has_connector_metadata() returns True during the forward pass.

        Args:
            connector_metadata: DaserConnectorMeta from build_connector_meta.
        """
        super().bind_connector_metadata(connector_metadata)
        self._meta = connector_metadata
        self._store_futures = []
        self._pending_commits = set()
        # Per-request save staging: req_id → contiguous uint8 GPU buffer of
        # size (num_slots * SLOT_SIZE). Filled in save_kv_layer over the
        # NUM_LAYERS calls for this forward; flushed as a single pwrite per
        # request in wait_for_save.
        self._save_staging: dict[str, torch.Tensor] = {}

    def clear_connector_metadata(self) -> None:
        """Clear metadata after forward pass completes."""
        super().clear_connector_metadata()
        self._meta = None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        """Load all KV cache blocks for cache-hit requests.

        Issues one coalesced GDS read per request covering the entire
        contiguous [start_slot, start_slot+num_slots) byte range of its
        chunk on disk. The per-layer-per-block decomposition is done
        afterwards as GPU-local copies into the KV cache tensors.

        Rationale: on-disk layout places slot_i layer_idx at
        offset = (start_slot + slot_i) * SLOT_SIZE + layer_idx * LAYER_SIZE,
        so each chunk is a single contiguous range. Reducing per-chunk
        submissions from (num_layers * num_blocks) to 1 removes ~600×
        Python/FFI overhead on a typical prompt.

        All work is completed before this method returns so correctness
        is preserved under FULL CUDA graph mode, where per-layer Python
        hooks (wait_for_layer_load) are not invoked during replay.

        Args:
            forward_context: vLLM ForwardContext for this forward pass.
        """
        if self._meta is None or not self._meta.reqs_to_load:
            return
        logger.info(
            "[CONNECTOR] start_load_kv: %d reqs to load",
            len(self._meta.reqs_to_load),
        )
        if self._gds is None:
            return

        num_layers = len(self._layer_names)
        if num_layers == 0:
            return
        layer_size = self._slot_size // num_layers

        # One staging buffer per request, sized to the full chunk range.
        # per_req_entries: list of (cp_staging, torch_staging, spec)
        per_req: list[tuple[cupy.ndarray, torch.Tensor, ReqLoadSpec]] = []
        coros: list = []
        sample_tensor = next(iter(self._kv_caches.values()), None)
        if sample_tensor is None:
            return

        for spec in self._meta.reqs_to_load.values():
            num_slots = len(spec.block_ids)
            if num_slots == 0:
                continue
            total_bytes = num_slots * self._slot_size
            staging = torch.empty(
                total_bytes, dtype=torch.uint8, device=sample_tensor.device
            )
            cp_staging = cupy.asarray(staging)

            async def _read(
                cp: cupy.ndarray = cp_staging,
                off: int = spec.start_slot * self._slot_size,
                nb: int = total_bytes,
            ) -> int:
                return await self._gds.read_into_async(cp, off, nb)

            coros.append(_read())
            per_req.append((cp_staging, staging, spec))

        if not coros:
            return

        # Submit all per-request reads concurrently on the background loop
        # and block until each finishes.
        async def _run_all(cs: list) -> list:
            return await asyncio.gather(*cs)

        asyncio.run_coroutine_threadsafe(_run_all(coros), self._bg_loop).result(
            timeout=120.0
        )

        # Copy from each per-request staging buffer into the per-layer KV
        # caches. The staging buffer layout is:
        #   [slot_0 layer_0][slot_0 layer_1]...[slot_0 layer_{L-1}]
        #   [slot_1 layer_0]...[slot_{S-1} layer_{L-1}]
        # so the slice for (slot_i, layer_idx) starts at
        #   slot_i * self._slot_size + layer_idx * layer_size.
        total_copies = 0
        for _, staging, spec in per_req:
            for slot_i, block_id in enumerate(spec.block_ids):
                slot_base = slot_i * self._slot_size
                for layer_idx, layer_name in enumerate(self._layer_names):
                    kv_tensor = self._kv_caches.get(layer_name)
                    if kv_tensor is None:
                        continue
                    if kv_tensor.dim() >= 2:
                        dst = kv_tensor[:, block_id]
                    else:
                        dst = kv_tensor[block_id]
                    offset = slot_base + layer_idx * layer_size
                    src_bytes = staging[offset : offset + dst.nbytes]
                    dst.copy_(src_bytes.view(kv_tensor.dtype).view(dst.shape))
                    if self._load_key_scale != 1.0:
                        dst[0].mul_(self._load_key_scale)
                    if self._load_value_scale != 1.0:
                        dst[1].mul_(self._load_value_scale)
                    if (
                        spec.pos_offset
                        and kv_tensor.dim() >= 5
                        and dst.dim() == 4
                        and dst.shape[0] >= 2
                        and self._rope_rotary_dim > 0
                    ):
                        # vLLM's FlashAttention KV cache uses the inverse
                        # relocation for independently prefetched chunks.
                        _apply_rope_delta_to_key_block(
                            dst[0],
                            delta=round(spec.pos_offset * self._rope_delta_scale),
                            rope_base=self._rope_base,
                            rotary_dim=self._rope_rotary_dim,
                            is_neox_style=self._rope_is_neox_style,
                        )
                    total_copies += 1

        logger.info(
            "[CONNECTOR] start_load_kv: %d reqs, %d GPU copies, %d GDS reads",
            len(per_req),
            total_copies,
            len(coros),
        )

    def wait_for_layer_load(self, layer_name: str) -> None:
        """No-op: all KV loading is done eagerly in start_load_kv.

        start_load_kv now blocks until all GDS reads complete and all
        staging buffers are copied into the KV cache before returning.
        This ensures correctness under FULL CUDA graph mode where vLLM
        replays the attention kernels without re-executing per-layer
        Python hooks.

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
        """Aggregate this layer's KV into each request's save staging buffer.

        No I/O is issued here. Data is copied on the GPU into a per-request
        contiguous staging buffer that matches the on-disk layout:
            staging[slot_i * SLOT_SIZE + layer_idx * LAYER_SIZE : ...]
        which allows wait_for_save to flush each request in a single
        kvikio.pwrite instead of NUM_LAYERS × num_slots submissions.

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

        for req_id, spec in self._meta.reqs_to_store.items():
            self._pending_commits.add(spec.chunk_key)
            num_slots = len(spec.block_ids)
            if num_slots == 0:
                continue

            staging = self._save_staging.get(req_id)
            if staging is None:
                staging = torch.empty(
                    num_slots * self._slot_size,
                    dtype=torch.uint8,
                    device=kv_layer.device,
                )
                self._save_staging[req_id] = staging

            for slot_i, block_id in enumerate(spec.block_ids):
                dst_off = slot_i * self._slot_size + layer_idx * layer_size
                if kv_layer.dim() >= 2:
                    src = kv_layer[:, block_id]
                else:
                    src = kv_layer[block_id]
                dst = staging[dst_off : dst_off + src.nbytes]
                dst.copy_(src.reshape(-1).view(torch.uint8))

    def wait_for_save(self) -> None:
        """Flush all per-request save staging buffers, then commit.

        One kvikio.pwrite per request covers the chunk's full contiguous
        range on disk. All requests' writes are submitted concurrently to
        the background asyncio loop; commit_chunk RPCs are batched in one
        asyncio.gather after writes land.
        """
        if self._meta is None or not self._save_staging:
            return

        coros: list = []
        _keep: list[torch.Tensor] = []  # prevent GC until writes complete
        for req_id, staging in self._save_staging.items():
            spec = self._meta.reqs_to_store.get(req_id)
            if spec is None:
                continue
            cp_staging = cupy.asarray(staging)
            file_offset = spec.start_slot * self._slot_size
            nbytes = staging.nbytes

            async def _write(
                cp: cupy.ndarray = cp_staging,
                off: int = file_offset,
                nb: int = nbytes,
            ) -> int:
                return await self._gds.write_async(cp, off, nb)

            coros.append(_write())
            _keep.append(staging)

        if coros:

            async def _run_all(cs: list) -> list:
                return await asyncio.gather(*cs)

            asyncio.run_coroutine_threadsafe(_run_all(coros), self._bg_loop).result(
                timeout=120.0
            )

        self._save_staging.clear()

        if self._pending_commits:

            async def _commit_all(keys: list[str]) -> None:
                await asyncio.gather(*(self._ipc_async.commit_chunk(k) for k in keys))

            asyncio.run_coroutine_threadsafe(
                _commit_all(list(self._pending_commits)), self._bg_loop
            ).result()
        self._pending_commits.clear()

    def shutdown(self) -> None:
        """Close GDS file handle and stop the background IO loop.

        Only the WORKER role owns _gds and the background thread; this
        method is a no-op when called on the SCHEDULER role instance.
        """
        if self._role != KVConnectorRole.WORKER:
            return
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

    def _init_rope_config(self, vllm_config: "VllmConfig") -> None:
        """Extract default RoPE settings from vLLM model config.

        Args:
            vllm_config: vLLM runtime config passed to the connector.
        """
        model_config = getattr(vllm_config, "model_config", None)
        if model_config is None:
            return
        try:
            head_size = int(model_config.get_head_size())
        except Exception:  # noqa: BLE001
            logger.warning("[CONNECTOR] could not infer RoPE head size")
            return

        hf_text_config = getattr(model_config, "hf_text_config", None)
        rope_parameters = getattr(hf_text_config, "rope_parameters", None) or {}
        if not isinstance(rope_parameters, dict):
            rope_parameters = {}
        model_type = str(getattr(hf_text_config, "model_type", ""))
        if "qwen" in model_type and "rope_theta" not in rope_parameters:
            rope_base = 1000000.0
        else:
            rope_base = float(rope_parameters.get("rope_theta", 10000.0))
        partial = float(rope_parameters.get("partial_rotary_factor", 1.0))
        rotary_dim = int(head_size * partial)

        self._rope_base = rope_base
        self._rope_rotary_dim = rotary_dim
        self._rope_is_neox_style = True
        logger.info(
            "[CONNECTOR] rope base=%s rotary_dim=%d neox=%s",
            self._rope_base,
            self._rope_rotary_dim,
            self._rope_is_neox_style,
        )
        logger.info(
            "[CONNECTOR] load tuning rope_delta_scale=%s key_scale=%s value_scale=%s",
            self._rope_delta_scale,
            self._load_key_scale,
            self._load_value_scale,
        )
