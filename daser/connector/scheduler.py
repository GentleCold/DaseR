# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.v1.request import Request

# First Party
from daser.connector.helpers import PendingStore, hash_tokens
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.logging import init_logger

logger = init_logger(__name__)


def _get_kv_transfer_flag(request: "Request", key: str) -> Any:
    """Return ``request.kv_transfer_params[key]`` if present, else ``None``.

    Args:
        request: vLLM ``Request`` or compatible object.
        key: connector-specific flag name to extract.

    Returns:
        The value under ``key``, or ``None`` when absent.
    """
    params = getattr(request, "kv_transfer_params", None)
    if not isinstance(params, dict):
        return None
    return params.get(key)


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


def _trim_chunk_to_external_window(
    chunk: dict[str, Any],
    block_ids: list[int],
    external_start: int,
    num_external_tokens: int,
    block_tokens: int,
    slot_size: int,
) -> bool:
    """Trim chunk metadata to the external token interval vLLM requested.

    Args:
        chunk: Mutable chunk metadata returned by the server.
        block_ids: Full vLLM block allocation for the request.
        external_start: Token offset where external KV loading begins.
        num_external_tokens: Number of tokens accepted from the connector.
        block_tokens: Tokens per vLLM block.
        slot_size: Bytes per DaseR slot.

    Returns:
        True when the chunk still covers at least one whole KV block.
    """
    if external_start % block_tokens != 0 or num_external_tokens <= 0:
        return False
    target_start = int(chunk.get("target_token_start", 0))
    target_end = target_start + int(chunk["token_count"])
    external_end = external_start + num_external_tokens
    load_start = max(target_start, external_start)
    load_end = min(target_end, external_end)
    load_start = ((load_start + block_tokens - 1) // block_tokens) * block_tokens
    load_end = ((load_end + block_tokens - 1) // block_tokens) * block_tokens
    load_end = min(load_end, target_end)
    if load_end <= load_start:
        return False

    skip_slots = (load_start - target_start) // block_tokens
    num_slots = (load_end - load_start) // block_tokens
    if load_start < external_start:
        return False
    block_start = (load_start - external_start) // block_tokens
    block_end = block_start + num_slots
    if block_start < 0 or block_end > len(block_ids):
        return False

    chunk["start_slot"] = int(chunk["start_slot"]) + skip_slots
    chunk["file_offset"] = int(chunk["file_offset"]) + skip_slots * slot_size
    chunk["num_slots"] = num_slots
    chunk["token_count"] = num_slots * block_tokens
    chunk["target_token_start"] = load_start
    chunk["block_ids"] = block_ids[block_start:block_end]
    return bool(chunk["block_ids"])


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


class SchedulerConnectorMixin:
    """Scheduler-role vLLM connector behavior.

    Async/thread-safety:
        These methods run on vLLM's scheduler thread and use the synchronous
        IPC client owned by the connector instance.
    """

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> "tuple[int | None, bool]":
        """Query DaseR for cached KV matching request tokens.

        Args:
            request: vLLM Request with prompt_token_ids.
            num_computed_tokens: tokens already in vLLM's KV cache.

        Returns:
            (num_external_tokens, is_async) - (0, False) on miss.
        """
        tokens = list(request.prompt_token_ids)
        self._req_tokens[request.request_id] = tokens
        if not getattr(self, "_runtime_config_ready", True):
            self._refresh_runtime_config()

        start = num_computed_tokens
        available = len(tokens) - start
        if available < self._block_tokens:
            return 0, False

        aligned = (available // self._block_tokens) * self._block_tokens
        prefix = tokens[: start + aligned]
        full_aligned = (len(tokens) // self._block_tokens) * self._block_tokens
        skip_save = bool(_get_kv_transfer_flag(request, "daser_skip_save"))
        if skip_save or full_aligned == 0:
            store_key = ""
        else:
            store_key = hash_tokens(tokens[:full_aligned])

        try:
            chunks = self._ipc_sync.lookup(prefix, self._model_id)
        except Exception as exc:
            logger.warning("[CONNECTOR] lookup failed: %s", exc)
            return 0, False

        if not chunks:
            if store_key:
                self._pending_alloc[request.request_id] = PendingStore(
                    chunk_key=store_key,
                    token_count=full_aligned,
                )
            logger.debug("[CONNECTOR] cache miss req=%s", request.request_id[:8])
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
            extra_tokens = available - 1
            if extra_tokens <= 0:
                return 0, False

        if len(chunks) == 1:
            self._pending_loads[request.request_id] = dict(
                chunks[0], num_computed_tokens=num_computed_tokens
            )
        else:
            self._pending_loads[request.request_id] = {
                str(i): dict(chunk, num_computed_tokens=num_computed_tokens)
                for i, chunk in enumerate(chunks)
            }

        logger.debug(
            "[CONNECTOR] cache hit req=%s chunks=%d prefix_tokens=%d",
            request.request_id[:8],
            len(chunks),
            extra_tokens,
        )
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
        block_ids: list[int] = [blk.block_id for blk in blocks.blocks[0]]

        if req_id in self._pending_loads:
            chunks = self._pending_loads[req_id]
            if "chunk_key" in chunks:
                chunk = chunks
                if not _trim_chunk_to_external_window(
                    chunk=chunk,
                    block_ids=block_ids,
                    external_start=int(chunk.get("num_computed_tokens", 0)),
                    num_external_tokens=num_external_tokens,
                    block_tokens=self._block_tokens,
                    slot_size=self._slot_size,
                ):
                    del self._pending_loads[req_id]
                    return
                logger.debug(
                    "[CONNECTOR] load blocks req=%s blocks=%s",
                    req_id,
                    chunk["block_ids"],
                )
                return
            for key, chunk in list(chunks.items()):
                if not _trim_chunk_to_external_window(
                    chunk=chunk,
                    block_ids=block_ids,
                    external_start=int(chunk.get("num_computed_tokens", 0)),
                    num_external_tokens=num_external_tokens,
                    block_tokens=self._block_tokens,
                    slot_size=self._slot_size,
                ):
                    logger.warning(
                        "[CONNECTOR] skip load req=%s key=%s target=%d slots=%d",
                        req_id[:8],
                        chunk.get("chunk_key", "")[:8],
                        int(chunk.get("target_token_start", 0)),
                        int(chunk["num_slots"]),
                    )
                    del chunks[key]
                    continue
                logger.debug(
                    "[CONNECTOR] load blocks req=%s key=%s blocks=%s",
                    req_id,
                    chunk.get("chunk_key", "")[:8],
                    chunk["block_ids"],
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

        if meta.reqs_to_store:
            meta.reqs_to_store = self._filter_live_store_specs(meta.reqs_to_store)

        if logger.isEnabledFor(logging.DEBUG):
            for req_id, spec in meta.reqs_to_load.items():
                logger.debug(
                    "[CONNECTOR] meta LOAD  req=%s start_slot=%d blocks=%d tokens=%d",
                    req_id[:8],
                    spec.start_slot,
                    len(spec.block_ids),
                    spec.token_count,
                )
            for req_id, spec in meta.reqs_to_store.items():
                logger.debug(
                    "[CONNECTOR] meta STORE req=%s start_slot=%d blocks=%d tokens=%d",
                    req_id[:8],
                    spec.start_slot,
                    len(spec.block_ids),
                    spec.token_count,
                )
        return meta

    def _filter_live_store_specs(
        self,
        specs: dict[str, ReqStoreSpec],
    ) -> dict[str, ReqStoreSpec]:
        """Drop store specs whose server allocation was already evicted.

        Args:
            specs: Store specs built for the current scheduler step.

        Returns:
            Specs that still own their allocated server slot ranges.
        """
        try:
            live_keys = self._ipc_sync.live_allocations(
                [
                    {
                        "chunk_key": spec.chunk_key,
                        "start_slot": spec.start_slot,
                        "num_slots": spec.num_slots,
                    }
                    for spec in specs.values()
                ]
            )
        except Exception as exc:
            logger.warning("[CONNECTOR] live_allocations failed: %s", exc)
            return specs
        return {
            req_id: spec
            for req_id, spec in specs.items()
            if spec.chunk_key in live_keys
        }

    def _record_cached_store_blocks(self, scheduler_output: "SchedulerOutput") -> None:
        """Append blocks from later chunked-prefill steps to store trackers.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.
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
            if (
                req_id in getattr(cached_reqs, "resumed_req_ids", set())
                and block_ids[: len(pending_store.block_ids)] == pending_store.block_ids
            ):
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
            (False, None) - no async cleanup needed.
        """
        self._req_tokens.pop(request.request_id, None)
        self._discard_pending_request(request.request_id)
        return False, None
