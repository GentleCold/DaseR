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
from daser.connector.helpers import PendingStore
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.connector.reuse import build_cache_reuse_strategy
from daser.logging import init_logger

logger = init_logger(__name__)


def _base_req_id(req_id: str) -> str:
    """Return the original request ID for synthetic scheduler sub-work IDs.

    Args:
        req_id: Request ID or scheduler-generated sub-work ID.

    Returns:
        Base vLLM request ID.
    """
    return req_id.split(":store:", 1)[0]


def _store_slot_index(req_id: str) -> int | None:
    """Return the rolling-prefix store slot encoded in a synthetic request ID.

    Args:
        req_id: vLLM request ID or ``<req_id>:store:<slot>`` synthetic ID.

    Returns:
        Slot index for synthetic store IDs, or None for regular IDs.
    """
    if ":store:" not in req_id:
        return None
    try:
        return int(req_id.rsplit(":store:", 1)[1])
    except ValueError:
        return None


def _matches_request_or_store_id(req_id: str, base_req_id: str) -> bool:
    """Return whether ``req_id`` belongs to a base request.

    Args:
        req_id: vLLM request ID or synthetic ``<req_id>:store:<slot>`` ID.
        base_req_id: Base vLLM request ID to match.

    Returns:
        True when ``req_id`` is the base request or one of its synthetic store
        entries.
    """
    return req_id == base_req_id or req_id.startswith(f"{base_req_id}:store:")


def _computed_tokens_after_step(
    scheduler_output: "SchedulerOutput",
) -> dict[str, int]:
    """Return per-request token counts that are valid after this step.

    Args:
        scheduler_output: vLLM SchedulerOutput for this step.

    Returns:
        Mapping from request ID to ``num_computed_tokens + scheduled_tokens``.
        Falls back to the scheduled token count when older or test scheduler
        outputs do not expose prior computed-token metadata.
    """
    scheduled = dict(getattr(scheduler_output, "num_scheduled_tokens", {}))
    computed_after = {req_id: int(tokens) for req_id, tokens in scheduled.items()}

    for req_data in getattr(scheduler_output, "scheduled_new_reqs", []) or []:
        req_id = str(getattr(req_data, "req_id", ""))
        if req_id in scheduled:
            computed_after[req_id] = int(
                getattr(req_data, "num_computed_tokens", 0)
            ) + int(scheduled[req_id])

    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached_reqs is not None:
        req_ids = getattr(cached_reqs, "req_ids", [])
        prior_counts = getattr(cached_reqs, "num_computed_tokens", [])
        for req_id, prior in zip(req_ids, prior_counts, strict=False):
            req_id = str(req_id)
            if req_id in scheduled:
                computed_after[req_id] = int(prior) + int(scheduled[req_id])

    return computed_after


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
    block_start = load_start // block_tokens
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
            self._record_external_prefix_cache_miss(available)
            return 0, False

        skip_load = bool(_get_kv_transfer_flag(request, "daser_skip_load"))
        if skip_load:
            logger.debug("[CONNECTOR] skip load req=%s", request.request_id[:8])
            self._record_external_prefix_cache_miss(available)
            full_aligned = (len(tokens) // self._block_tokens) * self._block_tokens
            skip_save = bool(_get_kv_transfer_flag(request, "daser_skip_save"))
            pending_store = (
                None
                if skip_save
                else self._reuse_strategy().prepare_store(tokens, full_aligned)
            )
            if pending_store is not None:
                self._pending_alloc[request.request_id] = pending_store
            return 0, False

        aligned = (available // self._block_tokens) * self._block_tokens
        prefix = tokens[: start + aligned]
        full_aligned = (len(tokens) // self._block_tokens) * self._block_tokens
        skip_save = bool(_get_kv_transfer_flag(request, "daser_skip_save"))

        try:
            chunks = self._lookup_with_external_prefix_metrics(
                prefix,
                self._model_id,
                max(0, len(tokens) - num_computed_tokens),
                num_computed_tokens,
            )
        except Exception as exc:
            logger.warning("[CONNECTOR] lookup failed: %s", exc)
            self._runtime_config_ready = False
            return 0, False

        if not chunks:
            pending_store = (
                None
                if skip_save
                else self._reuse_strategy().prepare_store(tokens, full_aligned)
            )
            if pending_store is not None:
                self._pending_alloc[request.request_id] = pending_store
            logger.debug("[CONNECTOR] cache miss req=%s", request.request_id[:8])
            return 0, False

        extra_tokens = _contiguous_prefix_tokens(chunks, num_computed_tokens)
        if extra_tokens <= 0:
            return 0, False

        pending_store = (
            None
            if skip_save
            else self._reuse_strategy().prepare_store(
                tokens,
                full_aligned,
                chunks,
            )
        )
        if pending_store is not None:
            self._pending_alloc[request.request_id] = pending_store

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
                    self._record_pending_store_blocks(req_id, block_ids)
                    return
                logger.debug(
                    "[CONNECTOR] load blocks req=%s blocks=%s",
                    req_id,
                    chunk["block_ids"],
                )
                self._record_pending_store_blocks(req_id, block_ids)
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
                    logger.debug(
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
        self._record_pending_store_blocks(req_id, block_ids)

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
        self._drop_preempted_pending_state(scheduler_output)
        scheduled_ids: set[str] = set(scheduler_output.num_scheduled_tokens.keys())
        computed_after = _computed_tokens_after_step(scheduler_output)
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
            if scheduled_tokens <= 0 and ":store:" in req_id:
                scheduled_tokens = scheduler_output.num_scheduled_tokens.get(
                    _base_req_id(req_id),
                    0,
                )
            base_req_id = _base_req_id(req_id)
            computed_tokens = computed_after.get(base_req_id, scheduled_tokens)
            slot_index = _store_slot_index(req_id)
            required_tokens = (
                (slot_index + 1) * self._block_tokens
                if slot_index is not None
                else int(alloc["token_count"])
            )
            should_store = (
                base_req_id in scheduled_ids
                and scheduled_tokens > 0
                and computed_tokens >= required_tokens
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

    def _drop_preempted_pending_state(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Discard pending scheduler state whose KV blocks were preempted.

        Args:
            scheduler_output: vLLM SchedulerOutput for this step.

        Async/thread-safety:
            Runs on the scheduler thread before metadata is handed to workers.
        """
        preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", set())
        for req_id in preempted_req_ids:
            base_req_id = str(req_id)
            for pending_req_id in list(self._pending_loads):
                if _matches_request_or_store_id(pending_req_id, base_req_id):
                    self._pending_loads.pop(pending_req_id, None)
            for pending_req_id in list(self._pending_stores):
                if _matches_request_or_store_id(pending_req_id, base_req_id):
                    self._drop_pending_store(pending_req_id)
            for pending_req_id in list(self._pending_alloc):
                if _matches_request_or_store_id(pending_req_id, base_req_id):
                    self._pending_alloc.pop(pending_req_id, None)

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

    def _lookup_with_external_prefix_metrics(
        self,
        tokens: list[int],
        model_id: str,
        queries: int,
        num_computed_tokens: int,
    ) -> list[dict[str, Any]]:
        """Run lookup while passing vLLM external-prefix query count when supported.

        Args:
            tokens: token prefix sent to DaseR lookup.
            model_id: model identifier.
            queries: vLLM external prefix query token count.
            num_computed_tokens: tokens already computed locally by vLLM.

        Returns:
            Matching chunk dicts returned by the IPC client.

        Thread-safety:
            Runs on the scheduler thread and uses the synchronous IPC client.
        """
        try:
            return self._ipc_sync.lookup(
                tokens,
                model_id,
                external_prefix_queries=queries,
                num_computed_tokens=num_computed_tokens,
            )
        except TypeError as exc:
            if "external_prefix_queries" not in str(exc):
                raise
            return self._ipc_sync.lookup(tokens, model_id)

    def _record_external_prefix_cache_miss(self, queries: int) -> None:
        """Record a connector external-prefix miss when lookup is skipped.

        Args:
            queries: vLLM external prefix query token count.

        Returns:
            None.

        Thread-safety:
            Runs on the scheduler thread and uses the synchronous IPC client
            when it supports the diagnostic operation.
        """
        recorder = getattr(self._ipc_sync, "record_external_prefix_cache", None)
        if recorder is None:
            return
        recorder(max(0, int(queries)), 0)

    def _record_pending_store_blocks(self, req_id: str, block_ids: list[int]) -> None:
        """Record request KV blocks for a pending scheduler store.

        Args:
            req_id: vLLM request ID.
            block_ids: KV block IDs allocated for the request.
        """
        pending_store = self._pending_alloc.get(req_id)
        if pending_store is None:
            return
        requested_tokens = pending_store.token_count
        pending_store.block_ids = block_ids[
            : math.ceil(requested_tokens / self._block_tokens)
        ]
        self._maybe_allocate_pending_store(req_id, pending_store)

    def _init_reuse_strategy(self) -> None:
        """Initialize the scheduler cache reuse strategy from current config."""
        self._cache_reuse_strategy = build_cache_reuse_strategy(
            "chunk",
            self._block_tokens,
        )

    def _reuse_strategy(self) -> Any:
        """Return the configured cache reuse strategy.

        Returns:
            Cache reuse strategy initialized from connector runtime config.
        """
        strategy = getattr(self, "_cache_reuse_strategy", None)
        if strategy is None:
            self._init_reuse_strategy()
            strategy = self._cache_reuse_strategy
        return strategy

    def allocate_store_chunk(
        self,
        chunk_key: str,
        token_count: int,
    ) -> dict[str, Any]:
        """Allocate server metadata for a pending scheduler store.

        Args:
            chunk_key: cache key to allocate.
            token_count: number of tokens covered by the allocation.

        Returns:
            Mutable server allocation metadata.
        """
        return self._ipc_sync.alloc_chunk(chunk_key, token_count, self._model_id)

    def allocate_store_chunks(
        self,
        chunks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Allocate server metadata for multiple pending scheduler stores.

        Args:
            chunks: chunk descriptors with chunk_key and token_count.

        Returns:
            Mutable server allocation metadata for each chunk.
        """
        alloc_chunks = getattr(self._ipc_sync, "alloc_chunks", None)
        if alloc_chunks is None:
            return [
                {
                    **self.allocate_store_chunk(
                        str(chunk["chunk_key"]),
                        int(chunk["token_count"]),
                    ),
                    "chunk_key": str(chunk["chunk_key"]),
                }
                for chunk in chunks
            ]
        return alloc_chunks(chunks, self._model_id)

    def set_pending_store(self, req_id: str, alloc: dict[str, Any]) -> None:
        """Record an allocated store for later connector metadata packaging.

        Args:
            req_id: vLLM request ID or synthetic store work ID.
            alloc: mutable allocation metadata.
        """
        self._pending_stores[req_id] = alloc

    def has_pending_store(self, req_id: str) -> bool:
        """Return whether a pending store entry already exists.

        Args:
            req_id: vLLM request ID or synthetic store work ID.

        Returns:
            True when the store is already pending.
        """
        return req_id in self._pending_stores

    def count_pending_stores_for_request(self, req_id: str) -> int:
        """Return number of synthetic slot stores pending for a request.

        Args:
            req_id: base vLLM request ID.

        Returns:
            Count of pending slot-store entries.
        """
        prefix = f"{req_id}:store:"
        return len([key for key in self._pending_stores if key.startswith(prefix)])

    def drop_pending_alloc(self, req_id: str) -> None:
        """Remove pending allocation state for a request.

        Args:
            req_id: vLLM request ID.
        """
        self._pending_alloc.pop(req_id, None)

    def _drop_pending_store(self, req_id: str) -> None:
        """Remove a pending store and release its server writer claim.

        Args:
            req_id: vLLM request ID or synthetic store work ID.
        """
        alloc = self._pending_stores.pop(req_id, None)
        if alloc is None:
            return
        release = getattr(self._ipc_sync, "release_chunk_writer", None)
        if release is None:
            return
        try:
            release(
                str(alloc["chunk_key"]),
                int(alloc["start_slot"]),
                int(alloc["num_slots"]),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CONNECTOR] release_chunk_writer failed: %s", exc)

    def _maybe_allocate_pending_store(
        self, req_id: str, pending_store: PendingStore
    ) -> None:
        """Allocate a DaseR chunk once a pending store has full KV coverage.

        Args:
            req_id: vLLM request ID being tracked.
            pending_store: store tracker for the request.
        """
        requested_tokens = pending_store.token_count
        strategy = self._reuse_strategy()
        if not strategy.ready_to_allocate(pending_store):
            return
        tokens = self._req_tokens.get(req_id, [])
        if len(tokens) < requested_tokens:
            return
        strategy.allocate_store(self, req_id, pending_store, tokens)

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
