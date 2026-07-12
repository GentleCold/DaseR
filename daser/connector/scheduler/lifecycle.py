# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlocks
    from vllm.v1.core.scheduler import SchedulerOutput
    from vllm.v1.request import Request

from daser.connector.helpers import PendingStore
from daser.connector.metadata import DaserConnectorMeta, ReqLoadSpec, ReqStoreSpec
from daser.connector.scheduler.planning import (
    _base_req_id,
    _computed_tokens_after_step,
    _contiguous_prefix_tokens,
    _get_kv_transfer_flag,
    _load_spec_from_chunk,
    _matches_request_or_store_id,
    _merge_adjacent_load_specs,
    _store_slot_index,
    _trim_chunk_to_external_window,
)
from daser.connector.scheduler.reuse import build_cache_reuse_strategy
from daser.logging import init_logger

logger = init_logger(__name__)


class RequestLifecycle:
    """Own scheduler request state and synchronous IPC orchestration.

    Async/thread-safety:
        These methods run on vLLM's scheduler thread and use the synchronous
        IPC client owned by the connector instance.
    """

    def __init__(
        self,
        *,
        ipc_client: Any,
        block_tokens: int,
        slot_size: int,
        model_id: str,
        cache_reuse_mode: str,
        runtime_config_ready: bool,
    ) -> None:
        self._ipc_sync = ipc_client
        self._block_tokens = block_tokens
        self._slot_size = slot_size
        self._model_id = model_id
        self._cache_reuse_mode = cache_reuse_mode
        self._runtime_config_ready = runtime_config_ready
        self._cache_reuse_strategy = build_cache_reuse_strategy(
            cache_reuse_mode,
            block_tokens,
        )
        self._pending_loads: dict[str, dict[str, Any]] = {}
        self._pending_stores: dict[str, dict[str, Any]] = {}
        self._pending_alloc: dict[str, PendingStore] = {}
        self._pending_async_saves: set[str] = set()
        self._req_tokens: dict[str, list[int]] = {}

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
            if "chunk_key" in chunks:
                chunk = chunks
                if "block_ids" in chunk:
                    meta.reqs_to_load[req_id] = _load_spec_from_chunk(chunk)
                    del self._pending_loads[req_id]
                continue
            ready = True
            load_specs: list[ReqLoadSpec] = []
            for chunk in chunks.values():
                if "block_ids" not in chunk:
                    ready = False
                    continue
                load_specs.append(_load_spec_from_chunk(chunk))
            merged_specs = _merge_adjacent_load_specs(load_specs, self._slot_size)
            for idx, spec in enumerate(merged_specs):
                load_id = req_id if len(merged_specs) == 1 else f"{req_id}:load:{idx}"
                meta.reqs_to_load[load_id] = spec
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
            pending_async_saves = self._pending_async_save_ids()
            for req_id in meta.reqs_to_store:
                pending_async_saves.add(_base_req_id(req_id))

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

    def refresh_runtime_config(self) -> None:
        """Refresh scheduler geometry and reuse policy from the DaseR server.

        Returns:
            None.

        Async/thread-safety:
            Called on the scheduler thread and performs synchronous control-plane
            IPC during connector initialization or recovery, never worker IO.
        """
        self._refresh_runtime_config()

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
        pending_async_saves = self._pending_async_save_ids()
        for req_id in preempted_req_ids:
            base_req_id = str(req_id)
            pending_async_saves.discard(base_req_id)
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

    def _refresh_runtime_config(self) -> None:
        """Refresh scheduler geometry and reuse policy over owned sync IPC."""
        try:
            config = self._ipc_sync.get_runtime_config()
        except Exception as exc:  # noqa: BLE001
            logger.info("[CONNECTOR] runtime config unavailable: %s", exc)
            return
        self._slot_size = int(config.get("slot_size", self._slot_size))
        block_tokens = int(config.get("block_tokens", self._block_tokens))
        self._model_id = str(config.get("model_id", self._model_id))
        cache_reuse_mode = str(config.get("cache_reuse_mode", self._cache_reuse_mode))
        if (
            cache_reuse_mode != self._cache_reuse_mode
            or block_tokens != self._block_tokens
        ):
            self._cache_reuse_mode = cache_reuse_mode
            self._block_tokens = block_tokens
            self._cache_reuse_strategy = build_cache_reuse_strategy(
                cache_reuse_mode,
                self._block_tokens,
            )
        else:
            self._block_tokens = block_tokens
        self._runtime_config_ready = bool(self._slot_size)

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

    def _discard_pending_request(self, req_id: str) -> None:
        """Clear scheduler-side pending state for a request.

        Args:
            req_id: vLLM request ID.
        """
        self._pending_loads.pop(req_id, None)
        if req_id in self._pending_stores:
            self._drop_pending_store(req_id)
        for pending_req_id in list(self._pending_stores):
            if pending_req_id.startswith(f"{req_id}:store:"):
                self._drop_pending_store(pending_req_id)
        self._pending_alloc.pop(req_id, None)

    def _pending_async_save_ids(self) -> set[str]:
        """Return request IDs whose worker-side saves are still pending.

        Returns:
            Mutable set of base vLLM request IDs.

        Thread-safety:
            Runs on the scheduler thread. The lazy initialization supports
            tests and mixin probes that do not call ``DaserConnector.__init__``.
        """
        pending = getattr(self, "_pending_async_saves", None)
        if pending is None:
            pending = set()
            self._pending_async_saves = pending
        return pending

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
        plan = strategy.plan_store(
            req_id,
            pending_store,
            tokens,
            set(self._pending_stores),
        )
        if plan.invalid:
            self._pending_alloc.pop(req_id, None)
            return
        if plan.intents:
            try:
                if len(plan.intents) == 1 and plan.intents[0].req_id == req_id:
                    intent = plan.intents[0]
                    allocations = [
                        self.allocate_store_chunk(
                            intent.chunk_key,
                            intent.token_count,
                        )
                    ]
                else:
                    allocations = self.allocate_store_chunks(
                        [
                            {
                                "chunk_key": intent.chunk_key,
                                "token_count": intent.token_count,
                            }
                            for intent in plan.intents
                        ]
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CONNECTOR] store allocation failed: %s", exc)
                return
            if len(allocations) != len(plan.intents):
                logger.warning(
                    "[CONNECTOR] allocation returned %d entries for %d intents",
                    len(allocations),
                    len(plan.intents),
                )
                return
            for intent, alloc in zip(plan.intents, allocations, strict=True):
                if bool(alloc.get("skipped", False)):
                    continue
                alloc["chunk_key"] = str(alloc.get("chunk_key", intent.chunk_key))
                alloc["token_count"] = intent.token_count
                alloc["num_slots"] = len(intent.block_ids)
                alloc["block_ids"] = intent.block_ids
                self._pending_stores[intent.req_id] = alloc
        pending_store.rolling_key = plan.next_key
        pending_store.rolling_slot_index = plan.next_slot
        if plan.complete:
            pending_store.chunk_key = plan.next_key
            self._pending_alloc.pop(req_id, None)

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
            (True, None) when DaseR is still storing this request's KV blocks,
            otherwise (False, None).
        """
        del block_ids
        if request.request_id in self._pending_async_save_ids():
            return True, None
        self._req_tokens.pop(request.request_id, None)
        self._discard_pending_request(request.request_id)
        return False, None

    def update_connector_output(self, connector_output: Any) -> None:
        """Update scheduler state from worker-side transfer completions.

        Args:
            connector_output: vLLM KVConnectorOutput carrying finished request
                IDs from workers.

        Async/thread-safety:
            Runs on vLLM's scheduler thread after worker connector polling.
        """
        pending_async_saves = self._pending_async_save_ids()
        for req_id in getattr(connector_output, "finished_sending", None) or ():
            pending_async_saves.discard(req_id)
            self._req_tokens.pop(req_id, None)
            self._discard_pending_request(req_id)
        for req_id in getattr(connector_output, "finished_recving", None) or ():
            if req_id not in self._pending_alloc:
                self._req_tokens.pop(req_id, None)
            for pending_req_id in list(self._pending_loads):
                if _matches_request_or_store_id(pending_req_id, req_id):
                    self._pending_loads.pop(pending_req_id, None)
