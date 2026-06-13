# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side cache reuse strategies."""

# Standard
from abc import ABC, abstractmethod
import math
from typing import Any

# First Party
from daser.config import CACHE_REUSE_CHUNK, CACHE_REUSE_PREFIX
from daser.connector.helpers import (
    ROLLING_PREFIX_SEED,
    PendingStore,
    hash_tokens,
    rolling_prefix_keys,
)
from daser.logging import init_logger

logger = init_logger(__name__)


class CacheReuseStrategy(ABC):
    """Compute store keys and allocate scheduler-side store work.

    Args:
        block_tokens: number of token IDs represented by one KV slot.

    Async/thread-safety:
        Strategy methods run on vLLM's scheduler thread. Implementations keep
        no shared cross-request mutable state.
    """

    def __init__(self, block_tokens: int) -> None:
        self._block_tokens = block_tokens

    @abstractmethod
    def prepare_store(
        self,
        tokens: list[int],
        aligned_tokens: int,
        cached_chunks: list[dict[str, Any]] | None = None,
    ) -> PendingStore | None:
        """Return pending store state for an aligned prompt prefix.

        Args:
            tokens: full prompt token IDs.
            aligned_tokens: number of block-aligned prompt tokens to store.
            cached_chunks: optional contiguous chunks already loaded for this
                prompt.

        Returns:
            Pending store state, or None when no store should be scheduled.
        """

    def ready_to_allocate(self, pending_store: PendingStore) -> bool:
        """Return whether the pending store has enough blocks to allocate.

        Args:
            pending_store: pending scheduler-side store state.

        Returns:
            True when ``allocate_store`` can make progress.
        """
        num_slots = math.ceil(pending_store.token_count / self._block_tokens)
        return len(pending_store.block_ids) >= num_slots

    @abstractmethod
    def allocate_store(
        self,
        owner: Any,
        req_id: str,
        pending_store: PendingStore,
        tokens: list[int],
    ) -> None:
        """Allocate server-side store metadata once block IDs are known.

        Args:
            owner: scheduler connector object with ``_ipc_sync``,
                ``_model_id``, ``_slot_size``, and ``_pending_*`` attributes.
            req_id: vLLM request ID.
            pending_store: pending store state for this request.
            tokens: full prompt token IDs.
        """


class ChunkReuseStrategy(CacheReuseStrategy):
    """Whole-prefix chunk store strategy."""

    def store_key(self, tokens: list[int], aligned_tokens: int) -> str:
        """Return a content hash for the whole aligned prefix.

        Args:
            tokens: full prompt token IDs.
            aligned_tokens: number of block-aligned prompt tokens to store.

        Returns:
            xxh3 token hash for the aligned prefix.
        """
        if aligned_tokens <= 0:
            return ""
        return hash_tokens(tokens[:aligned_tokens])

    def prepare_store(
        self,
        tokens: list[int],
        aligned_tokens: int,
        cached_chunks: list[dict[str, Any]] | None = None,
    ) -> PendingStore | None:
        """Return pending whole-prefix chunk store state.

        Args:
            tokens: full prompt token IDs.
            aligned_tokens: number of block-aligned prompt tokens to store.
            cached_chunks: ignored for chunk stores.

        Returns:
            Pending store state, or None when aligned_tokens is zero.
        """
        del cached_chunks
        chunk_key = self.store_key(tokens, aligned_tokens)
        if not chunk_key:
            return None
        return PendingStore(chunk_key=chunk_key, token_count=aligned_tokens)

    def allocate_store(
        self,
        owner: Any,
        req_id: str,
        pending_store: PendingStore,
        tokens: list[int],
    ) -> None:
        """Allocate one store covering the whole aligned prefix.

        Args:
            owner: scheduler connector object.
            req_id: vLLM request ID.
            pending_store: pending store state for this request.
            tokens: full prompt token IDs.
        """
        requested_tokens = pending_store.token_count
        num_slots = math.ceil(requested_tokens / self._block_tokens)
        chunk_key = pending_store.chunk_key
        if chunk_key != self.store_key(tokens, requested_tokens):
            logger.warning("[CONNECTOR] pending store key mismatch req=%s", req_id[:8])
            owner.drop_pending_alloc(req_id)
            return
        try:
            alloc = owner.allocate_store_chunk(
                chunk_key,
                requested_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[CONNECTOR] alloc_chunk failed: %s", exc)
            return
        if bool(alloc.get("skipped", False)):
            owner.drop_pending_alloc(req_id)
            logger.debug(
                "[CONNECTOR] skip duplicate store req=%s key=%s",
                req_id[:8],
                chunk_key[:8],
            )
            return
        alloc["chunk_key"] = chunk_key
        alloc["token_count"] = requested_tokens
        alloc["num_slots"] = num_slots
        alloc["block_ids"] = pending_store.block_ids[:num_slots]
        owner.set_pending_store(req_id, alloc)
        owner.drop_pending_alloc(req_id)
        logger.debug(
            "[CONNECTOR] alloc store req=%s key=%s tokens=%d/%d",
            req_id,
            alloc["chunk_key"][:8],
            requested_tokens,
            requested_tokens,
        )


class PrefixReuseStrategy(CacheReuseStrategy):
    """Rolling-prefix slot store strategy."""

    def prepare_store(
        self,
        tokens: list[int],
        aligned_tokens: int,
        cached_chunks: list[dict[str, Any]] | None = None,
    ) -> PendingStore | None:
        """Return pending rolling-prefix slot store state.

        Args:
            tokens: full prompt token IDs.
            aligned_tokens: number of block-aligned prompt tokens to store.
            cached_chunks: optional contiguous chunks already loaded for this
                prompt.

        Returns:
            Pending store state that resumes rolling from the latest cached
            slot when possible.
        """
        if aligned_tokens <= 0:
            return None
        cached_tokens = _contiguous_cached_tokens(cached_chunks or [])
        cached_slots = min(cached_tokens // self._block_tokens, aligned_tokens)
        rolling_key = _last_contiguous_chunk_key(
            cached_chunks or [],
            cached_slots,
            self._block_tokens,
        )
        if not rolling_key:
            rolling_key = ROLLING_PREFIX_SEED
            cached_slots = 0
        return PendingStore(
            chunk_key="",
            token_count=aligned_tokens,
            start_slot_index=cached_slots,
            rolling_key=rolling_key,
            rolling_slot_index=cached_slots,
        )

    def ready_to_allocate(self, pending_store: PendingStore) -> bool:
        """Return whether the next missing rolling-prefix slot has a block ID.

        Args:
            pending_store: pending scheduler-side store state.

        Returns:
            True when at least one additional slot store can be allocated.
        """
        return len(pending_store.block_ids) > pending_store.rolling_slot_index

    def allocate_store(
        self,
        owner: Any,
        req_id: str,
        pending_store: PendingStore,
        tokens: list[int],
    ) -> None:
        """Allocate one store target for each missing rolling-prefix slot.

        Args:
            owner: scheduler connector object.
            req_id: vLLM request ID.
            pending_store: pending store state for this request.
            tokens: full prompt token IDs.
        """
        requested_tokens = pending_store.token_count
        num_slots = math.ceil(requested_tokens / self._block_tokens)
        slot_i = pending_store.rolling_slot_index
        key = pending_store.rolling_key or ROLLING_PREFIX_SEED
        keys = rolling_prefix_keys(
            tokens,
            self._block_tokens,
            start_slot=slot_i,
            initial_key=key,
        )

        run: list[tuple[int, str]] = []
        for next_key in keys:
            if slot_i >= num_slots or slot_i >= len(pending_store.block_ids):
                break
            key = next_key
            if slot_i >= pending_store.start_slot_index:
                store_id = f"{req_id}:store:{slot_i}"
                if not owner.has_pending_store(store_id):
                    run.append((slot_i, key))
            slot_i += 1

        if run:
            try:
                allocations = owner.allocate_store_chunks(
                    [
                        {"chunk_key": chunk_key, "token_count": self._block_tokens}
                        for _slot_i, chunk_key in run
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("[CONNECTOR] alloc_chunks failed: %s", exc)
                return
            if len(allocations) != len(run):
                logger.warning(
                    "[CONNECTOR] alloc_chunks returned %d allocations for %d slots",
                    len(allocations),
                    len(run),
                )
                return
            for (store_slot_i, chunk_key), alloc in zip(
                run,
                allocations,
                strict=True,
            ):
                if bool(alloc.get("skipped", False)):
                    continue
                alloc["chunk_key"] = str(alloc.get("chunk_key", chunk_key))
                alloc["token_count"] = self._block_tokens
                alloc["num_slots"] = 1
                alloc["block_ids"] = [pending_store.block_ids[store_slot_i]]
                owner.set_pending_store(f"{req_id}:store:{store_slot_i}", alloc)

        pending_store.rolling_key = key
        pending_store.rolling_slot_index = slot_i
        if slot_i >= num_slots:
            if pending_store.chunk_key and pending_store.chunk_key != key:
                logger.warning(
                    "[CONNECTOR] pending store key mismatch req=%s", req_id[:8]
                )
                owner.drop_pending_alloc(req_id)
                return
            pending_store.chunk_key = key

        if slot_i >= num_slots:
            owner.drop_pending_alloc(req_id)
        if run:
            logger.debug(
                "[CONNECTOR] alloc rolling-prefix stores req=%s slots=%d",
                req_id[:8],
                len(run),
            )


def build_cache_reuse_strategy(
    cache_reuse_mode: str,
    block_tokens: int,
) -> CacheReuseStrategy:
    """Build a scheduler cache reuse strategy.

    Args:
        cache_reuse_mode: either ``"chunk"`` or ``"prefix"``.
        block_tokens: number of token IDs represented by one KV slot.

    Returns:
        CacheReuseStrategy implementation for the selected mode.

    Raises:
        ValueError: if cache_reuse_mode is unknown.
    """
    if cache_reuse_mode == CACHE_REUSE_CHUNK:
        return ChunkReuseStrategy(block_tokens)
    if cache_reuse_mode == CACHE_REUSE_PREFIX:
        return PrefixReuseStrategy(block_tokens)
    raise ValueError(f"unknown cache reuse mode: {cache_reuse_mode}")


def _contiguous_cached_tokens(chunks: list[dict[str, Any]]) -> int:
    """Return token coverage from prompt start for already cached chunks.

    Args:
        chunks: server chunk payloads with target_token_start and token_count.

    Returns:
        Contiguous token coverage from prompt start.
    """
    covered_until = 0
    for chunk in sorted(
        chunks,
        key=lambda item: int(item.get("target_token_start", 0)),
    ):
        start = int(chunk.get("target_token_start", 0))
        token_count = int(chunk["token_count"])
        if start > covered_until:
            break
        covered_until = max(covered_until, start + token_count)
    return covered_until


def _last_contiguous_chunk_key(
    chunks: list[dict[str, Any]],
    cached_slots: int,
    block_tokens: int,
) -> str:
    """Return key for the last contiguous cached slot.

    Args:
        chunks: server chunk payloads with target_token_start and token_count.
        cached_slots: number of contiguous cached slots from prompt start.
        block_tokens: tokens per KV slot.

    Returns:
        Chunk key of the last cached slot, or an empty string when unavailable.
    """
    if cached_slots <= 0:
        return ""
    target_start = (cached_slots - 1) * block_tokens
    for chunk in chunks:
        start = int(chunk.get("target_token_start", 0))
        token_count = int(chunk["token_count"])
        if start <= target_start < start + token_count:
            return str(chunk["chunk_key"])
    return ""
