# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side cache reuse strategies."""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StoreIntent:
    """Describe one server allocation needed for pending store work."""

    req_id: str
    chunk_key: str
    token_count: int
    block_ids: list[int]


@dataclass(frozen=True)
class StoreIntentPlan:
    """Return store intents together with the next strategy cursor state."""

    intents: tuple[StoreIntent, ...]
    next_key: str
    next_slot: int
    complete: bool
    invalid: bool = False


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
    def plan_store(
        self,
        req_id: str,
        pending_store: PendingStore,
        tokens: list[int],
        pending_store_ids: set[str],
    ) -> StoreIntentPlan:
        """Build store allocation intents once block IDs are known.

        Args:
            req_id: vLLM request ID.
            pending_store: pending store state for this request.
            tokens: full prompt token IDs.
            pending_store_ids: synthetic or base IDs already allocated.

        Returns:
            Immutable allocation intent plan for request lifecycle execution.
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

    def plan_store(
        self,
        req_id: str,
        pending_store: PendingStore,
        tokens: list[int],
        pending_store_ids: set[str],
    ) -> StoreIntentPlan:
        """Plan one store covering the whole aligned prefix.

        Args:
            req_id: vLLM request ID.
            pending_store: pending store state for this request.
            tokens: full prompt token IDs.
            pending_store_ids: existing allocated work IDs.

        Returns:
            One whole-prefix intent, or an invalid plan on key mismatch.
        """
        del pending_store_ids
        requested_tokens = pending_store.token_count
        num_slots = math.ceil(requested_tokens / self._block_tokens)
        chunk_key = pending_store.chunk_key
        if chunk_key != self.store_key(tokens, requested_tokens):
            logger.warning("[CONNECTOR] pending store key mismatch req=%s", req_id[:8])
            return StoreIntentPlan((), chunk_key, num_slots, True, invalid=True)
        return StoreIntentPlan(
            intents=(
                StoreIntent(
                    req_id=req_id,
                    chunk_key=chunk_key,
                    token_count=requested_tokens,
                    block_ids=pending_store.block_ids[:num_slots],
                ),
            ),
            next_key=chunk_key,
            next_slot=num_slots,
            complete=True,
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

    def plan_store(
        self,
        req_id: str,
        pending_store: PendingStore,
        tokens: list[int],
        pending_store_ids: set[str],
    ) -> StoreIntentPlan:
        """Plan one store target for each missing rolling-prefix slot.

        Args:
            req_id: vLLM request ID.
            pending_store: pending store state for this request.
            tokens: full prompt token IDs.
            pending_store_ids: existing allocated work IDs.

        Returns:
            Missing slot intents and the next rolling-prefix cursor state.
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
                if store_id not in pending_store_ids:
                    run.append((slot_i, key))
            slot_i += 1
        intents = tuple(
            StoreIntent(
                req_id=f"{req_id}:store:{store_slot_i}",
                chunk_key=chunk_key,
                token_count=self._block_tokens,
                block_ids=[pending_store.block_ids[store_slot_i]],
            )
            for store_slot_i, chunk_key in run
        )
        if slot_i >= num_slots:
            if pending_store.chunk_key and pending_store.chunk_key != key:
                logger.warning(
                    "[CONNECTOR] pending store key mismatch req=%s", req_id[:8]
                )
                return StoreIntentPlan((), key, slot_i, True, invalid=True)
        return StoreIntentPlan(
            intents=intents,
            next_key=key,
            next_slot=slot_i,
            complete=slot_i >= num_slots,
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
