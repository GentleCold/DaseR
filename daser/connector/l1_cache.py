# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

# Third Party
import torch

# First Party
from daser.connector.transfer import TransferStats


@dataclass
class PinnedChunkEntry:
    """One chunk stored in pinned L1 memory.

    Attributes:
        chunk_key: cache key for the chunk.
        buffer: contiguous uint8 host tensor.
        nbytes: number of valid bytes in buffer.
        durable: True once the chunk also has an L2 copy.
        durable_pin_count: protection count while L2 write is pending.
        load_pin_count: protection count while a load uses the entry.

    Async/thread-safety:
        Mutated only on the worker data-plane thread or background IO loop that
        owns the cache instance.
    """

    chunk_key: str
    buffer: torch.Tensor
    nbytes: int
    durable: bool
    durable_pin_count: int = 0
    load_pin_count: int = 0

    @property
    def evictable(self) -> bool:
        """Return True when this entry can be evicted by LRU."""
        return self.durable and self.durable_pin_count == 0 and self.load_pin_count == 0


class EvictionPolicy(Protocol):
    """Policy interface for choosing L1 eviction victims."""

    def touch(self, chunk_key: str) -> None:
        """Record an access or insertion for a chunk key."""
        ...

    def remove(self, chunk_key: str) -> None:
        """Remove a chunk key from policy state."""
        ...

    def victim(self, entries: dict[str, PinnedChunkEntry]) -> str | None:
        """Return the next evictable chunk key, or None."""
        ...


class LRUEvictionPolicy:
    """Least-recently-used eviction policy for PinnedL1Cache."""

    def __init__(self) -> None:
        self._order: OrderedDict[str, None] = OrderedDict()

    def touch(self, chunk_key: str) -> None:
        """Mark a chunk as most recently used.

        Args:
            chunk_key: accessed chunk key.
        """
        self._order.pop(chunk_key, None)
        self._order[chunk_key] = None

    def remove(self, chunk_key: str) -> None:
        """Remove a chunk key from the LRU order.

        Args:
            chunk_key: chunk key to remove.
        """
        self._order.pop(chunk_key, None)

    def victim(self, entries: dict[str, PinnedChunkEntry]) -> str | None:
        """Return the oldest evictable chunk key.

        Args:
            entries: active L1 entries by chunk key.

        Returns:
            Evictable chunk key, or None when every entry is protected.
        """
        for chunk_key in self._order:
            entry = entries.get(chunk_key)
            if entry is not None and entry.evictable:
                return chunk_key
        return None


def _default_pinned_allocator(nbytes: int) -> torch.Tensor:
    """Allocate a pinned host uint8 tensor.

    Args:
        nbytes: number of bytes to allocate.

    Returns:
        Pinned CPU tensor.
    """
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


class PinnedL1Cache:
    """Byte-limited chunk cache backed by pinned host memory.

    Args:
        capacity_bytes: hard byte limit for all entries.
        allocator: optional host tensor allocator, primarily for tests.
        eviction_policy: optional eviction policy; defaults to LRU.
        on_evict: optional callback invoked with evicted chunk keys.

    Async/thread-safety:
        Intended to be owned by a single worker transfer layer. External callers
        should serialize access if using it from multiple threads.
    """

    def __init__(
        self,
        capacity_bytes: int,
        allocator: Callable[[int], torch.Tensor] | None = None,
        eviction_policy: EvictionPolicy | None = None,
        on_evict: Callable[[str], None] | None = None,
    ) -> None:
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self._capacity_bytes = capacity_bytes
        self._allocator = allocator or _default_pinned_allocator
        self._policy = eviction_policy or LRUEvictionPolicy()
        self._on_evict = on_evict
        self._entries: dict[str, PinnedChunkEntry] = {}
        self._used_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def used_bytes(self) -> int:
        """Return bytes currently held by L1 entries."""
        return self._used_bytes

    def reserve(
        self,
        chunk_key: str,
        nbytes: int,
        durable: bool,
        durable_pin: bool = False,
    ) -> PinnedChunkEntry:
        """Reserve or replace an L1 entry.

        Args:
            chunk_key: cache key to reserve.
            nbytes: entry size in bytes.
            durable: True when the chunk has an L2 copy.
            durable_pin: True to protect the entry until L2 commit.

        Returns:
            Reserved cache entry with a writable buffer.

        Raises:
            ValueError: if nbytes is invalid or exceeds capacity.
            MemoryError: if no evictable entries can make room.
        """
        if nbytes <= 0:
            raise ValueError("nbytes must be positive")
        if nbytes > self._capacity_bytes:
            raise ValueError("entry exceeds L1 capacity")
        existing = self._entries.get(chunk_key)
        if existing is not None:
            self._remove_entry(chunk_key)
        self._evict_until_available(nbytes)
        buffer = self._allocator(nbytes)
        entry = PinnedChunkEntry(
            chunk_key=chunk_key,
            buffer=buffer,
            nbytes=nbytes,
            durable=durable,
            durable_pin_count=1 if durable_pin else 0,
        )
        self._entries[chunk_key] = entry
        self._used_bytes += nbytes
        self._policy.touch(chunk_key)
        return entry

    def get(self, chunk_key: str) -> PinnedChunkEntry | None:
        """Return an L1 entry and mark it recently used.

        Args:
            chunk_key: cache key to look up.

        Returns:
            Entry on hit, otherwise None.
        """
        entry = self._entries.get(chunk_key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        self._policy.touch(chunk_key)
        return entry

    def pin_for_load(self, chunk_key: str) -> PinnedChunkEntry | None:
        """Pin an entry for in-flight load.

        Args:
            chunk_key: cache key to pin.

        Returns:
            Entry on hit, otherwise None.
        """
        entry = self.get(chunk_key)
        if entry is not None:
            entry.load_pin_count += 1
        return entry

    def release_load_pin(self, chunk_key: str) -> None:
        """Release one in-flight load pin.

        Args:
            chunk_key: cache key to release.
        """
        entry = self._entries.get(chunk_key)
        if entry is not None and entry.load_pin_count > 0:
            entry.load_pin_count -= 1

    def release_durable_pin(self, chunk_key: str) -> None:
        """Release one durable-write pin.

        Args:
            chunk_key: cache key to release.
        """
        entry = self._entries.get(chunk_key)
        if entry is not None and entry.durable_pin_count > 0:
            entry.durable_pin_count -= 1

    def mark_durable(self, chunk_key: str) -> None:
        """Mark an entry as having a durable L2 copy.

        Args:
            chunk_key: cache key to update.
        """
        entry = self._entries.get(chunk_key)
        if entry is not None:
            entry.durable = True

    def remove(self, chunk_key: str) -> None:
        """Remove an entry without invoking the eviction callback.

        Args:
            chunk_key: cache key to remove.
        """
        self._remove_entry(chunk_key)

    def stats(self) -> TransferStats:
        """Return L1 cache counters as TransferStats."""
        return TransferStats(
            l1_hits=self._hits,
            l1_misses=self._misses,
            l1_evictions=self._evictions,
            l1_bytes=self._used_bytes,
        )

    def _evict_until_available(self, nbytes: int) -> None:
        """Evict LRU entries until nbytes can fit.

        Args:
            nbytes: bytes needed by the pending allocation.

        Raises:
            MemoryError: if protected entries leave insufficient capacity.
        """
        while self._used_bytes + nbytes > self._capacity_bytes:
            victim = self._policy.victim(self._entries)
            if victim is None:
                raise MemoryError("no evictable L1 entries")
            self._remove_entry(victim)
            self._evictions += 1
            if self._on_evict is not None:
                self._on_evict(victim)

    def _remove_entry(self, chunk_key: str) -> None:
        """Remove an entry and update accounting.

        Args:
            chunk_key: cache key to remove.
        """
        entry = self._entries.pop(chunk_key, None)
        if entry is None:
            return
        self._used_bytes -= entry.nbytes
        self._policy.remove(chunk_key)
