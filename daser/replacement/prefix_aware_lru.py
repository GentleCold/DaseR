# SPDX-License-Identifier: Apache-2.0

"""Prefix-aware LRU replacement policy."""

# Standard
from collections.abc import Callable, Hashable, Iterable
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Generic, TypeVar

# First Party
from daser.replacement.base import ReplacementPolicy

K = TypeVar("K")


@dataclass(frozen=True, order=True)
class _Order:
    epoch: int
    prefix_rank: int
    sequence: int


class PrefixAwareLRUReplacementPolicy(ReplacementPolicy[K], Generic[K]):
    """LRU policy that ages request suffix keys before prefix keys.

    Async/thread-safety:
        Not thread-safe by itself. Owners should call it from one event loop or
        guard it with their own lock.
    """

    def __init__(self) -> None:
        self._clock = 0
        self._sequence = 0
        self._orders: dict[K, _Order] = {}
        self._heap: list[tuple[int, int, int, K]] = []
        self._group_epochs: dict[Hashable, int] = {}
        self._key_groups: dict[K, Hashable] = {}
        self._group_counts: dict[Hashable, int] = {}

    def insert(self, key: K) -> None:
        """Insert or refresh a standalone key.

        Args:
            key: Cache key to track.
        """
        self._set_order(key, self._next_epoch(), 0, None)

    def insert_prefix(self, key: K, group: Hashable, prefix_index: int) -> None:
        """Insert or refresh one key using prefix-aware group ordering.

        Args:
            key: Cache key to track.
            group: Stable request/chunk identity shared by sibling prefix keys.
            prefix_index: Zero-based slot index inside the request prefix.
        """
        epoch = self._group_epochs.get(group)
        if epoch is None:
            epoch = self._next_epoch()
            self._group_epochs[group] = epoch
        self._set_order(key, epoch, -int(prefix_index), group)

    def access(self, key: K) -> None:
        """Mark an existing key as recently used.

        Args:
            key: Cache key that was read or written.
        """
        if key in self._orders:
            self._set_order(key, self._next_epoch(), 0, None)

    def access_prefix(self, keys: Iterable[tuple[K, int]]) -> None:
        """Refresh sibling prefix keys with suffix-before-prefix aging.

        Args:
            keys: ``(key, prefix_index)`` pairs in any order.
        """
        present = [
            (key, int(prefix_index))
            for key, prefix_index in keys
            if key in self._orders
        ]
        if not present:
            return
        epoch = self._next_epoch()
        for key, prefix_index in present:
            self._set_order(key, epoch, -prefix_index, None)

    def remove(self, key: K) -> None:
        """Remove a key from replacement tracking.

        Args:
            key: Cache key to forget.
        """
        if key not in self._orders:
            return
        self._orders.pop(key, None)
        self._drop_key_group(key)

    def evict(self) -> K | None:
        """Return and remove the least-recently-used key.

        Returns:
            Victim key, or None when empty.
        """
        while self._heap:
            epoch, prefix_rank, sequence, key = heappop(self._heap)
            order = self._orders.get(key)
            if order != _Order(epoch, prefix_rank, sequence):
                continue
            self.remove(key)
            return key
        return None

    def evict_matching(self, predicate: Callable[[K], bool]) -> K | None:
        """Return and remove the oldest key accepted by ``predicate``.

        Args:
            predicate: Function returning True for eligible victim keys.

        Returns:
            Victim key, or None when no tracked key matches.
        """
        victims = [
            (order, key) for key, order in self._orders.items() if predicate(key)
        ]
        victim = min(victims) if victims else None
        if victim is None:
            return None
        _order, key = victim
        self.remove(key)
        return key

    def _next_epoch(self) -> int:
        """Return the next recency epoch."""
        self._clock += 1
        return self._clock

    def _set_order(
        self,
        key: K,
        epoch: int,
        prefix_rank: int,
        group: Hashable | None,
    ) -> None:
        """Install a key order and push a lazy heap record."""
        self._drop_key_group(key)
        self._sequence += 1
        order = _Order(epoch, prefix_rank, self._sequence)
        self._orders[key] = order
        if group is not None:
            self._key_groups[key] = group
            self._group_counts[group] = self._group_counts.get(group, 0) + 1
        heappush(self._heap, (order.epoch, order.prefix_rank, order.sequence, key))

    def _drop_key_group(self, key: K) -> None:
        """Remove a key from its prefix group accounting."""
        group = self._key_groups.pop(key, None)
        if group is None:
            return
        count = self._group_counts.get(group, 0) - 1
        if count > 0:
            self._group_counts[group] = count
            return
        self._group_counts.pop(group, None)
        self._group_epochs.pop(group, None)
