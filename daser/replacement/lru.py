# SPDX-License-Identifier: Apache-2.0

# Standard
from collections import OrderedDict
from typing import Generic, TypeVar

# First Party
from daser.replacement.base import ReplacementPolicy

K = TypeVar("K")


class LRUReplacementPolicy(ReplacementPolicy[K], Generic[K]):
    """Least-recently-used replacement policy.

    The left side of the internal order is oldest; the right side is newest.

    Async/thread-safety:
        Not thread-safe by itself. Owners should call it from one event loop or
        guard it with their own lock.
    """

    def __init__(self) -> None:
        self._order: OrderedDict[K, None] = OrderedDict()

    def insert(self, key: K) -> None:
        """Insert or refresh a key.

        Args:
            key: Cache key to track.
        """
        self._order[key] = None
        self._order.move_to_end(key)

    def access(self, key: K) -> None:
        """Mark an existing key as recently used.

        Args:
            key: Cache key that was read or written.
        """
        if key in self._order:
            self._order.move_to_end(key)

    def remove(self, key: K) -> None:
        """Remove a key from replacement tracking.

        Args:
            key: Cache key to forget.
        """
        self._order.pop(key, None)

    def evict(self) -> K | None:
        """Return and remove the least-recently-used key.

        Returns:
            Victim key, or None when empty.
        """
        if not self._order:
            return None
        key, _ = self._order.popitem(last=False)
        return key
