# SPDX-License-Identifier: Apache-2.0

"""Scan-resistant recency policy with sparse insertion at the MRU end."""

from collections import OrderedDict
from typing import Generic, TypeVar

from daser.replacement.base import ReplacementPolicy

K = TypeVar("K")


class BIPReplacementPolicy(ReplacementPolicy[K], Generic[K]):
    """Retain reused allocations while admitting every new allocation.

    Args:
        mru_interval: Insert one in this many new keys at the MRU end. Other
            keys enter at the LRU end. A value of one gives ordinary LRU.

    Async/thread-safety:
        The cache owner must serialize calls. The policy owns only keys; it
        never releases buffers or decides whether a transfer may proceed.
    """

    def __init__(self, mru_interval: int = 32) -> None:
        if mru_interval <= 0:
            raise ValueError("mru_interval must be positive")
        self._interval = mru_interval
        self._insertions = 0
        self._order: OrderedDict[K, None] = OrderedDict()

    def insert(self, key: K) -> None:
        """Register a new key without refreshing existing shared allocations.

        Args:
            key: Physical allocation identity, possibly published by several
                independently addressable range children.

        Returns:
            None. The caller must serialize metadata access.
        """
        if key in self._order:
            return
        self._insertions = (self._insertions + 1) % self._interval
        self._order[key] = None
        self._order.move_to_end(key, last=self._insertions == 0)

    def access(self, key: K) -> None:
        """Promote a resident key after a real read or in-place update.

        Args:
            key: Resident physical allocation identity.

        Returns:
            None. Unknown keys are ignored; owner serialization is required.
        """
        if key in self._order:
            self._order.move_to_end(key)

    def remove(self, key: K) -> None:
        """Forget a key whose last resident child was removed.

        Args:
            key: Physical allocation identity to forget.

        Returns:
            None. Safe to repeat under the owner's metadata lock.
        """
        self._order.pop(key, None)

    def evict(self) -> K | None:
        """Remove the least protected key for the owner to reclaim.

        Returns:
            Victim identity, or None when empty. The owner remains responsible
            for writer, lease, and DMA barriers before releasing its storage.
        """
        if not self._order:
            return None
        key, _ = self._order.popitem(last=False)
        return key
