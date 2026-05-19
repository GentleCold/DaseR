# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

K = TypeVar("K")


class ReplacementPolicy(ABC, Generic[K]):
    """Abstract replacement policy for bounded cache tiers.

    Public methods mutate only policy bookkeeping, not the cached data.

    Async/thread-safety:
        Implementations are intended to be called from one asyncio event loop
        or protected by the owner if shared across threads.
    """

    @abstractmethod
    def insert(self, key: K) -> None:
        """Insert or refresh a key.

        Args:
            key: Cache key to track.
        """

    @abstractmethod
    def access(self, key: K) -> None:
        """Mark an existing key as recently used.

        Args:
            key: Cache key that was read or written.
        """

    @abstractmethod
    def remove(self, key: K) -> None:
        """Remove a key from replacement tracking.

        Args:
            key: Cache key to forget.
        """

    @abstractmethod
    def evict(self) -> K | None:
        """Return and remove the next victim key.

        Returns:
            Victim key, or None when the policy is empty.
        """
