# SPDX-License-Identifier: Apache-2.0

"""Compression-density replacement policy for exact cached records."""

from collections import OrderedDict
from collections.abc import Callable
from typing import Generic, TypeVar

from daser.replacement.base import ReplacementPolicy

K = TypeVar("K")


class CompressionDensityReplacementPolicy(ReplacementPolicy[K], Generic[K]):
    """Evict the resident allocation with the lowest raw/stored density.

    Args:
        density_provider: Callable returning the raw-equivalent bytes divided
            by stored bytes for a resident allocation.

    Async/thread-safety:
        The policy only mutates local bookkeeping and is not thread-safe. The
        owning L1 cache calls it under its metadata lock. The provider must
        observe the same protected cache state while ``evict`` runs.
    """

    def __init__(self, density_provider: Callable[[K], float]) -> None:
        self._density_provider = density_provider
        self._order: OrderedDict[K, None] = OrderedDict()

    def insert(self, key: K) -> None:
        """Insert or refresh an allocation in the density policy.

        Args:
            key: Allocation identifier to track.
        """
        self._order[key] = None
        self._order.move_to_end(key)

    def access(self, key: K) -> None:
        """Refresh recency without changing the allocation's density.

        Args:
            key: Allocation identifier that was accessed.
        """
        if key in self._order:
            self._order.move_to_end(key)

    def remove(self, key: K) -> None:
        """Forget an allocation that no longer has resident children.

        Args:
            key: Allocation identifier to remove.
        """
        self._order.pop(key, None)

    def evict(self) -> K | None:
        """Return the least useful allocation and remove it from tracking.

        Returns:
            The lowest-density allocation; ties use oldest recency. ``None``
            is returned when no allocation is resident.
        """
        if not self._order:
            return None
        victim = min(
            enumerate(self._order),
            key=lambda item: (self._density_provider(item[1]), item[0]),
        )[1]
        self._order.pop(victim, None)
        return victim
