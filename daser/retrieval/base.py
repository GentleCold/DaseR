# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass

# First Party
from daser.server.metadata_store import ChunkMeta


@dataclass(frozen=True)
class RetrievalMatch:
    """One retrieval hit plus its target prompt position.

    Attributes:
        meta: stored chunk metadata for the cache hit.
        target_token_start: token offset in the lookup prompt where the
            chunk should be loaded.
    """

    meta: ChunkMeta
    target_token_start: int = 0


class RetrievalIndex(ABC):
    """Pluggable retrieval interface for DaseR's KV cache index.

    Implementations map token sequences to stored ChunkMeta objects.
    PrefixHashIndex uses chained rolling-prefix keys for slot-granular exact
    prefix reuse; ChunkReuseIndex matches block-aligned document chunks at
    arbitrary prompt offsets. Future implementations may use vector similarity
    or hybrid strategies.

    The base provides concrete ``insert``/``remove`` operating on a
    subclass-owned ``_index`` dict keyed by ``chunk_key``. Subclasses keep
    ``lookup`` abstract, initialize ``self._index`` in ``__init__``, and may
    maintain secondary structures through the ``_on_insert``/``_on_remove``
    hooks.
    """

    _index: dict[str, ChunkMeta]

    @abstractmethod
    async def lookup(self, tokens: list[int], model_id: str) -> list[RetrievalMatch]:
        """Find cached chunks matching the given token sequence.

        Args:
            tokens: full token sequence for the request.
            model_id: only chunks with this model_id are returned.

        Returns:
            List of retrieval matches ordered by implementation-specific
            reuse order. May be empty.
        """
        ...

    async def insert(self, meta: ChunkMeta) -> None:
        """Add a committed chunk to the retrieval index.

        Args:
            meta: ChunkMeta describing the stored chunk.

        Async/thread-safety:
            Records ``meta`` in the primary index and notifies ``_on_insert``.
        """
        self._index[meta.chunk_key] = meta
        self._on_insert(meta)

    async def remove(self, chunk_key: str) -> None:
        """Remove an evicted chunk from the retrieval index.

        Args:
            chunk_key: key of the chunk to remove; ignored when absent.

        Async/thread-safety:
            Drops the chunk from the primary index and notifies ``_on_remove``.
        """
        meta = self._index.pop(chunk_key, None)
        self._on_remove(chunk_key, meta)

    def _on_insert(self, meta: ChunkMeta) -> None:  # noqa: B027
        """Update subclass secondary structures after a primary insert.

        Args:
            meta: chunk metadata just inserted into the primary index.
        """

    def _on_remove(self, chunk_key: str, meta: ChunkMeta | None) -> None:  # noqa: B027
        """Update subclass secondary structures after a primary removal.

        Args:
            chunk_key: key removed from the primary index.
            meta: removed chunk metadata, or None when the key was absent.
        """
