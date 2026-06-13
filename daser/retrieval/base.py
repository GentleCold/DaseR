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
    """

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

    @abstractmethod
    async def insert(self, meta: ChunkMeta) -> None:
        """Add a committed chunk to the retrieval index.

        Args:
            meta: ChunkMeta describing the stored chunk.
        """
        ...

    @abstractmethod
    async def remove(self, chunk_key: str) -> None:
        """Remove an evicted chunk from the retrieval index.

        Args:
            chunk_key: key of the chunk to remove.
        """
        ...
