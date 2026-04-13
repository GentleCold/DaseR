# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod

# First Party
from daser.server.metadata_store import ChunkMeta


class RetrievalIndex(ABC):
    """Pluggable retrieval interface for DaseR's KV cache index.

    Implementations map token sequences to stored ChunkMeta objects.
    The first implementation is PrefixHashIndex (exact prefix hash matching).
    Future implementations may use vector similarity or hybrid strategies.
    """

    @abstractmethod
    async def lookup(self, tokens: list[int], model_id: str) -> list[ChunkMeta]:
        """Find cached chunks matching the given token sequence.

        Args:
            tokens: full token sequence for the request.
            model_id: only chunks with this model_id are returned.

        Returns:
            List of matching ChunkMeta, ordered by match quality
            (longest / best match first). May be empty.
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
