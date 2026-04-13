# SPDX-License-Identifier: Apache-2.0

# Standard
from abc import ABC, abstractmethod

# First Party
from daser.server.metadata_store import ChunkMeta


class PositionEncoder(ABC):
    """Pluggable position encoding strategy for DaseR KV chunks.

    Manages position offsets so that KV computed at one position can be
    reused when the chunk is loaded at a (potentially different) position.

    The first implementation is FixedOffsetEncoder, which stores the
    position offset at insert time and returns it unchanged at load time.
    Future implementations may support dynamic position remapping.
    """

    @abstractmethod
    def assign_offset(self, chunk_key: str, token_count: int) -> int:
        """Return the position offset to store for this chunk.

        Called at alloc_chunk time, before the chunk is written.

        Args:
            chunk_key: unique key for the chunk being allocated.
            token_count: number of tokens in the chunk.

        Returns:
            Position offset (first position ID) to record in ChunkMeta.
        """
        ...

    @abstractmethod
    def get_offset(self, meta: ChunkMeta) -> int:
        """Return the position offset to apply when loading this chunk.

        Called at load time, after the chunk key is resolved.

        Args:
            meta: ChunkMeta of the chunk being loaded.

        Returns:
            Position offset to use when inserting KV into the model.
        """
        ...
