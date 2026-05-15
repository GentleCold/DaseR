# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.logging import init_logger
from daser.position.base import PositionEncoder
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)


class ChunkPositionEncoder(PositionEncoder):
    """Position encoder for chunk reuse at arbitrary prompt offsets.

    New chunks are stored with an initial offset. At load time, a target
    token start can override the stored offset so reused chunks can be
    placed at their current prompt position.

    Args:
        initial_offset: offset assigned to newly allocated chunks.
    """

    def __init__(self, initial_offset: int = 0) -> None:
        self._initial_offset = initial_offset

    def assign_offset(self, chunk_key: str, token_count: int) -> int:
        """Return the initial offset for a newly allocated chunk.

        Args:
            chunk_key: key of the chunk being allocated.
            token_count: number of tokens in the chunk.

        Returns:
            The configured initial offset.
        """
        logger.debug(
            "[INDEX] chunk assign_offset chunk_key=%s offset=%d tokens=%d",
            chunk_key[:8],
            self._initial_offset,
            token_count,
        )
        return self._initial_offset

    def get_offset(self, meta: ChunkMeta, target_token_start: int | None = None) -> int:
        """Return the target-aware position offset for a loaded chunk.

        Args:
            meta: chunk metadata.
            target_token_start: prompt token offset where the chunk will
                be loaded. When omitted, the stored metadata offset is
                returned.

        Returns:
            target_token_start when provided, otherwise meta.pos_offset.
        """
        if target_token_start is None:
            return meta.pos_offset
        return target_token_start
