# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.logging import init_logger
from daser.position.base import PositionEncoder
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)


class FixedOffsetEncoder(PositionEncoder):
    """Position encoder that stores and returns a fixed position offset.

    The offset is assigned at construction time and used for every chunk.
    This is the simplest strategy: all chunks start at the same position,
    which is appropriate when RoPE is re-applied at load time or when
    chunks are always prefixed at position 0.

    Args:
        fixed_offset: position offset assigned to every chunk (default 0).
    """

    def __init__(self, fixed_offset: int = 0) -> None:
        self._offset = fixed_offset

    def assign_offset(self, chunk_key: str, token_count: int) -> int:
        """Return the fixed offset for any new chunk.

        Args:
            chunk_key: not used; present for interface compatibility.
            token_count: not used; present for interface compatibility.

        Returns:
            The fixed offset set at construction time.
        """
        logger.debug(
            "[INDEX] assign_offset chunk_key=%s offset=%d", chunk_key[:8], self._offset
        )
        return self._offset

    def get_offset(self, meta: ChunkMeta) -> int:
        """Return the stored pos_offset from ChunkMeta.

        Args:
            meta: ChunkMeta whose pos_offset was set by assign_offset.

        Returns:
            meta.pos_offset unchanged.
        """
        return meta.pos_offset
