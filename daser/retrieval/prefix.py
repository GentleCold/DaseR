# SPDX-License-Identifier: Apache-2.0

# Standard
import hashlib

# First Party
from daser.logging import init_logger
from daser.retrieval.base import RetrievalIndex
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)


def _hash_tokens(tokens: list[int]) -> str:
    """Return a hex SHA256 of the token ID sequence.

    Args:
        tokens: list of integer token IDs.

    Returns:
        64-character hex string.
    """
    h = hashlib.sha256()
    for tok in tokens:
        h.update(tok.to_bytes(4, "little"))
    return h.hexdigest()


class PrefixHashIndex(RetrievalIndex):
    """Exact token-prefix hash retrieval index.

    Stores chunks indexed by SHA256(token_ids). For a lookup query,
    iterates over prefix lengths (at block_token boundaries) from
    longest to shortest and returns the first (longest) match.

    Args:
        block_tokens: vLLM block size in tokens (default 16). Prefix
                      lengths are quantised to multiples of this value.
    """

    def __init__(self, block_tokens: int = 16) -> None:
        self._block_tokens = block_tokens
        self._index: dict[str, ChunkMeta] = {}

    async def lookup(self, tokens: list[int], model_id: str) -> list[ChunkMeta]:
        """Return the longest cached prefix that matches tokens.

        Iterates prefix lengths from len(tokens) down to block_tokens
        in steps of block_tokens, hashing each prefix. Returns the
        first hit (longest match) or an empty list.

        Args:
            tokens: full token sequence to match against.
            model_id: only chunks with this model_id are returned.

        Returns:
            List with at most one ChunkMeta (the longest prefix match).
        """
        n = len(tokens)
        n = (n // self._block_tokens) * self._block_tokens
        while n >= self._block_tokens:
            key = _hash_tokens(tokens[:n])
            meta = self._index.get(key)
            if meta is not None and meta.model_id == model_id:
                logger.debug("[INDEX] prefix hit key=%s matched=%d tokens", key[:8], n)
                return [meta]
            n -= self._block_tokens
        return []

    async def insert(self, meta: ChunkMeta) -> None:
        """Insert a chunk into the prefix index.

        Args:
            meta: ChunkMeta with chunk_key = SHA256(token_ids).
        """
        self._index[meta.chunk_key] = meta
        logger.debug("[INDEX] insert chunk_key=%s", meta.chunk_key[:8])

    async def remove(self, chunk_key: str) -> None:
        """Remove a chunk from the prefix index.

        Args:
            chunk_key: key to remove; silently ignored if not present.
        """
        self._index.pop(chunk_key, None)
        logger.debug("[INDEX] remove chunk_key=%s", chunk_key[:8])
