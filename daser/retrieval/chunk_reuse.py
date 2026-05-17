# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.logging import init_logger
from daser.retrieval.base import RetrievalIndex, RetrievalMatch
from daser.retrieval.prefix import _hash_tokens
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)


class ChunkReuseIndex(RetrievalIndex):
    """Block-aligned chunk-window retrieval index.

    Unlike PrefixHashIndex, this implementation does not require a match
    to start at token 0. It scans block-aligned prompt windows using the
    token counts of committed chunks and returns every full chunk hit.

    Args:
        block_tokens: vLLM block size in tokens.
    """

    def __init__(self, block_tokens: int = 16) -> None:
        self._block_tokens = block_tokens
        self._index: dict[str, ChunkMeta] = {}

    async def lookup(self, tokens: list[int], model_id: str) -> list[RetrievalMatch]:
        """Return block-aligned cached chunks found inside ``tokens``.

        Args:
            tokens: full token sequence to scan.
            model_id: only chunks with this model_id are returned.

        Returns:
            Retrieval matches ordered by target token start.
        """
        matches: list[RetrievalMatch] = []
        seen_keys: set[str] = set()
        token_counts = sorted(
            {meta.token_count for meta in self._index.values()},
            reverse=True,
        )
        for start in range(0, len(tokens), self._block_tokens):
            for token_count in token_counts:
                end = start + token_count
                if end > len(tokens):
                    continue
                key = _hash_tokens(tokens[start:end])
                meta = self._index.get(key)
                if meta is None or meta.model_id != model_id:
                    continue
                if key in seen_keys:
                    continue
                matches.append(RetrievalMatch(meta=meta, target_token_start=start))
                seen_keys.add(key)
                logger.debug(
                    "[INDEX] chunk hit key=%s start=%d tokens=%d",
                    key[:8],
                    start,
                    token_count,
                )
                break
        return matches

    async def insert(self, meta: ChunkMeta) -> None:
        """Insert a committed chunk into the chunk reuse index.

        Args:
            meta: committed chunk metadata.
        """
        self._index[meta.chunk_key] = meta
        logger.debug("[INDEX] chunk insert key=%s", meta.chunk_key[:8])

    async def remove(self, chunk_key: str) -> None:
        """Remove an evicted chunk from the chunk reuse index.

        Args:
            chunk_key: key to remove; ignored when absent.
        """
        self._index.pop(chunk_key, None)
        logger.debug("[INDEX] chunk remove key=%s", chunk_key[:8])
