# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.connector.helpers import hash_tokens
from daser.logging import init_logger
from daser.retrieval.base import RetrievalIndex, RetrievalMatch
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
        self._by_token_count: dict[int, dict[str, ChunkMeta]] = {}
        self._token_counts_desc: list[int] = []

    async def lookup(self, tokens: list[int], model_id: str) -> list[RetrievalMatch]:
        """Return block-aligned cached chunks found inside ``tokens``.

        Args:
            tokens: full token sequence to scan.
            model_id: only chunks with this model_id are returned.

        Returns:
            Retrieval matches ordered by target token start.
        """
        matches: list[RetrievalMatch] = []
        token_counts = self._token_counts_desc
        start = 0
        while start < len(tokens):
            matched_tokens = 0
            for token_count in token_counts:
                end = start + token_count
                if end > len(tokens):
                    continue
                key = hash_tokens(tokens[start:end])
                meta = self._by_token_count[token_count].get(key)
                if meta is None or meta.model_id != model_id:
                    continue
                matches.append(RetrievalMatch(meta=meta, target_token_start=start))
                logger.debug(
                    "[INDEX] chunk hit key=%s start=%d tokens=%d",
                    key[:8],
                    start,
                    token_count,
                )
                matched_tokens = token_count
                break
            start += matched_tokens if matched_tokens else self._block_tokens
        return matches

    async def insert(self, meta: ChunkMeta) -> None:
        """Insert a committed chunk into the chunk reuse index.

        Args:
            meta: committed chunk metadata.
        """
        self._index[meta.chunk_key] = meta
        bucket = self._by_token_count.setdefault(meta.token_count, {})
        bucket[meta.chunk_key] = meta
        self._refresh_token_counts()
        logger.debug("[INDEX] chunk insert key=%s", meta.chunk_key[:8])

    async def remove(self, chunk_key: str) -> None:
        """Remove an evicted chunk from the chunk reuse index.

        Args:
            chunk_key: key to remove; ignored when absent.
        """
        meta = self._index.pop(chunk_key, None)
        if meta is not None:
            bucket = self._by_token_count.get(meta.token_count)
            if bucket is not None:
                bucket.pop(chunk_key, None)
                if not bucket:
                    del self._by_token_count[meta.token_count]
                    self._refresh_token_counts()
        logger.debug("[INDEX] chunk remove key=%s", chunk_key[:8])

    def _refresh_token_counts(self) -> None:
        """Refresh cached chunk token lengths in descending match order."""
        self._token_counts_desc = sorted(self._by_token_count, reverse=True)
