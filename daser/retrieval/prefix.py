# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.connector.helpers import rolling_prefix_keys
from daser.logging import init_logger
from daser.retrieval.base import RetrievalIndex, RetrievalMatch
from daser.server.metadata_store import ChunkMeta

logger = init_logger(__name__)

__all__ = ["PrefixHashIndex"]


class PrefixHashIndex(RetrievalIndex):
    """Rolling token-prefix hash retrieval index.

    Stores one KV slot per committed block. Slot ``i`` is indexed by a
    chained prefix key ``H(prev_key, block_tokens_i)`` so the key commits to
    the whole prompt prefix ending at that slot while the stored payload stays
    one block wide.

    Args:
        block_tokens: vLLM block size in tokens (default 16). Prefix
                      lengths are quantised to multiples of this value.
    """

    def __init__(self, block_tokens: int = 16) -> None:
        self._block_tokens = block_tokens
        self._index: dict[str, ChunkMeta] = {}

    async def lookup(self, tokens: list[int], model_id: str) -> list[RetrievalMatch]:
        """Return contiguous cached rolling-prefix slots for tokens.

        Computes chained prefix keys for each full block and returns committed
        slot hits from the start of the prompt until the first missing slot.

        Args:
            tokens: full token sequence to match against.
            model_id: only chunks with this model_id are returned.

        Returns:
            Retrieval matches ordered by target token start.
        """
        matches: list[RetrievalMatch] = []
        run_meta: ChunkMeta | None = None
        run_target_start = 0
        aligned = (len(tokens) // self._block_tokens) * self._block_tokens

        def flush_run() -> None:
            nonlocal run_meta, run_target_start
            if run_meta is not None:
                matches.append(
                    RetrievalMatch(
                        meta=run_meta,
                        target_token_start=run_target_start,
                    )
                )
            run_meta = None
            run_target_start = 0

        for slot_i, key in enumerate(
            rolling_prefix_keys(tokens[:aligned], self._block_tokens)
        ):
            meta = self._index.get(key)
            if meta is None or meta.model_id != model_id:
                break
            target_token_start = slot_i * self._block_tokens
            logger.debug(
                "[INDEX] rolling prefix hit key=%s target=%d",
                key[:8],
                target_token_start,
            )
            can_extend = (
                run_meta is not None
                and meta.start_slot == run_meta.start_slot + run_meta.num_slots
                and meta.pos_offset == run_meta.pos_offset
                and meta.model_id == run_meta.model_id
                and target_token_start == run_target_start + run_meta.token_count
            )
            if can_extend:
                run_meta = ChunkMeta(
                    chunk_key=meta.chunk_key,
                    start_slot=run_meta.start_slot,
                    num_slots=run_meta.num_slots + meta.num_slots,
                    token_count=run_meta.token_count + meta.token_count,
                    pos_offset=run_meta.pos_offset,
                    model_id=run_meta.model_id,
                    created_at=run_meta.created_at,
                    doc_ids=list(run_meta.doc_ids),
                )
                continue
            flush_run()
            run_meta = meta
            run_target_start = target_token_start
        flush_run()
        return matches

    async def insert(self, meta: ChunkMeta) -> None:
        """Insert a chunk into the prefix index.

        Args:
            meta: ChunkMeta keyed by a rolling-prefix slot key.
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
