# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.connector.helpers import TokenSequence, rolling_prefix_keys
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

    async def lookup(
        self, tokens: TokenSequence, model_id: str
    ) -> list[RetrievalMatch]:
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
        run_first_meta: ChunkMeta | None = None
        run_last_key = ""
        run_num_slots = 0
        run_token_count = 0
        run_target_start = 0
        hit_count = 0

        def flush_run() -> None:
            nonlocal run_first_meta, run_last_key, run_num_slots
            nonlocal run_token_count, run_target_start
            if run_first_meta is not None:
                matches.append(
                    RetrievalMatch(
                        # Keep the hot loop allocation-free for contiguous
                        # hits.  A long prompt can contain thousands of
                        # indexed blocks; constructing one replacement
                        # ChunkMeta per block only to update aggregate counts
                        # needlessly copies doc_ids and creates garbage on the
                        # server event loop.  Build the public aggregate once
                        # when the run ends.
                        meta=ChunkMeta(
                            chunk_key=run_last_key,
                            start_slot=run_first_meta.start_slot,
                            num_slots=run_num_slots,
                            token_count=run_token_count,
                            pos_offset=run_first_meta.pos_offset,
                            model_id=run_first_meta.model_id,
                            created_at=run_first_meta.created_at,
                            doc_ids=list(run_first_meta.doc_ids),
                        ),
                        target_token_start=run_target_start,
                    )
                )
            run_first_meta = None
            run_last_key = ""
            run_num_slots = 0
            run_token_count = 0
            run_target_start = 0

        for slot_i, key in enumerate(rolling_prefix_keys(tokens, self._block_tokens)):
            meta_entry = self._index.get(key)
            if meta_entry is None or meta_entry.model_id != model_id:
                break
            hit_count += 1
            target_token_start = slot_i * self._block_tokens
            can_extend = run_first_meta is not None and (
                meta_entry.start_slot == run_first_meta.start_slot + run_num_slots
                and meta_entry.pos_offset == run_first_meta.pos_offset
                and meta_entry.model_id == run_first_meta.model_id
                and target_token_start == run_target_start + run_token_count
            )
            if can_extend:
                run_last_key = meta_entry.chunk_key
                run_num_slots += meta_entry.num_slots
                run_token_count += meta_entry.token_count
                continue
            flush_run()
            run_first_meta = meta_entry
            run_last_key = meta_entry.chunk_key
            run_num_slots = meta_entry.num_slots
            run_token_count = meta_entry.token_count
            run_target_start = target_token_start
        flush_run()
        if hit_count:
            logger.debug(
                "[INDEX] rolling prefix lookup hits=%d tokens=%d matches=%d",
                hit_count,
                hit_count * self._block_tokens,
                len(matches),
            )
        return matches

    def candidate_keys(self, tokens: TokenSequence, model_id: str) -> set[str]:
        """Return rolling-prefix keys potentially matching this prompt.

        Args:
            tokens: full prompt token IDs.
            model_id: model identifier, accepted for the common retrieval API.

        Returns:
            Rolling keys for every full block in ``tokens``.  Prefix keys are
            model-independent; ``model_id`` is enforced when lookup resolves
            a committed metadata entry.

        Async/thread-safety:
            Pure CPU hashing with no index mutation or blocking I/O.
        """
        del model_id
        return set(rolling_prefix_keys(tokens, self._block_tokens))
