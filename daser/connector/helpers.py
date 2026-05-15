# SPDX-License-Identifier: Apache-2.0
"""Dependency-light connector helpers used by scheduler-side tests."""

# Standard
import array
from dataclasses import dataclass, field

# Third Party
import xxhash


def hash_tokens(tokens: list[int]) -> str:
    """Return hex xxh3_128 of token ID sequence.

    Args:
        tokens: list of integer token IDs.

    Returns:
        32-character hex string.

    Async/thread-safety:
        Pure CPU helper with no shared mutable state; safe to call from any
        thread or asyncio task.
    """
    # Pack as a contiguous C-int array for a single hash pass; avoids
    # the per-token Python-loop overhead of repeated h.update() calls.
    buf = bytes(array.array("i", tokens))
    return xxhash.xxh3_128(buf).hexdigest()


@dataclass
class PendingStore:
    """Scheduler-side state for a prompt KV store that may span steps.

    Attributes:
        chunk_key: xxh3_128 hex of the full block-aligned prompt prefix.
        token_count: number of aligned prompt tokens that must be computed
            before the chunk can be published.
        block_ids: vLLM block IDs covering the prompt prefix seen so far.
    """

    chunk_key: str
    token_count: int
    block_ids: list[int] = field(default_factory=list)
