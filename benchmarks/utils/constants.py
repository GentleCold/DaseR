# SPDX-License-Identifier: Apache-2.0
"""Shared benchmark constants."""

from __future__ import annotations

BYTES_PER_GIB: int = 1024**3
BLOCK_TOKENS: int = 16
NUM_KV_HEADS: int = 8
HEAD_DIM: int = 128
NUM_LAYERS: int = 36
DTYPE_BYTES: int = 2
BENCHMARK_SEED: int = 42

COMPARISON_GDS = "gds-vs-lmcache-local-ssd"
COMPARISON_IOURING_MEM = "iouring-mem-vs-lmcache-local-ssd-mem"

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful assistant answering questions using the following documents.\n\n"
)
DEFAULT_IMDB_QUESTION: str = "Summarize the sentiment of this review."


def slot_size_for_block_tokens(block_tokens: int) -> int:
    """Return bytes required for one model KV block.

    Args:
        block_tokens: Number of tokens in a vLLM KV block.

    Returns:
        Slot size in bytes for the benchmark model geometry.

    Thread-safety:
        Pure calculation over constants.
    """
    return NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS * block_tokens * DTYPE_BYTES


SLOT_SIZE: int = slot_size_for_block_tokens(BLOCK_TOKENS)
