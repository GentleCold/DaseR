# SPDX-License-Identifier: Apache-2.0
"""Shared benchmark constants."""

from __future__ import annotations

from daser.config import model_geometry_from_path

BYTES_PER_GIB: int = 1024**3
BLOCK_TOKENS: int = 128

COMPARISON_IOURING_MEM = "iouring-mem-vs-lmcache-local-ssd-mem"

DEFAULT_SYSTEM_PROMPT: str = (
    "You are a helpful assistant answering questions using the following documents.\n\n"
)
DEFAULT_IMDB_QUESTION: str = "Summarize the sentiment of this review."


def slot_size_for_block_tokens(
    model_path: str,
    block_tokens: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Return bytes required for one model KV block across all TP ranks.

    Args:
        model_path: HuggingFace model directory containing ``config.json``.
        block_tokens: Number of tokens in a vLLM KV block.
        tensor_parallel_size: Number of vLLM tensor-parallel ranks.

    Returns:
        Aggregate slot size derived from the model geometry.

    Thread-safety:
        Reads the model config without mutating shared state.
    """
    return model_geometry_from_path(model_path).slot_size_for_block_tokens(
        block_tokens,
        tensor_parallel_size,
    )
