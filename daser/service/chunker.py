# SPDX-License-Identifier: Apache-2.0

# Standard
import array
from dataclasses import dataclass

# Third Party
import xxhash

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)


def hash_tokens(tokens: list[int]) -> str:
    """Return hex xxh3_128 of a token ID sequence.

    Mirrors daser.connector.daser_connector.hash_tokens and
    daser.retrieval.prefix._hash_tokens so that service-produced
    chunk_keys match keys the connector and retrieval layer compute.

    Args:
        tokens: list of integer token IDs.

    Returns:
        32-character hex string.
    """
    buf = bytes(array.array("i", tokens))
    return xxhash.xxh3_128(buf).hexdigest()


@dataclass
class TokenChunk:
    """A block-aligned token chunk ready to prefill.

    Attributes:
        tokens: token IDs in this chunk.
        chunk_key: xxh3_128 of tokens; doubles as the DaseR cache key.
    """

    tokens: list[int]
    chunk_key: str


class Chunker:
    """Split tokenized documents into block-aligned chunks.

    The initial implementation does the simplest thing that lines up
    with DaseR's on-disk layout: fixed-size chunks of
    ``chunk_blocks * block_tokens`` tokens each. Anything past the
    last aligned boundary is discarded.

    Args:
        block_tokens: vLLM block size (default 16). Must match the
            server configuration so chunk keys line up.
        chunk_blocks: how many blocks go into a single chunk. A larger
            value amortises per-chunk IPC overhead but reduces cache
            granularity.
    """

    def __init__(self, block_tokens: int = 16, chunk_blocks: int = 16) -> None:
        if block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        if chunk_blocks <= 0:
            raise ValueError("chunk_blocks must be positive")
        self._block_tokens = block_tokens
        self._chunk_blocks = chunk_blocks
        self._chunk_tokens = block_tokens * chunk_blocks

    @property
    def block_tokens(self) -> int:
        """vLLM block size in tokens."""
        return self._block_tokens

    @property
    def chunk_tokens(self) -> int:
        """Number of tokens per chunk (block_tokens * chunk_blocks)."""
        return self._chunk_tokens

    def chunk(self, tokens: list[int]) -> list[TokenChunk]:
        """Split ``tokens`` into fixed-size TokenChunks.

        Tokens past the last ``chunk_tokens`` boundary are dropped so
        every returned chunk has exactly ``chunk_tokens`` tokens.

        Args:
            tokens: tokenized document.

        Returns:
            List of TokenChunk; empty when the input is shorter than
            one chunk.
        """
        chunks: list[TokenChunk] = []
        n = len(tokens)
        aligned = (n // self._chunk_tokens) * self._chunk_tokens
        for start in range(0, aligned, self._chunk_tokens):
            slice_ = list(tokens[start : start + self._chunk_tokens])
            chunks.append(TokenChunk(tokens=slice_, chunk_key=hash_tokens(slice_)))
        if aligned < n:
            logger.debug(
                "[CHUNKER] dropped %d trailing tokens (not chunk-aligned)",
                n - aligned,
            )
        return chunks
