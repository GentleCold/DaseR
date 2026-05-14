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
    ``chunk_blocks * block_tokens`` tokens each. Callers can either drop
    the final partial chunk or pad it to the next chunk boundary.

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

    def pad_to_chunk_boundary(self, tokens: list[int], pad_token: int) -> list[int]:
        """Return a copy of ``tokens`` padded to a chunk boundary.

        Args:
            tokens: tokenized input.
            pad_token: token ID used to fill the final partial chunk.

        Returns:
            Padded token list. Empty input remains empty.
        """
        padded = list(tokens)
        if not padded:
            return padded
        remainder = len(padded) % self._chunk_tokens
        if remainder:
            padded.extend([pad_token] * (self._chunk_tokens - remainder))
        return padded

    def chunk(
        self, tokens: list[int], pad_token: int | None = None
    ) -> list[TokenChunk]:
        """Split ``tokens`` into fixed-size TokenChunks.

        When ``pad_token`` is not provided, tokens past the last
        ``chunk_tokens`` boundary are dropped. When provided, the final
        partial chunk is padded so every input token is represented in a
        full-size chunk.

        Args:
            tokens: tokenized document.
            pad_token: optional token ID used to pad the final partial
                chunk.

        Returns:
            List of TokenChunk; empty when the input is shorter than
            one chunk and ``pad_token`` is not provided.
        """
        chunks: list[TokenChunk] = []
        chunked_tokens = (
            self.pad_to_chunk_boundary(tokens, pad_token)
            if pad_token is not None
            else list(tokens)
        )
        n = len(chunked_tokens)
        aligned = (n // self._chunk_tokens) * self._chunk_tokens
        for start in range(0, aligned, self._chunk_tokens):
            slice_ = list(chunked_tokens[start : start + self._chunk_tokens])
            chunks.append(TokenChunk(tokens=slice_, chunk_key=hash_tokens(slice_)))
        if pad_token is None and aligned < n:
            logger.debug(
                "[CHUNKER] dropped %d trailing tokens (not chunk-aligned)",
                n - aligned,
            )
        return chunks
