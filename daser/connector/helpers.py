# SPDX-License-Identifier: Apache-2.0
"""Dependency-light connector helpers used by scheduler-side tests."""

# Standard
import array
from dataclasses import dataclass, field

# Third Party
import xxhash

ROLLING_PREFIX_SEED = xxhash.xxh3_128(b"daser:rolling-prefix:v1").hexdigest()
TokenSequence = list[int] | bytes


def token_count(tokens: TokenSequence) -> int:
    """Return the number of native signed-int tokens in a list or wire buffer.

    Args:
        tokens: Token IDs as a Python list or packed native-int bytes.

    Returns:
        Number of token IDs.
    """
    if isinstance(tokens, bytes):
        if len(tokens) % 4:
            raise ValueError("packed token buffer is not int32 aligned")
        return len(tokens) // 4
    return len(tokens)


def token_window(tokens: TokenSequence, start: int, end: int) -> TokenSequence:
    """Return a token window while preserving packed bytes on the hot path.

    Args:
        tokens: Token IDs as a Python list or packed native-int bytes.
        start: Inclusive token index.
        end: Exclusive token index.

    Returns:
        The requested list slice or four-byte aligned byte slice.
    """
    if isinstance(tokens, bytes):
        return tokens[start * 4 : end * 4]
    return tokens[start:end]


def hash_tokens(tokens: TokenSequence) -> str:
    """Return hex xxh3_128 of token ID sequence.

    Args:
        tokens: list of integer token IDs.

    Returns:
        32-character hex string.

    Async/thread-safety:
        Pure CPU helper with no shared mutable state; safe to call from any
        thread or asyncio task.
    """
    # IPC lookup requests use packed bytes, avoiding a second msgpack list
    # encoding and server-side array conversion. Keep list support for public
    # callers and existing metadata/index tests.
    buf = tokens if isinstance(tokens, bytes) else bytes(array.array("i", tokens))
    return xxhash.xxh3_128(buf).hexdigest()


def rolling_prefix_key(prev_key: str, block_tokens: TokenSequence) -> str:
    """Return the next chained rolling-prefix key.

    Args:
        prev_key: previous rolling-prefix key as a 32-character hex string.
        block_tokens: token IDs for one vLLM block.

    Returns:
        32-character hex key for the prefix ending at this block.

    Async/thread-safety:
        Pure CPU helper with no shared mutable state; safe to call from any
        thread or asyncio task.
    """
    h = xxhash.xxh3_128()
    h.update(bytes.fromhex(prev_key))
    h.update(
        block_tokens
        if isinstance(block_tokens, bytes)
        else bytes(array.array("i", block_tokens))
    )
    return h.hexdigest()


def rolling_prefix_keys(
    tokens: TokenSequence,
    block_tokens: int,
    seed: str = ROLLING_PREFIX_SEED,
    start_slot: int = 0,
    initial_key: str | None = None,
) -> list[str]:
    """Return chained rolling-prefix keys for a token sequence.

    Args:
        tokens: full prompt token IDs.
        block_tokens: number of token IDs in one KV slot.
        seed: initial rolling-prefix seed.
        start_slot: slot index where key generation starts.
        initial_key: optional key immediately before ``start_slot``.

    Returns:
        Rolling keys for every full block from ``start_slot`` onward.

    Async/thread-safety:
        Pure CPU helper with no shared mutable state; safe to call from any
        thread or asyncio task.
    """
    if block_tokens <= 0:
        raise ValueError("block_tokens must be positive")
    aligned = (token_count(tokens) // block_tokens) * block_tokens
    start = start_slot * block_tokens
    if start < 0 or start > aligned:
        return []
    key_bytes = bytes.fromhex(initial_key or seed)
    # Pack the prompt once. Rebuilding an ``array.array`` for every block
    # dominated the control-plane profile on long prompts; byte slicing keeps
    # the exact signed-C-int wire representation while removing repeated
    # Python allocation and conversion work.
    packed_tokens = memoryview(
        tokens if isinstance(tokens, bytes) else array.array("i", tokens[:aligned])
    ).cast("B")
    token_stride = array.array("i").itemsize
    keys: list[str] = []
    for offset in range(start, aligned, block_tokens):
        h = xxhash.xxh3_128()
        h.update(key_bytes)
        begin = offset * token_stride
        end = (offset + block_tokens) * token_stride
        h.update(packed_tokens[begin:end])
        key_bytes = h.digest()
        keys.append(key_bytes.hex())
    return keys


def base_req_id(req_id: str) -> str:
    """Return the original request ID for synthetic scheduler sub-work IDs.

    Args:
        req_id: Request ID or scheduler-generated sub-work ID.

    Returns:
        Base vLLM request ID.

    Thread-safety:
        Pure string helper with no shared state.
    """
    return req_id.split(":store:", 1)[0].split(":load:", 1)[0]


@dataclass
class PendingStore:
    """Scheduler-side state for a prompt KV store that may span steps.

    Attributes:
        chunk_key: cache key for the final block-aligned store target.
        token_count: number of aligned prompt tokens that must be computed
            before the chunk can be published.
        block_ids: vLLM block IDs covering the prompt prefix seen so far.
        start_slot_index: first slot index that should be stored.
        rolling_key: rolling-prefix key immediately before
            rolling_slot_index. Empty for non-prefix strategies.
        rolling_slot_index: next slot index to process with rolling_key.
    """

    chunk_key: str
    token_count: int
    block_ids: list[int] = field(default_factory=list)
    start_slot_index: int = 0
    rolling_key: str = ""
    rolling_slot_index: int = 0
