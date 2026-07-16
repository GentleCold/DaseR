# SPDX-License-Identifier: Apache-2.0
"""Build a GLM-native binary token pool for the TraceLab replay runner."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
import sys
from typing import Any


def build_token_pool(corpus: Path, tokenizer: Any, limit: int) -> list[int]:
    """Tokenize a text corpus into ordinary model-native token IDs.

    Args:
        corpus: UTF-8 text corpus with one sample per line.
        tokenizer: Hugging Face-compatible tokenizer with ``encode`` and
            ``vocab_size`` attributes.
        limit: Maximum number of token IDs to return.

    Returns:
        Up to ``limit`` token IDs, excluding tokenizer-added special tokens.

    Thread-safety:
        Performs synchronous file IO and tokenizer calls. Do not call it from
        an asyncio hot path or share a mutable tokenizer concurrently.
    """

    if limit <= 0:
        raise ValueError("token limit must be positive")
    vocab_size = int(tokenizer.vocab_size)
    if vocab_size <= 0:
        raise ValueError("tokenizer vocab_size must be positive")
    special_token_ids = {
        int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())
    }

    pool: list[int] = []
    with corpus.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            token_ids = tokenizer.encode(line, add_special_tokens=False)
            for token_id in token_ids:
                value = int(token_id)
                if value in special_token_ids:
                    continue
                if value < 0 or value >= vocab_size:
                    raise ValueError(
                        f"ordinary token ID {value} is outside [0, {vocab_size})"
                    )
                pool.append(value)
                if len(pool) == limit:
                    return pool

    if not pool:
        raise ValueError("text corpus produced an empty token pool")
    return pool


def write_token_pool(token_ids: list[int], output: Path) -> None:
    """Write token IDs as headerless little-endian unsigned 32-bit integers.

    Args:
        token_ids: Non-negative IDs that fit in ``u32``.
        output: Destination consumed by TraceLab ``--token-pool-file``.

    Returns:
        ``None``.

    Thread-safety:
        Performs synchronous file IO and replaces ``output``. The destination
        must not be read or written concurrently.
    """

    if not token_ids:
        raise ValueError("token pool must not be empty")
    values = array("I", token_ids)
    if values.itemsize != 4:
        raise RuntimeError("this platform does not provide 32-bit unsigned arrays")
    if sys.byteorder != "little":
        values.byteswap()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as file:
        values.tofile(file)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a model-native u32 token pool for TraceLab"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1_000_000)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Generate the token pool described by command-line arguments.

    Args:
        None.

    Returns:
        ``None``.

    Thread-safety:
        Runs synchronously as a standalone preparation command.
    """

    from transformers import AutoTokenizer

    args = _parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    token_ids = build_token_pool(args.text_file, tokenizer, args.limit)
    write_token_pool(token_ids, args.output)
    print(
        f"wrote {len(token_ids)} model-native tokens to {args.output} "
        f"(min={min(token_ids)}, max={max(token_ids)}, "
        f"vocab_size={tokenizer.vocab_size})"
    )


if __name__ == "__main__":
    main()
