# SPDX-License-Identifier: Apache-2.0
"""Warm one deterministic shared prefix before a timed vLLM benchmark."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import aiohttp
from vllm.benchmarks.datasets import RandomDataset
from vllm.benchmarks.lib.endpoint_request_func import (
    AIOHTTP_TIMEOUT,
    RequestFuncInput,
    async_request_openai_completions,
)
from vllm.tokenizers import get_tokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse deterministic prefix warm-up arguments.

    Args:
        argv: Optional arguments without the executable name.

    Returns:
        Parsed command-line namespace.

    Asyncio/thread-safety:
        Pure apart from argparse's standard error handling.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source-num-prompts", type=int, required=True)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--random-prefix-len", type=int, required=True)
    parser.add_argument("--random-range-ratio", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.source_num_prompts <= 0:
        parser.error("--source-num-prompts must be positive")
    return args


async def run_warmup(args: argparse.Namespace) -> dict[str, Any]:
    """Generate the timed workload's first prompt and send it once.

    RandomDataset advances its generator for all request lengths and offsets
    before it creates the shared prefix. Generating the full configured sample
    count here is therefore necessary to reproduce the exact prefix used by
    the later timed ``vllm bench serve`` process; only the first prompt is sent.

    Args:
        args: Parsed warm-up configuration.

    Returns:
        Compact evidence describing the completed warm-up request.

    Raises:
        RuntimeError: If the completion request fails.

    Asyncio/thread-safety:
        Performs one asynchronous HTTP request and has no shared mutable state.
    """
    tokenizer = get_tokenizer(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    samples = RandomDataset(random_seed=args.seed).sample(
        tokenizer=tokenizer,
        num_requests=args.source_num_prompts,
        prefix_len=args.random_prefix_len,
        range_ratio=args.random_range_ratio,
        input_len=args.input_len,
        output_len=args.output_len,
    )
    sample = samples[0]
    request = RequestFuncInput(
        prompt=sample.prompt,
        api_url=f"{args.base_url.rstrip('/')}/v1/completions",
        prompt_len=sample.prompt_len,
        output_len=sample.expected_output_len,
        model=args.model,
        ignore_eos=True,
        extra_body={"temperature": 0.0, "top_p": 1.0},
    )
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        result = await async_request_openai_completions(request, session)
    if not result.success:
        raise RuntimeError(f"prefix warm-up request failed: {result.error}")
    prompt_text = (
        sample.prompt if isinstance(sample.prompt, str) else repr(sample.prompt)
    )
    return {
        "seed": args.seed,
        "source_num_prompts": args.source_num_prompts,
        "sent_requests": 1,
        "prompt_tokens": sample.prompt_len,
        "prefix_tokens": args.random_prefix_len,
        "suffix_tokens": args.input_len,
        "output_tokens": result.output_tokens,
        "ttft_ms": result.ttft * 1000.0,
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
    }


def main(argv: list[str] | None = None) -> None:
    """Run one warm-up request and persist compact JSON evidence.

    Args:
        argv: Optional arguments without the executable name.

    Returns:
        None.

    Asyncio/thread-safety:
        Owns one event loop and should be invoked once per process.
    """
    args = parse_args(argv)
    result = asyncio.run(run_warmup(args))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
