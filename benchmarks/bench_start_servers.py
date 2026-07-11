# SPDX-License-Identifier: Apache-2.0
"""Start benchmark service processes and write a run manifest."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from benchmarks.utils.servers import ServerManager
from benchmarks.utils.sizing import parse_size_bytes
from benchmarks.utils.system import apply_gpu_selection


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("vllm", "lmcache", "daser"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--store-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--gpu-id", default="auto")
    parser.add_argument("--gpu-util", type=float, default=0.85)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--l1-size", type=parse_size_bytes, default="256gib")
    parser.add_argument("--l2-size", type=parse_size_bytes, default="300gib")
    parser.add_argument(
        "--cache-reuse-mode", choices=("chunk", "prefix"), default="chunk"
    )
    parser.add_argument(
        "--transfer-mode", choices=("iouring", "gds"), default="iouring"
    )
    parser.add_argument(
        "--skip-l2",
        action="store_true",
        help="Disable backend L2 persistence/adapters for no-evict L1-only runs.",
    )
    parser.add_argument("--vllm-port", type=int, default=8001)
    parser.add_argument("--daser-port", type=int, default=2026)
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    return parser.parse_args(argv)


async def main_async(args: argparse.Namespace) -> None:
    """Start services and write manifest."""
    selected_gpu = apply_gpu_selection(args.gpu_id) or args.gpu_id
    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    manager = ServerManager(
        run_id=run_id,
        backend=args.backend,
        model=args.model,
        store_dir=args.store_dir,
        gpu_id=str(selected_gpu),
        gpu_util=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        l1_size_bytes=args.l1_size,
        l2_size_bytes=args.l2_size,
        max_num_batched_tokens=(
            args.max_num_batched_tokens if args.max_num_batched_tokens > 0 else None
        ),
        block_size=args.block_size,
        reuse_mode=args.cache_reuse_mode,
        transfer_mode=args.transfer_mode,
        vllm_port=args.vllm_port,
        daser_port=args.daser_port,
        startup_timeout=args.startup_timeout,
        max_model_len=args.max_model_len if args.max_model_len > 0 else None,
        skip_l2=args.skip_l2,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=args.trust_remote_code,
    )
    manifest = await manager.start()
    print(f"manifest={args.store_dir}/manifest.json")
    print(f"backend={manifest.backend} gpu={selected_gpu}")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    main()
