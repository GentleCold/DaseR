# SPDX-License-Identifier: Apache-2.0
"""Launch a DaseR server tuned for an example run.

This is a thin wrapper around ``daser.server.run_server`` that accepts CLI flags
and pre-allocates the NVMe store file if missing. Run in a dedicated terminal
before launching ``vllm_cold_warm.py``.

Usage:
    python examples/run_daser_server.py \\
        --store-path /tmp/daser_example/daser.store \\
        --socket-path /tmp/daser_example/daser.sock

Default model geometry targets Qwen3-8B (36 layers, 8 KV heads, bf16).
"""

# Future
from __future__ import annotations

# Standard
import argparse
import asyncio
import os

# First Party
from daser.config import DaserConfig
from daser.logging import init_logger
from daser.server.__main__ import run_server

logger = init_logger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse CLI flags for the example server launcher.

    Returns:
        argparse.Namespace with the server configuration fields.
    """
    parser = argparse.ArgumentParser(
        description="Launch a DaseR server configured for Qwen3-8B by default.",
    )
    parser.add_argument("--store-path", required=True, help="NVMe store file path")
    parser.add_argument(
        "--socket-path",
        default="/tmp/daser.sock",
        help="Unix socket path (default: /tmp/daser.sock)",
    )
    parser.add_argument(
        "--index-path",
        default="/tmp/daser.index",
        help="Metadata index snapshot path (default: /tmp/daser.index)",
    )
    parser.add_argument("--total-slots", type=int, default=128)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--model-id", default="qwen3-8b")
    return parser.parse_args()


def _preallocate_store(path: str, size_bytes: int) -> None:
    """Ensure ``path`` exists and is at least ``size_bytes`` long.

    Args:
        path: absolute path to the store file.
        size_bytes: total ring-buffer capacity in bytes.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) >= size_bytes:
        return
    logger.info("[EXAMPLE] pre-allocating store %s (%d bytes)", path, size_bytes)
    with open(path, "wb") as fh:
        fh.truncate(size_bytes)


def main() -> None:
    """Build a ``DaserConfig`` from CLI args and run the server until signal."""
    args = _parse_args()
    cfg = DaserConfig(
        store_path=args.store_path,
        ipc_socket_path=args.socket_path,
        index_path=args.index_path,
        total_slots=args.total_slots,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        block_tokens=args.block_tokens,
        dtype_bytes=args.dtype_bytes,
        model_id=args.model_id,
    )
    _preallocate_store(cfg.store_path, cfg.total_slots * cfg.resolved_slot_size())
    logger.info(
        "[EXAMPLE] starting server: socket=%s slot_size=%d total_slots=%d",
        cfg.ipc_socket_path,
        cfg.resolved_slot_size(),
        cfg.total_slots,
    )
    asyncio.run(run_server(cfg))


if __name__ == "__main__":
    main()
