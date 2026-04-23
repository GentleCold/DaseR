# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import asyncio
import os
import signal

# First Party
from daser.config import DaserConfig
from daser.logging import init_logger
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.doc_registry import DocRegistry
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

logger = init_logger(__name__)

_DEFAULT_STORE_SIZE = 10 * 1024 * 1024 * 1024  # 10 GiB, matches docs/development.md
_DEFAULT_SLOT_SIZE = 2 * 1024 * 1024  # 2 MiB, matches docs/development.md


def _parse_args() -> argparse.Namespace:
    """Parse daser.server command-line arguments.

    Flags mirror docs/development.md. Model-param flags are only consulted
    when --slot-size is 0, in which case DaserConfig.resolved_slot_size()
    derives the slot size from (num_kv_heads, head_dim, num_layers,
    block_tokens, dtype_bytes).
    """
    parser = argparse.ArgumentParser(
        prog="daser.server",
        description="DaseR control-plane server (IPC only)",
    )
    parser.add_argument(
        "--store-path",
        required=True,
        help="Absolute path to the pre-allocated daser.store KV data file",
    )
    parser.add_argument(
        "--store-size",
        type=int,
        default=_DEFAULT_STORE_SIZE,
        help="Total store capacity in bytes; used to derive total_slots",
    )
    parser.add_argument(
        "--socket-path",
        default="/tmp/daser.sock",
        help="Unix domain socket path for connector <-> server IPC",
    )
    parser.add_argument(
        "--index-path",
        default="/tmp/daser.index",
        help="Path to the serialized metadata index",
    )
    parser.add_argument(
        "--slot-size",
        type=int,
        default=_DEFAULT_SLOT_SIZE,
        help="Bytes per KV slot; 0 means derive from model params",
    )
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=0,
        help="Only needed when --slot-size 0",
    )
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--model-id", default="default")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _build_daser_config(args: argparse.Namespace) -> DaserConfig:
    """Build a DaserConfig from parsed CLI arguments.

    total_slots is derived as store_size // resolved slot size so the ring
    buffer exactly covers the pre-allocated store file.

    Args:
        args: argparse namespace from _parse_args().

    Returns:
        Fully populated DaserConfig.

    Raises:
        ValueError: if store_size is not a positive multiple of the
                    resolved slot size.
    """
    cfg = DaserConfig(
        store_path=args.store_path,
        index_path=args.index_path,
        slot_size=args.slot_size,
        ipc_socket_path=args.socket_path,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        block_tokens=args.block_tokens,
        dtype_bytes=args.dtype_bytes,
        model_id=args.model_id,
        log_level=args.log_level,
    )
    slot_size = cfg.resolved_slot_size()
    if args.store_size <= 0 or args.store_size % slot_size != 0:
        raise ValueError(
            f"--store-size ({args.store_size}) must be a positive multiple "
            f"of slot_size ({slot_size})"
        )
    cfg.total_slots = args.store_size // slot_size
    return cfg


async def run_server(cfg: DaserConfig) -> None:
    """Start the DaseR server and run until SIGTERM/SIGINT.

    Args:
        cfg: DaserConfig instance.
    """
    slot_size = cfg.resolved_slot_size()
    store = MetadataStore(total_slots=cfg.total_slots)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=cfg.total_slots,
        metadata_store=store,
        doc_registry=doc_registry,
    )

    if os.path.exists(cfg.index_path):
        try:
            cm.load(cfg.index_path)
            logger.info("[CHUNK] restored index from %s", cfg.index_path)
        except Exception as exc:
            logger.warning("[CHUNK] cold start — index load failed: %s", exc)

    ri = PrefixHashIndex(block_tokens=cfg.block_tokens)
    pe = FixedOffsetEncoder(fixed_offset=0)

    # Repopulate retrieval index from recovered metadata.
    for meta in list(store.iter_chunks()):
        await ri.insert(meta)

    server = IPCServer(
        socket_path=cfg.ipc_socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=slot_size,
        block_tokens=cfg.block_tokens,
        doc_registry=doc_registry,
    )
    await server.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    logger.info("[SERVER] DaseR server ready")
    await stop_event.wait()

    logger.info("[SERVER] shutting down — saving index to %s", cfg.index_path)
    parent = os.path.dirname(cfg.index_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    cm.save(cfg.index_path)
    await server.stop()
    logger.info("[SERVER] shutdown complete")


def main() -> None:
    """Entry point: python -m daser.server"""
    cfg = _build_daser_config(_parse_args())
    asyncio.run(run_server(cfg))


if __name__ == "__main__":
    main()
