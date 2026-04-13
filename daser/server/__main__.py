# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import os
import signal

# First Party
from daser.config import DaserConfig
from daser.logging import init_logger
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore

logger = init_logger(__name__)


async def run_server(cfg: DaserConfig) -> None:
    """Start the DaseR server and run until SIGTERM/SIGINT.

    Args:
        cfg: DaserConfig instance.
    """
    slot_size = cfg.resolved_slot_size()
    store = MetadataStore(total_slots=cfg.total_slots)
    cm = ChunkManager(total_slots=cfg.total_slots, metadata_store=store)

    if os.path.exists(cfg.index_path):
        try:
            cm.load(cfg.index_path)
            logger.info("[CHUNK] restored index from %s", cfg.index_path)
        except Exception as exc:
            logger.warning("[CHUNK] cold start — index load failed: %s", exc)

    ri = PrefixHashIndex(block_tokens=cfg.block_tokens)
    pe = FixedOffsetEncoder(fixed_offset=0)

    server = IPCServer(
        socket_path=cfg.ipc_socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=slot_size,
        block_tokens=cfg.block_tokens,
    )
    await server.start()

    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    logger.info("[SERVER] DaseR server ready")
    await stop_event.wait()

    logger.info("[SERVER] shutting down — saving index to %s", cfg.index_path)
    os.makedirs(os.path.dirname(cfg.index_path), exist_ok=True)
    cm.save(cfg.index_path)
    await server.stop()
    logger.info("[SERVER] shutdown complete")


def main() -> None:
    """Entry point: python -m daser.server"""
    cfg = DaserConfig()
    asyncio.run(run_server(cfg))


if __name__ == "__main__":
    main()
