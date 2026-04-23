# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import asyncio
import os
import signal

# Third Party
import uvicorn

# First Party
from daser.config import DaserConfig
from daser.logging import init_logger
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.doc_registry import DocRegistry
from daser.server.ipc_server import IPCServer
from daser.server.metadata_store import MetadataStore
from daser.service.http_api import ServiceConfig, build_service_app

logger = init_logger(__name__)


def _parse_args() -> argparse.Namespace:
    """Parse daser.service command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="daser.service",
        description="DaseR RAG service (HTTP API + embedded DaseR control plane)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--vllm-base-url",
        required=True,
        help="Base URL of the vllm serve instance (e.g. http://127.0.0.1:8001)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model identifier to pass to vLLM's OpenAI API",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="HuggingFace tokenizer name/path used for document tokenization",
    )
    parser.add_argument(
        "--socket-path", default="/tmp/daser.sock", help="DaseR IPC socket path"
    )
    parser.add_argument(
        "--store-path",
        default="/mnt/xfs/daser.store",
        help="Path to daser.store (KV data file, created on first launch)",
    )
    parser.add_argument(
        "--index-path",
        default="/mnt/xfs/daser.index",
        help="Path to daser.index (metadata snapshot)",
    )
    parser.add_argument("--total-slots", type=int, default=1024)
    parser.add_argument("--slot-size", type=int, default=0)
    parser.add_argument("--block-tokens", type=int, default=16)
    parser.add_argument(
        "--chunk-blocks",
        type=int,
        default=16,
        help="Blocks per chunk used by the service Chunker",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=0,
        help="Only needed when --slot-size 0; see DaserConfig",
    )
    parser.add_argument("--head-dim", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--model-id", default="default")
    return parser.parse_args()


def _build_daser_config(args: argparse.Namespace) -> DaserConfig:
    """Build a DaserConfig from parsed CLI arguments."""
    return DaserConfig(
        store_path=args.store_path,
        index_path=args.index_path,
        total_slots=args.total_slots,
        slot_size=args.slot_size,
        ipc_socket_path=args.socket_path,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        num_layers=args.num_layers,
        block_tokens=args.block_tokens,
        dtype_bytes=args.dtype_bytes,
        model_id=args.model_id,
    )


async def _run(args: argparse.Namespace) -> None:
    """Start the IPC server + HTTP API and run until a shutdown signal.

    The HTTP server is started in-process via uvicorn.Server so we can
    run it alongside the IPC server on the same asyncio event loop.

    Args:
        args: parsed CLI namespace.
    """
    cfg = _build_daser_config(args)
    slot_size = cfg.resolved_slot_size() if cfg.slot_size else 0

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
            logger.info("[SERVICE] restored DaseR index from %s", cfg.index_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SERVICE] cold start — index load failed: %s", exc)

    ri = PrefixHashIndex(block_tokens=cfg.block_tokens)
    pe = FixedOffsetEncoder(fixed_offset=0)

    # Repopulate retrieval index from any recovered chunks so lookups
    # hit right after a restart.
    for meta in list(store.iter_chunks()):
        await ri.insert(meta)

    # slot_size may be 0 when model params are not supplied; in that
    # case the connector computes it on first register_kv_caches. The
    # server only needs it to turn start_slot into a byte offset, which
    # the service layer never consumes directly.
    ipc_server = IPCServer(
        socket_path=cfg.ipc_socket_path,
        chunk_manager=cm,
        retrieval_index=ri,
        position_encoder=pe,
        slot_size=slot_size,
        block_tokens=cfg.block_tokens,
        doc_registry=doc_registry,
    )
    await ipc_server.start()

    service_cfg = ServiceConfig(
        vllm_base_url=args.vllm_base_url,
        model=args.model,
        tokenizer=args.tokenizer,
        socket_path=cfg.ipc_socket_path,
        block_tokens=cfg.block_tokens,
        chunk_blocks=args.chunk_blocks,
    )
    app = build_service_app(service_cfg)

    uvicorn_config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="none",  # reuse the current asyncio loop
    )
    http_server = uvicorn.Server(uvicorn_config)
    http_task = asyncio.create_task(http_server.serve(), name="daser-service-http")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    logger.info(
        "[SERVICE] DaseR service ready (HTTP=%s:%d, IPC=%s)",
        args.host,
        args.port,
        cfg.ipc_socket_path,
    )

    try:
        done, pending = await asyncio.wait(
            [http_task, asyncio.create_task(stop_event.wait())],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
    finally:
        http_server.should_exit = True
        try:
            await asyncio.wait_for(http_task, timeout=5)
        except Exception:  # noqa: BLE001
            pass

        logger.info("[SERVICE] shutting down — saving DaseR state")
        parent = os.path.dirname(cfg.index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        cm.save(cfg.index_path)
        await ipc_server.stop()
        logger.info("[SERVICE] shutdown complete")


def main() -> None:
    """CLI entry point: ``python -m daser.service``."""
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
