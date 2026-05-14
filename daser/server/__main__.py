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
from daser.server.connector_api import ConnectorAPIServer
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.metadata_store import MetadataStore
from daser.server.rag_api import RAGAPIConfig, build_rag_api

logger = init_logger(__name__)

_DEFAULT_STORE_SIZE = 10 * 1024 * 1024 * 1024
_DEFAULT_SLOT_SIZE = 2 * 1024 * 1024


def _parse_args() -> argparse.Namespace:
    """Parse ``python -m daser.server`` command-line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(
        prog="daser.server",
        description=(
            "DaseR server: North Bound RAG HTTP API + South Bound Connector IPC API"
        ),
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
        help="HuggingFace tokenizer name/path used by the North Bound RAG API",
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
        help="Unix domain socket path for the South Bound Connector API",
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
        "--chunk-blocks",
        type=int,
        default=16,
        help="Blocks per document chunk for the North Bound RAG API",
    )
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

    Args:
        args: parsed argparse namespace.

    Returns:
        Fully populated DaseR config.

    Raises:
        ValueError: if store size is not a positive slot multiple.
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


def _build_rag_config(args: argparse.Namespace) -> RAGAPIConfig:
    """Build North Bound RAG API config from parsed arguments.

    Args:
        args: parsed argparse namespace.

    Returns:
        RAG API config.
    """
    return RAGAPIConfig(
        vllm_base_url=args.vllm_base_url,
        model=args.model,
        tokenizer=args.tokenizer,
        block_tokens=args.block_tokens,
        chunk_blocks=args.chunk_blocks,
    )


async def _build_core(cfg: DaserConfig) -> ServerCore:
    """Construct and restore the shared server core.

    Args:
        cfg: DaseR runtime config.

    Returns:
        Restored ServerCore.
    """
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
            logger.info("[SERVER] restored index from %s", cfg.index_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SERVER] cold start; index load failed: %s", exc)

    core = ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=cfg.block_tokens),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=cfg.resolved_slot_size(),
        block_tokens=cfg.block_tokens,
    )
    await core.rebuild_retrieval_index()
    return core


async def run_server(args: argparse.Namespace) -> None:
    """Run the unified DaseR server until SIGTERM/SIGINT.

    Args:
        args: parsed CLI arguments.
    """
    cfg = _build_daser_config(args)
    core = await _build_core(cfg)

    connector_api = ConnectorAPIServer(
        socket_path=cfg.ipc_socket_path,
        core=core,
    )
    await connector_api.start()

    app = build_rag_api(_build_rag_config(args), core)
    uvicorn_config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="none",
    )
    http_server = uvicorn.Server(uvicorn_config)
    http_task = asyncio.create_task(http_server.serve(), name="daser-nb-http")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    logger.info(
        "[SERVER] ready (NB HTTP=%s:%d, SB IPC=%s)",
        args.host,
        args.port,
        cfg.ipc_socket_path,
    )

    stop_task = asyncio.create_task(stop_event.wait(), name="daser-stop")
    try:
        done, pending = await asyncio.wait(
            [http_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            task.result()
    finally:
        http_server.should_exit = True
        try:
            await asyncio.wait_for(http_task, timeout=5)
        except Exception:  # noqa: BLE001
            pass

        logger.info("[SERVER] shutting down; saving index to %s", cfg.index_path)
        parent = os.path.dirname(cfg.index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            core.chunk_manager.save(cfg.index_path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SERVER] failed to save index: %s", exc)
        await connector_api.stop()
        logger.info("[SERVER] shutdown complete")


def main() -> None:
    """CLI entry point for ``python -m daser.server``."""
    asyncio.run(run_server(_parse_args()))


if __name__ == "__main__":
    main()
