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
from daser.position.base import PositionEncoder
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.base import RetrievalIndex
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.http import HTTPServerConfig, build_http_app
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

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
        description="DaseR server: HTTP API + IPC server",
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
        help="HuggingFace tokenizer name/path used by the HTTP server",
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
        help="Unix domain socket path for the IPC server",
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
        help="Blocks per document chunk for the HTTP server",
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
    parser.add_argument(
        "--cache-reuse-mode",
        choices=("prefix", "chunk"),
        default="prefix",
        help="Cache reuse strategy: prefix preserves current behavior; chunk "
        "enables block-aligned chunk reuse inside RAG prompts.",
    )
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
        cache_reuse_mode=args.cache_reuse_mode,
    )
    slot_size = cfg.resolved_slot_size()
    if args.store_size <= 0 or args.store_size % slot_size != 0:
        raise ValueError(
            f"--store-size ({args.store_size}) must be a positive multiple "
            f"of slot_size ({slot_size})"
        )
    cfg.total_slots = args.store_size // slot_size
    return cfg


def _build_http_config(args: argparse.Namespace) -> HTTPServerConfig:
    """Build HTTP server config from parsed arguments.

    Args:
        args: parsed argparse namespace.

    Returns:
        HTTP server config.
    """
    return HTTPServerConfig(
        vllm_base_url=args.vllm_base_url,
        model=args.model,
        tokenizer=args.tokenizer,
        block_tokens=args.block_tokens,
        chunk_blocks=args.chunk_blocks,
        align_document_chunks=args.cache_reuse_mode == "chunk",
    )


def _build_index_components(
    cache_reuse_mode: str, block_tokens: int
) -> tuple[RetrievalIndex, PositionEncoder]:
    """Build retrieval and position modules for a cache reuse mode.

    Args:
        cache_reuse_mode: either "prefix" or "chunk".
        block_tokens: vLLM block size in tokens.

    Returns:
        RetrievalIndex and PositionEncoder selected for the mode.

    Raises:
        ValueError: if cache_reuse_mode is unknown.
    """
    if cache_reuse_mode == "prefix":
        return PrefixHashIndex(block_tokens=block_tokens), FixedOffsetEncoder(
            fixed_offset=0
        )
    if cache_reuse_mode == "chunk":
        return ChunkReuseIndex(block_tokens=block_tokens), ChunkPositionEncoder(
            initial_offset=0
        )
    raise ValueError(f"unknown cache reuse mode: {cache_reuse_mode}")


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

    retrieval_index, position_encoder = _build_index_components(
        cfg.cache_reuse_mode, cfg.block_tokens
    )

    core = ServerCore(
        chunk_manager=cm,
        retrieval_index=retrieval_index,
        position_encoder=position_encoder,
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

    ipc_server = IPCServer(
        socket_path=cfg.ipc_socket_path,
        core=core,
    )
    await ipc_server.start()

    app = build_http_app(_build_http_config(args), core)
    uvicorn_config = uvicorn.Config(
        app=app,
        host=args.host,
        port=args.port,
        log_level="info",
        loop="none",
    )
    http_server = uvicorn.Server(uvicorn_config)
    http_task = asyncio.create_task(http_server.serve(), name="daser-http")

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop_event.set)
    loop.add_signal_handler(signal.SIGINT, stop_event.set)

    logger.info(
        "[SERVER] ready (HTTP=%s:%d, IPC=%s)",
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
        await ipc_server.stop()
        logger.info("[SERVER] shutdown complete")


def main() -> None:
    """CLI entry point for ``python -m daser.server``."""
    asyncio.run(run_server(_parse_args()))


if __name__ == "__main__":
    main()
