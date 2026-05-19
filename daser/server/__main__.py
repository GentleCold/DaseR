# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import asyncio
import os
import re
import signal

# Third Party
import uvicorn

# First Party
from daser.config import BLOCK_TOKENS, DaserConfig
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
from daser.server.http import HTTPServerConfig, VLLMClient, build_http_app
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

logger = init_logger(__name__)

_DEFAULT_STORE_SIZE = 10 * 1024 * 1024 * 1024

_SIZE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def _parse_size_bytes(value: str) -> int:
    """Parse a byte size with optional human-readable suffix.

    Args:
        value: integer byte count or value with kb/mb/gb/kib/mib/gib suffix.

    Returns:
        Size in bytes.

    Raises:
        argparse.ArgumentTypeError: if the value is invalid.
    """
    match = re.fullmatch(r"\s*(\d+)\s*([a-zA-Z]*)\s*", value)
    if match is None:
        raise argparse.ArgumentTypeError(f"invalid size: {value}")
    number = int(match.group(1))
    unit = match.group(2).lower()
    if unit not in _SIZE_UNITS:
        raise argparse.ArgumentTypeError(f"unsupported size unit: {unit}")
    return number * _SIZE_UNITS[unit]


def _ensure_store_file(cfg: DaserConfig) -> None:
    """Create the KV store file if it does not exist.

    Args:
        cfg: resolved DaseR server config.

    Raises:
        ValueError: if an existing store file has the wrong size.
    """
    os.makedirs(cfg.store_dir, exist_ok=True)
    if os.path.exists(cfg.store_path):
        existing = os.path.getsize(cfg.store_path)
        if existing != cfg.aligned_store_bytes:
            raise ValueError(
                f"existing store file {cfg.store_path} has size {existing}, "
                f"expected {cfg.aligned_store_bytes}"
            )
        return
    with open(cfg.store_path, "wb") as f:
        f.truncate(cfg.aligned_store_bytes)


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
        "--model-path",
        default=None,
        help="HuggingFace model path used by vLLM and tokenizer loading",
    )
    parser.add_argument(
        "--store-dir",
        required=True,
        help="Directory for daser.store and daser.index",
    )
    parser.add_argument(
        "--store-size",
        type=_parse_size_bytes,
        default=_DEFAULT_STORE_SIZE,
        help="Total store capacity, e.g. 10gb, 10gib, 512mb, or bytes",
    )
    parser.add_argument(
        "--socket-path",
        default="/tmp/daser.sock",
        help="Unix domain socket path for the IPC server",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--cache-reuse-mode",
        choices=("prefix", "chunk"),
        default="prefix",
        help="Cache reuse strategy: prefix preserves current behavior; chunk "
        "enables block-aligned chunk reuse inside RAG prompts.",
    )
    parser.add_argument(
        "--transfer-backend",
        choices=("gds", "iouring-mem"),
        default="gds",
        help="Worker transfer backend: gds uses kvikio/cuFile; iouring-mem "
        "uses pinned host L1 plus SSD IO.",
    )
    parser.add_argument(
        "--l1-cache-size",
        type=_parse_size_bytes,
        default=0,
        help="Pinned host L1 cache capacity for --transfer-backend iouring-mem.",
    )
    return parser.parse_args()


def _resolve_model_paths(
    args: argparse.Namespace,
    vllm_model_id: str,
) -> tuple[str, str]:
    """Resolve vLLM model ID and local model path for server startup.

    Args:
        args: parsed CLI arguments.
        vllm_model_id: first model ID returned by vLLM ``/v1/models``.

    Returns:
        ``(model_id, model_path)``.

    Raises:
        ValueError: if no usable local model path can be resolved.
    """
    model_path = args.model_path
    if model_path:
        return vllm_model_id, model_path
    config_path = os.path.join(vllm_model_id, "config.json")
    if os.path.exists(config_path):
        return vllm_model_id, vllm_model_id
    raise ValueError(
        "--model-path is required when vLLM /v1/models does not return "
        "a local model directory"
    )


async def _read_vllm_model_id(vllm_base_url: str) -> str:
    """Return the first model ID served by vLLM.

    Args:
        vllm_base_url: vLLM OpenAI-compatible base URL.

    Returns:
        First model ID from ``/v1/models``.

    Raises:
        RuntimeError: if vLLM reports no models.
    """
    client = VLLMClient(base_url=vllm_base_url, model="")
    try:
        models = await client.list_models()
    finally:
        await client.close()
    if not models:
        raise RuntimeError(f"vLLM at {vllm_base_url} returned no models")
    return models[0]


def _build_daser_config(args: argparse.Namespace) -> DaserConfig:
    """Build a DaserConfig from parsed CLI arguments.

    Args:
        args: parsed argparse namespace.

    Returns:
        Fully populated DaseR config.

    Raises:
        ValueError: if store size is not a positive slot multiple.
    """
    vllm_model_id = getattr(args, "vllm_model_id", None) or args.model_path
    if vllm_model_id is None:
        raise ValueError("vLLM model id has not been resolved")
    model_id, model_path = _resolve_model_paths(args, str(vllm_model_id))
    args.vllm_model_id = model_id
    args.model_path = model_path
    cfg = DaserConfig(
        model_path=model_path,
        vllm_model_id=model_id,
        store_dir=args.store_dir,
        total_store_bytes=args.store_size,
        ipc_socket_path=args.socket_path,
        log_level=args.log_level,
        cache_reuse_mode=args.cache_reuse_mode,
        transfer_backend=args.transfer_backend,
        l1_cache_size=args.l1_cache_size,
    )
    if cfg.transfer_backend == "iouring-mem" and cfg.l1_cache_size <= 0:
        raise ValueError("--l1-cache-size must be positive for iouring-mem")
    slot_size = cfg.resolved_slot_size()
    if cfg.total_store_bytes <= 0 or cfg.total_slots <= 0:
        raise ValueError(
            f"--store-size ({cfg.total_store_bytes}) must be at least one "
            f"slot ({slot_size} bytes)"
        )
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
        model=getattr(args, "vllm_model_id", None) or args.model_path,
        tokenizer=args.model_path,
        block_tokens=BLOCK_TOKENS,
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
    args.vllm_model_id = await _read_vllm_model_id(args.vllm_base_url)
    cfg = _build_daser_config(args)
    _ensure_store_file(cfg)
    core = await _build_core(cfg)

    ipc_server = IPCServer(
        socket_path=cfg.ipc_socket_path,
        core=core,
        runtime_config=cfg.runtime_config(),
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
    logger.info(
        "[SERVER] model_id=%s model_path=%s store=%s slots=%d",
        cfg.model_id,
        cfg.model_path,
        cfg.store_path,
        cfg.total_slots,
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
