# SPDX-License-Identifier: Apache-2.0

# Standard
import argparse
import asyncio
import os
import re
import signal
from typing import Any, Awaitable, Callable

# Third Party
import uvicorn

# First Party
from daser.config import (
    BLOCK_TOKENS,
    CACHE_REUSE_CHUNK,
    CACHE_REUSE_MODES,
    CACHE_REUSE_PREFIX,
    DEFAULT_CACHE_REUSE_MODE,
    DEFAULT_IOURING_L1_BYTES,
    DaserConfig,
)
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
from daser.version import __version__, startup_version_message

logger = init_logger(__name__)


class VLLMStartupError(RuntimeError):
    """Raised when DaseR cannot contact vLLM during startup.

    Async/thread-safety:
        Exception type only; safe to raise from sync or async startup paths.
    """


_DEFAULT_L2_SIZE = 10 * 1024 * 1024 * 1024

DASER_ASCII_BANNER = r"""

████▄   ▄▄▄   ▄▄▄▄ ▄▄▄▄▄ █████▄
██  ██ ██▀██ ███▄▄ ██▄▄  ██▄▄██▄
████▀  ██▀██ ▄▄██▀ ██▄▄▄ ██   ██
"""
DASER_BANNER_COLOR = "\033[38;2;102;178;255m"
DASER_BANNER_RESET = "\033[0m"

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


def _log_startup_banner() -> None:
    """Log the DaseR ASCII startup banner.

    Async/thread-safety:
        Synchronous logging helper called once during CLI startup before
        server tasks are created.
    """
    colored_banner = (
        "\n"
        "\n████▄   ▄▄▄   ▄▄▄▄ ▄▄▄▄▄ "
        f"{DASER_BANNER_COLOR}█████▄{DASER_BANNER_RESET}\n"
        "██  ██ ██▀██ ███▄▄ ██▄▄  "
        f"{DASER_BANNER_COLOR}██▄▄██▄{DASER_BANNER_RESET}\n"
        "████▀  ██▀██ ▄▄██▀ ██▄▄▄ "
        f"{DASER_BANNER_COLOR}██   ██{DASER_BANNER_RESET}\n"
    )
    logger.info("%s", colored_banner)


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
    if cfg.skip_l2:
        logger.info("[SERVER] skip_l2 enabled; not allocating L2 store file")
        return
    if os.path.exists(cfg.store_path):
        existing = os.path.getsize(cfg.store_path)
        if existing > cfg.aligned_store_bytes:
            logger.warning(
                "[SERVER] truncating store file %s from %d to aligned size %d",
                cfg.store_path,
                existing,
                cfg.aligned_store_bytes,
            )
            with open(cfg.store_path, "r+b") as f:
                f.truncate(cfg.aligned_store_bytes)
            return
        if existing < cfg.aligned_store_bytes:
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
    parser.add_argument("--port", type=int, default=2026)
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
        "--l2-size",
        type=_parse_size_bytes,
        default=_DEFAULT_L2_SIZE,
        help="L2 SSD-tier capacity, e.g. 10gb, 10gib, 512mb, or bytes",
    )
    parser.add_argument(
        "--socket-path",
        default="/tmp/daser.sock",
        help="Unix domain socket path for the IPC server",
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--cache-reuse-mode",
        choices=CACHE_REUSE_MODES,
        default=DEFAULT_CACHE_REUSE_MODE,
        help="Cache reuse strategy: chunk enables block-aligned chunk reuse "
        "inside RAG prompts; prefix enables rolling-prefix slot reuse.",
    )
    parser.add_argument(
        "--transfer-mode",
        choices=("gds", "iouring"),
        default="iouring",
        help="Server-owned transfer layer: gds uses kvikio direct GPU/SSD IO; "
        "iouring uses an L1 pinned-memory tier above an L2 SSD file.",
    )
    parser.add_argument(
        "--l1-size",
        type=_parse_size_bytes,
        default=None,
        help="L1 memory-tier capacity for --transfer-mode=iouring. Defaults "
        "to min(1GiB, --l2-size).",
    )
    parser.add_argument(
        "--skip-l2",
        action="store_true",
        help="Use volatile L1 memory only: do not allocate daser.store and do "
        "not persist daser.index. Incompatible with --transfer-mode=gds.",
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=BLOCK_TOKENS,
        help="vLLM KV block size in tokens. Must match vLLM --block-size.",
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
        RuntimeError: if vLLM is unreachable or reports no models.
    """
    client = VLLMClient(base_url=vllm_base_url, model="")
    try:
        try:
            models = await client.list_models()
        except Exception as exc:  # noqa: BLE001
            raise VLLMStartupError(
                f"vLLM is not reachable at {vllm_base_url}. "
                "Please start vLLM before starting DaseR, then retry. "
                f"Original error: {exc}"
            ) from exc
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
        ValueError: if L2 size is not a positive slot multiple.
    """
    vllm_model_id = getattr(args, "vllm_model_id", None) or args.model_path
    if vllm_model_id is None:
        raise ValueError("vLLM model id has not been resolved")
    model_id, model_path = _resolve_model_paths(args, str(vllm_model_id))
    args.vllm_model_id = model_id
    args.model_path = model_path
    skip_l2 = bool(getattr(args, "skip_l2", False))
    transfer_mode = str(args.transfer_mode)
    if skip_l2 and transfer_mode == "gds":
        raise ValueError(
            "--skip-l2 is incompatible with --transfer-mode=gds because "
            "GDS requires an L2 store file"
        )
    l1_size = (
        min(DEFAULT_IOURING_L1_BYTES, int(args.l2_size))
        if args.l1_size is None and (transfer_mode == "iouring" or skip_l2)
        else int(args.l1_size or 0)
    )
    total_store_bytes = l1_size if skip_l2 else int(args.l2_size)
    cfg = DaserConfig(
        model_path=model_path,
        vllm_model_id=model_id,
        store_dir=args.store_dir,
        total_store_bytes=total_store_bytes,
        ipc_socket_path=args.socket_path,
        log_level=args.log_level,
        block_tokens=int(args.block_tokens),
        cache_reuse_mode=args.cache_reuse_mode,
        transfer_mode=transfer_mode,
        l1_size_bytes=l1_size,
        skip_l2=skip_l2,
    )
    slot_size = cfg.resolved_slot_size()
    if cfg.total_store_bytes <= 0 or cfg.total_slots <= 0:
        size_arg = "--l1-size" if skip_l2 else "--l2-size"
        raise ValueError(
            f"{size_arg} ({cfg.total_store_bytes}) must be at least one "
            f"slot ({slot_size} bytes)"
        )
    if skip_l2 and cfg.l1_size_bytes != cfg.l2_size_bytes:
        cfg.l1_size_bytes = cfg.l2_size_bytes
    if cfg.transfer_mode == "iouring" and cfg.l1_size_bytes <= 0:
        raise ValueError("--l1-size must be positive for iouring transfer")
    if not skip_l2 and cfg.l1_size_bytes and cfg.l1_size_bytes > cfg.l2_size_bytes:
        raise ValueError("--l1-size must not exceed --l2-size")
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
        block_tokens=int(args.block_tokens),
        cache_reuse_mode=args.cache_reuse_mode,
        align_document_chunks=args.cache_reuse_mode == CACHE_REUSE_CHUNK,
        transfer_mode=args.transfer_mode,
    )


def _log_startup_version(version: str = __version__) -> None:
    """Log the resolved DaseR version during server startup.

    Args:
        version: resolved DaseR package version.

    Returns:
        None.

    Async/thread-safety:
        This helper performs a single logger call and is safe to invoke from
        the main asyncio startup path.
    """
    logger.info("[SERVER] %s", startup_version_message(version))


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
    if cache_reuse_mode == CACHE_REUSE_PREFIX:
        return PrefixHashIndex(block_tokens=block_tokens), FixedOffsetEncoder(
            fixed_offset=0
        )
    if cache_reuse_mode == CACHE_REUSE_CHUNK:
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

    if cfg.skip_l2:
        logger.info("[SERVER] skip_l2 enabled; cold-starting volatile index")
    elif os.path.exists(cfg.index_path):
        try:
            cm.load(
                cfg.index_path,
                expected_cache_reuse_mode=cfg.cache_reuse_mode,
            )
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


async def _shutdown_server(
    http_server: uvicorn.Server,
    http_task: asyncio.Task[Any],
    ipc_server: IPCServer,
    core: ServerCore,
    index_path: str,
    cache_reuse_mode: str | None = None,
    skip_l2: bool = False,
    wait_for: Callable[[Awaitable[Any], float], Awaitable[Any]] = asyncio.wait_for,
) -> None:
    """Persist a fast consistent snapshot and close server resources.

    Args:
        http_server: running uvicorn server instance.
        http_task: task executing ``http_server.serve``.
        ipc_server: DaseR IPC server.
        core: shared server core whose chunk manager owns persistence.
        index_path: destination path for the saved control-plane snapshot.
        cache_reuse_mode: cache reuse mode to record in the snapshot.
        skip_l2: when True, do not persist metadata because L1-only bytes are
            volatile and have no backing store.
        wait_for: injectable awaitable timeout helper for tests.

    Async/thread-safety:
        Runs on the main DaseR asyncio event loop during SIGTERM/SIGINT
        shutdown. It stops new HTTP and IPC acceptance before saving the
        current in-memory index.
    """
    http_server.should_exit = True
    if not http_task.done():
        try:
            await wait_for(http_task, 5)
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            pass

    await ipc_server.stop_accepting()

    if skip_l2:
        logger.info("[SERVER] skip_l2 enabled; not saving volatile index")
    else:
        logger.info("[SERVER] shutting down; saving index to %s", index_path)
        parent = os.path.dirname(index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            if cache_reuse_mode is None:
                core.chunk_manager.save(index_path)
            else:
                core.chunk_manager.save(index_path, cache_reuse_mode=cache_reuse_mode)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SERVER] failed to save index: %s", exc)
    await ipc_server.close()
    logger.info("[SERVER] shutdown complete")


def _consume_completed_task(task: asyncio.Task[Any]) -> None:
    """Read a completed server task result without surfacing cancellation.

    Args:
        task: completed asyncio task from the main server wait set.

    Async/thread-safety:
        Called on the main server event loop after ``asyncio.wait`` returns.
    """
    try:
        task.result()
    except asyncio.CancelledError:
        logger.debug("[SERVER] task %s cancelled during shutdown", task.get_name())


async def run_server(args: argparse.Namespace) -> None:
    """Run the unified DaseR server until SIGTERM/SIGINT.

    Args:
        args: parsed CLI arguments.
    """
    _log_startup_banner()
    _log_startup_version()
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

    # Eagerly initialize the transfer layer so the first inference batch
    # does not pay the cost of pinned-memory allocation (which can take
    # tens of seconds for large L1 pools).
    if cfg.transfer_mode != "gds":
        await ipc_server.initialize_transfer()

    app = build_http_app(
        _build_http_config(args),
        core,
        drain_transfer=ipc_server.drain_transfer,
    )
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
        "L1-only" if cfg.skip_l2 else cfg.store_path,
        cfg.total_slots,
    )
    if cfg.skip_l2:
        logger.info(
            "[SERVER] skip_l2 enabled: lookup/store use volatile L1 only; "
            "GDS is disabled because it requires an L2 store file",
        )

    stop_task = asyncio.create_task(stop_event.wait(), name="daser-stop")
    try:
        done, pending = await asyncio.wait(
            [http_task, stop_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            if task is not http_task:
                task.cancel()
        for task in done:
            _consume_completed_task(task)
    finally:
        await _shutdown_server(
            http_server=http_server,
            http_task=http_task,
            ipc_server=ipc_server,
            core=core,
            index_path=cfg.index_path,
            cache_reuse_mode=cfg.cache_reuse_mode,
            skip_l2=cfg.skip_l2,
        )


def main() -> None:
    """CLI entry point for ``python -m daser.server``."""
    try:
        asyncio.run(run_server(_parse_args()))
    except (VLLMStartupError, ValueError) as exc:
        logger.error("[SERVER] %s", exc)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
