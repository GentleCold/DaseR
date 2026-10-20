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

from daser.compression import (
    CompressedStoreGeometry,
    default_online_codebooks,
    load_calibrated_codebooks,
)
from daser.compression.format import ONLINE_TILE_SCALARS, digest_bytes

# First Party
from daser.config import (
    BLOCK_TOKENS,
    CACHE_REUSE_CHUNK,
    CACHE_REUSE_MODES,
    CACHE_REUSE_PREFIX,
    DEFAULT_CACHE_REUSE_MODE,
    DEFAULT_IOURING_L1_BYTES,
    DEFAULT_L2_BYTES,
    DEFAULT_TIER_GRAIN,
    L1_ACCOUNTING_MODES,
    STORAGE_FORMAT_COMPRESSED_ONLINE,
    STORAGE_FORMAT_RAW,
    STORAGE_FORMATS,
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
from daser.server.store_file import ensure_store_file
from daser.server.tiers import TierLayout

logger = init_logger(__name__)


class VLLMStartupError(RuntimeError):
    """Raised when DaseR cannot contact vLLM during startup.

    Async/thread-safety:
        Exception type only; safe to raise from sync or async startup paths.
    """


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
    """Create the store directory and, unless skip_l2, the L2 store file.

    Args:
        cfg: resolved DaseR server config.

    Raises:
        ValueError: if an existing store file is smaller than one L2 lane per
            TP rank.
    """
    os.makedirs(cfg.store_dir, exist_ok=True)
    if cfg.skip_l2:
        logger.info("[SERVER] skip_l2 enabled; not allocating L2 store file")
        return
    ensure_store_file(cfg.store_path, cfg.store_file_bytes)


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
        "--trust-remote-code",
        action="store_true",
        help="Allow custom model repository code when loading the tokenizer.",
    )
    parser.add_argument(
        "--store-dir",
        required=True,
        help="Directory for daser.store and daser.index",
    )
    parser.add_argument(
        "--l2-size",
        type=_parse_size_bytes,
        default=DEFAULT_L2_BYTES,
        help="L2 SSD-tier capacity across all TP ranks, e.g. 10gb, 10gib, "
        "512mb, or bytes; each rank gets a grain-aligned lane of 1/TP.",
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
        help="Pinned-memory budget for --transfer-mode=iouring: the L1 lanes "
        "plus a staging pool of min(256MiB, 1/8) when L2 is enabled. Defaults "
        "to 1GiB; ignored by --transfer-mode=gds.",
    )
    parser.add_argument(
        "--tier-grain",
        type=_parse_size_bytes,
        default=DEFAULT_TIER_GRAIN,
        help="Allocation unit of the L1 and L2 tiers; a power of two multiple "
        "of 4KiB (default 64KiB).",
    )
    parser.add_argument(
        "--skip-l2",
        action="store_true",
        help="Use volatile L1 memory only: do not allocate daser.store and do "
        "not persist daser.index. Incompatible with --transfer-mode=gds.",
    )
    parser.add_argument(
        "--storage-format",
        choices=STORAGE_FORMATS,
        default=STORAGE_FORMAT_RAW,
        help="Physical KV storage format. compressed-online losslessly packs "
        "KV on store and decodes it on load.",
    )
    parser.add_argument(
        "--compressed-codebooks",
        default=None,
        help="Directory of an offline calibration artifact (metadata.json, "
        "codebooks.bin) whose per-plane codebooks replace the generic palette "
        "for --storage-format=compressed-online (default: generic palette).",
    )
    parser.add_argument(
        "--prefetch-lease-budget",
        type=_parse_size_bytes,
        default=0,
        help="Cap on L1 bytes pinned by in-flight scheduler prefetch leases "
        "for --transfer-mode=iouring, counted in --l1-accounting units; a "
        "prefetch over the cap waits for earlier leases (default 0: no cap).",
    )
    parser.add_argument(
        "--l2-read-stripe",
        type=_parse_size_bytes,
        default=0,
        help="For --transfer-mode=iouring, split each large L2 read into parts "
        "of about this size submitted together, keeping many IOs in flight per "
        "request; must be a multiple of 4 KiB (default 0: one serial read).",
    )
    parser.add_argument(
        "--extra-kv-layers",
        type=int,
        default=0,
        help="Attention layers vLLM registers beyond the model's own, stored "
        "in every slot with the same KV shape (for example 1 for a one-layer "
        "EAGLE drafter; default 0).",
    )
    parser.add_argument(
        "--l1-accounting",
        choices=L1_ACCOUNTING_MODES,
        default="stored",
        help="L1 capacity charge: stored bytes (default) or raw-equivalent "
        "bytes carried by each exact KV record.",
    )
    parser.add_argument(
        "--block-tokens",
        type=int,
        default=BLOCK_TOKENS,
        help="vLLM KV block size in tokens. Must match vLLM --block-size.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="vLLM tensor parallel size. Must match vLLM --tensor-parallel-size.",
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
        ValueError: if the tier sizes cannot hold one record or the selected
            options conflict.
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
    if transfer_mode == "gds":
        if args.l1_size:
            logger.warning("[SERVER] --transfer-mode=gds has no L1; ignoring --l1-size")
        l1_size = 0
    elif args.l1_size is None:
        l1_size = DEFAULT_IOURING_L1_BYTES
    else:
        l1_size = int(args.l1_size)
    cfg = DaserConfig(
        model_path=model_path,
        vllm_model_id=model_id,
        store_dir=args.store_dir,
        l2_size_bytes=0 if skip_l2 else int(args.l2_size),
        ipc_socket_path=args.socket_path,
        log_level=args.log_level,
        block_tokens=int(args.block_tokens),
        cache_reuse_mode=args.cache_reuse_mode,
        transfer_mode=transfer_mode,
        l1_size_bytes=l1_size,
        skip_l2=skip_l2,
        tier_grain=int(getattr(args, "tier_grain", DEFAULT_TIER_GRAIN)),
        tensor_parallel_size=int(args.tensor_parallel_size),
        storage_format=str(getattr(args, "storage_format", STORAGE_FORMAT_RAW)),
        l1_accounting=str(getattr(args, "l1_accounting", "stored")),
        prefetch_lease_budget_bytes=int(getattr(args, "prefetch_lease_budget", 0)),
        l2_read_stripe_bytes=int(getattr(args, "l2_read_stripe", 0)),
        extra_kv_layers=int(getattr(args, "extra_kv_layers", 0)),
    )
    if cfg.transfer_mode == "iouring" and cfg.l1_size_bytes <= 0:
        raise ValueError("--l1-size must be positive for iouring transfer")
    # A raw record is the largest record a chunk can store, so each enabled
    # tier lane must hold at least one.
    raw_record = cfg.resolved_local_slot_size()
    min_lane = -(-raw_record // cfg.tier_grain) * cfg.tier_grain
    if cfg.l1_enabled and cfg.l1_lane_bytes < min_lane:
        raise ValueError(
            f"--l1-size ({cfg.l1_size_bytes}) leaves {cfg.l1_lane_bytes} bytes "
            f"per TP rank after {cfg.staging_bytes} staging bytes; one record "
            f"needs {min_lane}"
        )
    if cfg.l2_enabled and cfg.l2_lane_bytes < min_lane:
        raise ValueError(
            f"--l2-size ({cfg.l2_size_bytes}) leaves {cfg.l2_lane_bytes} bytes "
            f"per TP rank; one record needs {min_lane}"
        )
    if cfg.prefetch_lease_budget_bytes > cfg.l1_size_bytes:
        raise ValueError("--prefetch-lease-budget must not exceed --l1-size")
    if cfg.storage_format == STORAGE_FORMAT_COMPRESSED_ONLINE:
        geometry = cfg.model_geometry()
        violations = []
        if cfg.transfer_mode != "iouring":
            violations.append("--transfer-mode=iouring")
        if cfg.cache_reuse_mode != CACHE_REUSE_PREFIX:
            violations.append("--cache-reuse-mode=prefix")
        if cfg.tensor_parallel_size != 1:
            violations.append("--tensor-parallel-size=1")
        if cfg.block_tokens != 128:
            violations.append("--block-tokens=128")
        if geometry.dtype_name != "bfloat16":
            violations.append("BF16 KV dtype")
        if violations:
            raise ValueError(f"{cfg.storage_format} requires " + ", ".join(violations))
    return cfg


def _tier_layout(cfg: DaserConfig) -> TierLayout:
    """Return the tier geometry of a finalized server config.

    Args:
        cfg: Config accepted by ``_build_daser_config``.

    Returns:
        TierLayout with the config's grain, lanes and L1 record charge.
    """
    return TierLayout(
        grain=cfg.tier_grain,
        l1_lane_bytes=cfg.l1_lane_bytes,
        l2_lane_bytes=cfg.l2_lane_bytes,
        tensor_parallel_size=cfg.tensor_parallel_size,
        l1_record_bytes=cfg.l1_record_bytes,
    )


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
        trust_remote_code=bool(args.trust_remote_code),
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
    if cache_reuse_mode == CACHE_REUSE_PREFIX:
        return PrefixHashIndex(block_tokens=block_tokens), FixedOffsetEncoder(
            fixed_offset=0
        )
    if cache_reuse_mode == CACHE_REUSE_CHUNK:
        return ChunkReuseIndex(block_tokens=block_tokens), ChunkPositionEncoder(
            initial_offset=0
        )
    raise ValueError(f"unknown cache reuse mode: {cache_reuse_mode}")


def _new_core(cfg: DaserConfig) -> ServerCore:
    """Construct an empty server core for a config.

    Args:
        cfg: DaseR runtime config.

    Returns:
        ServerCore with no chunks.
    """
    retrieval_index, position_encoder = _build_index_components(
        cfg.cache_reuse_mode, cfg.block_tokens
    )
    return ServerCore(
        chunk_manager=ChunkManager(MetadataStore(), DocRegistry()),
        retrieval_index=retrieval_index,
        position_encoder=position_encoder,
        tier_layout=_tier_layout(cfg),
        block_tokens=cfg.block_tokens,
    )


async def _build_core(cfg: DaserConfig) -> ServerCore:
    """Construct the shared server core and restore a persisted index.

    Args:
        cfg: DaseR runtime config.

    Returns:
        ServerCore holding the restored chunks, or empty on a cold start.

    Async/thread-safety:
        Startup only; the index read blocks the event loop briefly.
    """
    core = _new_core(cfg)
    if cfg.skip_l2:
        logger.info("[SERVER] skip_l2 enabled; cold-starting volatile index")
    elif not cfg.persists_index:
        # Compressed records overwrite L2 extents, so a prior snapshot must
        # not survive for a later raw restart to trust.
        if os.path.exists(cfg.index_path):
            os.remove(cfg.index_path)
        logger.info(
            "[SERVER] %s keeps a volatile index; cold-starting", cfg.storage_format
        )
    elif os.path.exists(cfg.index_path):
        try:
            await core.load_index(
                cfg.index_path,
                expected_cache_reuse_mode=cfg.cache_reuse_mode,
            )
            logger.info("[SERVER] restored index from %s", cfg.index_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[SERVER] cold start; index load failed: %s", exc)
            core = _new_core(cfg)
        # A clean snapshot describes the extents before this process started.
        # Once we admit writes those extents can be reused. Consume even a
        # rejected snapshot, and fail startup if it cannot be removed, so a
        # later dirty restart cannot expose overwritten bytes under old keys.
        os.remove(cfg.index_path)
    return core


async def _shutdown_server(
    http_server: uvicorn.Server,
    http_task: asyncio.Task[Any],
    ipc_server: IPCServer,
    core: ServerCore,
    index_path: str,
    cache_reuse_mode: str | None = None,
    persist_index: bool = True,
    wait_for: Callable[[Awaitable[Any], float], Awaitable[Any]] = asyncio.wait_for,
) -> None:
    """Persist a fast consistent snapshot and close server resources.

    Args:
        http_server: running uvicorn server instance.
        http_task: task executing ``http_server.serve``.
        ipc_server: DaseR IPC server.
        core: shared server core whose index is persisted.
        index_path: destination path for the saved control-plane snapshot.
        cache_reuse_mode: cache reuse mode to record in the snapshot.
        persist_index: when False, do not save metadata because the store
            cannot be described by a snapshot (memory-only or compressed-online).
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
    # Finish in-flight L2 writes so their chunks are saved with full copies.
    try:
        await ipc_server.drain_transfer()
    except Exception as exc:  # noqa: BLE001
        logger.exception("[SERVER] transfer drain before save failed: %s", exc)

    if not persist_index:
        logger.info("[SERVER] not saving volatile index")
    else:
        logger.info("[SERVER] shutting down; saving index to %s", index_path)
        parent = os.path.dirname(index_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        try:
            core.save_index(index_path, cache_reuse_mode=cache_reuse_mode)
        except Exception as exc:  # noqa: BLE001
            logger.exception("[SERVER] failed to save index: %s", exc)
    await ipc_server.close()
    logger.info("[SERVER] shutdown complete")


def _compression_runtime_config(
    cfg: DaserConfig, args: argparse.Namespace
) -> dict[str, Any]:
    """Select the online codec tables published to workers over IPC.

    Args:
        cfg: Finalized server config.
        args: Parsed CLI arguments; ``compressed_codebooks`` optionally names
            an offline calibration artifact directory.

    Returns:
        Runtime-config entries for compressed-online, or an empty dict for
        the raw storage format.

    Raises:
        ValueError: If the calibration artifact does not match the geometry.

    Async/thread-safety:
        Synchronous startup file read before the IPC server starts.
    """
    if cfg.storage_format != STORAGE_FORMAT_COMPRESSED_ONLINE:
        return {}
    model = cfg.model_geometry()
    geometry = CompressedStoreGeometry(
        slot_size=cfg.resolved_local_slot_size(),
        block_tokens=cfg.block_tokens,
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
        dtype_bytes=model.dtype_bytes,
        tile_scalars=ONLINE_TILE_SCALARS,
    )
    codebooks_dir = getattr(args, "compressed_codebooks", None)
    codebooks = (
        load_calibrated_codebooks(codebooks_dir, geometry)
        if codebooks_dir
        else default_online_codebooks(geometry)
    )
    logger.info(
        "[SERVER] online codebooks=%s sha256=%s",
        "calibrated" if codebooks_dir else "generic",
        digest_bytes(codebooks).hex()[:16],
    )
    return {
        "compressed_codebooks": codebooks,
        "compressed_codebook_hash": digest_bytes(codebooks),
        "compressed_tile_scalars": geometry.tile_scalars,
    }


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
    args.vllm_model_id = await _read_vllm_model_id(args.vllm_base_url)
    cfg = _build_daser_config(args)
    _ensure_store_file(cfg)
    core = await _build_core(cfg)

    runtime_config = cfg.runtime_config()
    runtime_config.update(_compression_runtime_config(cfg, args))

    ipc_server = IPCServer(
        socket_path=cfg.ipc_socket_path,
        core=core,
        runtime_config=runtime_config,
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
        "[SERVER] model_id=%s model_path=%s store=%s",
        cfg.model_id,
        cfg.model_path,
        "L1-only" if cfg.skip_l2 else cfg.store_path,
    )
    logger.info(
        "[SERVER] tiers grain=%d tp=%d l1_lane=%d l2_lane=%d staging=%d",
        cfg.tier_grain,
        cfg.tensor_parallel_size,
        cfg.l1_lane_bytes,
        cfg.l2_lane_bytes,
        cfg.staging_bytes,
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
            persist_index=cfg.persists_index,
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
