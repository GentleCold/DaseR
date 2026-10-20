# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import json
from pathlib import Path
import sys
from typing import Any

# Third Party
import pytest

# First Party
from daser.compression import (
    CompressedStoreGeometry,
    calibrate_codebooks,
    default_online_codebooks,
    write_calibration_artifact,
)
from daser.config import (
    DEFAULT_IOURING_L1_BYTES,
    DEFAULT_TIER_GRAIN,
    model_geometry_from_path,
)
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.__main__ import (
    DASER_ASCII_BANNER,
    DASER_BANNER_COLOR,
    DASER_BANNER_RESET,
    VLLMStartupError,
    _build_core,
    _build_daser_config,
    _build_http_config,
    _build_index_components,
    _compression_runtime_config,
    _consume_completed_task,
    _ensure_store_file,
    _log_startup_banner,
    _parse_args,
    _parse_size_bytes,
    _read_vllm_model_id,
    _resolve_model_paths,
    _shutdown_server,
    main,
)
from tests.server.core_helpers import alloc_and_store


def _run_parse(argv: list[str]):
    saved = sys.argv
    sys.argv = ["daser.server", *argv]
    try:
        return _parse_args()
    finally:
        sys.argv = saved


def _write_model_config(path: Path, *, dtype: str = "bfloat16") -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 28,
                "torch_dtype": dtype,
            }
        ),
        encoding="utf-8",
    )


def _compressed_args(
    tmp_path: Path,
    *,
    dtype: str = "bfloat16",
    extra: list[str] | None = None,
) -> Any:
    """Build parsed arguments for the constrained compressed startup mode."""
    model_path = tmp_path / "model"
    _write_model_config(model_path, dtype=dtype)
    geometry = model_geometry_from_path(str(model_path))
    argv = [
        "--model-path",
        str(model_path),
        "--store-dir",
        str(tmp_path / "store"),
        "--vllm-base-url",
        "http://127.0.0.1:8001",
        "--l2-size",
        str(geometry.slot_size_for_block_tokens(128)),
        "--storage-format",
        "compressed-online",
        "--cache-reuse-mode",
        "prefix",
        "--block-tokens",
        "128",
    ]
    return _run_parse([*argv, *(extra or [])])


def test_parse_size_bytes_accepts_human_readable_units() -> None:
    assert _parse_size_bytes("10gb") == 10 * 1000**3
    assert _parse_size_bytes("10gib") == 10 * 1024**3
    assert _parse_size_bytes("512mb") == 512 * 1000**2
    assert _parse_size_bytes("512mib") == 512 * 1024**2
    assert _parse_size_bytes("2097152") == 2097152


def test_documented_flags_populate_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
            "--socket-path",
            "/tmp/daser.sock",
            "--transfer-mode",
            "iouring",
            "--l1-size",
            "1gb",
            "--trust-remote-code",
        ]
    )
    cfg = _build_daser_config(args)

    assert args.port == 2026
    assert cfg.model_path == str(model_path)
    assert cfg.vllm_model_id == str(model_path)
    assert cfg.store_path == str(store_dir / "daser.store")
    assert cfg.ipc_socket_path == "/tmp/daser.sock"
    assert cfg.index_path == str(store_dir / "daser.index")
    assert cfg.transfer_mode == "iouring"
    assert cfg.l1_size_bytes == 1000**3
    assert cfg.l2_size_bytes == 10 * 1000**3
    assert cfg.tier_grain == DEFAULT_TIER_GRAIN
    assert cfg.l2_lane_bytes == 10 * 1000**3 // DEFAULT_TIER_GRAIN * DEFAULT_TIER_GRAIN

    http_cfg = _build_http_config(args)
    assert http_cfg.vllm_base_url == "http://127.0.0.1:8001"
    assert http_cfg.model == str(model_path)
    assert http_cfg.tokenizer == str(model_path)
    assert http_cfg.align_document_chunks is True
    assert http_cfg.trust_remote_code is True
    assert args.cache_reuse_mode == "chunk"
    runtime = cfg.runtime_config()
    assert runtime["transfer_mode"] == "iouring"
    assert runtime["l1_size_bytes"] == 1000**3
    assert runtime["l2_size_bytes"] == cfg.store_file_bytes
    assert runtime["tier_grain"] == DEFAULT_TIER_GRAIN
    assert runtime["l1_enabled"] is True and runtime["l2_enabled"] is True
    assert runtime["staging_bytes"] == 1000**3 // 8 // DEFAULT_TIER_GRAIN * (
        DEFAULT_TIER_GRAIN
    )
    assert runtime["l1_lane_bytes"] == cfg.l1_lane_bytes > 0
    assert "coalesce_load_misses" not in runtime
    assert "bip_enabled" not in runtime


def test_tier_grain_flag_sets_the_allocation_unit(tmp_path: Path) -> None:
    """``--tier-grain`` reaches config and runtime config and is validated."""
    base = [
        "--model-path",
        str(tmp_path / "model"),
        "--store-dir",
        str(tmp_path / "store"),
        "--vllm-base-url",
        "http://127.0.0.1:8001",
    ]
    _write_model_config(tmp_path / "model")

    cfg = _build_daser_config(_run_parse([*base, "--tier-grain", "1MiB"]))

    assert cfg.tier_grain == 1 << 20
    assert cfg.l2_lane_bytes % (1 << 20) == 0
    assert cfg.runtime_config()["tier_grain"] == 1 << 20
    with pytest.raises(ValueError, match="tier_grain"):
        _build_daser_config(_run_parse([*base, "--tier-grain", "96KiB"]))


def test_gds_has_no_l1_tier(tmp_path: Path) -> None:
    """GDS ignores --l1-size: no L1 lanes and no staging pool."""
    _write_model_config(tmp_path / "model")
    args = _run_parse(
        [
            "--model-path",
            str(tmp_path / "model"),
            "--store-dir",
            str(tmp_path / "store"),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--transfer-mode",
            "gds",
            "--l1-size",
            "1GiB",
        ]
    )

    cfg = _build_daser_config(args)
    runtime = cfg.runtime_config()

    assert cfg.l1_size_bytes == 0
    assert runtime["l1_enabled"] is False
    assert runtime["l1_lane_bytes"] == 0
    assert runtime["staging_bytes"] == 0
    assert runtime["l2_lane_bytes"] == cfg.l2_lane_bytes > 0


def test_prefetch_lease_budget_flag_populates_runtime_config(tmp_path: Path) -> None:
    """``--prefetch-lease-budget`` reaches config and runtime config."""
    base = [
        "--model-path",
        str(tmp_path / "model"),
        "--store-dir",
        str(tmp_path / "store"),
        "--vllm-base-url",
        "http://127.0.0.1:8001",
        "--l1-size",
        "2GiB",
        "--l2-size",
        "4GiB",
    ]
    _write_model_config(tmp_path / "model")

    default_cfg = _build_daser_config(_run_parse(base))
    cfg = _build_daser_config(_run_parse([*base, "--prefetch-lease-budget", "1GiB"]))

    assert default_cfg.runtime_config()["prefetch_lease_budget_bytes"] == 0
    assert cfg.prefetch_lease_budget_bytes == 1 << 30
    assert cfg.runtime_config()["prefetch_lease_budget_bytes"] == 1 << 30
    with pytest.raises(ValueError, match="--prefetch-lease-budget"):
        _build_daser_config(_run_parse([*base, "--prefetch-lease-budget", "3GiB"]))


def test_l2_read_stripe_flag_populates_runtime_config(tmp_path: Path) -> None:
    """``--l2-read-stripe`` reaches config and runtime config."""
    base = [
        "--model-path",
        str(tmp_path / "model"),
        "--store-dir",
        str(tmp_path / "store"),
        "--vllm-base-url",
        "http://127.0.0.1:8001",
    ]
    _write_model_config(tmp_path / "model")

    default_cfg = _build_daser_config(_run_parse(base))
    cfg = _build_daser_config(_run_parse([*base, "--l2-read-stripe", "64MiB"]))

    assert default_cfg.runtime_config()["l2_read_stripe_bytes"] == 0
    assert cfg.runtime_config()["l2_read_stripe_bytes"] == 64 << 20


def test_extra_kv_layers_grow_every_slot(tmp_path: Path) -> None:
    """``--extra-kv-layers`` adds layers of the target KV shape to each slot."""
    base = [
        "--model-path",
        str(tmp_path / "model"),
        "--store-dir",
        str(tmp_path / "store"),
        "--vllm-base-url",
        "http://127.0.0.1:8001",
    ]
    _write_model_config(tmp_path / "model")
    layers = model_geometry_from_path(str(tmp_path / "model")).num_layers

    default_cfg = _build_daser_config(_run_parse(base))
    cfg = _build_daser_config(_run_parse([*base, "--extra-kv-layers", "1"]))

    per_layer = default_cfg.resolved_local_slot_size() // layers
    assert cfg.resolved_local_slot_size() == per_layer * (layers + 1)
    assert cfg.runtime_config()["slot_size"] == per_layer * (layers + 1)
    with pytest.raises(ValueError, match="extra_kv_layers"):
        _build_daser_config(_run_parse([*base, "--extra-kv-layers", "-1"]))


def test_default_transfer_mode_is_iouring(tmp_path: Path) -> None:
    """The server defaults to iouring unless a transfer mode is specified."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
        ]
    )

    cfg = _build_daser_config(args)

    assert args.transfer_mode == "iouring"
    assert cfg.transfer_mode == "iouring"
    assert cfg.l1_size_bytes == DEFAULT_IOURING_L1_BYTES
    assert cfg.runtime_config()["transfer_mode"] == "iouring"
    assert cfg.runtime_config()["l1_size_bytes"] == cfg.l1_size_bytes


def test_skip_l2_populates_memory_only_runtime_config(tmp_path: Path) -> None:
    """--skip-l2 drops the L2 tier, the store file and index persistence."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
            "--skip-l2",
        ]
    )

    cfg = _build_daser_config(args)
    runtime = cfg.runtime_config()

    assert cfg.skip_l2 is True
    assert cfg.transfer_mode == "iouring"
    assert cfg.l1_size_bytes == DEFAULT_IOURING_L1_BYTES
    assert cfg.persists_index is False
    assert runtime["skip_l2"] is True
    assert runtime["transfer_mode"] == "iouring"
    assert runtime["store_path"] == ""
    assert runtime["l2_enabled"] is False
    assert runtime["l2_size_bytes"] == 0
    assert runtime["l2_lane_bytes"] == 0


def test_skip_l2_gives_the_whole_l1_budget_to_l1_lanes(
    tmp_path: Path,
) -> None:
    """Without L2 nothing stages, so the L1 lanes take the whole budget."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l1-size",
            "2gib",
            "--skip-l2",
        ]
    )

    cfg = _build_daser_config(args)
    runtime = cfg.runtime_config()

    assert cfg.skip_l2 is True
    assert cfg.l1_size_bytes == 2 * 1024**3
    assert runtime["store_path"] == ""
    assert runtime["staging_bytes"] == 0
    assert runtime["l1_lane_bytes"] == 2 * 1024**3
    assert runtime["l2_size_bytes"] == 0


def test_skip_l2_rejects_gds_with_clear_message(tmp_path: Path) -> None:
    """GDS requires an L2 store file, so it cannot run with --skip-l2."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--transfer-mode",
            "gds",
            "--skip-l2",
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        _build_daser_config(args)

    message = str(exc_info.value)
    assert "--skip-l2 is incompatible with --transfer-mode=gds" in message
    assert "GDS requires an L2 store file" in message


def test_cache_reuse_mode_chunk_selects_chunk_components(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--cache-reuse-mode",
            "chunk",
        ]
    )

    retrieval, position = _build_index_components(args.cache_reuse_mode, 16)
    http_cfg = _build_http_config(args)

    assert isinstance(retrieval, ChunkReuseIndex)
    assert isinstance(position, ChunkPositionEncoder)
    assert http_cfg.align_document_chunks is True


def test_cache_reuse_mode_prefix_selects_prefix_components(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--cache-reuse-mode",
            "prefix",
        ]
    )

    retrieval, position = _build_index_components(args.cache_reuse_mode, 16)

    assert isinstance(retrieval, PrefixHashIndex)
    assert isinstance(position, FixedOffsetEncoder)


@pytest.mark.parametrize("flag", ["--l2-size", "--l1-size"])
def test_each_tier_lane_must_fit_one_record(tmp_path: Path, flag: str) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            flag,
            # One raw 16-token record of the test model is 896 KiB.
            "512kib",
        ]
    )
    with pytest.raises(ValueError, match=f"{flag}.*one record needs"):
        _build_daser_config(args)


def test_ensure_store_file_truncates_larger_legacy_file(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
        ]
    )
    cfg = _build_daser_config(args)
    store_dir.mkdir()
    with open(cfg.store_path, "wb") as f:
        f.truncate(cfg.store_file_bytes + cfg.tier_grain)

    _ensure_store_file(cfg)

    assert Path(cfg.store_path).stat().st_size == cfg.store_file_bytes


def test_ensure_store_file_rejects_smaller_existing_file(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "10gb",
        ]
    )
    cfg = _build_daser_config(args)
    store_dir.mkdir()
    with open(cfg.store_path, "wb") as f:
        f.truncate(cfg.store_file_bytes - 1)

    with pytest.raises(ValueError, match="has size"):
        _ensure_store_file(cfg)


def test_skip_l2_does_not_create_store_file(tmp_path: Path) -> None:
    """Memory-only mode must not allocate daser.store at startup."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(store_dir),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--skip-l2",
        ]
    )
    cfg = _build_daser_config(args)

    _ensure_store_file(cfg)

    assert store_dir.exists()
    assert not Path(cfg.store_path).exists()


@pytest.mark.parametrize(
    ("extra", "required"),
    [
        (["--transfer-mode", "gds"], "--transfer-mode=iouring"),
        (["--cache-reuse-mode", "chunk"], "--cache-reuse-mode=prefix"),
        (["--tensor-parallel-size", "2"], "--tensor-parallel-size=1"),
        (["--block-tokens", "16"], "--block-tokens=128"),
    ],
)
def test_compressed_online_rejects_unsupported_runtime_modes(
    tmp_path: Path,
    extra: list[str],
    required: str,
) -> None:
    args = _compressed_args(tmp_path, extra=extra)

    with pytest.raises(ValueError, match=required):
        _build_daser_config(args)


def test_compressed_online_accepts_memory_only_store(tmp_path: Path) -> None:
    args = _compressed_args(tmp_path, extra=["--skip-l2", "--l1-size", "64mib"])

    cfg = _build_daser_config(args)

    assert cfg.skip_l2 is True
    assert cfg.storage_format == "compressed-online"
    assert cfg.l1_lane_bytes == 64 * 1024**2
    assert cfg.l2_lane_bytes == 0
    assert cfg.persists_index is False


def test_compressed_online_rejects_fp16_model(tmp_path: Path) -> None:
    args = _compressed_args(tmp_path, dtype="float16")

    with pytest.raises(ValueError, match="BF16 KV dtype"):
        _build_daser_config(args)


def _calibrated_dir(tmp_path: Path, cfg: Any, *, num_layers: int) -> Path:
    """Fit a calibration artifact whose high bytes favor a fixed palette."""
    model = cfg.model_geometry()
    geometry = CompressedStoreGeometry(
        slot_size=num_layers * 2 * 128 * model.num_kv_heads * model.head_dim * 2,
        block_tokens=128,
        num_layers=num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
    )
    raw = bytearray(geometry.slot_size)
    high = bytes([0x05, 0x06, 0x07]) * (geometry.slot_size // 6 + 1)
    raw[1::2] = high[: geometry.slot_size // 2]
    artifact = calibrate_codebooks(
        [bytes(raw)], model_id="test/model", geometry=geometry
    )
    out = tmp_path / f"calibration-{num_layers}"
    write_calibration_artifact(artifact, out)
    return out


def test_compression_runtime_config_defaults_to_generic_palette(
    tmp_path: Path,
) -> None:
    args = _compressed_args(tmp_path)
    cfg = _build_daser_config(args)
    model = cfg.model_geometry()

    runtime = _compression_runtime_config(cfg, args)

    geometry = CompressedStoreGeometry(
        slot_size=cfg.resolved_local_slot_size(),
        block_tokens=128,
        num_layers=model.num_layers,
        num_kv_heads=model.num_kv_heads,
        head_dim=model.head_dim,
    )
    assert args.compressed_codebooks is None
    assert runtime["compressed_codebooks"] == default_online_codebooks(geometry)


def test_compression_runtime_config_publishes_calibrated_codebooks(
    tmp_path: Path,
) -> None:
    args = _compressed_args(tmp_path)
    cfg = _build_daser_config(args)
    calibrated = _calibrated_dir(
        tmp_path, cfg, num_layers=cfg.model_geometry().num_layers
    )
    (tmp_path / "second").mkdir()
    args = _compressed_args(
        tmp_path / "second", extra=["--compressed-codebooks", str(calibrated)]
    )
    cfg = _build_daser_config(args)

    runtime = _compression_runtime_config(cfg, args)

    assert (
        runtime["compressed_codebooks"] == (calibrated / "codebooks.bin").read_bytes()
    )
    assert runtime["compressed_codebooks"][:3] == bytes([0x05, 0x06, 0x07])


def test_compression_runtime_config_rejects_foreign_geometry(tmp_path: Path) -> None:
    args = _compressed_args(tmp_path)
    cfg = _build_daser_config(args)
    args.compressed_codebooks = str(
        _calibrated_dir(tmp_path, cfg, num_layers=cfg.model_geometry().num_layers + 1)
    )

    with pytest.raises(ValueError, match="num_layers"):
        _compression_runtime_config(cfg, args)


def test_compression_runtime_config_is_empty_for_raw_storage(tmp_path: Path) -> None:
    args = _compressed_args(tmp_path)
    args.storage_format = "raw"
    cfg = _build_daser_config(args)

    assert _compression_runtime_config(cfg, args) == {}


@pytest.mark.asyncio
async def test_compressed_online_discards_stale_index(
    tmp_path: Path,
) -> None:
    """Online packed refs are volatile, so no snapshot may outlive the run."""
    cfg = _build_daser_config(_compressed_args(tmp_path))
    Path(cfg.store_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.index_path).write_bytes(b"stale-raw-index")

    assert not cfg.persists_index
    await _build_core(cfg)

    assert not Path(cfg.index_path).exists()


def _raw_config(tmp_path: Path, *extra: str) -> Any:
    """Build a raw-storage config with a small L2 tier."""
    model_path = tmp_path / "model"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--model-path",
            str(model_path),
            "--store-dir",
            str(tmp_path / "store"),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
            "--l2-size",
            "64mib",
            "--cache-reuse-mode",
            "prefix",
            *extra,
        ]
    )
    return _build_daser_config(args)


@pytest.mark.asyncio
async def test_build_core_restores_a_saved_index(tmp_path: Path) -> None:
    cfg = _raw_config(tmp_path)
    Path(cfg.store_dir).mkdir(parents=True)
    core = await _build_core(cfg)
    await alloc_and_store(core, "a", token_count=16)
    core.save_index(cfg.index_path, cache_reuse_mode=cfg.cache_reuse_mode)

    restored = await _build_core(cfg)

    assert restored.is_chunk_committed("a")
    assert restored.tier_layout == core.tier_layout


@pytest.mark.asyncio
async def test_dirty_restart_cannot_reuse_a_clean_snapshot(tmp_path: Path) -> None:
    """A loaded checkpoint cannot survive later writes and an interrupted run."""
    cfg = _raw_config(tmp_path)
    Path(cfg.store_dir).mkdir(parents=True)
    core = await _build_core(cfg)
    await alloc_and_store(core, "old", token_count=16)
    core.save_index(cfg.index_path, cache_reuse_mode=cfg.cache_reuse_mode)
    restored = await _build_core(cfg)
    assert restored.is_chunk_committed("old")
    assert not Path(cfg.index_path).exists()
    await restored.evict_chunk("old")
    await alloc_and_store(restored, "replacement", token_count=16)
    # No clean shutdown publishes the replacement's checkpoint.
    restarted = await _build_core(cfg)
    assert not restarted.is_chunk_committed("old")
    assert not restarted.is_chunk_committed("replacement")


@pytest.mark.asyncio
@pytest.mark.parametrize("store_state", ["missing", "oversized", "intact"])
async def test_store_recreation_invalidates_old_checkpoint(
    tmp_path: Path, store_state: str
) -> None:
    """Only an unchanged original store may recover its previous chunk keys."""
    cfg = _raw_config(tmp_path)
    Path(cfg.store_dir).mkdir(parents=True)
    core = await _build_core(cfg)
    await alloc_and_store(core, "old", token_count=16)
    core.save_index(cfg.index_path, cache_reuse_mode=cfg.cache_reuse_mode)
    if store_state != "missing":
        with open(cfg.store_path, "wb") as stream:
            stream.truncate(
                cfg.store_file_bytes
                + (cfg.tier_grain if store_state == "oversized" else 0)
            )
    _ensure_store_file(cfg)
    assert Path(cfg.index_path).exists() == (store_state == "intact")
    restored = await _build_core(cfg)
    assert restored.is_chunk_committed("old") == (store_state == "intact")


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["corrupt", "layout"])
async def test_build_core_cold_starts_on_an_unusable_index(
    tmp_path: Path, change: str
) -> None:
    """A corrupt index or one saved for other lanes falls back to cold start."""
    cfg = _raw_config(tmp_path)
    Path(cfg.store_dir).mkdir(parents=True)
    if change == "corrupt":
        Path(cfg.index_path).write_bytes(b"not-an-index")
    else:
        core = await _build_core(cfg)
        await alloc_and_store(core, "a", token_count=16)
        core.save_index(cfg.index_path, cache_reuse_mode=cfg.cache_reuse_mode)
        (tmp_path / "other").mkdir()
        cfg = _raw_config(tmp_path / "other", "--tier-grain", "128KiB")
        cfg.store_dir = str(tmp_path / "store")

    restored = await _build_core(cfg)

    assert not restored.is_chunk_committed("a")
    assert restored.chunk_manager.next_slot == 0


def test_model_path_is_optional_when_vllm_model_is_local_path(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    _write_model_config(model_path)
    args = _run_parse(
        [
            "--store-dir",
            str(tmp_path / "store"),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
        ]
    )

    model_id, resolved_model_path = _resolve_model_paths(
        args,
        vllm_model_id=str(model_path),
    )

    assert model_id == str(model_path)
    assert resolved_model_path == str(model_path)


def test_model_path_is_required_when_vllm_model_is_not_local_path(
    tmp_path: Path,
) -> None:
    args = _run_parse(
        [
            "--store-dir",
            str(tmp_path / "store"),
            "--vllm-base-url",
            "http://127.0.0.1:8001",
        ]
    )
    with pytest.raises(ValueError, match="--model-path is required"):
        _resolve_model_paths(args, vllm_model_id="served-alias")


@pytest.mark.asyncio
async def test_read_vllm_model_id_explains_when_vllm_is_unavailable(
    monkeypatch,
) -> None:
    """Startup should tell users to start vLLM before DaseR."""

    class FakeVLLMClient:
        def __init__(self, base_url: str, model: str) -> None:
            self.base_url = base_url
            self.model = model

        async def list_models(self) -> list[str]:
            raise OSError("connection refused")

        async def close(self) -> None:
            return None

    monkeypatch.setattr("daser.server.__main__.VLLMClient", FakeVLLMClient)

    with pytest.raises(RuntimeError) as exc_info:
        await _read_vllm_model_id("http://127.0.0.1:8001")

    message = str(exc_info.value)
    assert "vLLM is not reachable at http://127.0.0.1:8001" in message
    assert "Please start vLLM before starting DaseR" in message
    assert "connection refused" in message


def test_startup_banner_logs_daser_ascii_art(monkeypatch) -> None:
    """DaseR startup should print a recognizable terminal banner."""
    expected_banner = """

████▄   ▄▄▄   ▄▄▄▄ ▄▄▄▄▄ █████▄
██  ██ ██▀██ ███▄▄ ██▄▄  ██▄▄██▄
████▀  ██▀██ ▄▄██▀ ██▄▄▄ ██   ██
"""
    expected_colored = (
        "\n"
        "\n████▄   ▄▄▄   ▄▄▄▄ ▄▄▄▄▄ "
        f"{DASER_BANNER_COLOR}█████▄{DASER_BANNER_RESET}\n"
        "██  ██ ██▀██ ███▄▄ ██▄▄  "
        f"{DASER_BANNER_COLOR}██▄▄██▄{DASER_BANNER_RESET}\n"
        "████▀  ██▀██ ▄▄██▀ ██▄▄▄ "
        f"{DASER_BANNER_COLOR}██   ██{DASER_BANNER_RESET}\n"
    )
    messages: list[str] = []
    monkeypatch.setattr(
        "daser.server.__main__.logger.info",
        lambda message, *args: messages.append(message % args if args else message),
    )

    _log_startup_banner()

    assert DASER_ASCII_BANNER == expected_banner
    assert DASER_BANNER_COLOR == "\033[38;2;102;178;255m"
    assert DASER_BANNER_RESET == "\033[0m"
    assert expected_colored in messages


def test_main_exits_cleanly_when_vllm_is_unavailable(monkeypatch) -> None:
    """The CLI should show the vLLM hint without a Python traceback."""
    messages: list[str] = []

    async def fake_run_server(_args: Any) -> None:
        raise VLLMStartupError("Please start vLLM before starting DaseR")

    monkeypatch.setattr("daser.server.__main__._parse_args", lambda: object())
    monkeypatch.setattr("daser.server.__main__.run_server", fake_run_server)
    monkeypatch.setattr(
        "daser.server.__main__.logger.error",
        lambda message, *args: messages.append(message % args if args else message),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert messages == ["[SERVER] Please start vLLM before starting DaseR"]


def test_main_exits_cleanly_on_invalid_startup_config(monkeypatch) -> None:
    """The CLI should print configuration errors without a Python traceback."""
    messages: list[str] = []

    async def fake_run_server(_args: Any) -> None:
        raise ValueError(
            "--skip-l2 is incompatible with --transfer-mode=gds because "
            "GDS requires an L2 store file"
        )

    monkeypatch.setattr("daser.server.__main__._parse_args", lambda: object())
    monkeypatch.setattr("daser.server.__main__.run_server", fake_run_server)
    monkeypatch.setattr(
        "daser.server.__main__.logger.error",
        lambda message, *args: messages.append(message % args if args else message),
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    assert messages == [
        "[SERVER] --skip-l2 is incompatible with --transfer-mode=gds because "
        "GDS requires an L2 store file"
    ]


def test_store_dir_is_required(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    _write_model_config(model_path)
    with pytest.raises(SystemExit):
        _run_parse(
            [
                "--model-path",
                str(model_path),
                "--vllm-base-url",
                "http://127.0.0.1:8001",
            ]
        )


def test_http_flags_are_required():
    with pytest.raises(SystemExit):
        _run_parse(["--store-dir", "/tmp/store"])


def test_consume_completed_task_ignores_cancelled_http_task() -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError

    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(cancelled())
        with pytest.raises(asyncio.CancelledError):
            loop.run_until_complete(task)

        _consume_completed_task(task)
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_shutdown_server_stops_acceptance_before_saving(tmp_path: Path) -> None:
    events: list[str] = []
    index_path = str(tmp_path / "daser.index")

    class FakeHTTPTask:
        def done(self) -> bool:
            return False

    class FakeHTTPServer:
        should_exit = False

    class FakeCore:
        def save_index(self, path: str, cache_reuse_mode: str | None) -> None:
            events.append(f"save:{path}:{cache_reuse_mode}")

    class FakeIPCServer:
        async def stop_accepting(self) -> None:
            events.append("stop_accepting")

        async def close(self) -> None:
            events.append("close")

    async def fake_wait_for(task: Any, timeout: float) -> None:
        assert isinstance(task, FakeHTTPTask)
        assert timeout == 5
        events.append("http_wait")

    http_server = FakeHTTPServer()
    await _shutdown_server(
        http_server=http_server,
        http_task=FakeHTTPTask(),
        ipc_server=FakeIPCServer(),
        core=FakeCore(),
        index_path=index_path,
        cache_reuse_mode="prefix",
        wait_for=fake_wait_for,
    )

    assert http_server.should_exit is True
    assert events == [
        "http_wait",
        "stop_accepting",
        f"save:{index_path}:prefix",
        "close",
    ]


@pytest.mark.asyncio
async def test_shutdown_server_saves_after_cancelled_http_task(
    tmp_path: Path,
) -> None:
    events: list[str] = []
    index_path = str(tmp_path / "daser.index")

    class FakeHTTPTask:
        def done(self) -> bool:
            return False

    class FakeHTTPServer:
        should_exit = False

    class FakeCore:
        def save_index(self, path: str, cache_reuse_mode: str | None) -> None:
            events.append(f"save:{path}:{cache_reuse_mode}")

    class FakeIPCServer:
        async def stop_accepting(self) -> None:
            events.append("stop_accepting")

        async def close(self) -> None:
            events.append("close")

    async def cancelled_wait_for(_task: Any, _timeout: float) -> None:
        events.append("http_cancelled")
        raise asyncio.CancelledError

    http_server = FakeHTTPServer()
    await _shutdown_server(
        http_server=http_server,
        http_task=FakeHTTPTask(),
        ipc_server=FakeIPCServer(),
        core=FakeCore(),
        index_path=index_path,
        wait_for=cancelled_wait_for,
    )

    assert http_server.should_exit is True
    assert events == [
        "http_cancelled",
        "stop_accepting",
        f"save:{index_path}:None",
        "close",
    ]


@pytest.mark.asyncio
async def test_shutdown_server_skips_index_save_when_l2_is_skipped(
    tmp_path: Path,
) -> None:
    """Memory-only mode must not persist daser.index during shutdown."""
    events: list[str] = []
    index_path = str(tmp_path / "daser.index")

    class FakeHTTPTask:
        def done(self) -> bool:
            return True

    class FakeHTTPServer:
        should_exit = False

    class FakeCore:
        def save_index(self, path: str, cache_reuse_mode: str | None) -> None:
            events.append(f"save:{path}:{cache_reuse_mode}")

    class FakeIPCServer:
        async def stop_accepting(self) -> None:
            events.append("stop_accepting")

        async def close(self) -> None:
            events.append("close")

    await _shutdown_server(
        http_server=FakeHTTPServer(),
        http_task=FakeHTTPTask(),
        ipc_server=FakeIPCServer(),
        core=FakeCore(),
        index_path=index_path,
        persist_index=False,
    )

    assert events == ["stop_accepting", "close"]
    assert not Path(index_path).exists()


def test_compressed_l2_is_sized_by_l2_size_alone(tmp_path: Path) -> None:
    """Compressed-online sizes L2 like raw: --l2-size, without an L1 arena."""
    args = _compressed_args(tmp_path)
    config = _build_daser_config(args)
    runtime = config.runtime_config()

    assert config.l2_size_bytes == int(args.l2_size)
    assert config.store_file_bytes == config.l2_lane_bytes == int(args.l2_size)
    assert runtime["l2_size_bytes"] == config.store_file_bytes
    assert "l2_budget_bytes" not in runtime
    assert "total_store_bytes" not in runtime
