# SPDX-License-Identifier: Apache-2.0

# Standard
import json
from pathlib import Path
import sys

# Third Party
import pytest

# First Party
from daser.position.chunk_position import ChunkPositionEncoder
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.chunk_reuse import ChunkReuseIndex
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.__main__ import (
    _build_daser_config,
    _build_http_config,
    _build_index_components,
    _parse_args,
    _parse_size_bytes,
    _resolve_model_paths,
)


def _run_parse(argv: list[str]):
    saved = sys.argv
    sys.argv = ["daser.server", *argv]
    try:
        return _parse_args()
    finally:
        sys.argv = saved


def _write_model_config(path: Path) -> None:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "hidden_size": 1024,
                "num_attention_heads": 8,
                "num_key_value_heads": 4,
                "num_hidden_layers": 28,
                "torch_dtype": "bfloat16",
            }
        ),
        encoding="utf-8",
    )


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
            "--store-size",
            "10gb",
            "--socket-path",
            "/tmp/daser.sock",
        ]
    )
    cfg = _build_daser_config(args)

    assert cfg.model_path == str(model_path)
    assert cfg.vllm_model_id == str(model_path)
    assert cfg.store_path == str(store_dir / "daser.store")
    assert cfg.ipc_socket_path == "/tmp/daser.sock"
    assert cfg.index_path == str(store_dir / "daser.index")
    assert cfg.total_slots > 0
    assert cfg.aligned_store_bytes <= 10 * 1000**3
    assert cfg.aligned_store_bytes == cfg.total_slots * cfg.resolved_slot_size()

    http_cfg = _build_http_config(args)
    assert http_cfg.vllm_base_url == "http://127.0.0.1:8001"
    assert http_cfg.model == str(model_path)
    assert http_cfg.tokenizer == str(model_path)
    assert http_cfg.align_document_chunks is False
    assert args.cache_reuse_mode == "prefix"
    assert args.transfer_backend == "gds"
    assert args.l1_cache_size == 0
    assert cfg.transfer_backend == "gds"
    assert cfg.l1_cache_size == 0


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


def test_store_size_must_fit_at_least_one_slot(tmp_path: Path) -> None:
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
            "--store-size",
            "1",
        ]
    )
    with pytest.raises(ValueError, match="at least one slot"):
        _build_daser_config(args)


def test_iouring_mem_requires_positive_l1_cache_size(tmp_path: Path) -> None:
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
            "--transfer-backend",
            "iouring-mem",
        ]
    )
    with pytest.raises(ValueError, match="--l1-cache-size"):
        _build_daser_config(args)


def test_iouring_mem_accepts_l1_cache_size(tmp_path: Path) -> None:
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
            "--transfer-backend",
            "iouring-mem",
            "--l1-cache-size",
            "32gb",
        ]
    )
    cfg = _build_daser_config(args)

    assert cfg.transfer_backend == "iouring-mem"
    assert cfg.l1_cache_size == 32 * 1000**3


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
