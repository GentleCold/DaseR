# SPDX-License-Identifier: Apache-2.0

# Standard
import json
from pathlib import Path

# First Party
from daser.config import (
    BLOCK_TOKENS,
    DEFAULT_IOURING_L1_BYTES,
    DaserConfig,
    model_geometry_from_path,
)


def _write_model_config(path: Path, payload: dict[str, object]) -> None:
    path.mkdir()
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_model_geometry_from_path_reads_hf_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "bfloat16",
        },
    )

    geometry = model_geometry_from_path(str(model_path))

    assert geometry.num_kv_heads == 4
    assert geometry.head_dim == 128
    assert geometry.num_layers == 28
    assert geometry.dtype_bytes == 2
    assert geometry.slot_size == 4 * 128 * 2 * 28 * BLOCK_TOKENS * 2


def test_model_geometry_uses_explicit_head_dim_and_float32_dtype(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "model"
    _write_model_config(
        model_path,
        {
            "head_dim": 64,
            "hidden_size": 1024,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 2,
            "torch_dtype": "float32",
        },
    )

    geometry = model_geometry_from_path(str(model_path))

    assert geometry.head_dim == 64
    assert geometry.dtype_bytes == 4
    assert geometry.slot_size == 8 * 64 * 2 * 2 * BLOCK_TOKENS * 4


def test_daser_config_derives_paths_and_slot_size(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "float16",
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        total_store_bytes=4 * 128 * 2 * 28 * BLOCK_TOKENS * 2 * 4,
    )

    assert cfg.store_path == str(store_dir / "daser.store")
    assert cfg.index_path == str(store_dir / "daser.index")
    assert cfg.model_id == str(model_path)
    assert cfg.resolved_slot_size() == 4 * 128 * 2 * 28 * BLOCK_TOKENS * 2
    assert cfg.total_slots == 4
    assert cfg.aligned_store_bytes == cfg.resolved_slot_size() * 4


def test_daser_config_uses_configured_block_tokens_for_slot_size(
    tmp_path: Path,
) -> None:
    """Resolved slot size follows the configured vLLM block size."""
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "float16",
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        block_tokens=128,
    )

    assert cfg.resolved_slot_size() == 4 * 128 * 2 * 28 * 128 * 2
    assert cfg.runtime_config()["block_tokens"] == 128


def test_runtime_config_reuses_server_parameters(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "num_hidden_layers": 4,
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        total_store_bytes=512 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        ipc_socket_path="/tmp/custom.sock",
    )

    runtime_config = cfg.runtime_config()

    assert runtime_config == {
        "socket_path": "/tmp/custom.sock",
        "store_path": str(store_dir / "daser.store"),
        "slot_size": 8 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        "block_tokens": BLOCK_TOKENS,
        "model_id": str(model_path),
        "cache_reuse_mode": "chunk",
        "transfer_mode": "iouring",
        "l1_size_bytes": DEFAULT_IOURING_L1_BYTES,
        "l2_size_bytes": 512 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        "total_slots": 64,
        "total_store_bytes": 512 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        "skip_l2": False,
    }


def test_runtime_config_omits_store_path_when_l2_is_skipped(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "num_hidden_layers": 4,
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        total_store_bytes=512 * 64 * 2 * 4 * BLOCK_TOKENS * 2,
        skip_l2=True,
    )

    runtime_config = cfg.runtime_config()

    assert runtime_config["store_path"] == ""
    assert runtime_config["skip_l2"] is True
    assert runtime_config["l2_size_bytes"] == cfg.aligned_store_bytes
    assert runtime_config["total_slots"] == cfg.total_slots


def test_runtime_config_uses_aligned_l2_capacity(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 1024,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "num_hidden_layers": 28,
            "torch_dtype": "bfloat16",
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        total_store_bytes=10 * 1000**3,
    )

    runtime = cfg.runtime_config()

    assert cfg.aligned_store_bytes < cfg.total_store_bytes
    assert runtime["l2_size_bytes"] == cfg.aligned_store_bytes
    assert runtime["total_store_bytes"] == cfg.aligned_store_bytes


def test_model_id_can_differ_from_model_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    store_dir = tmp_path / "store"
    _write_model_config(
        model_path,
        {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "num_hidden_layers": 4,
        },
    )
    cfg = DaserConfig(
        model_path=str(model_path),
        store_dir=str(store_dir),
        vllm_model_id="served-model",
    )

    assert cfg.model_id == "served-model"
    assert cfg.runtime_config()["model_id"] == "served-model"
