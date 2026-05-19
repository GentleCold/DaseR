# SPDX-License-Identifier: Apache-2.0

# Standard
import json
from pathlib import Path

# First Party
from daser.config import BLOCK_TOKENS, DaserConfig, model_geometry_from_path


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
        "transfer_backend": "gds",
        "l1_cache_size": 0,
    }


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
