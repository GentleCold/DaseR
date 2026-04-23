# SPDX-License-Identifier: Apache-2.0

# Standard
import sys

# Third Party
import pytest

# First Party
from daser.server.__main__ import _build_daser_config, _parse_args


def _run_parse(argv: list[str]):
    saved = sys.argv
    sys.argv = ["daser.server", *argv]
    try:
        return _parse_args()
    finally:
        sys.argv = saved


def test_documented_flags_populate_config():
    args = _run_parse(
        [
            "--store-path",
            "/tmp/daser.store",
            "--store-size",
            str(10 * 1024 * 1024 * 1024),
            "--socket-path",
            "/tmp/daser.sock",
            "--index-path",
            "/tmp/daser.index",
            "--slot-size",
            "2097152",
        ]
    )
    cfg = _build_daser_config(args)

    assert cfg.store_path == "/tmp/daser.store"
    assert cfg.ipc_socket_path == "/tmp/daser.sock"
    assert cfg.index_path == "/tmp/daser.index"
    assert cfg.slot_size == 2097152
    assert cfg.total_slots == 5120
    assert cfg.total_slots * cfg.resolved_slot_size() == 10 * 1024 * 1024 * 1024


def test_store_size_must_be_slot_aligned():
    args = _run_parse(
        [
            "--store-path",
            "/tmp/daser.store",
            "--store-size",
            "2097153",  # one byte past 2 MiB, not a slot multiple
            "--slot-size",
            "2097152",
        ]
    )
    with pytest.raises(ValueError, match="multiple of slot_size"):
        _build_daser_config(args)


def test_store_path_is_required():
    with pytest.raises(SystemExit):
        _run_parse(["--slot-size", "2097152"])


def test_slot_size_zero_uses_model_params():
    args = _run_parse(
        [
            "--store-path",
            "/tmp/daser.store",
            "--store-size",
            str(8 * 128 * 2 * 28 * 16 * 2 * 4),  # 4 slots worth
            "--slot-size",
            "0",
            "--num-kv-heads",
            "8",
            "--head-dim",
            "128",
            "--num-layers",
            "28",
        ]
    )
    cfg = _build_daser_config(args)

    assert cfg.slot_size == 0
    assert cfg.resolved_slot_size() == 8 * 128 * 2 * 28 * 16 * 2
    assert cfg.total_slots == 4
