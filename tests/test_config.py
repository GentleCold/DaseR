# SPDX-License-Identifier: Apache-2.0
import pytest
from daser.config import DaserConfig


def test_explicit_slot_size():
    cfg = DaserConfig(slot_size=1024)
    assert cfg.resolved_slot_size() == 1024


def test_computed_slot_size():
    cfg = DaserConfig(
        num_kv_heads=8,
        head_dim=128,
        num_layers=28,
        block_tokens=16,
        dtype_bytes=2,
    )
    expected = 8 * 128 * 2 * 28 * 16 * 2
    assert cfg.resolved_slot_size() == expected


def test_missing_model_param_raises():
    cfg = DaserConfig(num_kv_heads=8, head_dim=128)  # num_layers missing
    with pytest.raises(ValueError, match="num_layers"):
        cfg.resolved_slot_size()
