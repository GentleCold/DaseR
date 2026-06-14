# SPDX-License-Identifier: Apache-2.0

# Standard
from dataclasses import dataclass

# Third Party
import pytest

# First Party
from daser.ops import rope_apply


@dataclass
class _FakeTensor:
    """Minimal stand-in exposing only what the rope shape guard reads.

    The ``rotary_dim`` boundary check runs before any real tensor op or CUDA
    dispatch, so the contract is verified without importing a real torch
    runtime (the CPU CI installs a torch stub with no tensor ops).
    """

    shape: tuple[int, ...]


def test_apply_rope_delta_to_key_block_rejects_rotary_dim_over_head_dim() -> None:
    """rotary_dim larger than head_dim is a misconfiguration and must raise.

    Previously this silently returned, masking a configuration error that
    would leave KV unrotated.
    """
    key_block = _FakeTensor(shape=(4, 2, 8))

    with pytest.raises(ValueError, match="rotary_dim must not exceed head_dim"):
        rope_apply.apply_rope_delta_to_key_block(
            key_block,
            delta=4,
            rope_base=10000.0,
            rotary_dim=16,
            is_neox_style=True,
        )


def test_apply_rope_delta_to_kv_key_block_rejects_rotary_dim_over_head_dim() -> None:
    """The KV-block wrapper raises on rotary_dim larger than head_dim."""
    kv_block = _FakeTensor(shape=(1, 2, 2, 4, 2, 8))

    with pytest.raises(ValueError, match="rotary_dim must not exceed head_dim"):
        rope_apply.apply_rope_delta_to_kv_key_block(
            kv_block,
            delta=4,
            rope_base=10000.0,
            rotary_dim=16,
            is_neox_style=True,
        )


def test_apply_rope_delta_noop_returns_without_shape_check() -> None:
    """delta==0 and rotary_dim<=0 remain no-ops regardless of head_dim."""
    key_block = _FakeTensor(shape=(4, 2, 8))

    # delta == 0 short-circuits before the shape guard.
    rope_apply.apply_rope_delta_to_key_block(
        key_block,
        delta=0,
        rope_base=10000.0,
        rotary_dim=16,
        is_neox_style=True,
    )
    # rotary_dim <= 0 short-circuits before the shape guard.
    rope_apply.apply_rope_delta_to_key_block(
        key_block,
        delta=4,
        rope_base=10000.0,
        rotary_dim=0,
        is_neox_style=True,
    )
