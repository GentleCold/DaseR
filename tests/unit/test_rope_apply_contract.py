# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest
import torch

# First Party
from daser.ops import rope_apply


def test_apply_rope_delta_to_key_block_rejects_rotary_dim_over_head_dim() -> None:
    """rotary_dim larger than head_dim is a misconfiguration and must raise.

    The shape guard runs before any CUDA dispatch, so this is exercised on
    CPU tensors. Previously this silently returned, masking a configuration
    error that would leave KV unrotated.
    """
    key_block = torch.zeros(4, 2, 8, dtype=torch.float32)

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
    kv_block = torch.zeros(1, 2, 2, 4, 2, 8, dtype=torch.float32)

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
    key_block = torch.zeros(4, 2, 8, dtype=torch.float32)

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
