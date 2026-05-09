# SPDX-License-Identifier: Apache-2.0

# Third Party
import torch

# First Party
from daser.connector.rope import apply_rope_delta


def test_apply_rope_delta_zero_delta_returns_equal_tensor() -> None:
    key = torch.randn(2, 4, 6)

    rotated = apply_rope_delta(
        key,
        source_start=8,
        target_start=8,
        block_tokens=4,
    )

    assert torch.allclose(rotated, key)
    assert rotated.data_ptr() != key.data_ptr()


def test_apply_rope_delta_rotates_known_pair() -> None:
    key = torch.tensor([[[1.0, 0.0]]])

    rotated = apply_rope_delta(
        key,
        source_start=0,
        target_start=1,
        block_tokens=1,
        rope_theta=10000.0,
    )

    expected = torch.tensor(
        [[[torch.cos(torch.tensor(1.0)), torch.sin(torch.tensor(1.0))]]]
    )
    assert torch.allclose(rotated, expected, atol=1e-6)
