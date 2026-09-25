# SPDX-License-Identifier: Apache-2.0
"""Protect the PyTorch/CUDA priority ordering used by load and store streams."""

from types import SimpleNamespace

import pytest

# CPU CI installs an import-only torch stub without the CUDA namespace.
pytest.importorskip("torch.cuda")

import torch

from daser.ops.stream_priority import cuda_stream_priority


@pytest.mark.parametrize("most_urgent", [-1, -3])
def test_priority_names_follow_pytorch_range_order(
    monkeypatch: pytest.MonkeyPatch, most_urgent: int
) -> None:
    """Named priorities respect PyTorch's tuple order on each device range."""
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: SimpleNamespace(priority_range=lambda: (0, most_urgent)),
    )

    assert cuda_stream_priority("low") == 0
    assert cuda_stream_priority("high") == most_urgent
