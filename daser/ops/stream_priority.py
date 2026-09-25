# SPDX-License-Identifier: Apache-2.0
"""Resolve CUDA priority names against PyTorch's device-specific range."""

import torch


def cuda_stream_priority(priority: str) -> int:
    """Return the supported numeric priority for a low or high priority stream.

    Args:
        priority: ``"low"`` for the least urgent priority or ``"high"`` for
            the most urgent priority on the current CUDA device.

    Returns:
        An integer suitable for ``torch.cuda.Stream(priority=...)``.

    Raises:
        ValueError: If priority is neither ``"low"`` nor ``"high"``.

    Async/thread-safety:
        Query once during stream initialization on its owning thread, after
        selecting the CUDA device. Does not synchronize or submit GPU work.
    """
    if priority not in {"low", "high"}:
        raise ValueError("CUDA stream priority must be 'low' or 'high'")
    # PyTorch returns (least urgent, most urgent), commonly (0, -1).
    # CUDA's higher scheduling priority has the smaller numeric value.
    least_urgent, most_urgent = torch.cuda.current_stream().priority_range()
    return least_urgent if priority == "low" else most_urgent
