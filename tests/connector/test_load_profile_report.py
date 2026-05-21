# SPDX-License-Identifier: Apache-2.0
"""Unit tests for connector load profile reporting."""

from pathlib import Path

# Third Party
import pytest

pytest.importorskip("torch.profiler")
import torch

# First Party
from daser.connector.worker import (
    _format_load_profile_report,
    _tensorboard_trace_handler,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_format_load_profile_report_lists_daser_segments() -> None:
    """Profiler report includes labeled DaseR load segments."""
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function("daser::index_copy"):
            tensor = torch.randn(64, 64, device="cuda")
            tensor.add_(1.0)
        with record_function("daser::rope_delta"):
            tensor.mul_(2.0)

    report = _format_load_profile_report(prof, wall_segments={"daser::index_copy": 1.0})
    assert "daser::index_copy" in report
    assert "daser::rope_delta" in report
    assert "cuda_ms" in report
    assert "Wall clock" in report
    assert "daser::index_copy" in report.split("Wall clock")[-1]


def test_tensorboard_handler_skips_empty_load(tmp_path: Path) -> None:
    """TensorBoard export is skipped when no GDS read occurred."""
    from torch.profiler import ProfilerActivity, profile, record_function

    log_dir = str(tmp_path / "tb")
    wall: dict[str, float] = {}
    handler = _tensorboard_trace_handler(log_dir, wall)

    with profile(
        activities=[ProfilerActivity.CPU],
        on_trace_ready=handler,
    ):
        with record_function("daser::noop"):
            pass

    assert not list(tmp_path.glob("**/*.pt.trace.json"))

    wall["daser::gds_read"] = 1.0
    handler2 = _tensorboard_trace_handler(log_dir, wall)
    with profile(
        activities=[ProfilerActivity.CPU],
        on_trace_ready=handler2,
    ):
        with record_function("daser::index_copy"):
            x = torch.zeros(4)
            x.add_(1)

    assert list(tmp_path.glob("**/*.pt.trace.json"))
