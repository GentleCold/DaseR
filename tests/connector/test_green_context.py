# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from daser.ops import green_context
from daser.ops.green_context import create_green_context, parse_green_context_sm_count


def test_green_context_sm_count_defaults_to_disabled() -> None:
    """Missing and blank settings preserve the primary CUDA stream path."""
    assert parse_green_context_sm_count(None, "TEST_GREEN") == 0
    assert parse_green_context_sm_count("", "TEST_GREEN") == 0
    assert parse_green_context_sm_count("  ", "TEST_GREEN") == 0


@pytest.mark.parametrize("value", ["0", "8", "32", " 64 "])
def test_green_context_sm_count_accepts_non_negative_integers(value: str) -> None:
    """Explicit counts are parsed without changing their magnitude."""
    assert parse_green_context_sm_count(value, "TEST_GREEN") == int(value)


@pytest.mark.parametrize("value", ["-1", "eight", "1.5"])
def test_green_context_sm_count_rejects_invalid_values(value: str) -> None:
    """Invalid opt-in settings fail before a worker starts serving traffic."""
    with pytest.raises(ValueError, match="TEST_GREEN"):
        parse_green_context_sm_count(value, "TEST_GREEN")


def test_green_context_reports_missing_driver_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported drivers fail with an actionable Green Context error."""
    monkeypatch.setattr(
        green_context.ctypes,
        "CDLL",
        lambda _name: SimpleNamespace(),
    )
    device = SimpleNamespace(type="cuda", index=0)
    with pytest.raises(RuntimeError, match="driver does not expose"):
        create_green_context(device=device, sm_count=8)  # type: ignore[arg-type]
