# SPDX-License-Identifier: Apache-2.0

"""Tests for pytest scratch directory resolution and cleanup."""

# Standard
from pathlib import Path

# Third Party
import pytest

# First Party
from tests import scratch


def test_resolve_scratch_root_honors_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``DASER_TEST_SCRATCH_ROOT`` overrides the default data-disk location."""
    custom_root = tmp_path / "custom-scratch"
    monkeypatch.setenv(scratch._ENV_SCRATCH_ROOT, str(custom_root))

    resolved = scratch.resolve_scratch_root()

    assert resolved == custom_root.resolve()
    assert resolved.is_dir()


def test_cleanup_ephemeral_scratch_preserves_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Session cleanup removes pytest temp dirs but keeps ``results/``."""
    root = tmp_path / "scratch-root"
    pytest_dir = root / scratch._PYTEST_DIR_NAME
    results_dir = root / scratch._RESULTS_DIR_NAME
    legacy_dir = root / "pytest-of-sza"
    pytest_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    (pytest_dir / "daser.store").write_bytes(b"tmp")
    (results_dir / "report.json").write_text("{}")
    (legacy_dir / "old.store").write_bytes(b"old")

    monkeypatch.setenv(scratch._ENV_SCRATCH_ROOT, str(root))
    scratch.cleanup_ephemeral_scratch()

    assert not pytest_dir.exists()
    assert not legacy_dir.exists()
    assert (results_dir / "report.json").read_text() == "{}"


def test_resolve_results_dir_is_under_scratch_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The preserved results directory lives under the scratch root."""
    root = tmp_path / "scratch-root"
    monkeypatch.setenv(scratch._ENV_SCRATCH_ROOT, str(root))

    results_dir = scratch.resolve_results_dir()

    assert results_dir == (root / scratch._RESULTS_DIR_NAME).resolve()
    assert results_dir.is_dir()
