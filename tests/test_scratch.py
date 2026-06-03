# SPDX-License-Identifier: Apache-2.0

"""Tests for pytest scratch directory resolution and cleanup."""

# Standard
import os
from pathlib import Path
import time

# Third Party
import pytest

# First Party
from tests import scratch


def test_resolve_scratch_root_honors_env_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``DASER_TEST_SCRATCH_ROOT`` overrides the default data-disk location."""
    custom_root = tmp_path / "custom-scratch"
    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(custom_root))

    resolved = scratch.resolve_scratch_root()

    assert resolved == custom_root.resolve()
    assert resolved.is_dir()


def test_allocate_pytest_session_basetemp_is_unique_per_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Each pytest session gets its own basetemp subdirectory."""
    root = tmp_path / "scratch-root"
    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(root))

    first = scratch.allocate_pytest_session_basetemp()
    second = scratch.allocate_pytest_session_basetemp()

    assert first != second
    assert first.parent == second.parent == (root / scratch.PYTEST_DIR_NAME).resolve()
    assert scratch.is_managed_session_dir_name(first.name)
    assert scratch.is_managed_session_dir_name(second.name)


def test_cleanup_ephemeral_scratch_removes_only_managed_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Session cleanup deletes one managed session dir and leaves others."""
    root = tmp_path / "scratch-root"
    sessions_parent = root / scratch.PYTEST_DIR_NAME
    active = sessions_parent / "1000-aabbccdd"
    other = sessions_parent / "2000-bbccddee"
    active.mkdir(parents=True)
    other.mkdir(parents=True)
    (active / "daser.store").write_bytes(b"active")
    (other / "daser.store").write_bytes(b"other")

    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(root))
    scratch.cleanup_ephemeral_scratch(session_basetemp=active)

    assert not active.exists()
    assert other.exists()
    assert (other / "daser.store").read_bytes() == b"other"


def test_cleanup_ephemeral_scratch_preserves_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Session cleanup removes pytest temp dirs but keeps ``results/``."""
    root = tmp_path / "scratch-root"
    sessions_parent = root / scratch.PYTEST_DIR_NAME
    session_dir = sessions_parent / "3000-ccddeeff"
    results_dir = root / scratch.RESULTS_DIR_NAME
    legacy_dir = root / "pytest-of-sza"
    session_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)
    legacy_dir.mkdir(parents=True)
    (session_dir / "daser.store").write_bytes(b"tmp")
    (results_dir / "report.json").write_text("{}")
    (legacy_dir / "old.store").write_bytes(b"old")

    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(root))
    scratch.cleanup_ephemeral_scratch(session_basetemp=session_dir)

    assert not session_dir.exists()
    assert not legacy_dir.exists()
    assert (results_dir / "report.json").read_text() == "{}"


def test_cleanup_ignores_unmanaged_basetemp_outside_scratch_layout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Explicit external basetemp paths are not deleted by session cleanup."""
    root = tmp_path / "scratch-root"
    external = tmp_path / "external-basetemp"
    external.mkdir(parents=True)
    (external / "keep.txt").write_text("stay")

    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(root))
    scratch.cleanup_ephemeral_scratch(session_basetemp=external)

    assert external.exists()
    assert (external / "keep.txt").read_text() == "stay"


def test_allocate_prunes_stale_managed_sessions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Stale managed session directories are pruned on allocate."""
    root = tmp_path / "scratch-root"
    sessions_parent = root / scratch.PYTEST_DIR_NAME
    stale = sessions_parent / "4000-aaaabbbb"
    fresh = sessions_parent / "5000-bbbbcccc"
    sessions_parent.mkdir(parents=True)
    stale.mkdir()
    fresh.mkdir()
    stale_mtime = time.time() - scratch.STALE_SESSION_MAX_AGE_SECONDS - 60
    os.utime(stale, (stale_mtime, stale_mtime))

    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(root))
    scratch.allocate_pytest_session_basetemp()

    assert not stale.exists()
    assert fresh.exists()


def test_allocate_removes_legacy_flat_pytest_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Legacy files in the shared ``pytest/`` root are removed on allocate."""
    root = tmp_path / "scratch-root"
    sessions_parent = root / scratch.PYTEST_DIR_NAME
    sessions_parent.mkdir(parents=True)
    legacy_store = sessions_parent / "daser.store"
    legacy_store.write_bytes(b"legacy")

    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(root))
    session_dir = scratch.allocate_pytest_session_basetemp()

    assert not legacy_store.exists()
    assert session_dir.parent == sessions_parent.resolve()
    assert scratch.is_managed_session_dir_name(session_dir.name)


def test_resolve_results_dir_is_under_scratch_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The preserved results directory lives under the scratch root."""
    root = tmp_path / "scratch-root"
    monkeypatch.setenv(scratch.ENV_SCRATCH_ROOT, str(root))

    results_dir = scratch.resolve_results_dir()

    assert results_dir == (root / scratch.RESULTS_DIR_NAME).resolve()
    assert results_dir.is_dir()
