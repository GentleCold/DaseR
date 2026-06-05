# SPDX-License-Identifier: Apache-2.0

"""Resolve pytest scratch directories on the data disk with CI fallback."""

# Standard
import os
from pathlib import Path
import re
import shutil
import time
import uuid

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_SCRATCH_ROOT = "DASER_TEST_SCRATCH_ROOT"
PYTEST_DIR_NAME = "pytest"
RESULTS_DIR_NAME = "results"
CONFIG_SESSION_BASETEMP_KEY = "daser_session_basetemp"

_DEFAULT_DATA_ROOT = Path("/data")
_FALLBACK_DIR_NAME = ".test-scratch"
_SESSION_DIR_RE = re.compile(r"^\d+-[a-f0-9]{8}$")
# Orphaned session dirs from crashed runs; safe to prune after this age.
STALE_SESSION_MAX_AGE_SECONDS = 6 * 3600


def resolve_scratch_root() -> Path:
    """Return the writable root directory for ephemeral test artifacts.

    Resolution order:

    1. ``DASER_TEST_SCRATCH_ROOT`` when set.
    2. ``/data/<USER>/daser-test`` when ``/data`` exists and is writable.
    3. ``<repo>/.test-scratch`` for CI runners and other environments without
       a shared data disk.

    Returns:
        Absolute scratch root path. The directory is created when possible.

    Async/thread-safety:
        Synchronous filesystem helper intended for pytest startup/shutdown.
    """
    env_root = os.environ.get(ENV_SCRATCH_ROOT)
    if env_root:
        path = Path(env_root).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    username = os.environ.get("USER") or os.environ.get("LOGNAME") or "daser"
    if _DEFAULT_DATA_ROOT.is_dir():
        data_root = (_DEFAULT_DATA_ROOT / username / "daser-test").resolve()
        if _is_writable_directory(data_root):
            return data_root

    fallback = (REPO_ROOT / _FALLBACK_DIR_NAME).resolve()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def allocate_pytest_session_basetemp() -> Path:
    """Return a unique per-pytest-session ``--basetemp`` directory.

    Each pytest invocation gets its own subdirectory under
    ``<scratch-root>/pytest/`` named ``<pid>-<uuid>``. This avoids pytest
    deleting another run's active temp files when two sessions share a parent
    ``--basetemp`` path.

    Stale session directories from earlier crashed runs are pruned on allocate.

    Returns:
        Absolute path to the new session directory.

    Async/thread-safety:
        Synchronous filesystem helper intended for pytest startup.
    """
    sessions_parent = resolve_scratch_root() / PYTEST_DIR_NAME
    sessions_parent.mkdir(parents=True, exist_ok=True)
    _cleanup_legacy_pytest_root_entries(sessions_parent)
    _prune_stale_session_dirs(sessions_parent)
    session_dir = sessions_parent / _new_session_dir_name()
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def resolve_results_dir() -> Path:
    """Return the preserved results directory under the scratch root.

    Returns:
        Absolute path to ``<scratch-root>/results``. The directory is created
        when missing.

    Async/thread-safety:
        Synchronous filesystem helper safe for pytest fixtures.
    """
    path = resolve_scratch_root() / RESULTS_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def cleanup_ephemeral_scratch(*, session_basetemp: Path | None = None) -> None:
    """Delete ephemeral scratch artifacts while preserving ``results/``.

    When ``session_basetemp`` is provided and lies under the managed
    ``<scratch-root>/pytest/<pid>-<uuid>`` layout, only that session directory
    is removed. Otherwise no active session tree is deleted (for example when
    the caller passed an explicit ``--basetemp`` on the CLI).

    Always prunes stale managed session directories and legacy ``pytest-of-*``
    siblings at the scratch root. The ``results/`` subtree is never deleted.

    Args:
        session_basetemp: Basetemp directory for the finishing pytest session,
            or ``None`` when pytest did not allocate a managed session path.

    Async/thread-safety:
        Synchronous cleanup intended for pytest session shutdown.
    """
    root = resolve_scratch_root()
    sessions_parent = root / PYTEST_DIR_NAME

    if session_basetemp is not None:
        session_path = session_basetemp.expanduser().resolve()
        if _is_managed_session_dir(session_path, root):
            shutil.rmtree(session_path, ignore_errors=True)

    if sessions_parent.is_dir():
        _prune_stale_session_dirs(sessions_parent)

    if not root.exists():
        return

    for entry in root.iterdir():
        if entry.name == RESULTS_DIR_NAME:
            continue
        if entry.is_dir() and entry.name.startswith("pytest-of-"):
            shutil.rmtree(entry, ignore_errors=True)


def _new_session_dir_name() -> str:
    """Return a unique managed session directory name.

    Returns:
        ``<pid>-<uuid>`` string safe for directory names.

    Async/thread-safety:
        Synchronous helper for pytest startup.
    """
    return f"{os.getpid()}-{uuid.uuid4().hex[:8]}"


def _is_managed_session_dir(path: Path, scratch_root: Path) -> bool:
    """Return whether ``path`` is a DaseR-managed pytest session directory.

    Args:
        path: Candidate session basetemp path.
        scratch_root: Resolved scratch root for the current environment.

    Returns:
        ``True`` when ``path`` is ``<scratch-root>/pytest/<pid>-<uuid8>``.

    Async/thread-safety:
        Synchronous path check for cleanup guards.
    """
    sessions_parent = (scratch_root / PYTEST_DIR_NAME).resolve()
    try:
        rel = path.resolve().relative_to(sessions_parent)
    except ValueError:
        return False
    if len(rel.parts) != 1:
        return False
    return _SESSION_DIR_RE.fullmatch(rel.parts[0]) is not None


def is_managed_session_dir_name(name: str) -> bool:
    """Return whether ``name`` matches the managed session directory pattern.

    Args:
        name: Final path component under ``<scratch-root>/pytest/``.

    Returns:
        ``True`` when ``name`` is ``<pid>-<uuid8>``.

    Async/thread-safety:
        Pure string check with no I/O.
    """
    return _SESSION_DIR_RE.fullmatch(name) is not None


def _cleanup_legacy_pytest_root_entries(sessions_parent: Path) -> None:
    """Remove pre-session-layout artifacts directly under ``pytest/``.

    Older scratch helpers placed ``daser.store`` and sockets in the shared
    ``pytest/`` root. Current runs use per-session subdirectories instead.

    Args:
        sessions_parent: ``<scratch-root>/pytest`` directory.

    Async/thread-safety:
        Synchronous cleanup helper for pytest startup.
    """
    if not sessions_parent.is_dir():
        return

    for entry in sessions_parent.iterdir():
        if entry.is_dir() and is_managed_session_dir_name(entry.name):
            continue
        if entry.is_file():
            entry.unlink(missing_ok=True)
        elif entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)


def _prune_stale_session_dirs(sessions_parent: Path) -> None:
    """Remove managed session directories older than the stale threshold.

    Args:
        sessions_parent: ``<scratch-root>/pytest`` directory.

    Async/thread-safety:
        Synchronous cleanup helper; safe to call during pytest startup/shutdown.
    """
    if not sessions_parent.is_dir():
        return

    now = time.time()
    for entry in sessions_parent.iterdir():
        if not entry.is_dir():
            continue
        if _SESSION_DIR_RE.fullmatch(entry.name) is None:
            continue
        try:
            age_seconds = now - entry.stat().st_mtime
        except OSError:
            continue
        if age_seconds >= STALE_SESSION_MAX_AGE_SECONDS:
            shutil.rmtree(entry, ignore_errors=True)


def _is_writable_directory(path: Path) -> bool:
    """Return whether ``path`` can be created and written by the current user.

    Args:
        path: Candidate scratch directory.

    Returns:
        ``True`` when the directory exists or can be created and probed.

    Async/thread-safety:
        Synchronous probe that creates the directory when needed.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError:
        return False
