# SPDX-License-Identifier: Apache-2.0

"""Resolve pytest scratch directories on the data disk with CI fallback."""

# Standard
import os
from pathlib import Path
import shutil

REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_SCRATCH_ROOT = "DASER_TEST_SCRATCH_ROOT"
PYTEST_DIR_NAME = "pytest"
RESULTS_DIR_NAME = "results"

_DEFAULT_DATA_ROOT = Path("/data")
_FALLBACK_DIR_NAME = ".test-scratch"


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


def resolve_pytest_basetemp() -> Path:
    """Return the pytest ``--basetemp`` directory under the scratch root.

    Returns:
        Absolute path to ``<scratch-root>/pytest``.

    Async/thread-safety:
        Synchronous filesystem helper intended for pytest startup.
    """
    path = resolve_scratch_root() / PYTEST_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def cleanup_ephemeral_scratch() -> None:
    """Delete ephemeral scratch directories while preserving ``results/``.

    Removes ``<scratch-root>/pytest`` and any legacy ``pytest-of-*`` siblings
    that may remain from older pytest defaults. The ``results/`` subtree is
    never deleted by this helper.

    Async/thread-safety:
        Synchronous cleanup intended for pytest session shutdown.
    """
    root = resolve_scratch_root()
    pytest_dir = root / PYTEST_DIR_NAME
    if pytest_dir.exists():
        shutil.rmtree(pytest_dir, ignore_errors=True)

    if not root.exists():
        return

    for entry in root.iterdir():
        if entry.name == RESULTS_DIR_NAME:
            continue
        if entry.is_dir() and entry.name.startswith("pytest-of-"):
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
