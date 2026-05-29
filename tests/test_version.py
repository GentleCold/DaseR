# SPDX-License-Identifier: Apache-2.0

# Standard
from importlib.metadata import PackageNotFoundError
from pathlib import Path

# Third Party
import tomli

# First Party
from daser import version


def _pyproject_version() -> str:
    """Read the package version declared in pyproject.toml.

    Args:
        None.

    Returns:
        Project version string.

    Async/thread-safety:
        This helper performs a read-only filesystem access and is safe to run
        concurrently with other tests.
    """
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomli.loads(pyproject_path.read_text(encoding="utf-8"))
    return str(pyproject["project"]["version"])


def test_resolve_version_falls_back_when_package_metadata_missing() -> None:
    """Source-tree version lookup should work before package installation.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        This test is synchronous and mutates no shared state.
    """

    def missing_metadata(_: str) -> str:
        raise PackageNotFoundError

    assert version.resolve_version(metadata_version=missing_metadata) == (
        _pyproject_version()
    )


def test_startup_version_message_includes_resolved_version() -> None:
    """Startup version text should include the package version.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        This test is synchronous and mutates no shared state.
    """
    assert version.startup_version_message("0.2.0") == "DaseR version=0.2.0"
