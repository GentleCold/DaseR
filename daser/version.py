# SPDX-License-Identifier: Apache-2.0
"""DaseR package version helpers."""

# Standard
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as metadata_version

_PACKAGE_NAME = "daser"
_SOURCE_VERSION = "0.1.0"


def resolve_version(
    package_name: str = _PACKAGE_NAME,
    metadata_version: Callable[[str], str] = metadata_version,
) -> str:
    """Resolve the DaseR package version.

    Args:
        package_name: installed distribution name to query.
        metadata_version: package metadata lookup function, injectable for tests.

    Returns:
        Installed package version, or the source-tree fallback version when
        package metadata is unavailable.

    Async/thread-safety:
        This helper is synchronous, performs no mutation, and is safe to call
        from asyncio tasks or regular threads.
    """
    try:
        return metadata_version(package_name)
    except PackageNotFoundError:
        return _SOURCE_VERSION


def startup_version_message(version: str) -> str:
    """Build the version text printed during server startup.

    Args:
        version: resolved DaseR package version.

    Returns:
        Human-readable startup version message.

    Async/thread-safety:
        This helper is pure and safe to call from any thread or asyncio task.
    """
    return f"DaseR version={version}"


__version__ = resolve_version()
