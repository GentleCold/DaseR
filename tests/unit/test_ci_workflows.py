# SPDX-License-Identifier: Apache-2.0
"""Tests for GitHub Actions workflow guardrails."""

# Standard
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_workflow(name: str) -> str:
    """Read a workflow file from ``.github/workflows``.

    Args:
        name: Workflow file name.

    Returns:
        Workflow file contents.

    Async/thread-safety:
        This helper performs a short synchronous read in tests only and
        maintains no shared state.
    """
    return (REPO_ROOT / ".github" / "workflows" / name).read_text()


def test_commit_lint_allows_perf_and_revert_types() -> None:
    """Commit lint must accept all documented commit message types.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        This test performs read-only filesystem access and is safe to run
        concurrently with other tests.
    """
    workflow = _read_workflow("commit-lint.yml")

    assert "feat|fix|perf|refactor|revert|chore|test|docs" in workflow
    assert "feat | fix | perf | refactor | revert | chore | test | docs" in workflow


def test_ci_runs_all_non_integration_tests() -> None:
    """CI unit tests must cover every test outside integration scope.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        This test performs read-only filesystem access and is safe to run
        concurrently with other tests.
    """
    workflow = _read_workflow("ci.yml")

    assert 'PYTHONHASHSEED: "0"' in workflow
    assert "pytest -q" in workflow
    assert '-m "not integration"' in workflow
    assert "--ignore=tests/integration" in workflow
    assert "--ignore=tests/connector/test_daser_connector.py" in workflow
    assert "--ignore=tests/connector/test_gds_transfer.py" in workflow
    assert "tests/" in workflow
    old_explicit_targets = [
        "          tests/test_config.py",
        "          tests/server/",
        "          tests/retrieval/",
        "          tests/position/",
    ]
    for target in old_explicit_targets:
        assert target not in workflow


def test_hardware_connector_tests_are_marked_integration() -> None:
    """GPU/GDS connector tests must not run in the CPU unit-test job.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        This test performs read-only filesystem access and is safe to run
        concurrently with other tests.
    """
    hardware_test_paths = [
        REPO_ROOT / "tests" / "connector" / "test_daser_connector.py",
        REPO_ROOT / "tests" / "connector" / "test_gds_transfer.py",
    ]

    for path in hardware_test_paths:
        assert "pytestmark = pytest.mark.integration" in path.read_text()
