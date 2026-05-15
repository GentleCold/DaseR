# SPDX-License-Identifier: Apache-2.0
"""Tests for importing scheduler-side connector helpers without GPU deps."""

# Standard
import subprocess
import sys
import textwrap


def test_scheduler_helpers_import_without_gpu_runtime_dependencies() -> None:
    """Pure connector helpers must import in the CPU-only unit-test job.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        The test runs an isolated Python subprocess and does not share
        mutable state with other tests.
    """
    code = textwrap.dedent(
        """
        import importlib.abc
        import sys


        class BlockedRuntimeDeps(importlib.abc.MetaPathFinder):
            blocked = ("cupy", "kvikio", "torch", "vllm")

            def find_spec(self, fullname, path, target=None):
                if fullname.split(".", 1)[0] in self.blocked:
                    raise ImportError(f"blocked optional dependency: {fullname}")
                return None


        sys.meta_path.insert(0, BlockedRuntimeDeps())

        from daser.connector.helpers import PendingStore, hash_tokens

        assert hash_tokens([1, 2, 3]) == hash_tokens([1, 2, 3])
        PendingStore(hash_tokens([1, 2, 3]), 3)
        """
    )

    subprocess.run([sys.executable, "-c", code], check=True)
