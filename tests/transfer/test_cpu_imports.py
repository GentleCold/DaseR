# SPDX-License-Identifier: Apache-2.0

# Standard
import subprocess
import sys
import textwrap


def test_transfer_package_imports_without_cupy() -> None:
    """Verify CPU transfer imports do not require optional CUDA packages.

    Args:
        None.

    Returns:
        None.

    Async/thread-safety:
        Runs a subprocess-only import check and does not share state with the
        test process.
    """
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCupy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cupy" or fullname.startswith("cupy."):
                    raise ModuleNotFoundError("No module named 'cupy'")
                return None

        sys.meta_path.insert(0, BlockCupy())

        import daser.transfer
        from daser.transfer.iouring_pinned import IOUringPinnedTransferLayer
        from daser.server.ipc import IPCServer

        assert daser.transfer.TransferLayer is not None
        assert IOUringPinnedTransferLayer is not None
        assert IPCServer is not None
        """
    )

    subprocess.run([sys.executable, "-c", script], check=True)
