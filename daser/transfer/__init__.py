# SPDX-License-Identifier: Apache-2.0

from daser.transfer.base import TransferLayer, TransferMode, TransferStats
from daser.transfer.iouring import TieredIOUringTransferLayer
from daser.transfer.memory import L1OnlyTransferLayer

__all__ = [
    "GDSTransferLayer",
    "L1OnlyTransferLayer",
    "TieredIOUringTransferLayer",
    "TransferBackend",
    "TransferLayer",
    "TransferMode",
    "TransferStats",
]


def __getattr__(name: str) -> object:
    """Lazily import CUDA-backed transfer symbols.

    Args:
        name: Attribute requested from ``daser.transfer``.

    Returns:
        The requested transfer symbol.

    Async/thread-safety:
        This function is synchronous and import-time only. It avoids importing
        optional CUDA dependencies in CPU-only test and CI environments.

    Raises:
        AttributeError: If ``name`` is not exported by this package.
    """
    if name == "GDSTransferLayer":
        from daser.transfer.gds import GDSTransferLayer

        return GDSTransferLayer
    if name == "TransferBackend":
        from daser.transfer.gds import TransferBackend

        return TransferBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
