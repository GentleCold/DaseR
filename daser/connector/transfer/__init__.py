# SPDX-License-Identifier: Apache-2.0

"""Data-plane transfer backends for DaseR connectors."""

from daser.connector.transfer.base import (
    BaseTransferLayer,
    TransferBackendName,
    TransferCallbacks,
    TransferConfig,
    TransferLayer,
    TransferStats,
)
from daser.connector.transfer.factory import build_transfer_layer
from daser.connector.transfer.gds import GDSTransferLayer, KvikIOTransferBackend
from daser.connector.transfer.iouring import (
    FileIOEngine,
    IOUringMemTransferLayer,
    NativeIOUring,
    NativeIOUringEngine,
    NativeIOUringError,
    PinnedL1Cache,
    PreadPwriteTestEngine,
)

__all__ = [
    "BaseTransferLayer",
    "FileIOEngine",
    "GDSTransferLayer",
    "IOUringMemTransferLayer",
    "KvikIOTransferBackend",
    "NativeIOUring",
    "NativeIOUringEngine",
    "NativeIOUringError",
    "PinnedL1Cache",
    "PreadPwriteTestEngine",
    "TransferBackendName",
    "TransferCallbacks",
    "TransferConfig",
    "TransferLayer",
    "TransferStats",
    "build_transfer_layer",
]
