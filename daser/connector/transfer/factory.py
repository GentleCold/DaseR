# SPDX-License-Identifier: Apache-2.0

"""Factory for constructing connector transfer layers."""

# First Party
from daser.connector.transfer.base import (
    TransferBackendName,
    TransferCallbacks,
    TransferConfig,
    TransferLayer,
)
from daser.connector.transfer.gds import GDSTransferLayer
from daser.connector.transfer.iouring import IOUringMemTransferLayer


def build_transfer_layer(
    config: TransferConfig,
    callbacks: TransferCallbacks,
) -> TransferLayer:
    """Build the configured transfer implementation.

    Args:
        config: immutable transfer configuration from the server runtime config.
        callbacks: server publication callbacks invoked by the transfer layer.

    Returns:
        Initialized transfer layer.

    Raises:
        ValueError: if the backend is unsupported or the backend-specific
            configuration is invalid.

    Async/thread-safety:
        Called on the worker thread during transfer initialization before hot
        path IO starts. The returned backend is immutable after construction.
    """
    if config.backend_name == TransferBackendName.GDS:
        transfer = GDSTransferLayer(config.store_path)
        transfer.set_commit_callback(callbacks.commit_chunk)
        return transfer

    if config.backend_name == TransferBackendName.IOURING_MEM:
        if config.l1_cache_size <= 0:
            raise ValueError("iouring-mem requires a positive l1_cache_size")
        return IOUringMemTransferLayer(
            path=config.store_path,
            l1_cache_size=config.l1_cache_size,
            commit_l1=callbacks.commit_l1,
            commit_l2=callbacks.commit_l2,
            evict_l1=callbacks.evict_l1,
        )

    raise ValueError(f"unsupported transfer backend: {config.backend_name}")
