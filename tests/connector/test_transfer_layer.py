# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

pytest.importorskip("cupy")
pytest.importorskip("kvikio")
pytest.importorskip("torch")

# First Party
from daser.connector.transfer import (
    GDSTransferLayer,
    IOUringMemTransferLayer,
    TransferBackendName,
    TransferCallbacks,
    TransferConfig,
    TransferStats,
    build_transfer_layer,
)

pytestmark = pytest.mark.integration


def test_gds_transfer_exposes_transfer_layer_interface(tmp_path) -> None:
    """GDSTransferLayer exposes the common TransferLayer surface."""
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 4096)

    transfer = GDSTransferLayer(str(store_path))
    try:
        assert transfer.backend_name == TransferBackendName.GDS
        assert isinstance(transfer.stats(), TransferStats)
    finally:
        transfer.close()


def test_factory_builds_gds_transfer_with_commit_callback(tmp_path) -> None:
    """Transfer factory wires GDS publication without connector branching."""
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 4096)
    commits: list[str] = []

    transfer = build_transfer_layer(
        TransferConfig(
            backend_name=TransferBackendName.GDS,
            store_path=str(store_path),
        ),
        TransferCallbacks(
            commit_chunk=lambda key: commits.append(key),
            commit_l1=lambda key: None,
            commit_l2=lambda key: None,
            evict_l1=lambda key: None,
        ),
    )

    try:
        assert isinstance(transfer, GDSTransferLayer)
        assert transfer.backend_name == TransferBackendName.GDS
    finally:
        transfer.close()


def test_factory_builds_iouring_mem_transfer_with_l1_callbacks(tmp_path) -> None:
    """Transfer factory wires iouring-mem L1/L2 callbacks."""
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 4096)

    transfer = build_transfer_layer(
        TransferConfig(
            backend_name=TransferBackendName.IOURING_MEM,
            store_path=str(store_path),
            l1_cache_size=4096,
        ),
        TransferCallbacks(
            commit_chunk=lambda key: None,
            commit_l1=lambda key: None,
            commit_l2=lambda key: None,
            evict_l1=lambda key: None,
        ),
    )

    try:
        assert isinstance(transfer, IOUringMemTransferLayer)
        assert transfer.backend_name == TransferBackendName.IOURING_MEM
    finally:
        transfer.close()


def test_factory_rejects_iouring_mem_without_l1(tmp_path) -> None:
    """iouring-mem must not construct an uncached transfer path."""
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 4096)

    with pytest.raises(ValueError, match="positive l1_cache_size"):
        build_transfer_layer(
            TransferConfig(
                backend_name=TransferBackendName.IOURING_MEM,
                store_path=str(store_path),
            ),
            TransferCallbacks(
                commit_chunk=lambda key: None,
                commit_l1=lambda key: None,
                commit_l2=lambda key: None,
                evict_l1=lambda key: None,
            ),
        )
