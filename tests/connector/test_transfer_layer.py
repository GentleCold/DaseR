# SPDX-License-Identifier: Apache-2.0

# Third Party
import pytest

# First Party
from daser.connector.gds_transfer import GDSTransferLayer
from daser.connector.transfer import TransferBackendName, TransferStats

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
