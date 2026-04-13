# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import cupy
import pytest
import torch

# First Party
from daser.connector.gds_transfer import GDSTransferLayer, TransferBackend

TEST_DIR = "/data/zwt/daser_test"


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture()
def store_file(tmp_path):
    """Pre-allocate a small store file for testing."""
    path = str(tmp_path / "test.store")
    size = 4 * 1024 * 1024  # 4 MB
    with open(path, "wb") as f:
        f.write(b"\x00" * size)
    return path


def test_backend_is_set(store_file):
    """Backend must be GDS or COMPAT after construction."""
    gds = GDSTransferLayer(store_file)
    assert gds.backend in (TransferBackend.GDS, TransferBackend.COMPAT)
    gds.close()


def test_write_and_read_roundtrip(store_file):
    """Write a GPU tensor, read it back, verify exact equality."""
    nbytes = 1024 * 1024  # 1 MB
    src = torch.randint(0, 256, (nbytes,), dtype=torch.uint8, device="cuda")
    src_cp = cupy.asarray(src)

    gds = GDSTransferLayer(store_file)
    written = _run(gds.write_async(src_cp, file_offset=0))
    assert written == nbytes

    dst = torch.zeros(nbytes, dtype=torch.uint8, device="cuda")
    dst_cp = cupy.asarray(dst)
    read = _run(gds.read_into_async(dst_cp, file_offset=0))
    assert read == nbytes

    assert torch.equal(src, dst)
    gds.close()


def test_multiple_offsets(store_file):
    """Write two non-overlapping regions and read both back correctly."""
    size = 512 * 1024  # 512 KB each
    t1 = torch.ones(size, dtype=torch.uint8, device="cuda")
    t2 = torch.full((size,), 2, dtype=torch.uint8, device="cuda")

    gds = GDSTransferLayer(store_file)
    _run(gds.write_async(cupy.asarray(t1), file_offset=0))
    _run(gds.write_async(cupy.asarray(t2), file_offset=size))

    r1 = torch.zeros(size, dtype=torch.uint8, device="cuda")
    r2 = torch.zeros(size, dtype=torch.uint8, device="cuda")
    _run(gds.read_into_async(cupy.asarray(r1), file_offset=0))
    _run(gds.read_into_async(cupy.asarray(r2), file_offset=size))

    assert torch.equal(t1, r1)
    assert torch.equal(t2, r2)
    gds.close()


def test_context_manager(store_file):
    """Context manager closes file without errors."""
    with GDSTransferLayer(store_file) as gds:
        assert gds.backend in (TransferBackend.GDS, TransferBackend.COMPAT)


def test_missing_file_raises(tmp_path):
    """FileNotFoundError raised when store file does not exist."""
    with pytest.raises(FileNotFoundError):
        GDSTransferLayer(str(tmp_path / "nonexistent.store"))
