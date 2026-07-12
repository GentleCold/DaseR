# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import cupy
import pytest
import torch

# First Party
from daser.transfer.gds import GDSTransferLayer, TransferBackend

pytestmark = pytest.mark.integration


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


def test_fixed_staging_pool_reuses_two_preallocated_buffers() -> None:
    """Fixed staging pool reuses bounded preallocated buffers."""
    # First Party
    from daser.connector.worker.memory import FixedCudaStagingPool

    pool = FixedCudaStagingPool(
        device=torch.device("cpu"),
        buffer_bytes=16,
        depth=2,
    )

    first = pool.acquire(8)
    second = pool.acquire(16)

    with pytest.raises(RuntimeError, match="no fixed staging buffers available"):
        pool.acquire(1)

    first.release()
    third = pool.acquire(4)

    assert third.tensor.data_ptr() == first.tensor.data_ptr()
    second.release()
    third.release()


def test_fixed_staging_pool_can_block_until_buffer_release() -> None:
    """Fixed staging pool can wait for a callback to release capacity."""
    # First Party
    from daser.connector.worker.memory import FixedCudaStagingPool

    pool = FixedCudaStagingPool(
        device=torch.device("cpu"),
        buffer_bytes=16,
        depth=1,
    )
    first = pool.acquire(8)
    waits = 0

    def release_first() -> None:
        nonlocal waits
        waits += 1
        first.release()

    second = pool.acquire(4, wait_for_release=release_first)

    assert waits == 1
    assert second.tensor.data_ptr() == first.tensor.data_ptr()
    second.release()


def test_fixed_staging_pool_rejects_oversized_request() -> None:
    """Fixed staging pool rejects requests larger than one buffer."""
    # First Party
    from daser.connector.worker.memory import FixedCudaStagingPool

    pool = FixedCudaStagingPool(
        device=torch.device("cpu"),
        buffer_bytes=16,
        depth=2,
    )

    with pytest.raises(ValueError, match="exceeds fixed staging buffer"):
        pool.acquire(17)
