# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest
import torch

# First Party
from daser.connector.transfer import (
    IOUringMemTransferLayer,
    PreadPwriteTestEngine,
    TransferBackendName,
)


def _run(coro):
    """Run a coroutine on the current test event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _cpu_allocator(nbytes: int) -> torch.Tensor:
    """Allocate a CPU uint8 tensor for transfer tests."""
    return torch.empty(nbytes, dtype=torch.uint8)


def _test_engine(path) -> PreadPwriteTestEngine:
    """Return the test-only file IO engine."""
    return PreadPwriteTestEngine(str(path))


def test_write_publishes_l1_before_l2_commit(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 64)
    commits_l1: list[str] = []
    commits_l2: list[str] = []
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=64,
        allocator=_cpu_allocator,
        io_engine=_test_engine(store_path),
        commit_l1=lambda key: commits_l1.append(key),
        commit_l2=lambda key: commits_l2.append(key),
    )
    src = torch.arange(16, dtype=torch.uint8)

    written = _run(
        transfer.write_chunk_async(
            chunk_key="chunk-a",
            buf=src,
            file_offset=0,
            nbytes=16,
        )
    )

    assert written == 16
    assert commits_l1 == ["chunk-a"]
    assert commits_l2 == ["chunk-a"]
    assert store_path.read_bytes()[:16] == bytes(range(16))
    assert transfer.backend_name == TransferBackendName.IOURING_MEM


def test_read_hits_l1_without_touching_l2(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(bytes([9]) * 64)
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=64,
        allocator=_cpu_allocator,
        io_engine=_test_engine(store_path),
    )
    src = torch.arange(16, dtype=torch.uint8)
    _run(transfer.write_chunk_async("chunk-a", src, file_offset=0, nbytes=16))

    store_path.write_bytes(bytes([3]) * 64)
    dst = torch.empty(16, dtype=torch.uint8)
    read = _run(
        transfer.read_chunk_into_async(
            chunk_key="chunk-a",
            buf=dst,
            file_offset=0,
            nbytes=16,
            l2_durable=True,
        )
    )

    assert read == 16
    assert dst.tolist() == list(range(16))
    assert transfer.stats().l1_hits == 1


def test_lookup_protected_l1_hit_cannot_be_evicted_until_release(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 64)
    evicted: list[str] = []
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=32,
        allocator=_cpu_allocator,
        io_engine=_test_engine(store_path),
        evict_l1=lambda key: evicted.append(key),
    )
    src = torch.arange(16, dtype=torch.uint8)
    large_src = torch.arange(32, dtype=torch.uint8)
    _run(transfer.write_chunk_async("chunk-a", src, file_offset=0, nbytes=16))

    dst = torch.empty(16, dtype=torch.uint8)
    _run(
        transfer.read_chunk_into_async(
            chunk_key="chunk-a",
            buf=dst,
            file_offset=0,
            nbytes=16,
            l2_durable=True,
            protect_lookup=True,
        )
    )

    with pytest.raises(MemoryError, match="no evictable"):
        _run(
            transfer.write_chunk_async("chunk-b", large_src, file_offset=16, nbytes=32)
        )

    transfer.release_lookup_pins(["chunk-a"])
    _run(transfer.write_chunk_async("chunk-b", large_src, file_offset=16, nbytes=32))

    assert dst.tolist() == list(range(16))
    assert evicted == ["chunk-a"]


def test_host_read_holds_pin_until_release(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 64)
    evicted: list[str] = []
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=32,
        allocator=_cpu_allocator,
        io_engine=_test_engine(store_path),
        evict_l1=lambda key: evicted.append(key),
    )
    src = torch.arange(16, dtype=torch.uint8)
    large_src = torch.arange(32, dtype=torch.uint8)
    _run(transfer.write_chunk_async("chunk-a", src, file_offset=0, nbytes=16))

    host = _run(
        transfer.read_chunk_host_async(
            chunk_key="chunk-a",
            file_offset=0,
            nbytes=16,
            l2_durable=True,
        )
    )

    with pytest.raises(MemoryError, match="no evictable"):
        _run(
            transfer.write_chunk_async("chunk-b", large_src, file_offset=16, nbytes=32)
        )

    transfer.release_chunk_host("chunk-a")
    _run(transfer.write_chunk_async("chunk-b", large_src, file_offset=16, nbytes=32))

    assert host.tolist() == list(range(16))
    assert evicted == ["chunk-a"]


def test_lookup_protected_l2_fill_cannot_be_evicted_until_release(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(bytes(range(64)))
    evicted: list[str] = []
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=32,
        allocator=_cpu_allocator,
        io_engine=_test_engine(store_path),
        evict_l1=lambda key: evicted.append(key),
    )
    dst = torch.empty(16, dtype=torch.uint8)

    read = _run(
        transfer.read_chunk_into_async(
            chunk_key="chunk-a",
            buf=dst,
            file_offset=8,
            nbytes=16,
            l2_durable=True,
            protect_lookup=True,
        )
    )

    with pytest.raises(MemoryError, match="no evictable"):
        _run(
            transfer.write_chunk_async(
                "chunk-b", torch.arange(32, dtype=torch.uint8), 16, 32
            )
        )

    transfer.release_lookup_pins(["chunk-a"])
    _run(
        transfer.write_chunk_async(
            "chunk-b", torch.arange(32, dtype=torch.uint8), 16, 32
        )
    )

    assert read == 16
    assert dst.tolist() == list(range(8, 24))
    assert evicted == ["chunk-a"]


def test_read_miss_loads_from_l2_and_fills_l1(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(bytes(range(64)))
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=64,
        allocator=_cpu_allocator,
        io_engine=_test_engine(store_path),
    )
    dst = torch.empty(16, dtype=torch.uint8)

    read = _run(
        transfer.read_chunk_into_async(
            chunk_key="chunk-a",
            buf=dst,
            file_offset=8,
            nbytes=16,
            l2_durable=True,
        )
    )

    assert read == 16
    assert dst.tolist() == list(range(8, 24))
    dst2 = torch.empty(16, dtype=torch.uint8)
    _run(transfer.read_chunk_into_async("chunk-a", dst2, 8, 16, True))
    assert dst2.tolist() == list(range(8, 24))
    assert transfer.stats().l1_misses == 1
    assert transfer.stats().l1_hits == 1


def test_l1_miss_without_l2_durable_raises(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 64)
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=64,
        allocator=_cpu_allocator,
        io_engine=_test_engine(store_path),
    )
    dst = torch.empty(16, dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="not durable"):
        _run(transfer.read_chunk_into_async("chunk-a", dst, 0, 16, False))


def test_default_engine_uses_native_iouring(tmp_path) -> None:
    store_path = tmp_path / "test.store"
    store_path.write_bytes(b"\0" * 64)
    transfer = IOUringMemTransferLayer(
        path=str(store_path),
        l1_cache_size=64,
        allocator=_cpu_allocator,
    )
    src = torch.arange(16, dtype=torch.uint8)
    dst = torch.empty(16, dtype=torch.uint8)

    try:
        assert _run(transfer.write_async(src, 0, 16)) == 16
        assert _run(transfer.read_into_async(dst, 0, 16)) == 16
    finally:
        transfer.close()

    assert dst.tolist() == list(range(16))
