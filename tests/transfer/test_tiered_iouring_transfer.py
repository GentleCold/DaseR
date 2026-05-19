# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# First Party
from daser.transfer.iouring_pinned import IOUringPinnedTransferLayer


def _run(coro: object) -> object:
    """Run a coroutine on the current test event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def test_iouring_pinned_load_hits_l1_before_l2(tmp_path) -> None:
    """Stored data is readable from L1 immediately before L2 persistence."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=16,
        l2_bytes=64,
    )
    src = bytearray(b"abcdefgh")
    dst = bytearray(8)

    written = _run(layer.store_bytes(src, file_offset=0, nbytes=8))
    loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=8))

    assert written == 8
    assert loaded == 8
    assert bytes(dst) == b"abcdefgh"
    assert layer.stats.l1_hits == 1
    assert layer.stats.l2_reads == 0
    layer.close()


def test_iouring_pinned_promotes_l2_miss_to_l1(tmp_path) -> None:
    """L1 eviction falls back to L2 and promotes the bytes back into L1."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=8,
        l2_bytes=64,
    )

    _run(layer.store_bytes(bytearray(b"aaaaaaaa"), file_offset=0, nbytes=8))
    _run(layer.store_bytes(bytearray(b"bbbbbbbb"), file_offset=8, nbytes=8))

    dst = bytearray(8)
    loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=8))

    assert loaded == 8
    assert bytes(dst) == b"aaaaaaaa"
    assert layer.stats.l1_misses == 1
    assert layer.stats.l2_reads == 1
    layer.close()


def test_iouring_pinned_parallel_l2_loads_use_independent_offsets(tmp_path) -> None:
    """Concurrent L2 loads read their requested byte ranges exactly."""

    async def scenario() -> None:
        block_size = 256 * 1024
        block_count = 128
        path = str(tmp_path / "daser.store")
        layer = IOUringPinnedTransferLayer(
            path=path,
            l1_bytes=block_size,
            l2_bytes=block_size * block_count,
        )
        try:
            for i in range(block_count):
                await layer.store_bytes(
                    bytearray([i]) * block_size,
                    file_offset=i * block_size,
                    nbytes=block_size,
                )
        finally:
            layer.close()

        layer = IOUringPinnedTransferLayer(
            path=path,
            l1_bytes=block_size,
            l2_bytes=block_size * block_count,
        )
        try:

            async def load_block(i: int) -> tuple[int, bytes]:
                dst = bytearray(block_size)
                await layer.load_bytes(
                    dst,
                    file_offset=i * block_size,
                    nbytes=block_size,
                )
                return i, bytes(dst)

            results = await asyncio.gather(*(load_block(i) for i in range(block_count)))
        finally:
            layer.close()

        for i, data in results:
            assert data == bytes([i]) * block_size

    _run(scenario())


def test_iouring_pinned_rejects_l2_overflow(tmp_path) -> None:
    """Writes beyond the configured L2 capacity are rejected."""
    layer = IOUringPinnedTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=8,
        l2_bytes=8,
    )

    try:
        _run(layer.store_bytes(bytearray(b"xx"), file_offset=7, nbytes=2))
    except ValueError as exc:
        assert "exceeds L2 capacity" in str(exc)
    else:
        raise AssertionError("expected ValueError")
    finally:
        layer.close()
