# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

# First Party
from daser.transfer.memory import L1OnlyTransferLayer

ALIGNMENT = 4096


def _run(coro: object) -> object:
    """Run a coroutine on the current test event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _block(byte: bytes, size: int = ALIGNMENT) -> bytearray:
    """Return one payload block filled with ``byte``."""
    return bytearray(byte * size)


def test_l1_only_transfer_stores_and_loads_without_store_file(tmp_path) -> None:
    """Memory-only transfer should never create or require daser.store."""
    layer = L1OnlyTransferLayer(l1_bytes=ALIGNMENT * 2)
    store_path = tmp_path / "daser.store"
    dst = bytearray(ALIGNMENT)

    try:
        written = _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT))
    finally:
        layer.close()

    assert written == ALIGNMENT
    assert loaded == ALIGNMENT
    assert bytes(dst) == bytes(_block(b"a"))
    assert not store_path.exists()
    assert layer.stats.l1_hits == 1
    assert layer.stats.l1_misses == 0
    assert layer.stats.l2_reads == 0
    assert layer.stats.l2_writes == 0


def test_l1_only_transfer_raises_on_l1_miss_after_eviction() -> None:
    """With no L2 fallback, evicted byte ranges are no longer loadable."""
    layer = L1OnlyTransferLayer(l1_bytes=ALIGNMENT)
    try:
        _run(layer.store_bytes(_block(b"a"), file_offset=0, nbytes=ALIGNMENT))
        _run(
            layer.store_bytes(
                _block(b"b"),
                file_offset=ALIGNMENT,
                nbytes=ALIGNMENT,
            )
        )
        dst = bytearray(ALIGNMENT)

        with pytest.raises(KeyError, match="L1-only cache miss"):
            _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT))
    finally:
        layer.close()

    assert layer.stats.l1_misses == 1
    assert layer.stats.l2_reads == 0


def test_l1_only_transfer_grouped_loads_l1_ranges() -> None:
    """Grouped operations should serve all spans from L1 memory."""
    layer = L1OnlyTransferLayer(l1_bytes=ALIGNMENT * 2)
    try:
        _run(
            layer.store_bytes_grouped(
                _block(b"a") + _block(b"b"),
                [
                    {"source_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "source_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )
        dst = bytearray(ALIGNMENT * 2)

        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "target_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )
    finally:
        layer.close()

    assert loaded == ALIGNMENT * 2
    assert bytes(dst) == bytes(_block(b"a") + _block(b"b"))
    assert layer.stats.l1_hits == 2
    assert layer.stats.l2_reads == 0
    assert layer.stats.l2_writes == 0


def test_l1_only_transfer_rejects_ranges_larger_than_l1() -> None:
    """A single store span must fit in the configured L1 capacity."""
    layer = L1OnlyTransferLayer(l1_bytes=ALIGNMENT)
    try:
        with pytest.raises(ValueError, match="exceeds L1 capacity"):
            _run(
                layer.store_bytes(
                    _block(b"a") + _block(b"b"),
                    file_offset=0,
                    nbytes=ALIGNMENT * 2,
                )
            )
    finally:
        layer.close()
