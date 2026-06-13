# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio

# Third Party
import pytest

# First Party
from daser.transfer.iouring import TieredIOUringTransferLayer

ALIGNMENT = 4096


def _run(coro: object) -> object:
    """Run a coroutine on the current test event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _block(byte: bytes, size: int = ALIGNMENT) -> bytearray:
    """Return one payload block filled with ``byte``."""
    return bytearray(byte * size)


def _layer(tmp_path, l1_bytes: int) -> TieredIOUringTransferLayer:
    """Return an iouring transfer layer with its L2 tier disabled."""
    return TieredIOUringTransferLayer(
        path=str(tmp_path / "daser.store"),
        l1_bytes=l1_bytes,
        l2_bytes=l1_bytes,
        skip_l2=True,
    )


def test_iouring_skip_l2_stores_and_loads_without_store_file(tmp_path) -> None:
    """iouring skip_l2 should never create or require daser.store."""
    layer = _layer(tmp_path, ALIGNMENT * 2)
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


def test_iouring_skip_l2_raises_on_l1_miss_after_eviction(tmp_path) -> None:
    """With no L2 fallback, evicted byte ranges are no longer loadable."""
    layer = _layer(tmp_path, ALIGNMENT)
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

        with pytest.raises(KeyError, match="skip_l2 cache miss"):
            _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT))
    finally:
        layer.close()

    assert layer.stats.l1_misses == 1
    assert layer.stats.l2_reads == 0


def test_iouring_skip_l2_grouped_loads_l1_ranges(tmp_path) -> None:
    """Grouped operations should serve all spans from L1 memory."""
    layer = _layer(tmp_path, ALIGNMENT * 2)
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


def test_iouring_grouped_store_uses_single_lock_pass(tmp_path, monkeypatch) -> None:
    """Grouped stores should avoid per-span store_bytes overhead."""
    layer = _layer(tmp_path, ALIGNMENT * 3)
    store_bytes_calls = 0
    lock_entries = 0
    original_store_bytes = layer.store_bytes
    original_lock = layer._lock  # noqa: SLF001

    async def counted_store_bytes(src, file_offset: int, nbytes: int) -> int:
        nonlocal store_bytes_calls
        store_bytes_calls += 1
        return await original_store_bytes(src, file_offset, nbytes)

    class CountedLock:
        def __init__(self, lock):
            self._lock = lock

        async def __aenter__(self):
            nonlocal lock_entries
            lock_entries += 1
            return await self._lock.__aenter__()

        async def __aexit__(self, exc_type, exc, tb):
            return await self._lock.__aexit__(exc_type, exc, tb)

    monkeypatch.setattr(layer, "store_bytes", counted_store_bytes)
    monkeypatch.setattr(layer, "_lock", CountedLock(original_lock))
    try:
        written = _run(
            layer.store_bytes_grouped(
                _block(b"a") + _block(b"b") + _block(b"c"),
                [
                    {"source_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "source_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT,
                    },
                    {
                        "source_offset": ALIGNMENT * 2,
                        "file_offset": ALIGNMENT * 2,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )
        dst = bytearray(ALIGNMENT * 3)
        loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT * 3))
    finally:
        layer.close()

    assert written == ALIGNMENT * 3
    assert loaded == ALIGNMENT * 3
    assert bytes(dst) == bytes(_block(b"a") + _block(b"b") + _block(b"c"))
    assert store_bytes_calls == 0
    assert lock_entries == 2


def test_iouring_skip_l2_overwrite_preserves_adjacent_coalesced_ranges(
    tmp_path,
) -> None:
    """Overwriting part of a coalesced L1 range should keep neighbors loadable."""
    layer = _layer(tmp_path, ALIGNMENT * 4)
    try:
        _run(
            layer.store_bytes(
                _block(b"a") + _block(b"b") + _block(b"c"),
                file_offset=0,
                nbytes=ALIGNMENT * 3,
            )
        )
        _run(
            layer.store_bytes(
                _block(b"x"),
                file_offset=ALIGNMENT,
                nbytes=ALIGNMENT,
            )
        )

        dst = bytearray(ALIGNMENT * 3)
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
                    {
                        "target_offset": ALIGNMENT * 2,
                        "file_offset": ALIGNMENT * 2,
                        "nbytes": ALIGNMENT,
                    },
                ],
            )
        )
    finally:
        layer.close()

    assert loaded == ALIGNMENT * 3
    assert bytes(dst) == bytes(_block(b"a") + _block(b"x") + _block(b"c"))
    assert layer.stats.l1_misses == 0
    assert layer.stats.l2_reads == 0


def test_iouring_skip_l2_partial_tail_overwrite_replaces_full_new_range(
    tmp_path,
) -> None:
    """A write that extends past an existing L1 entry should replace the range."""
    layer = _layer(tmp_path, ALIGNMENT * 4)
    try:
        _run(
            layer.store_bytes(
                _block(b"a") + _block(b"b"),
                file_offset=0,
                nbytes=ALIGNMENT * 2,
            )
        )
        _run(
            layer.store_bytes(
                _block(b"x") + _block(b"y"),
                file_offset=ALIGNMENT,
                nbytes=ALIGNMENT * 2,
            )
        )

        dst = bytearray(ALIGNMENT * 3)
        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "target_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT * 2,
                    },
                ],
            )
        )
    finally:
        layer.close()

    assert loaded == ALIGNMENT * 3
    assert bytes(dst) == bytes(_block(b"a") + _block(b"x") + _block(b"y"))
    assert layer.stats.l1_misses == 0
    assert layer.stats.l2_reads == 0


def test_iouring_skip_l2_grouped_partial_tail_overwrite_replaces_full_new_range(
    tmp_path,
) -> None:
    """Grouped writes should preserve old L1 fragments around overwrite tails."""
    layer = _layer(tmp_path, ALIGNMENT * 4)
    try:
        _run(
            layer.store_bytes(
                _block(b"a") + _block(b"b"),
                file_offset=0,
                nbytes=ALIGNMENT * 2,
            )
        )
        _run(
            layer.store_bytes_grouped(
                _block(b"x") + _block(b"y"),
                [
                    {
                        "source_offset": 0,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT * 2,
                    }
                ],
            )
        )

        dst = bytearray(ALIGNMENT * 3)
        loaded = _run(
            layer.load_bytes_grouped(
                dst,
                [
                    {"target_offset": 0, "file_offset": 0, "nbytes": ALIGNMENT},
                    {
                        "target_offset": ALIGNMENT,
                        "file_offset": ALIGNMENT,
                        "nbytes": ALIGNMENT * 2,
                    },
                ],
            )
        )
    finally:
        layer.close()

    assert loaded == ALIGNMENT * 3
    assert bytes(dst) == bytes(_block(b"a") + _block(b"x") + _block(b"y"))
    assert layer.stats.l1_misses == 0
    assert layer.stats.l2_reads == 0


def test_iouring_skip_l2_overwrite_evicts_preserved_fragments_when_l1_is_full(
    tmp_path,
) -> None:
    """Preserved fragments are best-effort when skip_l2 L1 has no spare room."""
    layer = _layer(tmp_path, ALIGNMENT * 2)
    try:
        _run(
            layer.store_bytes(
                _block(b"a") + _block(b"b"),
                file_offset=0,
                nbytes=ALIGNMENT * 2,
            )
        )
        _run(
            layer.store_bytes(
                _block(b"x") + _block(b"y"),
                file_offset=ALIGNMENT,
                nbytes=ALIGNMENT * 2,
            )
        )

        dst = bytearray(ALIGNMENT * 2)
        loaded = _run(
            layer.load_bytes(
                dst,
                file_offset=ALIGNMENT,
                nbytes=ALIGNMENT * 2,
            )
        )
        with pytest.raises(KeyError, match="skip_l2 cache miss"):
            _run(
                layer.load_bytes(
                    bytearray(ALIGNMENT),
                    file_offset=0,
                    nbytes=ALIGNMENT,
                )
            )
    finally:
        layer.close()

    assert loaded == ALIGNMENT * 2
    assert bytes(dst) == bytes(_block(b"x") + _block(b"y"))
    assert layer.stats.l2_reads == 0


def test_iouring_skip_l2_loads_across_adjacent_l1_ranges(tmp_path) -> None:
    """A logical load range may span multiple adjacent resident L1 entries."""
    layer = _layer(tmp_path, ALIGNMENT * 4)
    try:
        _run(
            layer.store_bytes(
                _block(b"a") + _block(b"b"),
                file_offset=0,
                nbytes=ALIGNMENT * 2,
            )
        )
        _run(
            layer.store_bytes(
                _block(b"c") + _block(b"d"),
                file_offset=ALIGNMENT * 2,
                nbytes=ALIGNMENT * 2,
            )
        )

        dst = bytearray(ALIGNMENT * 4)
        loaded = _run(layer.load_bytes(dst, file_offset=0, nbytes=ALIGNMENT * 4))
    finally:
        layer.close()

    assert loaded == ALIGNMENT * 4
    assert bytes(dst) == bytes(
        _block(b"a") + _block(b"b") + _block(b"c") + _block(b"d")
    )
    assert layer.stats.l1_hits == 1
    assert layer.stats.l1_misses == 0
    assert layer.stats.l2_reads == 0


def test_iouring_skip_l2_rejects_ranges_larger_than_l1(tmp_path) -> None:
    """A single store span must fit in the configured L1 capacity."""
    layer = _layer(tmp_path, ALIGNMENT)
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
