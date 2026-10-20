# SPDX-License-Identifier: Apache-2.0
"""Public batch-read contracts against actual kernel completions."""

import mmap
import os
from pathlib import Path

import pytest

from daser.transfer.iouring.native import NativeIOUring


def test_sparse_batch_callbacks_and_ring_reuse(tmp_path: Path) -> None:
    """Independent sources map to their own targets across ring wraparound."""
    path = tmp_path / "batch.store"
    payload = bytes(range(256)) * 32
    path.write_bytes(payload)
    fd = os.open(path, os.O_RDONLY)
    ring = NativeIOUring(entries=8)
    try:
        for _ in range(20):
            buffers = [bytearray(97) for _ in range(8)]
            offsets = [4001, 19, 987, 43, 7000, 0, 2048, 300]
            observed: dict[int, bytes] = {}

            def completed(
                index: int,
                count: int,
                observed: dict[int, bytes] = observed,
                buffers: list[bytearray] = buffers,
            ) -> None:
                assert count == 97
                observed[index] = bytes(buffers[index])

            assert (
                ring.read_batch_into(
                    fd,
                    list(zip(offsets, map(memoryview, buffers), strict=True)),
                    completed,
                )
                == 8 * 97
            )
            assert observed == {
                i: payload[offset : offset + 97] for i, offset in enumerate(offsets)
            }
        assert ring.read_batch_into(fd, []) == 0
    finally:
        ring.close()
        os.close(fd)


@pytest.mark.parametrize("failure", ["short", "callback", "bad_fd"])
def test_batch_drains_before_error_and_can_be_reused(
    tmp_path: Path, failure: str
) -> None:
    """A failed read or callback must not leave CQEs for the next caller."""
    path = tmp_path / "error.store"
    path.write_bytes(b"a" * 8192)
    fd = os.open(path, os.O_RDONLY)
    ring = NativeIOUring(entries=8)
    buffers = [bytearray(512) for _ in range(8)]

    def completed(index: int, count: int) -> None:
        if failure == "callback":
            raise RuntimeError("callback rejected")

    try:
        reads = [
            (8192 if failure == "short" and i == 0 else i * 512, memoryview(dst))
            for i, dst in enumerate(buffers)
        ]
        with pytest.raises(RuntimeError if failure == "callback" else OSError):
            ring.read_batch_into(-1 if failure == "bad_fd" else fd, reads, completed)
        dst = bytearray(8192)
        assert ring.read_into(fd, 0, memoryview(dst)) == 8192
        assert dst == b"a" * 8192
        if failure != "bad_fd":
            assert all(dst == b"a" * 512 for dst in buffers[1:])
    finally:
        ring.close()
        os.close(fd)


def test_batch_validation_precedes_any_io(tmp_path: Path) -> None:
    """Reject invalid buffers and oversized batches before modifying targets."""
    path = tmp_path / "validate.store"
    path.write_bytes(b"x" * 128)
    fd = os.open(path, os.O_RDONLY)
    ring = NativeIOUring(entries=2)
    dst = bytearray(16)
    try:
        for reads in (
            [(0, memoryview(dst)), (1, memoryview(b"readonly"))],
            [(0, memoryview(dst))] * 3,
            [(0, memoryview(dst)), (-1, memoryview(dst))],
        ):
            with pytest.raises(ValueError):
                ring.read_batch_into(fd, reads)
            assert dst == bytes(16)
        assert ring.read_into(fd, 0, memoryview(dst)) == 16
        assert dst == b"x" * 16
    finally:
        ring.close()
        os.close(fd)


def test_error_releases_internal_buffer_exports(tmp_path: Path) -> None:
    """A drained callback error cannot retain mmap exports until cyclic GC."""
    path = tmp_path / "exports.store"
    path.write_bytes(b"a" * 4096)
    fd = os.open(path, os.O_RDONLY)
    buf = mmap.mmap(-1, 4096)
    view = memoryview(buf)
    ring = NativeIOUring()
    saved: list[BaseException] = []

    def completed(index: int, nbytes: int) -> None:
        raise RuntimeError("callback rejected")

    try:
        try:
            ring.read_batch_into(fd, [(0, view)], completed)
        except RuntimeError as error:
            saved.append(error)
        assert len(saved) == 1
        assert buf[:] == b"a" * 4096
        view.release()
        # Keep the exception and its traceback alive deliberately. Only the
        # caller's view was released, so this fails if native views leak.
        buf.close()
    finally:
        ring.close()
        view.release()
        buf.close()
        os.close(fd)


@pytest.mark.parametrize(
    ("operation", "failure"),
    [
        ("read", "io"),
        ("read", "readonly"),
        ("write", "io"),
        ("write", "readonly"),
        ("read_batch", "io"),
        ("read_batch", "validation"),
        ("read_batch", "readonly"),
        ("write_batch", "io"),
        ("write_batch", "validation"),
        ("write_batch", "readonly"),
    ],
)
def test_failure_releases_exports_while_traceback_is_retained(
    operation: str, failure: str
) -> None:
    """Single IO errors and rejected batches cannot keep mappings exported."""
    buf = mmap.mmap(-1, 4096)
    view = memoryview(buf)
    readonly = view.toreadonly()
    ring = NativeIOUring()
    errors: list[BaseException] = []
    try:
        try:
            if operation == "read":
                ring.read_into(-1, 0, readonly if failure == "readonly" else view)
            elif operation == "write":
                ring.write(-1, 0, readonly if failure == "readonly" else view)
            else:
                extents = [(0, view)]
                if failure == "validation":
                    extents.append((-1, view))
                elif failure == "readonly":
                    extents.append((0, readonly))
                if operation == "read_batch":
                    ring.read_batch_into(-1, extents)
                else:
                    ring.write_batch_from(-1, extents)
        except (OSError, ValueError) as error:
            errors.append(error)
        assert len(errors) == 1
        view.release()
        readonly.release()
        # The exception, traceback and original argument containers remain
        # alive here; releasing the caller's views alone must be sufficient.
        buf.close()
    finally:
        for error in errors:
            error.__traceback__ = None
        ring.close()
        view.release()
        readonly.release()
        buf.close()


def test_sparse_write_batch_persists_extents_across_ring_reuse(
    tmp_path: Path,
) -> None:
    """Each source lands at its own offset; the ring is reusable after."""
    path = tmp_path / "write.store"
    path.write_bytes(bytes(8192))
    fd = os.open(path, os.O_RDWR)
    ring = NativeIOUring(entries=8)
    offsets = [4001, 19, 987, 43, 7000, 0, 2048, 300]
    try:
        for round_index in range(20):
            sources = [bytearray([round_index + i]) * 13 for i in range(8)]
            assert (
                ring.write_batch_from(
                    fd, list(zip(offsets, map(memoryview, sources), strict=True))
                )
                == 8 * 13
            )
            for offset, source in zip(offsets, sources, strict=True):
                assert os.pread(fd, 13, offset) == source
        assert ring.write_batch_from(fd, []) == 0
    finally:
        ring.close()
        os.close(fd)


def test_write_batch_drains_before_error_and_can_be_reused(tmp_path: Path) -> None:
    """A failed write batch leaves no CQEs for the next caller."""
    path = tmp_path / "write_error.store"
    path.write_bytes(b"a" * 4096)
    read_only = os.open(path, os.O_RDONLY)
    ring = NativeIOUring(entries=8)
    sources = [bytearray(b"b" * 512) for _ in range(8)]
    try:
        with pytest.raises(OSError):
            ring.write_batch_from(
                read_only,
                [(i * 512, memoryview(src)) for i, src in enumerate(sources)],
            )
        assert path.read_bytes() == b"a" * 4096
        dst = bytearray(4096)
        assert ring.read_into(read_only, 0, memoryview(dst)) == 4096
        assert dst == b"a" * 4096
    finally:
        ring.close()
        os.close(read_only)


def test_write_batch_validation_precedes_any_io(tmp_path: Path) -> None:
    """Reject invalid sources and oversized batches before writing anything."""
    path = tmp_path / "write_validate.store"
    path.write_bytes(b"x" * 128)
    fd = os.open(path, os.O_RDWR)
    ring = NativeIOUring(entries=2)
    src = bytearray(b"y" * 16)
    try:
        for writes in (
            [(0, memoryview(src)), (16, memoryview(b"readonly"))],
            [(0, memoryview(src))] * 3,
            [(0, memoryview(src)), (-1, memoryview(src))],
            [(0, memoryview(src)), (16, memoryview(bytearray()))],
        ):
            with pytest.raises(ValueError):
                ring.write_batch_from(fd, writes)
            assert path.read_bytes() == b"x" * 128
    finally:
        ring.close()
        os.close(fd)
