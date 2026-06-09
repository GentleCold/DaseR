# SPDX-License-Identifier: Apache-2.0

# Standard
import ctypes
import mmap
import os
import struct
import threading

_SYS_IO_URING_SETUP = 425
_SYS_IO_URING_ENTER = 426

_IORING_OP_READ = 22
_IORING_OP_WRITE = 23

_IORING_ENTER_GETEVENTS = 1

_IORING_OFF_SQ_RING = 0
_IORING_OFF_CQ_RING = 0x8000000
_IORING_OFF_SQES = 0x10000000

_IORING_FEAT_SINGLE_MMAP = 1
_MAX_RW_COUNT = 0x7FFFF000


class _SqringOffsets(ctypes.Structure):
    """Kernel ABI layout for ``struct io_sqring_offsets``."""

    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("dropped", ctypes.c_uint32),
        ("array", ctypes.c_uint32),
        ("resv1", ctypes.c_uint32),
        ("resv2", ctypes.c_uint64),
    ]


class _CqringOffsets(ctypes.Structure):
    """Kernel ABI layout for ``struct io_cqring_offsets``."""

    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("overflow", ctypes.c_uint32),
        ("cqes", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("resv1", ctypes.c_uint32),
        ("resv2", ctypes.c_uint64),
    ]


class _UringParams(ctypes.Structure):
    """Kernel ABI layout for ``struct io_uring_params``."""

    _fields_ = [
        ("sq_entries", ctypes.c_uint32),
        ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32),
        ("resv", ctypes.c_uint32 * 3),
        ("sq_off", _SqringOffsets),
        ("cq_off", _CqringOffsets),
    ]


class _Sqe(ctypes.Structure):
    """Kernel ABI layout for ``struct io_uring_sqe``."""

    _fields_ = [
        ("opcode", ctypes.c_uint8),
        ("flags", ctypes.c_uint8),
        ("ioprio", ctypes.c_uint16),
        ("fd", ctypes.c_int32),
        ("off", ctypes.c_uint64),
        ("addr", ctypes.c_uint64),
        ("len", ctypes.c_uint32),
        ("rw_flags", ctypes.c_uint32),
        ("user_data", ctypes.c_uint64),
        ("buf_index", ctypes.c_uint16),
        ("personality", ctypes.c_uint16),
        ("splice_fd_in", ctypes.c_int32),
        ("pad2", ctypes.c_uint64 * 2),
    ]


class _Cqe(ctypes.Structure):
    """Kernel ABI layout for ``struct io_uring_cqe``."""

    _fields_ = [
        ("user_data", ctypes.c_uint64),
        ("res", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


class NativeIOUring:
    """Small synchronous io_uring wrapper for positioned file reads/writes.

    Args:
        entries: Submission queue depth. Operations are serialized by this
            wrapper, but a small queue keeps the ABI setup conventional.

    Async/thread-safety:
        Methods are synchronous and protected by a thread lock. Callers that
        must not block an asyncio event loop should run them in an executor.
    """

    def __init__(self, entries: int = 8) -> None:
        if entries <= 0:
            raise ValueError("entries must be positive")
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._params = _UringParams()
        fd = self._libc.syscall(
            _SYS_IO_URING_SETUP,
            ctypes.c_uint32(entries),
            ctypes.byref(self._params),
        )
        if fd < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))

        self._ring_fd = int(fd)
        self._lock = threading.Lock()
        self._next_user_data = 1

        sq_ring_size = (
            self._params.sq_off.array
            + self._params.sq_entries * ctypes.sizeof(ctypes.c_uint32)
        )
        cq_ring_size = (
            self._params.cq_off.cqes + self._params.cq_entries * ctypes.sizeof(_Cqe)
        )
        if self._params.features & _IORING_FEAT_SINGLE_MMAP:
            ring_size = max(sq_ring_size, cq_ring_size)
            self._sq_ring = mmap.mmap(
                self._ring_fd,
                ring_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=_IORING_OFF_SQ_RING,
            )
            self._cq_ring = self._sq_ring
            self._cq_ring_owned = False
        else:
            self._sq_ring = mmap.mmap(
                self._ring_fd,
                sq_ring_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=_IORING_OFF_SQ_RING,
            )
            self._cq_ring = mmap.mmap(
                self._ring_fd,
                cq_ring_size,
                mmap.MAP_SHARED,
                mmap.PROT_READ | mmap.PROT_WRITE,
                offset=_IORING_OFF_CQ_RING,
            )
            self._cq_ring_owned = True

        self._sqes = mmap.mmap(
            self._ring_fd,
            self._params.sq_entries * ctypes.sizeof(_Sqe),
            mmap.MAP_SHARED,
            mmap.PROT_READ | mmap.PROT_WRITE,
            offset=_IORING_OFF_SQES,
        )

    def read(self, fd: int, file_offset: int, nbytes: int) -> bytes:
        """Read bytes at a file offset through io_uring.

        Args:
            fd: Open file descriptor.
            file_offset: Byte offset in the file.
            nbytes: Number of bytes to read.

        Returns:
            Bytes read from the file.

        Thread-safety:
            Serialized by the wrapper lock. The call blocks the caller until
            the io_uring completion arrives.
        """
        if nbytes == 0:
            return b""
        buf = bytearray(nbytes)
        view = memoryview(buf)
        cursor = 0
        while cursor < nbytes:
            chunk = min(_MAX_RW_COUNT, nbytes - cursor)
            self._submit_and_wait(
                _IORING_OP_READ,
                fd,
                file_offset + cursor,
                view[cursor : cursor + chunk],
                chunk,
            )
            cursor += chunk
        return bytes(buf)

    def read_into(self, fd: int, file_offset: int, dst: memoryview) -> int:
        """Read bytes at a file offset into a writable buffer.

        Args:
            fd: Open file descriptor.
            file_offset: Byte offset in the file.
            dst: Writable destination buffer.

        Returns:
            Number of bytes read into ``dst``.

        Thread-safety:
            Serialized by the wrapper lock. The destination buffer must not be
            mutated by another thread during the call.
        """
        view = memoryview(dst).cast("B")
        if view.readonly:
            raise ValueError("dst must be writable")
        total = len(view)
        cursor = 0
        while cursor < total:
            chunk = min(_MAX_RW_COUNT, total - cursor)
            self._submit_and_wait(
                _IORING_OP_READ,
                fd,
                file_offset + cursor,
                view[cursor : cursor + chunk],
                chunk,
            )
            cursor += chunk
        return total

    def readv_into(
        self,
        fd: int,
        reads: list[tuple[int, memoryview]],
    ) -> int:
        """Read multiple positioned ranges into writable buffers.

        Args:
            fd: Open file descriptor.
            reads: ``(file_offset, dst)`` pairs. Each destination is read fully.

        Returns:
            Total number of bytes read.

        Thread-safety:
            Serialized by the wrapper lock. Destination buffers must not be
            mutated by another thread until this call returns.
        """
        total = 0
        pending: list[tuple[int, memoryview]] = []
        for file_offset, dst in reads:
            view = memoryview(dst).cast("B")
            if view.readonly:
                raise ValueError("dst must be writable")
            cursor = 0
            while cursor < len(view):
                chunk = min(_MAX_RW_COUNT, len(view) - cursor)
                pending.append((file_offset + cursor, view[cursor : cursor + chunk]))
                total += chunk
                cursor += chunk
        if not pending:
            return 0

        results: list[int] = []
        batch_limit = int(self._params.sq_entries)
        with self._lock:
            for batch_start in range(0, len(pending), batch_limit):
                batch = pending[batch_start : batch_start + batch_limit]
                start_user_data = self._next_user_data
                self._next_user_data += len(batch)
                for idx, (file_offset, view) in enumerate(batch):
                    self._submit_locked(
                        _IORING_OP_READ,
                        fd,
                        file_offset,
                        view,
                        len(view),
                        start_user_data + idx,
                    )
                self._enter(
                    len(batch),
                    len(batch),
                    _IORING_ENTER_GETEVENTS,
                )
                results.extend(
                    self._wait_completions_locked(
                        start_user_data,
                        [len(view) for _offset, view in batch],
                    )
                )
        for idx, res in enumerate(results):
            if res < 0:
                raise OSError(-res, os.strerror(-res))
            expected = len(pending[idx][1])
            if res != expected:
                raise IOError(f"short io_uring result: {res} != {expected}")
        return total

    def write(self, fd: int, file_offset: int, data: bytes | memoryview) -> int:
        """Write bytes at a file offset through io_uring.

        Args:
            fd: Open file descriptor.
            file_offset: Byte offset in the file.
            data: Bytes to write.

        Returns:
            Number of bytes written.

        Thread-safety:
            Serialized by the wrapper lock. The source buffer must remain valid
            and unchanged until the call returns.
        """
        if not data:
            return 0
        view = memoryview(data).cast("B")
        total = len(view)
        cursor = 0
        while cursor < total:
            chunk = min(_MAX_RW_COUNT, total - cursor)
            chunk_view = view[cursor : cursor + chunk]
            if chunk_view.readonly:
                chunk_buf = bytearray(chunk_view)
                io_buf = memoryview(chunk_buf)
            else:
                io_buf = chunk_view
            self._submit_and_wait(
                _IORING_OP_WRITE,
                fd,
                file_offset + cursor,
                io_buf,
                chunk,
            )
            cursor += chunk
        return total

    def close(self) -> None:
        """Close the io_uring file descriptor and ring mappings.

        Returns:
            None.

        Thread-safety:
            Call after all read/write calls have returned. The method does not
            coordinate with in-flight operations.
        """
        self._sqes.close()
        if self._cq_ring_owned:
            self._cq_ring.close()
        self._sq_ring.close()
        os.close(self._ring_fd)

    def _submit_and_wait(
        self,
        opcode: int,
        fd: int,
        file_offset: int,
        buf: memoryview,
        nbytes: int,
    ) -> int:
        """Submit one read/write SQE and wait for its completion."""
        with self._lock:
            user_data = self._next_user_data
            self._next_user_data += 1
            self._submit_locked(opcode, fd, file_offset, buf, nbytes, user_data)
            self._enter(1, 1, _IORING_ENTER_GETEVENTS)
            res = self._wait_completion_locked(user_data)
        if res < 0:
            raise OSError(-res, os.strerror(-res))
        if res != nbytes:
            raise IOError(f"short io_uring result: {res} != {nbytes}")
        return res

    def _submit_locked(
        self,
        opcode: int,
        fd: int,
        file_offset: int,
        buf: memoryview,
        nbytes: int,
        user_data: int,
    ) -> None:
        """Write an SQE into the submission ring and enter the kernel."""
        sq_head = self._read_u32(self._sq_ring, self._params.sq_off.head)
        sq_tail = self._read_u32(self._sq_ring, self._params.sq_off.tail)
        sq_entries = self._read_u32(self._sq_ring, self._params.sq_off.ring_entries)
        if sq_tail - sq_head >= sq_entries:
            raise RuntimeError("io_uring submission queue is full")

        sq_mask = self._read_u32(self._sq_ring, self._params.sq_off.ring_mask)
        index = sq_tail & sq_mask
        sqe = _Sqe()
        sqe.opcode = opcode
        sqe.fd = fd
        sqe.off = file_offset
        sqe.addr = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        sqe.len = nbytes
        sqe.user_data = user_data
        self._write_struct(
            self._sqes,
            index * ctypes.sizeof(_Sqe),
            sqe,
        )
        self._write_u32(
            self._sq_ring,
            self._params.sq_off.array + index * ctypes.sizeof(ctypes.c_uint32),
            index,
        )
        self._write_u32(self._sq_ring, self._params.sq_off.tail, sq_tail + 1)

    def _enter(self, to_submit: int, min_complete: int, flags: int) -> None:
        """Enter the kernel to submit SQEs and optionally wait for CQEs."""
        ret = self._libc.syscall(
            _SYS_IO_URING_ENTER,
            ctypes.c_int(self._ring_fd),
            ctypes.c_uint32(to_submit),
            ctypes.c_uint32(min_complete),
            ctypes.c_uint32(flags),
            ctypes.c_void_p(0),
            ctypes.c_size_t(0),
        )
        if ret < 0:
            errno = ctypes.get_errno()
            raise OSError(errno, os.strerror(errno))

    def _wait_completion_locked(self, user_data: int) -> int:
        """Wait for a CQE matching ``user_data`` and return its result."""
        return self._wait_completions_locked(user_data, [0])[0]

    def _wait_completions_locked(
        self,
        start_user_data: int,
        expected_sizes: list[int],
    ) -> list[int]:
        """Wait for a contiguous user-data range and return ordered results."""
        remaining = len(expected_sizes)
        results: list[int | None] = [None] * remaining
        while True:
            cq_head = self._read_u32(self._cq_ring, self._params.cq_off.head)
            cq_tail = self._read_u32(self._cq_ring, self._params.cq_off.tail)
            if cq_head != cq_tail:
                cq_mask = self._read_u32(self._cq_ring, self._params.cq_off.ring_mask)
                index = cq_head & cq_mask
                cqe = self._read_struct(
                    self._cq_ring,
                    self._params.cq_off.cqes + index * ctypes.sizeof(_Cqe),
                    _Cqe,
                )
                res = int(cqe.res)
                seen_user_data = int(cqe.user_data)
                self._write_u32(self._cq_ring, self._params.cq_off.head, cq_head + 1)
                index = seen_user_data - start_user_data
                if index < 0 or index >= len(expected_sizes):
                    raise RuntimeError(
                        "unexpected io_uring completion "
                        f"{seen_user_data}, expected range "
                        f"[{start_user_data}, {start_user_data + len(expected_sizes)})"
                    )
                results[index] = res
                remaining -= 1
                if remaining == 0:
                    return [int(result) for result in results]
                continue

            self._enter(0, 1, _IORING_ENTER_GETEVENTS)

    def _read_u32(self, buf: mmap.mmap, offset: int) -> int:
        """Read a uint32 from a ring mapping without exporting the buffer."""
        return struct.unpack_from("I", buf, offset)[0]

    def _write_u32(self, buf: mmap.mmap, offset: int, value: int) -> None:
        """Write a uint32 to a ring mapping without exporting the buffer."""
        struct.pack_into("I", buf, offset, value)

    def _read_struct(
        self,
        buf: mmap.mmap,
        offset: int,
        struct_type: type[ctypes.Structure],
    ) -> ctypes.Structure:
        """Read a ctypes structure from a mapping without retaining exports."""
        raw = buf[offset : offset + ctypes.sizeof(struct_type)]
        return struct_type.from_buffer_copy(raw)

    def _write_struct(
        self,
        buf: mmap.mmap,
        offset: int,
        value: ctypes.Structure,
    ) -> None:
        """Write a ctypes structure to a mapping without retaining exports."""
        raw = ctypes.string_at(ctypes.addressof(value), ctypes.sizeof(value))
        buf[offset : offset + len(raw)] = raw
