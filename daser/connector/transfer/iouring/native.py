# SPDX-License-Identifier: Apache-2.0

# Standard
import ctypes
import errno
import mmap
import os

# Third Party
import torch


class NativeIOUringError(OSError):
    """Raised when a native io_uring syscall or completion fails."""


class _IOSqringOffsets(ctypes.Structure):
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


class _IOCqringOffsets(ctypes.Structure):
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


class _IOUringParams(ctypes.Structure):
    _fields_ = [
        ("sq_entries", ctypes.c_uint32),
        ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32),
        ("resv", ctypes.c_uint32 * 3),
        ("sq_off", _IOSqringOffsets),
        ("cq_off", _IOCqringOffsets),
    ]


class _IOUringSqe(ctypes.Structure):
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
        ("__pad2", ctypes.c_uint64 * 2),
    ]


class _IOUringCqe(ctypes.Structure):
    _fields_ = [
        ("user_data", ctypes.c_uint64),
        ("res", ctypes.c_int32),
        ("flags", ctypes.c_uint32),
    ]


_SYS_IO_URING_SETUP = 425
_SYS_IO_URING_ENTER = 426

_IORING_OP_READ = 22
_IORING_OP_WRITE = 23

_IORING_ENTER_GETEVENTS = 1
_IORING_OFF_SQ_RING = 0
_IORING_OFF_CQ_RING = 0x8000000
_IORING_OFF_SQES = 0x10000000
_IORING_FEAT_SINGLE_MMAP = 1

_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.syscall.restype = ctypes.c_long
_LIBC.syscall.argtypes = [
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_void_p,
    ctypes.c_long,
    ctypes.c_long,
    ctypes.c_void_p,
]


def _syscall(number: int, *args: object) -> int:
    """Run a libc syscall and raise NativeIOUringError on failure.

    Args:
        number: syscall number.
        args: syscall arguments.

    Returns:
        Raw syscall result.
    """
    padded = list(args) + [0] * (5 - len(args))
    result = _LIBC.syscall(number, *padded[:5])
    if result < 0:
        err = ctypes.get_errno()
        raise NativeIOUringError(err, os.strerror(err))
    return int(result)


def _u32_view(mm: mmap.mmap, offset: int) -> ctypes.c_uint32:
    """Return a uint32 view into a ring mmap.

    Args:
        mm: mmap containing the ring.
        offset: byte offset within the mmap.

    Returns:
        ctypes uint32 view backed by the mmap.
    """
    return ctypes.c_uint32.from_buffer(mm, offset)


class NativeIOUring:
    """Minimal native io_uring wrapper for one-at-a-time file reads/writes.

    Args:
        entries: requested queue depth.

    Async/thread-safety:
        This class is synchronous and not internally locked. Submit calls must
        be serialized by the owning async transfer layer.
    """

    def __init__(self, entries: int = 64) -> None:
        params = _IOUringParams()
        fd = _syscall(
            _SYS_IO_URING_SETUP,
            entries,
            ctypes.byref(params),
        )
        self._fd = fd
        self._params = params
        self._closed = False
        sq_ring_sz = params.sq_off.array + params.sq_entries * ctypes.sizeof(
            ctypes.c_uint32
        )
        cq_ring_sz = params.cq_off.cqes + params.cq_entries * ctypes.sizeof(_IOUringCqe)
        if params.features & _IORING_FEAT_SINGLE_MMAP:
            ring_sz = max(sq_ring_sz, cq_ring_sz)
            self._sq_ring = mmap.mmap(
                fd,
                ring_sz,
                flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=_IORING_OFF_SQ_RING,
            )
            self._cq_ring = self._sq_ring
        else:
            self._sq_ring = mmap.mmap(
                fd,
                sq_ring_sz,
                flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=_IORING_OFF_SQ_RING,
            )
            self._cq_ring = mmap.mmap(
                fd,
                cq_ring_sz,
                flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=_IORING_OFF_CQ_RING,
            )
        self._sqes_mmap = mmap.mmap(
            fd,
            params.sq_entries * ctypes.sizeof(_IOUringSqe),
            flags=mmap.MAP_SHARED | mmap.MAP_POPULATE,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
            offset=_IORING_OFF_SQES,
        )
        self._sq_head = _u32_view(self._sq_ring, params.sq_off.head)
        self._sq_tail = _u32_view(self._sq_ring, params.sq_off.tail)
        self._sq_mask = _u32_view(self._sq_ring, params.sq_off.ring_mask)
        self._sq_array = (ctypes.c_uint32 * params.sq_entries).from_buffer(
            self._sq_ring, params.sq_off.array
        )
        self._sqes = (_IOUringSqe * params.sq_entries).from_buffer(self._sqes_mmap)
        self._cq_head = _u32_view(self._cq_ring, params.cq_off.head)
        self._cq_tail = _u32_view(self._cq_ring, params.cq_off.tail)
        self._cq_mask = _u32_view(self._cq_ring, params.cq_off.ring_mask)
        self._cqes = (_IOUringCqe * params.cq_entries).from_buffer(
            self._cq_ring,
            params.cq_off.cqes,
        )
        self._next_user_data = 1

    def read_into(
        self,
        fd: int,
        dst: torch.Tensor,
        file_offset: int,
        nbytes: int,
    ) -> int:
        """Read file bytes into a contiguous CPU tensor.

        Args:
            fd: file descriptor.
            dst: destination CPU tensor.
            file_offset: byte offset in file.
            nbytes: bytes to read.

        Returns:
            Number of bytes read.
        """
        return self._submit(fd, dst, file_offset, nbytes, _IORING_OP_READ)

    def write_from(
        self,
        fd: int,
        src: torch.Tensor,
        file_offset: int,
        nbytes: int,
    ) -> int:
        """Write file bytes from a contiguous CPU tensor.

        Args:
            fd: file descriptor.
            src: source CPU tensor.
            file_offset: byte offset in file.
            nbytes: bytes to write.

        Returns:
            Number of bytes written.
        """
        return self._submit(fd, src, file_offset, nbytes, _IORING_OP_WRITE)

    def close(self) -> None:
        """Close ring mmaps and the ring file descriptor."""
        if self._closed:
            return
        self._closed = True
        self._release_views()
        self._sqes_mmap.close()
        if self._cq_ring is not self._sq_ring:
            self._cq_ring.close()
        self._sq_ring.close()
        os.close(self._fd)

    def _submit(
        self,
        fd: int,
        buf: torch.Tensor,
        file_offset: int,
        nbytes: int,
        opcode: int,
    ) -> int:
        """Submit one read or write SQE and wait for its CQE."""
        if self._closed:
            raise NativeIOUringError(errno.EBADF, "io_uring is closed")
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if nbytes == 0:
            return 0
        if buf.device.type != "cpu":
            raise ValueError("native io_uring buffers must be CPU tensors")
        if not buf.is_contiguous():
            raise ValueError("native io_uring buffers must be contiguous")
        if buf.numel() < nbytes:
            raise ValueError("buffer is smaller than nbytes")

        tail = self._sq_tail.value
        index = tail & self._sq_mask.value
        user_data = self._next_user_data
        self._next_user_data += 1

        sqe = self._sqes[index]
        ctypes.memset(ctypes.byref(sqe), 0, ctypes.sizeof(sqe))
        sqe.opcode = opcode
        sqe.fd = fd
        sqe.off = file_offset
        sqe.addr = int(buf.data_ptr())
        sqe.len = nbytes
        sqe.user_data = user_data
        self._sq_array[index] = index
        self._sq_tail.value = tail + 1

        _syscall(
            _SYS_IO_URING_ENTER,
            self._fd,
            1,
            1,
            _IORING_ENTER_GETEVENTS,
            None,
        )
        return self._consume_completion(user_data)

    def _consume_completion(self, user_data: int) -> int:
        """Consume the completion matching user_data."""
        while self._cq_head.value == self._cq_tail.value:
            _syscall(
                _SYS_IO_URING_ENTER,
                self._fd,
                0,
                1,
                _IORING_ENTER_GETEVENTS,
                None,
            )
        head = self._cq_head.value
        index = head & self._cq_mask.value
        cqe = self._cqes[index]
        result = int(cqe.res)
        seen_user_data = int(cqe.user_data)
        self._cq_head.value = head + 1
        if seen_user_data != user_data:
            raise NativeIOUringError(
                errno.EIO,
                f"unexpected completion user_data={seen_user_data}",
            )
        if result < 0:
            err = -result
            raise NativeIOUringError(err, os.strerror(err))
        return result

    def _release_views(self) -> None:
        """Release ctypes views before closing their backing mmaps."""
        del self._sq_head
        del self._sq_tail
        del self._sq_mask
        del self._sq_array
        del self._sqes
        del self._cq_head
        del self._cq_tail
        del self._cq_mask
        del self._cqes
