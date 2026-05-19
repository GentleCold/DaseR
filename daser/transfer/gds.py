# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import enum
import os
from typing import Optional

# Third Party
import cupy
import kvikio
import kvikio.cufile
import kvikio.defaults

# First Party
from daser.logging import init_logger
from daser.transfer.base import TransferLayer

logger = init_logger(__name__)


class TransferBackend(enum.Enum):
    """Active IO backend for GDSTransferLayer."""

    GDS = "gds"
    COMPAT = "compat"


class GDSTransferLayer(TransferLayer):
    """Async NVMe<->GPU IO using kvikio.

    Args:
        path: absolute path to the pre-allocated store file.
        nthreads: kvikio thread-pool size used in compat mode.

    Async/thread-safety:
        IO futures are awaited through the event loop executor. The backend is
        selected once during construction and never changes at runtime.
    """

    def __init__(self, path: str, nthreads: int = 4) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Store file not found: {path}")

        kvikio.defaults.set("compat_mode", kvikio.CompatMode.OFF)
        try:
            self._file = kvikio.cufile.CuFile(path, "r+")
            self._backend = TransferBackend.GDS
        except RuntimeError as exc:
            logger.warning(
                "[TRANSFER:gds] direct cuFile open failed, falling back to compat: %s",
                exc,
            )
            kvikio.defaults.set("compat_mode", kvikio.CompatMode.ON)
            self._backend = TransferBackend.COMPAT
            kvikio.defaults.set("num_threads", nthreads)
            self._file = kvikio.cufile.CuFile(path, "r+")

        logger.info(
            "[TRANSFER:gds] backend=%s nthreads=%d path=%s",
            self._backend.name,
            nthreads,
            path,
        )

    @property
    def backend(self) -> TransferBackend:
        """The active IO backend."""
        return self._backend

    async def write_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Write from a GPU buffer to the store file.

        Args:
            buf: CuPy ndarray on device or host.
            file_offset: byte offset in the store file.
            nbytes: bytes to write; defaults to full buffer size.

        Returns:
            Number of bytes written.
        """
        loop = asyncio.get_event_loop()
        io_future = self._file.pwrite(buf, nbytes, file_offset)
        return await loop.run_in_executor(None, io_future.get)

    async def read_into_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Read from the store file into a GPU buffer.

        Args:
            buf: pre-allocated CuPy ndarray.
            file_offset: byte offset in the store file.
            nbytes: bytes to read; defaults to full buffer size.

        Returns:
            Number of bytes read.
        """
        loop = asyncio.get_event_loop()
        io_future = self._file.pread(buf, nbytes, file_offset)
        return await loop.run_in_executor(None, io_future.get)

    async def load_bytes(self, dst: cupy.ndarray, file_offset: int, nbytes: int) -> int:
        """Load bytes into a GPU buffer.

        Args:
            dst: destination CuPy ndarray.
            file_offset: byte offset in L2.
            nbytes: bytes to read.

        Returns:
            Number of bytes read.
        """
        return await self.read_into_async(dst, file_offset, nbytes)

    async def store_bytes(
        self, src: cupy.ndarray, file_offset: int, nbytes: int
    ) -> int:
        """Store bytes from a GPU buffer.

        Args:
            src: source CuPy ndarray.
            file_offset: byte offset in L2.
            nbytes: bytes to write.

        Returns:
            Number of bytes written.
        """
        return await self.write_async(src, file_offset, nbytes)

    def close(self) -> None:
        """Close the underlying kvikio file handle."""
        self._file.close()
        logger.debug("[TRANSFER:gds] file closed")

    def __enter__(self) -> "GDSTransferLayer":
        """Return this transfer layer for context-manager use."""
        return self

    def __exit__(self, *_: object) -> None:
        """Close the transfer layer on context-manager exit."""
        self.close()
