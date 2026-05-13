# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import enum
import os
import time
from typing import Optional

# Third Party
import cupy
import kvikio
import kvikio.cufile
import kvikio.defaults

# First Party
from daser.logging import init_logger, init_perf_logger
from daser.perf import LatencyHistogram

logger = init_logger(__name__)
perf = init_perf_logger(__name__)


class TransferBackend(enum.Enum):
    """Active IO backend for GDSTransferLayer."""

    GDS = "gds"  # cuFile GDS — direct NVMe↔GPU DMA, no CPU involvement
    COMPAT = "compat"  # kvikio compat mode — POSIX thread-pool + CPU bounce buffer


class GDSTransferLayer:
    """Async NVMe↔GPU IO using kvikio (cuFile GDS or compat-mode fallback).

    Opens a pre-existing file for read+write. Exposes coroutine-compatible
    methods that wrap kvikio IOFuture in asyncio via run_in_executor so
    callers stay in a pure asyncio event loop.

    Backend is selected once at construction from kvikio.defaults.get("compat_mode"):
    - CompatMode.OFF  → GDS path (direct DMA)
    - CompatMode.ON   → compat path (POSIX + CPU bounce, still async via thread pool)
    - CompatMode.AUTO → treated as COMPAT unless GDS actually activates

    Args:
        path: absolute path to the pre-allocated store file.
        nthreads: kvikio thread-pool size used in compat mode (default 4).
            Increasing this overlaps GPU→CPU staging with disk IO.
            Ignored when GDS direct-DMA is active.
    """

    def __init__(self, path: str, nthreads: int = 4) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Store file not found: {path}")

        mode = kvikio.defaults.get("compat_mode")
        if mode == kvikio.CompatMode.OFF:
            self._backend = TransferBackend.GDS
        else:
            self._backend = TransferBackend.COMPAT
            # In compat mode, kvikio uses a POSIX thread pool for IO.
            # Default is 1 thread which serialises all writes; 4 threads
            # overlaps GPU→CPU staging with disk IO on btrfs/NVMe workloads.
            kvikio.defaults.set("num_threads", nthreads)

        self._file = kvikio.cufile.CuFile(path, "r+")
        self._perf_histograms: dict[str, LatencyHistogram] = {}
        self._perf_enabled = os.environ.get("DASER_PERF_LOG", "0") == "1"
        logger.info(
            "[GDS] backend=%s nthreads=%d path=%s", self._backend.name, nthreads, path
        )

    @property
    def backend(self) -> TransferBackend:
        """The active IO backend (immutable after init)."""
        return self._backend

    def _get_histogram(self, name: str) -> "LatencyHistogram | None":
        """Return the histogram for *name*, creating it lazily when perf is on.

        Args:
            name: metric name, e.g. "gds_read" or "gds_write".

        Returns:
            LatencyHistogram, or None when DASER_PERF_LOG is off.
        """
        if not self._perf_enabled:
            return None
        hist = self._perf_histograms.get(name)
        if hist is None:
            hist = LatencyHistogram(name, unit="ms", scale=1000)
            self._perf_histograms[name] = hist
        return hist

    def get_histograms(self) -> dict[str, LatencyHistogram]:
        """Return all collected latency histograms for this transfer layer.

        Returns:
            Dict mapping metric name to LatencyHistogram.
        """
        return dict(self._perf_histograms)

    async def write_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Write from a GPU buffer to the store file at file_offset.

        Non-blocking: submits IO and suspends the coroutine until the
        IOFuture completes in the kvikio thread pool.

        Args:
            buf: cupy ndarray on device (or host in compat mode).
            file_offset: byte offset in the store file to write at.
            nbytes: bytes to write; defaults to full buf size.

        Returns:
            Number of bytes written.
        """
        hist = self._get_histogram("gds_write")
        t0 = time.perf_counter() if hist is not None else 0.0
        # Measure kvikio submission overhead separately
        t_kvikio = time.perf_counter()
        loop = asyncio.get_event_loop()
        io_future = self._file.pwrite(buf, nbytes, file_offset)
        t_submit = time.perf_counter() - t_kvikio
        result = await loop.run_in_executor(None, io_future.get)
        if hist is not None:
            elapsed = time.perf_counter() - t0
            hist.record(elapsed)
            perf.record("gds_write_submit_ms", t_submit * 1000, "ms")
            nbytes_val = nbytes if nbytes is not None else buf.nbytes
            perf.record("gds_write_mb", nbytes_val / 1e6, "MB")
        return result

    async def read_into_async(
        self,
        buf: cupy.ndarray,
        file_offset: int,
        nbytes: Optional[int] = None,
    ) -> int:
        """Read from the store file into a GPU buffer at file_offset.

        Non-blocking: submits IO and suspends the coroutine until the
        IOFuture completes in the kvikio thread pool.

        Args:
            buf: pre-allocated cupy ndarray on device to read into.
            file_offset: byte offset in the store file to read from.
            nbytes: bytes to read; defaults to full buf size.

        Returns:
            Number of bytes read.
        """
        hist = self._get_histogram("gds_read")
        t0 = time.perf_counter() if hist is not None else 0.0
        t_kvikio = time.perf_counter()
        loop = asyncio.get_event_loop()
        io_future = self._file.pread(buf, nbytes, file_offset)
        t_submit = time.perf_counter() - t_kvikio
        result = await loop.run_in_executor(None, io_future.get)
        if hist is not None:
            elapsed = time.perf_counter() - t0
            hist.record(elapsed)
            perf.record("gds_read_submit_ms", t_submit * 1000, "ms")
            nbytes_val = nbytes if nbytes is not None else buf.nbytes
            perf.record("gds_read_mb", nbytes_val / 1e6, "MB")
        return result

    def close(self) -> None:
        """Close the underlying kvikio file handle."""
        self._file.close()
        logger.debug("[GDS] file closed")

    def __enter__(self) -> "GDSTransferLayer":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
