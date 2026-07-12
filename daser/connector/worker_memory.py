# SPDX-License-Identifier: Apache-2.0
"""Bounded worker-side CUDA staging memory."""

from __future__ import annotations

# Standard
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

# Third Party
import torch

DEFAULT_STORE_STAGING_BYTES = 1536 << 20
DEFAULT_PENDING_STORE_STAGING_BYTES = 3072 << 20
MIN_STORE_STAGING_BYTES = 64 << 20
MIN_STORE_STAGING_POOL_DEPTH = 1


class _CudaStagingLeaseOwner(Protocol):
    """Pool interface required by ``CudaStagingLease``."""

    def release(self, lease: "CudaStagingLease") -> None:
        """Return a lease to its owning pool.

        Args:
            lease: Lease previously returned by that pool.
        """


@dataclass
class CudaStagingLease:
    """One logical staging allocation leased from a fixed CUDA pool.

    Args:
        pool: Owning pool that receives the allocation on release.
        tensor: Backing tensor, possibly larger than ``nbytes``.
        nbytes: Logical byte count used by the current transfer.

    Async/thread-safety:
        The worker thread releases a lease only after every CUDA IPC user has
        finished with it.
    """

    pool: _CudaStagingLeaseOwner
    tensor: torch.Tensor
    nbytes: int
    _released: bool = False

    @property
    def view(self) -> torch.Tensor:
        """Return the active one-dimensional uint8 tensor view."""
        return self.tensor[: self.nbytes]

    def release(self) -> None:
        """Return this lease to its owning pool once."""
        if self._released:
            return
        self._released = True
        self.pool.release(self)


class FixedCudaStagingPool:
    """Preallocate and lease fixed-size worker CUDA staging buffers.

    Args:
        device: Device on which staging tensors are allocated.
        buffer_bytes: Size of each fixed staging buffer.
        depth: Number of fixed buffers to allocate.

    Async/thread-safety:
        One vLLM worker thread owns the pool. It never allocates after
        construction; callers release leases after async transfer completes.
    """

    def __init__(
        self,
        device: torch.device,
        buffer_bytes: int,
        depth: int,
    ) -> None:
        if buffer_bytes <= 0:
            raise ValueError("buffer_bytes must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        self._buffer_bytes = buffer_bytes
        self._buffers: list[torch.Tensor] = [
            torch.empty(buffer_bytes, dtype=torch.uint8, device=device)
            for _ in range(depth)
        ]
        self._free_indices: list[int] = list(range(depth))

    @property
    def buffer_bytes(self) -> int:
        """Return the fixed capacity of each staging buffer."""
        return self._buffer_bytes

    @property
    def available(self) -> int:
        """Return the number of currently free staging buffers."""
        return len(self._free_indices)

    @property
    def depth(self) -> int:
        """Return the number of fixed buffers in the pool."""
        return len(self._buffers)

    def buffer(self, index: int) -> torch.Tensor:
        """Return a backing tensor for one-time CUDA IPC registration.

        Args:
            index: Fixed buffer index.

        Returns:
            The full preallocated backing tensor.

        Async/thread-safety:
            The caller must not mutate the tensor outside the lease protocol.
        """
        if index < 0 or index >= len(self._buffers):
            raise ValueError(f"fixed staging buffer index out of range: {index}")
        return self._buffers[index]

    def acquire(
        self,
        nbytes: int,
        wait_for_release: Callable[[], None] | None = None,
    ) -> CudaStagingLease:
        """Lease one preallocated staging buffer.

        Args:
            nbytes: Logical transfer byte count.
            wait_for_release: Optional callback invoked once when every buffer
                is leased.

        Returns:
            Lease whose view is limited to ``nbytes``.

        Raises:
            ValueError: If ``nbytes`` is invalid.
            RuntimeError: If no buffer is available after the callback.
        """
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if nbytes > self._buffer_bytes:
            raise ValueError(
                f"staging request {nbytes} exceeds fixed staging buffer "
                f"{self._buffer_bytes}"
            )
        if not self._free_indices and wait_for_release is not None:
            wait_for_release()
        if not self._free_indices:
            raise RuntimeError("no fixed staging buffers available")
        return self.acquire_index(self._free_indices[0], nbytes)

    def acquire_index(self, index: int, nbytes: int) -> CudaStagingLease:
        """Lease a specific preallocated staging buffer.

        Args:
            index: Fixed buffer index.
            nbytes: Logical transfer byte count.

        Returns:
            Lease whose view is limited to ``nbytes``.

        Raises:
            ValueError: If ``index`` or ``nbytes`` is invalid.
            RuntimeError: If the requested buffer is already leased.
        """
        if index < 0 or index >= len(self._buffers):
            raise ValueError(f"fixed staging buffer index out of range: {index}")
        if nbytes < 0:
            raise ValueError("nbytes must be non-negative")
        if nbytes > self._buffer_bytes:
            raise ValueError(
                f"staging request {nbytes} exceeds fixed staging buffer "
                f"{self._buffer_bytes}"
            )
        if index not in self._free_indices:
            raise RuntimeError(f"fixed staging buffer {index} is not available")
        self._free_indices.remove(index)
        return CudaStagingLease(
            pool=self,
            tensor=self._buffers[index],
            nbytes=nbytes,
        )

    def release(self, lease: CudaStagingLease) -> None:
        """Return a lease to this pool.

        Args:
            lease: Lease previously returned by this pool.

        Raises:
            ValueError: If the lease belongs to another pool.
        """
        for index, tensor in enumerate(self._buffers):
            if tensor is lease.tensor:
                if index not in self._free_indices:
                    self._free_indices.append(index)
                    self._free_indices.sort()
                return
        raise ValueError("lease does not belong to this fixed staging pool")


def derive_staging_limits(device: torch.device) -> tuple[int, int]:
    """Return bounded staging batch and pending byte limits for a device.

    Args:
        device: Device that owns worker-side staging tensors.

    Returns:
        Tuple of single-buffer bytes and total pending bytes.

    Async/thread-safety:
        Reads CUDA device properties during worker initialization before
        request traffic starts.
    """
    if device.type != "cuda":
        return DEFAULT_STORE_STAGING_BYTES, DEFAULT_PENDING_STORE_STAGING_BYTES
    props = torch.cuda.get_device_properties(device)
    total = int(props.total_memory)
    try:
        free, _ = torch.cuda.mem_get_info(device)
        free = int(free)
    except (RuntimeError, TypeError, ValueError):
        free = total
    batch = min(
        DEFAULT_STORE_STAGING_BYTES,
        max(MIN_STORE_STAGING_BYTES, min(total // 50, free // 10)),
    )
    pending = min(
        DEFAULT_PENDING_STORE_STAGING_BYTES,
        max(batch, min(total // 25, free // 5)),
    )
    return batch, pending


def store_staging_pool_depth(
    buffer_bytes: int,
    pending_limit_bytes: int,
) -> int:
    """Return fixed store staging pool depth for a byte budget.

    Args:
        buffer_bytes: Capacity of one fixed staging buffer.
        pending_limit_bytes: Total pending store staging byte budget.

    Returns:
        Number of fixed store staging buffers to preallocate.

    Async/thread-safety:
        Pure startup calculation.
    """
    if buffer_bytes <= 0:
        raise ValueError("buffer_bytes must be positive")
    if pending_limit_bytes <= 0:
        return MIN_STORE_STAGING_POOL_DEPTH
    return max(
        MIN_STORE_STAGING_POOL_DEPTH,
        pending_limit_bytes // buffer_bytes,
    )


def load_staging_pool_depth(
    buffer_bytes: int,
    pending_limit_bytes: int,
    device: torch.device,
    max_inflight: int,
    reserve_bytes: int,
) -> int:
    """Return fixed load staging depth under memory and inflight limits.

    Args:
        buffer_bytes: Capacity of one fixed staging buffer.
        pending_limit_bytes: Existing worker staging byte budget.
        device: Device used for staging allocation.
        max_inflight: Maximum request loads allowed in flight.
        reserve_bytes: Free CUDA memory to leave unallocated.

    Returns:
        Number of fixed load staging buffers to preallocate.

    Async/thread-safety:
        Called during worker initialization before request traffic. It may query
        CUDA free memory but does not allocate.
    """
    depth = min(
        max_inflight,
        store_staging_pool_depth(buffer_bytes, pending_limit_bytes),
    )
    if device.type != "cuda":
        return max(1, depth)
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
    except Exception:  # noqa: BLE001
        return max(1, depth)
    usable_bytes = max(0, int(free_bytes) - reserve_bytes)
    memory_depth = max(1, usable_bytes // buffer_bytes)
    return max(1, min(depth, memory_depth))
