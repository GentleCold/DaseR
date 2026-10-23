# SPDX-License-Identifier: Apache-2.0
"""Host-only capacity snapshot shared by FragPack attention and its connector."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

_CAPACITY: FragPackCapacity | None = None


def publish_capacity(capacity: FragPackCapacity) -> None:
    """Publish the actual allocation after a store is created or rebound.

    Args:
        capacity: Fully initialized, host-only allocation snapshot.

    Returns:
        None.

    Async/thread-safety:
        Called on the attention forward thread; replaces one process-local
        reference atomically. Does not perform I/O or touch CUDA tensors.
    """
    global _CAPACITY
    _CAPACITY = capacity


def get_capacity() -> FragPackCapacity | None:
    """Return the published allocation, or None before store initialization.

    Returns:
        Current snapshot; absence must not be interpreted as zero capacity.

    Async/thread-safety:
        Lock-free host reference read on the worker thread; no I/O.
    """
    return _CAPACITY


@dataclass(frozen=True)
class FragPackCapacity:
    """Allocated capacity of one worker's FragPack attention store.

    Attributes:
        blocks: Allocated blocks, including the engine's reserved null block.
        block_tokens: Token slots per block.
        layers: Attention layers on this worker.
        kv_heads: KV heads per layer on this worker.
        page_bytes: Allocated bytes per block-layer.
        raw_page_bytes: BF16 bytes for the same block-layer geometry.
        memory: Persistent tensor bytes by component; excludes model weights,
            allocator reserve, CUDA graphs and per-step temporary tensors.

    Async/thread-safety:
        Published once per store binding on the forward thread. Treat memory
        as immutable; consumers receive independent dictionaries via to_dict.
    """

    blocks: int
    block_tokens: int
    layers: int
    kv_heads: int
    page_bytes: int
    raw_page_bytes: int
    memory: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return an independent, serializable snapshot without GPU access.

        Returns:
            Dictionary of geometry and component bytes.

        Async/thread-safety:
            Reads immutable host metadata; never synchronizes a CUDA stream.
        """
        return asdict(self)
