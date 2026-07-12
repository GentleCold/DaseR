# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import threading
from typing import Any

import torch

from daser.connector.ipc_client import IPCClientAsync
from daser.connector.metadata import ReqStoreSpec, StoreWriteSpan
from daser.connector.worker_memory import (
    DEFAULT_STORE_STAGING_BYTES,
    CudaStagingLease,
)


class StorePipeline:
    """Own store-side asyncio, IPC, staging, and future state.

    Args:
        socket_path: DaseR server Unix socket path.

    Async/thread-safety:
        Construction and public methods run on the vLLM worker thread. Async
        IPC runs exclusively on the private ``daser-store-io`` thread.
    """

    def __init__(self, socket_path: str) -> None:
        self._client = IPCClientAsync(socket_path)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="daser-store-io",
        )
        self.save_futures: list[Any] = []
        self.pending_staging_bytes = 0
        self.staging_bytes = 0
        self.pending_staging_limit_bytes = 0
        self.staging_pool: Any | None = None
        self.pending_finished_saves: dict[str, Any] = {}
        self._thread.start()

    @property
    def client(self) -> IPCClientAsync:
        """Return the store IPC client for pipeline-owned coroutines."""
        return self._client

    def submit(self, coro: Any) -> Any:
        """Submit a coroutine to the private store event loop."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        """Close the IPC client and stop the private store event loop."""
        self.submit(self._client.close()).result(timeout=5.0)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5.0)

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()


@dataclass(frozen=True)
class StagedStoreBatch:
    """Hold a worker CUDA snapshot until its async store completes."""

    buffer: torch.Tensor
    ready_event: torch.cuda.Event | None
    spans: list[StoreWriteSpan]
    lease: CudaStagingLease | None


def build_staging_store_batches(
    reqs_to_store: dict[str, ReqStoreSpec],
    slot_size: int,
    max_batch_bytes: int = DEFAULT_STORE_STAGING_BYTES,
) -> list[tuple[list[int], list[StoreWriteSpan]]]:
    """Split store requests into bounded slot-major staging batches.

    Args:
        reqs_to_store: Request IDs mapped to store specifications.
        slot_size: Bytes stored for one rank-local KV slot.
        max_batch_bytes: Maximum GPU staging bytes in one batch.

    Returns:
        Ordered block ID and server write-span batches.

    Async/thread-safety:
        Pure CPU planning; safe to call from worker or store-loop threads.
    """
    if slot_size <= 0:
        raise ValueError("slot_size must be positive")
    max_slots = max(1, max_batch_bytes // slot_size)
    batches: list[tuple[list[int], list[StoreWriteSpan]]] = []
    batch_blocks: list[int] = []
    batch_spans: list[StoreWriteSpan] = []
    written_specs: set[tuple[str, int, int, int, int]] = set()

    def flush_batch() -> None:
        nonlocal batch_blocks, batch_spans
        if batch_blocks:
            batches.append((batch_blocks, batch_spans))
        batch_blocks = []
        batch_spans = []

    for spec in reqs_to_store.values():
        source_key = (
            spec.chunk_key,
            spec.start_slot,
            spec.num_slots,
            spec.file_offset,
            len(spec.block_ids),
        )
        if source_key in written_specs:
            continue
        written_specs.add(source_key)
        cursor = 0
        while cursor < len(spec.block_ids):
            if len(batch_blocks) >= max_slots:
                flush_batch()
            available = max_slots - len(batch_blocks)
            take = min(available, len(spec.block_ids) - cursor)
            if take <= 0:
                flush_batch()
                continue
            source_slot = len(batch_blocks)
            part = spec.block_ids[cursor : cursor + take]
            batch_blocks.extend(part)
            batch_spans.append(
                StoreWriteSpan(
                    source_offset=source_slot * slot_size,
                    nbytes=take * slot_size,
                    file_offset=spec.file_offset + cursor * slot_size,
                    chunk_key=spec.chunk_key,
                    start_slot=spec.start_slot,
                    num_slots=spec.num_slots,
                )
            )
            cursor += take
    flush_batch()
    return batches
