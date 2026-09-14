# SPDX-License-Identifier: Apache-2.0
"""Track unchanged bytes in fixed, worker-owned load staging buffers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BufferKey = tuple[int, int]
Signature = tuple[tuple[int, int, int], ...]


def _signature(spans: list[dict[str, Any]]) -> Signature:
    result = tuple(
        (int(s["file_offset"]), int(s.get("target_offset", 0)), int(s["nbytes"]))
        for s in spans
    )
    if any(offset < 0 or target < 0 or size < 0 for offset, target, size in result):
        raise ValueError("staging content spans must be nonnegative ranges")
    # Preserve the transfer API's empty-span no-op. Such a span neither proves
    # reusable content nor overlaps a writer.
    return tuple(span for span in result if span[2])


def _overlaps(left: Signature, right: Signature) -> bool:
    return any(
        a < b + size_b and b < a + size_a
        for a, _, size_a in left
        for b, _, size_b in right
    )


@dataclass(eq=False)
class StagingContent:
    """Identify one pending load; readiness never owns or extends a CUDA lease."""

    spans: Signature
    ready: bool = False


class StagingContentIndex:
    """Prove content reuse across read-only consumers of registered load buffers.

    The IPC event loop exclusively owns this index. All methods are synchronous
    metadata operations without IO or CUDA calls. Callers must bracket every
    storage write and invalidate a destination on every load or registration.
    Entries describe existing bytes; they never retain a worker staging lease.
    """

    def __init__(self) -> None:
        self._buffers: dict[BufferKey, StagingContent] = {}
        self._writes: dict[object, Signature] = {}

    def begin_load(
        self, key: BufferKey, spans: list[dict[str, Any]], *, reusable: bool
    ) -> tuple[bool, StagingContent | None]:
        """Check unchanged content or reserve a proof for a new physical load.

        Args:
            key: Registered worker PID and load ring index.
            spans: Complete physical-source and destination byte mappings.
            reusable: Whether the worker promises read-only packed consumption.
        Returns:
            A cache-hit flag and an optional pending load identity.
        Async/thread-safety:
            Call on the owning IPC loop before yielding to the transfer layer.
        """
        previous = self._buffers.pop(key, None)
        if not reusable or not spans:
            return False, None
        signature = _signature(spans)
        if not signature:
            return False, None
        writing = any(_overlaps(signature, write) for write in self._writes.values())
        if (
            previous is not None
            and previous.ready
            and previous.spans == signature
            and not writing
        ):
            self._buffers[key] = previous
            return True, None
        candidate = StagingContent(signature)
        self._buffers[key] = candidate
        return False, candidate

    def finish_load(
        self, key: BufferKey, candidate: StagingContent | None, *, success: bool
    ) -> None:
        """Publish a drained load only if its proof survived intervening writes.

        Args:
            key: Registered worker PID and load ring index.
            candidate: Identity returned by begin_load, or None for ineligible IO.
            success: Whether all transfer and GPU-copy completion checks passed.
        Returns:
            None. Failed or invalidated candidates never become reusable.
        Async/thread-safety:
            Call on the IPC loop after IO/H2D completion, including errors.
        """
        if candidate is None or self._buffers.get(key) is not candidate:
            return
        if not success or any(
            _overlaps(candidate.spans, write) for write in self._writes.values()
        ):
            self._buffers.pop(key)
        else:
            candidate.ready = True

    def is_current(self, key: BufferKey, candidate: StagingContent) -> bool:
        """Check that a source load has not been revoked by replacement or writes.

        Args:
            key: Registered source PID and ring index.
            candidate: Pending source identity, possibly still awaiting H2D.
        Returns:
            True only while the identity is current and no writer overlaps it.
        Async/thread-safety:
            IPC-loop-only; callers must retain the source RPC through device use.
        """
        return self._buffers.get(key) is candidate and not any(
            _overlaps(candidate.spans, write) for write in self._writes.values()
        )

    def begin_write(self, spans: list[dict[str, Any]]) -> object:
        """Invalidate overlapping proofs before a possibly partial storage write.

        Args:
            spans: Actual physical write spans, after packed placement.
        Returns:
            A token that must be passed once to end_write in a finally block.
        Async/thread-safety:
            IPC-loop-only; call immediately before awaiting the storage write.
        """
        token = object()
        signature = _signature(spans)
        self._writes[token] = signature
        self._invalidate(signature)
        return token

    def end_write(self, token: object) -> None:
        """Invalidate loads started during a write, including a failed write.

        Args:
            token: Identity returned by begin_write.
        Returns:
            None. A duplicate or unknown token raises KeyError.
        Async/thread-safety:
            Call on the IPC loop after the write coroutine has drained.
        """
        self._invalidate(self._writes.pop(token))

    def forget(self, key: BufferKey) -> None:
        """Discard content when a registered buffer is replaced.

        Args:
            key: Worker PID and load ring index being replaced.
        Returns:
            None; absence is allowed.
        Async/thread-safety:
            IPC-loop-only, before publishing the new CUDA mapping.
        """
        self._buffers.pop(key, None)

    def clear(self) -> None:
        """Discard all proofs after transfers drain during server shutdown.

        Args:
            None.
        Returns:
            None.
        Async/thread-safety:
            IPC-loop-only; the caller owns transfer shutdown and CUDA mappings.
        """
        self._buffers.clear()
        self._writes.clear()

    def _invalidate(self, spans: Signature) -> None:
        for key, content in list(self._buffers.items()):
            if _overlaps(content.spans, spans):
                del self._buffers[key]
