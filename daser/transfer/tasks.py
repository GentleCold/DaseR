# SPDX-License-Identifier: Apache-2.0

"""Cancellation-safe await helpers shared by transfer backends."""

# Standard
import asyncio
from collections.abc import Awaitable, Sequence
from typing import Any


async def settle(awaitables: Sequence[Awaitable[Any]]) -> None:
    """Wait for every awaitable, deferring cancellation until all ended.

    Cancelling an executor await or a copy poll does not stop the underlying
    IO or DMA, so buffers must stay owned until the real work ends.

    Args:
        awaitables: Futures or tasks to wait for; none is cancelled.

    Raises:
        BaseException: The first failure among ``awaitables``, else a
            cancellation of the caller received while waiting.
    """
    if not awaitables:
        return
    # Most data-plane calls own one copy or IO future. A one-element gather
    # adds a callback and event-loop turn without extending buffer lifetime.
    # Normalize a lone coroutine too: the public API accepts any Awaitable.
    single = len(awaitables) == 1
    pending = (
        asyncio.ensure_future(awaitables[0])
        if single
        else asyncio.gather(*awaitables, return_exceptions=True)
    )
    cancellation: asyncio.CancelledError | None = None
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError as exc:
            cancellation = exc
    results = (pending.result(),) if single else pending.result()
    for result in results:
        if isinstance(result, BaseException):
            raise result
    if cancellation is not None:
        raise cancellation


async def wait_shared(futures: Sequence[asyncio.Future[Any]]) -> None:
    """Wait for shared work without cancelling its owner when this caller leaves.

    Args:
        futures: Owner-managed IO, lease or promotion futures.

    Returns:
        None after success; the first dependency failure propagates.

    Async/thread-safety:
        Use on the futures' event loop. Caller cancellation is immediate and
        never cancels dependencies. A detached aggregate observes later errors.
    """
    if not futures:
        return
    pending = futures[0] if len(futures) == 1 else asyncio.gather(*futures)
    try:
        # One shield covers the aggregate; wrapping every dependency creates
        # avoidable callbacks under concurrent, overlapping prefetch loads.
        await asyncio.shield(pending)
    except asyncio.CancelledError:
        pending.add_done_callback(consume_exception)
        raise


def resolve(future: asyncio.Future[None], error: BaseException | None) -> None:
    """Complete ``future`` with ``error`` (or success) if still pending.

    Args:
        future: Handle future to complete.
        error: Failure to report; a cancellation cancels the future.
    """
    if future.done():
        return
    if error is None:
        future.set_result(None)
    elif isinstance(error, asyncio.CancelledError):
        future.cancel()
    else:
        future.set_exception(error)


def consume_exception(future: asyncio.Future[Any]) -> None:
    """Done callback marking a handle future's exception as retrieved.

    Args:
        future: Completed handle future; ``drain`` reports its error.
    """
    if not future.cancelled():
        future.exception()
