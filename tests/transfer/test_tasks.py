# SPDX-License-Identifier: Apache-2.0
"""Owned asynchronous work must finish before cancellation releases its buffers."""

import asyncio

import pytest

from daser.transfer.tasks import settle


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("cancellations", [0, 1, 3])
async def test_settle_defers_cancellation_until_every_owner_finishes(
    count: int, cancellations: int
) -> None:
    """Single and batched work retain ownership through repeated cancellation."""
    futures = [asyncio.get_running_loop().create_future() for _ in range(count)]
    draining = asyncio.create_task(settle(futures))
    await asyncio.sleep(0)
    for _ in range(cancellations):
        draining.cancel()
        await asyncio.sleep(0)
        assert not draining.done()
    assert all(not future.cancelled() for future in futures)
    for future in futures:
        future.set_result(None)
    if cancellations:
        with pytest.raises(asyncio.CancelledError):
            await draining
    else:
        await draining


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 2])
async def test_settle_drains_siblings_and_prioritizes_io_failure(count: int) -> None:
    """Cancellation must neither hide an IO error nor detach unfinished siblings."""
    futures = [asyncio.get_running_loop().create_future() for _ in range(count)]
    draining = asyncio.create_task(settle(futures))
    await asyncio.sleep(0)
    draining.cancel()
    await asyncio.sleep(0)
    futures[0].set_exception(OSError("injected IO failure"))
    await asyncio.sleep(0)
    for future in futures[1:]:
        assert not draining.done()
        future.set_result(None)
    with pytest.raises(OSError, match="injected IO failure"):
        await draining


@pytest.mark.asyncio
async def test_settle_accepts_a_lone_coroutine() -> None:
    """The optimized path keeps the public Awaitable input contract."""
    finished = asyncio.Event()

    async def work() -> None:
        await asyncio.sleep(0)
        finished.set()

    await settle([work()])
    assert finished.is_set()
