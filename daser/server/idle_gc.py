# SPDX-License-Identifier: Apache-2.0

"""Schedule cyclic GC only at idle points in the DaseR event loop."""

import asyncio
from collections.abc import Callable
import gc

from starlette.types import ASGIApp, Receive, Scope, Send


class IdleGCMiddleware:
    """Defer automatic cyclic GC while HTTP, IPC or transfer work is active.

    Args:
        app: Wrapped ASGI application, including its lifespan protocol.
        is_idle: Public IPC probe including requests and background transfers.
        interval_s: Seconds between idle collection opportunities.

    Async/thread-safety:
        Owns interpreter-wide automatic GC during the ASGI lifespan. Request
        counters, the idle probe and explicit collection run on one asyncio
        loop; there is no await between checking idleness and collecting.
        Executor threads therefore cannot trigger automatic cyclic GC. Explicit
        ``gc.collect`` calls elsewhere and reference-count destruction remain
        outside this policy. Continuous traffic can defer collection forever;
        newly arriving requests may queue behind a collection already running.
    """

    def __init__(
        self,
        app: ASGIApp,
        is_idle: Callable[[], bool],
        interval_s: float = 0.1,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("GC polling interval must be positive")
        self._app = app
        self._is_idle = is_idle
        self._interval_s = interval_s
        self._requests = 0

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Serve an ASGI scope and manage GC during application lifespan.

        Args:
            scope: ASGI connection or lifespan scope.
            receive: Async ASGI message receiver.
            send: Async ASGI message sender.

        Returns:
            None after the wrapped application finishes this scope.

        Async/thread-safety:
            All scopes must execute on the same event loop. HTTP scopes stay
            active through response transmission and ASGI background tasks.
            Cancellation and failures always release their activity count.
        """
        if scope["type"] == "lifespan":
            was_enabled = gc.isenabled()
            gc.disable()
            collector = asyncio.create_task(self._collect(), name="daser-idle-gc")
            try:
                await self._app(scope, receive, send)
            finally:
                collector.cancel()
                try:
                    await asyncio.gather(collector, return_exceptions=True)
                finally:
                    # Preserve the embedding application's original GC policy,
                    # including when startup or shutdown raises/cancels.
                    if was_enabled:
                        gc.enable()
        elif scope["type"] == "http":
            self._requests += 1
            try:
                await self._app(scope, receive, send)
            finally:
                self._requests -= 1
        else:
            await self._app(scope, receive, send)

    async def _collect(self) -> None:
        """Perform due collections only when both application planes are idle."""
        while True:
            await asyncio.sleep(self._interval_s)
            if self._requests or not self._is_idle():
                continue
            counts = gc.get_count()
            thresholds = gc.get_threshold()
            if not thresholds[0] or counts[0] <= thresholds[0]:
                continue
            # Preserve young-generation collections instead of scanning
            # every long-lived object at each idle opportunity. Explicit
            # collections advance CPython's generation counters too.
            generation = next(gen for gen in (2, 1, 0) if counts[gen] > thresholds[gen])
            gc.collect(generation)
