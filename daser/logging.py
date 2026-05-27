# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
from typing import IO, Optional
import weakref

DASER_LOG_COLOR = "\033[38;2;102;178;255m"
RESET_COLOR = "\033[0m"
_DASER_HANDLERS: "weakref.WeakSet[logging.Handler]" = weakref.WeakSet()


class _DaseRFormatter(logging.Formatter):
    """Formatter that colors the DaseR log source label."""

    def __init__(self) -> None:
        super().__init__(
            f"%(asctime)s %(levelname)s {DASER_LOG_COLOR}DaseR{RESET_COLOR} "
            "%(name)s %(message)s"
        )


def init_logger(
    name: str, level: Optional[str] = None, stream: Optional[IO[str]] = None
) -> logging.Logger:
    """Return a logger for the given module name.

    The log level is taken from the DASER_LOG_LEVEL environment variable
    (default INFO). Callers should use component tags in messages:
    [GDS], [INDEX], [CHUNK], [IPC], [CONNECTOR].

    Args:
        name: typically __name__ of the calling module.
        level: override log level string (e.g. "DEBUG"). If None, reads
               DASER_LOG_LEVEL env var, defaulting to "INFO".
        stream: optional text stream for the handler. Production callers leave
            this unset; tests pass an in-memory stream.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    daser_handlers = [
        handler for handler in logger.handlers if handler in _DASER_HANDLERS
    ]
    if stream is not None:
        for handler in daser_handlers:
            logger.removeHandler(handler)
        daser_handlers = []
    if not daser_handlers:
        handler = logging.StreamHandler(stream)
        handler.setFormatter(_DaseRFormatter())
        _DASER_HANDLERS.add(handler)
        logger.addHandler(handler)

    resolved_level = level or os.environ.get("DASER_LOG_LEVEL", "INFO")
    logger.setLevel(resolved_level.upper())
    return logger
