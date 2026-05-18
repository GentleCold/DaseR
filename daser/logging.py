# SPDX-License-Identifier: Apache-2.0

# Standard
import logging
import os
from typing import Optional


def init_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a logger for the given module name.

    The log level is taken from the DASER_LOG_LEVEL environment variable
    (default INFO). Callers should use component tags in messages:
    [GDS], [INDEX], [CHUNK], [IPC], [CONNECTOR].

    Args:
        name: typically __name__ of the calling module.
        level: override log level string (e.g. "DEBUG"). If None, reads
               DASER_LOG_LEVEL env var, defaulting to "INFO".

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    resolved_level = level or os.environ.get("DASER_LOG_LEVEL", "INFO")
    logger.setLevel(resolved_level.upper())
    return logger
