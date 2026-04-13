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


class PerfLogger:
    """Logger for performance metrics (latency, hit rate, throughput).

    Writes structured records to a dedicated logger at DEBUG level.
    Enable with DASER_PERF_LOG=1.

    Args:
        name: module name for the underlying logger.
    """

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(f"{name}.perf")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s PERF %(name)s %(message)s")
            )
            self._logger.addHandler(handler)
        self._logger.setLevel(logging.DEBUG)
        self._enabled = os.environ.get("DASER_PERF_LOG", "0") == "1"

    def record(self, metric: str, value: float, unit: str = "") -> None:
        """Record a single metric value.

        Args:
            metric: metric name, e.g. "gds_read_latency_ms".
            value: numeric value.
            unit: optional unit string for clarity.
        """
        if self._enabled:
            self._logger.debug("%s=%.4f%s", metric, value, unit)


def init_perf_logger(name: str) -> PerfLogger:
    """Return a PerfLogger for the given module name.

    Args:
        name: typically __name__ of the calling module.

    Returns:
        PerfLogger instance.
    """
    return PerfLogger(name)
