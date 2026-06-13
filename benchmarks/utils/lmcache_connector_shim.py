# SPDX-License-Identifier: Apache-2.0
"""Benchmark-only alias for LMCache's external MP connector."""

from __future__ import annotations

from lmcache.integration.vllm.lmcache_mp_connector import (
    LMCacheMPConnector as _ExternalLMCacheMPConnector,
)


class DaseRBenchLMCacheMPConnector(_ExternalLMCacheMPConnector):
    """Alias that bypasses vLLM's built-in ``LMCacheMPConnector`` registry entry.

    Thread-safety:
        Same as ``lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector``.
    """
