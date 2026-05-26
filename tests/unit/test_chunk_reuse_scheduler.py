# SPDX-License-Identifier: Apache-2.0

# Standard
from typing import Any

# First Party
from daser.connector.scheduler import SchedulerConnectorMixin

BLOCK_TOKENS = 4


class _SchedulerProbe(SchedulerConnectorMixin):
    """Minimal scheduler-role connector for chunk reuse credit tests."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self._runtime_config_ready = True
        self._block_tokens = BLOCK_TOKENS
        self._model_id = "m"
        self._req_tokens: dict[str, list[int]] = {}
        self._pending_loads: dict[str, Any] = {}
        self._pending_alloc: dict[str, Any] = {}
        self._ipc_sync = self
        self._chunks = chunks

    def lookup(self, tokens: list[int], model_id: str) -> list[dict[str, Any]]:
        """Return the configured chunks for a scheduler lookup.

        Args:
            tokens: prompt tokens supplied by the scheduler.
            model_id: served model identifier supplied by the scheduler.

        Returns:
            Configured server-style chunk dictionaries.
        """
        del tokens, model_id
        return self._chunks

    def _refresh_runtime_config(self) -> None:
        """No-op; tests seed runtime config directly."""
        return

    @property
    def pending_loads(self) -> dict[str, Any]:
        """Return scheduler pending load state for assertions."""
        return self._pending_loads


class _Request:
    """Minimal vLLM request stand-in."""

    request_id = "req"
    prompt_token_ids = list(range(12))
    kv_transfer_params = {"daser_skip_save": True}


def test_single_non_prefix_chunk_does_not_credit_missing_prefix() -> None:
    """A lone chunk hit can only provide external tokens at its target offset."""
    connector = _SchedulerProbe(
        chunks=[
            {
                "chunk_key": "doc",
                "start_slot": 5,
                "num_slots": 1,
                "file_offset": 160,
                "token_count": 4,
                "target_token_start": 4,
                "pos_offset": 4,
            }
        ]
    )

    assert connector.get_num_new_matched_tokens(_Request(), 0) == (0, False)
    assert connector.pending_loads == {}


def test_single_prefix_chunk_still_credits_external_suffix() -> None:
    """A single prefix chunk still credits the contiguous uncached suffix."""
    connector = _SchedulerProbe(
        chunks=[
            {
                "chunk_key": "prefix",
                "start_slot": 1,
                "num_slots": 2,
                "file_offset": 32,
                "token_count": 8,
                "target_token_start": 0,
                "pos_offset": 0,
            }
        ]
    )

    assert connector.get_num_new_matched_tokens(_Request(), 4) == (4, False)
    assert connector.pending_loads["req"]["chunk_key"] == "prefix"
