# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per-request KV transfer flags.

Covers the contract that ``kv_transfer_params={"daser_skip_save": True}``
on the vLLM Request causes the scheduler-side connector to skip delayed
write-back allocation. Default behavior (flag absent / False) must still
track the block-aligned hash so write-back keeps working after vLLM decides
how many blocks it will compute.

Run with:
    python -m pytest tests/unit/test_skip_save_flag.py -xvs
"""

# Standard
from typing import Any, Optional

# First Party
from daser.connector.daser_connector import DaserConnector
from daser.connector.helpers import PendingStore, hash_tokens

BLOCK_TOKENS = 16


class _MockRequest:
    """Minimal stand-in for vllm.v1.request.Request.

    Only carries the attributes DaserConnector touches on the scheduler
    path so the connector logic can be exercised without depending on
    vLLM internals.
    """

    def __init__(
        self,
        request_id: str,
        token_ids: list[int],
        kv_transfer_params: Optional[dict[str, Any]] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt_token_ids = token_ids
        self.kv_transfer_params = kv_transfer_params


class _RecordingIPCClient:
    """Captures the (prefix, store_key, model_id) match_and_alloc call.

    The connector passes an empty store_key because allocation is delayed
    until ``update_state_after_alloc``. Tests assert the connector's
    pending allocation state rather than expecting this RPC to reserve
    slots immediately.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], str, str]] = []
        self.response: dict[str, Any] = {"chunks": [], "alloc": None}

    def match_and_alloc(
        self, prefix: list[int], store_key: str, model_id: str
    ) -> dict[str, Any]:
        self.calls.append((list(prefix), store_key, model_id))
        return self.response


class _MockDaserConnector(DaserConnector):
    """Test connector that bypasses vLLM init for scheduler-path testing.

    Mirrors the subclass pattern used by ``test_block_aligned_bug`` so
    private attribute initialization happens inside the class hierarchy
    and ruff's SLF001 rule stays clean.
    """

    def __init__(self, ipc: _RecordingIPCClient) -> None:
        self._block_tokens = BLOCK_TOKENS
        self._socket_path = "/tmp/test.sock"
        self._store_path = "/tmp/test.store"
        self._slot_size = 2359296
        self._model_id = "test"
        self._ipc_sync = ipc
        self._pending_loads = {}
        self._pending_stores = {}
        self._pending_alloc = {}
        self._req_tokens = {}

    @property
    def captured_alloc(self) -> dict[str, PendingStore]:
        """Expose the alloc map so tests can assert without SLF001."""
        return self._pending_alloc


class TestSkipSaveFlag:
    def test_skip_save_flag_sends_empty_store_key(self) -> None:
        """``daser_skip_save=True`` must blank the store_key (no alloc)."""
        ipc = _RecordingIPCClient()
        connector = _MockDaserConnector(ipc)
        request = _MockRequest(
            "req-skip",
            list(range(BLOCK_TOKENS * 2)),
            kv_transfer_params={"daser_skip_save": True},
        )

        num_external, is_async = connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )

        assert (num_external, is_async) == (0, False)
        assert len(ipc.calls) == 1
        _, store_key, _ = ipc.calls[0]
        assert store_key == "", (
            "skip-save requests must not pre-allocate a store chunk; "
            f"got store_key={store_key!r}"
        )
        assert request.request_id not in connector.captured_alloc

    def test_default_behavior_records_pending_block_aligned_alloc(self) -> None:
        """No flag → connector preserves write-back allocation intent."""
        tokens = list(range(BLOCK_TOKENS * 2))
        ipc = _RecordingIPCClient()
        connector = _MockDaserConnector(ipc)
        request = _MockRequest("req-default", tokens)

        connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        assert len(ipc.calls) == 1
        _, store_key, _ = ipc.calls[0]
        assert store_key == ""
        full_aligned = (len(tokens) // BLOCK_TOKENS) * BLOCK_TOKENS
        pending_store = connector.captured_alloc[request.request_id]
        assert pending_store.chunk_key == hash_tokens(tokens[:full_aligned])
        assert pending_store.token_count == full_aligned

    def test_skip_save_false_records_pending_block_aligned_alloc(self) -> None:
        """Explicit False is treated like absent — keep storing."""
        tokens = list(range(BLOCK_TOKENS * 2))
        ipc = _RecordingIPCClient()
        connector = _MockDaserConnector(ipc)
        request = _MockRequest(
            "req-explicit-false",
            tokens,
            kv_transfer_params={"daser_skip_save": False},
        )

        connector.get_num_new_matched_tokens(request, num_computed_tokens=0)

        _, store_key, _ = ipc.calls[0]
        assert store_key == ""
        full_aligned = (len(tokens) // BLOCK_TOKENS) * BLOCK_TOKENS
        pending_store = connector.captured_alloc[request.request_id]
        assert pending_store.chunk_key == hash_tokens(tokens[:full_aligned])
        assert pending_store.token_count == full_aligned

    def test_missing_attribute_does_not_crash(self) -> None:
        """Older Request objects without the attribute must still work."""
        tokens = list(range(BLOCK_TOKENS * 2))
        ipc = _RecordingIPCClient()
        connector = _MockDaserConnector(ipc)

        class _LegacyRequest:
            def __init__(self) -> None:
                self.request_id = "legacy"
                self.prompt_token_ids = tokens

        connector.get_num_new_matched_tokens(_LegacyRequest(), num_computed_tokens=0)

        _, store_key, _ = ipc.calls[0]
        assert store_key == ""
        full_aligned = (len(tokens) // BLOCK_TOKENS) * BLOCK_TOKENS
        pending_store = connector.captured_alloc["legacy"]
        assert pending_store.chunk_key == hash_tokens(tokens[:full_aligned])
        assert pending_store.token_count == full_aligned
