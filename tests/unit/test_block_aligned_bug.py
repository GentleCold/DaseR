# SPDX-License-Identifier: Apache-2.0
"""Unit tests for block-aligned token boundary bug.

Run with:
    python -m pytest tests/unit/test_block_aligned_bug.py -xvs
"""

# Third Party
import pytest

# First Party
from daser.connector.daser_connector import DaserConnector
from daser.connector.helpers import PendingStore, hash_tokens


class TestTokeniseAndTruncateBug:
    def test_exact_block_aligned_length_not_multiple_of_block_tokens(self):
        from benchmarks.bench_common import tokenise_and_truncate

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(16))

        tokenizer = MockTokenizer()
        prompts = ["x"]
        max_tokens = 200
        block_tokens = 16

        result = tokenise_and_truncate(prompts, tokenizer, max_tokens, block_tokens)

        assert len(result[0]) % block_tokens != 0, (
            f"Expected length % {block_tokens} != 0 (not a multiple), "
            f"got {len(result[0])} which is a multiple of {block_tokens}. "
            "A prompt with exactly block_tokens should NOT remain block-aligned."
        )

    def test_block_aligned_plus_one_unchanged(self):
        from benchmarks.bench_common import tokenise_and_truncate

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(17))

        tokenizer = MockTokenizer()
        prompts = ["x"]
        max_tokens = 200
        block_tokens = 16

        result = tokenise_and_truncate(prompts, tokenizer, max_tokens, block_tokens)

        assert len(result[0]) == 17, f"Expected length == 17, got {len(result[0])}"
        assert 17 % 16 != 0, "17 should not be a multiple of 16"


class TestGetNumNewMatchedTokensBug:
    def test_extra_tokens_equal_to_available_returns_zero(self):
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def lookup(self, prefix, model_id):
                return [
                    {
                        "chunk_key": hash_tokens(prefix),
                        "start_slot": 0,
                        "num_slots": 1,
                        "file_offset": 0,
                        "token_count": len(prefix),
                    }
                ]

        class MockDaserConnector(DaserConnector):
            def __init__(self):
                self._block_tokens = BLOCK_TOKENS
                self._socket_path = "/tmp/test.sock"
                self._store_path = "/tmp/test.store"
                self._slot_size = 2359296
                self._model_id = "test"
                self._ipc_sync = MockIPCClientSync()
                self._pending_loads = {}
                self._pending_alloc = {}
                self._req_tokens = {}

        connector = MockDaserConnector()

        request = MockRequest("test-req-1", list(range(16)))

        num_external_tokens, is_async = connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )

        assert num_external_tokens == 15, (
            f"Expected num_external_tokens == 15 when extra_tokens == available "
            f"(both are 16), but got {num_external_tokens}. "
            "DaseR follows LMCache by moving full KV blocks while reporting "
            "one fewer external token for full-prompt hits."
        )

    def test_extra_tokens_less_than_available_returns_correct_count(self):
        """When extra_tokens < available, should return extra_tokens normally."""
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def lookup(self, prefix, model_id):
                return [
                    {
                        "chunk_key": hash_tokens(prefix),
                        "start_slot": 0,
                        "num_slots": 1,
                        "file_offset": 0,
                        "token_count": len(prefix),
                    }
                ]

        class MockDaserConnector(DaserConnector):
            def __init__(self):
                self._block_tokens = BLOCK_TOKENS
                self._socket_path = "/tmp/test.sock"
                self._store_path = "/tmp/test.store"
                self._slot_size = 2359296
                self._model_id = "test"
                self._ipc_sync = MockIPCClientSync()
                self._pending_loads = {}
                self._pending_alloc = {}
                self._req_tokens = {}

        connector = MockDaserConnector()

        request = MockRequest("test-req-1", list(range(32)))

        num_external_tokens, is_async = connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )

        assert 0 < num_external_tokens < 32, (
            f"Expected 0 < num_external_tokens < 32, got {num_external_tokens}"
        )

    def test_partial_chunked_prefill_does_not_store_short_prefix(self):
        """A chunked prefill step must not publish a partial prompt prefix."""
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def __init__(self):
                self.alloc_calls = []

            def alloc_chunk(self, chunk_key, token_count, model_id):
                self.alloc_calls.append((chunk_key, token_count, model_id))
                return {
                    "start_slot": 39,
                    "num_slots": token_count // BLOCK_TOKENS,
                    "file_offset": 39 * 2359296,
                    "pos_offset": 0,
                }

        mock_ipc = MockIPCClientSync()

        class MockDaserConnector(DaserConnector):
            def __init__(self):
                self._block_tokens = BLOCK_TOKENS
                self._socket_path = "/tmp/test.sock"
                self._store_path = "/tmp/test.store"
                self._slot_size = 2359296
                self._model_id = "test"
                self._ipc_sync = mock_ipc
                self._pending_loads = {}
                self._pending_alloc = {
                    "test-req-1": PendingStore(
                        chunk_key=hash_tokens(list(range(624))),
                        token_count=624,
                    )
                }
                self._pending_stores = {}
                self._req_tokens = {"test-req-1": list(range(630))}

        class MockBlock:
            def __init__(self, block_id: int):
                self.block_id = block_id

        class MockBlocks:
            blocks = ([MockBlock(i) for i in range(6)],)

        class MockSchedulerOutput:
            num_scheduled_tokens = {"test-req-1": 96}

        connector = MockDaserConnector()
        request = MockRequest("test-req-1", list(range(630)))

        connector.update_state_after_alloc(request, MockBlocks(), num_external_tokens=0)
        meta = connector.build_connector_meta(MockSchedulerOutput())

        assert meta.reqs_to_store == {}
        assert mock_ipc.alloc_calls == []

    def test_full_chunked_prefill_step_stores_aligned_prompt(self):
        """A full scheduled prompt prefill stores the block-aligned prefix."""
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def __init__(self):
                self.alloc_calls = []

            def alloc_chunk(self, chunk_key, token_count, model_id):
                self.alloc_calls.append((chunk_key, token_count, model_id))
                return {
                    "start_slot": 39,
                    "num_slots": token_count // BLOCK_TOKENS,
                    "file_offset": 39 * 2359296,
                    "pos_offset": 0,
                }

        mock_ipc = MockIPCClientSync()

        class MockDaserConnector(DaserConnector):
            def __init__(self):
                self._block_tokens = BLOCK_TOKENS
                self._socket_path = "/tmp/test.sock"
                self._store_path = "/tmp/test.store"
                self._slot_size = 2359296
                self._model_id = "test"
                self._ipc_sync = mock_ipc
                self._pending_loads = {}
                self._pending_alloc = {
                    "test-req-1": PendingStore(
                        chunk_key=hash_tokens(list(range(256))),
                        token_count=256,
                    )
                }
                self._pending_stores = {}
                self._req_tokens = {"test-req-1": list(range(260))}

        class MockBlock:
            def __init__(self, block_id: int):
                self.block_id = block_id

        class MockBlocks:
            blocks = ([MockBlock(i) for i in range(16)],)

        class MockSchedulerOutput:
            num_scheduled_tokens = {"test-req-1": 256}

        connector = MockDaserConnector()
        request = MockRequest("test-req-1", list(range(260)))

        connector.update_state_after_alloc(request, MockBlocks(), num_external_tokens=0)
        meta = connector.build_connector_meta(MockSchedulerOutput())

        store_spec = meta.reqs_to_store["test-req-1"]
        expected_key = hash_tokens(list(range(256)))

        assert store_spec.token_count == 256
        assert store_spec.num_slots == 16
        assert store_spec.chunk_key == expected_key
        assert store_spec.block_ids == list(range(16))
        assert mock_ipc.alloc_calls == [(expected_key, 256, "test")]

    def test_chunked_prefill_completion_stores_full_aligned_prompt(self):
        """A multi-step chunked prefill stores after the aligned prefix completes."""
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids
                self.num_computed_tokens = 0

        class MockIPCClientSync:
            def __init__(self):
                self.alloc_calls = []

            def alloc_chunk(self, chunk_key, token_count, model_id):
                self.alloc_calls.append((chunk_key, token_count, model_id))
                return {
                    "start_slot": 39,
                    "num_slots": token_count // BLOCK_TOKENS,
                    "file_offset": 39 * 2359296,
                    "pos_offset": 0,
                }

        mock_ipc = MockIPCClientSync()
        tokens = list(range(630))

        class MockDaserConnector(DaserConnector):
            def __init__(self):
                self._block_tokens = BLOCK_TOKENS
                self._socket_path = "/tmp/test.sock"
                self._store_path = "/tmp/test.store"
                self._slot_size = 2359296
                self._model_id = "test"
                self._ipc_sync = mock_ipc
                self._pending_loads = {}
                self._pending_alloc = {
                    "test-req-1": PendingStore(
                        chunk_key=hash_tokens(tokens[:624]),
                        token_count=624,
                    )
                }
                self._pending_stores = {}
                self._req_tokens = {"test-req-1": tokens}

        class MockBlock:
            def __init__(self, block_id: int):
                self.block_id = block_id

        class MockBlocks:
            def __init__(self, block_ids: list[int]):
                self.blocks = ([MockBlock(i) for i in block_ids],)

        class MockSchedulerOutput:
            def __init__(
                self,
                scheduled_tokens: int,
                new_block_ids: list[int] | None = None,
                num_computed_tokens: int = 0,
            ):
                self.num_scheduled_tokens = {"test-req-1": scheduled_tokens}

                class MockCachedRequestData:
                    pass

                self.scheduled_cached_reqs = MockCachedRequestData()
                self.scheduled_cached_reqs.req_ids = (
                    ["test-req-1"] if new_block_ids is not None else []
                )
                self.scheduled_cached_reqs.resumed_req_ids = set()
                self.scheduled_cached_reqs.new_block_ids = (
                    [(new_block_ids,)] if new_block_ids is not None else []
                )
                self.scheduled_cached_reqs.num_computed_tokens = (
                    [num_computed_tokens] if new_block_ids is not None else []
                )

        connector = MockDaserConnector()
        request = MockRequest("test-req-1", tokens)

        connector.update_state_after_alloc(
            request,
            MockBlocks(list(range(6))),
            num_external_tokens=0,
        )
        meta = connector.build_connector_meta(MockSchedulerOutput(96))

        assert meta.reqs_to_store == {}
        assert mock_ipc.alloc_calls == []

        request.num_computed_tokens = 96
        meta = connector.build_connector_meta(
            MockSchedulerOutput(
                528,
                new_block_ids=list(range(6, 39)),
                num_computed_tokens=96,
            )
        )

        store_spec = meta.reqs_to_store["test-req-1"]
        expected_key = hash_tokens(tokens[:624])

        assert store_spec.token_count == 624
        assert store_spec.num_slots == 39
        assert store_spec.chunk_key == expected_key
        assert store_spec.block_ids == list(range(39))
        assert mock_ipc.alloc_calls == [(expected_key, 624, "test")]

    def test_extra_tokens_equals_available_exact_aligned_case(self):
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def lookup(self, prefix, model_id):
                return [
                    {
                        "chunk_key": hash_tokens(prefix),
                        "start_slot": 0,
                        "num_slots": 2,
                        "file_offset": 0,
                        "token_count": len(prefix),
                    }
                ]

        class MockDaserConnector(DaserConnector):
            def __init__(self):
                self._block_tokens = BLOCK_TOKENS
                self._socket_path = "/tmp/test.sock"
                self._store_path = "/tmp/test.store"
                self._slot_size = 2359296
                self._model_id = "test"
                self._ipc_sync = MockIPCClientSync()
                self._pending_loads = {}
                self._pending_alloc = {}
                self._req_tokens = {}

        connector = MockDaserConnector()

        request = MockRequest("test-req-1", list(range(32)))

        num_external_tokens, is_async = connector.get_num_new_matched_tokens(
            request, num_computed_tokens=0
        )

        assert num_external_tokens == 31, (
            f"Expected when available=32 and block_tokens=16, "
            f"got {num_external_tokens}. "
            "This follows LMCache by subtracting only the final token."
        )

    def test_trim_external_window_keeps_full_block_for_lmcache_style_minus_one(self):
        from daser.connector.scheduler import _trim_chunk_to_external_window

        chunk = {
            "chunk_key": "k",
            "start_slot": 0,
            "num_slots": 2,
            "file_offset": 0,
            "token_count": 32,
            "target_token_start": 0,
        }

        ok = _trim_chunk_to_external_window(
            chunk=chunk,
            block_ids=[10, 11],
            external_start=0,
            num_external_tokens=31,
            block_tokens=16,
            slot_size=100,
        )

        assert ok
        assert chunk["num_slots"] == 2
        assert chunk["token_count"] == 32
        assert chunk["block_ids"] == [10, 11]


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
