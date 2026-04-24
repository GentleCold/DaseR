# SPDX-License-Identifier: Apache-2.0
"""Unit tests for block-aligned token boundary bug.

Tests the fix for:
1. tokenise_and_truncate: condition >= block_tokens + 1 misses len(ids) == block_tokens
   - After fix: len(ids) == 16 gets truncated to 15, then padded to 17 (not a multiple of 16)
2. get_num_new_matched_tokens: returns all available tokens when extra_tokens == available,
   causing vLLM scheduler assert num_new_tokens > 0 to fail.
   - After fix: when extra_tokens >= available, returns (available // block_tokens) * block_tokens,
     or 0 if that equals available

Run with:
    python -m pytest tests/unit/test_block_aligned_bug.py -xvs
"""

# Third Party
import pytest

# First Party
from daser.connector.daser_connector import DaserConnector, hash_tokens


class TestTokeniseAndTruncateBug:
    """Test that tokenise_and_truncate handles exact block-aligned lengths.

    The goal is to ensure the output length is NEVER an exact multiple of block_tokens.
    """

    def test_exact_block_aligned_length_not_multiple_of_block_tokens(self):
        """When tokenized prompt length is exactly 16 (block_tokens), output should not be 16.

        The bug: old condition `>= block_tokens + 1` (i.e., >= 17) misses len(ids) == 16.
        So a prompt that tokenizes to exactly 16 tokens would keep that length,
        which later causes get_num_new_matched_tokens to return all 16 tokens,
        leaving num_new_tokens = 0 in vLLM's scheduler.

        After fix: len(ids) == 16 is truncated to 15, then padded to 17 (not a multiple of 16).
        """
        import sys
        sys.path.insert(0, "/home/ld/DaseR")
        from benchmarks.bench_e2e_daser_vs_lmcache import tokenise_and_truncate

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(16))

        tokenizer = MockTokenizer()
        prompts = ["x"]
        max_tokens = 200
        block_tokens = 16

        result = tokenise_and_truncate(prompts, tokenizer, max_tokens, block_tokens)

        assert len(result[0]) % block_tokens != 0, (
            f"Expected length % {block_tokens} != 0 (not a multiple), got {len(result[0])} "
            f"which is a multiple of {block_tokens}. "
            "A prompt with exactly block_tokens should NOT remain block-aligned."
        )

    def test_block_aligned_plus_one_unchanged(self):
        """When tokenized prompt length is 17, it should remain unchanged (not a multiple of 16)."""
        import sys
        sys.path.insert(0, "/home/ld/DaseR")
        from benchmarks.bench_e2e_daser_vs_lmcache import tokenise_and_truncate

        class MockTokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(17))

        tokenizer = MockTokenizer()
        prompts = ["x"]
        max_tokens = 200
        block_tokens = 16

        result = tokenise_and_truncate(prompts, tokenizer, max_tokens, block_tokens)

        assert len(result[0]) == 17, (
            f"Expected length == 17, got {len(result[0])}"
        )
        assert 17 % 16 != 0, "17 should not be a multiple of 16"


class TestGetNumNewMatchedTokensBug:
    """Test that get_num_new_matched_tokens doesn't return all available tokens.

    The core issue: when extra_tokens == available (e.g., both are 16),
    vLLM scheduler computes num_new_tokens = 0 and crashes with assert num_new_tokens > 0.
    """

    def test_extra_tokens_equal_to_available_returns_zero(self):
        """When extra_tokens == available, should return 0 to avoid num_new_tokens == 0.

        Bug scenario:
        - prompt has 16 tokens (exact multiple of block_tokens)
        - num_computed_tokens = 0
        - cache returns token_count = 16
        - extra_tokens = 16 - 0 = 16
        - available = 16 - 0 = 16
        - OLD code returns 16, causing vLLM to compute num_new_tokens = 0 and crash
        - FIXED code should return 0 (since extra_tokens >= available, and after
          alignment adjustment extra_tokens becomes 0)
        """
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def match_and_alloc(self, prefix, store_key, model_id):
                return {
                    "chunks": [{
                        "chunk_key": hash_tokens(prefix),
                        "start_slot": 0,
                        "num_slots": 1,
                        "file_offset": 0,
                        "token_count": len(prefix),
                    }],
                    "alloc": None,
                }

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

        assert num_external_tokens == 0, (
            f"Expected num_external_tokens == 0 when extra_tokens == available "
            f"(both are 16), but got {num_external_tokens}. "
            "Returning all 16 tokens would cause vLLM to compute num_new_tokens = 0 "
            "and crash with assert num_new_tokens > 0."
        )

    def test_extra_tokens_less_than_available_returns_correct_count(self):
        """When extra_tokens < available, should return extra_tokens normally."""
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def match_and_alloc(self, prefix, store_key, model_id):
                return {
                    "chunks": [{
                        "chunk_key": hash_tokens(prefix),
                        "start_slot": 0,
                        "num_slots": 1,
                        "file_offset": 0,
                        "token_count": len(prefix),
                    }],
                    "alloc": None,
                }

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

    def test_extra_tokens_equals_available_exact_aligned_case(self):
        """Test the exact edge case where aligned extra_tokens == available.

        Scenario: 32 tokens, block_tokens=16
        - available = 32
        - aligned = (32 // 16) * 16 = 32
        - If cache returns token_count = 32, extra_tokens = 32
        - available = 32, so extra_tokens >= available is True
        - extra_tokens = (32 // 16) * 16 = 32
        - extra_tokens == available is True, so extra_tokens -= 16 = 16
        - 16 > 0, so returns 16

        This leaves 16 tokens for vLLM to compute (32 - 16 = 16), which is > 0.
        """
        BLOCK_TOKENS = 16

        class MockRequest:
            def __init__(self, request_id: str, token_ids: list[int]):
                self.request_id = request_id
                self.prompt_token_ids = token_ids

        class MockIPCClientSync:
            def match_and_alloc(self, prefix, store_key, model_id):
                return {
                    "chunks": [{
                        "chunk_key": hash_tokens(prefix),
                        "start_slot": 0,
                        "num_slots": 2,
                        "file_offset": 0,
                        "token_count": len(prefix),
                    }],
                    "alloc": None,
                }

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

        assert num_external_tokens == 16, (
            f"Expected num_external_tokens == 16 when available=32 and block_tokens=16, "
            f"got {num_external_tokens}. This ensures 16 tokens remain for vLLM to compute."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])