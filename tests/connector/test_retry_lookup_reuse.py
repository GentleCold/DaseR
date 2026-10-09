# SPDX-License-Identifier: Apache-2.0
"""Reuse of DaseR lookups across vLLM block-allocation retries."""

from types import SimpleNamespace

from daser.connector.scheduler.lifecycle import RequestLifecycle

PREFIX = list(range(8))


class _EpochIPC:
    """IPC stub holding one 8-token chunk and a server index epoch."""

    def __init__(self) -> None:
        self.lookups = 0
        self.epoch_checks = 0
        self.epoch = 0
        self.present = True

    def lookup(
        self,
        tokens,
        model_id,
        external_prefix_queries=None,
        num_computed_tokens=0,
        defer_pending=None,
    ):
        return self.lookup_versioned(
            tokens, model_id, external_prefix_queries, num_computed_tokens
        )[0]

    def lookup_versioned(
        self,
        tokens,
        model_id,
        external_prefix_queries=None,
        num_computed_tokens=0,
        defer_pending=None,
    ):
        del model_id, external_prefix_queries, num_computed_tokens, defer_pending
        self.lookups += 1
        if not self.present or list(tokens[:8]) != PREFIX:
            return [], self.epoch
        chunk = {
            "chunk_key": "doc",
            "start_slot": 0,
            "num_slots": 2,
            "file_offset": 0,
            "token_count": 8,
            "target_token_start": 0,
            "pos_offset": 0,
        }
        return [chunk], self.epoch

    def index_epoch(self) -> int:
        self.epoch_checks += 1
        return self.epoch

    def evict(self) -> None:
        """Remove the chunk; the server advances its epoch on removal."""
        self.present = False
        self.epoch += 1


class _UnversionedIPC(_EpochIPC):
    """IPC stub whose client reports no index epoch."""

    lookup_versioned = None  # type: ignore[assignment]

    def lookup(
        self,
        tokens,
        model_id,
        external_prefix_queries=None,
        num_computed_tokens=0,
        defer_pending=None,
    ):
        return _EpochIPC.lookup_versioned(self, tokens, model_id)[0]


def _lifecycle(ipc: _EpochIPC) -> RequestLifecycle:
    return RequestLifecycle(
        ipc_client=ipc,
        block_tokens=4,
        slot_size=32,
        model_id="model",
        cache_reuse_mode="chunk",
        runtime_config_ready=True,
    )


def _request(req_id: str = "a") -> SimpleNamespace:
    return SimpleNamespace(
        request_id=req_id,
        prompt_token_ids=PREFIX + [100] * 4,
        kv_transfer_params={"daser_skip_save": True},
    )


def _alloc(lifecycle: RequestLifecycle, request: SimpleNamespace) -> None:
    blocks = SimpleNamespace(blocks=[[SimpleNamespace(block_id=i) for i in range(3)]])
    lifecycle.update_state_after_alloc(request, blocks, 8)


def _next_step(lifecycle: RequestLifecycle) -> None:
    """End the current scheduler step."""
    lifecycle.build_connector_meta(SimpleNamespace(num_scheduled_tokens={}))


def test_allocation_retry_reuses_lookup_while_epoch_unchanged() -> None:
    """A next-step retry with an unchanged epoch skips the token lookup."""
    ipc = _EpochIPC()
    lifecycle = _lifecycle(ipc)
    request = _request()
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (8, True)
    # vLLM could not allocate blocks and asks again next step.
    _next_step(lifecycle)
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (8, True)
    assert (ipc.lookups, ipc.epoch_checks) == (1, 1)


def test_epoch_is_read_once_per_scheduler_step() -> None:
    """Retries in one step share one epoch RPC; a lookup reply supplies it free."""
    ipc = _EpochIPC()
    lifecycle = _lifecycle(ipc)
    first, second = _request("a"), _request("b")
    lifecycle.get_num_new_matched_tokens(first, 0)
    lifecycle.get_num_new_matched_tokens(second, 0)
    _next_step(lifecycle)
    lifecycle.get_num_new_matched_tokens(first, 0)
    lifecycle.get_num_new_matched_tokens(second, 0)
    assert (ipc.lookups, ipc.epoch_checks) == (2, 1)
    # The same step's lookup reply already carries the epoch.
    _next_step(lifecycle)
    lifecycle.get_num_new_matched_tokens(_request("c"), 0)
    lifecycle.get_num_new_matched_tokens(first, 0)
    assert (ipc.lookups, ipc.epoch_checks) == (3, 1)


def test_removal_between_retries_forces_a_fresh_lookup() -> None:
    """An advanced epoch invalidates the reused result, so no stale hit loads."""
    ipc = _EpochIPC()
    lifecycle = _lifecycle(ipc)
    request = _request()
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (8, True)
    _next_step(lifecycle)
    ipc.evict()
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (0, False)
    assert ipc.lookups == 2


def test_changed_window_forces_a_fresh_lookup() -> None:
    """A retry whose local prefix hit moved is not an identical lookup."""
    ipc = _EpochIPC()
    lifecycle = _lifecycle(ipc)
    request = _request()
    lifecycle.get_num_new_matched_tokens(request, 0)
    lifecycle.get_num_new_matched_tokens(request, 4)
    assert ipc.lookups == 2
    assert ipc.epoch_checks == 0


def test_successful_allocation_ends_reuse() -> None:
    """After vLLM allocates, the next call is a new decision and looks up."""
    ipc = _EpochIPC()
    lifecycle = _lifecycle(ipc)
    request = _request()
    lifecycle.get_num_new_matched_tokens(request, 0)
    _alloc(lifecycle, request)
    lifecycle.get_num_new_matched_tokens(request, 0)
    assert ipc.lookups == 2


def test_reused_result_is_not_shared_with_pending_loads() -> None:
    """Trimming a pending load must not change the result kept for retries."""
    ipc = _EpochIPC()
    lifecycle = _lifecycle(ipc)
    request = _request()
    lifecycle.get_num_new_matched_tokens(request, 0)
    _alloc(lifecycle, request)
    retry = _request("b")
    lifecycle.get_num_new_matched_tokens(retry, 0)
    assert lifecycle.get_num_new_matched_tokens(retry, 0) == (8, True)


def test_client_without_epoch_looks_up_every_retry() -> None:
    """Without an index epoch nothing proves a result fresh, so none is reused."""
    ipc = _UnversionedIPC()
    lifecycle = _lifecycle(ipc)
    request = _request()
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (8, True)
    _next_step(lifecycle)
    assert lifecycle.get_num_new_matched_tokens(request, 0) == (8, True)
    assert (ipc.lookups, ipc.epoch_checks) == (2, 0)
