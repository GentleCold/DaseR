# SPDX-License-Identifier: Apache-2.0

import pytest

from daser.server.staging_content import StagingContentIndex


def _spans(offset: int = 0, target: int = 0) -> list[dict[str, int]]:
    return [{"file_offset": offset, "target_offset": target, "nbytes": 128}]


@pytest.mark.parametrize("offset", [0, 64, 127, 128])
def test_writes_invalidate_only_overlapping_sources(offset: int) -> None:
    index = StagingContentIndex()
    hit, load = index.begin_load((42, 0), _spans(), reusable=True)
    assert not hit
    index.finish_load((42, 0), load, success=True)
    write = index.begin_write(_spans(offset))
    index.end_write(write)
    hit, _ = index.begin_load((42, 0), _spans(), reusable=True)
    assert hit == (offset == 128)


@pytest.mark.parametrize("load_finishes_first", [False, True])
def test_load_started_during_write_never_publishes_stale_content(
    load_finishes_first: bool,
) -> None:
    index = StagingContentIndex()
    write = index.begin_write(_spans())
    _, load = index.begin_load((42, 0), _spans(), reusable=True)
    if load_finishes_first:
        index.finish_load((42, 0), load, success=True)
        index.end_write(write)
    else:
        index.end_write(write)
        index.finish_load((42, 0), load, success=True)
    assert not index.begin_load((42, 0), _spans(), reusable=True)[0]


def test_write_revokes_inflight_load_before_late_completion() -> None:
    index = StagingContentIndex()
    _, load = index.begin_load((42, 0), _spans(), reusable=True)
    write = index.begin_write(_spans())
    index.end_write(write)
    index.finish_load((42, 0), load, success=True)
    assert not index.begin_load((42, 0), _spans(), reusable=True)[0]


def test_failed_or_replaced_load_cannot_publish_or_remove_later_owner() -> None:
    index = StagingContentIndex()
    _, failed = index.begin_load((42, 0), _spans(), reusable=True)
    index.finish_load((42, 0), failed, success=False)
    assert not index.begin_load((42, 0), _spans(), reusable=True)[0]
    index.forget((42, 0))
    _, replacement = index.begin_load((42, 0), _spans(256), reusable=True)
    index.finish_load((42, 0), replacement, success=True)
    index.finish_load((42, 0), failed, success=False)
    assert index.begin_load((42, 0), _spans(256), reusable=True)[0]


@pytest.mark.parametrize("key,target", [((43, 0), 0), ((42, 1), 0), ((42, 0), 8)])
def test_reuse_respects_producer_ring_and_destination_layout(
    key: tuple[int, int],
    target: int,
) -> None:
    index = StagingContentIndex()
    _, load = index.begin_load((42, 0), _spans(), reusable=True)
    index.finish_load((42, 0), load, success=True)
    assert not index.begin_load(key, _spans(target=target), reusable=True)[0]


def test_ineligible_load_and_shutdown_discard_prior_content() -> None:
    index = StagingContentIndex()
    _, load = index.begin_load((42, 0), _spans(), reusable=True)
    index.finish_load((42, 0), load, success=True)
    assert index.begin_load((42, 0), _spans(), reusable=False) == (False, None)
    hit, load = index.begin_load((42, 0), _spans(), reusable=True)
    assert not hit
    index.finish_load((42, 0), load, success=True)
    index.clear()
    assert not index.begin_load((42, 0), _spans(), reusable=True)[0]
