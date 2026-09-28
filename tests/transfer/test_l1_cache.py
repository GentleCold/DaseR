# SPDX-License-Identifier: Apache-2.0

"""Public L1 ownership behavior across shared slices and eviction."""

import pytest

pytest.importorskip("cupy")

from daser.replacement import LRUReplacementPolicy, ReplacementPolicy
from daser.replacement.bip import BIPReplacementPolicy
from daser.transfer.iouring.l1_cache import L1Cache


def test_resident_identity_tracks_children_and_capacity_eviction() -> None:
    """Shared allocation children remain independently resident and evictable."""
    cache = L1Cache(8192, 4096, lambda _key, _data: False)
    try:
        parent = cache.reserve_untracked(8192)
        assert parent is not None
        first = parent.subslice(0, 4096)
        second = parent.subslice(4096, 4096)
        parent.close()
        assert not cache.contains_slice(first)
        cache.put_reserved_group([((0, 4096), first), ((4096, 4096), second)])
        assert cache.resident_slice_ids() == {id(first), id(second)}
        assert cache.contains_slice(first) and cache.contains_slice(second)

        cache.drop_overlapping(0, 4096)
        assert not cache.contains_slice(first)
        assert cache.contains_slice(second)
        held = cache.reserve_untracked(8192)
        assert held is not None
        assert not cache.contains_slice(second)
        assert cache.resident_slice_ids() == set()
        held.close()
    finally:
        cache.close()


@pytest.mark.parametrize("policy_type", [LRUReplacementPolicy, BIPReplacementPolicy])
def test_hot_child_keeps_its_allocation_without_evicting_unrelated_children(
    policy_type: type[ReplacementPolicy[int]],
) -> None:
    """A pool reservation reclaims one cold allocation and keeps the hot one."""
    cache = L1Cache(16384, 4096, lambda _key, _data: False, policy_type())
    try:
        first = cache.reserve_untracked(8192)
        assert first is not None
        first.view()[:] = b"a" * 4096 + b"b" * 4096
        a, b = first.subslice(0, 4096), first.subslice(4096, 4096)
        cache.put_reserved_group([((0, 4096), a), ((16384, 4096), b)])
        first.close()
        second = cache.reserve_untracked(8192)
        assert second is not None
        c, d = second.subslice(0, 4096), second.subslice(4096, 4096)
        cache.put_reserved_group([((32768, 4096), c), ((49152, 4096), d)])
        second.close()

        # Only one child is read. Its sibling still occupies the same physical
        # allocation, so dropping that sibling first would release no space.
        hits, misses = cache.resolve_subranges(0, 0, 4096)
        assert not misses
        cache.record_hits(hits)
        reserved = cache.reserve_untracked(4096)
        assert reserved is not None
        try:
            assert cache.get((0, 4096)) is a
            assert cache.get((16384, 4096)) is b
            assert bytes(b.view()) == b"b" * 4096
            assert not cache.contains_slice(c)
            assert not cache.contains_slice(d)
            assert cache.bytes_used == 8192
        finally:
            reserved.close()
    finally:
        cache.close()


@pytest.mark.parametrize("policy_type", [LRUReplacementPolicy, BIPReplacementPolicy])
def test_group_eviction_does_not_reuse_writer_held_pool_bytes(
    policy_type: type[ReplacementPolicy[int]],
) -> None:
    """Detached grouped children remain alive until the writer releases them."""
    pinned: set[int] = set()
    cache = L1Cache(8192, 4096, lambda _key, data: id(data) in pinned, policy_type())
    parent = cache.reserve_untracked(8192)
    assert parent is not None
    a, b = parent.subslice(0, 4096), parent.subslice(4096, 4096)
    cache.put_reserved_group([((0, 4096), a), ((16384, 4096), b)])
    parent.close()
    pinned.add(id(a))
    try:
        assert cache.reserve_untracked(4096) is None
        assert cache.resident_slice_ids() == set()
        a.view()[:] = b"x" * 4096
        assert bytes(a.view()) == b"x" * 4096
        pinned.clear()
        cache.release((0, 4096), a)
        reserved = cache.reserve_untracked(8192)
        assert reserved is not None
        reserved.close()
    finally:
        a.close()
        cache.close()


def test_standalone_allocations_keep_lru_order() -> None:
    """Single-allocation raw entries retain their existing LRU behavior."""
    cache = L1Cache(8192, 4096, lambda _key, _data: False)
    try:
        first = cache.reserve_or_raise((0, 4096), 4096)
        cache.put((0, 4096), first)
        second = cache.reserve_or_raise((4096, 4096), 4096)
        cache.put((4096, 4096), second)
        cache.touch((0, 4096))
        third = cache.reserve_or_raise((8192, 4096), 4096)
        cache.put((8192, 4096), third)
        assert cache.contains_slice(first)
        assert not cache.contains_slice(second)
        assert cache.contains_slice(third)
    finally:
        cache.close()


def test_accounted_charge_controls_capacity_eviction() -> None:
    """Raw-equivalent charges evict a physically smaller resident slice."""
    cache = L1Cache(8192, 4096, lambda _key, _data: False)
    try:
        first = cache.reserve_or_raise(
            (0, 4096),
            4096,
            accounted_nbytes=8192,
        )
        cache.put((0, 4096), first, accounted_nbytes=8192)
        second = cache.reserve_or_raise((4096, 4096), 4096)
        cache.put((4096, 4096), second)

        assert not cache.contains_slice(first)
        assert cache.contains_slice(second)
        assert cache.bytes_used == 4096
    finally:
        cache.close()


def test_overwrite_preserves_bytes_with_new_resident_slice_identities() -> None:
    """Detached writer-held data is not resident after preserved-tail replacement."""
    cache = L1Cache(24576, 4096, lambda _key, _data: True)
    original = cache.reserve_or_raise((0, 12288), 12288)
    original.view()[:] = b"a" * 4096 + b"b" * 4096 + b"c" * 4096
    cache.put((0, 12288), original)
    try:
        cache.drop_overlapping(4096, 4096, preserve_remainder=True)
        assert not cache.contains_slice(original)
        # The external writer still owns original bytes. Free that owner only
        # after checking that residency no longer confuses it with fragments.
        assert bytes(original.view()[4096:8192]) == b"b" * 4096
        left = cache.get((0, 4096))
        right = cache.get((8192, 4096))
        assert left is not None
        assert right is not None
        assert cache.resident_slice_ids() == {id(left), id(right)}
        assert cache.contains_slice(left) and cache.contains_slice(right)
        assert bytes(left.view()) == b"a" * 4096
        assert bytes(right.view()) == b"c" * 4096
    finally:
        original.close()
        cache.close()


@pytest.mark.parametrize("writer_held", [False, True])
def test_repeated_overwrite_reuses_fragments_without_extra_pool_capacity(
    writer_held: bool,
) -> None:
    """Preserved children survive parent close and share its charged allocation."""
    pinned: set[int] = set()
    cache = L1Cache(16384, 4096, lambda _key, data: id(data) in pinned)
    original = cache.reserve_or_raise((0, 16384), 16384)
    payload = bytes(range(256)) * 64
    original.view()[:] = payload
    original_ptr = original.ptr_at()
    allocation = original.allocation_id
    cache.put((0, 16384), original)
    if writer_held:
        pinned.add(id(original))
    try:
        # The pool is full. Interior and edge overwrites must only split
        # ownership; allocating copied fragments would fail or evict them.
        cache.drop_overlapping(4096, 4096, preserve_remainder=True)
        cache.drop_overlapping(10240, 2048, preserve_remainder=True)
        cache.drop_overlapping(0, 1024, preserve_remainder=True)
        cache.drop_overlapping(15360, 1024, preserve_remainder=True)
        ranges = [(1024, 3072), (8192, 2048), (12288, 3072)]
        assert cache.bytes_used == sum(size for _, size in ranges)
        for offset, size in ranges:
            child = cache.get((offset, size))
            assert child is not None
            assert child.allocation_id == allocation
            assert child.ptr_at() == original_ptr + offset
            assert bytes(child.view()) == payload[offset : offset + size]
        if writer_held:
            assert bytes(original.view()) == payload
            pinned.clear()
            cache.release((0, 16384), original)
        # Dropping the external owner must leave all retained children valid.
        for offset, size in ranges:
            child = cache.get((offset, size))
            assert child is not None
            assert bytes(child.view()) == payload[offset : offset + size]
        for offset, size in ranges:
            cache.drop_overlapping(offset, size)
        assert cache.bytes_used == 0
        reused = cache.reserve_untracked(16384)
        assert reused is not None
        assert reused.ptr_at() == original_ptr
        # Final child close must return the allocation exactly once.
        assert cache.reserve_untracked(4096) is None
        reused.close()
    finally:
        original.close()
        cache.close()


def test_fragment_eviction_cannot_reuse_detached_writer_allocation() -> None:
    """Even after every fragment is evicted, the old writer retains the pages."""
    pinned: set[int] = set()
    cache = L1Cache(12288, 4096, lambda _key, data: id(data) in pinned)
    original = cache.reserve_or_raise((0, 12288), 12288)
    original.view()[:] = b"x" * 12288
    cache.put((0, 12288), original)
    pinned.add(id(original))
    try:
        cache.drop_overlapping(4096, 4096, preserve_remainder=True)
        assert cache.bytes_used == 8192
        assert cache.reserve_untracked(4096) is None
        assert cache.bytes_used == 0
        assert bytes(original.view()) == b"x" * 12288
        pinned.clear()
        cache.release((0, 12288), original)
        reused = cache.reserve_untracked(12288)
        assert reused is not None
        reused.close()
    finally:
        original.close()
        cache.close()
