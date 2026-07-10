# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.replacement.lru import LRUReplacementPolicy
from daser.replacement.prefix_aware_lru import PrefixAwareLRUReplacementPolicy


def test_lru_policy_evicts_least_recently_used_key() -> None:
    """LRU policy chooses the oldest untouched key first."""
    policy = LRUReplacementPolicy[str]()

    policy.insert("a")
    policy.insert("b")
    policy.access("a")

    assert policy.evict() == "b"
    assert policy.evict() == "a"
    assert policy.evict() is None


def test_lru_policy_remove_disables_future_eviction() -> None:
    """Removed keys do not appear in future eviction decisions."""
    policy = LRUReplacementPolicy[str]()

    policy.insert("a")
    policy.insert("b")
    policy.remove("a")

    assert policy.evict() == "b"
    assert policy.evict() is None


def test_prefix_aware_lru_evicts_request_suffix_before_prefix() -> None:
    """Prefix-aware LRU keeps earlier request slots newer than suffix slots."""
    policy = PrefixAwareLRUReplacementPolicy[str]()

    for index, key in enumerate(("a", "b", "c")):
        policy.insert_prefix(key, ("req", 0, 3), index)

    assert policy.evict() == "c"
    assert policy.evict() == "b"
    assert policy.evict() == "a"


def test_prefix_aware_lru_prefers_older_request_before_newer_suffix() -> None:
    """Request recency remains the primary LRU order."""
    policy = PrefixAwareLRUReplacementPolicy[str]()

    for index, key in enumerate(("a0", "b0")):
        policy.insert_prefix(key, ("req0", 0, 2), index)
    for index, key in enumerate(("a1", "b1")):
        policy.insert_prefix(key, ("req1", 2, 2), index)

    assert policy.evict() == "b0"
    assert policy.evict() == "a0"
    assert policy.evict() == "b1"
    assert policy.evict() == "a1"
