# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.replacement.lru import LRUReplacementPolicy


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
