# SPDX-License-Identifier: Apache-2.0

# First Party
from daser.replacement.compression_density import (
    CompressionDensityReplacementPolicy,
)
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


def test_compression_density_policy_evicts_low_density_before_oldest() -> None:
    """Compression density outranks recency while ties remain oldest-first."""
    density = {"low": 1.0, "high": 2.0, "high_old": 2.0}
    policy = CompressionDensityReplacementPolicy(density.__getitem__)

    policy.insert("low")
    policy.insert("high")
    policy.insert("high_old")
    policy.access("low")

    assert policy.evict() == "low"
    assert policy.evict() == "high"
    assert policy.evict() == "high_old"
