# SPDX-License-Identifier: Apache-2.0

"""Public scan resistance, adaptation, and allocation identity contracts."""

from daser.replacement.bip import BIPReplacementPolicy


def test_reads_protect_residents_from_a_scan() -> None:
    """All scan keys are admitted, while reused keys outlive the scan."""
    policy = BIPReplacementPolicy[str]()
    for key in ("a", "b", "c"):
        policy.insert(key)
    policy.access("b")
    for index in range(10):
        key = str(index)
        policy.insert(key)
        assert policy.evict() == key
    assert [policy.evict() for _ in range(4)] == ["c", "a", "b", None]


def test_sparse_mru_insertion_adapts_to_new_keys() -> None:
    """Periodic insertions survive earlier untouched keys without an access."""
    policy = BIPReplacementPolicy[int](mru_interval=4)
    for key in range(5):
        policy.insert(key)
    assert [policy.evict() for _ in range(5)] == [4, 2, 1, 0, 3]


def test_shared_children_do_not_reset_recency_or_advance_insertion_clock() -> None:
    """Publishing another child cannot demote an already read allocation."""
    policy = BIPReplacementPolicy[str](mru_interval=4)
    policy.insert("a")
    policy.insert("b")
    policy.access("b")
    policy.insert("b")
    policy.insert("c")
    policy.insert("d")
    assert [policy.evict() for _ in range(4)] == ["c", "a", "b", "d"]


def test_removal_allows_fresh_identity_and_interval_one_is_lru() -> None:
    """Removed keys can be inserted again and unknown accesses are harmless."""
    policy = BIPReplacementPolicy[str](mru_interval=1)
    policy.insert("a")
    policy.insert("b")
    policy.remove("a")
    policy.access("a")
    policy.insert("a")
    assert [policy.evict() for _ in range(3)] == ["b", "a", None]
