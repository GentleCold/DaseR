# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

# Third Party
import pytest

torch = pytest.importorskip("torch")

# First Party
from daser.connector.transfer import PinnedL1Cache


def _cpu_allocator(nbytes: int) -> torch.Tensor:
    """Allocate a CPU uint8 tensor for deterministic unit tests."""
    return torch.empty(nbytes, dtype=torch.uint8)


def test_insert_get_and_stats() -> None:
    cache = PinnedL1Cache(capacity_bytes=16, allocator=_cpu_allocator)

    entry = cache.reserve("a", 8, durable=False)
    entry.buffer.fill_(7)

    hit = cache.get("a")
    assert hit is not None
    assert hit.buffer.tolist() == [7] * 8
    assert cache.stats().l1_hits == 1
    assert cache.stats().l1_misses == 0
    assert cache.stats().l1_bytes == 8


def test_lru_evicts_oldest_unpinned_entry() -> None:
    evicted: list[str] = []
    cache = PinnedL1Cache(
        capacity_bytes=16,
        allocator=_cpu_allocator,
        on_evict=evicted.append,
    )
    cache.reserve("a", 8, durable=True)
    cache.reserve("b", 8, durable=True)
    assert cache.get("a") is not None

    cache.reserve("c", 8, durable=True)

    assert cache.get("a") is not None
    assert cache.get("b") is None
    assert cache.get("c") is not None
    assert evicted == ["b"]
    assert cache.stats().l1_evictions == 1


def test_durable_pin_prevents_eviction_until_released() -> None:
    cache = PinnedL1Cache(capacity_bytes=16, allocator=_cpu_allocator)
    cache.reserve("a", 8, durable=False, durable_pin=True)
    cache.reserve("b", 8, durable=False, durable_pin=True)

    with pytest.raises(MemoryError):
        cache.reserve("c", 8, durable=True)

    cache.mark_durable("a")
    cache.release_durable_pin("a")
    cache.reserve("c", 8, durable=True)

    assert cache.get("a") is None
    assert cache.get("b") is not None
    assert cache.get("c") is not None


def test_load_pin_prevents_eviction_until_release() -> None:
    cache = PinnedL1Cache(capacity_bytes=16, allocator=_cpu_allocator)
    cache.reserve("a", 8, durable=True)
    cache.reserve("b", 8, durable=True)
    assert cache.pin_for_load("a") is not None

    cache.reserve("c", 8, durable=True)

    assert cache.get("a") is not None
    assert cache.get("b") is None
    cache.release_load_pin("a")
    assert cache.get("c") is not None
    cache.reserve("d", 8, durable=True)
    assert cache.get("a") is None


def test_lookup_pin_prevents_eviction_until_release() -> None:
    cache = PinnedL1Cache(capacity_bytes=16, allocator=_cpu_allocator)
    cache.reserve("a", 8, durable=True)
    cache.reserve("b", 8, durable=True)
    assert cache.pin_for_lookup("a")

    cache.reserve("c", 8, durable=True)

    assert cache.get("a") is not None
    assert cache.get("b") is None
    cache.release_lookup_pin("a")
    assert cache.get("c") is not None
    cache.reserve("d", 8, durable=True)
    assert cache.get("a") is None


def test_oversized_entry_raises() -> None:
    cache = PinnedL1Cache(capacity_bytes=16, allocator=_cpu_allocator)

    with pytest.raises(ValueError, match="exceeds L1 capacity"):
        cache.reserve("huge", 17, durable=True)
