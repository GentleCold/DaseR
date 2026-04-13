# SPDX-License-Identifier: Apache-2.0

# Standard

# Third Party

# First Party
from daser.server.chunk_manager import ChunkManager
from daser.server.metadata_store import MetadataStore


def make_manager(total_slots: int = 8) -> ChunkManager:
    store = MetadataStore(total_slots=total_slots)
    return ChunkManager(total_slots=total_slots, metadata_store=store)


def test_alloc_returns_correct_start_slot():
    mgr = make_manager(8)
    slot = mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert slot == 0


def test_alloc_advances_head():
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    slot = mgr.alloc("key2", num_slots=2, token_count=32, model_id="m", pos_offset=48)
    assert slot == 3


def test_free_slots_decreases():
    mgr = make_manager(8)
    assert mgr.free_slots == 8
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert mgr.free_slots == 5


def test_evicts_oldest_when_full():
    mgr = make_manager(6)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    mgr.alloc("key3", num_slots=3, token_count=48, model_id="m", pos_offset=96)
    assert mgr.store.get("key1") is None
    assert mgr.store.get("key2") is not None
    assert mgr.store.get("key3") is not None


def test_evict_advances_tail():
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    assert mgr.tail == 0
    mgr.evict_oldest()
    assert mgr.tail == 3
    assert mgr.store.get("key1") is None


def test_wrap_around():
    # total=8; key1(3) at [0..2], key2(3) at [3..5] → head=6
    # key3(4) needs 4 slots; only 2 remain at tail → SKIP [6,7], head wraps to 0
    # must evict key1 then key2 to make room; key3 lands at slot 0
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    slot = mgr.alloc("key3", num_slots=4, token_count=64, model_id="m", pos_offset=96)
    assert slot == 0
    assert mgr.store.get("key3") is not None


def test_wrap_skip_advances_tail():
    # After wrap: key3 at [0..3], SKIP at [6,7], tail at 6
    # alloc key4 (2 slots): tail must consume SKIP [6,7] → tail wraps to 0
    # then key4 sits at slot 4
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=3, token_count=48, model_id="m", pos_offset=48)
    mgr.alloc("key3", num_slots=4, token_count=64, model_id="m", pos_offset=96)
    mgr.alloc("key4", num_slots=2, token_count=32, model_id="m", pos_offset=160)
    assert mgr.store.get("key4") is not None
    assert mgr.store.get("key4").start_slot == 4


def test_save_and_load_state(tmp_path):
    mgr = make_manager(8)
    mgr.alloc("key1", num_slots=3, token_count=48, model_id="m", pos_offset=0)
    mgr.alloc("key2", num_slots=2, token_count=32, model_id="m", pos_offset=48)
    path = str(tmp_path / "daser.index")
    mgr.save(path)

    store2 = MetadataStore(total_slots=8)
    mgr2 = ChunkManager(total_slots=8, metadata_store=store2)
    mgr2.load(path)
    assert mgr2.head == mgr.head
    assert mgr2.tail == mgr.tail
    assert mgr2.store.get("key1") is not None
    assert mgr2.store.get("key2") is not None
