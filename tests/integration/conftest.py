# SPDX-License-Identifier: Apache-2.0

# Standard
import asyncio
import threading

# Third Party
import pytest

# First Party
from daser.position.fixed_offset import FixedOffsetEncoder
from daser.retrieval.prefix import PrefixHashIndex
from daser.server.chunk_manager import ChunkManager
from daser.server.core import ServerCore
from daser.server.doc_registry import DocRegistry
from daser.server.ipc import IPCServer
from daser.server.metadata_store import MetadataStore

# ---------------------------------------------------------------------------
# Qwen3-8B KV geometry (verified against config.json)
# ---------------------------------------------------------------------------
NUM_KV_HEADS: int = 8
HEAD_DIM: int = 128
NUM_LAYERS: int = 36
BLOCK_TOKENS: int = 16  # vLLM default block size
DTYPE_BYTES: int = 2  # bfloat16

# Bytes per vLLM block across all layers.
# Must match DaserConnector's computed slot_size so that file_offset
# arithmetic agrees between the server and the connector.
# = 8 * 128 * 2 * 36 * 16 * 2 = 2,359,296
SLOT_SIZE: int = NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS * BLOCK_TOKENS * DTYPE_BYTES

TOTAL_SLOTS: int = 128  # ring buffer capacity for the test


@pytest.fixture(scope="module")
def daser_server(tmp_path_factory: pytest.TempPathFactory):
    """Start a real DaseR IPC server in a background asyncio thread.

    Yields:
        tuple[str, str, int]: (socket_path, store_path, slot_size)

    The server stays alive for the entire test module so that both LLM
    instances share the same index and store file.
    """
    tmp = tmp_path_factory.mktemp("daser")
    socket_path = str(tmp / "daser.sock")
    store_path = str(tmp / "daser.store")

    # Pre-allocate the store file so GDS writes never need to extend it.
    store_size = TOTAL_SLOTS * SLOT_SIZE
    with open(store_path, "wb") as f:
        f.write(b"\x00" * store_size)

    metadata_store = MetadataStore(total_slots=TOTAL_SLOTS)
    doc_registry = DocRegistry()
    cm = ChunkManager(
        total_slots=TOTAL_SLOTS,
        metadata_store=metadata_store,
        doc_registry=doc_registry,
    )
    core = ServerCore(
        chunk_manager=cm,
        retrieval_index=PrefixHashIndex(block_tokens=BLOCK_TOKENS),
        position_encoder=FixedOffsetEncoder(fixed_offset=0),
        slot_size=SLOT_SIZE,
        block_tokens=BLOCK_TOKENS,
    )
    server = IPCServer(
        socket_path=socket_path,
        core=core,
        runtime_config={
            "socket_path": socket_path,
            "store_path": store_path,
            "slot_size": SLOT_SIZE,
            "block_tokens": BLOCK_TOKENS,
            "model_id": "qwen3-8b",
        },
    )

    # Run the server's asyncio event loop in a dedicated daemon thread.
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        started.set()
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True, name="daser-test-server")
    thread.start()
    assert started.wait(timeout=10.0), "DaseR test server failed to start in 10s"

    yield socket_path, store_path, SLOT_SIZE

    # Teardown: stop server gracefully, then stop the event loop.
    stop_future = asyncio.run_coroutine_threadsafe(server.stop(), loop)
    stop_future.result(timeout=10.0)
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=10.0)
