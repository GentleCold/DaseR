# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test for the per-request KV skip-save flag.

Runs two real vLLM forward passes through DaserConnector and verifies
behavior by querying the DaseR retrieval index over IPC after each one.
Avoids log-string matching since the connector logs come from the
EngineCore subprocess and would be brittle to parse from the test
process.

Phase 1 (default ``SamplingParams``):
    Connector should compute ``store_key = hash(tokens)`` and the chunk
    should land in DaseR. After the forward pass returns, an IPC
    ``lookup(tokens, model_id)`` MUST yield at least one chunk.

Phase 2 (``extra_args={"kv_transfer_params": {"daser_skip_save": True}}``):
    Connector sees ``daser_skip_save=True`` and sends an empty
    ``store_key``; the server therefore allocates no chunk. The
    subsequent IPC ``lookup`` MUST return an empty list.

Run with:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    LD_LIBRARY_PATH=/data/sza/conda_env/daser/lib \\
    pytest -xvs tests/integration/test_skip_save_e2e.py -m integration \\
    --log-cli-level=INFO

Same hardware / model prerequisites as ``test_vllm_e2e.py``.
"""

# Standard
import time

# Third Party
import pytest
from transformers import AutoTokenizer
from vllm import SamplingParams

# First Party
from daser.connector.daser_connector import hash_tokens
from daser.connector.ipc_client import IPCClientSync
from daser.logging import init_logger

# Local (reuse helpers from the existing E2E test to keep LLM
# construction / teardown identical to the other integration test).
from .test_vllm_e2e import (
    BLOCK_TOKENS,
    MODEL_PATH,
    _destroy_llm,
    _make_llm,
)

logger = init_logger(__name__)

MODEL_ID: str = "qwen3-8b"

# Two distinct prompts, each long enough to be block-aligned to at least
# one full block (16 tokens) after tokenization. We pick clearly
# different content so prompt_A's hash cannot accidentally match
# prompt_B's prefix and produce false-positive cache hits.
PROMPT_DEFAULT: str = (
    "The migration patterns of monarch butterflies span thousands of "
    "kilometres across North America every autumn. Each generation "
    "travels a specific leg of the route and never sees the destination."
)
PROMPT_SKIP_SAVE: str = (
    "Quantum entanglement allows two particles to share a state such "
    "that measuring one instantly determines the other, regardless of "
    "the physical distance separating them in the lab apparatus."
)


def _block_aligned(tokens: list[int]) -> list[int]:
    """Trim a token list down to a multiple of ``BLOCK_TOKENS``.

    The connector hashes ``tokens[:block_aligned_len]`` for its
    ``store_key`` (see ``DaserConnector.get_num_new_matched_tokens``),
    so the test must query the index with the same trimmed prefix.

    Args:
        tokens: raw token id list.

    Returns:
        ``tokens[:n]`` where ``n`` is the largest multiple of
        ``BLOCK_TOKENS`` that is ``<= len(tokens)``.
    """
    n = (len(tokens) // BLOCK_TOKENS) * BLOCK_TOKENS
    return tokens[:n]


@pytest.mark.integration
@pytest.mark.slow
def test_skip_save_blocks_chunk_alloc(daser_server: tuple[str, str, int]) -> None:
    """``daser_skip_save=True`` must keep the new chunk out of DaseR.

    Args:
        daser_server: ``(socket_path, store_path, slot_size)`` from the
            module-scoped fixture in ``conftest.py``.

    Asserts:
        - Phase 1 (default sampling): the prompt's block-aligned prefix
          becomes visible to ``IPCClient.lookup`` after the forward pass.
        - Phase 2 (skip-save flag set): the prompt's block-aligned prefix
          is *not* found by ``IPCClient.lookup``.
    """
    socket_path, _store_path, _slot_size = daser_server
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    ipc = IPCClientSync(socket_path)

    tokens_default = list(
        tokenizer(PROMPT_DEFAULT, add_special_tokens=False)["input_ids"]
    )
    tokens_skip = list(
        tokenizer(PROMPT_SKIP_SAVE, add_special_tokens=False)["input_ids"]
    )
    aligned_default = _block_aligned(tokens_default)
    aligned_skip = _block_aligned(tokens_skip)
    assert len(aligned_default) >= BLOCK_TOKENS, (
        "PROMPT_DEFAULT must tokenize to at least one full block"
    )
    assert len(aligned_skip) >= BLOCK_TOKENS, (
        "PROMPT_SKIP_SAVE must tokenize to at least one full block"
    )

    # Pre-condition: neither prompt is cached yet.
    assert ipc.lookup(aligned_default, MODEL_ID) == [], (
        "PROMPT_DEFAULT must not be in the index before Phase 1"
    )
    assert ipc.lookup(aligned_skip, MODEL_ID) == [], (
        "PROMPT_SKIP_SAVE must not be in the index before Phase 2"
    )

    llm = _make_llm(socket_path, _store_path, _slot_size)
    try:
        # --------------------------------------------------------------
        # Phase 1: default SamplingParams — connector must STORE
        # --------------------------------------------------------------
        params_default = SamplingParams(temperature=0.0, max_tokens=4)
        logger.info("[SKIP-SAVE] Phase 1: default sampling -> expect STORE")
        llm.generate([PROMPT_DEFAULT], params_default)

        # commit_chunk runs on the connector's background asyncio loop;
        # give it a brief window to land before querying the index.
        _wait_until_present(ipc, aligned_default, MODEL_ID, timeout=10.0)

        hits = ipc.lookup(aligned_default, MODEL_ID)
        assert len(hits) >= 1, (
            "Phase 1 (no flag): expected the prompt to be cached, "
            f"got {len(hits)} chunks. Connector default-path STORE broke."
        )
        expected_key = hash_tokens(aligned_default)
        assert any(c.get("chunk_key") == expected_key for c in hits), (
            f"Phase 1: chunk_key mismatch — expected {expected_key[:12]}…, "
            f"got {[c.get('chunk_key', '?')[:12] for c in hits]}"
        )
        logger.info("[SKIP-SAVE] Phase 1 OK — chunk %s… cached", expected_key[:12])

        # --------------------------------------------------------------
        # Phase 2: skip-save flag set — connector must NOT STORE
        # --------------------------------------------------------------
        params_skip = SamplingParams(
            temperature=0.0,
            max_tokens=4,
            extra_args={"kv_transfer_params": {"daser_skip_save": True}},
        )
        logger.info("[SKIP-SAVE] Phase 2: daser_skip_save=True -> expect NO STORE")
        llm.generate([PROMPT_SKIP_SAVE], params_skip)

        # Same delay as Phase 1 so we don't race a possible (incorrect)
        # delayed commit: if a STORE were going to happen, this is more
        # than enough time for it to land.
        time.sleep(2.0)

        hits_skip = ipc.lookup(aligned_skip, MODEL_ID)
        assert hits_skip == [], (
            "Phase 2 (skip-save): expected NO chunk cached, "
            f"got {len(hits_skip)} chunks: "
            f"{[c.get('chunk_key', '?')[:12] for c in hits_skip]}. "
            "daser_skip_save flag was not honored end-to-end."
        )
        logger.info("[SKIP-SAVE] Phase 2 OK — no chunk allocated")

        # --------------------------------------------------------------
        # Cross-check: the Phase 1 chunk is still present after Phase 2.
        # Skip-save must not have evicted or shadowed prior cache state.
        # --------------------------------------------------------------
        hits_after = ipc.lookup(aligned_default, MODEL_ID)
        assert len(hits_after) >= 1, (
            "Phase 2 should not have affected Phase 1's cached chunk, "
            "but it disappeared from the index."
        )
        logger.info("[SKIP-SAVE] cross-check OK — Phase 1 chunk still indexed")
    finally:
        _destroy_llm(llm)
        ipc.close()


def _wait_until_present(
    ipc: IPCClientSync,
    tokens: list[int],
    model_id: str,
    timeout: float,
) -> None:
    """Poll ``ipc.lookup`` until it returns a hit or ``timeout`` elapses.

    The connector commits chunks on a background asyncio loop after the
    forward pass, so a non-trivial wall-clock delay is possible between
    ``LLM.generate`` returning and the chunk becoming visible. Polling
    keeps the test fast in the common case (hit lands in <100 ms) while
    still giving slow CI hosts some headroom before declaring failure.

    Args:
        ipc: connected IPC client.
        tokens: block-aligned token prefix to look up.
        model_id: model identifier matching the DaseR runtime config.
        timeout: maximum seconds to wait before giving up.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if ipc.lookup(tokens, model_id):
            return
        time.sleep(0.05)
