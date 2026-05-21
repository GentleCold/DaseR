# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test: DaserConnector + vLLM LLM offline API.

Run with:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
    pytest -xvs tests/integration/test_vllm_e2e.py -m integration \\
    --log-cli-level=INFO

Requires:
    - CUDA GPU with >= 24 GB VRAM
    - Qwen3-8B weights at /data/zwt/model/models/Qwen/Qwen3-8B
    - DaseR installed in editable mode (pip install -e .)
    - vLLM installed in the active venv
"""

# Standard
import gc
from time import perf_counter, sleep, time

# Third Party
import pytest
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# First Party
from daser.connector.ipc_client import IPCClientSync
from daser.logging import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/data/zwt/model/models/Qwen/Qwen3-8B"
MODEL_ID: str = "qwen3-8b"
BLOCK_TOKENS: int = 16  # must match conftest.BLOCK_TOKENS
MAX_NEW_TOKENS: int = 1

# Two prompts each > 64 tokens so ≥ 4 full KV blocks are cached per prompt.
PROMPTS: list[str] = [
    (
        "Artificial intelligence is transforming the way we work and live. "
        "From natural language processing to computer vision, machine learning "
        "models are being deployed in healthcare, finance, transportation, and "
        "education. As these systems become more capable, questions about "
        "safety, alignment, and interpretability grow more urgent. Researchers "
        "at universities and companies around the world are working to ensure "
        "that AI systems remain beneficial and controllable as they scale. "
        "Describe the key technical challenges in AI alignment:"
    ),
    (
        "The history of computing spans eight decades, from vacuum tube "
        "machines weighing several tons to pocket-sized devices more powerful "
        "than the supercomputers of the 1990s. The invention of the transistor,"
        " the integrated circuit, and the microprocessor each triggered an "
        "order-of-magnitude leap in capability. Today, GPU clusters connected "
        "by high-bandwidth interconnects power large language models trained on"
        " trillions of tokens. Summarize the most important inflection points "
        "in computer hardware history:"
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_llm(socket_path: str, store_path: str, slot_size: int) -> LLM:
    """Create a vLLM LLM instance wired to DaserConnector.

    Args:
        socket_path: Unix socket path for DaseR IPC.
        store_path: unused fixture compatibility value.
        slot_size: unused fixture compatibility value.

    Returns:
        vLLM LLM instance ready for offline generation.
    """
    del store_path, slot_size
    kv_transfer_config: dict = {
        "kv_connector": "DaserConnector",
        "kv_connector_module_path": "daser.connector.daser_connector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "socket_path": socket_path,
        },
    }
    return LLM(
        model=MODEL_PATH,
        kv_transfer_config=kv_transfer_config,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        enable_prefix_caching=False,
        disable_hybrid_kv_cache_manager=True,
    )


def _destroy_llm(llm: LLM) -> None:
    """Shut down the vLLM engine and release all GPU memory.

    Calls engine_core.shutdown() to trigger the full teardown chain:
    EngineCoreClient → EngineCore.shutdown() → scheduler.shutdown()
    → DaserConnector.shutdown() (stops background GDS thread, closes fd).

    Args:
        llm: The LLM instance to destroy.
    """
    try:
        llm.llm_engine.engine_core.shutdown(timeout=30.0)
    except TypeError:
        # Some EngineCoreClient implementations do not accept a timeout kwarg.
        llm.llm_engine.engine_core.shutdown()
    except Exception as exc:
        logger.warning("[E2E] engine_core.shutdown raised: %s", exc)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()


def _block_aligned(tokens: list[int]) -> list[int]:
    """Trim a token list to the DaseR block-aligned cache prefix."""
    n = (len(tokens) // BLOCK_TOKENS) * BLOCK_TOKENS
    return tokens[:n]


def _wait_until_visible(
    ipc: IPCClientSync,
    aligned_prompts: list[list[int]],
    timeout: float,
) -> None:
    """Wait until all aligned prompt prefixes are visible through IPC lookup.

    Args:
        ipc: connected DaseR IPC client.
        aligned_prompts: block-aligned prompt token prefixes.
        timeout: maximum seconds to wait.
    """
    deadline = time() + timeout
    while time() < deadline:
        visible = sum(1 for tokens in aligned_prompts if ipc.lookup(tokens, MODEL_ID))
        if visible == len(aligned_prompts):
            return
        sleep(0.05)
    raise AssertionError(
        "DaseR cold stores did not become visible before warm run "
        f"({visible}/{len(aligned_prompts)} visible)"
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_output_correctness_and_perf(daser_server: tuple[str, str, int]) -> None:
    """Verify that cache-hit output matches cold output.

    Phase 1 (cold): LLM #1 computes KV from scratch; DaserConnector stores
    it to NVMe and commits to the DaseR index.

    Phase 2 (warm): the same LLM sees a DaseR cache hit, loads KV from NVMe
    via GDS, and produces identical one-token output.

    Args:
        daser_server: (socket_path, store_path, slot_size) from fixture.

    Asserts:
        - Output text is identical across cold and warm runs.
        - Cold stores become visible through DaseR lookup before the warm run.
    """
    socket_path, store_path, slot_size = daser_server
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    warm_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_NEW_TOKENS,
        extra_args={"kv_transfer_params": {"daser_skip_save": True}},
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    aligned_prompts = [
        _block_aligned(tokenizer.encode(prompt, add_special_tokens=False))
        for prompt in PROMPTS
    ]
    ipc = IPCClientSync(socket_path)

    # ------------------------------------------------------------------
    # Phase 1: cold run — DaseR miss → compute KV → store to NVMe
    # ------------------------------------------------------------------
    llm1 = _make_llm(socket_path, store_path, slot_size)
    logger.info("[E2E] Phase 1: cold inference starting")
    t0 = perf_counter()
    cold_outputs = llm1.generate(PROMPTS, params)
    cold_gen_time = perf_counter() - t0
    logger.info("[E2E] Phase 1: cold generation done in %.2fs", cold_gen_time)
    _wait_until_visible(ipc, aligned_prompts, timeout=10.0)
    ipc.transfer_drain()

    # ------------------------------------------------------------------
    # Phase 2: warm run — DaseR hit → load KV from NVMe. Keep this in the same
    # LLM process because server-managed CUDA IPC/GDS initializes CUDA in the
    # pytest parent process, and a second forked vLLM EngineCore can fail CUDA
    # driver initialization.
    # ------------------------------------------------------------------
    logger.info("[E2E] Phase 2: warm inference starting")
    t1 = perf_counter()
    warm_outputs = llm1.generate(PROMPTS, warm_params)
    warm_gen_time = perf_counter() - t1
    logger.info("[E2E] Phase 2: warm generation done in %.2fs", warm_gen_time)
    _destroy_llm(llm1)
    ipc.close()

    # ------------------------------------------------------------------
    # Correctness: each prompt must produce identical output tokens
    # ------------------------------------------------------------------
    for i, (cold, warm) in enumerate(zip(cold_outputs, warm_outputs, strict=False)):
        cold_text = cold.outputs[0].text
        warm_text = warm.outputs[0].text
        assert cold_text == warm_text, (
            f"Prompt {i}: output mismatch\n  cold: {cold_text!r}\n  warm: {warm_text!r}"
        )
    logger.info("[E2E] correctness check passed: all %d outputs match", len(PROMPTS))

    # This test intentionally logs timing rather than asserting a speedup:
    # with only two short prompts the single-run timing variance can exceed
    # the transfer signal. End-to-end performance gates live in the benchmark.
    speedup = cold_gen_time / warm_gen_time if warm_gen_time > 0 else float("inf")
    logger.info(
        "[E2E] cold_gen=%.2fs  warm_gen=%.2fs  speedup=%.2fx",
        cold_gen_time,
        warm_gen_time,
        speedup,
    )
