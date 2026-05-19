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
from time import perf_counter

# Third Party
import pytest
import torch
from vllm import LLM, SamplingParams

# First Party
from daser.logging import init_logger

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Test configuration
# ---------------------------------------------------------------------------
MODEL_PATH: str = "/data/zwt/model/models/Qwen/Qwen3-8B"
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_output_correctness_and_perf(daser_server: tuple[str, str, int]) -> None:
    """Verify that cache-hit output matches cold output and warm run is faster.

    Phase 1 (cold): LLM #1 computes KV from scratch; DaserConnector stores
    it to NVMe and commits to the DaseR index.

    Phase 2 (warm): the same LLM sees a DaseR cache hit, loads KV from NVMe
    via GDS, and produces identical one-token output.

    Args:
        daser_server: (socket_path, store_path, slot_size) from fixture.

    Asserts:
        - Output text is identical across cold and warm runs.
        - Warm generation time is less than cold generation time.
    """
    socket_path, store_path, slot_size = daser_server
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    # ------------------------------------------------------------------
    # Phase 1: cold run — DaseR miss → compute KV → store to NVMe
    # ------------------------------------------------------------------
    llm1 = _make_llm(socket_path, store_path, slot_size)
    logger.info("[E2E] Phase 1: cold inference starting")
    t0 = perf_counter()
    cold_outputs = llm1.generate(PROMPTS, params)
    cold_gen_time = perf_counter() - t0
    logger.info("[E2E] Phase 1: cold generation done in %.2fs", cold_gen_time)

    # ------------------------------------------------------------------
    # Phase 2: warm run — DaseR hit → load KV from NVMe. Keep this in the same
    # LLM process because server-managed CUDA IPC/GDS initializes CUDA in the
    # pytest parent process, and a second forked vLLM EngineCore can fail CUDA
    # driver initialization.
    # ------------------------------------------------------------------
    logger.info("[E2E] Phase 2: warm inference starting")
    t1 = perf_counter()
    warm_outputs = llm1.generate(PROMPTS, params)
    warm_gen_time = perf_counter() - t1
    logger.info("[E2E] Phase 2: warm generation done in %.2fs", warm_gen_time)
    _destroy_llm(llm1)

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

    # ------------------------------------------------------------------
    # Performance: warm generation must be faster than cold
    # ------------------------------------------------------------------
    speedup = cold_gen_time / warm_gen_time if warm_gen_time > 0 else float("inf")
    logger.info(
        "[E2E] cold_gen=%.2fs  warm_gen=%.2fs  speedup=%.2fx",
        cold_gen_time,
        warm_gen_time,
        speedup,
    )
    assert warm_gen_time < cold_gen_time, (
        f"Warm run ({warm_gen_time:.2f}s) was not faster than "
        f"cold run ({cold_gen_time:.2f}s). "
        "DaseR cache hit may not have occurred — check connector logs."
    )
