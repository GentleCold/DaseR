# DaseR–vLLM End-to-End Integration Test Design

**Date:** 2026-04-14
**Status:** Approved

---

## Why

The existing test suite covers individual components (IPCServer, IPCClient, GDSTransferLayer, ChunkManager, PrefixHashIndex) in isolation. There is no test that exercises the full path:

```
DaserConnector (SCHEDULER) → IPC → DaseR server
DaserConnector (WORKER) → GDS → NVMe → GDS → GPU
```

Without this test, a regression in the connector↔server wire protocol, slot-offset arithmetic, or the GDS roundtrip under vLLM's execution model would go undetected.

The integration test must verify two things:
1. **Correctness**: output tokens from a warm (cache-hit) inference are identical to those from the cold run.
2. **Performance**: the warm run completes faster than the cold run (DaseR actually reduces generation wall time).

---

## What

### Test scenario

A two-phase test using the real `vllm.LLM` offline API and a real DaseR IPCServer:

```
Phase 1 — cold:
  LLM #1 (DaserConnector, kv_both) → infer prompts
  → DaserConnector stores KV blocks to NVMe via GDS
  → commit_chunk IPC → DaseR index updated
  del LLM #1 ; torch.cuda.empty_cache()   ← vLLM memory cache cleared

Phase 2 — warm:
  LLM #2 (same config, same socket/store) → infer same prompts
  → DaserConnector lookup → cache hit
  → load KV from NVMe into GPU via GDS
  del LLM #2 ; torch.cuda.empty_cache()

Assert: for each prompt, out1.text == out2.text
Assert: generation_time_warm < generation_time_cold
```

### Files

| File | Description |
|------|-------------|
| `tests/integration/__init__.py` | Package marker |
| `tests/integration/conftest.py` | `daser_server` module-scoped fixture |
| `tests/integration/test_vllm_e2e.py` | Integration test functions |

---

## Interface / Fixture Contract

### `daser_server` fixture (`conftest.py`)

- **Scope**: `module` — server stays alive across both LLM instances.
- **What it does**:
  1. Resolves `SLOT_SIZE` (see constants below).
  2. Pre-allocates `daser.store` under `tmp_path_factory`: size = `TOTAL_SLOTS × SLOT_SIZE`.
  3. Instantiates `MetadataStore → ChunkManager → PrefixHashIndex → FixedOffsetEncoder → IPCServer`.
  4. Runs the server's asyncio event loop in a `threading.Thread` (daemon=True).
  5. Yields `(socket_path: str, store_path: str, slot_size: int)`.
  6. Teardown: signals the loop to stop, joins the thread, removes socket + store files.

### Constants

```python
MODEL_PATH    = "/data/zwt/model/models/Qwen/Qwen3-8B"
# Qwen3-8B KV params (from config.json)
NUM_KV_HEADS  = 8
HEAD_DIM      = 128
NUM_LAYERS    = 36
BLOCK_TOKENS  = 16        # vLLM default block size
DTYPE_BYTES   = 2         # bfloat16
SLOT_SIZE     = NUM_KV_HEADS * HEAD_DIM * 2 * NUM_LAYERS * BLOCK_TOKENS * DTYPE_BYTES
# = 2,359,296 bytes (~2.25 MB per slot)
TOTAL_SLOTS   = 64        # enough for the test prompts
```

`SLOT_SIZE` is passed identically to the `IPCServer` fixture and to `kv_connector_extra_config`
so that file offset arithmetic agrees between server and connector.

### `_make_llm` helper

```python
def _make_llm(socket_path, store_path, slot_size) -> LLM:
    kv_cfg = KVTransferConfig(
        kv_connector="DaserConnector",
        kv_connector_module_path="daser.connector.daser_connector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "socket_path": socket_path,
            "store_path": store_path,
            "slot_size": slot_size,
            "block_tokens": BLOCK_TOKENS,
            "model_id": "qwen3-8b",
        },
    )
    return LLM(
        model=MODEL_PATH,
        kv_transfer_config=kv_cfg,
        gpu_memory_utilization=0.7,
        max_model_len=2048,
        disable_hybrid_kv_cache_manager=True,  # DaserConnector does not implement HMA
    )
```

---

## Test Functions

### `test_output_correctness_and_perf`

```
markers: @pytest.mark.integration, @pytest.mark.slow
```

```
# --- Phase 1: cold ---
llm1 = _make_llm(...)          # model loaded here; exclude from timing
t0 = perf_counter()
cold_out = llm1.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=64))
cold_gen_time = perf_counter() - t0
del llm1 ; cuda.empty_cache()

# --- Phase 2: warm ---
llm2 = _make_llm(...)          # model loaded here; exclude from timing
t1 = perf_counter()
warm_out = llm2.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=64))
warm_gen_time = perf_counter() - t1
del llm2 ; cuda.empty_cache()

# Correctness
for c, w in zip(cold_out, warm_out):
    assert c.outputs[0].text == w.outputs[0].text

# Performance: warm generation must be faster
logger.info("cold_gen=%.2fs  warm_gen=%.2fs  speedup=%.2fx",
            cold_gen_time, warm_gen_time, cold_gen_time / warm_gen_time)
assert warm_gen_time < cold_gen_time
```

Model load time is excluded by starting the timer after `LLM()` returns (the constructor loads the model in offline mode). The timing therefore measures only generation latency, making the cold/warm comparison meaningful.

### Test prompts

Two prompts, each > 64 tokens, so ≥ 4 full vLLM blocks are cached and reloaded:

```python
PROMPTS = [
    "Artificial intelligence is transforming the way we work and live. "
    "From natural language processing to computer vision, machine learning "
    "models are being deployed in healthcare, finance, transportation, and "
    "education. As these systems become more capable, questions about safety, "
    "alignment, and interpretability grow more urgent. Researchers at "
    "universities and companies around the world are working to ensure that "
    "AI systems remain beneficial and controllable as they scale. Describe "
    "the key technical challenges in AI alignment:",

    "The history of computing spans eight decades, from vacuum tube machines "
    "weighing several tons to pocket-sized devices more powerful than the "
    "supercomputers of the 1990s. The invention of the transistor, the "
    "integrated circuit, and the microprocessor each triggered an order-of-"
    "magnitude leap in capability. Today, GPU clusters connected by "
    "high-bandwidth interconnects power large language models trained on "
    "trillions of tokens. Summarize the most important inflection points in "
    "computer hardware history:",
]
```

---

## Trade-offs

| Option | Pro | Con |
|--------|-----|-----|
| Two LLM instances (chosen) | vLLM memory cache is fully cleared; test truly exercises DaseR cache | Slow (loads Qwen3-8B twice) |
| Memory-pressure eviction | Single LLM instance | Fragile; hard to guarantee eviction |
| Subprocess `vllm serve` + OpenAI API | Closest to production | Adds HTTP layer complexity; harder to set up in pytest |

---

## Out of Scope

- Multi-GPU / tensor-parallel setups.
- GDS hardware path (test uses IOUring fallback if `libcufile` is unavailable).
- Ring-buffer eviction under pressure (covered by `test_chunk_manager.py`).
- Benchmarking / throughput measurement beyond the cold/warm comparison in this test.
