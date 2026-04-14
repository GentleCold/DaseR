# CLAUDE.md

Guidelines for AI coding agents (Claude Code, Copilot, Cursor, etc.) working in this repository.

## Project Overview

DaseR is a RAG-native KV cache service for LLM inference. It runs as an independent server process and integrates with vLLM via the `KVConnectorBase_V1` interface. Its core differentiators are:

- **GDS-first data path**: cuFile async for NVMe↔GPU DMA, io_uring fallback when GDS is unavailable
- **Pluggable retrieval**: `RetrievalIndex` ABC decouples retrieval strategy from storage
- **Ring buffer storage**: single large XFS file with fixed-size slots, external index
- **Independent server**: DaseR is a separate process; vLLM is the client

DaseR does **not** depend on LMCache. LMCache and vLLM in this repo are references only.

## Repository

Default branch: `master`. Base all new branches and pull requests against `master`.

All production code lives under `DaseR/daser/`. Do not place source code outside this directory.

## Python Environment

```bash
source /data/zwt/vllm/bin/activate
```

Install in editable mode:

```bash
cd DaseR
pip install -e .
```

## Key Paths

| Resource | Path |
|----------|------|
| Python venv | `/data/zwt/vllm/` |
| LMCache reference | `/home/zwt/daser_project/LMCache/` |
| vLLM reference | `/home/zwt/daser_project/vllm/` |
| Design docs | `DaseR/doc/design/` |
| Server resources | `doc/source.md` |
| Default model | `/data/zwt/model/models/Qwen/Qwen3-8B` |
| Test working directory | `/data/zwt/daser_test/` |

Server topology (GPU inventory, NVMe devices, PCIe bus IDs, NUMA config, default model and test paths) is documented in [`doc/source.md`](doc/source.md). Read it before writing storage paths, CUDA device indices, or NUMA assumptions into code.

## Architecture Rules

These rules reflect core design decisions. Do not violate them without updating the design doc first.

- **Cross-layer calls go through ABCs only.** `DaserConnector` calls `GDSTransferLayer` via its public API. `GDSTransferLayer` never imports from `daser.server`. Enforce boundaries.
- **Unified async model.** All IO is `asyncio` + `cuFile async` (or `io_uring`). Never mix `threading.Thread` with `asyncio.run_coroutine_threadsafe`. Never use `concurrent.futures.Future` alongside asyncio Futures.
- **GDS backend is immutable after startup.** `GDSTransferLayer` selects `CuFileBackend` or `IOUringBackend` once at init. There is no runtime switching.
- **Data plane stays in vLLM process.** GDS DMA (`cuFileReadAsync` / `cuFileWriteAsync`) must execute in the vLLM worker process, which owns the CUDA context and registered GPU buffers. Never move KV tensor IO to the DaseR server process.
- **Control plane stays in DaseR server.** Index lookups, chunk allocation, position offset management, and metadata serialization belong in `daser/server/`. The connector only calls these via IPC.
- **No LMCache imports.** `daser/` must not import from `lmcache`. Use it as a reference only.

## Testing

### Running Tests

```bash
# Full test suite
pytest -xvs tests/

# Single file
pytest -xvs tests/test_chunk_manager.py

# Single test
pytest -xvs tests/test_chunk_manager.py::test_ring_wrap
```

### Testing Practices

- Test against the **public interface and docstring contract**, not the implementation.
- Do not access private members (`_`-prefixed) in tests unless strictly necessary.
- Unit tests must use `IOUringBackend` (or a mock) in place of `CuFileBackend` — do not require real GDS hardware to run the test suite.
- All new features and bug fixes must include corresponding tests.
- Ensure existing tests pass before submitting.

## Linting & Code Quality

```bash
# Run all checks
pre-commit run --all-files

# Individual tools
ruff check .          # lint
ruff format .         # format (line-length 88)
isort .               # import sorting
mypy .                # type checking
```

### Import Ordering

```python
# Standard
import os

# Third Party
import torch

# First Party
from daser.config import DaserConfig

# Local
from .utils import helper
```

### File Header

Every Python file must begin with:

```python
# SPDX-License-Identifier: Apache-2.0
```

## Coding Conventions

### Type Hints

All functions and methods must have type hints for arguments and return values.

### Docstrings

Every public function and method must have a docstring covering:
- What the function does
- Arguments (with types and descriptions)
- Return values
- Any asyncio / thread-safety considerations

### Logging

Use the unified logger. Never use `print()` in production code.

```python
from daser.logging import init_logger
logger = init_logger(__name__)
```

Performance-sensitive paths use the perf logger:

```python
from daser.logging import init_perf_logger
perf = init_perf_logger(__name__)
perf.record("gds_read_latency_ms", latency)
```

Always log the active GDS backend at startup:

```python
logger.info("[GDS] backend=%s", self._backend.name)
```

### Encapsulation

Never access private members (`_`-prefixed) of other classes. Interact only through public APIs.

### Code Organization

- Module-level helper functions go at the top of the file (after imports, before classes).
- Private/helper methods within a class go at the end, after all public methods.

## Writing Design Docs

Before implementing a new module or making a significant architectural change, write a design doc in `doc/design/YYYY-MM-DD-<topic>.md`. The design doc should cover:

- **Why**: motivation and the problem being solved
- **What**: the interface (ABC or dataclass), not the implementation
- **Trade-offs**: alternatives considered and why this approach was chosen
- **Out of scope**: explicitly state what is deferred

Keep design docs concise. Use ASCII diagrams for component interactions.

## Code Review Checklist

### Correctness
- [ ] Code does what the design doc / PR description claims
- [ ] Edge cases handled (empty inputs, None, boundary conditions)
- [ ] No regressions — existing tests still pass

### Architecture
- [ ] No cross-layer direct imports (connector does not import server internals)
- [ ] No `threading` + `asyncio` mixing
- [ ] GDS backend not accessed outside `GDSTransferLayer`
- [ ] No LMCache imports in `daser/`

### Style & Standards
- [ ] `pre-commit run --all-files` passes
- [ ] All new/modified functions have type hints and docstrings
- [ ] License header present on all Python files
- [ ] Import ordering follows Standard / Third Party / First Party / Local

### Testing
- [ ] New features and bug fixes include tests
- [ ] Tests use `IOUringBackend` or mock — no GDS hardware required
- [ ] Tests target public interface, not internals

### Performance
- [ ] No unnecessary memory copies in hot paths (especially connector ↔ GDSTransferLayer)
- [ ] CUDA resources properly released (streams, events, registered buffers)
- [ ] Async IO is truly non-blocking — no hidden `time.sleep` or synchronous syscalls in hot paths

### Documentation
- [ ] Non-obvious design decisions have a comment explaining *why*
- [ ] If a new module was added, `doc/design/` has a corresponding design doc
