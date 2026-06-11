# CLAUDE.md

Guidelines for AI coding agents (Claude Code, Copilot, Cursor, etc.) working in this repository.

## Project

DaseR is a RAG-native KV cache service for LLM inference. It integrates with vLLM via `KVConnectorBase_V1` and stores KV tensors directly to NVMe using NVIDIA cuFile (GDS) or io_uring as a fallback.

- Default branch: `master`. Base all branches and PRs against `master`.
- All production code lives under `daser/`. Do not place source code elsewhere.
- See [docs/development.md](docs/development.md) for environment setup, commands, and vLLM integration.
- See [docs/design](docs/design) for system design and component overview.
- See [docs/insights/](docs/insights/) for project insights — research motivation, related work analysis, and roadmap.
- See [docs/optimizations/](docs/optimizations/) for optimization records and performance comparison benchmarks.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, commit format, and PR process — follow these conventions for all contributions.

## Rules

Architecture constraints — do not violate without updating the design doc first.

1. **Cross-layer calls go through ABCs and IPC only.** Never import across layer boundaries directly. `DaserConnector` calls DaseR server through the IPC client API; transfer implementations live behind `daser.transfer.TransferLayer` and never import from `daser.connector`.
2. **All IO is asyncio-based.** Do not introduce synchronous blocking calls on the hot path.
3. **Transfer backend is immutable after startup.** `python -m daser.server --transfer-mode` selects `gds` or `iouring` once at startup — no runtime switching.
4. **Data plane is server-managed.** DaseR server owns SSD files, transfer backend selection, L1/L2 sizing, and replacement policy. The vLLM worker exposes staging buffers through CUDA IPC and does not open or manage SSD files.
5. **Control plane stays in the DaseR server.** Index lookups, chunk allocation, position offset management, transfer orchestration, and metadata serialization belong in `daser/server/`. The connector calls these via IPC only.
6. **Do not modify vLLM or LMCache source code without explicit permission.** Treat both as read-only third-party dependencies. If an upstream change is required, raise it with the user first.

Behavioral guidelines.

7. **Run benchmarks to completion.** When running benchmarks or performance tests, always execute them to completion within the session and report results. Do not stop after writing code edits without running them.
8. **Prefer minimal, targeted changes.** Avoid broad refactors. If a simpler approach exists, propose it first.
9. **Verify command syntax.** Primary language is Python. Always verify exact flag syntax before suggesting CLI commands.
10. **Avoid leaking private paths.** Commit messages, issue bodies, PR
    descriptions, and committed code should not expose machine-specific private
    paths or local-only resource locations unless the user explicitly requests
    it.
11. **Prioritize long-term maintainability.** When writing code, keep the
    design maintainable over time. Refactor design where necessary, remove dead
    code instead of preserving it, and avoid excessive backward-compatibility
    scaffolding unless it is explicitly required.
12. **Do not use `/tmp` for tests.** The server root filesystem is small. Put
    test scratch files, sockets, generated data, and temporary stores under
    `<data-dir>/`, preferably `<data-dir>/daser_test/` for tests and
    `<data-dir>/daser_bench/` for benchmarks.
13. **Clean test store files after runs.** After tests or benchmarks complete,
    remove leftover store files and per-run scratch directories so repeated
    runs do not accumulate large artifacts on disk.

## Conventions

**File header** — every Python file must begin with:
```python
# SPDX-License-Identifier: Apache-2.0
```

**Type hints** — all functions and methods must have type hints for arguments and return values.

**Docstrings** — every public function and method must have a docstring covering: what it does, arguments (with types), return values, and any asyncio/thread-safety considerations.

**Logging** — use the unified logger; never use `print()` in production code:
```python
from daser.logging import init_logger
logger = init_logger(__name__)
```

**Code organization** — module-level helpers go at the top of the file (after imports, before classes). Private/helper methods within a class go at the end, after all public methods.

**Encapsulation** — never access private members (`_`-prefixed) of other classes. Interact only through public APIs.

**Testing** — Test against the public interface and docstring contract, not implementation internals.

**Commit format** — always follow `CONTRIBUTING.md`: `<type>(<scope>): <short description>`. Scope is mandatory. Always include a commit body describing the changes in this commit as `- ` bulleted points, one per change, not prose. For documentation restructures/tooling, prefer `chore(docs)` over `docs(docs)`; reserve `docs(...)` for cases where the type is clearly documentation-only and not redundant.

## Server Resources

Private paths and hardware information for this development machine. Use these when writing test commands, storage paths, or CUDA device assumptions, but do not hard-code them in committed source or docs.

### Python Venv

`<data-dir>/vllm/` — activate with `source <data-dir>/vllm/bin/activate`.

### Reference Repos

- LMCache: `<user-home>/daser_project/LMCache/`
- vLLM: `<user-home>/daser_project/vllm/`

### Default Model

`<data-dir>/model/models/Qwen/Qwen3-8B` — standard model for integration tests and benchmarks.

### Test Working Directory

`<data-dir>/daser_test/` — scratch area for ring-buffer files, IPC sockets,
and test artifacts. Use this instead of `/tmp`, because the server root
filesystem is small. Clean between runs and remove leftover store files after
tests complete.

### Benchmark Store Directory

`<data-dir>/daser_bench/` — default scratch base for benchmark `--store-dir`.
Create per-run subdirectories or let benchmark scripts use unique temporary
subdirectories under this path so repeated runs do not reuse stale store files.
Set `VLLM_WORKER_MULTIPROC_METHOD=spawn` for long benchmark runs that start
vLLM workers from these scripts; otherwise CUDA may fail to initialize after a
forked worker process.

### GPU Inventory

| Index | Model | VRAM | PCIe Bus ID |
|-------|-------|------|-------------|
| 0 | NVIDIA H800 | 80 GB | 0000:09:00.0 |
| 1 | NVIDIA GeForce RTX 4090 | 24 GB | 0000:11:00.0 |
| 2 | NVIDIA H800 | 80 GB | 0000:33:00.0 |
| 3 | NVIDIA H800 | 80 GB | 0000:38:00.0 |
| 4 | NVIDIA H800 PCIe | 80 GB | 0000:3C:00.0 |

GPUs 0, 2, 3, and 4 are H800s. GPU 1 is RTX 4090 for development and small-model testing. Default: `cuda:0`.

### NVMe Inventory

| Device | Capacity | Mount |
|--------|----------|-------|
| `/dev/nvme1n1` | 3.5 TiB | — |
| `/dev/nvme2n1` | 3.5 TiB | — |
| `/dev/nvme3n1` | 3.5 TiB | `/data` (btrfs, primary) |
| `/dev/nvme4n1` | 3.5 TiB | — |

All are MEMBLAZE P6531DT on NUMA node 0. Primary data volume: `/data` (`nvme3n1`, btrfs).

### NUMA Topology

Node 0: CPUs 0-47, 96-143, about 504 GB RAM.
Node 1: CPUs 48-95, 144-191, about 504 GB RAM.

All NVMes are on NUMA node 0. Use `numactl --cpunodebind=0` for GDS workloads.
