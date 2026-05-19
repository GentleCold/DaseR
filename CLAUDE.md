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
3. **Transfer backend is immutable after startup.** `python -m daser.server --transfer-mode` selects `gds` or `iouring-pinned` once at startup — no runtime switching.
4. **Data plane is server-managed.** DaseR server owns SSD files, transfer backend selection, L1/L2 sizing, and replacement policy. The vLLM worker exposes staging buffers through CUDA IPC and does not open or manage SSD files.
5. **Control plane stays in the DaseR server.** Index lookups, chunk allocation, position offset management, transfer orchestration, and metadata serialization belong in `daser/server/`. The connector calls these via IPC only.
6. **Do not modify vLLM or LMCache source code without explicit permission.** Treat both as read-only third-party dependencies. If an upstream change is required, raise it with the user first.

Behavioral guidelines.

7. **Run benchmarks to completion.** When running benchmarks or performance tests, always execute them to completion within the session and report results. Do not stop after writing code edits without running them.
8. **Prefer minimal, targeted changes.** Avoid broad refactors. If a simpler approach exists, propose it first.
9. **Verify command syntax.** Primary language is Python. Always verify exact flag syntax before suggesting CLI commands.

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
