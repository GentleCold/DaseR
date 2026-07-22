# AGENTS.md

Guidelines for AI coding agents (Codex, Claude Code, Copilot, Cursor, etc.) working in this repository.

## Project

DaseR is a RAG-native KV cache service for LLM inference. It integrates with vLLM via `KVConnectorBase_V1` and stores KV tensors directly to NVMe using NVIDIA cuFile (GDS) or io_uring as a fallback.

- Default branch: `master`. Base all branches and PRs against `master`.
- All production code lives under `daser/`. Do not place source code elsewhere.
- See [docs/development.md](docs/development.md) for environment setup, commands, and vLLM integration.
- See [docs/design](docs/design) for system design and component overview.
- See [docs/insights/](docs/insights/) for project insights -- research motivation, related work analysis, and roadmap.
- See [docs/optimizations/](docs/optimizations/) for optimization records and performance comparison benchmarks.
- See [CONTRIBUTING.md](CONTRIBUTING.md) for branch naming, commit format, and PR process -- follow these conventions for all contributions.

## Rules

Architecture constraints -- do not violate without updating the design doc first.

1. **Cross-layer calls use public ABCs, APIs, or IPC.** `DaserConnector` communicates with the DaseR server through IPC; `IPCServer` invokes `TransferLayer` through its public API. Transfer implementations must not import from `daser.server`.
2. **All IO is asyncio-based.** Do not introduce synchronous blocking calls on the hot path.
3. **Transfer backend selection is immutable after startup.** The server selects the configured transfer backend once at initialization; runtime switching is not supported.
4. **Data-plane responsibilities are split by ownership.** The vLLM worker owns the CUDA context, KV cache tensors, and staging buffers. The DaseR server owns `TransferLayer`, CUDA IPC mappings, and transfer orchestration. Transfer operations use worker-exported buffers through the server-side transfer layer.
5. **Control plane stays in the DaseR server.** Index lookups, chunk allocation, position offset management, and metadata serialization belong in `daser/server/`. The connector calls these via IPC only.
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
    scaffolding unless it is explicitly required. Do not increase code
    complexity for speculative fallback or compatibility paths; prefer simple,
    maintainable code as the primary design criterion. When code complexity
    becomes too high, introduce appropriate module boundaries, abstractions, and
    structure instead of adding more ad hoc logic.
12. **Do not use `/tmp` for large tests or benchmarks.** Use a project-approved
    scratch directory outside the repository for sockets, generated data,
    temporary stores, and benchmark output.
13. **Clean test store files after runs.** After tests or benchmarks complete,
    remove leftover store files and per-run scratch directories so repeated
    runs do not accumulate large artifacts on disk.
14. **Avoid unnecessary unit tests and red tests.** Unit tests should cover
    core logic, behavior contracts, regressions, and meaningful edge cases.
    Do not add tests for trivial configuration/default-value edits, mechanical
    renames, constant changes, or other changes where a test only restates the
    implementation without protecting behavior.

## Conventions

**File header** -- every Python file must begin with:
```python
# SPDX-License-Identifier: Apache-2.0
```

**Type hints** -- all functions and methods must have type hints for arguments and return values.

**Docstrings** -- every public function and method must have a docstring covering: what it does, arguments (with types), return values, and any asyncio/thread-safety considerations.

**Code comments** -- write detailed comments for non-obvious code. Explain the
reasoning, invariants, concurrency/asyncio assumptions, performance constraints,
and hardware or IO requirements that are not clear from the code itself. Avoid
comments that merely restate what a line of code does.

**Logging** -- use the unified logger; never use `print()` in production code:
```python
from daser.logging import init_logger
logger = init_logger(__name__)
```

**Code organization** -- module-level helpers go at the top of the file (after imports, before classes). Private/helper methods within a class go at the end, after all public methods.

**Encapsulation** -- never access private members (`_`-prefixed) of other classes. Interact only through public APIs.

**Testing** -- Test against the public interface and docstring contract, not implementation internals.

**Commit format** -- always follow `CONTRIBUTING.md`: `<type>(<scope>): <short description>`. Scope is mandatory. Always include a commit body describing the changes in this commit as `- ` bulleted points, one per change, not prose. For documentation restructures/tooling, prefer `chore(docs)` over `docs(docs)`; reserve `docs(...)` for cases where the type is clearly documentation-only and not redundant.

<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

This project is managed by Trellis. The working knowledge you need lives under `.trellis/`:

- `.trellis/workflow.md` - development phases, when to create tasks, skill routing
- `.trellis/spec/` - package- and layer-scoped coding guidelines (read before writing code in a given layer)
- `.trellis/workspace/` - per-developer journals and session traces
- `.trellis/tasks/` - active and archived tasks (PRDs, research, jsonl context)

If a Trellis command is available on your platform (e.g. `/trellis:finish-work`, `/trellis:continue`), prefer it over manual steps. Not every platform exposes every command.

If you're using Codex or another agent-capable tool, additional project-scoped helpers may live in:
- `.agents/skills/` - reusable Trellis skills
- `.codex/agents/` - optional custom subagents

Managed by Trellis. Edits outside this block are preserved; edits inside may be overwritten by a future `trellis update`.

<!-- TRELLIS:END -->
