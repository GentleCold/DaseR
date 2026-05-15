# Contributing to DaseR

## Branch Naming

Create a branch for every change. Never commit directly to `master`.

```
feat/<topic>       # new feature          e.g. feat/storage-eviction
fix/<topic>        # bug fix              e.g. fix/ring-buffer-wrap
perf/<topic>       # performance change   e.g. perf/gds-read-path
refactor/<topic>   # refactor             e.g. refactor/ipc-client
revert/<topic>     # revert a change      e.g. revert/prefix-cache
chore/<topic>      # tooling / config     e.g. chore/ci-setup
test/<topic>       # tests only           e.g. test/connector-coverage
docs/<topic>       # documentation        e.g. docs/architecture
```

## Commit Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/).

```
<type>(<scope>): <short description>
```

**Types:** `feat` | `fix` | `perf` | `refactor` | `revert` | `chore` | `test` | `docs`

**Scopes:** `scaffold` | `storage` | `server` | `connector` | `tests` | `ci` | `docs`

**Examples:**
```
feat(connector): add async IPC client for worker role
fix(storage): correct ring buffer wrap-around offset calculation
perf(connector): reduce GDS read staging overhead
revert(server): restore previous chunk allocation policy
test(e2e): add cold-read latency assertion
chore(ci): add ruff pre-commit hook
```

## Merging to master

- All branches are merged via **squash merge** — the entire branch becomes one commit on `master`.
- The PR title is the squash commit message and must follow the commit format above.
- Force-pushing to `master` is prohibited.

## Pull Request Checklist

Use the pr templates in `.github/pull_request_template.md`.

## Reporting Issues

Use the issue templates in `.github/ISSUE_TEMPLATE/`. For bugs, include a minimal reproduction. For features, describe the motivation and how it fits the architecture.
