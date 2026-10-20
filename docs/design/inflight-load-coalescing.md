# In-flight load coalescing

## Problem

vLLM publishes externally loaded blocks to its prefix cache only after the
asynchronous load lands. When two requests share a prompt prefix that is not
yet on the GPU, both see a local miss and an external hit, so each allocates
its own GPU blocks and each triggers its own L1/L2 read and H2D copy of the
same bytes. Agent and long-document workloads hit this often: several requests
for the same document or session prefix arrive while the first load is still
in flight.

## Mechanism

Coalescing is a scheduler-side policy in `RequestLifecycle`
(`daser/connector/scheduler/lifecycle.py`). It is off by default and enabled
with `kv_connector_extra_config={"coalesce_loads": true}`.

vLLM prefix caching must be enabled for followers to reuse the leader's
loaded blocks locally.

- **Leader registration.** When `get_num_new_matched_tokens` returns an
  external hit for a request, the request becomes a leader keyed by
  `(prefix end, prefix digest)`. The digest is an in-process hash of
  `tokens[:end]`. If vLLM later cannot allocate the blocks, the next lookup
  for that request replaces the entry.
- **Follower wait.** On a fresh lookup, `get_num_new_matched_tokens` checks for
  a leader whose prefix end lies beyond the request's local hit and inside its
  prompt, with a matching digest. If one exists, it returns `(None, False)`,
  and vLLM keeps the request waiting for this step without allocating blocks.
- **Landing.** When the worker reports `finished_recving` for a leader, the
  entry stays joinable for `COALESCE_LANDING_STEPS` (1) more scheduler step.
  vLLM caches the landed blocks when it promotes the leader in the next
  `schedule()` call, and that call may visit a follower first. After that
  step the entry is dropped.
- **Join.** A follower rechecks on each step. Once the leader's blocks are in
  the prefix cache, its local hit covers the prefix and it loads only the
  remainder, if any.
- **Bound.** A follower waits at most `COALESCE_MAX_WAIT_S` (10 s). After that
  it loads its own copy, which guards against a stalled or preempted leader.

## Correctness

- A digest collision only makes a request wait for a load it cannot reuse.
  vLLM's block hashes still decide what is actually hit, so the output tokens
  cannot change.
- A fresh lookup for a request drops any leader entry it registered earlier,
  for example when vLLM could not allocate its blocks. `request_finished`
  clears all per-request state.
- All state lives on vLLM's scheduler thread. No IPC or server change is
  involved, and architecture rules 1 and 5 hold.

## Observability

Cumulative counters are logged as
`[CONNECTOR] load coalescing: followers=N joined=N missed=N inflight=N`:

- `joined` counts followers whose local hit grew after waiting.
- `missed` counts followers that gave up or found no reuse.

## Measured effect

Setup: B1 long-document benchmark, single H800, GPU KV 24 GiB, L1 8 GiB, L2
160 GiB, `max_num_seqs` 16, one smoke run (not a formal result).

- External load fell by about 164k tokens, roughly 6% of loaded tokens. All of
  those tokens became GPU prefix hits.
- TTFT p50 fell from 595 ms to 491 ms, and p90 fell by 10%.

The benefit scales with how often a request arrives while another request for
the same prefix has not yet produced its first token. In the same benchmark
this fraction rose from 0.01 at low load to 0.58 (raw) and 0.66 (compressed)
at saturation.

## Tests

`tests/connector/test_load_coalescing.py` covers:

- the disabled baseline;
- follower wait followed by a local hit;
- non-matching prefixes;
- expiry of landed loads;
- release on leader finish;
- the stalled-leader timeout;
- requests whose local hit already covers the prefix.
