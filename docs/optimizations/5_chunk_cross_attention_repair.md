# Chunk-Mode Cross-Attention Repair: Analysis and Optimization Directions

**Date:** 2026-06-10
**Target:** accuracy of multi-chunk KV reuse in `--cache-reuse-mode chunk`
**Scope:** `daser/server/`, `daser/connector/`, `daser/ops/`, `benchmarks/`
**Status:** analysis and staged proposal; no code change in this document

Chunks are prefilled independently, so their KV tensors carry no
cross-attention against the system prompt or other documents. This document
decomposes the resulting accuracy error, states the hard constraints the
current architecture imposes on any repair, and lays out optimization
directions ranked by leverage. It supersedes the Stage C/D portions of the
earlier multi-document production plan where the two overlap.

## 1. Current State

The only load-time transform on master is positional: RoPE delta relocation
plus an optional global K/V scale in
`daser/connector/staging.py::_transform_loaded_staging_batch`. No
cross-attention repair exists.

The chunk-mode pipeline:

1. **Upload** (`POST /documents`): the document is padded to a block boundary
   (`Chunker.pad_to_block_boundary`, default `block_tokens=16`) and prefilled
   as a standalone prompt through vLLM (`max_tokens=1`). The chunk's KV is
   computed in isolation starting at position 0 and stored with
   `pos_offset=0`. Fixed segments (`chat_prefix`, `doc_separator`) are
   prewarmed the same way.
2. **Lookup** (`/infer`): `ChunkReuseIndex` scans block-aligned prompt windows
   and returns full-chunk hits at arbitrary offsets.
3. **Supply**: `SchedulerConnectorMixin.get_num_new_matched_tokens` accepts
   only the contiguous chunk coverage starting at `num_computed_tokens`
   (`_contiguous_prefix_tokens`), capped at `available - 1` so at least the
   final prompt token is always recomputed.
4. **Load**: staging bytes are repositioned by a TileLang RoPE delta kernel
   (`daser/ops/rope_apply.py`) and copied into the vLLM KV cache.

## 2. Error Decomposition

Four independent error sources, in decreasing order of impact:

1. **Fake attention sinks.** Each chunk is prefilled from position 0, so the
   model treats its first tokens as an attention sink (abnormally large K
   norms). A concatenated prompt with N chunks contains N fake sinks that
   compete for attention mass. Published ablations (EPIC/LegoLink) find this
   — not the missing cross-attention itself — is the dominant cause of
   accuracy loss in position-independent caching.
2. **Missing cross-chunk attention.** Document B's KV never attended to the
   system prompt or document A.
3. **Pad-token noise.** Block-boundary padding KV is stored, loaded, and
   attended to by every subsequent token (up to `block_tokens - 1` pad
   tokens per segment).
4. **RoPE relocation numerics.** bf16 rotation error at large position
   deltas. Minor relative to the above.

The query/task suffix is always recomputed with full attention over all
loaded KV, so the tail seam is healthy by construction; the damage is
concentrated inside and between the reused chunks.

## 3. Hard Constraints

Any repair plan must respect these; the first two are properties of the vLLM
`KVConnectorBase_V1` interface (read-only third-party dependency, CLAUDE.md
rule 6), the third is a property of our supply chain.

1. **Contiguous-only external supply.** `get_num_new_matched_tokens` returns
   a single int. "Skip the first k tokens of each chunk and let vLLM
   recompute them" (LegoLink-style selective recompute) is not expressible
   without vLLM changes. Holes can only be ceded at the head or tail of the
   whole reused prefix.
2. **No attention introspection.** The connector cannot read QK^T scores or
   attention weights, and cannot change the attention computation
   (temperature, masking) in-kernel.
3. **Block granularity.** `_trim_chunk_to_external_window` and the whole
   supply path align to `block_tokens` (16). Token-grain trim parameters
   round up to whole blocks; plan trims in block units.

## 4. Optimization Directions

### 4.1 Shared-prefix prefill — eliminate fake sinks at store time

**Highest leverage.** Instead of correcting sink drift numerically at load
time, prevent the fake sink from forming: prefill `[chat_prefix + doc]` at
upload time and store only the KV of the doc's block range.

- This is the shared-prefix mechanism from APE (ICLR'25): every chunk
  computes its KV facing the same real sink, so K-norm distributions stay
  normal and chunk heads are never "sinkified".
- Cost is marginal: chunk-mode lookup already hits the prewarmed
  `chat_prefix` chunk, so the prefix portion of the upload prefill loads
  from cache and only doc-attends-prefix compute is added.
- The missing piece is **sub-range store**: `ChunkReuseStrategy` only
  supports whole-prefix stores keyed by the full aligned prefix hash. The
  store spec needs a "store only the trailing N blocks, keyed by the doc
  segment hash" form. The load side already does the symmetric metadata
  surgery in `_trim_chunk_to_external_window`, so this is a contained
  connector + HTTP-upload change, entirely inside `daser/`.
- Orthogonal to load-time sink correction; once landed, the numeric
  correction term shrinks and may be disabled.

### 4.2 Composed-prefix partial hits + canonical chunk ordering — the lossless path

The contiguous-supply constraint is also an opportunity: "reuse a composed
prefix `[SYS, A, B]` + fully recompute the tail `[C, task]`" is natively
expressible and **strictly lossless** — the composed prefix carries full
internal cross-attention and the tail is recomputed.

- Requires a composed-KV index keyed by the ordered chunk sequence with
  longest-common-prefix matching, alongside the existing per-chunk index.
  Full-combination hits are a special case; partial-prefix hits are where
  the storage reuse rate comes from.
- **Canonical ordering** multiplies the hit rate: when assembling the
  `/infer` prompt, normalize `doc_ids` order (by access frequency, then a
  stable tiebreak) so `[A, C, B]` and `[A, B, C]` collapse into one
  combination. RAG document order is usually semantically exchangeable;
  gate it behind a flag for callers that require explicit order.
  Frequency-first ordering also concentrates heat near the prefix-tree
  root (RAGCache's knowledge-tree insight), which composes well with
  ring-buffer eviction.

### 4.3 Seam patches — residuals instead of seam chunks

Storing whole doc-pair seam KV costs O(pair length). The difference between
full-prefill KV and independently concatenated KV is concentrated in a small
number of boundary tokens (the HKVD observation behind CacheBlend's ~15%
recompute budget). Store only the **KV residual** of the k seam tokens and
add it back in `_transform_loaded_staging_batch` after RoPE relocation.

- Storage drops from O(pair) to O(k_seam) and the residual compresses well.
- Residual generation needs no HF dependency: a background full prefill of
  the combination (`daser_skip_load=true` + store) diffed against the
  independent chunk KV produces it, and the same pass yields the online
  calibration signal of §4.6.

### 4.4 Per-chunk load-time correction — upgrade the existing hooks

`load_key_scale` / `load_value_scale` already exist as connector-level
constants (default 1.0, plumbed from `kv_connector_extra_config` through
`copy_staging_to_kv_cache`). APE shows per-chunk K/V scaling factors close
most of the parallel-vs-sequential encoding gap when combined with a shared
prefix.

- Promote the two scalars to per-chunk metadata: `ChunkMeta` →
  IPC `lookup` payload → `ReqLoadSpec` → staging transform.
- Watch the copy-run fragmentation: `build_load_copy_runs` merges adjacent
  loads only when `pos_offset` matches; per-chunk parameters would split
  runs and multiply kernel launches. Extend the TileLang kernels to accept
  per-slot parameter tables (same pattern as the existing cos/sin tables)
  so one launch handles heterogeneous chunks.
- Sink-drift correction coefficients, if kept, should be calibrated
  per-layer rather than as a global constant — error concentrates in
  retrieval heads and shallow layers, and the stat tensor is already
  shaped `[layers, kv_heads, head_dim]`.
- APE's third knob (attention temperature) requires kernel changes and
  stays out of scope; uniform K scaling is its first-order approximation.

### 4.5 Low-cost hygiene fixes

- **Pad strategy.** Replace `pad_token` block padding with semantically
  neutral natural tokens (newline / separator text), or zero the V rows of
  pad positions at load time. Removes error source 3 outright.
- **Tail awareness.** `extra_tokens = available - 1` already guarantees one
  recomputed token; head-side block-aligned trim
  (`_trim_chunk_to_external_window`) is the only other recompute lever and
  rounds to 16-token blocks — size any head-trim parameter accordingly.

### 4.6 Traffic-driven composed upgrade + KV-space calibration

The cost-based decision between "independent chunks + correction" and
"composed KV" needs an execution mechanism that does not depend on users
registering doc-sets:

- First sighting of a combination: serve via independent chunks (fast,
  lossy), then schedule a low-priority background full prefill
  (`daser_skip_load` + store) that writes the composed KV. Second sighting
  onward hits losslessly. Combination-level access counters plus a bounded
  background scheduler are the only new pieces; GPU contention is managed
  by concurrency limits and off-peak scheduling.
- The same shadow prefill produces an **online accuracy proxy**: per-layer
  distance between composed and concatenated KV, measured in KV space on
  the serving model. This replaces offline HF
  `output_attentions` calibration — no separate pipeline, no
  calibration-vs-production model drift.

## 5. Priority and Sequencing

1. **Accuracy baseline first — done, see §7.**
   `benchmarks/bench_accuracy_baseline.py` measures raw concatenation vs
   full prefill on a LongBench QA subset. The measured gap (extracted-answer
   F1 +0.21 to +0.45, recall +0.05 to +0.26) is large enough to justify the
   full sequence below; multi-hop QA (HotpotQA) is confirmed as the
   highest-impact, highest-signal regression set.
2. **Shared-prefix prefill (§4.1)** — root-cause fix for the dominant error
   term, contained change, no vLLM involvement.
3. **Pad hygiene (§4.5)** — trivial cost, pure win.
4. **Composed partial hits + canonical ordering (§4.2)** — the lossless
   path; mostly control-plane work in `daser/server/`.
5. **Per-chunk correction (§4.4), seam patches (§4.3), background upgrade
   (§4.6)** — invest based on the residual gap the baseline shows after
   steps 2-4.

The overall stance: load-time numeric correction is the last safety net, not
the primary repair. The biggest levers are **store-time conditions** (§4.1,
which removes the dominant error source) and **combination reuse rate**
(§4.2/§4.6, which routes ever more traffic off the lossy path entirely).

## 6. Out of Scope (requires vLLM or kernel changes)

Tracked for upstream watching, not actionable inside `daser/` today:

- Token-grain selective recompute (CacheBlend-style HKVD) — needs
  non-contiguous external supply and attention-weight export.
- Per-chunk head-token recompute (LegoLink) — needs non-contiguous supply.
- In-kernel attention temperature / sink masking (APE's full recipe).
- NoPE-format KV storage with RoPE fused at attention time — removes
  relocation numerics entirely.

## 7. Accuracy Baseline Results (2026-06-10)

### 7.1 Experiment Setup

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4090 (24 GiB) |
| Model | Qwen3-8B, bf16, max_model_len=16384 |
| DaseR transfer | iouring, L1=4 GiB, L2=30 GiB |
| Dataset | LongBench QA subset: hotpotqa, 2wikimqa, musique |
| Decoding | greedy (temperature=0) |
| Max new tokens | 256 (an initial run used 64 and truncated 35–69% of reuse answers; see §7.7) |
| Max context tokens | 14,500 (filtered before chat-template expansion) |
| Script | `benchmarks/bench_accuracy_baseline.py` |

### 7.2 Service Startup

```bash
# vLLM (GPU 1, port 8021)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
vllm serve /data/zwt/model/models/Qwen/Qwen3-8B \
  --port 8021 \
  --no-enable-prefix-caching \
  --max-model-len 16384 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.92 \
  --kv-transfer-config '{"kv_connector":"DaserConnector",...}'

# DaseR HTTP server (port 2046)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
python -m daser.server \
  --vllm-base-url http://127.0.0.1:8021 \
  --model-path /data/zwt/model/models/Qwen/Qwen3-8B \
  --store-dir /data/$USER/daser-baseline/store \
  --l2-size 30gb --l1-size 4gb \
  --transfer-mode iouring \
  --cache-reuse-mode chunk \
  --socket-path /tmp/daser_baseline_$USER.sock \
  --host 127.0.0.1 --port 2046
```

### 7.3 Experiment Script

```bash
# IMPORTANT: restart DaseR (and clean daser.store) before each dataset
# to prevent ring-buffer exhaustion from cross-dataset chunk accumulation.
for dataset in hotpotqa 2wikimqa musique; do
  python benchmarks/bench_accuracy_baseline.py \
    --daser-url http://127.0.0.1:2046 \
    --tokenizer /data/zwt/model/models/Qwen/Qwen3-8B \
    --data-dir /data/$USER/daser-baseline/data \
    --datasets $dataset \
    --num-samples 20 \
    --max-context-tokens 14500 \
    --max-new-tokens 256 \
    --out /data/$USER/daser-baseline/results/${dataset}_v2.json
done
```

Each sample uploads every passage in the LongBench context as a separate
document (5–20 docs per sample) via `POST /documents`, then runs two
`POST /infer` requests with identical greedy parameters:

- `full`:  `use_kv_cache=false` — full prefill (accuracy upper bound).
- `reuse`: `use_kv_cache=true`  — concatenated chunk KV with RoPE relocation,
  no cross-attention repair (the lossy path).

Documents are kept after each sample (script default, recorded as
`delete_docs=false` in the result JSON). Keeping them avoids delete/ring-
eviction interplay within a dataset run; cross-dataset accumulation is
handled by the per-dataset DaseR restart above.

### 7.4 Results

Accuracy means are computed only over samples with a verified chunk-KV load
(N below); the one HotpotQA sample whose reuse request silently fell back to
full recompute is excluded.

| Dataset | N | Full F1 | Reuse F1 | **F1 Gap** | **Ext-F1 Gap** | Full Rec | Reuse Rec | **Rec Gap** | Agree |
|---------|---|---------|----------|------------|----------------|----------|-----------|-------------|-------|
| HotpotQA | 19/20 | **0.7338** | 0.1714 | **+0.5624** | +0.4486 | 0.6842 | 0.4211 | **+0.2632** | 5% |
| 2WikimQA | 20/20 | 0.2951 | 0.0665 | **+0.2286** | +0.2060 | 0.4500 | 0.4000 | **+0.0500** | 0% |
| MuSiQue | 16/16 | 0.2784 | 0.0138 | **+0.2646** | +0.2073 | 0.1875 | 0.0625 | **+0.1250** | 0% |

Columns:

- **F1 / Rec**: token-level QA F1 and gold-answer substring recall against
  LongBench gold answers, on the raw answer text.
- **Ext-F1 Gap**: F1 gap after extracting the concise answer span (bold
  ``**span**`` or first sentence) from both answers. The difference between
  F1 Gap and Ext-F1 Gap is the portion attributable to verbosity-induced
  token-F1 dilution rather than factual error.
- **Agree**: fraction of samples where the normalized full answer equals the
  normalized reuse answer. Near-zero is expected given the verbosity contrast
  (terse vs rambling); treat as a diagnostic, not an accuracy measure.
- **N**: samples with a TTFT-verified chunk-KV load / scored samples.

### 7.5 Key Findings

1. **Chunk reuse causes significant accuracy loss on every dataset, on every
   metric.** The raw F1 gap is +0.23 to +0.56; the extracted-answer F1 gap —
   which removes verbosity dilution — remains +0.21 to +0.45; gold-answer
   recall drops on all three datasets (+0.05 to +0.26). The loss is real
   factual degradation, not merely formatting drift.

2. **The degradation has two distinguishable components.**
   - *Factual errors*: wrong entities hallucinated from unrelated context
     ("Hafsa Hatun", "Sven Nijdam"-style substitutions), captured by the
     recall gap and the extracted-F1 gap.
   - *Instruction-following collapse*: reuse answers average 320–850
     characters versus 29–58 for full prefill, frequently rambling,
     self-contradicting ("Wait, I cannot determine…"), or failing to stop —
     14/16 MuSiQue reuse answers still hit the 256-token cap. This unbounded-
     generation behavior is itself a symptom of attention-sink damage (§2.1)
     and makes answer-length ratio a cheap regression indicator for repair
     work.

3. **HotpotQA suffers the worst (+0.56 F1, +0.26 Rec).** HotpotQA's
   multi-hop reasoning requires aggregating evidence across passages, which
   directly exercises the cross-chunk attention paths that chunk-mode reuse
   breaks. This aligns with the §2 prediction that multi-doc QA is the
   highest-impact scenario. With full-prefill F1 at 0.73, it also has the
   best signal-to-noise ratio and should be the primary regression set for
   §4 repair work.

4. **Even tasks where the model already struggles are made worse.**
   2WikimQA and MuSiQue full-prefill F1 is only ~0.28–0.30 (Qwen3-8B at
   16k context is insufficient for these tasks), but chunk reuse pushes F1
   down to 0.01–0.07 and MuSiQue recall from 0.19 to 0.06.

5. **Loaded ≈ 100% with coverage ≈ 99% confirms the gap is caused by KV
   content differences, not cache misses.** The chunks are found, loaded
   (reuse TTFT 112–208 ms vs full 800–1464 ms), and copied into the vLLM KV
   cache. The accuracy loss comes from what is **inside** those chunks:
   independently-prefilled KV with fake sinks at every chunk boundary.

### 7.6 Measurement Pitfall: Ring-Buffer Exhaustion

An initial run that processed all three datasets in a single process (without
restarting DaseR between them) reported zero F1 gap for 2WikimQA and MuSiQue.
Investigation revealed the cause:

- The ring buffer (12,715 slots, L2=30 GiB) accumulated chunks across all
  datasets. After the first dataset (HotpotQA, ~20 samples × ~10 docs × ~75
  chunks/doc ≈ 15,000 chunks), the buffer was full.
- By the time 2WikimQA and MuSiQue ran, older chunks had been evicted and
  the vLLM external prefix cache hit rate had decayed from ~30% to ~9%.
- The reuse path fell back to full prefill, producing byte-identical output
  and masking the true accuracy gap.

**Remedy:** restart DaseR (or use `--skip-l2` for volatile L1-only runs)
between datasets so every sample in the reuse path exercises a real chunk-KV
load. The `Loaded` column in the report detects this condition: a dataset
with Loaded ≪ 100% should be re-run. Since the v2 run, `summarize()` also
excludes non-loaded samples from accuracy means automatically, so isolated
fallbacks no longer dilute the gap.

### 7.7 Measurement Pitfall: Generation-Cap Truncation

The initial run used `max_new_tokens=64` and reported a HotpotQA F1 gap of
+0.53. Per-sample inspection showed that degraded reuse answers are not just
wrong but **verbose** (4–5× the character length of full-prefill answers),
and 35–69% of reuse answers were cut mid-sentence at the 64-token cap. The
truncation depressed both reuse F1 (long answers dilute token precision) and
reuse recall (the gold span can be cut off before it appears), conflating
generation-cap artifacts with genuine KV damage.

The v2 measurement therefore made three corrections to
`benchmarks/bench_accuracy_baseline.py`:

1. `max_new_tokens` raised 64 → 256 (script default), removing most
   truncation; only answers that genuinely fail to stop still hit the cap.
2. Accuracy means computed only over samples with a verified chunk-KV load
   (`reuse_loaded`), reported as `samples_loaded`.
3. Added extracted-answer F1 (bold span / first sentence) to decompose the
   raw F1 gap into factual error versus verbosity dilution, and recorded
   `delete_docs` in the result JSON for reproducibility.

The corrected numbers (§7.4) confirm the original conclusion — the v1 and v2
gaps agree within a few points — while making the metric decomposition and
the per-sample validity explicit. Result artifacts are written to
`/data/$USER/daser-baseline/results/{hotpotqa,2wikimqa,musique}_v2.json`.
