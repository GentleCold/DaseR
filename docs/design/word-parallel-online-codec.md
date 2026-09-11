# Word-parallel online KV codec

The online 256-scalar format uses one warp per tile. Each lane handles eight
consecutive BF16 scalars, so it can retain their primary symbols in one 24-bit
word and their secondary symbols in another word. The stored representation,
static codebooks, slot headers, side indexes and raw fallback remain unchanged.
Other tile sizes and generic persisted formats retain their existing kernels.

## Rank calculation

Every lane computes its number of primary escapes and raw escapes. Two warp
inclusive scans, minus the current lane's counts, locate the beginning of its
secondary and raw segments. Adding the persisted tile prefix produces a plane
offset. Local symbol order determines offsets within each segment.

Decode identifies three-bit escape symbols by intersecting a word with its
one-bit and two-bit shifts, retaining bit positions 0, 3, 6, and so on. A
population count gives the number of preceding escapes without a serial
ballot/carry round for each scalar group. Packed words remain unsigned for
logical shifts; indices and scan subtraction use signed integers.

Inactive lanes still participate in warp scans with zero counts. Inactive
tiles never access the side index. A partial primary or secondary word reads
only the bytes containing live symbols. Secondary words hold at most 24 data
bits plus a seven-bit initial offset, which fits in one 32-bit register.

## Store and load ownership

The escape-count stage owns one complete slot/plane per CTA. It retains tile
counts in shared memory, then scans groups of 256 tiles after a block barrier.
Warp scans and a scan of the eight warp totals produce the original two prefix
tables. Inactive threads contribute zero and participate in every barrier;
each group finishes reading shared totals before the next group overwrites them.
A uniform carry connects groups, including partial final warps. This removes
the separate prefix launch and the two persistent device count arrays.
Planes mark their slot's overflow flag with an atomic maximum after the existing
same-stream zero initialization. The following layout launch therefore sees all
plane totals and overflow decisions without changing the stored format.

The layout stage assigns one warp to the ordered slot prefix. Within a slot,
lanes scan groups of plane payload lengths; inactive lanes contribute zero.
The final lane's inclusive sum advances the uniform record cursor before the
next plane group. All lanes retain the same staging cursor, and only lane zero
publishes slot metadata. A slot that exceeds its envelope clears every tentative
plane descriptor and advances by the raw stride. This preserves the original
byte offsets and alignment without a host metadata round trip or extra buffer.

The compact store stage reads the live KV tensor after the existing producer
event. A lane forms primary and secondary words from its eight scalars, writes
low bytes and primary bytes, and contributes its secondary word to the existing
zeroed escape scratch. Contributions use disjoint bit ranges; a word crossing
a 32-bit boundary is split between two atomic additions. Their sum cannot
carry into another lane's bits. The existing CTA barrier precedes final escape
byte emission. Raw escape bytes have unique writers in original scalar order.
Invalid token tails encode canonical zero bits.

The decoder retains eight restored values in registers and writes each target
directly in the vLLM KV cache. The fixed geometry permits vector stores where
aligned. Multiple destinations reuse the same decoded values; they do not
require an intermediate raw payload.

All buffer budgets, stream ownership, staging leases and transfer/commit
boundaries remain unchanged. Completion still covers every destination write.
Both decoder variants and the store bundle compile and execute during startup;
request slot count and destination fanout do not trigger JIT compilation.

## Validation

The decoder also exposes `prepare(...) -> PreparedKVRestore | None` for
experiments that keep payload IO intact while varying GPU submission size.
Preparation validates and deduplicates sources, uploads metadata once, and
launches no payload kernel. `submit_next(max_sources)` submits up to a positive
number of remaining sources on the prepared stream and returns that number;
it returns zero after exhaustion. The existing `decode(...)` entry point
prepares and submits all sources at once, preserving the production policy.

Source segments are tensor views of the original allocation. A fanout segment
keeps the full destination vector because its CSR offsets remain absolute;
singleton segments slice their destination vector with their sources. No raw
intermediate tensor or additional payload copy is introduced. The load owner
must retain the staging lease and exclusive metadata ring until an event after
the last submission completes, including a failed or abandoned partial plan.
References in the plan do not prevent the owner's ring allocator from reusing
memory. Queue-depth control belongs to the caller and is not implicit in this
API. Different segment lengths use the warmed dynamic-shape kernels.

Tests compare restored KV with the original bytes and read GPU-produced packed
records with an independent CPU decoder. Cases include sparse escapes, full
secondary streams, consecutive raw escapes, raw overflow, non-contiguous
destinations, repeated metadata-ring use, partial tiles and partial token
tails. The consecutive-escape pattern has an odd period so it crosses lane,
byte and tile boundaries while the whole slot remains compressed.
The mixed-batch test also covers a nonempty partial second warp group of planes
and raw fallback followed by further packed records. A 258-tile plane covers
carry between block-scan groups and a partial final warp. CUDA memory and shared
memory race checks cover the fused count/prefix boundaries.

Performance acceptance requires a complete endpoint workload in addition to
the real-KV microbenchmark and concurrent-compute screen. A kernel speedup
alone does not establish a TTFT improvement.
