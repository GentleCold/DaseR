# Word-parallel online KV codec

The online 256-scalar format uses one warp per tile. Each lane handles eight
consecutive BF16 scalars, so it can retain their primary symbols in one 24-bit
word and their secondary symbols in another word. The stored representation,
static codebooks, slot headers and raw fallback remain unchanged. The generic
kernel supports other tile sizes, but the server/worker online runtime fixes
`ONLINE_TILE_SCALARS=256`.

## Rank calculation

Every lane computes its number of primary escapes and raw escapes. Two warp
inclusive scans, minus the current lane's counts, locate the beginning of its
secondary and raw segments. Adding the persisted tile prefix produces a plane
offset. Local symbol order determines offsets within each segment.

Decode spreads the eight three-bit symbols of a word to one nibble each and
identifies escapes by intersecting the nibbles with their one-bit and two-bit
shifts, retaining bit 0 of each escaped nibble. A population count gives the
number of escapes without a serial ballot/carry round for each scalar group.
Packed words remain unsigned for logical shifts; indices and scan subtraction
use signed integers.

Inactive lanes still participate in warp scans with zero counts. Inactive
tiles never access the prefix tables. A partial primary or secondary word reads
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

The 256-scalar decoder has a single implementation. One warp decodes a tile
and one lane eight scalars. A CTA reads the plane descriptor once and decodes
32 consecutive groups of eight tiles, prefetching the next group's primary
word, two low-byte words and two prefix entries while the current group
decodes. Record fields are read as 32-bit words; two consecutive words share
their middle load. A lane writes its eight scalars as one 128-bit store when
the head dimension is divisible by eight, and one scalar at a time otherwise.

The kernel compiles without TileLang's per-access bounds guards. Containment
of a corrupt record (media bit flip, torn write or packer bug) rests on three
properties instead of the guards:

- Every global write address derives from host-validated metadata only:
  destination block IDs, the plane range and grid indices. Record bytes
  choose written values, never written locations.
- Every read addressed by record bytes (descriptor offsets, prefixes, ranks)
  stays inside `[0, staging extent)`: word reads take unsigned 32-bit
  offsets from the record's first word (a corrupt negative rank wraps high)
  and clamp both aligned word indices to the last staging word. A CTA skips
  the clamps of its prefetched reads only after checking, once, that the
  descriptor offsets bound every such read of its grid inside the staging
  tensor; reads addressed by scanned ranks are always clamped.
- Header, descriptor, codebook and metadata reads use offsets the host has
  already range-checked against the staging tensor.

A corrupt record therefore restores wrong scalars into its own destination
blocks and nothing else. A GPU fuzz test flips bits in descriptors, prefix
tables and payload and checks for CUDA faults and writes to neighbouring
blocks.

Decoding is branchless and handles a lane's eight scalars together:
per-scalar escape branches would diverge in most warps because escapes are
scattered across lanes. Codebooks are repacked to 16 bytes per plane (primary
entries in bytes 0-6, secondary in 8-14, the escape entry 7 zero) so each level
is one PRMT byte pair and a spread symbol nibble is directly a PRMT selector.
Escaped scalars take their high byte from an in-order stream: secondary
codebook entries, with a raw escape's byte gathered in from the lane's raw
window (fewer than eight bytes, loaded as one eight-byte window). A gather is
one PRMT whose selector nibbles hold each flagged scalar's escape rank,
computed for all nibbles by one multiply, and is ORed into the codebook bytes
because escape entries decode to zero. Eight high bytes therefore cost eight
PRMTs instead of a per-scalar select chain. Two PRMTs per output word
interleave low and high bytes before the 128-bit store. PRMT and the symbol
spread are emitted as inline PTX: `__byte_perm` masks every non-constant
selector, and nvcc expands a C-level spread per symbol.

On Qwen3-8B-shaped 512-tile planes of real KV (one H800 PCIe at 1755 MHz),
this restores about 845 GB/s of raw-equivalent bytes, about 79% of a
device-to-device copy of the same bytes, against about 711 GB/s for the
per-scalar select loop.

The codec does not detect corruption. GPU-packed records carry no digest of
the raw bytes (the CPU reference stores a SHA-256 that only its own decoder
checks), so lossless restore assumes the storage path returns the bytes that
were written.

All buffer budgets, stream ownership, staging leases and transfer/commit
boundaries remain unchanged. Completion still covers every destination write.
Both decoder variants (singleton and fanout) and the store bundle compile and execute during startup;
request slot count and destination fanout do not trigger JIT compilation.

The optional layerwise path requires a valid L1 prefetch lease.
`daser/compression/layer_major.py` loads the header pages and current layer
group's plane bytes, rebases the descriptors in staging, and passes
`plane_range` to the same decoder. Raw fallback records use their raw layer
ranges. Exclusive tiers do not grant leases and use ordinary full-record loads.

## Validation

The decoder also exposes `prepare(...) -> PreparedKVRestore | None` so the
load pipeline can validate a batch and upload its metadata before the batch's
payload transfer completes. Preparation deduplicates sources and launches no
payload kernel; `submit()` later decodes every source once on the prepared
stream. `decode(...)` prepares and submits in one call. The load owner must
retain the staging lease and exclusive metadata ring until an event after the
submission completes, including for a prepared plan that is abandoned before
submission. References in the plan do not prevent the owner's ring allocator
from reusing memory.

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

## Measured effect

All numbers are for GLM-4-9B-chat-1m (80 KiB of KV per token) on a single
H800.

- **Ratio.** Stored bytes are about 0.737 of raw (generic codec, 256-scalar
  tiles). The measured real-text inputs gave 0.736–0.737; other distributions
  can change the escape rate and may use raw fallback.
- **Decode throughput.** The fused decoder restores about 270 GB/s of
  raw-equivalent KV. That is well above both the H2D rate (about 55 GB/s)
  and the SSD rate (about 6.5 GB/s), so decoding is not on the critical path
  of an L1 or L2 load.
- **Long-context agent replay.** Setup: 300k–545k-token sessions, GPU KV 48 GiB,
  L1 64 GiB, L2 512 GiB. Summed TTFT over cache-hit turns fell from 357 s to
  319 s (1.12×). Wall-clock time barely moved because cold prefill dominates
  it.
- **Long-document benchmark near saturation.** Setup: GPU KV 24 GiB, L1 8 GiB,
  L2 160 GiB. Peak SLO goodput rose from 4.84 to 6.57 requests/s (1.36×),
  because the same L1/L2 capacity holds about 1.36× more KV.

Bytes only become visible when KV loading is a large share of the request
time. When prefill compute or decode dominates, the gain approaches
`1 / (1 - 0.263 * load_share)`.
