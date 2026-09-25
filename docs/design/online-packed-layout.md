# Online packed FIFO layout experiment

Online packed records currently occupy separate raw slot envelopes for rolling
prefix keys. This experiment coalesces newly written, source-contiguous records
in consecutive live ring slots, without a separate variable-length allocator.
The DaseR server owns placement; the worker still supplies exact record lengths
and the server publishes physical references only after transfer completes.

For a run of slots `[a, b)`, reserve its existing raw envelope and put the records
contiguously at its **end**. Every record is at most one raw slot long. Therefore
record `i` begins at or after raw slot `i`, and ends within raw slot `b - 1`.
FIFO reclaims record `i` before reclaiming any later slot that contains its bytes.
This is the reason tail alignment is safe; packing toward the beginning would
let a newer record occupy an earlier slot that can be reused while it is live.

Runs must stop at source gaps, logical gaps, physical wrap, the current FIFO tail,
and any already assigned record. Crossing the FIFO tail could combine a new
allocation with an older allocation whose physical slot happens to follow it.
Existing allocation envelopes still validate worker input before relocation.
The resulting range must stay inside the union of the run's raw envelopes.

The layout owner remembers a placement per rank and logical slot, together with
the actual `ChunkMeta` object that owns that generation. An IPC retry, including
a partial retry with a different grouping, reuses the original placement. A
different length or encoding for the same generation fails before any write.
Reusing a slot with a new metadata object replaces its placement. State is
bounded by the existing logical slot count; there is no payload copy, added
capacity, compaction pass, independent free list, or change to raw storage.
As with the current online physical index, this state is specific to the live
server; it does not define a new persistent-store recovery format.

Placement is synchronous control-plane bookkeeping with no suspension between
validation and reservation. Transfer error retains the reservation so retries
remain stable. Original staging leases, L2 write ordering, committed-chunk
visibility and load completion barriers still apply. Delayed writes continue
to pass the existing current-allocation check before transfer.

Validation must cover all-live byte preservation across repeated FIFO reuse,
partial/reordered retries, shrinking and expanding records in new generations,
tail/wrap boundaries, exact capacity, invalid input without state mutation, and
real transfer round trips. Endpoint measurements must verify that dispatch and
physical read counts fall, in addition to complete output equality and TTFT.
This is an experimental layout mechanism, not a measured performance claim.
