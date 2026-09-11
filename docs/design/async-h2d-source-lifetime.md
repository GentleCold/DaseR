# Asynchronous H2D source ownership

An io_uring load can submit a CUDA copy and then suspend while other L1
operations run. Submission does not finish the DMA: every source allocation
must remain immutable until an event recorded after that copy completes.
The IPC destination synchronization alone cannot protect unleased L1 sources.

The transfer layer borrows the existing source-slice reference counts before
submitting each grouped CUDA copy. This is the same ownership registry used
by request leases, so eviction and overwrite preserve a borrowed allocation.
A transfer-owned event task polls the copy event without blocking the server
loop, releases the references under the metadata lock and wakes pool waiters.
No payload is copied again and no extra pinned or GPU capacity is allocated.

Each load awaits its own copy tasks before returning or propagating an error.
Completed copies release memory independently while later L2 reads are still
pending, including when a request exceeds L1 capacity. Transfer drain also
waits for outstanding copy tasks. CPU destinations keep their synchronous
copy behavior. CUDA event creation precedes copy submission; an error after
partial submission still records and drains the completion event.

The server retains its existing IPC synchronization and request-lease release
order. Worker staging, decode and vLLM completion contracts are unchanged.
Validation covers delayed DMA with L1 replacement, overlapping stores, L2
promotion, cancellation and byte-exact restoration on a real CUDA device.

Packed stores have the inverse ownership boundary: an executor thread copies
the worker's source into a reserved pinned destination. Cancelling the asyncio
await cannot stop that thread. The store therefore shields and drains the copy
task before propagating cancellation or freeing either buffer, including when
the caller cancels repeatedly. Failed snapshots release their reservation and
wake pool waiters before returning, so unrelated stores can reuse the pages.

Overwriting part of a resident range preserves the untouched fragments as
reference-counted child slices of its existing allocation. The children are
created before releasing the old resident, and have separate identities from
any writer or DMA owner still borrowing the original slice. No payload is
copied under the metadata lock. Resident byte counters exclude the overwritten
holes; the fixed pinned pool still charges the entire original allocation
until its final child or external owner closes. Replacement continues to evict
whole allocations, so a small surviving fragment can temporarily reduce
effective capacity but cannot permit premature reuse of an in-flight buffer.
