# Asynchronous H2D source ownership

An io_uring load can submit a CUDA copy and then suspend while other
operations run. Submission does not finish the DMA: every source extent must
remain immutable until an event recorded after that copy completes.

Tier extents are owned by the server's `TierIndex`. Before a load submits
its moves, the server takes an `io_pin` on the L1 or L2 copy it reads. A pinned
copy is never an eviction candidate and its extent cannot be reused. For an
L2 read that does not promote into L1, the temporary host source instead
comes from the transfer layer's `StagingPool`; its reservation remains held
through H2D completion. These are separate ownership mechanisms.

The transfer layer polls the copy event without blocking the server loop and
reports completion. The server releases tier pins and wakes allocations
waiting on `io_released`; the transfer layer releases its temporary staging
slices only after their copies finish. No extra payload snapshot is needed.

Each load awaits its own copy events before returning or propagating an
error. `IPCServer` can overlap several ready load batches with pending
promotions, but releases the request's accumulated tier pins together after
all batches have finished, including on failure. Temporary transfer staging
slices can be returned as their copies finish. Transfer drain also waits for
outstanding copy events. CPU destinations keep their synchronous copy behavior. CUDA event creation
precedes copy submission; an error after partial submission still records and
drains the completion event before the pins are released.

Stores have the inverse ownership boundary: an executor thread copies the
worker's source into the placed L1 extent or a staging slice. Cancelling the
asyncio await cannot stop that thread. The store therefore shields and drains
the copy task before propagating cancellation or releasing the extent,
including when the caller cancels repeatedly. The L1 extent stays io-pinned
until its L2 write completes. Failed snapshots release their placement and
wake waiters before returning.

Tier extents are placed and released at chunk granularity. A raw store may
populate an allocation in several requests, but the copy is not readable until
all of its records and TP ranks are valid. Another allocation cannot reuse
part of a pinned extent.

Validation covers delayed DMA under eviction pressure, L2 promotion,
cancellation and byte-exact restoration on a real CUDA device.
