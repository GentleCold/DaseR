# Deferred stores across request preemption

The worker accumulates immutable store specifications after forward execution,
then submits them when vLLM reports request completion. Preemption can recycle
the original KV blocks before completion. Dropping only the scheduler's pending
state leaves a worker specification referencing those recycled blocks.

Scheduler metadata therefore carries the base IDs of preempted requests whose
store specifications were already published. The worker consumes those IDs in
the public `handle_preemptions` hook, before vLLM can overwrite source blocks.
This hook also runs in no-forward steps, which skip `wait_for_save`; consuming
cancellation in the save hook would leave stale specifications in those steps.
It discards only unsent
stores and releases their writer claims through asynchronous public IPC, using
the original allocation identities. Cancellation never reports the old request
as a completed send. A resumed request may publish fresh specifications.

Writer-release futures are independent of request-completion futures. Worker
polling reaps completed releases without blocking, surfaces failures, and never
mistakes release completion for permission to free model KV blocks. Shutdown
drains outstanding releases before closing the IPC connection. Already submitted
finished stores keep their existing snapshot/transfer completion contract.

This repairs the deferred-save path. It does not make arbitrary prefill-time
background packing safe: early submission additionally needs immutable snapshots
or protection before preempted source blocks are overwritten. In particular,
waiting on a CUDA event that has not been recorded cannot establish that lifetime.
