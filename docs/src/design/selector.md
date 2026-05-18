# Selector

`src/selector.rs` defines two traits (`Sampler` and `Remover`) and one
FIFO implementation of each. The split is deliberate: a future
`PrioritisedSampler` and a future `LRURemover` will sit behind the same
traits without touching the rest of the codebase.

Today, only FIFO is implemented.

## The commit-counter trick

The interesting part is `FifoSampler::commit` / `select`. It has to
satisfy three constraints simultaneously: drainers commit samples
concurrently in any order; the consumer must wake **exactly once per full
batch** and never on a partial one; and detecting "this batch is full"
cannot involve scanning all in-flight samples (that would be
O(batch_size) per commit).

The implementation uses one atomic counter **per batch-slot in the ring**
(so `num_buffers` counters total). When a drainer commits position `pos`:

```rust
let batch_id = (pos / batch_size) % num_buffers;
let prev = self.batch_counts[batch_id].fetch_add(1, AcqRel);
if prev + 1 == batch_size {
    // last committer for this batch-slot; exactly one drainer hits this
    self.cv.notify_one();
}
```

The drainer whose `fetch_add` brings the counter from `batch_size - 1` to
`batch_size` is unambiguously the last committer for that batch-slot.
That's the one that takes the (uncontended) mutex and signals the
condvar. No scanning, no wake storm.

The consumer's `select`:

```rust
let batch_id = (read_cursor / batch_size) % num_buffers;
if batch_counts[batch_id].load(Acquire) >= batch_size {
    batch_counts[batch_id].store(0, Release);  // reset for next cycle
    break;
}
```

## Why the reset is race-free

The counter is reset to zero by the consumer **after** it observes
`>= batch_size`. The next set of drainers that will write into this
batch-slot can only start incrementing it once `read_cursor` advances past
this batch, which only happens after the consumer calls
`release_previous_batch` on the next `sample()` call. And drainers can't
advance past `read_cursor` because of the `Store::write_cursor`
backpressure.

So the order is always:

1. Drainers fill batch N's counter to `batch_size`.
2. Consumer reads, resets counter, advances `read_cursor` (next `sample`).
3. Drainers can start writing into batch N again.

No drainer can be mid-increment during step 2.

## `commit_batch`

When a drainer reserves `n` contiguous slots (via `try_reserve_slots`), it
calls `commit_batch(start, n)` instead of `n` separate `commit` calls.
That's one `fetch_add(n)` instead of `n` × `fetch_add(1)`. The contract is
that `[start, start+n)` must stay within one batch, which
`try_reserve_slots` already guarantees by capping `n` at the batch
boundary.

## Acquire/Release ordering

- Drainer's `fetch_add(AcqRel)` releases the memcpy that preceded it.
- Consumer's `load(Acquire)` synchronises with that release, so by the
  time it sees `count >= batch_size`, every memcpy is visible.
- The Mutex + Condvar is only there to park the consumer while it waits;
  the actual handshake is via the atomic.
