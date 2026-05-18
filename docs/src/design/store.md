# Store

`src/store.rs` is the composition layer. It owns the pre-allocated
[`PytreeRingBuf`](ring-buffer.md), a boxed [`Sampler`](selector.md), and a
boxed `Remover`, and exposes the actual insertion strategy used by both
the drainer pool and the direct `submit()` path. The path from SPSC pop
to consumer wake lives here.

## Cursors

Three cursors govern the ring:

| Cursor | Owner | Role |
|---|---|---|
| `write_cursor` (atomic) | drainers (and `submit`) | Next absolute slot to reserve. Advanced by CAS. |
| commit counters (atomic, in `FifoSampler`) | drainers | Per-batch-slot fill counter; wake when full. |
| `read_cursor` (in `FifoRemover`) | the single consumer | Next absolute slot to read. |

Cursors are absolute (`usize` that monotonically grows) and only modded
into `[0, capacity)` when actually addressing the ring. That keeps wrap-
around bookkeeping out of the comparisons.

## The reserve-memcpy-commit dance

`insert_batch` is the hot path for drainers. One round of the loop:

1. **Reserve.** Call `try_reserve_slots(want)`. This does a single CAS on
   `write_cursor` to claim up to `want` contiguous slots, capped at the
   batch boundary so the reservation never spans two batch-slots. Returns
   `(start, n)` on success.
2. **Memcpy.** For each of the `n` slots, copy each array of the sample
   straight into the ring slot. `slot_mut` returns a raw pointer; nothing
   synchronises because the reservation guarantees disjoint access.
3. **Commit.** Call `sampler.commit_batch(start, n)`, one `fetch_add(n)`
   on the batch-slot counter. The drainer whose increment lands the
   counter at `batch_size` wakes the consumer (see
   [Selector](selector.md)).

If `try_reserve_slots` returns `None` (ring is full), the caller awaits on
`space_available` (a `tokio::sync::Notify`) and tries again. The consumer
wakes producers by calling `space_available.notify_waiters()` from
`release_previous_batch`.

## Why cap reservations at the batch boundary

`try_reserve_slots` reserves `min(want, available, in_batch_remain)` slots.
That last term is what keeps reservations from crossing a batch boundary.

The reason: commit is a `fetch_add(n)` on one batch-slot counter. A
reservation that straddled two batches would require two `fetch_add`s and
the wake detection (which drainer's commit brings the counter to
`batch_size`) would become ambiguous. Two drainers could each commit
half of a batch and neither would observe `counter == batch_size` for
the half already finished by the other drainer. Capping at the boundary
means one reservation, one batch-slot, one commit, unambiguous wake.

The cost is that a drainer with `2*batch_size` samples queued might do
two reservations instead of one. Each is still a single CAS, and the
second one starts from `write + n` which is the next batch-slot's first
position, so the CAS retries are bounded by genuine contention, not by
this rule.

## Two insert paths

`insert_batch` (async) is the drainer path. It awaits on `space_available`
when the ring is full.

`insert_sync` is the in-process `submit()` path. It spins on a CAS retry
loop without going through tokio. This is fine because in-process use is
single-threaded by construction, so the ring is never full long enough
for spinning to matter.

Both go through the same `write_slot` / `try_reserve_slots` CAS, so the
correctness story is identical.

## Releasing slots

The consumer doesn't release slots when it reads; it releases the
*previous* batch's slots on entry to the next `sample()` call. That's why
`Store` carries `has_previous_batch: Cell<bool>`: it's how the consumer
remembers there's a batch outstanding from last time.

This matters because the numpy views from `sample()` are still pointing
into those slots until the *next* `sample()` is called; see
[Zero-copy batches](../guides/zero-copy.md). Releasing eagerly would let a
drainer overwrite a slot that the learner is still reading.

`release_previous_batch` calls `remover.remove(batch_size)`, which advances
`read_cursor`. If the ring was full at the time of release, it also calls
`space_available.notify_waiters()` to unpark any blocked inserters.

## Why `Cell<bool>` is sound

`has_previous_batch` is a `Cell<bool>`, non-`Sync`. That's only OK because
*only the consumer touches it*, and there's only one consumer. The
`unsafe impl Sync for Store` is conditional on this invariant. If
multi-consumer sampling is ever added, this field needs to become an
atomic or move into the consumer's stack frame.
