# Ingress & drainers

`src/ingress.rs` is everything between the transport and the ring buffer.
It owns the SPSC queues, the drainer pool, the per-connection buffer
recycling pool, and the contract for waking producers when the queue
drains.

## The shape

```text
Conn 0 ──> SPSC 0 ──┐
Conn 1 ──> SPSC 1 ──┼── Drainer 0 ──┐
Conn 2 ──> SPSC 2 ──┘               │
Conn 3 ──> SPSC 3 ──┐               ├── store.insert_batch()
Conn 4 ──> SPSC 4 ──┼── Drainer 1 ──┘
Conn 5 ──> SPSC 5 ──┘
```

Every transport connection gets its own
`crossbeam::ArrayQueue<TransportQueueItem>`. Multiple connections fan into
a single drainer task, and a fixed number of drainer tasks fan into the
`Store`. This is what collapses N connections of contention down to N
drainers' worth.

## Why SPSC, not MPSC

An MPSC queue would be one queue total, with contention proportional to
the number of producers, potentially hundreds. A per-connection SPSC
gives each producer its own queue with no other writers, eliminating
that contention.

## Buffer recycling

Reading a sample off the network needs a `Vec<u8>` to read into. The
ingress layer recycles these buffers: every connection carries a free
pool of `Vec<u8>`s that cycle through producer → queue → drainer → back
to pool with no allocation in steady state.

The cycle:

1. The transport calls `sender.acquire(payload_size)` to get a buffer. The
   call pops one from the connection's free pool, or allocates a fresh
   `Vec<u8>` only if the pool is empty (initial fill-up or an unusually
   deep burst).
2. The transport reads `payload_size` bytes from the socket into that
   buffer, then `sender.push(buf).await` puts a
   `TransportQueueItem { data: buf, free_pool: ... }` onto the SPSC.
3. The drainer pops the item, references its `data` slice during
   `store.insert_batch`, then drops the item.
4. `TransportQueueItem::Drop` returns its `Vec<u8>` to the free pool via
   `std::mem::take(&mut self.data)`. The next `acquire` reuses it.

The buffer's `len` is preserved across the cycle; the bytes left in it
are stale, but the next read overwrites them, so the cycle skips a
`clear()` and the corresponding resize/zero-fill.

This recycling is built around the TCP transport, which calls `acquire`
before reading the socket. A custom transport that arrives with its
payload already in a `Vec<u8>` (e.g. a hypothetical gRPC transport
decoded by prost) can skip `acquire` and call `push` directly; the
`TransportQueueItem` will drop that `Vec` back into the free pool on
the first cycle and steady state reuses it from there.

### Pool sizing: `2Q + 1`

The free pool is sized to `2 * producer_queue_size + 1`. That's the
worst-case in-flight depth:

- Up to `Q` items sitting in the SPSC queue.
- Up to `Q` items the drainer is holding during `insert_batch` (one full
  round's batch).
- Plus 1 the producer is currently filling.

Sized that way, the pool never fills up before a buffer can be returned to
it, and steady state is fully allocation-free.

## Push backpressure

`SampleSender::push` is async. If the queue is full, instead of busy
looping it parks on a per-connection `Notify`:

```rust
let notified = self.space_available.notified();  // register interest BEFORE push
match self.queue.push(entry) {
    Ok(()) => { self.drainer_wake.notify_one(); return; }
    Err(rejected) => { entry = rejected; notified.await; }
}
```

The `notified()` call is registered *before* the push attempt, so a
drainer that pops and calls `notify_one` between the failed push and the
`.await` still lands as a permit, not a lost wake. This pattern matters
because the `Notify` API is permit-based; calling `notified()` after
`notify_one()` would miss the wake.

`metrics.record_push_blocked()` is incremented every time the push fails.
Sustained growth on `push_blocked_count` means the drainer can't keep up
with this connection's producer.

## Drainer round

`drain_round` is the body of one drainer iteration. It does two phases:

1. **Snapshot-bounded pop across all my queues.** For each queue, pop at
   most `queue.len()` (the depth at the moment we arrived). Cap at the
   snapshot length so a flooding producer can't starve other queues this
   round. After popping from a queue, call `space_available.notify_one()`
   so the producer wakes immediately and doesn't have to wait for the
   rest of phase 1 to finish.
2. **One `store.insert_batch` for everything.** All popped samples for
   the round go into one call. That collapses up to N CAS attempts down
   to `ceil(total / batch_size)`, the structural lower bound, because
   `try_reserve_slots` caps each reservation at the batch boundary.

`drain_round` returns `did_work`. When `false`, the drainer parks on its
own `wake` `Notify`. Producers and the transport accept loop both call
`notify_one` on it when there's work to do.

The popped `TransportQueueItem`s are held in a local `Vec` for the entire
phase 2. Their `Drop` doesn't fire until the vec goes out of scope at the
end of the round, which is exactly when the buffers are no longer needed
and can safely go back to the free pool.

## Per-round rotation

Drainers iterate their queues with an offset of `round_counter % N`. A
fixed iteration order biases per-connection latency (queue 0 always wakes
first) and lands adjacent-index samples adjacent in the ring. Rotation preserves
fairness (each queue visits each iteration position exactly once over N
rounds).

Different drainers advance their counters independently so they drift out
of phase, which keeps their `insert_batch` arrivals at the `write_cursor`
from synchronising.

## Connection pruning

`SampleSender` and the drainer's `TransportHandle` share an
`Arc<AtomicBool>` (`closed`). `SampleSender::Drop` flips it to `true`.
After each round, the drainer prunes transports whose `closed` flag is
set. This is how disconnected connections get forgotten; no explicit
signal beyond the producer-side `Drop` is needed.

When a connection's drainer entry is pruned, the per-connection free pool
goes with it (it's owned only by `SampleSender` and the in-flight
`TransportQueueItem`s, all of which are gone by then). Buffers from that
connection are simply freed.

## Detailed-metrics path

When built with `--features detailed-metrics`:

- `TransportQueueItem` carries a push-time `Instant`. The drainer
  subtracts it on pop to record per-sample `queue_dwell_ns`.
- `drain_round` records wall-clock per call (only when it did work).
- Per-reservation `memcpy_ns` is recorded inside `store.insert_batch`.

In the default build all three are no-ops with zero runtime cost. See
[Reading metrics](../guides/metrics.md).
