# Rust internals: overview

This section walks through how the Rust side is put together. The goal is
to give a reader enough of a map to navigate the source, not to repeat what
the comments in each file already say.

## What it has to do

A learner produces one batch every training step. Rollout workers (possibly
hundreds of them) produce one sample every env step. The Python-side
flatten/unflatten is unavoidable; everything else is bounded by memory
bandwidth.

That gives a few constraints:

- **No per-sample allocation on the consumer side.** Pre-allocate one
  contiguous buffer per array, slice into it.
- **No mutex around the hot path.** Many producers, one consumer; use a
  CAS on a write cursor and per-batch atomic counters.
- **Drop the GIL while blocking.** The consumer's `sample()` parks on a
  Rust condvar, not on a Python lock.

## Data flow

```text
Python actor                Rust server                       Python learner
─────────────              ────────────────────              ──────────────
client.send(pytree)
  │ flatten + concat
  ▼
  ──── TCP ─────────►  transport task
                        │ push(bytes)
                        ▼
                      SPSC queue (one per connection)
                        │
                        ▼ [drainer wakes via Notify]
                      drainer task
                        │ pop, batch, try_reserve_slots (CAS)
                        │ memcpy into ring
                        │ commit_batch (fetch_add)
                        ▼
                      ring buffer (pre-allocated)
                        │ [last committer notifies consumer]
                        └────────── wake ──────────────────►  server.sample()
                                                                │ build numpy views
                                                                ▼
                                                              sample.batch (zero-copy)
```

## Module map

| File | Role |
|---|---|
| [`transport/`](transport.md) | Accept connections, push raw bytes into per-connection SPSC queues. |
| [`ingress.rs`](ingress.md) | Drainer pool. Owns the SPSC queues, drains them, calls `store.insert_batch`. |
| [`store.rs`](store.md) | Composes the ring buffer, sampler and remover. Does the CAS-reserve-then-memcpy dance. |
| [`ring_buf.rs`](ring-buffer.md) | Raw pre-allocated buffer with `UnsafeCell` interior mutability; no synchronisation of its own. |
| [`selector.rs`](selector.md) | `Sampler` and `Remover` traits, plus the FIFO implementation that does the commit-counter wake. |
| `metrics.rs` | Per-drainer counters + optional hdrhistograms. See [Reading metrics](../guides/metrics.md). |
| `py_bindings.rs` | PyO3 layer. Constructs the `Store`, picks the transport, releases the GIL during `sample()`. |
| `array_spec.rs` | Tiny value type: shape + dtype size per leaf. |

## Key design choices

- **Pytree-agnostic Rust.** The server receives shapes and dtype sizes at
  construction, then only raw bytes. All structure handling stays in
  Python where `optree` does it well and the Rust hot path stays simple.
- **One memcpy per sample, ever.** Bytes are copied off the socket into
  a `TransportQueueItem`, popped, then memcpy'd into the ring. The numpy
  arrays yielded to Python are pointers into the ring; no further copy
  unless the caller asks for one.
- **Steady-state zero-allocation ingress.** The transport-side `Vec<u8>`
  buffers cycle through a per-connection free pool sized for the
  worst-case in-flight depth, so after a brief warm-up the ingress path
  does no `malloc`/`free`. See [Ingress & drainers](ingress.md).
- **Per-connection SPSC instead of one MPSC.** One queue per connection
  means producers never contend with each other on the push path. The
  drainers fan in from many SPSCs to one ring, and contention shows up
  only at the `write_cursor` CAS, which is one CAS per *reservation*,
  not per sample.
- **Drainer pool, not one drainer per connection.** A few hundred
  connections collapse to a fixed N drainers. Each drainer owns a subset
  of SPSC queues and visits them in a per-round-rotated order; see
  [Ingress & drainers](ingress.md).
- **Commit-counter wake.** Drainers `fetch_add` into a per-batch-slot
  atomic counter; whichever drainer's increment brings the counter to
  `batch_size` wakes the consumer. Exactly one notify per batch, no
  scanning. See [Selector](selector.md).
