# Ring buffer

`PytreeRingBuf` (`src/ring_buf.rs`) is the raw backing store: one
`Vec<u8>` per array in the flattened pytree, sized for `capacity` slots.
There is **no synchronisation in this type at all**.

## Why no synchronisation

Adding atomics or locks to the buffer itself would mean either a global
lock around every slot access, or a per-slot atomic flag that doubles the
cacheline traffic for no real benefit (the writers already coordinate
through the `Store`'s write cursor).

Instead, the contract is that the caller (always the `Store` in this
codebase) must guarantee disjoint access. Concurrent writers reserve
different slot indices via CAS on the cursor; the consumer reads only
after writers have committed. That makes the buffer itself a dumb byte
array with `UnsafeCell` for interior mutability.

The `unsafe impl Sync for PytreeRingBuf` is sound only because of this
external contract. The safety comment in the source documents it in
detail.

## Layout

`capacity` slots, each split across `num_arrays` separate `Vec<u8>`s. Slot
`i` of array `j` lives at offset `i * slot_bytes[j]` inside `buffers[j]`.

So a 32-batch with three arrays takes three separate `Vec<u8>`s of
`32 * slot_bytes[j]` each, not one interleaved buffer. This matters for
**zero-copy reads**: `range_ptr(j, start, count)` returns a `(ptr, len)`
slice covering `count` contiguous slots of array `j`. The Python side wraps
that pointer in a numpy view with stride `slot_bytes[j]`.

An interleaved layout would force a strided numpy view or a copy to get
one array out. Separate buffers per array means contiguous, stride-free
numpy views, and the consumer never sees a slow path.

## Capacity must be a multiple of `batch_size`

The constructor asserts `capacity % batch_size == 0`. This is what
guarantees that any one batch lives in a contiguous slot range; no batch
ever spans the wrap-around point. That property is what lets the
`Contiguous { start, count }` sample result type exist at all; without it
we'd need a scatter-gather variant.

## What this type does not do

- Track which slots are in use (the `Store` does that via the write/read
  cursors).
- Notify anything (the `Sampler` does that via its condvar).
- Free anything (it's pre-allocated for the server's lifetime).
