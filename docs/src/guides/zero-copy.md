# Zero-copy batches

`server.sample()` and `server.dataset_iter()` return numpy arrays that are
**views into memory owned by the Rust ring buffer**, not Python-owned
copies. This is what makes the consumer path fast (no allocation, no
`memcpy`, no GC pressure), but it puts two constraints on how you use the
arrays.

## The two rules

1. **Slot recycling.** Arrays returned by one `sample()` call stay valid
   only until the **next** `sample()` call on the same server. After that,
   their slots are released for drainers to overwrite. Old references
   then point at memory that is being concurrently written.
2. **Server lifetime.** The Rust ring buffer is owned by the `Server`. If
   the `Server` is closed or garbage-collected while you still hold a
   sample array, the underlying buffer is freed and the array becomes a
   dangling pointer. Keep the `Server` alive at least as long as any
   sample arrays you've kept.

## What this means in practice

- **Read, compute, train, then iterate.** The normal training loop
  consumes the batch before fetching the next one.
- **Pass the batch into a framework that copies on entry.** Most
  optimised paths (host-to-device transfers, `jnp.asarray`, etc.) copy
  the data, after which the original view is no longer needed.
- **Don't** store batches in a list across iterations.
- **Don't** hand a batch off to another thread that may outlive the
  current iteration.
- **Don't** close the `Server` while any sample arrays are still in use.

```python
# Wrong: second batch silently overwrites the first.
batches = []
for sample in server.dataset_iter():
    batches.append(sample.batch)

# Right: copy if you need to keep.
batches = []
for sample in server.dataset_iter():
    batches.append(optree.tree_map(np.copy, sample.batch))
```

## Getting a copy when you need one

`dataset_iter` takes a `copy=True` flag that deep-copies every batch
before yielding it. Arrays are then safe to store indefinitely:

```python
for sample in server.dataset_iter(copy=True):
    keep_for_later.append(sample.batch)
```

For one-off copies, `optree.tree_map(np.copy, batch)` or `np.copy(arr)`
on individual leaves works fine.

## Why it works this way

The Rust ring buffer is pre-allocated at server construction. Drainers
write into slots and the consumer reads from slots; both sides agree on
ownership via cursors, not via Python reference counts. The numpy arrays
yielded to Python are constructed with `NPY_ARRAY_OWNDATA` *unset*, so
numpy knows not to free the memory and trusts the producer to keep it
alive.

Once `sample()` is called again the consumer's previous batch is
released back into the writable pool, and the next round of drainers
can memcpy on top of it. That's how the system avoids per-sample
allocation entirely.

If you need every batch to outlive the next, you're trading the
zero-copy property for safety, and `copy=True` is the explicit way to do
that.
