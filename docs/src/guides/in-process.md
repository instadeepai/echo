# In-process use

When the producer and consumer live in the same Python process (no
distributed rollouts on separate machines), the network transport can be
skipped entirely. This is the right shape for unit tests, microbenchmarks,
single-process notebooks, and anything else where the "rollout worker"
and the "learner" are the same Python interpreter.

Omit `transport=` when constructing the `Server` and call `submit()`
directly:

```python
import numpy as np
from echo import Server

example = {"obs": np.zeros((4,), dtype=np.float32)}
server = Server(example, batch_size=32)  # no transport=

for i in range(64):
    server.submit({"obs": np.full((4,), i, dtype=np.float32)})

sample = server.sample()
assert sample.batch["obs"].shape == (32, 4)
```

## What this skips

When you omit `transport=`, the `Server` does **not** spin up:

- The TCP accept loop and worker threads (or any custom transport).
- The drainer pool.
- The per-connection SPSC queues.

Instead, `submit()` flattens the pytree, concatenates leaves to bytes, and
calls straight into `Store::insert_sync` on the Rust side, which spin-waits
if the ring is full. This is the cheapest path through the system.

## When to use it

- **Unit tests** that want to drive the pytree shape and the sampler logic
  without touching sockets.
- **Microbenchmarks** that want to measure the ring/sampler in isolation.
- **Single-process notebooks** where there are no rollout workers.

## What you lose

Samples submitted via `submit()` are not counted in `samples_inserted_total`;
that counter only tracks samples moved by the drainer. Most other metrics
(`store_size`, `notify_count`) are still updated by the commit path, so they
work the same way.

If you want network-style backpressure metrics, you need a transport.
