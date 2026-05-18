# Getting started

## Install

```bash
pip install id-echo
```

Requires Python 3.10+.

## Quickstart

A typical setup runs the server on the learner node and clients on rollout
workers (separate processes or machines).

The `example` pytree defines the contract for the rest of the pipeline.
Its structure and per-leaf shape/dtype must **exactly** match what every
`client.send(...)` call sends. Both ends validate it on connection.

### Learner side

```python
import numpy as np
from echo import Server, TcpTransport

example = {
    "obs": np.zeros((4,), dtype=np.float32),
    "reward": np.zeros((1,), dtype=np.float32),
}

server = Server(example, batch_size=32, transport=TcpTransport(port=50051))
server.start()

for sample in server.dataset_iter():
    batch = sample.batch  # {"obs": (32, 4), "reward": (32, 1)}
    # ... feed to training step
```

### Rollout worker side

```python
import numpy as np
from echo import TcpClient

example = {
    "obs": np.zeros((4,), dtype=np.float32),
    "reward": np.zeros((1,), dtype=np.float32),
}

client = TcpClient("learner-host", 50051, example)
for _ in range(1000):
    client.send({
        "obs": np.random.randn(4).astype(np.float32),
        "reward": np.array([1.0], dtype=np.float32),
    })
client.close()
```

`sample.batch` is a pytree with the same structure as `example`, but every
leaf has shape `(batch_size, *leaf.shape)`.

!!! warning "These arrays are zero-copy views, not copies"
    Leaves of `sample.batch` point directly into Rust-owned ring-buffer
    memory and are **invalidated on the next iteration**. Don't store
    references across loop iterations without copying. See
    [Zero-copy batches](guides/zero-copy.md) for the full rules.

## Accumulating multi-step rollouts

If a rollout worker produces several transitions per step, use
[`TrajectoryAccumulator`](guides/trajectory-accumulator.md) to fill a
pre-allocated pytree and send it as one message:

```python
from echo import TrajectoryAccumulator

T = 16
transition_example = {
    "obs": np.zeros((T, 4), dtype=np.float32),
    "reward": np.zeros((T,), dtype=np.float32),
}
buf = TrajectoryAccumulator({"step": transition_example})
for _ in range(T):
    buf.add("step", {
        "obs": np.zeros((4,), dtype=np.float32),
        "reward": np.float32(0.0),
    })
client.send(buf.build())
```

See the [TrajectoryAccumulator guide](guides/trajectory-accumulator.md)
for details.
