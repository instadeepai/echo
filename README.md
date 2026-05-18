# echo

<p align="center">
  <a href="https://instadeepai.github.io/echo/"><img src="https://img.shields.io/badge/docs-instadeepai.github.io%2Fecho-blue?style=flat-square&logo=materialformkdocs" alt="Docs"></a>
  <a href="https://pypi.org/project/id-echo/"><img src="https://img.shields.io/pypi/v/id-echo?style=flat-square&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://github.com/instadeepai/echo/blob/main/pyproject.toml"><img src="https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Finstadeepai%2Fecho%2Fmain%2Fpyproject.toml&style=flat-square&logo=python&logoColor=white&label=python" alt="Python"></a>
</p>

A very fast distributed replay buffer for reinforcement learning. The core
is a lockfree, pre-allocated Rust ring buffer. Batches come back to Python
as zero-copy numpy views, with the GIL released while you wait for the next
batch.

<p align="center"><a href="https://instadeepai.github.io/echo/"><b>📖 Documentation</b></a></p>

## Install

```bash
pip install id-echo
```

## What it does

Distributed RL training pipelines typically have many hundreds or even
thousands of rollout workers producing samples and a small number of learners
consuming batches of them. `echo` exists because I found that data transfer
and stacking was often the bottleneck of these systems. `echo` receives
individual samples, assembles them into fixed-size batches, and serves those
batches with minimal copying and no Python-side contention.

The core is a lockfree ring buffer written in Rust. Clients push samples into
queues over the network, a pool of drainers moves data into the ring
buffer, and the consumer pulls full batches out. Pytrees are flattened with
`optree` on the client and unflattened on the python server. Rust itself is
pytree-agnostic.

## Features

- Lockfree ring buffer, pre-allocated (no copies after ingress)
- Extensible transports, ships with a custom TCP protocol (extendible to RDMA, gRPC, etc)
- Pytree-shaped samples (nested dicts/tuples of numpy arrays) via optree
- Zero-copy batches: `sample()` returns numpy views into Rust-owned memory
- GIL released while waiting for batches
- FIFO sampling (with more strategies planned)
- Detailed metrics exposed per batch

## Example

A typical setup runs the server on the learner node and clients on rollout
workers (separate processes or machines).

**Learner side** (one process):

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

**Rollout worker side** (one or many processes):

```python
import numpy as np
from echo import TcpClient

example = {
    "obs": np.zeros((4,), dtype=np.float32),
    "reward": np.zeros((1,), dtype=np.float32),
}

client = TcpClient("localhost", 50051, example)
for _ in range(1000):
    client.send({
        "obs": np.random.randn(4).astype(np.float32),
        "reward": np.array([1.0], dtype=np.float32),
    })
client.close()
```

For in-process use (tests, benchmarks), omit the transport and call
`server.submit(data)` directly.

## Development

```bash
cargo test                       # Rust tests
uv run pytest python/tests/ -v   # Python tests
just bench                       # Benchmarks
just                             # List all commands
```

## Name

The name is a nod to [Reverb](https://github.com/google-deepmind/reverb),
DeepMind's RL replay buffer that inspired this project, an echo being a
faster, simpler kind of reverberation.
