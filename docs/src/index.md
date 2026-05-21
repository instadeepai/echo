<p align="center">
  <img src="assets/logo.svg" alt="echo" width="320">
</p>

A very fast distributed replay buffer for reinforcement learning. The core
is a lockfree, pre-allocated Rust ring buffer; batches come back to Python
as zero-copy numpy views, with the GIL released while you wait.

## What it does

Distributed RL training pipelines typically have many hundreds or even
thousands of rollout workers producing samples and a small number of learners
consuming batches of them. echo exists because data transfer and
stacking is often the bottleneck of these systems. It receives individual
samples, assembles them into fixed-size batches, and serves those batches
with minimal copying and no Python-side contention.

The core is a lockfree ring buffer written in Rust. Clients push samples
into SPSC queues over the network, a pool of drainers moves data into the
ring buffer, and the consumer pulls full batches out. Pytrees are flattened
with `optree` on the client and unflattened on the Python server. Rust
itself is pytree-agnostic.

## Features

- Lockfree ring buffer, pre-allocated (no copies after ingress)
- TCP transport on the wire (custom transports — gRPC, RDMA, … — can
  be plugged into the same Rust `Transport` trait)
- Pytree-shaped samples (nested dicts/tuples of numpy arrays) via `optree`
- Zero-copy batches: `sample()` returns numpy views into Rust-owned memory
- GIL released while waiting for batches
- FIFO sampling
- Detailed metrics exposed per batch

## Where to start

- **[Getting started](getting-started.md)**: install, then a runnable example.
- **[Zero-copy batches](guides/zero-copy.md)**: ownership and lifetime
  rules for the numpy views returned by `sample()`.
- **[Python API](api/server.md)**: reference for every public class.
- **[Rust internals](design/overview.md)**: how the lockfree path is put together.
- **[Reading metrics](guides/metrics.md)**: glossary for every field on `SampleInfo`.
