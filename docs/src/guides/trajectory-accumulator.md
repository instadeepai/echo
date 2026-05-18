# TrajectoryAccumulator: the client-side adder

`TrajectoryAccumulator` is a client-side helper for accumulating multi-timescale
pytree samples before calling `client.send`. It exists because RL rollout
workers often produce data at more than one rate (one transition per env
step, plus one summary statistic per episode), and you usually want to send
them together as a single message rather than as a stream of singleton
samples.

## When to use it

- You have a rollout of `T` transitions and want to send them in one shot,
  not `T` separate `send` calls.
- You have data at several timescales (per-step, per-N-step, per-episode)
  that share a single message.

## How it works

Construct it with a dict whose top-level keys are timescale names. Each
leaf's leading dimension is the number of `add()` calls expected before the
buffer is ready to send.

```python
import numpy as np
from echo import TrajectoryAccumulator, TcpClient

T = 64  # rollout length

example = {
    "step": {
        "obs": np.zeros((T, 4), dtype=np.float32),
        "reward": np.zeros((T, 1), dtype=np.float32),
    },
    "episode": {
        "return": np.zeros((1,), dtype=np.float32),
    },
}

# The server is constructed with the same example.
client = TcpClient("localhost", 50051, example)
buf = TrajectoryAccumulator(example)

for _ in range(num_rollouts):
    episode_return = 0.0
    for _ in range(T):
        obs, reward = env.step(...)
        buf.add("step", {"obs": obs, "reward": reward})
        episode_return += float(reward)

    buf.add("episode", {"return": np.array([episode_return], dtype=np.float32)})
    client.send(buf.build())
```

## Mental model

`TrajectoryAccumulator` is just pre-allocated numpy arrays plus per-timescale slot
counters. `add()` writes into the next free slot for that timescale via
slice assignment; `build()` returns the filled pytree, flips the active
buffer (so the next round writes into a fresh one without allocating), and
resets the counters.

There's no network or Rust involvement; it's purely a way to amortise the
flatten-and-send cost across many environment steps. The pytree it returns
goes through the normal `client.send` path.

## Two buffers, no allocation per rollout

`TrajectoryAccumulator` double-buffers internally: two copies of the pytree, with
`build()` swapping the active one. So while one buffer is being serialised
and sent, the next rollout can start filling the other without any
allocation. This matters when rollouts are short relative to flatten +
network latency.

## Common pitfalls

- **Leading-dimension mismatch within a timescale.** All leaves under one
  timescale key must share the same first axis size. That's what defines
  "how many `add` calls before the buffer is full". The constructor checks
  this and raises.
- **Adding past capacity.** If you call `add` more than the leading
  dimension allows, you get `IndexError`. Call `reset()` if you want to
  abort a partial rollout.
- **Dict-only at the top level.** The top-level pytree must be a `dict`
  with timescale names. Below that, leaves can be any pytree shape that
  `optree` understands.
