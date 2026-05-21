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
timescale is one of two kinds, inferred from the example pytree:

- **Buffered** — every leaf shares the same leading dim `N` (with `N > 1`).
  That leading dim is the timescale's capacity: the buffer fills after `N`
  `add()` calls and each call writes into `stored[s:s+1]`.

- **Single-item** — the timescale holds one trailing piece of context
  (e.g. an episode return, a bootstrap step) rather than a buffer of
  steps. Detected when at least one leaf is 0-d, *or* all leaves have
  `shape[0] == 1`. Capacity is `1` and `add()` replaces the whole leaf,
  so non-0-d leaves can carry any per-item shape (apart from the
  optional leading 1).

```python
import numpy as np
from echo import TrajectoryAccumulator, TcpClient

T = 64  # rollout length

example = {
    # Buffered timescale: leading dim T across every leaf.
    "step": {
        "obs": np.zeros((T, 4), dtype=np.float32),
        "reward": np.zeros((T,), dtype=np.float32),
    },
    # Single-item timescale: all leaves have shape (1, ...) — capacity 1,
    # `add()` replaces the whole leaf.
    "episode": {
        "return": np.zeros((1,), dtype=np.float32),
        "length": np.zeros((1,), dtype=np.float32)
    },
    # Single-item timescale: 0-d reward means add() replaces the whole leaf
    "final_step": {
        "obs": np.zeros((4,), dtype=np.float32),
        "reward": np.zeros((), dtype=np.int32),
    },
}

# The server is constructed with the same example.
client = TcpClient("localhost", 50051, example)
buf = TrajectoryAccumulator(example)

for _ in range(num_rollouts):
    episode_return = 0.0
    reward = 0.0
    obs = env.reset()
    for _ in range(T):
        buf.add("step", {"obs": obs, "reward": reward})
        obs, reward, ... = env.step(...)
        episode_return += float(reward)

    buf.add("episode", {"return": np.array([episode_return]), "length": np.array([length])})
    buf.add("final_step", {"obs": obs, "reward": reward})
    client.send(buf.build())
```

## Mental model

`TrajectoryAccumulator` is just pre-allocated numpy arrays plus per-timescale
slot counters. For buffered timescales, `add()` writes into the next free
slot via slice assignment (`stored[s:s+1] = incoming`); for single-item
timescales it replaces the whole leaf (`stored[...] = incoming`).
`build()` returns the filled pytree and resets the counters. The buffer is
reused — no allocation per rollout.

There's no network or Rust involvement; it's purely a way to amortise the
flatten-and-send cost across many environment steps. The pytree it returns
goes through the normal `client.send` path.

The tree returned by `build()` aliases the accumulator's internal buffers,
so the next `add()` will overwrite it. The usual `client.send(buf.build())`
pattern is safe because `send` is synchronous and copies the bytes before
returning — don't hold onto the returned tree across further `add()` calls.

## Common pitfalls

- **Leading-dimension mismatch in a *buffered* timescale.** Inside a
  buffered timescale, all leaves must share the same first axis size —
  that's what defines "how many `add` calls before the buffer is full".
  The constructor checks this and raises. If you actually meant a
  single-item timescale, make one leaf 0-d or add a leading dim to all leaves.
- **Adding past capacity.** If you call `add` more times than the
  timescale's capacity, you get `IndexError` with the timescale name,
  capacity, and offending index. Call `reset()` to abort a partial
  rollout.
- **Dict-only at the top level.** The top-level pytree must be a `dict`
  with timescale names. Below that, leaves can be any pytree shape that
  `optree` understands.
