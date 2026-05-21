# TrajectoryAccumulator

Pure-Python helper for accumulating multi-timescale pytree samples on the
client side before calling `client.send`. Useful when a single rollout step
produces several arrays at different rates (e.g. one transition per env
step, plus a single episode-level statistic per episode).

Each timescale is either:

- **Buffered** — every leaf shares the same leading dim `N` (`N > 1`);
  that's the capacity, filled by `N` `add()` calls.
- **Single-item** — capacity 1; detected when every leaf is 1-d or has
  at least one leaf that is 0-d. `add()` replaces the whole leaf.

See the [guide](../guides/trajectory-accumulator.md) for the rationale and
worked examples.

::: echo.trajectory_accumulator.TrajectoryAccumulator
