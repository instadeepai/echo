# TrajectoryAccumulator

Pure-Python helper for accumulating multi-timescale pytree samples on the
client side before calling `client.send`. Useful when a single rollout step
produces several arrays at different rates (e.g. one transition per env
step, plus a single episode-level statistic per episode).

::: echo.trajectory_accumulator.TrajectoryAccumulator
