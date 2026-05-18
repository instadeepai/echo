# SampleInfo

`SampleInfo` is a frozen PyO3 class returned alongside every batch
(`Sample.info`). It snapshots backpressure and throughput metrics at the
moment the batch was emitted.

See [Reading metrics](../guides/metrics.md) for what each field means and
how to interpret them.

## Fields

All fields are read-only attributes on a `SampleInfo` instance.

### Always-on (health)

| Field | Type | Meaning |
|---|---|---|
| `store_size` | `int` | Samples committed to the ring but not yet sampled by the consumer. |
| `active_connections` | `int` | Currently-open transport connections. |
| `queue_depth_sum` | `int` | Sum of all per-connection SPSC depths at the start of the last drain round. |
| `queue_depth_max` | `int` | Max single-queue depth observed in the last drain round, taken across all drainers. |
| `push_blocked_count` | `int` | Times `SampleSender::push` hit a full SPSC queue. |
| `samples_inserted_total` | `int` | Samples moved by the drainer pool from SPSC to store. Does not include `submit()`. |
| `notify_count` | `int` | Times the last-committer drainer woke the consumer (one per full batch). |

### Detailed build only

Populated only when built with `--features detailed-metrics`. In the default
build these are zero-valued. The `HistSnapshot` fields expose
`count`, `min_ns`, `max_ns`, `mean_ns`, `p50_ns`, `p90_ns`, `p99_ns`.

| Field | Type | Meaning |
|---|---|---|
| `cas_success_total` | `int` | Successful CAS on the store's `write_cursor`. |
| `cas_failure_total` | `int` | Failed CAS: two drainers raced for the same slot. |
| `reservation_retries_total` | `int` | Retries inside `try_reserve_slots`. |
| `drain_round` | `HistSnapshot` | Wall-clock per `drain_round` call that did work. |
| `memcpy` | `HistSnapshot` | Time to memcpy all samples for one reservation. |
| `queue_dwell` | `HistSnapshot` | Per-sample time from SPSC push to drainer pop. |

## HistSnapshot

| Field | Meaning |
|---|---|
| `count` | Total observations recorded. |
| `min_ns` / `max_ns` | Smallest / largest observation, in nanoseconds. |
| `mean_ns` | Arithmetic mean. Biased by outliers; prefer percentiles on skewed distributions. |
| `p50_ns` | Median. The "typical" observation. |
| `p90_ns`, `p99_ns` | Tail percentiles. `p99 / p50` is your spikiness signal. |

## Example

```python
for sample in server.dataset_iter():
    info = sample.info
    if info.push_blocked_count > 0:
        # producers are blocking, drainer can't keep up
        ...
    if info.memcpy.count > 0:  # detailed build
        print(f"memcpy p99: {info.memcpy.p99_ns / 1e6:.2f} ms")
    train_step(sample.batch)
```
