# echo metrics reference

Glossary for every field exposed on `SampleInfo` and returned from
`Server.sample()`.

## Two build modes

- **Default build**: health-only. Counters and gauges always on. No
  hdrhistogram dep, no per-sample timestamps.
- **`--features detailed-metrics`**: adds CAS counters and three timing
  histograms (`memcpy`, `drain_round`, `queue_dwell`). Use for perf work.

When the feature is off, the detailed fields are still present on
`SampleInfo` but report `0` / empty `HistSnapshot`. Python code checking
`if info.memcpy.count > 0:` works uniformly across both builds.

## Metric categories

1. **Counters** (`_total` / `_count`): monotonic, always go up. Divide by run duration to get a rate.
2. **Gauges** (e.g. `store_size`, `queue_depth_sum`): snapshot values, can go up or down.
3. **Histograms** (`memcpy`, `drain_round`, `queue_dwell`): report `count`, `min_ns`, `max_ns`, `mean_ns`, `p50_ns`, `p90_ns`, `p99_ns` of a duration distribution.

## Ingestion path: what the drainer did

| Metric | Feature? | What it measures |
|---|---|---|
| `samples_inserted_total` | always | Samples the drainer has moved from SPSC to store. Drainer path only; `Server.submit()` samples are **not** counted. |
| `cas_success_total` | detailed | Successful CAS on the store's `write_cursor`; one per reservation (a reservation can cover multiple samples). |
| `cas_failure_total` | detailed | Failed CAS: two drainers raced for the same slot. The loser retries. |
| `reservation_retries_total` | detailed | Retries inside `try_reserve_slots`. Usually equal to `cas_failure_total`. |
| `cas_failure / cas_success` ratio | detailed | CAS contention rate. >10 % is bad; 0.2 % is fine. |

## Ingress (SPSC) side: what the transport is doing

| Metric | What it measures |
|---|---|
| `active_connections` | Currently-open transport connections. |
| `push_blocked_count` | Every time `SampleSender::push` hit a full SPSC queue. One park = one increment. |
| `queue_depth_sum` | Sum of all this-drainer's SPSC queue depths, sampled at the start of each drain round. |
| `queue_depth_max` | Max single-queue depth at start of round, taken across all drainers. Equal to `producer_queue_size` = at least one queue was full. |

## Consumer side

| Metric | What it measures |
|---|---|
| `store_size` | Samples committed to the ring but not yet sampled by the consumer. In FIFO, hovers between 0 and `batch_size` under balanced rates. |
| `notify_count` | Times the last-committer drainer called `notify_one` on the consumer — one per full committed batch. Should roughly equal `samples_inserted_total / batch_size`. |

## Timing histograms (detailed build only)

All three are populated only when built with `--features detailed-metrics`.
In the default build they are zero-valued `HistSnapshot`s.

### `drain_round`
Wall-clock for one complete execution of `drain_round`, recorded only
when the round did work. p50 = how long a typical round takes;
p99/tail = stalls (e.g. `insert_batch` blocked on a full ring).

### `memcpy`
Time to memcpy all samples for **one reservation** (not per-sample).
`memcpy.count` ≈ `cas_success_total` because each successful reservation
produces one memcpy record.

p99/p50 ratio on this metric is the scheduler-preemption detector:
memcpy is near-constant-time for a given byte count, so a large p99/p50
means the OS parked the thread partway through.

### `queue_dwell`
For each sample, the time from `push` into SPSC to `pop` out by the
drainer. Per-sample.

- High p50: drainer cycle is slow; samples wait long for a visit.
- High p99: drainer occasionally stalled; samples pushed during the stall
  sat through it.
- Unlike `memcpy`, this is driven by scheduling, not by per-sample work.

## Reading them together

Always-on signals first:

| Signal | Interpretation |
|---|---|
| `push_blocked_count` rising quickly | Producers hitting full SPSC queues; drainer can't keep up or SPSC is too small |
| `queue_depth_max = producer_queue_size` persistently | At least one connection's queue is full; pair with `push_blocked_count` |
| `store_size ≈ 0` | Drainer and consumer in lock-step, or consumer outpacing drainer |
| `store_size ≈ num_buffers × batch_size` | Drainer outrunning consumer; downstream (usually GPU) is the ceiling |
| `active_connections` dropping unexpectedly | Actor disconnects; look at the actor side |

Detailed-build signals:

| Signal | Interpretation |
|---|---|
| `cas_failure / cas_success > 5%` | CAS contention: too many drainers fighting for `write_cursor` |
| `memcpy.p99_ns / memcpy.p50_ns > 10×` | Scheduler preemption during memcpy |
| `queue_dwell.p50_ns` high and `drain_round.p50_ns` low | Drainer is idle most of the time, wakes rarely |
| `queue_dwell.p99_ns` and `drain_round.p99_ns` both high | Drainer occasionally stalls (ring-full backpressure) |
