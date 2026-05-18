"""Distributed end-to-end benchmark using Ray.

Sweeps the TCP transport across actor counts, measuring latency and
throughput of the full batching pipeline.
"""

from __future__ import annotations

import math
import os

import numpy as np


BATCH_SIZE = 2048
ROLLOUT_LEN = 5
NUM_BATCHES = 50
WARMUP_BATCHES = 3
ACTOR_COUNTS = [1, 10, 100, 500, 1000]

IS_LOCAL_TEST = True
LOCAL_ACTOR_COUNTS = [1]
LOCAL_NUM_BATCHES = 3
LOCAL_BATCH_SIZE = 4

BASE_PORT = 54000
SWEEP_TIMEOUT_S = 300.0


def _rl_sample() -> dict:
    """A very large RL observation."""
    return {
        "action_mask": np.random.randn(10, 17).astype(np.float32),
        "feature_1": np.random.randn(2).astype(np.float32),
        "feature_2": np.random.randn(4).astype(np.float32),
        "feature_3": np.random.randn(3).astype(np.float32),
        "feature_4": np.random.randn(80, 2).astype(np.float32),
        "feature_5": np.random.randn(80, 2).astype(np.float32),
        "feature_6": np.random.randn(80, 1).astype(np.float32),
        "feature_7": np.random.randn(80, 8).astype(np.float32),
        "feature_8": np.random.randn(80, 2).astype(np.float32),
        "feature_9": np.random.randn(80, 6).astype(np.float32),
        "feature_10": np.random.randn(80, 2).astype(np.float32),
        "feature_11": np.random.randn(80).astype(np.float32),
        "feature_12": np.random.randint(0, 10, (80,), dtype=np.int32),
        "feature_13": np.random.randn(80, 16, 2, 5).astype(np.float16),
        "feature_14": np.random.randn(80, 16, 2, 5).astype(np.float16),
        "feature_15": np.random.randn(80, 16, 1, 5).astype(np.float16),
        "feature_16": np.random.randn(80, 16, 1, 5).astype(np.float16),
        "feature_17": np.random.randn(10, 17).astype(np.float32),
        "feature_18": np.random.randn(10, 8).astype(np.float32),
        "feature_19": np.random.randn(10, 2).astype(np.float32),
        "feature_20": np.random.randn(10, 2).astype(np.float32),
        "feature_21": np.random.randn(10, 9).astype(np.float32),
        "feature_22": np.random.randn(10, 8).astype(np.float32),
        "feature_23": np.random.randn(10, 8).astype(np.float32),
        "feature_24": np.random.randn(10, 4).astype(np.float32),
        "feature_25": np.random.randn(10, 1).astype(np.float32),
        "feature_26": np.random.randn(10, 5).astype(np.float32),
        "feature_27": np.random.randn(10, 40).astype(np.float32),
        "feature_28": np.random.randn(10, 30).astype(np.float32),
        "feature_29": np.random.randn(10, 10).astype(np.float32),
        "feature_30": np.random.randn(10, 8).astype(np.float32),
        "feature_31": np.random.randn(10, 8).astype(np.float32),
        "feature_32": np.random.randn(10, 8).astype(np.float32),
        "feature_33": np.random.randn(10, 2).astype(np.float32),
        "feature_34": np.random.randn(10, 6).astype(np.float32),
        "feature_35": np.random.randn(10, 2).astype(np.float32),
        "feature_36": np.random.randn(10).astype(np.float32),
        "feature_37": np.random.randn(10, 16, 6, 14).astype(np.float16),
        "feature_38": np.random.randn(10, 16, 6, 14).astype(np.float16),
        "feature_39": np.random.randn(10, 32, 24).astype(np.float32),
        "feature_40": np.random.randint(0, 10, (10,), dtype=np.int32),
        "feature_41": np.random.randn(10, 16, 3, 14).astype(np.float16),
        "feature_42": np.random.randn(10, 16, 3, 14).astype(np.float16),
    }


def build_step_example() -> dict:
    """rl_sample() with each leaf stacked ROLLOUT_LEN times along axis 0."""
    import optree

    single = _rl_sample()
    leaves, treedef = optree.tree_flatten(single)
    stacked_leaves = [np.stack([leaf] * ROLLOUT_LEN, axis=0) for leaf in leaves]
    return optree.tree_unflatten(treedef, stacked_leaves)


def build_server_example() -> dict:
    """Step pytree + scalar `_send_ts`."""
    step_example = build_step_example()
    return {**step_example, "_send_ts": np.zeros((), dtype=np.float64)}


def compute_metrics(
    measured_batches: list[tuple[float, np.ndarray]],
    batch_size: int,
    num_batches: int,
) -> dict:
    """Latency + throughput from a list of (arrival_ts, send_ts_array) pairs."""
    if len(measured_batches) < 2:
        raise ValueError("compute_metrics needs at least 2 batches")

    all_latencies_us = []
    for arrival_ts, send_ts_array in measured_batches:
        latencies = (arrival_ts - send_ts_array) * 1_000_000
        all_latencies_us.append(latencies)
    all_latencies_us = np.concatenate(all_latencies_us)

    mean_latency_us = float(np.mean(all_latencies_us))
    stddev_latency_us = float(np.std(all_latencies_us))

    arrival_times = [ts for ts, _ in measured_batches]
    total_time_s = arrival_times[-1] - arrival_times[0]
    throughput = num_batches * batch_size / total_time_s

    batch_throughputs = []
    for i in range(1, len(arrival_times)):
        delta_s = arrival_times[i] - arrival_times[i - 1]
        batch_throughputs.append(batch_size / delta_s)
    stddev_throughput = float(np.std(batch_throughputs))

    return {
        "mean_latency_us": mean_latency_us,
        "stddev_latency_us": stddev_latency_us,
        "throughput": throughput,
        "stddev_throughput": stddev_throughput,
    }


def save_distributed_results(
    out_dir,
    commit: str,
    branch: str,
    latency_benchmarks: list[dict],
    throughput_benchmarks: list[dict],
) -> None:
    """Save latency results merged into {commit}.json and throughput to {commit}-throughput.json."""
    import json
    from datetime import datetime, timezone
    from pathlib import Path as _Path

    out_dir = _Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    latency_path = out_dir / f"{commit}.json"
    if latency_path.exists():
        with open(latency_path) as f:
            existing = json.load(f)
        existing["benchmarks"] = [b for b in existing["benchmarks"] if b.get("source") != "distributed"]
        existing["benchmarks"].extend(latency_benchmarks)
        latency_path.write_text(json.dumps(existing, indent=2))
    else:
        data = {
            "commit": commit,
            "branch": branch,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "benchmarks": latency_benchmarks,
        }
        latency_path.write_text(json.dumps(data, indent=2))

    throughput_path = out_dir / f"{commit}-throughput.json"
    data = {
        "commit": commit,
        "branch": branch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmarks": throughput_benchmarks,
    }
    throughput_path.write_text(json.dumps(data, indent=2))
    print(f"Saved latency results to {latency_path}")
    print(f"Saved throughput results to {throughput_path}")


def _make_learner_class():
    import ray

    @ray.remote(resources={"learner": 1})
    class Learner:
        def __init__(
            self,
            port: int,
            num_batches: int = NUM_BATCHES,
            warmup_batches: int = WARMUP_BATCHES,
            batch_size: int = BATCH_SIZE,
        ) -> None:
            import time

            from echo import Server, TcpTransport

            server_example = build_server_example()
            self._server = Server(server_example, batch_size, transport=TcpTransport(port=port))
            self._server.start()
            time.sleep(0.3)  # wait for socket to bind
            self._port = port
            self._num_batches = num_batches
            self._warmup_batches = warmup_batches

        def get_address(self) -> tuple[str, int]:
            return ray.util.get_node_ip_address(), self._port

        def run(self) -> list[tuple[float, np.ndarray]]:
            import time

            batches = []
            total = self._warmup_batches + self._num_batches
            for _ in range(total):
                sample = self._server.sample()
                batch_arrival_ts = time.time()
                batches.append((batch_arrival_ts, sample.batch["_send_ts"].copy()))
            self._server.close()
            return batches

    return Learner


def _make_actor_class():
    import ray

    @ray.remote(resources={"actor": 1})
    class Actor:
        def __init__(self, host: str, port: int) -> None:
            from echo import TcpClient, TrajectoryAccumulator

            server_example = build_server_example()
            self._client = TcpClient(host, port, server_example, max_inflight_msgs=32)
            step_example = build_step_example()
            self._buffer = TrajectoryAccumulator({"step": step_example})

        def run(self, num_rollouts: int) -> None:
            import time

            try:
                for _ in range(num_rollouts):
                    for _ in range(ROLLOUT_LEN):
                        self._buffer.add("step", _rl_sample())
                    data = {**self._buffer.build()["step"]}
                    data["_send_ts"] = np.array(time.time(), dtype=np.float64)
                    self._client.send(data)
                self._client.wait()
            except (BrokenPipeError, OSError):
                pass  # server shut down after collecting enough batches
            self._client.close()

    return Actor


def main() -> None:
    import subprocess
    import tempfile

    import ray

    project_root = os.getcwd()

    # Ray packages the working dir into each task; switch to /tmp to keep
    # rsync small.
    os.chdir(tempfile.mkdtemp())

    if not IS_LOCAL_TEST:
        ray.init()
        actor_counts = ACTOR_COUNTS
        batch_size = BATCH_SIZE
        num_batches = NUM_BATCHES
    else:
        ray.init(resources={"learner": 10_000, "actor": 10_000})
        actor_counts = LOCAL_ACTOR_COUNTS
        batch_size = LOCAL_BATCH_SIZE
        num_batches = LOCAL_NUM_BATCHES

    Learner = _make_learner_class()
    Actor = _make_actor_class()

    latency_benchmarks = []
    throughput_benchmarks = []
    summary_rows = []

    for run_index, num_actors in enumerate(actor_counts):
        port = BASE_PORT + run_index
        print(f"\n[{run_index + 1}/{len(actor_counts)}] actors={num_actors} port={port}")

        learner = Learner.remote(
            port=port,
            num_batches=num_batches,
            warmup_batches=WARMUP_BATCHES,
            batch_size=batch_size,
        )
        host, actual_port = ray.get(learner.get_address.remote())

        actors = [Actor.remote(host=host, port=actual_port) for _ in range(num_actors)]

        # Slight overproduction so the learner's in-flight buffer never stalls
        # waiting on the actors to refill.
        in_flight = batch_size * 2
        total_items = (num_batches + WARMUP_BATCHES) * batch_size + in_flight
        num_rollouts = math.ceil(total_items / num_actors)
        learner_future = learner.run.remote()
        _actor_futures = [actor.run.remote(num_rollouts) for actor in actors]

        ready, _ = ray.wait([learner_future], timeout=SWEEP_TIMEOUT_S)
        if not ready:
            for actor in actors:
                ray.kill(actor)
            ray.kill(learner)
            raise TimeoutError(f"Benchmark sweep timed out after {SWEEP_TIMEOUT_S}s")
        all_batches = ray.get(learner_future)

        measured = all_batches[WARMUP_BATCHES:]
        metrics = compute_metrics(measured, batch_size, num_batches)

        latency_benchmarks.append(
            {
                "name": f"dist_tcp_latency_{num_actors}_actors",
                "source": "distributed",
                "unit": "us",
                "mean": metrics["mean_latency_us"],
                "stddev": metrics["stddev_latency_us"],
            }
        )

        throughput_benchmarks.append(
            {
                "name": f"dist_tcp_throughput_{num_actors}_actors",
                "source": "distributed-throughput",
                "unit": "samples/s",
                "mean": metrics["throughput"],
                "stddev": metrics["stddev_throughput"],
            }
        )

        summary_rows.append((num_actors, metrics))

        for actor in actors:
            ray.kill(actor)
        ray.kill(learner)

    ray.shutdown()

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_root, text=True).strip()
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root, text=True).strip()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    save_distributed_results(out_dir, commit, branch, latency_benchmarks, throughput_benchmarks)

    print("\n" + "=" * 80)
    print(
        f"{'Actors':>8} {'Latency (us)':>15} {'Stddev (us)':>13} {'Throughput (s/s)':>18} {'Stddev':>10}"
    )
    print("-" * 80)
    for num_actors, m in summary_rows:
        print(
            f"{num_actors:>8} "
            f"{m['mean_latency_us']:>15.1f} {m['stddev_latency_us']:>13.1f} "
            f"{m['throughput']:>18.1f} {m['stddev_throughput']:>10.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
