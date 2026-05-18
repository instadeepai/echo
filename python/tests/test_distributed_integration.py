"""End-to-end smoke test of the distributed benchmark (Ray pipeline)."""

import math
import pytest

ray = pytest.importorskip("ray", reason="ray not installed")

from benches.bench_distributed import (  # noqa: E402
    BASE_PORT,
    compute_metrics,
    _make_learner_class,
    _make_actor_class,
)

SMOKE_BATCH_SIZE = 64
SMOKE_NUM_BATCHES = 3
SMOKE_WARMUP_BATCHES = 1
SMOKE_PORT = BASE_PORT + 9000

PIPELINE_TIMEOUT_S = 60


@pytest.fixture(scope="module")
def ray_local():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(resources={"learner": 1, "actor": 10}, num_cpus=4)
    yield
    ray.shutdown()


def test_smoke_single_actor(ray_local):
    Learner = _make_learner_class()
    Actor = _make_actor_class()

    port = SMOKE_PORT

    learner = Learner.remote(
        port=port,
        num_batches=SMOKE_NUM_BATCHES,
        warmup_batches=SMOKE_WARMUP_BATCHES,
        batch_size=SMOKE_BATCH_SIZE,
    )
    host, actual_port = ray.get(learner.get_address.remote(), timeout=PIPELINE_TIMEOUT_S)

    actor = Actor.remote(host=host, port=actual_port)
    num_rollouts = math.ceil((SMOKE_NUM_BATCHES + SMOKE_WARMUP_BATCHES) * SMOKE_BATCH_SIZE / 1)

    learner_future = learner.run.remote()
    actor_future = actor.run.remote(num_rollouts)

    try:
        all_batches = ray.get(learner_future, timeout=PIPELINE_TIMEOUT_S)
        ray.get(actor_future, timeout=PIPELINE_TIMEOUT_S)
    finally:
        ray.kill(actor)
        ray.kill(learner)

    assert len(all_batches) == SMOKE_WARMUP_BATCHES + SMOKE_NUM_BATCHES

    for arrival_ts, send_ts_array in all_batches:
        assert isinstance(arrival_ts, float)
        assert send_ts_array.shape == (SMOKE_BATCH_SIZE,)
        assert send_ts_array.dtype.kind == "f"

    measured = all_batches[SMOKE_WARMUP_BATCHES:]
    metrics = compute_metrics(measured, SMOKE_BATCH_SIZE, SMOKE_NUM_BATCHES)
    assert metrics["mean_latency_us"] > 0
    assert metrics["throughput"] > 0
