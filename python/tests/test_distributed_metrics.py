import numpy as np
import pytest
from benches.bench_distributed import ROLLOUT_LEN, build_step_example, compute_metrics


def _make_batches(num_batches, batch_size, latency_s=0.01, interval_s=0.5):
    """Synthetic batch data: uniform latency, uniform inter-batch interval."""
    batches = []
    base_time = 1000.0
    for i in range(num_batches):
        arrival_ts = base_time + i * interval_s
        # All samples sent latency_s before arrival
        send_ts_array = np.full(batch_size, arrival_ts - latency_s, dtype=np.float64)
        batches.append((arrival_ts, send_ts_array))
    return batches


def test_compute_metrics_latency_mean():
    batches = _make_batches(10, batch_size=4, latency_s=0.010)
    m = compute_metrics(batches, batch_size=4, num_batches=10)
    assert abs(m["mean_latency_us"] - 10_000.0) < 1.0  # 10ms = 10000 us


def test_compute_metrics_latency_stddev_zero():
    """Uniform latency => stddev ~0."""
    batches = _make_batches(10, batch_size=4, latency_s=0.010)
    m = compute_metrics(batches, batch_size=4, num_batches=10)
    assert m["stddev_latency_us"] < 1.0


def test_compute_metrics_throughput():
    """10 batches of 4 samples, 0.5s apart => throughput = 10*4 / (9*0.5) ~ 8.89 samples/s."""
    batches = _make_batches(10, batch_size=4, latency_s=0.010, interval_s=0.5)
    m = compute_metrics(batches, batch_size=4, num_batches=10)
    expected = 10 * 4 / (9 * 0.5)
    assert abs(m["throughput"] - expected) < 0.01


def test_compute_metrics_throughput_stddev():
    """Uniform intervals => per-batch throughput is constant => stddev ~0."""
    batches = _make_batches(10, batch_size=4, latency_s=0.010, interval_s=0.5)
    m = compute_metrics(batches, batch_size=4, num_batches=10)
    assert m["stddev_throughput"] < 0.01


def test_compute_metrics_requires_at_least_two_batches():
    """Need >= 2 batches to compute inter-batch intervals."""
    batches = _make_batches(1, batch_size=4)
    with pytest.raises(ValueError, match="at least 2"):
        compute_metrics(batches, batch_size=4, num_batches=1)


def test_build_step_example_shape():
    step_example = build_step_example()
    import optree
    leaves = optree.tree_leaves(step_example)
    for leaf in leaves:
        assert leaf.shape[0] == ROLLOUT_LEN, (
            f"Leading dim should be ROLLOUT_LEN={ROLLOUT_LEN}, got {leaf.shape}"
        )


def test_build_server_example_has_send_ts():
    from benches.bench_distributed import build_server_example
    server_example = build_server_example()
    assert "_send_ts" in server_example
    assert server_example["_send_ts"].dtype == np.float64
    assert server_example["_send_ts"].shape == ()
