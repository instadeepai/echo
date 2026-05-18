"""End-to-end tests for Sample/SampleInfo returned by Server.sample()."""

import threading
import time

import numpy as np
import pytest

from echo import Server, TcpClient, TcpTransport
from echo.echo import SampleInfo
from echo.server import Sample

from conftest import free_port, wait_for_listen


def _make_server(batch_size: int = 2, num_buffers: int = 3) -> Server:
    example = {"x": np.zeros((4,), dtype=np.float32)}
    return Server(example=example, batch_size=batch_size, num_buffers=num_buffers)


def test_sample_returns_sample_object():
    server = _make_server()
    try:
        for i in range(2):
            server.submit({"x": np.full((4,), i, dtype=np.float32)})
        out = server.sample()
        assert isinstance(out, Sample)
        assert isinstance(out.info, SampleInfo)
        assert out.batch["x"].shape == (2, 4)
    finally:
        server.close()


def test_sample_info_fields_accessible():
    server = _make_server()
    try:
        for i in range(2):
            server.submit({"x": np.full((4,), i, dtype=np.float32)})
        out = server.sample()
        assert isinstance(out.info.store_size, int)
        assert isinstance(out.info.active_connections, int)
        assert isinstance(out.info.queue_depth_sum, int)
        assert isinstance(out.info.queue_depth_max, int)
        assert isinstance(out.info.push_blocked_count, int)
        assert out.info.active_connections == 0
        assert out.info.push_blocked_count == 0
    finally:
        server.close()


def test_active_connections_reflects_clients():
    port = free_port()
    example = {"x": np.zeros((4,), dtype=np.float32)}
    server = Server(
        example=example,
        batch_size=2,
        transport=TcpTransport(port=port),
        num_buffers=4,
    )
    server.start()
    wait_for_listen(port)

    collected: list[Sample] = []

    def drain():
        for s in server.dataset_iter():
            collected.append(s)

    t = threading.Thread(target=drain, daemon=True)
    t.start()

    c1 = c2 = None
    try:
        c1 = TcpClient(host="127.0.0.1", port=port, data_sample=example)
        c2 = TcpClient(host="127.0.0.1", port=port, data_sample=example)

        for i in range(6):
            c1.send({"x": np.full((4,), i, dtype=np.float32)})
            c2.send({"x": np.full((4,), i + 100, dtype=np.float32)})

        deadline = time.time() + 5.0
        while not collected and time.time() < deadline:
            time.sleep(0.01)
        assert collected, "no batches produced within 5s"

        sample = collected[-1]
        assert sample.info.active_connections >= 2
        assert sample.info.push_blocked_count == 0
        capacity = 2 * 4  # batch_size * num_buffers
        assert 0 <= sample.info.store_size <= capacity
    finally:
        if c1 is not None:
            c1.close()
        if c2 is not None:
            c2.close()
        server.close()
        t.join(timeout=2)


def test_push_blocked_rises_under_backpressure():
    port = free_port()
    example = {"x": np.zeros((4,), dtype=np.float32)}
    # Small ring + slow consumer → store backpressure → SPSC queue fills →
    # producer pushes are blocked.
    server = Server(
        example=example,
        batch_size=2,
        transport=TcpTransport(port=port),
        num_buffers=2,
    )
    server.start()
    wait_for_listen(port)

    client = None
    try:
        # Large client-side inflight so the *Rust* SPSC queue is the bottleneck,
        # not the client ack semaphore.
        client = TcpClient(host="127.0.0.1", port=port, data_sample=example, max_inflight_msgs=1024)

        for i in range(512):
            client.send({"x": np.full((4,), i, dtype=np.float32)})

        deadline = time.time() + 2.0
        sample = None
        while time.time() < deadline:
            sample = server.sample()
            if sample is not None and sample.info.push_blocked_count > 0:
                break
        assert sample is not None
        assert sample.info.push_blocked_count > 0

        # push_blocked_count is cumulative; a second sample must still see it.
        sample = server.sample()
        assert sample is not None
        assert sample.info.push_blocked_count > 0
    finally:
        if client is not None:
            client.close()
        server.close()


def test_reset_histograms_clears_window_in_sample_info():
    """reset_histograms() clears the histogram count window. Histograms are
    only populated by the drainer pool (transport path), so this test uses
    TCP; it skips when detailed-metrics isn't enabled."""
    port = free_port()
    example = {"x": np.zeros((4,), dtype=np.float32)}
    server = Server(
        example=example,
        batch_size=2,
        transport=TcpTransport(port=port),
        num_buffers=4,
    )
    server.start()
    wait_for_listen(port)

    collected: list[Sample] = []

    def drain():
        for s in server.dataset_iter():
            collected.append(s)

    t = threading.Thread(target=drain, daemon=True)
    t.start()

    client = None
    try:
        client = TcpClient(host="127.0.0.1", port=port, data_sample=example)

        N1 = 60
        for i in range(N1):
            client.send({"x": np.full((4,), i, dtype=np.float32)})

        deadline = time.time() + 5.0
        while len(collected) < (N1 // 4) and time.time() < deadline:
            time.sleep(0.01)
        assert collected

        s1 = collected[-1]
        if s1.info.memcpy.count == 0:
            pytest.skip("detailed-metrics feature not enabled in this build")
        pre_count = s1.info.memcpy.count
        assert pre_count >= 5

        server.reset_histograms()
        before_phase2_len = len(collected)
        N2 = 4
        for i in range(N2):
            client.send({"x": np.full((4,), 100 + i, dtype=np.float32)})

        deadline = time.time() + 5.0
        while len(collected) <= before_phase2_len and time.time() < deadline:
            time.sleep(0.01)
        assert len(collected) > before_phase2_len

        s2 = collected[-1]
        assert s2.info.memcpy.count < pre_count
        assert s2.info.memcpy.count <= N2 + 4
    finally:
        if client is not None:
            client.close()
        server.close()
