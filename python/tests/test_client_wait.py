import numpy as np
import pytest

from echo import Server, TcpClient, TcpTransport

from conftest import free_port, wait_for_listen

_SAMPLE = {"x": np.zeros((4,), dtype=np.float32)}


@pytest.fixture
def server():
    port = free_port()
    s = Server(_SAMPLE, batch_size=10_000, transport=TcpTransport(port=port), num_buffers=3)
    s.start()
    wait_for_listen(port)
    yield s, port
    s.close()


@pytest.fixture
def client(server):
    _, port = server
    c = TcpClient("localhost", port, _SAMPLE, max_inflight_msgs=16)
    yield c
    c.close()


class TestWait:
    def test_wait_returns_when_all_acked(self, client):
        for _ in range(5):
            client.send(_SAMPLE)
        client.wait()
        assert client._in_flight_count == 0

    def test_wait_on_empty_is_immediate(self, client):
        client.wait()
        assert client._in_flight_count == 0
