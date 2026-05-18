"""Shared fixtures for echo's Python tests."""
import socket
import time
from collections.abc import Iterator

import numpy as np
import pytest

from echo import Server, TcpClient, TcpTransport


def free_port() -> int:
    """Return an OS-assigned free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for_listen(port: int, host: str = "127.0.0.1", timeout: float = 2.0) -> None:
    """Block until `port` is accepting connections, or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.1):
                return
        except OSError:
            time.sleep(0.02)
    raise TimeoutError(f"port {port} not listening after {timeout}s")


@pytest.fixture
def make_server():
    """Factory: build, start, and clean up a Server bound to a free port.

    Yields ``(server, port)``. The server is started and the port confirmed
    listening before returning, and closed on teardown.
    """
    created: list[Server] = []

    def _make(example, batch_size: int, *, transport: str = "tcp", **kwargs) -> tuple[Server, int]:
        if transport != "tcp":
            raise ValueError(f"unsupported transport {transport!r}; only 'tcp' is shipped")
        port = free_port()
        tx = TcpTransport(port=port)
        server = Server(example, batch_size, transport=tx, **kwargs)
        server.start()
        wait_for_listen(port)
        created.append(server)
        return server, port

    yield _make

    for server in created:
        server.close()


@pytest.fixture
def make_client():
    """Factory for TcpClient that closes them at teardown."""
    created = []

    def _make(transport: str, host: str, port: int, example, **kwargs):
        if transport != "tcp":
            raise ValueError(f"unsupported transport {transport!r}; only 'tcp' is shipped")
        client = TcpClient(host, port, example, **kwargs)
        created.append(client)
        return client

    yield _make

    for client in created:
        client.close()


@pytest.fixture(params=["tcp"])
def transport_name(request) -> Iterator[str]:
    """Parametrize a test over the shipped transports."""
    yield request.param


# Convenience constants for tests that don't care about specific dtypes.
F32_OBS_EXAMPLE = {"obs": np.zeros((4,), dtype=np.float32)}
