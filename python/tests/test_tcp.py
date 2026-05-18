"""TCP-specific tests. Generic round-trip coverage lives in test_server.py
(parametrized over both transports). This file covers behaviour that is only
exposed via TCP: handshake validation, magic/version, and graceful close
semantics."""
import socket

import numpy as np
import pytest

from echo import ConnectionClosedError, Server, TcpClient, TcpTransport

from conftest import free_port, wait_for_listen


class TestTcpSpecValidation:
    def test_spec_mismatch_raises(self, make_server):
        server_example = {"obs": np.zeros((4,), dtype=np.float32)}
        client_example = {"obs": np.zeros((8,), dtype=np.float32)}
        server, port = make_server(server_example, batch_size=10, transport="tcp")

        with pytest.raises(ValueError, match="Server spec mismatch"):
            TcpClient("localhost", port, client_example)

    def test_wrong_payload_size_raises(self, make_server, make_client):
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server, port = make_server(example, batch_size=10, transport="tcp")
        client = make_client("tcp", "localhost", port, example)

        with pytest.raises(ValueError, match="Payload size"):
            client.send({"obs": np.array([1, 2], dtype=np.float32)})


class TestHandshake:
    """The wire format starts with magic 'ECHO' + 1-byte version, so a client
    pointed at an unrelated server (or a mismatched echo version) fails
    fast with a clear error."""

    def test_unknown_peer_rejects_with_magic_error(self):
        # Spin up a plain socket that accepts but sends garbage.
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def serve():
            conn, _ = listener.accept()
            conn.sendall(b"HTTP")  # 4 wrong magic bytes
            conn.close()

        import threading
        t = threading.Thread(target=serve, daemon=True)
        t.start()

        example = {"obs": np.zeros((4,), dtype=np.float32)}
        with pytest.raises(ValueError, match="magic"):
            TcpClient("127.0.0.1", port, example)

        t.join(timeout=1.0)
        listener.close()


class TestTcpGracefulClose:
    def test_send_after_close_raises_connection_closed(self, make_server, make_client):
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server, port = make_server(example, batch_size=10, transport="tcp")
        client = make_client("tcp", "localhost", port, example)

        client.close()
        with pytest.raises(ConnectionClosedError):
            client.send({"obs": np.array([1, 2, 3, 4], dtype=np.float32)})

    def test_send_after_server_shutdown_raises_connection_closed(self):
        port = free_port()
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server = Server(example, batch_size=10, transport=TcpTransport(port=port))
        server.start()
        wait_for_listen(port)

        client = TcpClient("localhost", port, example)
        server.close()

        # Block until the ack thread observes the server-side FIN — without
        # this, a fast send() could win the race against the close detection.
        assert client._closed.wait(timeout=2.0)

        with pytest.raises(ConnectionClosedError):
            client.send({"obs": np.array([1, 2, 3, 4], dtype=np.float32)})
        client.close()

    def test_close_is_idempotent(self, make_server, make_client):
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server, port = make_server(example, batch_size=10, transport="tcp")
        client = make_client("tcp", "localhost", port, example)

        client.close()
        client.close()  # second close must not raise


class TestTcpFlowControl:
    def test_wait_for_acks(self, make_server, make_client):
        example = {"val": np.zeros((1,), dtype=np.int32)}
        server, port = make_server(example, batch_size=5, transport="tcp")
        client = make_client("tcp", "localhost", port, example, max_inflight_msgs=32)

        for i in range(5):
            client.send({"val": np.array([i], dtype=np.int32)})

        sample = server.sample()
        assert sample is not None
        client.wait()


class TestTcpPartialReads:
    """The server reassembles payloads that the kernel splits across recv
    boundaries. Test by sending a payload one byte at a time."""

    def test_byte_by_byte_payload_is_reassembled(self, make_server):
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server, port = make_server(example, batch_size=1, transport="tcp")

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", port))
        try:
            magic = b""
            while len(magic) < 4:
                chunk = sock.recv(4 - len(magic))
                assert chunk
                magic += chunk
            assert magic == b"ECHO"
            # Consume rest of handshake by reading any pending bytes.
            sock.setblocking(False)
            try:
                while sock.recv(4096):
                    pass
            except BlockingIOError:
                pass
            sock.setblocking(True)

            payload = np.array([1, 2, 3, 4], dtype=np.float32).tobytes()
            for byte in payload:
                sock.sendall(bytes([byte]))

            # Server acks one byte, then we read it (consume to keep flow control honest).
            sock.recv(1)
        finally:
            sock.close()

        sample = server.sample()
        assert sample is not None
        np.testing.assert_array_equal(sample.batch["obs"], [[1, 2, 3, 4]])
