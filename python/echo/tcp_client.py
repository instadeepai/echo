import logging
import socket
import struct
import threading

import numpy as np
import optree

_log = logging.getLogger(__name__)


class ConnectionClosedError(ConnectionError):
    """Raised on send() after a TcpClient has been closed or its peer has disconnected."""


class TcpClient:
    """TCP client that sends pytree samples to an echo Server.

    Args:
        host: Server hostname or IP
        port: Server TCP port
        data_sample: Example pytree (used to validate against the server spec)
        max_inflight_msgs: Max unacknowledged messages before send() blocks
    """

    def __init__(
        self,
        host: str,
        port: int,
        data_sample: optree.PyTree[np.ndarray],
        max_inflight_msgs: int = 32,
    ):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._peer = f"{host}:{port}"

        # Anything below this connect() may leave self._sock open; the outer
        # try/except guarantees we close it if construction fails partway.
        try:
            self._sock.connect((host, port))

            magic = self._recv_exact(4)
            if magic != b"ECHO":
                raise ValueError(
                    f"Unexpected handshake magic {magic!r} from {self._peer}; "
                    "is the peer an echo server?"
                )
            version = self._recv_exact(1)[0]
            if version != 1:
                raise ValueError(
                    f"Unsupported echo wire version {version} from {self._peer} "
                    "(this client supports 1)"
                )

            num_arrays = struct.unpack("<I", self._recv_exact(4))[0]
            shapes = []
            dtype_sizes = []
            for _ in range(num_arrays):
                dtype_size = struct.unpack("<I", self._recv_exact(4))[0]
                num_dims = struct.unpack("<I", self._recv_exact(4))[0]
                shape = list(struct.unpack(f"<{num_dims}I", self._recv_exact(num_dims * 4)))
                shapes.append(shape)
                dtype_sizes.append(dtype_size)

            leaves, _ = optree.tree_flatten(data_sample)
            if not all(isinstance(leaf, np.ndarray) for leaf in leaves):
                raise TypeError("All leaves of data_sample must be numpy arrays")
            expected_shapes = [list(leaf.shape) for leaf in leaves]
            expected_dtype_sizes = [leaf.dtype.itemsize for leaf in leaves]

            if shapes != expected_shapes or dtype_sizes != expected_dtype_sizes:
                raise ValueError(
                    f"Server spec mismatch: "
                    f"shapes={shapes} vs expected={expected_shapes}, "
                    f"dtype_sizes={dtype_sizes} vs expected={expected_dtype_sizes}"
                )

            self._payload_size = sum(
                int(np.prod(s)) * d for s, d in zip(shapes, dtype_sizes)
            )

            self._inflight_msgs = threading.BoundedSemaphore(max_inflight_msgs)
            self._in_flight_count = 0
            self._in_flight_zero = threading.Condition(threading.Lock())
            self._closed = threading.Event()
            self._ack_thread = threading.Thread(target=self._consume_acks, daemon=True)
            self._ack_thread.start()
        except BaseException:
            self._sock.close()
            raise

    def _recv_exact(self, n: int) -> bytes:
        parts = []
        remaining = n
        while remaining > 0:
            chunk = self._sock.recv(remaining)
            if not chunk:
                raise ConnectionError("Connection closed during recv")
            parts.append(chunk)
            remaining -= len(chunk)
        return b"".join(parts)

    def _consume_acks(self):
        try:
            while not self._closed.is_set():
                ack = self._sock.recv(1)
                if not ack:
                    self._closed.set()
                    break
                if ack == b"\x01":
                    self._inflight_msgs.release()
                    with self._in_flight_zero:
                        self._in_flight_count -= 1
                        if self._in_flight_count == 0:
                            self._in_flight_zero.notify_all()
        except OSError as e:
            if not self._closed.is_set():
                _log.warning("ack thread %s exiting: %s", self._peer, e)
            self._closed.set()
        finally:
            with self._in_flight_zero:
                self._in_flight_zero.notify_all()

    def send(self, data: optree.PyTree[np.ndarray]) -> None:
        """Send a pytree sample to the server.

        Raises:
            ConnectionClosedError: client closed or peer disconnected.
            ValueError: payload-size mismatch.
        """
        if self._closed.is_set():
            raise ConnectionClosedError(f"TcpClient for {self._peer} is closed")

        leaves, _ = optree.tree_flatten(data)
        payload = b"".join(leaf.tobytes() for leaf in leaves)

        if len(payload) != self._payload_size:
            raise ValueError(
                f"Payload size {len(payload)} != expected {self._payload_size}"
            )

        with self._in_flight_zero:
            self._in_flight_count += 1

        self._inflight_msgs.acquire()

        try:
            self._sock.sendall(payload)
        except OSError as e:
            self._closed.set()
            self._inflight_msgs.release()
            with self._in_flight_zero:
                self._in_flight_count -= 1
                if self._in_flight_count == 0:
                    self._in_flight_zero.notify_all()
            raise ConnectionClosedError(
                f"TcpClient for {self._peer} connection lost"
            ) from e

    def wait(self) -> None:
        """Block until all in-flight messages have been acknowledged."""
        with self._in_flight_zero:
            self._in_flight_zero.wait_for(lambda: self._in_flight_count == 0)

    def close(self) -> None:
        """Close the client. Idempotent."""
        if self._closed.is_set():
            return
        self._closed.set()
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self._sock.close()
        # Ack thread is daemon, but we still join briefly so callers can rely
        # on close() leaving no live threads behind in tests.
        self._ack_thread.join(timeout=1.0)
