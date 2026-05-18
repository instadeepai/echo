from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import optree

from echo.echo import SampleInfo, TcpTransport, _Server


class Sample(NamedTuple):
    """A batch paired with a snapshot of backpressure metrics."""

    batch: Any
    info: SampleInfo


class Server:
    """
    Server that handles pytree flattening/unflattening on top of the Rust core.

    Args:
        example: Example pytree with numpy arrays (used to infer structure)
        batch_size: Number of samples per emitted batch
        transport: Optional TcpTransport. Omit for in-process use; call
            ``submit()`` to feed samples directly. Custom transports
            (gRPC, RDMA, …) can be wired by implementing the Rust
            ``Transport`` trait.
        num_buffers: Number of ring buffer batches (min 2, default 3)
        num_drainers: Number of threads draining from producer queues
        producer_queue_size: Per-connection queue size

    Lifetime: the numpy arrays returned by ``sample()`` are views into
    Rust-owned ring-buffer memory. They are invalidated as soon as the next
    batch reuses those slots, and the memory is freed when this Server is
    garbage-collected. Don't keep references to a batch beyond the next
    sample() call without ``np.copy``, and don't drop the Server while you
    still hold sample arrays.
    """

    def __init__(
        self,
        example: Any,
        batch_size: int,
        transport: TcpTransport | None = None,
        num_buffers: int = 3,
        num_drainers: int = 8,
        producer_queue_size: int = 8,
    ):
        leaves, self._treedef = optree.tree_flatten(example)

        if not all(isinstance(leaf, np.ndarray) for leaf in leaves):
            raise TypeError("All leaves must be numpy arrays")

        self._shapes = [leaf.shape for leaf in leaves]
        dtype_sizes = [leaf.dtype.itemsize for leaf in leaves]
        self._dtypes = [leaf.dtype for leaf in leaves]
        self._batch_size = batch_size

        self._server = _Server(
            shapes=list(self._shapes),
            dtype_sizes=list(dtype_sizes),
            batch_size=batch_size,
            transport=transport,
            num_buffers=num_buffers,
            num_drainers=num_drainers,
            producer_queue_size=producer_queue_size,
        )

    def start(self) -> None:
        """Start the transport (bind port, accept connections)."""
        self._server.start()

    def sample(self) -> Sample | None:
        """
        Block until a batch is ready. Returns None on shutdown.

        Each leaf of ``Sample.batch`` is a zero-copy view into Rust-owned
        memory and is invalidated on the next ``sample()`` call. Use
        ``np.copy`` (or ``dataset_iter(copy=True)``) if you need to retain a
        batch beyond the next call.
        """
        result = self._server.sample()
        if result is None:
            return None
        flat_arrays, info = result

        reshaped = [
            np.frombuffer(b, dtype=d).reshape((self._batch_size,) + tuple(s))
            for b, d, s in zip(flat_arrays, self._dtypes, self._shapes)
        ]

        batch = optree.tree_unflatten(self._treedef, reshaped)
        return Sample(batch=batch, info=info)

    def dataset_iter(self, *, copy: bool = False) -> Generator[Sample, None, None]:
        """Yield Samples as batches become ready; returns on shutdown.

        Pass ``copy=True`` to deep-copy each batch and free the underlying
        ring slots for reuse on the next iteration.
        """
        while True:
            sample = self.sample()
            if sample is None:
                return
            if copy:
                sample = Sample(
                    batch=optree.tree_map(np.copy, sample.batch),
                    info=sample.info,
                )
            yield sample

    def submit(self, data: Any) -> None:
        """Submit a single sample directly to the store (in-process / tests)."""
        leaves, _ = optree.tree_flatten(data)
        self._server.submit([leaf.tobytes() for leaf in leaves])

    def close(self) -> None:
        """Signal shutdown, causing sample() to return None."""
        self._server.shutdown()

    def reset_histograms(self) -> None:
        """Reset the histogram fields on subsequent SampleInfo snapshots.
        Counters and gauges are unchanged.
        """
        self._server.reset_histograms()
