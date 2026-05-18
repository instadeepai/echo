"""Type stubs for the Rust extension module."""

from collections.abc import Sequence
from typing import Any

import numpy as np


class TcpTransport:
    def __init__(self, port: int, num_threads: int = 8) -> None: ...


class HistSnapshot:
    count: int
    min_ns: int
    max_ns: int
    mean_ns: float
    p50_ns: int
    p90_ns: int
    p99_ns: int


class SampleInfo:
    store_size: int
    active_connections: int
    queue_depth_sum: int
    queue_depth_max: int
    push_blocked_count: int
    samples_inserted_total: int
    notify_count: int
    cas_success_total: int
    cas_failure_total: int
    reservation_retries_total: int
    drain_round: HistSnapshot
    memcpy: HistSnapshot
    queue_dwell: HistSnapshot


class _Server:
    def __init__(
        self,
        shapes: Sequence[Sequence[int]],
        dtype_sizes: Sequence[int],
        batch_size: int,
        transport: TcpTransport | None = None,
        num_buffers: int = 3,
        num_drainers: int = 8,
        producer_queue_size: int = 8,
    ) -> None: ...
    def start(self) -> None: ...
    def sample(self) -> tuple[list[np.ndarray[Any, np.dtype[np.uint8]]], SampleInfo] | None: ...
    def submit(self, data: Sequence[bytes]) -> None: ...
    def reset_histograms(self) -> None: ...
    def shutdown(self) -> None: ...
