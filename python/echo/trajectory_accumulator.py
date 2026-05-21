from typing import Any

import numpy as np
import optree


class TrajectoryAccumulator:
    """Multi-timescale accumulator: fixed-size pytree buffer.

    Args:
        example: Dict with timescale names as top-level keys. The leading
            dimension of each leaf array is the number of ``add()`` calls
            expected before the buffer is ready to send.
    """

    def __init__(self, example: dict[str, Any]):
        if not isinstance(example, dict):
            raise TypeError("example must be a dict with timescale names as top-level keys")

        self._counts: dict[str, int] = {}
        for name, subtree in example.items():
            leaves = optree.tree_leaves(subtree)
            if not leaves:
                raise ValueError(f"Timescale '{name}' has no array leaves")
            leading = leaves[0].shape[0] if leaves[0].ndim > 0 else 1
            if not all((leaf.shape[0] if leaf.ndim > 0 else 1) == leading for leaf in leaves):
                raise ValueError(
                    f"All leaves in timescale '{name}' must share the same leading dimension"
                )
            self._counts[name] = leading

        self._tree: dict[str, Any] = {
            n: optree.tree_map(np.zeros_like, sub) for n, sub in example.items()
        }
        self._slot: dict[str, int] = {name: 0 for name in example}

    def add(self, name: str, data: Any) -> None:
        """Write a single-item pytree into the next slot for timescale *name*."""
        if name not in self._counts:
            raise KeyError(f"Unknown timescale '{name}'. Known: {list(self._counts)}")
        s = self._slot[name]
        if s >= self._counts[name]:
            raise IndexError(
                f"Timescale '{name}' is already full ({self._counts[name]} slots). "
                "Call reset() or build() before adding more."
            )

        def _write_slot(stored, incoming):
            np.atleast_1d(stored)[s:s + 1] = incoming
            return stored

        optree.tree_map_(_write_slot, self._tree[name], data)
        self._slot[name] += 1

    def build(self) -> dict[str, Any]:
        """Return the filled pytree and reset slot counters.

        The returned tree aliases internal buffers; callers must finish using
        it (e.g. complete the synchronous send) before the next ``add()``.
        """
        self._slot = {name: 0 for name in self._slot}
        return self._tree

    def reset(self) -> None:
        """Reset slot counters without sending (e.g. on episode abort)."""
        self._slot = {name: 0 for name in self._slot}
