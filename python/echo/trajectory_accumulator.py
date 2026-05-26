from typing import Any

import numpy as np
import optree


class TrajectoryAccumulator:
    """Multi-timescale accumulator: fixed-size pytree buffer.

    Per timescale, the example pytree determines how many ``add()`` calls
    fit before the buffer is full:

    * **Buffered timescale** — every leaf shares the same leading dim ``N``
      (``N > 1``); the accumulator stores ``N`` per-add items into
      ``stored[s:s+1] = incoming`` slot-by-slot. ``N`` becomes the
      timescale's capacity.

    * **Single-item timescale** — the timescale holds one trailing piece of
      context (e.g. a bootstrap step, an episode return) rather than a
      buffer. Detected when at least one leaf is 0-d, or all leaves
      have ``shape[0] == 1``. Capacity is ``1``; ``add()`` replaces the
      whole leaf, so non-0-d leaves may have any per-item shape (apart
      from the optional leading 1).

    Args:
        example: Dict with timescale names as top-level keys. Each value is
            a pytree whose leaves declare the per-timescale layout per the
            rule above.
    """

    def __init__(self, example: dict[str, Any]):
        if not isinstance(example, dict):
            raise TypeError("example must be a dict with timescale names as top-level keys")

        self._counts: dict[str, int] = {}
        self._single_item: dict[str, bool] = {}
        for name, subtree in example.items():
            leaves = optree.tree_leaves(subtree)
            if not leaves:
                raise ValueError(f"Timescale '{name}' has no array leaves")

            # Single-item: any 0-d leaf OR every leaf with leading dim 1.
            if any(leaf.ndim == 0 for leaf in leaves) or all(leaf.shape[0] == 1 for leaf in leaves):
                self._counts[name] = 1
                self._single_item[name] = True
            else:
                leading = [leaf.shape[0] for leaf in leaves]
                if not all(s == leading[0] for s in leading):
                    raise ValueError(
                        f"All leaves in buffered timescale '{name}' must share the same "
                        f"leading dimension (got {leading}); make any leaf 0-d or "
                        f"all shape (1, ...) to mark the timescale single-item instead"
                    )
                self._counts[name] = leading[0]
                self._single_item[name] = False

        self._tree: dict[str, Any] = {n: optree.tree_map(np.zeros_like, sub) for n, sub in example.items()}
        self._slot: dict[str, int] = {name: 0 for name in example}

    def add(self, name: str, data: Any) -> None:
        """Write a single-item pytree into the next slot for timescale *name*."""
        if name not in self._counts:
            raise KeyError(f"Unknown timescale '{name}'. Known: {list(self._counts)}")
        s = self._slot[name]
        if s >= self._counts[name]:
            raise IndexError(f"Timescale '{name}' has {self._counts[name]} slots, but you tried to add at index {s}")

        # Single-item: replace the whole leaf
        # Buffered: write into the next slot of the leading dim.
        key = Ellipsis if self._single_item[name] else slice(s, s + 1)

        def _write_slot(stored, incoming):
            stored[key] = incoming
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
