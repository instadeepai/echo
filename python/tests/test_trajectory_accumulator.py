import numpy as np
import pytest
from conftest import free_port, wait_for_listen
from echo import Server, TcpClient, TcpTransport
from echo.trajectory_accumulator import TrajectoryAccumulator

_EXAMPLE_SMALL = {
    "transition": {"obs": np.empty((2, 4), dtype=np.float32)},
}

_EXAMPLE_MULTI = {
    "transition": {
        "obs": np.empty((2, 4), dtype=np.float32),
        "rew": np.empty((2,), dtype=np.float32),
    },
    "summary": {
        "ret": np.empty((1,), dtype=np.float32),
    },
}

N = 4
_EXAMPLE = {
    "transition": {
        "obs": np.empty((N, 12), dtype=np.float32),
        "rew": np.empty((N,), dtype=np.float32),
    },
    "summary": {
        "ret": np.empty((1,), dtype=np.float32),
    },
}


class TestTrajectoryAccumulatorInit:
    def test_counts_inferred_from_leading_dim(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        assert buf._counts == {"transition": N, "summary": 1}

    def test_rejects_mismatched_leading_dim_in_buffered_timescale(self):
        # Buffered timescales (no 0-d leaves) must have all leaves share
        # leading dim.
        bad = {
            "timescale": {
                "a": np.empty((4, 8), dtype=np.float32),  # leading=4
                "b": np.empty((3,), dtype=np.float32),  # leading=3
            },
        }
        with pytest.raises(ValueError, match="same leading dimension"):
            TrajectoryAccumulator(bad)

    def test_zero_d_leaf_marks_timescale_single_item(self):
        # A 0-d leaf signals "single-item timescale": capacity is 1 and
        # other leaves may have any per-item shape.
        spec = {
            "timescale": {
                "obs": np.empty((4, 8), dtype=np.float32),  # per-item shape
                "flag": np.empty((), dtype=np.bool_),  # 0-d marker
            },
        }
        buf = TrajectoryAccumulator(spec)
        assert buf._counts == {"timescale": 1}

    def test_all_unit_leading_marks_timescale_single_item(self):
        # Every leaf with shape (1, ...) → single-item, no 0-d marker needed.
        spec = {
            "timescale": {
                "obs": np.empty((1, 8), dtype=np.float32),
                "head": np.empty((1, 4, 2), dtype=np.float32),
            },
        }
        buf = TrajectoryAccumulator(spec)
        assert buf._counts == {"timescale": 1}

    def test_single_unit_leading_leaf_is_single_item(self):
        spec = {"timescale": {"a": np.empty((1,), dtype=np.float32)}}
        buf = TrajectoryAccumulator(spec)
        assert buf._counts == {"timescale": 1}

    def test_mixed_unit_and_non_unit_leading_rejected(self):
        # (1, *) and (5,) — neither all-unit nor matching leading-dim. Falls
        # through to the buffered branch and fails the invariant.
        bad = {
            "timescale": {
                "a": np.empty((1, 8), dtype=np.float32),  # leading=1
                "b": np.empty((5,), dtype=np.float32),  # leading=5
            },
        }
        with pytest.raises(ValueError, match="same leading dimension"):
            TrajectoryAccumulator(bad)

    def test_unit_leading_write_replaces_whole_leaf(self):
        spec = {"timescale": {"obs": np.empty((1, 4), dtype=np.float32)}}
        buf = TrajectoryAccumulator(spec)
        buf.add("timescale", {"obs": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)})
        np.testing.assert_array_equal(buf._tree["timescale"]["obs"], [[1.0, 2.0, 3.0, 4.0]])

    def test_over_index_raises_clear_indexerror(self):
        spec = {
            "transition": {
                "obs": np.empty((N, 12), dtype=np.float32),
                "rew": np.empty((N,), dtype=np.float32),
            },
        }
        buf = TrajectoryAccumulator(spec)
        item = {
            "obs": np.zeros(12, dtype=np.float32),
            "rew": np.zeros((), dtype=np.float32),
        }
        for _ in range(N):
            buf.add("transition", item)
        with pytest.raises(
            IndexError,
            match=rf"Timescale 'transition' has {N} slots, but you tried to add at index {N}",
        ):
            buf.add("transition", item)


class TestTrajectoryAccumulatorAdd:
    def test_add_writes_correct_values(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        obs = np.arange(12, dtype=np.float32)
        buf.add("transition", {"obs": obs, "rew": np.array(7.0, dtype=np.float32)})

        tree = buf._tree
        np.testing.assert_array_equal(tree["transition"]["obs"][0], obs)
        np.testing.assert_array_equal(tree["transition"]["rew"][0], 7.0)

    def test_add_increments_slot_counter(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        buf.add("transition", {"obs": np.zeros(12, dtype=np.float32), "rew": np.zeros((), dtype=np.float32)})
        assert buf._slot["transition"] == 1

    def test_add_dtype_cast(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        obs64 = np.ones(12, dtype=np.float64) * 1.5
        buf.add("transition", {"obs": obs64, "rew": np.zeros((), dtype=np.float64)})

        tree = buf._tree
        np.testing.assert_allclose(tree["transition"]["obs"][0], 1.5)

    def test_add_multiple_slots(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        for i in range(N):
            buf.add(
                "transition",
                {
                    "obs": np.full(12, float(i), dtype=np.float32),
                    "rew": np.array(float(i * 10), dtype=np.float32),
                },
            )

        tree = buf._tree
        for i in range(N):
            np.testing.assert_array_equal(tree["transition"]["obs"][i], float(i))
            np.testing.assert_array_equal(tree["transition"]["rew"][i], float(i * 10))


class TestTrajectoryAccumulatorScalarLeaves:
    """0-d leaves should be writable without padding to (1,)."""

    def _example(self):
        return {
            "step": {
                "obs": np.empty((3, 4), dtype=np.float32),
                "reward": np.empty((3,), dtype=np.float32),
            },
            "episode": {
                "ret": np.empty((), dtype=np.float32),
                "gen": np.empty((), dtype=np.int32),
            },
        }

    def test_capacity_inferred_for_zero_d_timescale(self):
        buf = TrajectoryAccumulator(self._example())
        assert buf._counts == {"step": 3, "episode": 1}

    def test_write_zero_d_leaves(self):
        buf = TrajectoryAccumulator(self._example())
        buf.add(
            "episode",
            {
                "ret": np.array(7.5, dtype=np.float32),
                "gen": np.array(42, dtype=np.int32),
            },
        )
        tree = buf._tree
        assert tree["episode"]["ret"].shape == ()
        assert tree["episode"]["gen"].shape == ()
        np.testing.assert_array_equal(tree["episode"]["ret"], 7.5)
        np.testing.assert_array_equal(tree["episode"]["gen"], 42)

    def test_mixed_zero_d_and_nd_in_same_build(self):
        buf = TrajectoryAccumulator(self._example())
        for i in range(3):
            buf.add(
                "step",
                {
                    "obs": np.full(4, float(i), dtype=np.float32),
                    "reward": np.array(float(i * 10), dtype=np.float32),
                },
            )
        buf.add(
            "episode",
            {
                "ret": np.array(99.0, dtype=np.float32),
                "gen": np.array(5, dtype=np.int32),
            },
        )
        tree = buf.build()

        assert tree["step"]["obs"].shape == (3, 4)
        assert tree["step"]["reward"].shape == (3,)
        assert tree["episode"]["ret"].shape == ()
        assert tree["episode"]["gen"].shape == ()

        for i in range(3):
            np.testing.assert_array_equal(tree["step"]["obs"][i], float(i))
            np.testing.assert_array_equal(tree["step"]["reward"][i], float(i * 10))
        np.testing.assert_array_equal(tree["episode"]["ret"], 99.0)
        np.testing.assert_array_equal(tree["episode"]["gen"], 5)

    def test_zero_d_timescale_full_after_one_add(self):
        buf = TrajectoryAccumulator(self._example())
        buf.add(
            "episode",
            {
                "ret": np.array(1.0, dtype=np.float32),
                "gen": np.array(1, dtype=np.int32),
            },
        )
        with pytest.raises(IndexError):
            buf.add(
                "episode",
                {
                    "ret": np.array(2.0, dtype=np.float32),
                    "gen": np.array(2, dtype=np.int32),
                },
            )


class TestTrajectoryAccumulatorBuild:
    def _fill(self, buf):
        for i in range(N):
            buf.add(
                "transition",
                {
                    "obs": np.full(12, float(i), dtype=np.float32),
                    "rew": np.array(float(i), dtype=np.float32),
                },
            )
        buf.add("summary", {"ret": np.array([99.0], dtype=np.float32)})

    def test_build_returns_dict(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        self._fill(buf)
        tree = buf.build()
        assert isinstance(tree, dict)
        assert "transition" in tree
        assert "summary" in tree

    def test_build_contains_correct_data(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        self._fill(buf)
        tree = buf.build()
        np.testing.assert_array_equal(tree["transition"]["obs"][0], 0.0)
        np.testing.assert_array_equal(tree["summary"]["ret"], [99.0])

    def test_build_resets_slot_counters(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        self._fill(buf)
        buf.build()
        assert all(s == 0 for s in buf._slot.values())

    def test_reset_clears_slot_counters(self):
        buf = TrajectoryAccumulator(_EXAMPLE)
        buf.add("transition", {"obs": np.zeros(12, dtype=np.float32), "rew": np.zeros((), dtype=np.float32)})
        buf.reset()
        assert buf._slot["transition"] == 0


@pytest.fixture
def multi_server():
    port = free_port()
    s = Server(_EXAMPLE_MULTI, batch_size=1, transport=TcpTransport(port=port), num_buffers=3)
    s.start()
    wait_for_listen(port)
    yield s, port
    s.close()


class TestClientSendFromTrajectoryAccumulator:
    def test_send_pytree_from_build(self):
        port = free_port()
        server = Server(_EXAMPLE_SMALL, batch_size=1, transport=TcpTransport(port=port), num_buffers=3)
        server.start()
        wait_for_listen(port)

        try:
            client = TcpClient("localhost", port, _EXAMPLE_SMALL, max_inflight_msgs=4)
            buf = TrajectoryAccumulator(_EXAMPLE_SMALL)
            buf.add("transition", {"obs": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)})
            buf.add("transition", {"obs": np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32)})

            client.send(buf.build())
            client.wait()

            sample = server.sample()
            np.testing.assert_array_equal(
                sample.batch["transition"]["obs"],
                [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
            )

            client.close()
        finally:
            server.close()


class TestRoundTrip:
    def test_round_trip_multi_timescale(self, multi_server):
        server, port = multi_server
        client = TcpClient("localhost", port, _EXAMPLE_MULTI, max_inflight_msgs=4)
        try:
            obs_rows = [
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
                np.array([5.0, 6.0, 7.0, 8.0], dtype=np.float32),
            ]
            rew_vals = np.array([0.5, 1.5], dtype=np.float32)
            ret_val = np.array([42.0], dtype=np.float32)

            buf = TrajectoryAccumulator(_EXAMPLE_MULTI)
            for obs, rew in zip(obs_rows, rew_vals):
                buf.add("transition", {"obs": obs, "rew": np.array(rew)})
            buf.add("summary", {"ret": ret_val})

            client.send(buf.build())
            client.wait()

            sample = server.sample()
            np.testing.assert_array_equal(
                sample.batch["transition"]["obs"],
                [[obs_rows[0], obs_rows[1]]],
            )
            np.testing.assert_array_equal(sample.batch["transition"]["rew"], [rew_vals])
            np.testing.assert_array_equal(sample.batch["summary"]["ret"], [ret_val])
        finally:
            client.close()

    def test_round_trip_multiple_samples(self, multi_server):
        server, port = multi_server
        client = TcpClient("localhost", port, _EXAMPLE_MULTI, max_inflight_msgs=4)
        try:
            samples = [
                {
                    "obs": np.array([[10.0, 11.0, 12.0, 13.0], [14.0, 15.0, 16.0, 17.0]], dtype=np.float32),
                    "rew": np.array([1.0, 2.0], dtype=np.float32),
                    "ret": np.array([100.0], dtype=np.float32),
                },
                {
                    "obs": np.array([[20.0, 21.0, 22.0, 23.0], [24.0, 25.0, 26.0, 27.0]], dtype=np.float32),
                    "rew": np.array([3.0, 4.0], dtype=np.float32),
                    "ret": np.array([200.0], dtype=np.float32),
                },
            ]

            buf = TrajectoryAccumulator(_EXAMPLE_MULTI)
            for s in samples:
                for obs, rew in zip(s["obs"], s["rew"]):
                    buf.add("transition", {"obs": obs, "rew": np.array(rew)})
                buf.add("summary", {"ret": s["ret"]})
                client.send(buf.build())
            client.wait()

            for s in samples:
                smp = server.sample()
                np.testing.assert_array_equal(smp.batch["transition"]["obs"], [s["obs"]])
                np.testing.assert_array_equal(smp.batch["transition"]["rew"], [s["rew"]])
                np.testing.assert_array_equal(smp.batch["summary"]["ret"], [s["ret"]])
        finally:
            client.close()
