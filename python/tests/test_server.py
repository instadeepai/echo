import threading

import numpy as np
import pytest
from echo import SampleInfo, Server
from echo.echo import _Server


class TestServerBasic:
    def test_single_sample_batch(self, make_server, make_client, transport_name):
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server, port = make_server(example, batch_size=1, transport=transport_name)
        client = make_client(transport_name, "localhost", port, example, max_inflight_msgs=32)

        client.send({"obs": np.array([1, 2, 3, 4], dtype=np.float32)})
        sample = server.sample()
        assert sample is not None
        np.testing.assert_array_equal(sample.batch["obs"], [[1, 2, 3, 4]])

    def test_multiple_samples_batch(self, make_server, make_client, transport_name):
        example = {"obs": np.zeros((2,), dtype=np.float32)}
        server, port = make_server(example, batch_size=3, transport=transport_name)
        client = make_client(transport_name, "localhost", port, example, max_inflight_msgs=32)

        client.send({"obs": np.array([1, 2], dtype=np.float32)})
        client.send({"obs": np.array([3, 4], dtype=np.float32)})
        client.send({"obs": np.array([5, 6], dtype=np.float32)})

        sample = server.sample()
        assert sample is not None
        assert sample.batch["obs"].shape == (3, 2)

    def test_shutdown_returns_none(self):
        example = {"obs": np.zeros((2,), dtype=np.float32)}
        server = Server(example, batch_size=10)
        server.close()
        assert server.sample() is None

    def test_sample_info_present(self):
        example = {"obs": np.zeros((2,), dtype=np.float32)}
        server = Server(example, batch_size=2)
        try:
            server.submit({"obs": np.array([1, 2], dtype=np.float32)})
            server.submit({"obs": np.array([3, 4], dtype=np.float32)})

            sample = server.sample()
            assert sample is not None
            assert isinstance(sample.info, SampleInfo)
            assert sample.info.active_connections == 0
            assert sample.info.push_blocked_count == 0
            assert sample.info.store_size >= 0
        finally:
            server.close()


class TestPytreeStructure:
    def test_nested_pytree_preserved(self, make_server, make_client, transport_name):
        example = {
            "obs": {
                "radars": np.zeros((8, 6), dtype=np.float32),
                "features": np.zeros((8,), dtype=np.float32),
            },
            "action": np.zeros((4,), dtype=np.int32),
        }
        server, port = make_server(example, batch_size=1, transport=transport_name)
        client = make_client(transport_name, "localhost", port, example, max_inflight_msgs=32)

        sample_in = {
            "obs": {
                "radars": np.ones((8, 6), dtype=np.float32),
                "features": np.arange(8, dtype=np.float32),
            },
            "action": np.array([1, 2, 3, 4], dtype=np.int32),
        }
        client.send(sample_in)

        sample = server.sample()
        assert sample is not None
        np.testing.assert_array_equal(sample.batch["obs"]["radars"], np.ones((1, 8, 6), dtype=np.float32))
        np.testing.assert_array_equal(sample.batch["obs"]["features"], np.arange(8, dtype=np.float32)[np.newaxis, :])
        np.testing.assert_array_equal(sample.batch["action"], [[1, 2, 3, 4]])

    def test_dtypes_preserved(self, make_server, make_client, transport_name):
        example = {
            "float32": np.zeros((2,), dtype=np.float32),
            "int32": np.zeros((2,), dtype=np.int32),
            "uint8": np.zeros((2,), dtype=np.uint8),
        }
        server, port = make_server(example, batch_size=1, transport=transport_name)
        client = make_client(transport_name, "localhost", port, example, max_inflight_msgs=32)

        client.send({
            "float32": np.array([1.5, 2.5], dtype=np.float32),
            "int32": np.array([100, 200], dtype=np.int32),
            "uint8": np.array([255, 128], dtype=np.uint8),
        })

        sample = server.sample()
        assert sample is not None
        assert sample.batch["float32"].dtype == np.float32
        assert sample.batch["int32"].dtype == np.int32
        assert sample.batch["uint8"].dtype == np.uint8
        np.testing.assert_array_equal(sample.batch["float32"], [[1.5, 2.5]])
        np.testing.assert_array_equal(sample.batch["int32"], [[100, 200]])
        np.testing.assert_array_equal(sample.batch["uint8"], [[255, 128]])

    def test_client_rejects_wrong_payload_size(self, make_server, make_client, transport_name):
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server, port = make_server(example, batch_size=10, transport=transport_name)
        client = make_client(transport_name, "localhost", port, example, max_inflight_msgs=32)

        with pytest.raises(ValueError, match="Payload size"):
            client.send({"obs": np.array([1, 2], dtype=np.float32)})


class TestMultipleActors:
    def test_multiple_actors_single_batch(self, make_server, make_client, transport_name):
        example = {"value": np.zeros((1,), dtype=np.int32)}
        server, port = make_server(example, batch_size=4, transport=transport_name)

        expected_values = {10, 20, 30, 40}
        for val in expected_values:
            c = make_client(transport_name, "localhost", port, example, max_inflight_msgs=32)
            c.send({"value": np.array([val], dtype=np.int32)})

        sample = server.sample()
        assert sample is not None
        assert set(sample.batch["value"].flatten().tolist()) == expected_values

    def test_concurrent_actors(self, make_server, make_client, transport_name):
        example = {"id": np.zeros((1,), dtype=np.int32)}
        num_actors = 10
        samples_per_actor = 10
        server, port = make_server(example, batch_size=num_actors * samples_per_actor, transport=transport_name)

        lock = threading.Lock()
        clients = []

        def actor(actor_id):
            c = make_client(transport_name, "localhost", port, example, max_inflight_msgs=32)
            with lock:
                clients.append(c)
            for i in range(samples_per_actor):
                c.send({"id": np.array([actor_id * 1000 + i], dtype=np.int32)})

        threads = [threading.Thread(target=actor, args=(i,)) for i in range(num_actors)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sample = server.sample()
        assert sample is not None
        values = sample.batch["id"].flatten().tolist()
        assert len(values) == num_actors * samples_per_actor
        assert len(set(values)) == num_actors * samples_per_actor


class TestZeroCopy:
    def test_arrays_are_views_not_copies(self):
        server = _Server(shapes=[[4]], dtype_sizes=[4], batch_size=1, num_buffers=2)
        server.submit([np.zeros(4, dtype=np.float32).tobytes()])
        result = server.sample()
        assert result is not None
        arrays, _info = result
        assert not arrays[0].flags.owndata
        server.shutdown()

    def test_data_correct_pool_cycling(self):
        server = _Server(shapes=[[2]], dtype_sizes=[1], batch_size=2, num_buffers=3)

        expected_batches = [
            ([1, 2], [3, 4]),
            ([5, 6], [7, 8]),
            ([9, 10], [11, 12]),
            ([13, 14], [15, 16]),
        ]

        for batch_samples in expected_batches:
            for s in batch_samples:
                server.submit([bytes(s)])
            result = server.sample()
            assert result is not None
            arrays, _info = result
            data = np.frombuffer(bytes(arrays[0]), dtype=np.uint8)
            expected = np.array([b for s in batch_samples for b in s], dtype=np.uint8)
            np.testing.assert_array_equal(data, expected)

        server.shutdown()

    def test_data_correct_across_sequential_batches(self, make_server, make_client):
        example = {"obs": np.zeros((2,), dtype=np.float32)}
        server, port = make_server(example, batch_size=1, transport="tcp", num_buffers=3)
        client = make_client("tcp", "localhost", port, example, max_inflight_msgs=32)

        client.send({"obs": np.array([1.0, 2.0], dtype=np.float32)})
        sample1 = server.sample()
        assert sample1 is not None
        saved = sample1.batch["obs"].copy()

        client.send({"obs": np.array([3.0, 4.0], dtype=np.float32)})
        sample2 = server.sample()
        assert sample2 is not None

        np.testing.assert_array_almost_equal(sample2.batch["obs"], [[3.0, 4.0]])
        np.testing.assert_array_almost_equal(saved, [[1.0, 2.0]])

    def test_num_buffers_accepted(self):
        example = {"obs": np.zeros((2,), dtype=np.float32)}
        for n in [2, 3, 5]:
            s = Server(example, batch_size=10, num_buffers=n)
            s.close()


class TestSubmit:
    def test_submit_single_sample(self):
        example = {"obs": np.zeros((4,), dtype=np.float32)}
        server = Server(example, batch_size=1)
        server.submit({"obs": np.array([1, 2, 3, 4], dtype=np.float32)})
        sample = server.sample()
        assert sample is not None
        np.testing.assert_array_equal(sample.batch["obs"], [[1, 2, 3, 4]])
        server.close()

    def test_submit_multiple_samples(self):
        example = {"obs": np.zeros((2,), dtype=np.float32)}
        server = Server(example, batch_size=3)
        server.submit({"obs": np.array([1, 2], dtype=np.float32)})
        server.submit({"obs": np.array([3, 4], dtype=np.float32)})
        server.submit({"obs": np.array([5, 6], dtype=np.float32)})

        sample = server.sample()
        assert sample is not None
        np.testing.assert_array_equal(sample.batch["obs"], [[1, 2], [3, 4], [5, 6]])
        server.close()

    def test_submit_nested_pytree(self):
        example = {
            "obs": {
                "pos": np.zeros((3,), dtype=np.float32),
                "vel": np.zeros((3,), dtype=np.float32),
            },
            "reward": np.zeros((1,), dtype=np.float32),
        }
        server = Server(example, batch_size=1)
        server.submit({
            "obs": {
                "pos": np.array([1, 2, 3], dtype=np.float32),
                "vel": np.array([4, 5, 6], dtype=np.float32),
            },
            "reward": np.array([7.0], dtype=np.float32),
        })

        sample = server.sample()
        assert sample is not None
        np.testing.assert_array_equal(sample.batch["obs"]["pos"], [[1, 2, 3]])
        np.testing.assert_array_equal(sample.batch["obs"]["vel"], [[4, 5, 6]])
        np.testing.assert_array_equal(sample.batch["reward"], [[7.0]])
        server.close()

    def test_submit_dtypes_preserved(self):
        example = {
            "f32": np.zeros((2,), dtype=np.float32),
            "i32": np.zeros((2,), dtype=np.int32),
            "u8": np.zeros((2,), dtype=np.uint8),
        }
        server = Server(example, batch_size=1)
        server.submit({
            "f32": np.array([1.5, 2.5], dtype=np.float32),
            "i32": np.array([100, 200], dtype=np.int32),
            "u8": np.array([255, 128], dtype=np.uint8),
        })

        sample = server.sample()
        assert sample is not None
        assert sample.batch["f32"].dtype == np.float32
        assert sample.batch["i32"].dtype == np.int32
        assert sample.batch["u8"].dtype == np.uint8
        server.close()

    def test_submit_sequential_batches(self):
        example = {"val": np.zeros((1,), dtype=np.int32)}
        server = Server(example, batch_size=2, num_buffers=3)

        for i in range(4):
            server.submit({"val": np.array([i * 2], dtype=np.int32)})
            server.submit({"val": np.array([i * 2 + 1], dtype=np.int32)})
            sample = server.sample()
            assert sample is not None
            np.testing.assert_array_equal(sample.batch["val"], [[i * 2], [i * 2 + 1]])

        server.close()


class TestDatasetIter:
    def test_zero_copy_iterates_batches(self):
        example = {"val": np.zeros((2,), dtype=np.float32)}
        server = Server(example, 1)
        it = server.dataset_iter()

        server.submit({"val": np.array([1.0, 2.0], dtype=np.float32)})
        r1 = next(it).batch["val"].copy()

        server.submit({"val": np.array([3.0, 4.0], dtype=np.float32)})
        r2 = next(it).batch["val"].copy()

        server.submit({"val": np.array([5.0, 6.0], dtype=np.float32)})
        r3 = next(it).batch["val"].copy()

        server.close()

        np.testing.assert_array_equal(r1, [[1.0, 2.0]])
        np.testing.assert_array_equal(r2, [[3.0, 4.0]])
        np.testing.assert_array_equal(r3, [[5.0, 6.0]])

    def test_copy_true_arrays_survive_next_iteration(self):
        example = {"val": np.zeros((2,), dtype=np.float32)}
        server = Server(example, 1)
        it = server.dataset_iter(copy=True)

        server.submit({"val": np.array([1.0, 2.0], dtype=np.float32)})
        sample1 = next(it)

        server.submit({"val": np.array([3.0, 4.0], dtype=np.float32)})
        sample2 = next(it)

        server.close()

        np.testing.assert_array_equal(sample1.batch["val"], [[1.0, 2.0]])
        np.testing.assert_array_equal(sample2.batch["val"], [[3.0, 4.0]])

    def test_shutdown_stops_iteration(self):
        example = {"val": np.zeros((2,), dtype=np.float32)}
        server = Server(example, 1)
        server.close()
        assert list(server.dataset_iter()) == []


class TestResetHistograms:
    def test_reset_does_not_raise(self):
        example = {"obs": np.zeros((2,), dtype=np.float32)}
        server = Server(example, batch_size=1)
        server.reset_histograms()
        server.close()
