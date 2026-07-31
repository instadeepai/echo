"""Tests for ``Server(pin_host_memory=...)``.

The contract under test is that *if the constructor returns, every ring buffer
is page-locked*. That is deliberately what echo offers instead of a status
object: construction succeeding is the assertion, and unlike an explicit check
it cannot be forgotten.

So the confirmation here does not ask echo what it did. It asks the CUDA runtime
directly, through ``ctypes``, for the registration flags on the address behind a
sampled view. A test that trusted echo's own account of its work would pass just
as happily against the silent no-op this feature replaced.

Note that ``VmLck`` in ``/proc/self/status`` is *not* a valid check: the NVIDIA
driver's page-locking does not go through mlock accounting, so it stays at zero
even when pinning demonstrably works.
"""
import ctypes
import glob
import os
import re
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from echo import Server

CUDA_SUCCESS = 0
CUDA_HOST_REGISTER_PORTABLE = 0x01

EXAMPLE = {
    "obs": np.zeros((16,), dtype=np.float32),
    "reward": np.zeros((1,), dtype=np.float32),
}


def _cudart_paths() -> list[str]:
    """Candidate CUDA runtime libraries, mirroring echo's resolution ladder.

    Independent of echo's implementation on purpose: the point of these tests is
    to reach the runtime without going through the code under test.
    """
    candidates = []
    with open("/proc/self/maps") as maps:
        candidates += re.findall(r"\s(/\S*libcudart\.so[.\d]*)", maps.read())
    for site in sys.path:
        candidates += glob.glob(os.path.join(site, "nvidia", "*", "lib", "libcudart.so*"))
    candidates += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    return [c for c in candidates if not c.endswith(".a")]


def _load_cudart() -> ctypes.CDLL | None:
    """The CUDA runtime, or None on a machine without one."""
    for path in _cudart_paths():
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue
    return None


CUDART = _load_cudart()


def _device_count(cudart: ctypes.CDLL) -> int:
    count = ctypes.c_int(0)
    if cudart.cudaGetDeviceCount(ctypes.byref(count)) != CUDA_SUCCESS:
        return 0
    return count.value


HAS_GPU = CUDART is not None and _device_count(CUDART) > 0
requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="no CUDA device available")


def registration_flags(address: int) -> tuple[int, int]:
    """The CUDA runtime's own account of how ``address`` is registered.

    Returns ``(cuda_error_code, flags)``; a non-zero code means the runtime does
    not know the address as page-locked host memory.
    """
    assert CUDART is not None
    flags = ctypes.c_uint(0)
    code = CUDART.cudaHostGetFlags(ctypes.byref(flags), ctypes.c_void_p(address))
    return code, flags.value


def batch_address(sample) -> int:
    """Address of the first leaf of a sampled batch.

    The first batch starts at ring slot 0, so this is the base of the buffer.
    """
    leaves = [sample.batch[key] for key in sorted(sample.batch)]
    return leaves[0].ctypes.data


class TestDefaultOff:
    def test_constructs_and_serves_batches(self):
        """Off by default, and off is the unchanged behaviour — this runs on
        CPU-only machines and CI."""
        server = Server(EXAMPLE, batch_size=2)
        try:
            for _ in range(2):
                server.submit({k: v.copy() for k, v in EXAMPLE.items()})
            sample = server.sample()
            assert sample is not None
            assert sample.batch["obs"].shape == (2, 16)
        finally:
            server.close()

    @pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs procfs")
    def test_loads_no_cuda_library(self):
        """The default path must behave like a build without the feature.

        Checked in a subprocess because this module loads the CUDA runtime
        through ``ctypes`` at import, which would mask a load by echo. Runs on
        CPU-only machines too — there it asserts the absence stays an absence.
        """
        script = textwrap.dedent(
            """
            import re
            import numpy as np
            from echo import Server

            def mapped():
                with open("/proc/self/maps") as f:
                    return set(re.findall(r"\\s(/\\S*libcudart\\.so[.\\d]*)", f.read()))

            before = mapped()
            server = Server({"obs": np.zeros((32,), dtype=np.float32)}, 4)
            assert mapped() == before, "default-off loaded a CUDA library"
            server.close()
            print("ok")
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "ok" in result.stdout


@pytest.mark.skipif(HAS_GPU, reason="a usable CUDA device is present")
def test_requesting_pinning_without_cuda_raises_and_lists_probed_paths():
    """The failure a misconfigured deployment gets: loud, at startup, specific."""
    with pytest.raises(RuntimeError) as excinfo:
        Server(EXAMPLE, batch_size=4, pin_host_memory=True)

    message = str(excinfo.value)
    assert "libcudart.so" in message, message
    # Every rung reports what it tried, so the user can fix their environment
    # without reading echo's source.
    assert "soname load" in message, message


@requires_gpu
@pytest.mark.gpu
def test_pins_when_no_runtime_is_loaded_yet():
    """A server constructed *before* the framework touches CUDA must still pin.

    This test module loads the runtime through ``ctypes`` at import, so the
    tests below resolve it from the process's own mappings. Here the subprocess
    has nothing mapped, which leaves finding the pip-installed wheel — the case
    the original defect broke, and the reason it needed no loader-path
    environment variable to be set.
    """
    script = textwrap.dedent(
        """
        import re
        import numpy as np
        with open("/proc/self/maps") as f:
            assert not re.search(r"libcudart\\.so", f.read()), "runtime already mapped"

        from echo import Server
        server = Server({"obs": np.zeros((32,), dtype=np.float32)}, 4, pin_host_memory=True)
        with open("/proc/self/maps") as f:
            assert re.search(r"libcudart\\.so", f.read()), "nothing was loaded"
        server.close()
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@requires_gpu
@pytest.mark.gpu
class TestPinningOnGpu:
    def test_construction_page_locks_every_ring_buffer(self):
        server = Server(EXAMPLE, batch_size=4, pin_host_memory=True)
        try:
            for _ in range(4):
                server.submit({k: v.copy() for k, v in EXAMPLE.items()})
            sample = server.sample()
            assert sample is not None

            for name in sorted(sample.batch):
                address = sample.batch[name].ctypes.data
                code, flags = registration_flags(address)
                assert code == CUDA_SUCCESS, f"{name} is not registered host memory"
                assert flags & CUDA_HOST_REGISTER_PORTABLE, (
                    f"{name} is registered but not portable (flags={flags:#x})"
                )
        finally:
            server.close()

    def test_batch_is_bit_identical_with_and_without_pinning(self):
        rng = np.random.default_rng(0)
        samples = [
            {
                "obs": rng.standard_normal(16).astype(np.float32),
                "reward": rng.standard_normal(1).astype(np.float32),
            }
            for _ in range(4)
        ]

        batches = []
        for pin in (False, True):
            server = Server(EXAMPLE, batch_size=4, pin_host_memory=pin)
            try:
                for sample in samples:
                    server.submit(sample)
                got = server.sample()
                assert got is not None
                batches.append({k: np.copy(v) for k, v in got.batch.items()})
            finally:
                server.close()

        unpinned, pinned = batches
        for name in unpinned:
            assert unpinned[name].tobytes() == pinned[name].tobytes(), name

    def test_two_servers_in_one_process_both_pin(self):
        """One server per GPU in one process needs no device configuration:
        registration is portable across every context in the process."""
        first = Server(EXAMPLE, batch_size=4, pin_host_memory=True)
        second = Server(EXAMPLE, batch_size=8, pin_host_memory=True)
        try:
            for server, batch_size in ((first, 4), (second, 8)):
                for _ in range(batch_size):
                    server.submit({k: v.copy() for k, v in EXAMPLE.items()})
                sample = server.sample()
                assert sample is not None
                code, flags = registration_flags(batch_address(sample))
                assert code == CUDA_SUCCESS
                assert flags & CUDA_HOST_REGISTER_PORTABLE
        finally:
            first.close()
            second.close()

    def test_pinning_survives_server_teardown_and_reconstruction(self):
        """A closed-and-dropped server unregisters, and the next one still pins:
        a long-lived process must not accumulate or exhaust registrations."""
        for _ in range(3):
            server = Server(EXAMPLE, batch_size=4, pin_host_memory=True)
            try:
                for _ in range(4):
                    server.submit({k: v.copy() for k, v in EXAMPLE.items()})
                sample = server.sample()
                assert sample is not None
                code, _ = registration_flags(batch_address(sample))
                assert code == CUDA_SUCCESS
            finally:
                server.close()
                del server
