# Host-memory pinning

`Server(..., pin_host_memory=True)` CUDA-page-locks echo's ring buffers, so
that copying a sampled batch to a GPU is a real DMA transfer instead of a
chunked copy the driver walks through by hand.

It is off by default and it is not free — the page-locked memory is not
swappable. This page covers what the mechanism actually is, what it measured,
how to size the footprint, and — the part that is easy to get wrong — **how to
confirm it engaged**.

```python
server = Server(example, batch_size=512, transport=TcpTransport(port=50051),
                pin_host_memory=True)
```

## The guarantee

> **If `Server(...)` returns, every ring buffer is page-locked.**

There is no state in which you asked for pinning and silently did not get it.
Anything that would prevent it — no CUDA runtime found, no usable device, a
registration rejected — raises `RuntimeError` from the constructor, naming every
path echo probed and the CUDA error symbolically:

```text
RuntimeError: pin_host_memory=True but no CUDA runtime could be loaded. Probed, in order:
  - libcudart.so.13 [soname load]: libcudart.so.13: cannot open shared object file: No such file or directory
  - libcudart.so.12 [soname load]: libcudart.so.12: cannot open shared object file: No such file or directory
  - libcudart.so [soname load]: libcudart.so: cannot open shared object file: No such file or directory
Install a CUDA runtime (for example the nvidia-cuda-runtime wheel), or construct
the Server after your framework has initialised CUDA.
```

That is deliberately all the API there is: no status object, no report method, no
warning. Construction succeeding *is* the assertion, and unlike an explicit check
you might add, it cannot be forgotten. If you would rather degrade than fail,
catch `RuntimeError`.

With the argument off, echo loads no CUDA library at all and behaves exactly like
a build without the feature.

## The mechanism: a pageable "async" copy is not async

Ordinary heap memory is *pageable* — the OS may move or swap it out, so the GPU
cannot safely DMA out of it. When you copy from pageable memory, the driver
cannot hand the transfer to the copy engine and walk away. Instead it:

1. copies a chunk of your data into a small internal page-locked staging buffer,
2. DMAs that chunk to the device,
3. repeats, thousands of times for a large batch.

Steps 1–3 run on the calling CPU thread and hold the driver lock. Two things
follow, and the second is usually the expensive one:

- the bytes move slower, because every chunk pays a round-trip;
- **`cudaMemcpyAsync` stops being asynchronous.** The call does not return until
  the staging loop is essentially done, and while it runs, the driver lock it
  holds blocks the *same thread's* kernel launches. On a learner issuing tens of
  thousands of small launches per step, that contention — not the bandwidth —
  is what shows up as host-side dispatch dominating the step while the GPU sits
  idle waiting for work.

Page-locking the ring buffers removes the staging loop. The pages cannot move, so
the driver writes one DMA descriptor to the copy engine and returns immediately.

Echo registers with `cudaHostRegisterPortable`, which makes the locked pages
valid in **every** CUDA context in the process, including contexts created later.
That is why there is no device argument: N servers across N GPUs in one process
each pin without any per-server configuration, and echo never selects a device or
allocates device memory.

## Measured

Local microbenchmark on a **NVIDIA GeForce RTX 5060 Ti (16 GB), driver
595.71.05, CUDA runtime 13.3**, sized like a large-batch learner:
`batch_size=512` over four arrays (34.1 MB per batch, 102.2 MB ring), with the
copy measurement isolated to a single 52.4 MB buffer. 11 repeats; median and
interquartile range:

Through echo's own ring buffers and sampled views (34.1 MB batch, 102.2 MB ring):

| arm | copy time | IQR | throughput | host-write | registration |
|---|---|---|---|---|---|
| pageable | 2.441 ms | 2.440–2.442 | 14.0 GB/s | 12.32 GB/s | – |
| page-locked (portable) | 2.401 ms | 2.399–2.407 | 14.2 GB/s | 12.39 GB/s | 8.9 ms |

And isolating one 52.4 MB buffer, to compare against driver-allocated pinned
memory and to time the copy call on its own:

| source memory | copy time | IQR | throughput | `cudaMemcpyAsync` returns after |
|---|---|---|---|---|
| pageable | 3.751 ms | 3.748–3.752 | 14.0 GB/s | 3.645 ms — **97% of the copy** |
| page-locked | 3.715 ms | 3.714–3.719 | 14.1 GB/s | 0.002 ms — **0%** |
| `cudaHostAlloc` (reference ceiling) | 3.658 ms | 3.657–3.658 | 14.3 GB/s | 0.002 ms |

Read the throughput and the return-latency columns separately, because they say
different things.

**Bandwidth barely moves on this machine, and that is expected.** Both paths
saturate the host's link at ~14 GB/s; driver-allocated pinned memory only reaches
14.3 GB/s, so there is nothing more to win here. A host with more PCIe headroom
will show a larger gap.

**Host-thread occupancy collapses by ~1800x.** That is the mechanism above,
measured: the same call goes from holding the calling thread for 3.645 ms to
0.002 ms. This is the component that scales into the driver-lock contention a
large-batch learner suffers, and it is the reason to turn pinning on.

**Nothing on the write side pays for it.** Host-write throughput into the ring —
what a drainer costs — is unchanged at 12.3 GB/s, and registering the 102 MB ring
took 8.9 ms once, at construction. Pinning changes only how the memory is mapped,
not any code on the ingest or sample path, so there is no per-sample or per-drain
cost. A batch sampled with pinning on is bit-identical to the same batch with it
off.

**Only the portable flag is used.** Two variants were measured and rejected.
Page-aligning the ring buffers gained 1.6% of copy time and nothing at all on
return latency, well under the bar set in advance, so the buffers stay plain
`Vec<u8>`. Read-only registration (`cudaHostRegisterReadOnly`) is not usable
here at all — `cudaHostRegister` returns `cudaErrorNotSupported` on this GPU,
whose `cudaDevAttrHostRegisterReadOnlySupported` is 0 — and the CUDA
documentation describes that flag as permission to register memory *mapped*
read-only rather than as a transfer optimisation, while saying nothing about host
writes to such a range. Echo's drainers write these pages continuously, so there
would be no documented basis for using it even where it is supported.

**What this microbenchmark cannot tell you.** A single consumer GPU cannot
reproduce the driver-lock contention of a real learner issuing tens of thousands
of small kernel launches concurrently with the staging copy. The table above
measures the bandwidth component and the mechanism; it does *not* size the win on
a large-batch learner, where the contention component dominates and the effect is
correspondingly larger than the 1–3% bandwidth figure here.

So do not read 1–3% as "pinning is worth 1–3%", and do not read the occupancy
column as a step-time prediction either. Measure your own workload: turn pinning
on, confirm it engaged (below), and compare step times.

To reproduce the table: allocate a buffer with `malloc`, `cudaMalloc` a
destination, and time `cudaMemcpyAsync` + `cudaStreamSynchronize` before and
after `cudaHostRegister(ptr, size, cudaHostRegisterPortable)`. Time the
`cudaMemcpyAsync` call *on its own*, without the synchronize, to see the
occupancy column.

## Sizing the footprint

The whole ring is locked, not one batch:

```text
page-locked bytes = batch_size x num_buffers x sum(leaf.nbytes for leaf in example)
```

For a batch of 512 with `num_buffers=3` and 67 KB per sample, that is
512 x 3 x 67 KB ≈ 102 MB.

Two things to hold onto:

- **It is not swappable.** Those pages are removed from the pool the kernel can
  reclaim under pressure, for the whole life of the server.
- **It multiplies by the number of servers.** One `Server` per GPU in a single
  process means N times the figure above. Size the host before the job does it
  for you.

Note that `VmLck` in `/proc/self/status` stays at **zero** even when pinning is
working, so a memory-lock resource limit (`ulimit -l`) does not bind here — see
[below](#verifying-that-pinning-engaged).

## Construct the server after CUDA is initialised

Registration needs an initialised CUDA runtime, so the supported order is:

1. let your framework initialise CUDA,
2. then construct the `Server`.

Echo does not depend on you getting this right — before registering it forces
initialisation best-effort, by freeing a null pointer. That frees nothing and
never *changes* which device is current, so it cannot perturb your framework's
device selection.

It is not entirely free, though, and this is the reason the ordering above is a
recommendation rather than a footnote: initialising the runtime creates the
primary CUDA context on whatever device is already current, and that context
costs device memory (~128 MB on the machine in the table above). Construct after
your framework has initialised CUDA and the context already exists, so echo adds
nothing. Construct before it, and echo creates the context first — which both
spends that memory early and may interact badly with a framework that
pre-allocates a fraction of *free* device memory.

Echo never allocates device buffers of its own and never calls a
set-device function.

If initialisation genuinely cannot happen, or no device is usable, the
constructor raises with the CUDA error named. It never degrades silently.

## Finding the CUDA runtime

Echo has no build-time or link-time CUDA dependency; one wheel installs on GPU
and CPU-only hosts alike. The runtime is located at pin time by trying, in order:

1. **What the process already has mapped.** Version- and path-agnostic; hits
   whenever your framework has already initialised CUDA.
2. **What the installed CUDA wheels ship.** Echo locates the `nvidia` vendor
   package through Python's import machinery and searches beneath it. It searches
   the vendor package rather than a named component because CUDA 13 ships one
   consolidated wheel (`nvidia/cu13/lib/`) whose layout differs from CUDA 12's
   per-component layout (`nvidia/cuda_runtime/lib/`).
3. **Sonames**, versioned first (`libcudart.so.13`, `libcudart.so.12`) and the
   unversioned `libcudart.so` last.

You should not need a soname symlink in your image or an `LD_LIBRARY_PATH` entry.
If you are carrying either as a workaround, remove it — and note that with them
in place, rung 3 hides whether rungs 1 and 2 work.

## Verifying that pinning engaged

This section is the reason this page exists. Pinning was once believed not to
help a workload that had in fact never executed the pinning code path, and the
knowledge of how to check lived in one person's head.

**In-process: the constructor.** `Server(..., pin_host_memory=True)` returning
*is* the assertion. There is nothing else to query.

**From a profile: the copy's memory-source classification.** Profile the learner
and look at the host-to-device memcpy rows. A profiler reports the source kind
for each transfer; it must say the source is pinned/page-locked rather than
pageable. This is the authoritative external signal.

**From a test, without a framework.** Ask the CUDA runtime directly for the flags
on the address behind a sampled batch — this is what echo's own test suite does,
precisely so that a test cannot pass merely because echo believes it worked:

```python
import ctypes

cudart = ctypes.CDLL("libcudart.so.13")
flags = ctypes.c_uint(0)
code = cudart.cudaHostGetFlags(
    ctypes.byref(flags), ctypes.c_void_p(batch["obs"].ctypes.data)
)
assert code == 0                # 0 = cudaSuccess; non-zero means not registered
assert flags.value & 0x01       # cudaHostRegisterPortable
```

!!! warning "`VmLck` is not a valid check"

    `VmLck` in `/proc/self/status` stays at **zero** even when pinning
    demonstrably works: the NVIDIA driver's page-locking does not go through
    mlock accounting. Anyone who reads it as a check will conclude pinning is off
    when it is on. The same goes for any tool built on mlock accounting. Use the
    profile trace or `cudaHostGetFlags`.

## What is not covered

- **Non-CUDA accelerators.** macOS wheels build and the default-off path is a
  clean no-op there, but no equivalent functionality is provided.
- **Anything other than the ring buffers.** Producer queues, transport staging
  buffers and accumulator storage are not registered: the ring buffers are the
  only source of a host-to-device copy.
- **Allocating pinned memory directly** instead of registering the existing
  buffers. Steady-state transfer performance would be identical, so it would
  change only construction cost while making the CUDA runtime mandatory at
  allocation time.
