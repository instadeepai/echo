# Host-memory pinning

`Server(..., pin_host_memory=True)` CUDA-page-locks echo's ring buffers, so that
copying a sampled batch to a GPU is a real DMA transfer instead of a chunked copy
the driver walks through by hand.

It is off by default, and it is not free: page-locked memory is not swappable.

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
  - (none) [already-loaded scan]: this process has no CUDA runtime mapped
  - (none) [installed-wheel search]: no CUDA vendor package is importable
  - libcudart.so.13 [soname load]: libcudart.so.13: cannot open shared object file: No such file or directory
  - libcudart.so.12 [soname load]: libcudart.so.12: cannot open shared object file: No such file or directory
  - libcudart.so [soname load]: libcudart.so: cannot open shared object file: No such file or directory
Install a CUDA runtime (for example the nvidia-cuda-runtime wheel), or construct
the Server after your framework has initialised CUDA.
```

A `(none)` line means that rung ran and found nothing, so the message
distinguishes it from a rung that never ran at all.

That is all the API there is: no status object, no report method, no warning. If
you would rather degrade than fail, catch `RuntimeError`.

With the argument off, echo loads no CUDA library at all and behaves exactly like
a build without the feature.

## Why a pageable "async" copy is not async

Ordinary heap memory is *pageable* — the OS may move or swap it out, so the GPU
cannot safely DMA out of it. When you copy from pageable memory, the driver
cannot hand the transfer to the copy engine and walk away. Instead it:

1. copies a chunk of your data into a small internal page-locked staging buffer,
2. DMAs that chunk to the device,
3. repeats, thousands of times for a large batch.

Steps 1–3 run on the calling CPU thread and hold the driver lock, with two
consequences:

- the bytes move slower, because every chunk pays a round-trip;
- **`cudaMemcpyAsync` stops being asynchronous.** The call does not return until
  the staging loop is essentially done, and while it runs, the driver lock it
  holds blocks the *same thread's* kernel launches. On a learner issuing tens of
  thousands of small launches per step, that contention — not the bandwidth — is
  what shows up as host-side dispatch dominating the step while the GPU sits idle.

Page-locking the ring buffers removes the staging loop. The pages cannot move, so
the driver writes one DMA descriptor to the copy engine and returns immediately.

Echo registers with `cudaHostRegisterPortable`, which makes the locked pages
valid in **every** CUDA context in the process, including contexts created later.
That is why there is no device argument: N servers across N GPUs in one process
each pin without any per-server configuration, and echo never selects a device or
allocates device memory.

## Sizing the footprint

The whole ring is locked, not one batch:

```text
page-locked bytes = batch_size x num_buffers x sum(leaf.nbytes for leaf in example)
```

For a batch of 512 with `num_buffers=3` and 67 KB per sample, that is
512 x 3 x 67 KB ≈ 102 MB.

- **It is not swappable.** Those pages are removed from the pool the kernel can
  reclaim under pressure, for the whole life of the server.
- **It multiplies by the number of servers.** One `Server` per GPU in a single
  process means N times the figure above, so size the host accordingly.

Note that `VmLck` in `/proc/self/status` stays at **zero** even when pinning is
working, so a memory-lock resource limit (`ulimit -l`) does not bind here — see
[below](#verifying-that-pinning-engaged).

## Construct the server after CUDA is initialised

Registration needs an initialised CUDA runtime, so the supported order is:

1. let your framework initialise CUDA,
2. then construct the `Server`.

Echo does not depend on you getting this right — before registering it forces
initialisation best-effort, by freeing a null pointer. 

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

The ladder and why each rung exists are covered in
[Host-memory pinning](../design/host-pinning.md) under Rust internals.

## What is not covered

- **Non-CUDA accelerators.** macOS wheels build and the default-off path is a
  clean no-op there, but no equivalent functionality is provided.
