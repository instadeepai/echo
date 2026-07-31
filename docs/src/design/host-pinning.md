# Host-memory pinning

`src/host_pinning/` optionally CUDA-page-locks the ring buffers, so a
host-to-device copy of a sampled batch is a DMA transfer rather than a chunked
staging copy. This page is about how the module is put together; for the
mechanism, the measured numbers and how to verify it engaged, see the
[host-memory pinning guide](../guides/host-memory-pinning.md).

Two jobs, one file each:

| File | Role |
|---|---|
| `mod.rs` | What both halves need: `CudaApi`, `Region`, `PinError`. |
| `resolve.rs` | Find the CUDA runtime. |
| `register.rs` | Page-lock memory with it, and roll back cleanly. |

They interact at exactly one point: `resolve` returns a `CudaApi` and `register`
takes one. `resolve` makes no CUDA calls at all beyond `dlsym`, which is why the
split is clean rather than nominal — and why resolution, three quarters of the
source, is testable with no GPU anywhere in sight.

## Why resolution is a ladder

The whole feature was a silent no-op for months because it did one thing:
`dlopen("libcudart.so")`. The pip CUDA runtime wheels ship only the *versioned*
soname, with no unversioned symlink and no ldconfig entry, so that call returned
NULL and every pin became a no-op that reported nothing.

So resolution tries three rungs in order, accumulating every attempted path:

1. **Already-loaded scan.** Parse `/proc/self/maps` for a mapped runtime and
   `dlopen` its absolute path. Version- and path-agnostic, and it hits whenever
   the framework has already initialised CUDA — the common case. `dlopen` on the
   path of a mapped library reuses that mapping rather than loading a second copy.
2. **Installed-wheel search.** Search beneath the CUDA *vendor* package
   directories, which `py_bindings` locates through Python's import machinery and
   passes down. Searching the vendor package rather than a named component
   subpackage is load-bearing: CUDA 13 ships one consolidated wheel laid out as
   `nvidia/cu13/lib/`, CUDA 12 one wheel per component as
   `nvidia/cuda_runtime/lib/`. Naming the component finds nothing on a CUDA 13
   install.
3. **Soname load**, versioned before unversioned, each trying
   `RTLD_NOLOAD` before a full load.

`candidates()` builds the whole ladder as a pure function of the maps text and
the vendor roots, so its *ordering* is unit-tested. That is deliberate: reordering
the rungs is precisely how this broke, and the ordering is the kind of thing an
unrelated edit can silently change.

A rung that yields no candidate still reports itself, so a failure message
distinguishes "searched and found nothing" from "never ran".

`resolve` is the only place that knows about CUDA layouts; `py_bindings` supplies
the vendor roots because that is where a `Python` token exists, and keeping pyo3
out of `host_pinning` is what lets `cargo test` exercise it with no interpreter.

## Why registration is not in the constructor

`PytreeRingBuf::new` and `Store::new` stay infallible. Page-locking is a separate
`pin_host_memory` call on a fully-constructed buffer.

The reason is rollback, not tidiness. A constructor that returns `Err` never runs
`Drop`, so registering inside one forces a hand-written unregister loop on the
error path — the exact code most likely to be wrong and least likely to be
exercised. Registering afterwards lets `Drop` own rollback for both the failure
path and normal teardown. `PytreeRingBuf` holds `pinned_with: Option<CudaApi>`,
which is both the cue for `Drop` and what keeps `Drop` off any process-global.

Within one attempt it is all or nothing: if the *n*th buffer is rejected,
`pin_all` unregisters the preceding *n−1* before returning the error, so a failed
construction leaves nothing for a retry to accumulate. Registrations are still
leaked if a reference-counted `Store` outlives process shutdown, which is
accepted.

## The injection seam

`CudaApi` is a struct of function pointers passed explicitly to `pin_all` /
`unpin_all`, rather than a global the two reach into.

That exists for one reason: rollback is unsafe code whose only externally visible
consequence is the *absence* of leaked registrations — which cannot be observed
from Python at all. With injection, a test hands in a stub that fails on the third
buffer and asserts exactly two unregister calls, on a machine with no GPU.
`tests/host_pinning.rs` does that, and also drives a real `PytreeRingBuf` through
registration and `Drop` with the same stubs.

The module is `pub`, like every other module in the crate, because `tests/` are
separate crates and can only see `pub` items — the same reason
`PytreeRingBuf::slot_mut` is public. `CudaApi`'s fields stay private behind
`unsafe fn CudaApi::new`, so being public does not let a caller assemble one from
arbitrary function pointers, and `pin_all`/`unpin_all` remain `unsafe fn` with
stated contracts.

## No CUDA dependency at build time

Nothing links CUDA and nothing is generated from CUDA headers: the four entry
points (`cudaHostRegister`, `cudaHostUnregister`, `cudaGetErrorName`, `cudaFree`)
are `dlsym`'d at pin time. One wheel installs on GPU and CPU-only hosts alike, and
with pinning off no library is loaded and no code here runs.

The library is opened `RTLD_LOCAL` so echo cannot change how anything else in the
process resolves its symbols, and the handle is never `dlclose`d — the
registrations it backs must outlive it.
