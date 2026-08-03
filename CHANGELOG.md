# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-08-03

### Added

- `Server(..., pin_host_memory=True)` CUDA-page-locks the ring buffers, so a
  downstream host-to-device copy of a sampled batch is a real DMA transfer on the
  copy engine instead of a chunked, driver-mediated staging copy. Off by default.
  Measured locally, `cudaMemcpyAsync` out of pageable memory holds the calling
  thread for 97% of the copy; out of page-locked memory it returns in 0.002 ms.
- If the constructor returns, every ring buffer is page-locked. Any inability to
  deliver that raises `RuntimeError` naming every path probed for the CUDA
  runtime and the CUDA error symbolically. There is no silent no-op.
- The CUDA runtime is resolved through a three-rung ladder — already-mapped
  libraries, then the installed CUDA wheels, then sonames (versioned before
  unversioned) — so no soname symlink or `LD_LIBRARY_PATH` entry is needed.
- New guide: [Host-memory pinning](https://instadeepai.github.io/echo/guides/host-memory-pinning/),
  covering the mechanism, the constructor guarantee, the unswappable footprint
  arithmetic, and how the CUDA runtime is located.

## [0.1.1] - 2026-05-26

- `TrajectoryAccumulator` better supports single and buffered timescales
    - Buffered if: all leaves have a leading dim > 1
    - Single if: all leaves have leading dims == 1 or there is a single scalar
- Guide and API reference cover both accumulator modes and the detection rule.
- New project logo and README polish (#3).

## Pull Requests
- docs: add logo by @sash-a in https://github.com/instadeepai/echo/pull/3
- feat: better accumulator by @sash-a in https://github.com/instadeepai/echo/pull/4

**Full Changelog**: https://github.com/instadeepai/echo/compare/v0.1.0...v0.1.1

## [0.1.0] - 2026-05-18

Initial public release.

- Lockfree, pre-allocated Rust ring buffer with zero-copy numpy batches.
- TCP transport with per-connection SPSC queues and a drainer pool.
- Only FIFO sampling/adding/removing
