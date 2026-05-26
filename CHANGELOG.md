# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
