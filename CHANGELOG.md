# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-05-18

Initial public release.

- Lockfree, pre-allocated Rust ring buffer with zero-copy numpy batches.
- TCP transport with per-connection SPSC queues and a drainer pool.
- Only FIFO sampling/adding/removing
