use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_utils::CachePadded;
use parking_lot::{Condvar, Mutex};

use crate::metrics::Metrics;

/// Result of a select() call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleResult {
    /// Contiguous range in the ring buffer (zero-copy path).
    Contiguous { start: usize, count: usize },
    // Future: Scattered(Vec<usize>) for uniform/prioritized
}

/// Determines which items to sample and when they're ready.
pub trait Sampler: Send + Sync {
    /// Signal that the item at absolute position `pos` has been written.
    /// Non-blocking: the implementation is responsible for ensuring the
    /// consumer only observes fully-written batches.
    fn commit(&self, pos: usize);

    /// Signal that `n` items starting at absolute position `start_pos` have
    /// been written. Caller must ensure `[start_pos, start_pos+n)` stays
    /// within one batch. Default implementation calls commit() per position.
    fn commit_batch(&self, start_pos: usize, n: usize) {
        for i in 0..n {
            self.commit(start_pos + i);
        }
    }

    /// Block until `count` items are ready, then select them.
    /// Returns None on shutdown.
    fn select(&self, count: usize) -> Option<SampleResult>;

    /// Non-blocking select. Returns None if fewer than `count` ready.
    fn try_select(&self, count: usize) -> Option<SampleResult>;

    /// Number of items currently committed but not yet sampled (commit − read).
    /// Relaxed — may be slightly stale; acceptable for metrics.
    fn queue_size(&self) -> usize;

    /// Cancel any blocked select() calls.
    fn shutdown(&self);
}

/// FIFO sampler. Drainers `fetch_add` a per-batch counter after memcpy; the
/// drainer that brings the counter to `batch_size` wakes the consumer. One
/// wake per batch, no scanning. Counters indexed by `(pos / batch_size) %
/// num_buffers`; the consumer resets to 0 after sampling. The Store's
/// write_cursor backpressure ensures no drainer touches a batch counter the
/// consumer is currently using.
pub struct FifoSampler {
    /// Per-batch commit count. Release on increment publishes the memcpy;
    /// Acquire on load synchronises. Cache-padded to avoid false sharing
    /// between drainers committing to different batches.
    batch_counts: Box<[CachePadded<AtomicUsize>]>,
    /// Next absolute position the consumer will read from.
    read_cursor: AtomicUsize,
    batch_size: usize,
    num_buffers: usize,
    /// Consumer blocks here until a batch is ready.
    mu: Mutex<()>,
    cv: Condvar,
    shutdown: AtomicBool,
    /// Optional metrics sink. When Some, commit/select record timing + notify
    /// counters. Tests that don't care about metrics pass None.
    metrics: Option<Arc<Metrics>>,
}

impl FifoSampler {
    pub fn new(batch_size: usize, capacity: usize) -> Self {
        Self::with_metrics(batch_size, capacity, None)
    }

    pub fn with_metrics(batch_size: usize, capacity: usize, metrics: Option<Arc<Metrics>>) -> Self {
        assert!(
            capacity.is_multiple_of(batch_size),
            "capacity must be a multiple of batch_size",
        );
        let num_buffers = capacity / batch_size;
        let batch_counts = (0..num_buffers)
            .map(|_| CachePadded::new(AtomicUsize::new(0)))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            batch_counts,
            read_cursor: AtomicUsize::new(0),
            batch_size,
            num_buffers,
            mu: Mutex::new(()),
            cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
            metrics,
        }
    }

    fn capacity(&self) -> usize {
        self.batch_size * self.num_buffers
    }

    fn on_notify(&self) {
        if let Some(m) = &self.metrics {
            m.record_notify();
        }
    }
}

impl Sampler for FifoSampler {
    fn commit(&self, pos: usize) {
        let batch_id = (pos / self.batch_size) % self.num_buffers;
        // AcqRel pairs with the consumer's Acquire load in `select`.
        let prev = self.batch_counts[batch_id].fetch_add(1, Ordering::AcqRel);
        if prev + 1 == self.batch_size {
            self.on_notify();
            let _lock = self.mu.lock();
            self.cv.notify_one();
        }
    }

    fn commit_batch(&self, start_pos: usize, n: usize) {
        // Caller guarantees [start_pos, start_pos+n) stays within one batch.
        debug_assert_eq!(
            start_pos / self.batch_size,
            (start_pos + n - 1) / self.batch_size,
            "commit_batch must not cross a batch boundary",
        );
        let batch_id = (start_pos / self.batch_size) % self.num_buffers;
        let prev = self.batch_counts[batch_id].fetch_add(n, Ordering::AcqRel);
        if prev < self.batch_size && prev + n >= self.batch_size {
            self.on_notify();
            let _lock = self.mu.lock();
            self.cv.notify_one();
        }
    }

    fn select(&self, count: usize) -> Option<SampleResult> {
        debug_assert_eq!(count, self.batch_size, "only batch_size sampling supported");
        let mut lock = self.mu.lock();
        loop {
            let read = self.read_cursor.load(Ordering::Acquire);
            let batch_id = (read / self.batch_size) % self.num_buffers;
            if self.batch_counts[batch_id].load(Ordering::Acquire) >= self.batch_size {
                // Backpressure on the Store's write_cursor keeps drainers from
                // touching this batch's counter until the consumer releases it.
                self.batch_counts[batch_id].store(0, Ordering::Release);
                break;
            }
            if self.shutdown.load(Ordering::Acquire) {
                return None;
            }
            self.cv.wait(&mut lock);
        }
        drop(lock);

        let start = self.read_cursor.fetch_add(count, Ordering::AcqRel) % self.capacity();
        Some(SampleResult::Contiguous { start, count })
    }

    fn try_select(&self, count: usize) -> Option<SampleResult> {
        debug_assert_eq!(count, self.batch_size, "only batch_size sampling supported");
        let read = self.read_cursor.load(Ordering::Acquire);
        let batch_id = (read / self.batch_size) % self.num_buffers;
        if self.batch_counts[batch_id].load(Ordering::Acquire) < self.batch_size {
            return None;
        }
        self.batch_counts[batch_id].store(0, Ordering::Release);
        let start = self.read_cursor.fetch_add(count, Ordering::AcqRel) % self.capacity();
        Some(SampleResult::Contiguous { start, count })
    }

    fn queue_size(&self) -> usize {
        // Approximate: sum of all batch counts. May slightly overcount during
        // reset transitions but fine for metrics.
        self.batch_counts
            .iter()
            .map(|c| c.load(Ordering::Relaxed))
            .sum()
    }

    fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        let _lock = self.mu.lock();
        self.cv.notify_all();
    }
}

/// Tracks which slots the consumer is done with.
pub trait Remover: Send + Sync {
    /// Advance the free cursor, releasing slots for writers.
    fn remove(&self, count: usize);

    /// Current free position (used by Store for backpressure).
    fn read_pos(&self) -> usize;
}

/// FIFO remover. Cursor that advances by batch_size on each release.
/// Cache-padded: hot read on the drainer side, cold write on the consumer side.
pub struct FifoRemover {
    read_cursor: CachePadded<AtomicUsize>,
}

impl FifoRemover {
    pub fn new() -> Self {
        Self {
            read_cursor: CachePadded::new(AtomicUsize::new(0)),
        }
    }
}

impl Default for FifoRemover {
    fn default() -> Self {
        Self::new()
    }
}

impl Remover for FifoRemover {
    fn remove(&self, count: usize) {
        self.read_cursor.fetch_add(count, Ordering::Release);
    }

    fn read_pos(&self) -> usize {
        self.read_cursor.load(Ordering::Acquire)
    }
}
