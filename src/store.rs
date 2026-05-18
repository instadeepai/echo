use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "detailed-metrics")]
use std::time::Instant;

use crossbeam_utils::CachePadded;
use tokio::sync::Notify;

use crate::array_spec::ArraySpec;
use crate::metrics::DrainerMetrics;
use crate::ring_buf::PytreeRingBuf;
use crate::selector::{Remover, SampleResult, Sampler};

/// Zero-copy view into a batch of samples.
/// Each entry in `arrays` is (raw_address, total_bytes) for one array across the batch.
pub struct ConsumerView {
    pub arrays: Vec<(usize, usize)>,
    pub count: usize,
}

pub struct Store {
    ring: PytreeRingBuf,
    sampler: Box<dyn Sampler>,
    remover: Box<dyn Remover>,
    specs: Vec<ArraySpec>,
    batch_size: usize,
    capacity: usize,

    /// Slot assignment cursor. Writers CAS this to claim a slot.
    /// Cache-padded to isolate from the adjacent Notify's state word.
    write_cursor: CachePadded<AtomicUsize>,
    /// Wakes async inserters when space becomes available.
    space_available: Notify,
    shutdown: AtomicBool,
    /// Only accessed by the single consumer thread.
    has_previous_batch: Cell<bool>,
}

impl Store {
    pub fn new(
        specs: Vec<ArraySpec>,
        batch_size: usize,
        num_batches: usize,
        sampler: Box<dyn Sampler>,
        remover: Box<dyn Remover>,
    ) -> Self {
        assert!(num_batches >= 2, "num_batches must be >= 2");

        let capacity = batch_size * num_batches;
        let slot_bytes: Vec<usize> = specs.iter().map(|s| s.num_bytes()).collect();
        let ring = PytreeRingBuf::new(slot_bytes, capacity, batch_size);

        Self {
            ring,
            sampler,
            remover,
            specs,
            batch_size,
            capacity,
            write_cursor: CachePadded::new(AtomicUsize::new(0)),
            space_available: Notify::new(),
            shutdown: AtomicBool::new(false),
            has_previous_batch: Cell::new(false),
        }
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn specs(&self) -> &[ArraySpec] {
        &self.specs
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Items currently committed but not yet sampled. Relaxed; metric-only.
    pub fn size(&self) -> usize {
        self.sampler.queue_size()
    }

    /// Try to claim a slot and write data. Returns true on success,
    /// false if the ring is full. CAS-loops on contention.
    fn write_slot(&self, arrays: &[&[u8]], metrics: Option<&DrainerMetrics>) -> bool {
        loop {
            let write = self.write_cursor.load(Ordering::Relaxed);
            let freed = self.remover.read_pos();
            if write.wrapping_sub(freed) >= self.capacity {
                return false;
            }
            if self
                .write_cursor
                .compare_exchange_weak(write, write + 1, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                if let Some(m) = metrics {
                    m.record_cas_success();
                }
                let slot = write % self.capacity;

                // Memcpy from temp buffer into ring slot.
                #[cfg(feature = "detailed-metrics")]
                let memcpy_start = metrics.map(|_| Instant::now());
                for (i, &data) in arrays.iter().enumerate() {
                    debug_assert_eq!(data.len(), self.specs[i].num_bytes());
                    unsafe {
                        let dst = self.ring.slot_mut(slot, i);
                        std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
                    }
                }
                if let Some(m) = metrics {
                    #[cfg(feature = "detailed-metrics")]
                    if let Some(t) = memcpy_start {
                        m.record_memcpy_ns(t.elapsed().as_nanos() as u64);
                    }
                    m.record_samples_inserted(1);
                }

                self.sampler.commit(write);
                return true;
            }
            if let Some(m) = metrics {
                m.record_cas_failure();
            }
        }
    }

    /// Async insert. Awaits on Notify when the ring is full.
    pub async fn insert(&self, arrays: &[&[u8]]) {
        loop {
            let notified = self.space_available.notified();
            if self.write_slot(arrays, None) {
                return;
            }
            notified.await;
        }
    }

    /// Synchronous insert. Spins when the ring is full.
    pub fn insert_sync(&self, arrays: &[&[u8]]) {
        while !self.write_slot(arrays, None) {
            std::hint::spin_loop();
        }
    }

    /// Reserve up to `want` contiguous slots in one CAS. Capped at the
    /// batch boundary so commit_batch can do a single fetch_add.
    fn try_reserve_slots(
        &self,
        want: usize,
        metrics: Option<&DrainerMetrics>,
    ) -> Option<(usize, usize)> {
        let mut retries = 0u64;
        loop {
            let write = self.write_cursor.load(Ordering::Relaxed);
            let freed = self.remover.read_pos();
            let used = write.wrapping_sub(freed);
            if used >= self.capacity {
                if let Some(m) = metrics {
                    m.record_reservation_retries(retries);
                }
                return None;
            }
            let available = self.capacity - used;
            let batch_offset = write % self.batch_size;
            let in_batch_remain = self.batch_size - batch_offset;
            let n = want.min(available).min(in_batch_remain);
            if n == 0 {
                if let Some(m) = metrics {
                    m.record_reservation_retries(retries);
                }
                return None;
            }
            if self
                .write_cursor
                .compare_exchange_weak(write, write + n, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                if let Some(m) = metrics {
                    m.record_cas_success();
                    m.record_reservation_retries(retries);
                }
                return Some((write, n));
            }
            if let Some(m) = metrics {
                m.record_cas_failure();
            }
            retries += 1;
        }
    }

    /// Async batch insert. Reserves up to batch_size slots per CAS and does
    /// one commit_batch fetch_add per reservation. Pass `metrics` from a
    /// drainer to record counters on its per-drainer cache line; pass None
    /// for in-process `submit`.
    pub async fn insert_batch(
        &self,
        samples: &[&[u8]],
        array_sizes: &[usize],
        metrics: Option<&DrainerMetrics>,
    ) {
        let mut done = 0;
        while done < samples.len() {
            let notified = self.space_available.notified();
            let (start, n) = match self.try_reserve_slots(samples.len() - done, metrics) {
                Some(x) => x,
                None => {
                    notified.await;
                    continue;
                }
            };
            #[cfg(feature = "detailed-metrics")]
            let memcpy_start = metrics.map(|_| Instant::now());
            for i in 0..n {
                let pos = start + i;
                let slot = pos % self.capacity;
                let sample = samples[done + i];
                let mut offset = 0;
                for (j, &size) in array_sizes.iter().enumerate() {
                    unsafe {
                        let dst = self.ring.slot_mut(slot, j);
                        std::ptr::copy_nonoverlapping(sample.as_ptr().add(offset), dst, size);
                    }
                    offset += size;
                }
            }
            if let Some(m) = metrics {
                #[cfg(feature = "detailed-metrics")]
                if let Some(t) = memcpy_start {
                    // Single record per reservation; dividing by n gives per-sample.
                    m.record_memcpy_ns(t.elapsed().as_nanos() as u64);
                }
                m.record_samples_inserted(n as u64);
            }
            // Single fetch_add for all n commits (same batch by construction).
            self.sampler.commit_batch(start, n);
            done += n;
        }
    }

    /// Release the previous batch's slots, making space for writers.
    fn release_previous_batch(&self) {
        if !self.has_previous_batch.get() {
            return;
        }
        self.has_previous_batch.set(false);

        let was_full = {
            let write = self.write_cursor.load(Ordering::Acquire);
            let freed = self.remover.read_pos();
            write.wrapping_sub(freed) >= self.capacity
        };

        self.remover.remove(self.batch_size);

        if was_full {
            // notify_waiters stores no permit and only wakes already-registered
            // waiters. Safe here because writers call `notified()` *before*
            // each try_reserve_slots — tokio's Notified future captures the
            // current generation at construction, so a notify_waiters that
            // fires between notified() and .await still wakes that await.
            self.space_available.notify_waiters();
        }
    }

    /// Build a ConsumerView from a contiguous range.
    fn build_view(&self, start: usize, count: usize) -> ConsumerView {
        let arrays = (0..self.ring.num_arrays())
            .map(|i| self.ring.range_ptr(i, start, count))
            .collect();
        ConsumerView { arrays, count }
    }

    /// Block until a batch is ready. Returns None on shutdown.
    /// Releases the previous batch's slots on entry.
    pub fn sample(&self) -> Option<ConsumerView> {
        self.release_previous_batch();

        let result = self.sampler.select(self.batch_size)?;
        let SampleResult::Contiguous { start, count } = result;

        self.has_previous_batch.set(true);
        Some(self.build_view(start, count))
    }

    /// Non-blocking sample. Returns None if not enough data.
    pub fn try_sample(&self) -> Option<ConsumerView> {
        self.release_previous_batch();

        let result = self.sampler.try_select(self.batch_size)?;
        let SampleResult::Contiguous { start, count } = result;

        self.has_previous_batch.set(true);
        Some(self.build_view(start, count))
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        self.sampler.shutdown();
        self.space_available.notify_waiters();
    }

    pub fn is_shutdown(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }
}

// Safety:
// - PytreeRingBuf uses UnsafeCell but Store guarantees disjoint slot access.
// - Cell<bool> (has_previous_batch) is only accessed by the single consumer thread.
// - All other shared state uses atomics or parking_lot synchronization.
unsafe impl Send for Store {}
unsafe impl Sync for Store {}
