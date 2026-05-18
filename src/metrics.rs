//! Runtime health + diagnostic metrics.
//!
//! Default build exposes cheap counters / gauges used for ongoing health
//! monitoring. Building with `--features detailed-metrics` additionally
//! enables per-op CAS counters and three timing histograms (memcpy,
//! drain_round, queue_dwell) for diagnosis.
//!
//! Call-site convention: hot paths never touch atomics or hdrhistogram
//! directly — they go through `Metrics::record_*` and
//! `DrainerMetrics::record_*`. Feature-gated metrics are no-ops when the
//! feature is off so call sites stay free of `#[cfg]`.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crossbeam_utils::CachePadded;

#[cfg(feature = "detailed-metrics")]
use hdrhistogram::Histogram;
#[cfg(feature = "detailed-metrics")]
use parking_lot::Mutex;

// ---------------------------------------------------------------------------
// Histogram config + snapshot shape (shape always present, values only
// populated when the feature is on).
// ---------------------------------------------------------------------------

#[cfg(feature = "detailed-metrics")]
const HIST_LOW_NS: u64 = 1;
#[cfg(feature = "detailed-metrics")]
const HIST_HIGH_NS: u64 = 10_000_000_000;
#[cfg(feature = "detailed-metrics")]
const HIST_SIGFIG: u8 = 3;

#[cfg(feature = "detailed-metrics")]
fn new_timing_hist() -> Histogram<u64> {
    Histogram::new_with_bounds(HIST_LOW_NS, HIST_HIGH_NS, HIST_SIGFIG)
        .expect("valid histogram bounds")
}

/// Snapshot of a timing distribution. Always exported; values are zero when
/// `detailed-metrics` is disabled.
#[derive(Clone, Copy, Debug, Default)]
pub struct HistSnapshot {
    pub count: u64,
    pub min_ns: u64,
    pub max_ns: u64,
    pub mean_ns: f64,
    pub p50_ns: u64,
    pub p90_ns: u64,
    pub p99_ns: u64,
}

#[cfg(feature = "detailed-metrics")]
impl HistSnapshot {
    fn from_hist(h: &Histogram<u64>) -> Self {
        Self {
            count: h.len(),
            min_ns: h.min(),
            max_ns: h.max(),
            mean_ns: h.mean(),
            p50_ns: h.value_at_quantile(0.50),
            p90_ns: h.value_at_quantile(0.90),
            p99_ns: h.value_at_quantile(0.99),
        }
    }
}

// ---------------------------------------------------------------------------
// DrainerMetrics — per-drainer, cache-padded in `Metrics.drainers`.
// ---------------------------------------------------------------------------

/// Metrics owned by one drainer. Cache-padded in [`Metrics::drainers`] so
/// concurrent writes from different drainers don't share a cache line.
pub struct DrainerMetrics {
    queue_depth_sum: AtomicUsize,
    queue_depth_max: AtomicUsize,
    samples_inserted: AtomicU64,

    #[cfg(feature = "detailed-metrics")]
    cas_success: AtomicU64,
    #[cfg(feature = "detailed-metrics")]
    cas_failure: AtomicU64,
    #[cfg(feature = "detailed-metrics")]
    reservation_retries: AtomicU64,

    #[cfg(feature = "detailed-metrics")]
    memcpy_ns_hist: Mutex<Histogram<u64>>,
    #[cfg(feature = "detailed-metrics")]
    drain_round_ns_hist: Mutex<Histogram<u64>>,
    #[cfg(feature = "detailed-metrics")]
    queue_dwell_ns_hist: Mutex<Histogram<u64>>,
}

impl DrainerMetrics {
    fn new() -> Self {
        Self {
            queue_depth_sum: AtomicUsize::new(0),
            queue_depth_max: AtomicUsize::new(0),
            samples_inserted: AtomicU64::new(0),
            #[cfg(feature = "detailed-metrics")]
            cas_success: AtomicU64::new(0),
            #[cfg(feature = "detailed-metrics")]
            cas_failure: AtomicU64::new(0),
            #[cfg(feature = "detailed-metrics")]
            reservation_retries: AtomicU64::new(0),
            #[cfg(feature = "detailed-metrics")]
            memcpy_ns_hist: Mutex::new(new_timing_hist()),
            #[cfg(feature = "detailed-metrics")]
            drain_round_ns_hist: Mutex::new(new_timing_hist()),
            #[cfg(feature = "detailed-metrics")]
            queue_dwell_ns_hist: Mutex::new(new_timing_hist()),
        }
    }

    // --- always on -----------------------------------------------------

    /// Record a batch of `n` samples inserted into the store by this drainer.
    pub fn record_samples_inserted(&self, n: u64) {
        self.samples_inserted.fetch_add(n, Ordering::Relaxed);
    }

    /// Publish the pre-drain sum/max queue depths observed at the start of a
    /// drain round. Stored (not accumulated) — latest round wins.
    pub fn record_queue_depth(&self, sum: usize, max: usize) {
        self.queue_depth_sum.store(sum, Ordering::Relaxed);
        self.queue_depth_max.store(max, Ordering::Relaxed);
    }

    pub fn samples_inserted(&self) -> u64 {
        self.samples_inserted.load(Ordering::Relaxed)
    }

    pub fn queue_depth_sum(&self) -> usize {
        self.queue_depth_sum.load(Ordering::Relaxed)
    }

    pub fn queue_depth_max(&self) -> usize {
        self.queue_depth_max.load(Ordering::Relaxed)
    }

    // --- detailed-only -------------------------------------------------
    //
    // When the feature is off these are no-op inlines so the call site
    // doesn't need `#[cfg]`.

    #[cfg(feature = "detailed-metrics")]
    #[inline]
    pub fn record_cas_success(&self) {
        self.cas_success.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(not(feature = "detailed-metrics"))]
    #[inline(always)]
    pub fn record_cas_success(&self) {}

    #[cfg(feature = "detailed-metrics")]
    #[inline]
    pub fn record_cas_failure(&self) {
        self.cas_failure.fetch_add(1, Ordering::Relaxed);
    }
    #[cfg(not(feature = "detailed-metrics"))]
    #[inline(always)]
    pub fn record_cas_failure(&self) {}

    #[cfg(feature = "detailed-metrics")]
    #[inline]
    pub fn record_reservation_retries(&self, n: u64) {
        self.reservation_retries.fetch_add(n, Ordering::Relaxed);
    }
    #[cfg(not(feature = "detailed-metrics"))]
    #[inline(always)]
    pub fn record_reservation_retries(&self, _n: u64) {}

    #[cfg(feature = "detailed-metrics")]
    #[inline]
    pub fn record_memcpy_ns(&self, ns: u64) {
        let _ = self.memcpy_ns_hist.lock().record(ns.max(HIST_LOW_NS));
    }
    #[cfg(not(feature = "detailed-metrics"))]
    #[inline(always)]
    pub fn record_memcpy_ns(&self, _ns: u64) {}

    #[cfg(feature = "detailed-metrics")]
    #[inline]
    pub fn record_drain_round_ns(&self, ns: u64) {
        let _ = self.drain_round_ns_hist.lock().record(ns.max(HIST_LOW_NS));
    }
    #[cfg(not(feature = "detailed-metrics"))]
    #[inline(always)]
    pub fn record_drain_round_ns(&self, _ns: u64) {}

    #[cfg(feature = "detailed-metrics")]
    #[inline]
    pub fn record_queue_dwell_ns(&self, ns: u64) {
        let _ = self.queue_dwell_ns_hist.lock().record(ns.max(HIST_LOW_NS));
    }
    #[cfg(not(feature = "detailed-metrics"))]
    #[inline(always)]
    pub fn record_queue_dwell_ns(&self, _ns: u64) {}

    // --- detailed-only -------------------------------------------------
    // (existing record_*_ns methods stay above this)

    /// Resets all three timing histograms (counts and min/max watermarks)
    /// for this drainer. Concurrent `record_*_ns` calls serialize on the
    /// same `Mutex` so they don't race; values recorded after this returns
    /// land in the new window.
    #[cfg(feature = "detailed-metrics")]
    pub fn reset_histograms(&self) {
        self.memcpy_ns_hist.lock().reset();
        self.drain_round_ns_hist.lock().reset();
        self.queue_dwell_ns_hist.lock().reset();
    }
    #[cfg(not(feature = "detailed-metrics"))]
    #[inline(always)]
    pub fn reset_histograms(&self) {}
}

// ---------------------------------------------------------------------------
// Metrics — global (always on) counters + accessor methods.
// ---------------------------------------------------------------------------

pub struct Metrics {
    active_connections: AtomicUsize,
    /// Cache-padded so the producer-side increment (every push that hits a
    /// full SPSC) doesn't false-share with `notify_count` on the drainer
    /// side, and vice versa.
    push_blocked_count: CachePadded<AtomicU64>,
    notify_count: CachePadded<AtomicU64>,
    drainers: Vec<CachePadded<DrainerMetrics>>,
}

impl Metrics {
    pub fn new(num_drainers: usize) -> Arc<Self> {
        let drainers = (0..num_drainers)
            .map(|_| CachePadded::new(DrainerMetrics::new()))
            .collect();
        Arc::new(Self {
            active_connections: AtomicUsize::new(0),
            push_blocked_count: CachePadded::new(AtomicU64::new(0)),
            notify_count: CachePadded::new(AtomicU64::new(0)),
            drainers,
        })
    }

    pub fn num_drainers(&self) -> usize {
        self.drainers.len()
    }

    pub fn drainer(&self, idx: usize) -> &DrainerMetrics {
        &self.drainers[idx]
    }

    // --- record (always on) --------------------------------------------

    pub fn record_connection_opened(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_connection_closed(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    pub fn record_push_blocked(&self) {
        self.push_blocked_count.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_notify(&self) {
        self.notify_count.fetch_add(1, Ordering::Relaxed);
    }

    // --- read ----------------------------------------------------------

    pub fn active_connections(&self) -> usize {
        self.active_connections.load(Ordering::Relaxed)
    }

    pub fn push_blocked_count(&self) -> u64 {
        self.push_blocked_count.load(Ordering::Relaxed)
    }

    pub fn notify_count(&self) -> u64 {
        self.notify_count.load(Ordering::Relaxed)
    }

    pub fn samples_inserted_total(&self) -> u64 {
        self.drainers.iter().map(|d| d.samples_inserted()).sum()
    }

    pub fn queue_depth_sum(&self) -> usize {
        self.drainers.iter().map(|d| d.queue_depth_sum()).sum()
    }

    pub fn queue_depth_max(&self) -> usize {
        self.drainers
            .iter()
            .map(|d| d.queue_depth_max())
            .max()
            .unwrap_or(0)
    }

    // --- read (detailed; always callable, zero when feature off) -------

    pub fn cas_success_total(&self) -> u64 {
        #[cfg(feature = "detailed-metrics")]
        {
            self.drainers
                .iter()
                .map(|d| d.cas_success.load(Ordering::Relaxed))
                .sum()
        }
        #[cfg(not(feature = "detailed-metrics"))]
        {
            0
        }
    }

    pub fn cas_failure_total(&self) -> u64 {
        #[cfg(feature = "detailed-metrics")]
        {
            self.drainers
                .iter()
                .map(|d| d.cas_failure.load(Ordering::Relaxed))
                .sum()
        }
        #[cfg(not(feature = "detailed-metrics"))]
        {
            0
        }
    }

    pub fn reservation_retries_total(&self) -> u64 {
        #[cfg(feature = "detailed-metrics")]
        {
            self.drainers
                .iter()
                .map(|d| d.reservation_retries.load(Ordering::Relaxed))
                .sum()
        }
        #[cfg(not(feature = "detailed-metrics"))]
        {
            0
        }
    }

    pub fn memcpy_aggregate(&self) -> HistSnapshot {
        #[cfg(feature = "detailed-metrics")]
        {
            self.aggregate_hist(|d| &d.memcpy_ns_hist)
        }
        #[cfg(not(feature = "detailed-metrics"))]
        {
            HistSnapshot::default()
        }
    }

    pub fn drain_round_aggregate(&self) -> HistSnapshot {
        #[cfg(feature = "detailed-metrics")]
        {
            self.aggregate_hist(|d| &d.drain_round_ns_hist)
        }
        #[cfg(not(feature = "detailed-metrics"))]
        {
            HistSnapshot::default()
        }
    }

    pub fn queue_dwell_aggregate(&self) -> HistSnapshot {
        #[cfg(feature = "detailed-metrics")]
        {
            self.aggregate_hist(|d| &d.queue_dwell_ns_hist)
        }
        #[cfg(not(feature = "detailed-metrics"))]
        {
            HistSnapshot::default()
        }
    }

    /// Reset all per-drainer timing histograms. Use at the boundary of a
    /// metrics window (typically the end of a logging interval) so the next
    /// `*_aggregate` snapshot reflects only events recorded after this call.
    /// Counters and gauges are not affected.
    pub fn reset_histograms(&self) {
        for d in &self.drainers {
            d.reset_histograms();
        }
    }

    #[cfg(feature = "detailed-metrics")]
    fn aggregate_hist<F>(&self, get: F) -> HistSnapshot
    where
        F: Fn(&DrainerMetrics) -> &Mutex<Histogram<u64>>,
    {
        let mut merged = new_timing_hist();
        for d in &self.drainers {
            let h = get(d);
            let g = h.lock();
            let _ = merged.add(&*g);
        }
        HistSnapshot::from_hist(&merged)
    }
}
