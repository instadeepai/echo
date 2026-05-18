//! Per-connection SPSC queues + drainer pool.
//!
//! Each transport connection writes samples into its own SPSC queue
//! (single producer, single consumer). A pool of N drainer tasks each
//! own a subset of these queues and feed samples into the Store.
//! This reduces contention from 100+ connections down to N drainers.
//!
//! ```text
//! Conn 0 ──> SPSC 0 ──┐
//! Conn 1 ──> SPSC 1 ──┼── Drainer 0 ──┐
//! Conn 2 ──> SPSC 2 ──┘               │
//! Conn 3 ──> SPSC 3 ──┐               ├── store.insert()
//! Conn 4 ──> SPSC 4 ──┼── Drainer 1 ──┘
//! Conn 5 ──> SPSC 5 ──┘
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
#[cfg(feature = "detailed-metrics")]
use std::time::Instant;

use crossbeam::queue::ArrayQueue;
use tokio::sync::{mpsc, Notify};

use crate::metrics::Metrics;
use crate::store::Store;

/// Raw sample bytes (concatenated arrays in pytree order).
type Sample = Vec<u8>;

/// One entry in a per-connection SPSC queue. Its `Drop` returns `data` to
/// the connection's free pool so buffers cycle without per-message alloc.
pub struct TransportQueueItem {
    #[cfg(feature = "detailed-metrics")]
    pub pushed_at: Instant,
    pub data: Sample,
    free_pool: Arc<ArrayQueue<Vec<u8>>>,
}

impl TransportQueueItem {
    pub fn new(data: Sample, free_pool: Arc<ArrayQueue<Vec<u8>>>) -> Self {
        Self {
            #[cfg(feature = "detailed-metrics")]
            pushed_at: Instant::now(),
            data,
            free_pool,
        }
    }
}

impl Drop for TransportQueueItem {
    fn drop(&mut self) {
        // Preserve len: bytes are stale but `read_exact` on the next `acquire`
        // overwrites them, which avoids a resize/zero-fill on acquisition.
        let buf = std::mem::take(&mut self.data);
        let _ = self.free_pool.push(buf);
    }
}

/// Producer side of a per-connection SPSC queue.
///
/// Given to transport connection handlers. Each connection gets its own
/// sender (and thus its own queue) so there is zero contention between
/// connections on the producer side.
pub struct SampleSender {
    queue: Arc<ArrayQueue<TransportQueueItem>>,
    space_available: Arc<Notify>,
    drainer_wake: Arc<Notify>,
    metrics: Arc<Metrics>,
    free_pool: Arc<ArrayQueue<Vec<u8>>>,
    closed: Arc<AtomicBool>,
}

impl SampleSender {
    pub fn acquire(&self, payload_size: usize) -> Vec<u8> {
        self.free_pool
            .pop()
            .unwrap_or_else(|| vec![0u8; payload_size])
    }

    pub async fn push(&self, data: Sample) {
        let mut entry = TransportQueueItem::new(data, self.free_pool.clone());
        loop {
            // Register interest before the push so a notify_one racing with us
            // between a failed push and the await lands as a permit.
            let notified = self.space_available.notified();
            match self.queue.push(entry) {
                Ok(()) => {
                    self.drainer_wake.notify_one();
                    return;
                }
                Err(rejected) => {
                    self.metrics.record_push_blocked();
                    entry = rejected;
                    notified.await;
                }
            }
        }
    }
}

impl Drop for SampleSender {
    fn drop(&mut self) {
        self.closed.store(true, Ordering::Release);
        self.metrics.record_connection_closed();
    }
}

/// Drainer-side handle for one transport connection. The matching
/// `SampleSender` sets `closed` in its `Drop` so the drainer can prune.
pub struct TransportHandle {
    pub queue: Arc<ArrayQueue<TransportQueueItem>>,
    pub space_available: Arc<Notify>,
    pub closed: Arc<AtomicBool>,
}

/// Pool of drainer tasks that read from per-connection SPSC queues
/// and insert into the Store.
pub struct DrainerPool {
    _runtime: tokio::runtime::Runtime,
    /// Per-drainer channel used to hand off newly-accepted transports to the
    /// drainer that owns them.
    transport_txs: Vec<mpsc::UnboundedSender<TransportHandle>>,
    wakes: Vec<Arc<Notify>>,
    next_drainer: AtomicUsize,
    shutdown: Arc<AtomicBool>,
    producer_queue_size: usize,
    metrics: Arc<Metrics>,
}

impl DrainerPool {
    pub fn new(
        store: Arc<Store>,
        num_drainers: usize,
        producer_queue_size: usize,
        metrics: Arc<Metrics>,
    ) -> Self {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(num_drainers)
            .max_blocking_threads(num_drainers * 2)
            .enable_all()
            .thread_name("drainer")
            .build()
            .expect("failed to build drainer runtime");

        let shutdown = Arc::new(AtomicBool::new(false));
        let mut transport_txs = Vec::with_capacity(num_drainers);
        let mut wakes = Vec::with_capacity(num_drainers);

        for drainer_idx in 0..num_drainers {
            // Channel used to hand newly-accepted transports to this drainer.
            let (transport_tx, transport_rx) = mpsc::unbounded_channel();
            let wake = Arc::new(Notify::new());

            let store = store.clone();
            let wake_clone = wake.clone();
            let shutdown_clone = shutdown.clone();
            let metrics_clone = metrics.clone();

            runtime.spawn(drainer_task(
                store,
                transport_rx,
                wake_clone,
                shutdown_clone,
                metrics_clone,
                drainer_idx,
            ));

            transport_txs.push(transport_tx);
            wakes.push(wake);
        }

        Self {
            _runtime: runtime,
            transport_txs,
            wakes,
            next_drainer: AtomicUsize::new(0),
            shutdown,
            producer_queue_size,
            metrics,
        }
    }

    pub fn metrics(&self) -> &Arc<Metrics> {
        &self.metrics
    }

    /// Create a new per-connection SPSC sender, assigned to a drainer via round-robin.
    pub fn new_sender(&self) -> SampleSender {
        let idx = self.next_drainer.fetch_add(1, Ordering::Relaxed) % self.transport_txs.len();
        let queue: Arc<ArrayQueue<TransportQueueItem>> =
            Arc::new(ArrayQueue::new(self.producer_queue_size));
        // 2Q+1: Q in the SPSC queue, Q held by the drainer during insert_batch,
        // 1 the producer is filling.
        let free_pool: Arc<ArrayQueue<Vec<u8>>> =
            Arc::new(ArrayQueue::new(2 * self.producer_queue_size + 1));
        let space_available = Arc::new(Notify::new());
        let closed = Arc::new(AtomicBool::new(false));
        let transport = TransportHandle {
            queue: queue.clone(),
            space_available: space_available.clone(),
            closed: closed.clone(),
        };
        let _ = self.transport_txs[idx].send(transport);
        self.wakes[idx].notify_one();

        self.metrics.record_connection_opened();

        SampleSender {
            queue,
            space_available,
            drainer_wake: self.wakes[idx].clone(),
            metrics: self.metrics.clone(),
            free_pool,
            closed,
        }
    }

    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
        for wake in &self.wakes {
            wake.notify_waiters();
        }
    }
}

/// Drain every transport queue once and hand the popped batch to the store.
/// Returns whether any work was done. Public so tests can drive it directly.
pub async fn drain_round(
    transports: &[TransportHandle],
    store: &Arc<Store>,
    array_sizes: &[usize],
    metrics: &crate::metrics::DrainerMetrics,
    round_counter: usize,
) -> bool {
    #[cfg(feature = "detailed-metrics")]
    let round_start = Instant::now();

    let mut round_sum = 0usize;
    let mut round_max = 0usize;

    // Rotate iteration order by round_counter so no single queue is always
    // first and no single one is always last.
    let n_queues = transports.len();
    let offset = if n_queues == 0 {
        0
    } else {
        round_counter % n_queues
    };

    // Pop everything currently in each queue into one round-batch, capped at
    // the per-queue snapshot length so no queue can starve the others.
    let mut batch: Vec<TransportQueueItem> = Vec::new();
    for step in 0..n_queues {
        let i = (step + offset) % n_queues;
        let transport = &transports[i];
        let n = transport.queue.len();
        round_sum += n;
        if n > round_max {
            round_max = n;
        }
        let before = batch.len();
        for _ in 0..n {
            match transport.queue.pop() {
                Some(s) => batch.push(s),
                None => break,
            }
        }
        if batch.len() > before {
            transport.space_available.notify_one();
        }
    }

    metrics.record_queue_depth(round_sum, round_max);

    if batch.is_empty() {
        return false;
    }

    // One aggregated insert_batch for everything popped this round.
    #[cfg(feature = "detailed-metrics")]
    {
        let now = Instant::now();
        for entry in &batch {
            let dwell = now.saturating_duration_since(entry.pushed_at).as_nanos() as u64;
            metrics.record_queue_dwell_ns(dwell);
        }
    }
    let slices: Vec<&[u8]> = batch.iter().map(|e| e.data.as_slice()).collect();
    store
        .insert_batch(&slices, array_sizes, Some(metrics))
        .await;

    #[cfg(feature = "detailed-metrics")]
    metrics.record_drain_round_ns(round_start.elapsed().as_nanos() as u64);

    true
}

async fn drainer_task(
    store: Arc<Store>,
    mut transport_rx: mpsc::UnboundedReceiver<TransportHandle>,
    wake: Arc<Notify>,
    shutdown: Arc<AtomicBool>,
    metrics: Arc<Metrics>,
    drainer_idx: usize,
) {
    let array_sizes: Vec<usize> = store.specs().iter().map(|s| s.num_bytes()).collect();
    let mut transports: Vec<TransportHandle> = Vec::new();
    let mut round_counter: usize = 0;

    loop {
        let notified = wake.notified();

        while let Ok(transport) = transport_rx.try_recv() {
            transports.push(transport);
        }

        if shutdown.load(Ordering::Relaxed) {
            break;
        }

        let did_work = drain_round(
            &transports,
            &store,
            &array_sizes,
            metrics.drainer(drainer_idx),
            round_counter,
        )
        .await;
        round_counter = round_counter.wrapping_add(1);

        // Prune connections whose SampleSender has been dropped (drained any
        // remaining items first via drain_round above).
        transports.retain(|t| !t.closed.load(Ordering::Acquire));

        if !did_work {
            notified.await;
        }
    }
}
