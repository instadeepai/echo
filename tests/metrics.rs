use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use crossbeam::queue::ArrayQueue;
use echo::array_spec::ArraySpec;
use echo::ingress::{drain_round, DrainerPool, TransportHandle, TransportQueueItem};
use echo::metrics::Metrics;
use echo::selector::{FifoRemover, FifoSampler};
use echo::store::Store;
use tokio::sync::Notify;

fn make_transport(capacity: usize) -> (TransportHandle, Arc<ArrayQueue<Vec<u8>>>) {
    let handle = TransportHandle {
        queue: Arc::new(ArrayQueue::new(capacity)),
        space_available: Arc::new(Notify::new()),
        closed: Arc::new(AtomicBool::new(false)),
    };
    let free_pool = Arc::new(ArrayQueue::new(capacity * 2 + 1));
    (handle, free_pool)
}

fn make_store_and_sizes() -> (Arc<Store>, Vec<usize>) {
    let specs = vec![ArraySpec::new(vec![2], 1)];
    let batch_size = 4;
    let num_batches = 4;
    let capacity = batch_size * num_batches;
    let store = Arc::new(Store::new(
        specs.clone(),
        batch_size,
        num_batches,
        Box::new(FifoSampler::new(batch_size, capacity)),
        Box::new(FifoRemover::new()),
    ));
    let array_sizes: Vec<usize> = specs.iter().map(|s| s.num_bytes()).collect();
    (store, array_sizes)
}

fn make_store() -> Arc<Store> {
    make_store_and_sizes().0
}

// ---------------------------------------------------------------------------
// Metrics smoke tests — only exercise the public API (no atomic poking).
// ---------------------------------------------------------------------------

#[test]
fn test_metrics_starts_zeroed() {
    let m = Metrics::new(4);
    assert_eq!(m.active_connections(), 0);
    assert_eq!(m.push_blocked_count(), 0);
    assert_eq!(m.notify_count(), 0);
    assert_eq!(m.queue_depth_sum(), 0);
    assert_eq!(m.queue_depth_max(), 0);
    assert_eq!(m.samples_inserted_total(), 0);
}

#[test]
fn test_metrics_queue_depth_sum_and_max() {
    let m = Metrics::new(3);
    m.drainer(0).record_queue_depth(10, 4);
    m.drainer(1).record_queue_depth(3, 11);
    m.drainer(2).record_queue_depth(7, 2);
    assert_eq!(m.queue_depth_sum(), 20);
    assert_eq!(m.queue_depth_max(), 11);
}

#[test]
fn test_metrics_active_connections_and_push_blocked() {
    let m = Metrics::new(1);
    m.record_connection_opened();
    m.record_connection_opened();
    m.record_connection_opened();
    m.record_connection_closed();
    assert_eq!(m.active_connections(), 2);

    for _ in 0..42 {
        m.record_push_blocked();
    }
    assert_eq!(m.push_blocked_count(), 42);
}

// ---------------------------------------------------------------------------
// drain_round — public behaviour.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_drain_round_publishes_pre_drain_sum_and_max() {
    let (store, array_sizes) = make_store_and_sizes();
    let metrics = Metrics::new(1);

    let (t1, p1) = make_transport(16);
    let (t2, p2) = make_transport(16);
    // t1 has 3 items, t2 has 5 → sum=8, max=5.
    for _ in 0..3 {
        t1.queue
            .push(TransportQueueItem::new(vec![0u8; 2], p1.clone()))
            .ok()
            .unwrap();
    }
    for _ in 0..5 {
        t2.queue
            .push(TransportQueueItem::new(vec![0u8; 2], p2.clone()))
            .ok()
            .unwrap();
    }

    let q1 = t1.queue.clone();
    let q2 = t2.queue.clone();
    let transports = vec![t1, t2];
    let did_work = drain_round(&transports, &store, &array_sizes, metrics.drainer(0), 0).await;

    assert!(did_work);
    assert_eq!(metrics.drainer(0).queue_depth_sum(), 8);
    assert_eq!(metrics.drainer(0).queue_depth_max(), 5);
    assert_eq!(q1.len(), 0, "q1 fully drained");
    assert_eq!(q2.len(), 0, "q2 fully drained");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_drain_round_empty_queues_zero_sum() {
    let (store, array_sizes) = make_store_and_sizes();
    let metrics = Metrics::new(1);

    let (t, _p) = make_transport(16);
    let transports = vec![t];
    let did_work = drain_round(&transports, &store, &array_sizes, metrics.drainer(0), 0).await;

    assert!(!did_work);
    assert_eq!(metrics.drainer(0).queue_depth_sum(), 0);
    assert_eq!(metrics.drainer(0).queue_depth_max(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_drain_round_records_samples_inserted() {
    let (store, array_sizes) = make_store_and_sizes();
    let metrics = Metrics::new(1);

    let (transport, pool) = make_transport(16);
    for _ in 0..4 {
        transport
            .queue
            .push(TransportQueueItem::new(vec![0u8; 2], pool.clone()))
            .ok()
            .unwrap();
    }
    let transports = vec![transport];

    let did_work = drain_round(&transports, &store, &array_sizes, metrics.drainer(0), 0).await;
    assert!(did_work);
    assert_eq!(metrics.drainer(0).samples_inserted(), 4);
    assert_eq!(metrics.samples_inserted_total(), 4);
}

#[cfg(feature = "detailed-metrics")]
#[tokio::test(flavor = "multi_thread")]
async fn test_drain_round_records_detailed_counters() {
    let (store, array_sizes) = make_store_and_sizes();
    let metrics = Metrics::new(1);

    let (transport, pool) = make_transport(16);
    for _ in 0..4 {
        transport
            .queue
            .push(TransportQueueItem::new(vec![0u8; 2], pool.clone()))
            .ok()
            .unwrap();
    }
    let transports = vec![transport];

    let _ = drain_round(&transports, &store, &array_sizes, metrics.drainer(0), 0).await;

    assert!(metrics.cas_success_total() >= 1);
    assert_eq!(metrics.memcpy_aggregate().count, 1);
    assert_eq!(metrics.queue_dwell_aggregate().count, 4);
    assert_eq!(metrics.drain_round_aggregate().count, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_drain_round_notifies_consumer_after_batch_fills() {
    let specs = vec![ArraySpec::new(vec![2], 1)];
    let batch_size = 4;
    let num_batches = 2;
    let capacity = batch_size * num_batches;
    let metrics = Metrics::new(1);
    let store = Arc::new(Store::new(
        specs.clone(),
        batch_size,
        num_batches,
        Box::new(FifoSampler::with_metrics(
            batch_size,
            capacity,
            Some(metrics.clone()),
        )),
        Box::new(FifoRemover::new()),
    ));
    let array_sizes: Vec<usize> = specs.iter().map(|s| s.num_bytes()).collect();

    let (transport, pool) = make_transport(16);
    for _ in 0..batch_size {
        transport
            .queue
            .push(TransportQueueItem::new(vec![0u8; 2], pool.clone()))
            .ok()
            .unwrap();
    }
    let transports = vec![transport];

    assert_eq!(metrics.notify_count(), 0);

    let _ = drain_round(&transports, &store, &array_sizes, metrics.drainer(0), 0).await;

    assert_eq!(metrics.notify_count(), 1);
    let _ = store.try_sample().expect("batch should be ready");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_drain_round_notifies_space_available() {
    // drain_round must notify_one on space_available after popping from a
    // queue, or a producer parked on space_available.notified() after a
    // failed push will sit there forever.
    let (store, array_sizes) = make_store_and_sizes();
    let metrics = Metrics::new(1);

    let (transport, pool) = make_transport(2);
    transport
        .queue
        .push(TransportQueueItem::new(vec![0u8; 2], pool.clone()))
        .ok()
        .unwrap();
    transport
        .queue
        .push(TransportQueueItem::new(vec![0u8; 2], pool.clone()))
        .ok()
        .unwrap();
    assert_eq!(transport.queue.len(), 2, "queue is full");

    // Park a task on space_available.notified(), then give the runtime a
    // yield so the waiter is genuinely asleep before drain_round pops.
    let space_available = transport.space_available.clone();
    let waiter = tokio::spawn(async move {
        let notified = space_available.notified();
        notified.await;
    });
    tokio::task::yield_now().await;

    let _ = drain_round(&[transport], &store, &array_sizes, metrics.drainer(0), 0).await;

    tokio::time::timeout(std::time::Duration::from_secs(1), waiter)
        .await
        .expect("producer was not woken by drainer's notify_one")
        .expect("waiter task panicked");
}

#[test]
fn test_sender_lifecycle_tracks_active_connections() {
    let store = make_store();
    let metrics = Metrics::new(2);
    let pool = DrainerPool::new(store, 2, 16, metrics.clone());

    let s1 = pool.new_sender();
    let s2 = pool.new_sender();
    assert_eq!(metrics.active_connections(), 2);

    drop(s1);
    assert_eq!(metrics.active_connections(), 1);

    let s3 = pool.new_sender();
    assert_eq!(metrics.active_connections(), 2);

    drop(s2);
    drop(s3);
    assert_eq!(metrics.active_connections(), 0);

    pool.shutdown();
}

#[cfg(feature = "detailed-metrics")]
#[test]
fn test_drainer_reset_histograms_clears_counts_and_extremes() {
    let m = Metrics::new(1);
    let d = m.drainer(0);

    d.record_memcpy_ns(1_000);
    d.record_memcpy_ns(50_000_000); // big value → hits high bucket
    d.record_drain_round_ns(2_000);
    d.record_queue_dwell_ns(3_000);

    let pre = m.memcpy_aggregate();
    assert_eq!(pre.count, 2);
    assert!(pre.max_ns >= 50_000_000);
    assert!(pre.min_ns <= 1_000);
    assert_eq!(m.drain_round_aggregate().count, 1);
    assert_eq!(m.queue_dwell_aggregate().count, 1);

    d.reset_histograms();

    let post_mc = m.memcpy_aggregate();
    let post_dr = m.drain_round_aggregate();
    let post_qd = m.queue_dwell_aggregate();
    // Counts cleared.
    assert_eq!(post_mc.count, 0);
    assert_eq!(post_dr.count, 0);
    assert_eq!(post_qd.count, 0);
    // mean over empty histogram is 0.
    assert_eq!(post_mc.mean_ns, 0.0);
    // After reset(), max should be back to the sentinel (0), not 50_000_000.
    assert_eq!(post_mc.max_ns, 0);

    // Recording again starts a fresh window.
    d.record_memcpy_ns(7_000);
    let next = m.memcpy_aggregate();
    assert_eq!(next.count, 1);
    assert!(next.min_ns <= 7_000);
    assert!(next.max_ns >= 7_000);
    // Should NOT see the 50_000_000 from before reset.
    assert!(next.max_ns < 50_000_000);
}

#[cfg(feature = "detailed-metrics")]
#[test]
fn test_metrics_reset_histograms_clears_all_drainers() {
    let m = Metrics::new(3);
    m.drainer(0).record_memcpy_ns(1_000);
    m.drainer(1).record_memcpy_ns(2_000);
    m.drainer(2).record_memcpy_ns(3_000);
    m.drainer(0).record_drain_round_ns(4_000);
    m.drainer(2).record_queue_dwell_ns(5_000);
    assert_eq!(m.memcpy_aggregate().count, 3);
    assert_eq!(m.drain_round_aggregate().count, 1);
    assert_eq!(m.queue_dwell_aggregate().count, 1);

    m.reset_histograms();

    assert_eq!(m.memcpy_aggregate().count, 0);
    assert_eq!(m.drain_round_aggregate().count, 0);
    assert_eq!(m.queue_dwell_aggregate().count, 0);
}

#[cfg(feature = "detailed-metrics")]
#[test]
fn test_concurrent_record_and_reset_no_panic() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    let m = Arc::new(Metrics::new(2));
    let stop = Arc::new(AtomicBool::new(false));

    // Two recorder threads, one per drainer, hammering all three histograms.
    let mut recorders = Vec::new();
    for drainer_idx in 0..2 {
        let m = m.clone();
        let stop = stop.clone();
        recorders.push(thread::spawn(move || {
            let mut i: u64 = 0;
            while !stop.load(Ordering::Relaxed) {
                let d = m.drainer(drainer_idx);
                d.record_memcpy_ns(1_000 + (i % 1_000_000));
                d.record_drain_round_ns(2_000 + (i % 1_000_000));
                d.record_queue_dwell_ns(3_000 + (i % 1_000_000));
                i = i.wrapping_add(1);
            }
        }));
    }

    // Resetter — at least a handful of resets while recorders run.
    for _ in 0..20 {
        thread::sleep(Duration::from_millis(5));
        m.reset_histograms();
    }
    stop.store(true, Ordering::Relaxed);
    for r in recorders {
        r.join().unwrap();
    }

    // Final reset → snapshot must be zero (no records can land after this).
    m.reset_histograms();
    let mc = m.memcpy_aggregate();
    let dr = m.drain_round_aggregate();
    let qd = m.queue_dwell_aggregate();
    for snap in [mc, dr, qd] {
        assert_eq!(snap.count, 0);
        assert_eq!(snap.max_ns, 0);
        assert_eq!(snap.mean_ns, 0.0);
    }
}
