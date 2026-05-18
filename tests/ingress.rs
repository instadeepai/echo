//! Tests for the ingress pipeline: per-connection SPSC queues, buffer pooling,
//! the `SampleSender` API, and the drainer pool.
//!
//! `drain_round` smoke tests (queue-depth metric, samples_inserted, batch
//! notify, space_available notify) live in `tests/metrics.rs` because they
//! pin metric semantics. The tests here focus on:
//!   * Buffer recycling via [`TransportQueueItem::Drop`] + free pool.
//!   * [`SampleSender::acquire`] / [`SampleSender::push`] behaviour.
//!   * The full pipeline through the real background `drainer_task`.
//!   * Pool lifecycle (multi-drainer fan-out, shutdown).

use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::queue::ArrayQueue;
use echo::array_spec::ArraySpec;
use echo::ingress::{DrainerPool, TransportQueueItem};
use echo::metrics::Metrics;
use echo::selector::{FifoRemover, FifoSampler};
use echo::store::{ConsumerView, Store};

fn make_store(batch_size: usize, num_batches: usize, elem_bytes: usize) -> Arc<Store> {
    let specs = vec![ArraySpec::new(vec![elem_bytes], 1)];
    let capacity = batch_size * num_batches;
    Arc::new(Store::new(
        specs,
        batch_size,
        num_batches,
        Box::new(FifoSampler::new(batch_size, capacity)),
        Box::new(FifoRemover::new()),
    ))
}

fn read_view(view: &ConsumerView) -> Vec<u8> {
    let (ptr, len) = view.arrays[0];
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() }
}

async fn wait_for_batch(store: &Store, deadline: Duration) -> Option<ConsumerView> {
    let start = Instant::now();
    while start.elapsed() < deadline {
        if let Some(v) = store.try_sample() {
            return Some(v);
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    None
}

/// `DrainerPool` owns a `tokio::runtime::Runtime`; dropping a runtime from
/// inside another runtime's async context panics on the join of its blocking
/// threads. Always tear down the pool through this helper.
async fn drop_pool(pool: DrainerPool) {
    tokio::task::spawn_blocking(move || drop(pool))
        .await
        .unwrap();
}

// ---------------------------------------------------------------------------
// TransportQueueItem::Drop — buffer recycling into the free pool.
// ---------------------------------------------------------------------------

#[test]
fn test_item_drop_recycles_buffer_into_pool() {
    // Length is preserved so a subsequent `read_exact` overwrites the buffer
    // in place — this is the optimization the Drop comment documents.
    let pool: Arc<ArrayQueue<Vec<u8>>> = Arc::new(ArrayQueue::new(4));
    let item = TransportQueueItem::new(vec![0xAA, 0xBB, 0xCC, 0xDD], pool.clone());
    drop(item);

    assert_eq!(pool.len(), 1);
    let buf = pool.pop().unwrap();
    assert_eq!(buf.len(), 4, "len preserved across Drop");
    assert_eq!(
        buf,
        vec![0xAA, 0xBB, 0xCC, 0xDD],
        "bytes preserved (stale-OK)"
    );
}

#[test]
fn test_item_drop_with_full_pool_does_not_panic() {
    // The free pool is bounded; Drop must use the fallible `push`, not
    // `expect`, so a full pool doesn't crash the producer/drainer.
    let pool: Arc<ArrayQueue<Vec<u8>>> = Arc::new(ArrayQueue::new(1));
    pool.push(vec![0u8; 8]).ok().unwrap();
    let item = TransportQueueItem::new(vec![1u8; 8], pool.clone());
    drop(item); // would panic if Drop unwrapped the push
    assert_eq!(pool.len(), 1, "full pool still holds exactly one buffer");
}

// ---------------------------------------------------------------------------
// SampleSender::acquire — empty pool returns a fresh zeroed buffer.
// ---------------------------------------------------------------------------

#[test]
fn test_acquire_returns_fresh_zeroed_buffer_when_pool_empty() {
    let store = make_store(4, 2, 16);
    let metrics = Metrics::new(1);
    let pool = DrainerPool::new(store, 1, 4, metrics);
    let sender = pool.new_sender();

    let buf = sender.acquire(16);
    assert_eq!(buf.len(), 16);
    assert!(buf.iter().all(|&b| b == 0), "fresh allocation is zeroed");

    pool.shutdown();
}

// ---------------------------------------------------------------------------
// Full pipeline: SampleSender::push -> drainer_task -> Store.
// Exercises the real background drainer, not just the bare `drain_round`.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_push_lands_samples_in_store() {
    let batch_size = 4;
    let elem_bytes = 2;
    let store = make_store(batch_size, 4, elem_bytes);
    let metrics = Metrics::new(1);
    let pool = DrainerPool::new(store.clone(), 1, 16, metrics);
    let sender = pool.new_sender();

    for i in 0u8..4 {
        let mut buf = sender.acquire(elem_bytes);
        buf[0] = i;
        buf[1] = 0xEE;
        sender.push(buf).await;
    }

    let view = wait_for_batch(&store, Duration::from_secs(2))
        .await
        .expect("batch should land within 2s");
    assert_eq!(view.count, 4);
    assert_eq!(read_view(&view), vec![0, 0xEE, 1, 0xEE, 2, 0xEE, 3, 0xEE]);
    drop(view);

    pool.shutdown();
    drop_pool(pool).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_multi_drainer_pool_distributes_load() {
    // Round-robin assignment + N senders => every drainer gets at least one
    // transport, so every drainer's `samples_inserted` should be non-zero.
    let batch_size = 4;
    let elem_bytes = 2;
    let num_drainers = 4;
    let store = make_store(batch_size, 16, elem_bytes); // capacity 64
    let metrics = Metrics::new(num_drainers);
    let pool = DrainerPool::new(store.clone(), num_drainers, 16, metrics.clone());

    let senders: Vec<_> = (0..num_drainers).map(|_| pool.new_sender()).collect();
    for (i, sender) in senders.iter().enumerate() {
        for j in 0u8..4 {
            let mut buf = sender.acquire(elem_bytes);
            buf[0] = i as u8;
            buf[1] = j;
            sender.push(buf).await;
        }
    }

    // Wait for every sample to reach the store.
    let total = (num_drainers * 4) as u64;
    let start = Instant::now();
    while metrics.samples_inserted_total() < total && start.elapsed() < Duration::from_secs(5) {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert_eq!(metrics.samples_inserted_total(), total);

    for d in 0..num_drainers {
        assert!(
            metrics.drainer(d).samples_inserted() > 0,
            "drainer {d} should have processed at least one sample under round-robin",
        );
    }

    drop(senders);
    pool.shutdown();
    drop_pool(pool).await;
}

// ---------------------------------------------------------------------------
// Backpressure: push must block (and record push_blocked) when the SPSC
// queue is full because downstream is saturated.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_push_blocks_and_records_metric_when_saturated() {
    // SPSC queue size 1 + a 4-slot store with no consumer: the drainer's
    // `insert_batch` will block once the store fills, the SPSC will then
    // fill, and the next `push` must wait on space_available. We never
    // sample so the saturation is permanent.
    let elem_bytes = 2;
    let store = make_store(2, 2, elem_bytes); // capacity 4
    let metrics = Metrics::new(1);
    let pool = DrainerPool::new(store, 1, 1, metrics.clone());
    let sender = pool.new_sender();

    let mut blocked = false;
    for i in 0u8..32 {
        let mut buf = sender.acquire(elem_bytes);
        buf[0] = i;
        buf[1] = 0;
        if tokio::time::timeout(Duration::from_millis(200), sender.push(buf))
            .await
            .is_err()
        {
            blocked = true;
            break;
        }
    }
    assert!(
        blocked,
        "push should eventually block when downstream is saturated"
    );
    assert!(
        metrics.push_blocked_count() > 0,
        "push_blocked must be recorded before the await on space_available",
    );

    drop(sender);
    pool.shutdown();
    drop_pool(pool).await;
}

// ---------------------------------------------------------------------------
// DrainerPool::shutdown stops the drainer tasks.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_shutdown_stops_drainers() {
    // Setup with capacity well above what we push so the drainer never
    // blocks inside `insert_batch` — that's the only case `shutdown()` can
    // unblock from.
    let elem_bytes = 2;
    let store = make_store(4, 4, elem_bytes); // capacity 16
    let metrics = Metrics::new(1);
    let pool = DrainerPool::new(store.clone(), 1, 16, metrics.clone());
    let sender = pool.new_sender();

    // Push and confirm the drainer was alive.
    for i in 0u8..4 {
        let mut buf = sender.acquire(elem_bytes);
        buf[0] = i;
        buf[1] = 0;
        sender.push(buf).await;
    }
    let _view = wait_for_batch(&store, Duration::from_secs(2))
        .await
        .expect("batch ready");
    let before = metrics.samples_inserted_total();
    assert_eq!(before, 4);

    pool.shutdown();
    // Give the drainer a moment to wake from notified.await and break out
    // of its loop.
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Push more — these can sit in the SPSC queue but the drainer is gone,
    // so they must not reach the store.
    for i in 4u8..8 {
        let mut buf = sender.acquire(elem_bytes);
        buf[0] = i;
        buf[1] = 0;
        if tokio::time::timeout(Duration::from_millis(50), sender.push(buf))
            .await
            .is_err()
        {
            break;
        }
    }
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(
        metrics.samples_inserted_total(),
        before,
        "no new samples should reach the store after shutdown",
    );

    drop(sender);
    drop_pool(pool).await;
}
