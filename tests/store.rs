use std::sync::Arc;
use std::thread;
use std::time::Duration;

use echo::array_spec::ArraySpec;
use echo::metrics::Metrics;
use echo::selector::{FifoRemover, FifoSampler};
use echo::store::Store;

fn make_store(batch_size: usize, num_batches: usize) -> Store {
    let specs = vec![ArraySpec::new(vec![2], 1)]; // 2 bytes per sample
    let capacity = batch_size * num_batches;
    Store::new(
        specs,
        batch_size,
        num_batches,
        Box::new(FifoSampler::new(batch_size, capacity)),
        Box::new(FifoRemover::new()),
    )
}

fn read_data(view: &echo::store::ConsumerView) -> Vec<u8> {
    assert_eq!(view.arrays.len(), 1);
    let (ptr, len) = view.arrays[0];
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() }
}

#[test]
fn test_insert_and_sample_basic() {
    let store = make_store(2, 2);
    store.insert_sync(&[&[0xAA, 0xBB]]);
    store.insert_sync(&[&[0xCC, 0xDD]]);

    let view = store.try_sample().expect("should have a batch");
    assert_eq!(view.count, 2);
    assert_eq!(read_data(&view), vec![0xAA, 0xBB, 0xCC, 0xDD]);
}

#[test]
fn test_multi_array() {
    let specs = vec![ArraySpec::new(vec![2], 1), ArraySpec::new(vec![3], 1)];
    let capacity = 4;
    let store = Store::new(
        specs,
        2,
        2,
        Box::new(FifoSampler::new(2, capacity)),
        Box::new(FifoRemover::new()),
    );

    store.insert_sync(&[&[0x01, 0x02], &[0x10, 0x20, 0x30]]);
    store.insert_sync(&[&[0x03, 0x04], &[0x40, 0x50, 0x60]]);

    let view = store.try_sample().unwrap();
    assert_eq!(view.count, 2);

    let d0 = unsafe { std::slice::from_raw_parts(view.arrays[0].0 as *const u8, view.arrays[0].1) };
    let d1 = unsafe { std::slice::from_raw_parts(view.arrays[1].0 as *const u8, view.arrays[1].1) };
    assert_eq!(d0, &[0x01, 0x02, 0x03, 0x04]);
    assert_eq!(d1, &[0x10, 0x20, 0x30, 0x40, 0x50, 0x60]);
}

#[test]
fn test_pool_cycling() {
    let store = make_store(2, 3);

    store.insert_sync(&[&[0x01, 0x02]]);
    store.insert_sync(&[&[0x03, 0x04]]);
    let v1 = store.try_sample().unwrap();
    assert_eq!(read_data(&v1), vec![0x01, 0x02, 0x03, 0x04]);

    store.insert_sync(&[&[0x05, 0x06]]);
    store.insert_sync(&[&[0x07, 0x08]]);
    let v2 = store.try_sample().unwrap();
    assert_eq!(read_data(&v2), vec![0x05, 0x06, 0x07, 0x08]);

    store.insert_sync(&[&[0x09, 0x0A]]);
    store.insert_sync(&[&[0x0B, 0x0C]]);
    let v3 = store.try_sample().unwrap();
    assert_eq!(read_data(&v3), vec![0x09, 0x0A, 0x0B, 0x0C]);
}

#[test]
fn test_try_sample_not_enough() {
    let store = make_store(4, 2);
    store.insert_sync(&[&[0x01, 0x02]]);
    store.insert_sync(&[&[0x03, 0x04]]);
    assert!(store.try_sample().is_none());
}

#[test]
fn test_try_sample_releases_previous() {
    let store = make_store(2, 2); // capacity=4

    store.insert_sync(&[&[0x01, 0x02]]);
    store.insert_sync(&[&[0x03, 0x04]]);
    let _v1 = store.try_sample().unwrap();

    store.insert_sync(&[&[0x05, 0x06]]);
    store.insert_sync(&[&[0x07, 0x08]]);

    // Ring is full (4 used). try_sample releases previous batch first.
    let v2 = store.try_sample().unwrap();
    assert_eq!(read_data(&v2), vec![0x05, 0x06, 0x07, 0x08]);

    // After releasing v1's slots, we can insert again.
    store.insert_sync(&[&[0x09, 0x0A]]);
    assert!(store.try_sample().is_none()); // only 1, need 2

    store.insert_sync(&[&[0x0B, 0x0C]]);
    assert!(store.try_sample().is_some());
}

#[test]
fn test_shutdown_returns_none() {
    let store = Arc::new(make_store(2, 2));
    let s = store.clone();
    let handle = thread::spawn(move || s.sample());

    thread::sleep(Duration::from_millis(50));
    store.shutdown();

    assert!(handle.join().unwrap().is_none());
}

#[test]
fn test_backpressure() {
    let store = Arc::new(make_store(2, 2)); // capacity=4

    store.insert_sync(&[&[0x01, 0x02]]);
    store.insert_sync(&[&[0x02, 0x03]]);
    store.insert_sync(&[&[0x03, 0x04]]);
    store.insert_sync(&[&[0x04, 0x05]]);
    // Ring full.

    let inserted = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let ins = inserted.clone();
    let s = store.clone();
    let handle = thread::spawn(move || {
        s.insert_sync(&[&[0xFF, 0xFE]]);
        ins.store(true, std::sync::atomic::Ordering::Release);
    });

    thread::sleep(Duration::from_millis(50));
    assert!(!inserted.load(std::sync::atomic::Ordering::Acquire));

    // sample() releases previous batch's slots on next call.
    let _v1 = store.sample().unwrap(); // consumes batch, marks has_previous
    let _v2 = store.sample().unwrap(); // releases batch 1, consumes batch 2

    handle.join().unwrap();
    assert!(inserted.load(std::sync::atomic::Ordering::Acquire));
}

#[test]
#[should_panic(expected = "num_batches must be >= 2")]
fn test_num_batches_minimum() {
    make_store(2, 1);
}

#[test]
fn test_store_size_tracks_commits_and_samples() {
    let store = make_store(2, 2);
    assert_eq!(store.size(), 0, "empty store");

    store.insert_sync(&[&[0xAA, 0xBB]]);
    store.insert_sync(&[&[0xCC, 0xDD]]);
    assert_eq!(store.size(), 2, "2 inserted, none sampled");

    let _view = store.try_sample().expect("batch ready");
    assert_eq!(store.size(), 0, "batch sampled → queue empty");
}

#[test]
fn test_many_batches_sequential() {
    let store = make_store(2, 3);
    for batch_idx in 0u8..10 {
        let a = batch_idx * 2;
        let b = batch_idx * 2 + 1;
        store.insert_sync(&[&[a, 0]]);
        store.insert_sync(&[&[b, 0]]);
        let view = store.sample().unwrap();
        let data = read_data(&view);
        assert_eq!(data[0], a);
        assert_eq!(data[2], b);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_insert_batch_splits_across_batch_boundary() {
    // batch_size=4, num_batches=3, capacity=12. Inserting 10 samples in one
    // call forces 3 reservations: 4+4+2.
    let store = Arc::new(make_store(4, 3));
    let metrics = Metrics::new(1);
    let array_sizes = vec![2usize]; // one array, 2 bytes per sample

    // Each sample is one pytree — 2 bytes. Tag byte 0 = index; byte 1 = 0xEE.
    let samples: Vec<Vec<u8>> = (0u8..10).map(|i| vec![i, 0xEE]).collect();
    let refs: Vec<&[u8]> = samples.iter().map(|v| v.as_slice()).collect();

    store
        .insert_batch(&refs, &array_sizes, Some(metrics.drainer(0)))
        .await;

    // Drainer-path samples are counted, so we can assert the exact count.
    assert_eq!(metrics.samples_inserted_total(), 10);

    // Pull batch 0 (samples 0..4) and batch 1 (samples 4..8). Batch 2
    // (only 2 of 4 slots filled) is not ready yet.
    let v0 = store.try_sample().expect("batch 0 ready");
    assert_eq!(v0.count, 4);
    assert_eq!(read_data(&v0), vec![0, 0xEE, 1, 0xEE, 2, 0xEE, 3, 0xEE]);

    let v1 = store.try_sample().expect("batch 1 ready");
    assert_eq!(v1.count, 4);
    assert_eq!(read_data(&v1), vec![4, 0xEE, 5, 0xEE, 6, 0xEE, 7, 0xEE]);

    assert!(store.try_sample().is_none(), "batch 2 only half full");
}

#[test]
fn test_insert_batch_concurrent_preserves_count_and_integrity() {
    // Concurrent drainers must neither lose nor duplicate samples.
    // batch=4, num_batches=6, capacity=24; 4 threads × 4 samples = 16 (no backpressure).
    // The goal is CAS contention on the reservation loop, not backpressure.
    const THREADS: u8 = 4;
    const PER_THREAD: u8 = 4;
    const TOTAL: usize = (THREADS as usize) * (PER_THREAD as usize);

    let store = Arc::new(make_store(4, 6));
    let metrics = Metrics::new(THREADS as usize);
    let array_sizes = Arc::new(vec![2usize]);

    let mut handles = Vec::new();
    for t in 0..THREADS {
        let store = store.clone();
        let metrics = metrics.clone();
        let array_sizes = array_sizes.clone();
        handles.push(thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async move {
                // Each thread submits PER_THREAD samples in a single call
                // so the reservation-CAS path is exercised under contention.
                let samples: Vec<Vec<u8>> = (0..PER_THREAD).map(|i| vec![t, i]).collect();
                let refs: Vec<&[u8]> = samples.iter().map(|v| v.as_slice()).collect();
                store
                    .insert_batch(&refs, &array_sizes, Some(metrics.drainer(t as usize)))
                    .await;
            });
        }));
    }
    for h in handles {
        h.join().unwrap();
    }

    // Each drainer reports its own samples_inserted; aggregate must equal total.
    assert_eq!(metrics.samples_inserted_total(), TOTAL as u64);

    // Pull every committed sample and rebuild the multiset. Regardless of
    // arrival order, we must see each (thread, index) pair exactly once.
    let mut seen: Vec<(u8, u8)> = Vec::with_capacity(TOTAL);
    for _ in 0..(TOTAL / 4) {
        let v = store.sample().expect("batch ready");
        let data = read_data(&v);
        for chunk in data.chunks(2) {
            seen.push((chunk[0], chunk[1]));
        }
    }

    seen.sort();
    let mut expected: Vec<(u8, u8)> = (0..THREADS)
        .flat_map(|t| (0..PER_THREAD).map(move |i| (t, i)))
        .collect();
    expected.sort();
    assert_eq!(seen, expected, "no loss, no duplication, no corruption");
}

#[test]
fn test_concurrent_insert_data_integrity() {
    let store = Arc::new(make_store(4, 3)); // batch=4, capacity=12

    let s = store.clone();
    let writers = thread::spawn(move || {
        let mut handles = vec![];
        for i in 0u8..12 {
            let s = s.clone();
            handles.push(thread::spawn(move || {
                s.insert_sync(&[&[i, i + 100]]);
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
    });

    let mut all_values = vec![];
    for _ in 0..3 {
        let view = store.sample().unwrap();
        let data = read_data(&view);
        for chunk in data.chunks(2) {
            all_values.push((chunk[0], chunk[1]));
        }
    }

    writers.join().unwrap();

    all_values.sort();
    let expected: Vec<(u8, u8)> = (0u8..12).map(|i| (i, i + 100)).collect();
    assert_eq!(all_values, expected);
}
