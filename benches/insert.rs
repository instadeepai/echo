//! Drainer-side insert path in isolation: `insert_batch` + `sample`, no
//! network, no tokio worker pool.
//!
//! Four pytree shapes, because the per-array overhead and the memcpy cost
//! trade off against each other:
//!   - `atari`: two big observation arrays, 55 KiB/sample; memcpy dominates.
//!   - `medium`: eight 1 KiB arrays; a realistic middle.
//!   - `scalars`: 32 eight-byte arrays; per-array overhead dominates.
//!   - `rollout`: the 44-array, 996.5 KiB/sample pytree that
//!     `benches/bench_distributed.py` actually sends over TCP.

use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use crossbeam::queue::ArrayQueue;
use echo::array_spec::ArraySpec;
use echo::ingress::{drain_round, TransportHandle, TransportQueueItem};
use echo::metrics::Metrics;
use echo::selector::{FifoRemover, FifoSampler};
use echo::store::Store;
use tokio::sync::Notify;

const BATCH_SIZE: usize = 256;
const NUM_BUFFERS: usize = 4;

/// One pytree shape: a name, and per array its shape and dtype size in bytes.
type Pytree = (&'static str, Vec<(Vec<usize>, usize)>);

fn shapes() -> Vec<Pytree> {
    vec![
        (
            "atari",
            vec![
                (vec![84, 84, 4], 1), // obs
                (vec![1], 4),          // action
                (vec![1], 4),          // reward
                (vec![1], 1),          // done
                (vec![84, 84, 4], 1), // next_obs
            ],
        ),
        ("medium", (0..8).map(|_| (vec![256], 4)).collect()),
        // The pytree from benches/bench_distributed.py: _rl_sample() with every
        // leaf stacked ROLLOUT_LEN=5 deep, plus the scalar _send_ts. 44 arrays,
        // 996.5 KiB per sample. Generated from the Python definition, in optree
        // flatten order.
        (
            "rollout",
            vec![
                (vec![], 8),
                (vec![5, 10, 17], 4),
                (vec![5, 2], 4),
                (vec![5, 80, 2], 4),
                (vec![5, 80], 4),
                (vec![5, 80], 4),
                (vec![5, 80, 16, 2, 5], 2),
                (vec![5, 80, 16, 2, 5], 2),
                (vec![5, 80, 16, 1, 5], 2),
                (vec![5, 80, 16, 1, 5], 2),
                (vec![5, 10, 17], 4),
                (vec![5, 10, 8], 4),
                (vec![5, 10, 2], 4),
                (vec![5, 4], 4),
                (vec![5, 10, 2], 4),
                (vec![5, 10, 9], 4),
                (vec![5, 10, 8], 4),
                (vec![5, 10, 8], 4),
                (vec![5, 10, 4], 4),
                (vec![5, 10, 1], 4),
                (vec![5, 10, 5], 4),
                (vec![5, 10, 40], 4),
                (vec![5, 10, 30], 4),
                (vec![5, 10, 10], 4),
                (vec![5, 3], 4),
                (vec![5, 10, 8], 4),
                (vec![5, 10, 8], 4),
                (vec![5, 10, 8], 4),
                (vec![5, 10, 2], 4),
                (vec![5, 10, 6], 4),
                (vec![5, 10, 2], 4),
                (vec![5, 10], 4),
                (vec![5, 10, 16, 6, 14], 2),
                (vec![5, 10, 16, 6, 14], 2),
                (vec![5, 10, 32, 24], 4),
                (vec![5, 80, 2], 4),
                (vec![5, 10], 4),
                (vec![5, 10, 16, 3, 14], 2),
                (vec![5, 10, 16, 3, 14], 2),
                (vec![5, 80, 2], 4),
                (vec![5, 80, 1], 4),
                (vec![5, 80, 8], 4),
                (vec![5, 80, 2], 4),
                (vec![5, 80, 6], 4),
            ],
        ),
        ("scalars", (0..32).map(|_| (vec![1], 8)).collect()),
    ]
}

fn make_store(specs: Vec<ArraySpec>) -> Store {
    let capacity = BATCH_SIZE * NUM_BUFFERS;
    Store::new(
        specs,
        BATCH_SIZE,
        NUM_BUFFERS,
        Box::new(FifoSampler::new(BATCH_SIZE, capacity)),
        Box::new(FifoRemover::new()),
    )
}

fn bench_insert(c: &mut Criterion) {
    // insert_batch is async but never yields while the ring has space, so a
    // current-thread runtime resolves it without ever parking.
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");

    let mut group = c.benchmark_group("insert_batch");

    for (name, dims) in shapes() {
        let specs: Vec<ArraySpec> = dims
            .iter()
            .map(|(shape, dtype)| ArraySpec::new(shape.clone(), *dtype))
            .collect();
        let array_sizes: Vec<usize> = specs.iter().map(|s| s.num_bytes()).collect();
        let payload: usize = array_sizes.iter().sum();

        let store = make_store(specs);
        // One reusable batch of source payloads, as the drainer sees them:
        // each sample is its arrays concatenated in pytree order.
        let owned: Vec<Vec<u8>> = (0..BATCH_SIZE).map(|i| vec![i as u8; payload]).collect();
        let slices: Vec<&[u8]> = owned.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Bytes((payload * BATCH_SIZE) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                rt.block_on(store.insert_batch(black_box(&slices), &array_sizes, None));
                black_box(store.sample());
            });
        });
    }

    group.finish();
}

/// A full drainer round: pop every connection's SPSC queue into one batch and
/// hand it to the store. Covers the per-round bookkeeping the `insert_batch`
/// group skips.
fn bench_drain_round(c: &mut Criterion) {
    // Enough connections that a round's batch is the size a real deployment
    // sees, which is what the per-round bookkeeping scales with.
    const CONNECTIONS: usize = 64;
    const QUEUE_DEPTH: usize = 8;

    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime");

    let mut group = c.benchmark_group("drain_round");

    for (name, dims) in shapes() {
        let specs: Vec<ArraySpec> = dims
            .iter()
            .map(|(shape, dtype)| ArraySpec::new(shape.clone(), *dtype))
            .collect();
        let array_sizes: Vec<usize> = specs.iter().map(|s| s.num_bytes()).collect();
        let payload: usize = array_sizes.iter().sum();

        // Enough headroom that the store never backpressures mid-round.
        let store = Arc::new(make_store(specs));
        let metrics = Metrics::new(1);

        let pools: Vec<Arc<ArrayQueue<Vec<u8>>>> = (0..CONNECTIONS)
            .map(|_| Arc::new(ArrayQueue::new(QUEUE_DEPTH * 2)))
            .collect();
        let transports: Vec<TransportHandle> = (0..CONNECTIONS)
            .map(|_| TransportHandle {
                queue: Arc::new(ArrayQueue::new(QUEUE_DEPTH)),
                space_available: Arc::new(Notify::new()),
                closed: Arc::new(AtomicBool::new(false)),
            })
            .collect();

        let per_round = CONNECTIONS * QUEUE_DEPTH;
        group.throughput(Throughput::Bytes((payload * per_round) as u64));
        // A round drains the queues, so they must be refilled between rounds.
        // `iter_batched` would run every setup before every routine, and the
        // queues are bounded, so only the first refill would survive; hand-roll
        // the timing instead and keep refill and release outside it.
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_custom(|iters| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iters {
                    for (t, pool) in transports.iter().zip(&pools) {
                        for _ in 0..QUEUE_DEPTH {
                            // Recycle through the pool, as a real connection does.
                            let buf = pool.pop().unwrap_or_else(|| vec![0u8; payload]);
                            let _ = t.queue.push(TransportQueueItem::new(buf, pool.clone()));
                        }
                    }

                    let start = Instant::now();
                    rt.block_on(drain_round(
                        &transports,
                        &store,
                        &array_sizes,
                        metrics.drainer(0),
                        0,
                    ));
                    elapsed += start.elapsed();

                    // Release what was inserted so the ring never backpressures.
                    while store.try_sample().is_some() {}
                }
                elapsed
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_insert, bench_drain_round);
criterion_main!(benches);
