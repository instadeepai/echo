//! Micro-benchmarks for the Rust hot path: ring writes, the drainer's insert
//! path, the FIFO sampler and the consumer's `sample`. No network and no
//! drainer runtime; each group drives the crate's own methods directly so a
//! change to one of them shows up undiluted.
//!
//! Four pytree shapes, because the per-array overhead and the memcpy cost
//! trade off against each other:
//!   - `atari`: two big observation arrays, 55 KiB/sample; memcpy dominates.
//!   - `small`: 32 eight-byte scalars; per-array overhead dominates.
//!   - `medium`: eight 1 KiB arrays; a realistic middle.
//!   - `large`: the 44-array, 996.5 KiB/sample pytree that
//!     `benches/bench_distributed.py` actually sends over TCP.

use std::hint::black_box;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use crossbeam::queue::ArrayQueue;
use echo::array_spec::ArraySpec;
use echo::ingress::{drain_round, TransportHandle, TransportQueueItem};
use echo::metrics::Metrics;
use echo::ring_buf::PytreeRingBuf;
use echo::selector::{FifoRemover, FifoSampler, Sampler};
use echo::store::Store;
use tokio::runtime::Runtime;
use tokio::sync::Notify;

const BATCH_SIZE: usize = 256;
/// The minimum `Store::new` accepts, and all these benches need: one batch
/// held by the consumer while the next is written.
const NUM_BUFFERS: usize = 2;
const CAPACITY: usize = BATCH_SIZE * NUM_BUFFERS;

/// Samples per `insert_batch` call, for the groups that isolate per-reservation
/// cost. 1 is a lone sample; `BATCH_SIZE` is a whole batch in one reservation.
const CHUNKS: [usize; 3] = [1, 16, BATCH_SIZE];

/// One pytree shape: a name, and per array its shape and dtype size in bytes.
type Pytree = (&'static str, Vec<(Vec<usize>, usize)>);

fn pytrees() -> Vec<Pytree> {
    vec![
        (
            "atari",
            vec![
                (vec![84, 84, 4], 1), // obs
                (vec![1], 4),         // action
                (vec![1], 4),         // reward
                (vec![1], 1),         // done
                (vec![84, 84, 4], 1), // next_obs
            ],
        ),
        ("small", (0..32).map(|_| (vec![1], 8)).collect()),
        ("medium", (0..8).map(|_| (vec![256], 4)).collect()),
        // The pytree from benches/bench_distributed.py: _rl_sample() with every
        // leaf stacked ROLLOUT_LEN=5 deep, plus the scalar _send_ts. Generated
        // from the Python definition, in optree flatten order.
        (
            "large",
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
    ]
}

fn to_specs(dims: &[(Vec<usize>, usize)]) -> Vec<ArraySpec> {
    dims.iter()
        .map(|(shape, dtype)| ArraySpec::new(shape.clone(), *dtype))
        .collect()
}

fn array_sizes(specs: &[ArraySpec]) -> Vec<usize> {
    specs.iter().map(ArraySpec::num_bytes).collect()
}

/// Built the way `PyServer::new` builds it, sampler metrics included.
fn make_store(
    specs: Vec<ArraySpec>,
    batch_size: usize,
    num_buffers: usize,
    metrics: &Arc<Metrics>,
) -> Arc<Store> {
    Arc::new(Store::new(
        specs,
        batch_size,
        num_buffers,
        Box::new(FifoSampler::with_metrics(
            batch_size,
            batch_size * num_buffers,
            Some(metrics.clone()),
        )),
        Box::new(FifoRemover::new()),
    ))
}

/// One source payload: a sample's arrays concatenated in pytree order.
///
/// Never all zeros. A zeroed `Vec` comes from calloc unwritten, so its pages
/// all map the kernel's shared zero page and copying from it reads cache, not
/// DRAM: about 20% faster on `large`. The transport's `read_exact` always
/// writes its buffers, so a zeroed source would flatter the bench.
fn make_payload(payload: usize, seed: usize) -> Vec<u8> {
    vec![1 + (seed % 255) as u8; payload]
}

/// One batch of source payloads, as the drainer sees them.
fn make_samples(payload: usize, count: usize) -> Vec<Vec<u8>> {
    (0..count).map(|i| make_payload(payload, i)).collect()
}

/// `insert_batch` is async but never yields while the ring has space, so a
/// current-thread runtime resolves it without ever parking.
fn runtime() -> Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("runtime")
}

/// The floor under `insert_batch`: the same per-slot, per-array copies into a
/// `PytreeRingBuf`, with no reservation, commit or metrics. The gap between
/// this group and `insert_batch` is what the synchronisation costs.
fn bench_memcpy(c: &mut Criterion) {
    let mut group = c.benchmark_group("memcpy");

    for (name, dims) in pytrees() {
        let sizes = array_sizes(&to_specs(&dims));
        let payload: usize = sizes.iter().sum();
        let ring = PytreeRingBuf::new(sizes.clone(), CAPACITY, BATCH_SIZE);
        let samples = make_samples(payload, BATCH_SIZE);
        // Advance through the ring as the store does, so both touch the same
        // amount of memory.
        let mut cursor = 0;

        group.throughput(Throughput::Bytes((payload * BATCH_SIZE) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                for (i, sample) in samples.iter().enumerate() {
                    let slot = (cursor + i) % CAPACITY;
                    let mut offset = 0;
                    for (j, &size) in sizes.iter().enumerate() {
                        // SAFETY: slot < CAPACITY, j < num_arrays, and this
                        // thread is the ring's only user.
                        unsafe {
                            let dst = ring.slot_mut(slot, j);
                            std::ptr::copy_nonoverlapping(sample.as_ptr().add(offset), dst, size);
                        }
                        offset += size;
                    }
                }
                cursor = (cursor + BATCH_SIZE) % CAPACITY;
                black_box(&ring);
            });
        });
    }

    group.finish();
}

/// The drainer's insert path for one whole batch, then the consumer's
/// `sample` to release it.
fn bench_insert_batch(c: &mut Criterion) {
    let rt = runtime();
    let mut group = c.benchmark_group("insert_batch");

    for (name, dims) in pytrees() {
        let specs = to_specs(&dims);
        let sizes = array_sizes(&specs);
        let payload: usize = sizes.iter().sum();
        let metrics = Metrics::new(1);
        let store = make_store(specs, BATCH_SIZE, NUM_BUFFERS, &metrics);
        let owned = make_samples(payload, BATCH_SIZE);
        let slices: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();

        group.throughput(Throughput::Bytes((payload * BATCH_SIZE) as u64));
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter(|| {
                rt.block_on(store.insert_batch(
                    black_box(&slices),
                    &sizes,
                    Some(metrics.drainer(0)),
                ));
                black_box(store.sample());
            });
        });
    }

    group.finish();
}

/// Per-reservation cost of `insert_batch`: one batch of 1-byte samples,
/// inserted `chunk` samples per call, so the CAS, commit and metrics overhead
/// is not buried under memcpy.
fn bench_reservation(c: &mut Criterion) {
    let rt = runtime();
    let mut group = c.benchmark_group("reservation");

    let specs = vec![ArraySpec::new(vec![1], 1)];
    let sizes = array_sizes(&specs);
    let metrics = Metrics::new(1);
    let store = make_store(specs, BATCH_SIZE, NUM_BUFFERS, &metrics);
    let owned = make_samples(1, BATCH_SIZE);
    let slices: Vec<&[u8]> = owned.iter().map(Vec::as_slice).collect();

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    for chunk in CHUNKS {
        group.bench_function(BenchmarkId::from_parameter(chunk), |b| {
            b.iter(|| {
                // One block_on per batch, so its overhead isn't counted per chunk.
                rt.block_on(async {
                    for part in slices.chunks(chunk) {
                        store
                            .insert_batch(black_box(part), &sizes, Some(metrics.drainer(0)))
                            .await;
                    }
                });
                black_box(store.sample());
            });
        });
    }

    group.finish();
}

/// `FifoSampler` alone: commit one batch `chunk` positions at a time, then
/// select it. The baseline any other `Sampler` gets compared against.
fn bench_fifo_sampler(c: &mut Criterion) {
    let mut group = c.benchmark_group("fifo_sampler");

    group.throughput(Throughput::Elements(BATCH_SIZE as u64));
    for chunk in CHUNKS {
        let sampler = FifoSampler::with_metrics(BATCH_SIZE, CAPACITY, Some(Metrics::new(1)));
        let mut pos = 0;

        group.bench_function(BenchmarkId::from_parameter(chunk), |b| {
            b.iter(|| {
                for start in (pos..pos + BATCH_SIZE).step_by(chunk) {
                    sampler.commit_batch(start, chunk);
                }
                // The batch is complete, so select returns without waiting.
                black_box(sampler.select(BATCH_SIZE));
                pos += BATCH_SIZE;
            });
        });
    }

    group.finish();
}

/// The consumer's `sample`: release the previous batch, select the next and
/// build the zero-copy view. Only the number of arrays matters to it, so each
/// leaf shrinks to one byte and a batch to one sample, keeping the untimed
/// refill cheap.
fn bench_sample(c: &mut Criterion) {
    // Samples inserted before each timed run of `sample` calls. The ring holds
    // one more, for the batch the consumer still has when the refill starts.
    const REFILL: usize = 64;

    let mut group = c.benchmark_group("sample");

    for (name, dims) in pytrees() {
        let specs: Vec<ArraySpec> = dims.iter().map(|_| ArraySpec::new(vec![1], 1)).collect();
        let metrics = Metrics::new(1);
        let store = make_store(specs, 1, REFILL + 1, &metrics);
        let sample = [0u8];
        let arrays: Vec<&[u8]> = dims.iter().map(|_| sample.as_slice()).collect();

        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            // iter_batched runs all REFILL setups, then times all REFILL
            // routines, so a batch is always ready and `sample` never blocks.
            b.iter_batched(
                || store.insert_sync(&arrays),
                |()| store.sample(),
                BatchSize::NumIterations(REFILL as u64),
            );
        });
    }

    group.finish();
}

/// A full drainer round: pop every connection's SPSC queue into one batch and
/// hand it to the store. Covers the per-round bookkeeping that `insert_batch`
/// skips.
fn bench_drain_round(c: &mut Criterion) {
    // A round's worth of samples is exactly one batch.
    const CONNECTIONS: usize = 32;
    const QUEUE_DEPTH: usize = BATCH_SIZE / CONNECTIONS;

    let rt = runtime();
    let mut group = c.benchmark_group("drain_round");

    for (name, dims) in pytrees() {
        let specs = to_specs(&dims);
        let sizes = array_sizes(&specs);
        let payload: usize = sizes.iter().sum();
        let metrics = Metrics::new(1);
        let store = make_store(specs, BATCH_SIZE, NUM_BUFFERS, &metrics);

        // Sized as `DrainerPool::new_sender` sizes them.
        let pools: Vec<Arc<ArrayQueue<Vec<u8>>>> = (0..CONNECTIONS)
            .map(|_| Arc::new(ArrayQueue::new(2 * QUEUE_DEPTH + 1)))
            .collect();
        let transports: Vec<TransportHandle> = (0..CONNECTIONS)
            .map(|_| TransportHandle {
                queue: Arc::new(ArrayQueue::new(QUEUE_DEPTH)),
                space_available: Arc::new(Notify::new()),
                closed: Arc::new(AtomicBool::new(false)),
            })
            .collect();

        group.throughput(Throughput::Bytes((payload * BATCH_SIZE) as u64));
        // A round drains the queues, so they must be refilled between rounds.
        // `iter_batched` would run every setup before every routine, and the
        // queues are bounded, so only the first refill would survive; hand-roll
        // the timing instead and keep refill and release outside it.
        group.bench_function(BenchmarkId::from_parameter(name), |b| {
            b.iter_custom(|iters| {
                let mut elapsed = Duration::ZERO;
                for round in 0..iters {
                    for (t, pool) in transports.iter().zip(&pools) {
                        for _ in 0..QUEUE_DEPTH {
                            // Recycle through the pool, as `SampleSender::acquire` does.
                            let buf = pool.pop().unwrap_or_else(|| make_payload(payload, 0));
                            let _ = t.queue.push(TransportQueueItem::new(buf, pool.clone()));
                        }
                    }

                    let start = Instant::now();
                    rt.block_on(drain_round(
                        &transports,
                        &store,
                        &sizes,
                        metrics.drainer(0),
                        round as usize,
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

/// Short runs: the point is a quick before/after on one machine, and these
/// benches are stable enough that 20 samples show a real change.
fn config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = config();
    targets = bench_memcpy,
        bench_insert_batch,
        bench_reservation,
        bench_fifo_sampler,
        bench_sample,
        bench_drain_round
}
criterion_main!(benches);
