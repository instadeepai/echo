//! Tests for the TCP transport: wire-format handshake, sample ingestion via
//! real sockets, multi-connection handling, and lifecycle.

use std::sync::Arc;
use std::time::{Duration, Instant};

use echo::array_spec::ArraySpec;
use echo::ingress::DrainerPool;
use echo::metrics::Metrics;
use echo::selector::{FifoRemover, FifoSampler};
use echo::store::{ConsumerView, Store};
use echo::transport::tcp::{encode_handshake, TcpTransport};
use echo::transport::Transport;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// Bind to port 0, read the OS-assigned port, drop the listener. There's a
/// small TOCTOU window before the transport rebinds, but it's the standard
/// pattern for end-to-end socket tests.
fn pick_free_port() -> u16 {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    port
}

fn make_pool(
    specs: &[ArraySpec],
    batch_size: usize,
    num_batches: usize,
) -> (Arc<Store>, Arc<DrainerPool>) {
    let capacity = batch_size * num_batches;
    let store = Arc::new(Store::new(
        specs.to_vec(),
        batch_size,
        num_batches,
        Box::new(FifoSampler::new(batch_size, capacity)),
        Box::new(FifoRemover::new()),
    ));
    let metrics = Metrics::new(1);
    let pool = Arc::new(DrainerPool::new(store.clone(), 1, 16, metrics));
    (store, pool)
}

async fn connect_with_retry(port: u16) -> TcpStream {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match TcpStream::connect(("127.0.0.1", port)).await {
            Ok(s) => return s,
            Err(_) if Instant::now() < deadline => {
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
            Err(e) => panic!("connect to 127.0.0.1:{port} failed: {e}"),
        }
    }
}

async fn read_exact(stream: &mut TcpStream, n: usize) -> Vec<u8> {
    let mut buf = vec![0u8; n];
    stream.read_exact(&mut buf).await.unwrap();
    buf
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

fn read_view(view: &ConsumerView) -> Vec<u8> {
    let (ptr, len) = view.arrays[0];
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len).to_vec() }
}

/// Both `TcpTransport` and `DrainerPool` own `tokio::runtime::Runtime`s.
/// Dropping a runtime from inside an async context panics — and that applies
/// even to `TcpTransport::shutdown`, which sets `state = None` and so drops
/// the inner runtime. Always tear both down through this helper.
async fn cleanup(transport: TcpTransport, pool: Arc<DrainerPool>) {
    tokio::task::spawn_blocking(move || {
        transport.shutdown();
        drop(transport);
        drop(pool);
    })
    .await
    .unwrap();
}

// ---------------------------------------------------------------------------
// End-to-end: TcpTransport over a real socket.
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn test_server_sends_handshake_on_connect() {
    let specs = vec![ArraySpec::new(vec![4], 1)];
    let (_store, pool) = make_pool(&specs, 4, 2);
    let port = pick_free_port();
    let transport = TcpTransport::new(1, pool.clone(), specs.clone(), port);
    transport.start().unwrap();

    let mut stream = connect_with_retry(port).await;
    let expected = encode_handshake(&specs);
    let got = read_exact(&mut stream, expected.len()).await;
    assert_eq!(
        got, expected,
        "handshake bytes should match encode_handshake"
    );

    drop(stream);
    pool.shutdown();
    cleanup(transport, pool).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_server_receives_sample_and_acks() {
    let payload = 4usize;
    let specs = vec![ArraySpec::new(vec![payload], 1)];
    let (store, pool) = make_pool(&specs, 4, 2);
    let port = pick_free_port();
    let transport = TcpTransport::new(1, pool.clone(), specs.clone(), port);
    transport.start().unwrap();

    let mut stream = connect_with_retry(port).await;
    let hs = encode_handshake(&specs);
    let _ = read_exact(&mut stream, hs.len()).await;

    // Send one batch worth of samples; expect a single-byte ack per sample.
    for i in 0u8..4 {
        let sample = vec![i, i.wrapping_add(1), i.wrapping_add(2), i.wrapping_add(3)];
        stream.write_all(&sample).await.unwrap();
        let ack = read_exact(&mut stream, 1).await;
        assert_eq!(ack, vec![0x01], "server should ack every sample with 0x01");
    }
    stream.shutdown().await.unwrap();

    let view = wait_for_batch(&store, Duration::from_secs(5))
        .await
        .expect("batch should land within 5s");
    assert_eq!(view.count, 4);
    let bytes = read_view(&view);
    let expected: Vec<u8> = (0u8..4)
        .flat_map(|i| vec![i, i.wrapping_add(1), i.wrapping_add(2), i.wrapping_add(3)])
        .collect();
    assert_eq!(bytes, expected);
    drop(view);

    pool.shutdown();
    cleanup(transport, pool).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_server_handles_concurrent_connections() {
    let payload = 2usize;
    let specs = vec![ArraySpec::new(vec![payload], 1)];
    let batch_size = 4;
    let (store, pool) = make_pool(&specs, batch_size, 4);
    let port = pick_free_port();
    let transport = TcpTransport::new(2, pool.clone(), specs.clone(), port);
    transport.start().unwrap();

    let num_clients: u8 = 4;
    let samples_per_client: u8 = 2;
    let total = (num_clients as usize) * (samples_per_client as usize);

    let mut handles = Vec::new();
    for client in 0..num_clients {
        let specs = specs.clone();
        handles.push(tokio::spawn(async move {
            let mut stream = connect_with_retry(port).await;
            let hs = encode_handshake(&specs);
            let _ = read_exact(&mut stream, hs.len()).await;
            for i in 0..samples_per_client {
                let sample = vec![client, i];
                stream.write_all(&sample).await.unwrap();
                let _ack = read_exact(&mut stream, 1).await;
            }
            stream.shutdown().await.unwrap();
        }));
    }
    for h in handles {
        h.await.unwrap();
    }

    // Drain every committed batch and verify the multiset of (client, i)
    // matches what each client sent. Order isn't observable across clients.
    let mut seen: Vec<(u8, u8)> = Vec::with_capacity(total);
    let deadline = Instant::now() + Duration::from_secs(5);
    while seen.len() < total && Instant::now() < deadline {
        if let Some(v) = store.try_sample() {
            let bytes = read_view(&v);
            for chunk in bytes.chunks(payload) {
                seen.push((chunk[0], chunk[1]));
            }
        } else {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
    seen.sort();
    let mut expected: Vec<(u8, u8)> = (0..num_clients)
        .flat_map(|c| (0..samples_per_client).map(move |i| (c, i)))
        .collect();
    expected.sort();
    assert_eq!(seen, expected, "every sample from every client should land");

    pool.shutdown();
    cleanup(transport, pool).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_server_double_start_fails() {
    let specs = vec![ArraySpec::new(vec![4], 1)];
    let (_store, pool) = make_pool(&specs, 4, 2);
    let port = pick_free_port();
    let transport = TcpTransport::new(1, pool.clone(), specs, port);
    transport.start().unwrap();
    assert!(
        transport.start().is_err(),
        "second start should fail while the first is still running",
    );
    pool.shutdown();
    cleanup(transport, pool).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn test_server_clean_disconnect_between_frames() {
    // Client receives the handshake and disconnects without sending a
    // sample. The server's `read_exact` returns UnexpectedEof, which is
    // mapped to Ok(()) — a clean shutdown, not an error. Verified
    // indirectly: a follow-up connection still completes the handshake.
    let specs = vec![ArraySpec::new(vec![4], 1)];
    let (_store, pool) = make_pool(&specs, 4, 2);
    let port = pick_free_port();
    let transport = TcpTransport::new(1, pool.clone(), specs.clone(), port);
    transport.start().unwrap();

    let mut s1 = connect_with_retry(port).await;
    let hs = encode_handshake(&specs);
    let _ = read_exact(&mut s1, hs.len()).await;
    drop(s1);

    let mut s2 = connect_with_retry(port).await;
    let got = read_exact(&mut s2, hs.len()).await;
    assert_eq!(
        got, hs,
        "server still accepting after a clean mid-frame disconnect"
    );

    drop(s2);
    pool.shutdown();
    cleanup(transport, pool).await;
}
