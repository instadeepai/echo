//! TCP transport with buffered reads.
//!
//! Reads each sample from the network into a temporary buffer, then
//! pushes it into a per-connection SPSC queue. A drainer pool pulls
//! from these queues and inserts into the Store.

use std::sync::Arc;

use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::array_spec::ArraySpec;
use crate::ingress::DrainerPool;

/// Wire-format identifier: clients verify these bytes before parsing
/// anything else, so an unrelated server (or a future incompatible echo)
/// surfaces as a clear handshake error.
pub const PROTOCOL_MAGIC: [u8; 4] = *b"ECHO";
pub const PROTOCOL_VERSION: u8 = 1;

/// Encode the handshake message (magic + version + array specs) into bytes.
/// Format: [4 magic] [u8 version] [u32 num_arrays] then per array
/// [u32 dtype_size] [u32 ndim] [u32 x ndim shape]
pub fn encode_handshake(specs: &[ArraySpec]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&PROTOCOL_MAGIC);
    buf.push(PROTOCOL_VERSION);
    buf.extend_from_slice(&(specs.len() as u32).to_le_bytes());
    for spec in specs {
        buf.extend_from_slice(&(spec.dtype_size as u32).to_le_bytes());
        buf.extend_from_slice(&(spec.shape.len() as u32).to_le_bytes());
        for &dim in &spec.shape {
            buf.extend_from_slice(&(dim as u32).to_le_bytes());
        }
    }
    buf
}

struct State {
    _runtime: tokio::runtime::Runtime,
    shutdown_tx: Option<tokio::sync::oneshot::Sender<()>>,
    /// What the listener actually bound to. Equals `requested_port` unless
    /// that was 0, in which case the OS chose it.
    port: u16,
}

pub struct TcpTransport {
    num_threads: usize,
    drainer_pool: Arc<DrainerPool>,
    specs: Vec<ArraySpec>,
    requested_port: u16,
    state: Mutex<Option<State>>,
}

impl TcpTransport {
    pub fn new(
        num_threads: usize,
        drainer_pool: Arc<DrainerPool>,
        specs: Vec<ArraySpec>,
        port: u16,
    ) -> Self {
        Self {
            num_threads,
            drainer_pool,
            specs,
            requested_port: port,
            state: Mutex::new(None),
        }
    }
}

impl super::Transport for TcpTransport {
    fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut state = self.state.lock();
        if state.is_some() {
            return Err("TCP server already started".into());
        }

        // Bind here rather than inside the spawned task: a `start` that
        // returns then means the port is bound and backlogging connections,
        // and `requested_port = 0` has an OS-assigned port to report.
        let addr: std::net::SocketAddr = ([0, 0, 0, 0], self.requested_port).into();
        let listener = std::net::TcpListener::bind(addr)?;
        listener.set_nonblocking(true)?;
        let port = listener.local_addr()?.port();

        // Bound transport threads so the drainer pool has CPU left.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(self.num_threads)
            .max_blocking_threads(self.num_threads)
            .enable_all()
            .thread_name("tcp")
            .build()?;
        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let drainer_pool = self.drainer_pool.clone();
        let specs = self.specs.clone();

        rt.spawn(async move {
            if let Err(e) = run_server(drainer_pool, specs, listener, shutdown_rx).await {
                eprintln!("TCP server error: {e}");
            }
        });

        *state = Some(State {
            _runtime: rt,
            shutdown_tx: Some(shutdown_tx),
            port,
        });
        Ok(())
    }

    fn shutdown(&self) {
        let mut state = self.state.lock();
        if let Some(ref mut s) = *state {
            if let Some(tx) = s.shutdown_tx.take() {
                let _ = tx.send(());
            }
        }
        *state = None;
    }

    fn port(&self) -> Option<u16> {
        self.state.lock().as_ref().map(|s| s.port)
    }
}

async fn run_server(
    drainer_pool: Arc<DrainerPool>,
    specs: Vec<ArraySpec>,
    listener: std::net::TcpListener,
    shutdown_rx: tokio::sync::oneshot::Receiver<()>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let handshake = encode_handshake(&specs);
    let payload_size: usize = specs.iter().map(|s| s.num_bytes()).sum();
    // Registers with this runtime's reactor, so it has to happen in-task.
    let listener = TcpListener::from_std(listener)?;

    tokio::select! {
        _ = async {
            loop {
                match listener.accept().await {
                    Ok((stream, peer)) => {
                        // Each connection gets its own SPSC queue.
                        let sender = drainer_pool.new_sender();
                        let handshake = handshake.clone();
                        tokio::spawn(async move {
                            if let Err(e) = handle_connection(
                                stream, sender, &handshake, payload_size,
                            ).await {
                                eprintln!("TCP connection error from {peer}: {e}");
                            }
                        });
                    }
                    Err(e) => {
                        eprintln!("TCP accept error: {e}");
                    }
                }
            }
        } => {}
        _ = shutdown_rx => {}
    }

    Ok(())
}

async fn handle_connection(
    mut stream: tokio::net::TcpStream,
    sender: crate::ingress::SampleSender,
    handshake: &[u8],
    payload_size: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    stream.set_nodelay(true)?;

    // Send handshake
    stream.write_all(handshake).await?;

    loop {
        // Recycled buffer from the per-connection free pool (allocates only
        // on initial fill-up).
        let mut buf = sender.acquire(payload_size);
        // Clean EOF between frames is a normal disconnect, not an error.
        match stream.read_exact(&mut buf).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.into()),
        }

        sender.push(buf).await;

        // Send 1-byte ack
        stream.write_all(&[0x01]).await?;
    }
}
