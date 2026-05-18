/// Transport layer for receiving samples from remote actors.
///
/// echo ships a single TCP transport, but the `Transport` trait is the
/// extension point for custom transports (gRPC, RDMA, shared memory, …):
/// implement `start`/`shutdown`, push received payloads into a
/// `DrainerPool` sender, and the ring-buffer side is unchanged.
pub mod tcp;

pub use tcp::TcpTransport;

pub trait Transport: Send + Sync {
    fn start(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    fn shutdown(&self);
}
