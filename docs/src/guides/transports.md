# Transports

echo ships a single server-side transport (TCP) and a matching `TcpClient`.
The wire format is intentionally minimal: a one-time handshake that
exchanges the array spec (shapes + dtype sizes), then raw concatenated
sample bytes with a one-byte ack per sample.

The Rust `Transport` trait is the extension point for custom transports:
implement `start`/`shutdown`, push received payloads into a
`DrainerPool` sender, and the ring-buffer side is unchanged. gRPC, RDMA,
shared memory, or anything else can be wired in without touching the
ingress / store / sampler path.

## TCP

The TCP transport sends a tiny handshake (4-byte magic `ECHO`, 1-byte
wire version, then one array spec per leaf) and then streams raw bytes
for every sample. There is **no per-message framing** on the wire after
the handshake. The server reads `payload_size` bytes, pushes them into
the connection's SPSC queue, and acks with a single byte.

That's the whole path: no protobuf, no length-prefixing, no metadata per
message. Throughput is governed by the network and the drainer rate.

## Constructor parameters

`TcpTransport` takes:

- `port`: bind port on the server side.
- `num_threads` (default 8): worker threads used by the server to accept
  connections and push into SPSC queues. Bumping this only helps if you
  have many concurrent connections.

`TcpClient` takes `max_inflight_msgs` (default 32), which caps the number
of un-acked sends in flight via a `BoundedSemaphore`.

## Backpressure model

Client sends, server receives, server pushes into the connection's SPSC
queue, drainer pops, drainer commits to the ring buffer. The pre-drainer
SPSC queue is a `crossbeam::ArrayQueue` sized by `producer_queue_size`
(default 8 entries per connection).

Backpressure flows in two stages:

1. **Server-side**: if the SPSC queue is full, the server-side `push` task
   parks on a `Notify` until the drainer pops. `push_blocked_count` is
   incremented every time this happens.
2. **Client-side**: the client's `BoundedSemaphore` caps the number of
   un-acked sends in flight. When the cap is reached, `send()` blocks
   until the server acks an earlier message.

In a healthy run, both queues stay shallow. Sustained `push_blocked_count`
growth means the drainer can't keep up; see
[Reading metrics](metrics.md).

## Graceful shutdown

`TcpClient.send` raises `ConnectionClosedError` (a subclass of
`ConnectionError`) when the server has disconnected. Custom transports
should follow the same convention or document their own shutdown signal.
