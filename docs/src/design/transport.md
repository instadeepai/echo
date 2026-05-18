# Transport

`src/transport/` is the server-side network layer. echo ships one
implementation, `tcp.rs`, behind the `Transport` trait in `mod.rs`. The
trait is the extension point for custom transports (gRPC, RDMA, shared
memory, …): drop in a new module that implements `start`/`shutdown` and
pushes received bytes into a `DrainerPool` sender, and the ingress /
store / sampler path stays the same.

## Trait shape

`Transport` is intentionally thin: just `start` and `shutdown`. Each
implementation owns its own tokio runtime and accept loop; the `Store`
doesn't know which one it's running.

## Handshake

The TCP transport sends the array spec (shapes + dtype sizes) to the
client on connection. The client validates it against its own
`data_sample` and closes the connection on mismatch. After the
handshake, the wire format is just concatenated raw bytes, one
fixed-size message per sample.

The handshake serves two purposes:

1. Validating shapes up front, so misaligned bytes are caught at connect
   time rather than silently memcpy'd into the ring as garbage.
2. Avoiding shape metadata per sample. After the handshake, there is
   zero per-sample framing on the wire.

A custom transport is free to handshake however it likes (gRPC could
send the spec as the first server-to-client message, for instance), as
long as it ends up calling `sender.push(bytes)` with payloads matching
the agreed-upon spec.

## TCP

`tcp.rs` is a stdlib-style accept loop on a tokio listener. For each
accepted socket it gets a `SampleSender` from the `DrainerPool`, sends
the handshake, then reads `payload_size` bytes at a time and calls
`sender.push(bytes).await`. After every successful push it sends a
1-byte ack back so the client's bounded semaphore can release a slot.

Frame format on the wire after handshake: nothing. Just `payload_size`
bytes per sample, where `payload_size` is `sum(prod(shape) * dtype_size)`
across the flattened pytree.

## Ack pacing

The TCP transport sends a 1-byte ack after the sample makes it onto the
SPSC queue (not after it lands in the ring). The client uses these acks
to drive a `BoundedSemaphore` capped at `max_inflight_msgs`. Send to the
server, wait for an ack before sending another N. That gives both ends
explicit backpressure without any rate-limiting heuristics.

Note: the ack fires after **push to SPSC**, not after the drainer has
committed to the ring. So the client's view of "in flight" lags the
truth, but only by the depth of the SPSC queue. The actual end-to-end
backpressure is the chain of SPSC depth + ring depth + consumer rate,
which is exactly what the metrics expose.
