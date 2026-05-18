# Transports

`TcpTransport` is a PyO3 class. It only configures the Rust-side server;
an instance is passed to `Server(transport=...)` and the class itself has
no methods to call directly.

For the user-facing tradeoff and the extension story for custom
transports (gRPC, RDMA, …), see [Transports](../guides/transports.md).

## TcpTransport

```python
from echo import TcpTransport

transport = TcpTransport(port=50051, num_threads=8)
```

**Parameters**

- `port: int`: bind port on the server side.
- `num_threads: int = 8`: worker threads used to accept connections and
  push samples into SPSC queues.
