# Server

The high-level server. Handles pytree flattening on `submit()`, unflattening
on `sample()`, and owns the underlying Rust `_Server`.

The `example` pytree passed to `Server` defines the shape and dtype of every
leaf, and the server will only accept samples that match it exactly. The
matching client must be constructed with a pytree of the **same structure
and per-leaf shape/dtype**, and every `client.send(...)` call must pass a
pytree with those exact shapes too. Mismatches are rejected at handshake
(by the client) or produce a hard size error inside `send()`.

::: echo.server.Server

::: echo.server.Sample
