# Clients

echo ships one client, `TcpClient`, paired with the built-in
`TcpTransport`. If you plug a custom transport (gRPC, RDMA, …) into the
Rust `Transport` trait, you'll write a matching client to go with it.

## TcpClient

::: echo.tcp_client.TcpClient

::: echo.tcp_client.ConnectionClosedError
