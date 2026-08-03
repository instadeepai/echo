# Development

## Build modes

| Command | Profile | Detailed metrics |
|---|---|---|
| `uv sync --extra dev` | dev (fast incremental) | off |
| `just install` | dev | off |
| `just develop` | dev | on |
| `just install-telemetry` | release | on |
| `just build-whl` | release | off |

`detailed-metrics` adds three hdrhistograms (memcpy, drain_round,
queue_dwell) and CAS counters. Off by default; turn on for perf
investigation. See [Reading metrics](guides/metrics.md) for what they
give you.

## Tests

```bash
cargo test                       # Rust unit + integration tests
uv run pytest python/tests/ -v   # Python tests
```

Rust tests live in `tests/`, one file per module — including
`tests/host_pinning.rs`. Python tests live in `python/tests/`.

### Working on host-memory pinning

See [Host-memory pinning](design/host-pinning.md) for how the module is laid out,
and the [guide](guides/host-memory-pinning.md) for what it does for a user.

Most of `tests/host_pinning.rs` needs no GPU: the resolution ladder is pure, and
registration and rollback run against injected stubs. The tests that do need a
device print a skip notice and pass when there isn't one, since Rust has no native
test skip. On the Python side the equivalent tests carry a `gpu` marker (registered
in `pyproject.toml`) and `pytest` skips them automatically. Either way CI needs no
per-runner configuration.

To exercise the CUDA-runtime resolution ladder locally, install a runtime wheel
into the checkout's venv:

```bash
uv pip install nvidia-cuda-runtime      # CUDA 13; use nvidia-cuda-runtime-cu12 for CUDA 12
```

Without it only rung 1 (already-mapped libraries) can hit, and on a machine with no
system CUDA install nothing resolves at all. The wheel is *not* a `dev` extra:
CI is CPU-only and should not download a CUDA runtime. Note that
`uv run` re-syncs the venv, so re-run the install if it disappears.

`cargo test` has no Python interpreter to ask for the wheel's location, so the
Rust tests look under `.venv/lib/python*/site-packages/nvidia` in the checkout.

## Benchmarks

```bash
just bench               # ray-based, requires uv sync --extra distributed
```

Benchmark results land in `benches/`.

## Docs

```bash
just docs-serve   # live-reload on http://127.0.0.1:8000/echo/
just docs         # static site to docs/site/
```

Note the **`/echo/` path prefix** — `mkdocs serve` honours `site_url`, so
`http://127.0.0.1:8000/` alone 404s. The address it prints on startup is correct.

Working on a remote box, `docs-serve` binds to loopback only and is invisible from
your laptop. Either forward the port from the client side, which needs no change
here:

```bash
ssh -L 8000:localhost:8000 you@remote-box    # then browse http://localhost:8000/echo/
```

or bind to an interface the client can reach:

```bash
just docs-serve-on                  # 0.0.0.0:8000 — all interfaces
just docs-serve-on 10.0.0.5:8000    # just this one
```

Prefer the tunnel, or a specific private interface, over `0.0.0.0` on an untrusted
network: `mkdocs serve` is a development server with no authentication.

The docs are built and deployed by `.github/workflows/docs.yml` on every
push to `main`.

The `mkdocstrings` plugin pulls signatures and docstrings live from
`python/echo/`. Edit a docstring there and `mkdocs serve` will
hot-reload the corresponding API page.

## Just recipes

`just` with no args lists every recipe. The commonly-used ones:

- `just install`: editable install in dev profile.
- `just develop`: dev build with `detailed-metrics`, for perf work.
- `just build-whl`: manylinux release wheel.
- `just bench`: benchmarks.
- `just docs-serve`: live-reload docs.
- `just docs`: build static docs.
