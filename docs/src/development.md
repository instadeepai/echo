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

## Benchmarks

```bash
just bench               # ray-based, requires uv sync --extra distributed
```

Benchmark results land in `benches/`.

## Docs

```bash
uv run --extra docs mkdocs serve   # live-reload on http://127.0.0.1:8000
uv run --extra docs mkdocs build   # static site to site/
```

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
