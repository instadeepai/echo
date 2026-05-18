# echo benchmark runner

# Default recipe: list available targets
default:
    @just --list

# Run distributed benchmark (requires ray: uv sync --extra distributed)
bench:
    uv run --extra distributed benches/bench_distributed.py

# Build a manylinux wheel (default python3.10, override with: just build-whl python3.12)
build-whl PYTHON="python3.10":
    maturin build --release -i {{PYTHON}}

# Install the lean build from the current checkout (editable, dev profile)
install:
    uv pip install --force-reinstall -e .

# Install from the current checkout with detailed-metrics (release profile)
install-telemetry:
    uv pip install --force-reinstall -e . \
      --config-settings=build-args='--release --features detailed-metrics'

# Install telemetry build from an origin branch (default main). Good for teammates without a local checkout
install-telemetry-from-git BRANCH="main":
    pip install --force-reinstall \
      "id-echo @ git+ssh://git@github.com/instadeepai/echo.git@{{BRANCH}}" \
      --config-settings=build-args='--release --features detailed-metrics'

# Build the current worktree into the active venv (fast local dev, no docker)
develop:
    uv run maturin develop --features detailed-metrics

# Live-reload docs on http://127.0.0.1:8000
docs-serve:
    uv run --extra docs mkdocs serve

# Build static docs into site/
docs:
    uv run --extra docs mkdocs build

