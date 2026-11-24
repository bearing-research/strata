#!/usr/bin/env bash
# Launch two Strata notebook HTTP workers on localhost so the arXiv
# classifier demo can dispatch cells to different workers without cloud
# deployment. Each process is just `strata-worker` bound to a different
# port.
#
#   df-cluster → http://127.0.0.1:9000/v1/execute (see notebook.toml)
#   gpu-fly    → http://127.0.0.1:9001/v1/execute
#
# Kill the workers with Ctrl-C (the trap below forwards SIGINT to children).
set -euo pipefail

cleanup() {
    echo
    echo "[workers] stopping..."
    kill "${DF_PID:-}" "${GPU_PID:-}" 2>/dev/null || true
    wait "${DF_PID:-}" "${GPU_PID:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "[workers] starting df-cluster on port 9000 (with datafusion)..."
# df-cluster needs DataFusion's Python bindings installed in its interpreter
# so cells that `from datafusion import SessionContext` can actually run.
# `uv run --with` installs the package into an ephemeral overlay for this
# invocation only, matching how the cloud df-cluster image will ship with
# datafusion pre-installed.
uv run --with datafusion strata-worker --port 9000 --log-level warning &
DF_PID=$!

echo "[workers] starting gpu-fly on port 9001..."
# gpu-fly runs `embed`, which imports sentence-transformers (and with it torch).
# With uv on PATH the worker runs each cell in the notebook's locked environment
# (its uv.lock, written when the server first syncs the notebook), so those come
# from the notebook's pyproject.toml rather than this interpreter.
uv run strata-worker --port 9001 --log-level warning &
GPU_PID=$!

echo "[workers] waiting for health..."
for port in 9000 9001; do
    for _ in $(seq 1 30); do
        if curl -sf "http://127.0.0.1:$port/health" >/dev/null; then
            echo "[workers] port $port healthy"
            break
        fi
        sleep 0.2
    done
done

echo "[workers] ready. Open the arxiv_classifier notebook in Strata and run cells."
echo "[workers] df-cluster pid=$DF_PID, gpu-fly pid=$GPU_PID"
wait
