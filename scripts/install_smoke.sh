#!/usr/bin/env bash
# Smoke-test an installed Strata: the server answers /health, and a notebook gets a
# package and runs a cell that imports it, once in the CLI's own process and once
# through the server.
#
#   scripts/install_smoke.sh <bin dir of the install> [expected uv version]
#
# Strata runs with no uv on PATH and a fresh HOME (no ~/.local/bin/uv), so the
# notebook's environment is built by the uv installed with it. With a second
# argument, the uv that built it must be that version.
set -euo pipefail

BIN=$(cd "$1" && pwd)
EXPECT_UV=${2:-}
WORK=$(cd "$(mktemp -d)" && pwd -P)
PORT=${STRATA_PORT:-8767}
URL="http://127.0.0.1:${PORT}"
# PATH without uv, and without the install's own bin dir: every command is called by path.
export PATH=/usr/bin:/bin
export HOME="$WORK/home"
mkdir -p "$HOME"
export STRATA_PORT=$PORT
# The server opens notebooks only under its storage dir, where this one is made.
export STRATA_NOTEBOOK_STORAGE_DIR="$WORK/storage"
mkdir -p "$STRATA_NOTEBOOK_STORAGE_DIR"
unset STRATA_UV VIRTUAL_ENV

if command -v uv >/dev/null; then
  echo "::error::uv is on PATH ($(command -v uv)); the smoke test needs a PATH without it"
  exit 1
fi

"$BIN/strata" --help >/dev/null
"$BIN/strata-worker" --help >/dev/null
echo "strata and strata-worker --help ok"

"$BIN/strata-notebook" >"$WORK/server.log" 2>&1 &
SERVER_PID=$!
# WORK holds the fresh HOME's uv cache (~300 MB); uv leaves some of it read-only.
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true
      chmod -R u+w "${WORK:?}"; rm -rf "${WORK:?}"' EXIT
for _ in $(seq 1 60); do
  curl -fsS "$URL/health" >/dev/null 2>&1 && break
  kill -0 "$SERVER_PID" 2>/dev/null || break  # it exited: say why now, not in 60s
  sleep 1
done
curl -fsS "$URL/health" >/dev/null || {
  echo "::error::the server did not answer /health"
  cat "$WORK/server.log"
  exit 1
}
echo "server answered /health"

# Each cell imports its package at the pinned version, so a status of ok proves it.
all_ok() {
  "$BIN/python" - "$1" <<'PY' || { cat "$1"; exit 1; }
import json, sys
data = json.load(open(sys.argv[1]))
cells = data["cells"] if "cells" in data else [data]
assert cells and all(c["status"] == "ok" for c in cells), "a cell did not run ok"
PY
}

NB="$STRATA_NOTEBOOK_STORAGE_DIR/smoke"
"$BIN/strata" new smoke --parent "$STRATA_NOTEBOOK_STORAGE_DIR" --no-git
"$BIN/strata" dep add "$NB" "six==1.17.0"
"$BIN/strata" cell add "$NB" -c 'import six
assert six.__version__ == "1.17.0", six.__version__'
"$BIN/strata" run "$NB" --format json >"$WORK/run.json" || {
  echo "::error::strata run failed"
  cat "$WORK/run.json"
  exit 1
}
all_ok "$WORK/run.json"
echo "the CLI added six and ran a cell importing it"

SESSION=$(curl -fsS -X POST "$URL/v1/notebooks/open" -H 'content-type: application/json' \
  -d "{\"path\": \"$NB\"}" | "$BIN/python" -c 'import json, sys; print(json.load(sys.stdin)["session_id"])')
ON_SERVER=(--server "$URL" --session "$SESSION")
"$BIN/strata" dep add "${ON_SERVER[@]}" "iniconfig==2.1.0"
CELL=$("$BIN/strata" cell add "${ON_SERVER[@]}" --format json -c 'import iniconfig
from importlib.metadata import version
assert version("iniconfig") == "2.1.0", version("iniconfig")' |
  "$BIN/python" -c 'import json, sys; print(json.load(sys.stdin)["id"])')
"$BIN/strata" cell run "${ON_SERVER[@]}" --format json "$CELL" >"$WORK/cell.json" || {
  echo "::error::the server's cell run failed"
  cat "$WORK/cell.json" "$WORK/server.log"
  exit 1
}
all_ok "$WORK/cell.json"
echo "the server added iniconfig and ran a cell importing it"

if [ -n "$EXPECT_UV" ]; then
  # The uv that built the notebook's environment records itself in its pyvenv.cfg.
  built_by=$(sed -n 's/^uv = //p' "$NB/.venv/pyvenv.cfg")
  if [ "$built_by" != "$EXPECT_UV" ]; then
    echo "::error::the notebook's environment was built by uv '$built_by', not $EXPECT_UV"
    exit 1
  fi
  echo "the notebook's environment was built by uv $built_by"
fi
echo "install smoke test passed"
