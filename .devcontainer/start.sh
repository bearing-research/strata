#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

SERVER_LOG="/tmp/strata-devcontainer.log"
SERVER_URL="http://127.0.0.1:8765/health"

# Boot the server if it isn't already responding. The default storage
# dir is ``~/.strata/notebooks`` (per strata.config), which Strata
# creates on demand — no mkdir needed here.
if ! curl -fsS "$SERVER_URL" >/dev/null 2>&1; then
    nohup env \
        STRATA_DEPLOYMENT_MODE=personal \
        strata-notebook \
        >"$SERVER_LOG" 2>&1 &
fi

for _ in $(seq 1 30); do
    if curl -fsS "$SERVER_URL" >/dev/null 2>&1; then
        exit 0
    fi
    sleep 1
done

echo "Strata server did not become ready; see $SERVER_LOG" >&2
exit 1
