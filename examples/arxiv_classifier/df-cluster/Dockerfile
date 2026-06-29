# Template: Strata worker image for Fly.io (CPU).
#
# Builds an image that runs `strata-worker` — an HTTP endpoint that
# accepts cell execution requests. Cells run in this container's Python
# environment, so pip-install whatever your cells import (pandas,
# datafusion, scikit-learn, ...).
#
# Deploy:
#   flyctl apps create strata-df-cluster   # first time only
#   flyctl deploy --remote-only
#
# Fly prints the URL — paste it into notebook.toml as a [[workers]]
# entry with transport = "http" and url = "<fly-url>/v1/execute".

# Digest-pinned for reproducible builds (Scorecard Pinned-Dependencies).
# Bump tag + digest together via `docker buildx imagetools inspect`.
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1 AS builder

# Rust toolchain — required to build Strata's native extension.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git ca-certificates && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:$PATH"

# Install Strata + your workload deps. Pin Strata to a commit or tag
# for reproducible builds; PyPI users will just write "strata".
RUN pip install --no-cache-dir \
    "strata-notebook[notebook] @ git+https://github.com/bearing-research/strata.git@bef5d94" \
    "datafusion>=42" \
    "pandas>=2" \
    "pyarrow>=18"

# Slim runtime — copy the installed packages from the builder stage.
FROM python:3.14-slim@sha256:b877e50bd90de10af8d82c57a022fc2e0dc731c5320d762a27986facfc3355c1

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

EXPOSE 8080
CMD ["strata-worker", "--host", "0.0.0.0", "--port", "8080"]
