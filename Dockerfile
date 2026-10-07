# syntax=docker/dockerfile:1
# Strata: Snapshot-aware serving layer for Iceberg tables
#
# Build: DOCKER_BUILDKIT=1 docker build -t strata .
#
# Run (the named volume keeps cache and metadata across restarts):
#   docker run --rm -p 8765:8765 \
#     -v strata_state:/home/strata/.strata \
#     -v /path/to/warehouse:/data \
#     strata
#
# Volumes:
#   /home/strata/.strata  - State directory (cache + metadata + uv cache)
#   /data                 - Mount your Iceberg warehouse here

# Pinned by digest for supply-chain safety (Scorecard Pinned-Dependencies).
# Bump tag and digest together; get the digest with
# `docker buildx imagetools inspect <image:tag>`.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.3-python3.13-trixie-slim@sha256:82f018bb3bd8b1d12c376c3e87da186ec1932cbf91bc8e73089feea6428fec00

# =============================================================================
# Stage 1a: Frontend Builder (Node.js)
# =============================================================================
FROM node:26-alpine@sha256:0b36e8c136b94cd4fcf02188228e76c31ad5872eef3fec8cbd2eee500cfd9e80 AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 1b: Backend Builder (uv + Rust)
# =============================================================================
FROM ${UV_IMAGE} AS builder
ENV UV_PYTHON=3.13
ENV CARGO_TARGET_DIR=/root/.cargo-target

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

ARG RUST_VERSION=1.92.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy only what the backend wheel needs, so frontend edits keep the
# Python/Rust build cache.
WORKDIR /build
COPY LICENSE README.md pyproject.toml uv.lock ./
COPY src ./src
COPY rust ./rust
# strata-client is a workspace member, not a server dependency; copied only so
# the workspace resolves during `uv export` / `uv build`.
COPY packages ./packages

# Export the exact runtime dependency set from uv.lock; the runtime stage
# installs it, then the wheel with --no-deps, so nothing is re-resolved.
#
# Extras:
#   --extra otel: without it, the OTEL_* env vars fly.example.toml sets are ignored.
#   --extra postgres: lets a deployment point STRATA_ARTIFACT_METADATA_DSN at
#     Postgres (several servers sharing one store) without rebuilding the
#     image. SQLite stays the default.
#   --extra sql, --extra sql-sqlite: SQL cells (DuckDB is a core dependency).
#     Without sql a notebook holding a SQL cell does not open; the SQLite driver
#     is what the shipped SQL example connects with.
RUN mkdir -p dist && \
    uv export \
      --frozen \
      --no-dev \
      --no-emit-workspace \
      --no-editable \
      --no-header \
      --no-annotate \
      --extra otel \
      --extra postgres \
      --extra sql \
      --extra sql-sqlite \
      --format requirements.txt \
      --output-file dist/runtime-requirements.txt

# Build only the root package. Pin the interpreter so maturin emits a cp313
# wheel that matches the runtime image.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=cache,target=/root/.cargo/registry \
    --mount=type=cache,target=/root/.cargo/git \
    --mount=type=cache,target=/root/.cargo-target \
    uv build --wheel --python 3.13 --out-dir dist

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM ${UV_IMAGE} AS runtime
ENV UV_LINK_MODE=copy

# Install into a uv venv: Strata refuses to start without the
# ``uv = <version>`` marker in pyvenv.cfg (src/strata/_uv_runtime.py), which
# ``uv venv`` writes and ``--system`` would not.
ENV VIRTUAL_ENV=/opt/strata-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

COPY --from=builder /build/dist/runtime-requirements.txt /tmp/
COPY --from=builder /build/dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv $VIRTUAL_ENV && \
    uv pip install -r /tmp/runtime-requirements.txt && \
    uv pip install --no-deps /tmp/*.whl && \
    rm /tmp/*.whl /tmp/runtime-requirements.txt

COPY --from=frontend-builder /build/frontend/dist /home/strata/frontend/dist

RUN useradd --create-home --shell /bin/bash strata

RUN mkdir -p /home/strata/.strata/cache /home/strata/.strata/uv-cache /tmp/strata-notebooks /data && \
    chown -R strata:strata /home/strata /tmp/strata-notebooks /data

USER strata
WORKDIR /home/strata

VOLUME ["/home/strata/.strata", "/data"]

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Overridable defaults. ``docker run -p`` needs a non-loopback bind, so allow
# remote clients in personal mode; service-mode operators override
# STRATA_DEPLOYMENT_MODE at runtime.
ENV UV_CACHE_DIR=/home/strata/.strata/uv-cache
ENV UV_PYTHON_DOWNLOADS=never
ENV STRATA_HOST=0.0.0.0
ENV STRATA_PORT=8765
ENV STRATA_DEPLOYMENT_MODE=personal
ENV STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL=true
ENV STRATA_CACHE_DIR=/home/strata/.strata/cache
ENV STRATA_METADATA_DB=/home/strata/.strata/meta.sqlite

# Stdlib only, so no extra dependency.
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:%s/health' % os.environ.get('STRATA_PORT', '8765')).read()"

EXPOSE 8765

# python -m is more robust under K8s than console scripts.
CMD ["python", "-m", "strata"]
