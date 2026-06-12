# Strata: Snapshot-aware serving layer for Iceberg tables
#
# Build: DOCKER_BUILDKIT=1 docker build -t strata .
#
# Run (recommended - persists cache and metadata across restarts):
#   docker run --rm -p 8765:8765 \
#     -v strata_state:/home/strata/.strata \
#     -v /path/to/warehouse:/data \
#     strata
#
# Volumes:
#   /home/strata/.strata  - State directory (cache + metadata + uv cache)
#   /data                 - Mount your Iceberg warehouse here
#
# Without the named volume, cache is lost on container restart!
#
# Multi-stage build:
# 1. Builder: uses uv + Rust to build the wheel (with BuildKit caching)
# 2. Runtime: minimal image with just the wheel installed
#
# syntax=docker/dockerfile:1

# Pinned by digest for reproducible, supply-chain-safe builds (Scorecard
# Pinned-Dependencies). Bump the tag and the digest together; resolve a new
# digest with `docker buildx imagetools inspect <image:tag>`.
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.11.3-python3.13-trixie-slim@sha256:82f018bb3bd8b1d12c376c3e87da186ec1932cbf91bc8e73089feea6428fec00

# =============================================================================
# Stage 1a: Frontend Builder (Node.js)
# =============================================================================
FROM node:26-alpine@sha256:3ad34ca6292aec4a91d8ddeb9229e29d9c2f689efd0dd242860889ac71842eba AS frontend-builder
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

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust with pinned toolchain for reproducible builds
ARG RUST_VERSION=1.92.0
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain ${RUST_VERSION}
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy only the files needed to build the backend wheel. This keeps frontend-
# only edits from invalidating the Python/Rust build cache.
WORKDIR /build
COPY LICENSE README.md pyproject.toml uv.lock ./
COPY src ./src
COPY rust ./rust
# strata-client is a workspace member (dev convenience) but NOT a runtime
# dependency of the server, so the image never installs it. Copied only so the
# workspace resolves during `uv export` / `uv build`.
COPY packages ./packages

# Export the exact runtime dependency set from uv.lock. The final image installs
# these first, then installs the built wheel with --no-deps to avoid resolving
# the project's dependencies again at image build time.
#
# Extras included in the image:
#   --extra otel: OpenTelemetry SDK + OTLP exporter. The fly.toml sets
#     OTEL_* env vars to wire spans to a collector; without the otel
#     extra installed, those env vars would be silently ignored.
RUN mkdir -p dist && \
    uv export \
      --frozen \
      --no-dev \
      --no-emit-workspace \
      --no-editable \
      --no-header \
      --no-annotate \
      --extra otel \
      --format requirements.txt \
      --output-file dist/runtime-requirements.txt

# Build the server wheel (maturin/Rust → cp313). Only the root package — the
# server image doesn't include strata-client. Pin the build interpreter so
# maturin emits a cp313 wheel that matches the runtime image.
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

# Install into a uv-managed venv. Strata refuses to start outside one
# (see src/strata/_uv_runtime.py); ``--system`` would leave the
# container's Python without the ``uv = <version>`` marker that the
# startup guard reads from pyvenv.cfg. ``uv venv`` writes that marker;
# subsequent ``uv pip install`` calls without ``--system`` target the
# active venv via $VIRTUAL_ENV.
ENV VIRTUAL_ENV=/opt/strata-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

# Copy and install the pinned runtime dependency set first, then the wheel.
# This keeps image installs aligned with uv.lock and avoids re-resolving the
# wheel's dependencies during the final install step.
COPY --from=builder /build/dist/runtime-requirements.txt /tmp/
COPY --from=builder /build/dist/*.whl /tmp/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv $VIRTUAL_ENV && \
    uv pip install -r /tmp/runtime-requirements.txt && \
    uv pip install --no-deps /tmp/*.whl && \
    rm /tmp/*.whl /tmp/runtime-requirements.txt

# Copy the built frontend
COPY --from=frontend-builder /build/frontend/dist /home/strata/frontend/dist

# Create non-root user
RUN useradd --create-home --shell /bin/bash strata

# Create directories for cache, notebook storage, and data (as root, then chown)
RUN mkdir -p /home/strata/.strata/cache /home/strata/.strata/uv-cache /tmp/strata-notebooks /data && \
    chown -R strata:strata /home/strata /tmp/strata-notebooks /data

# Switch to non-root user
USER strata
WORKDIR /home/strata

# Declare volumes for persistence across container restarts
# - /home/strata/.strata: State directory (cache + meta.sqlite + uv cache)
# - /data: Mount point for local warehouse data
VOLUME ["/home/strata/.strata", "/data"]

# Python runtime settings
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Strata configuration defaults (can be overridden).
#
# The container is fundamentally a multi-user shape: ``docker run -p
# 8765:8765`` only works when the server binds non-loopback. Set the
# personal-mode "allow remote clients" flag so the default
# personal-mode deployment doesn't refuse to bind to 0.0.0.0.
# Service-mode operators override STRATA_DEPLOYMENT_MODE at runtime;
# this set of defaults gets them to a working container immediately
# without an extra env var.
ENV UV_CACHE_DIR=/home/strata/.strata/uv-cache
ENV UV_PYTHON_DOWNLOADS=never
ENV STRATA_HOST=0.0.0.0
ENV STRATA_PORT=8765
ENV STRATA_DEPLOYMENT_MODE=personal
ENV STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL=true
ENV STRATA_CACHE_DIR=/home/strata/.strata/cache
ENV STRATA_METADATA_DB=/home/strata/.strata/meta.sqlite

# Health check (uses stdlib to avoid extra dependencies)
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health').read()"

# Expose the default port
EXPOSE 8765

# Run the server (python -m is more robust for K8s than console scripts)
CMD ["python", "-m", "strata"]
