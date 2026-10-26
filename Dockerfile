# syntax=docker/dockerfile:1
# Strata: Snapshot-aware serving layer for Iceberg tables
#
# Build: DOCKER_BUILDKIT=1 docker build -t strata .
#
# Run (the named volume keeps cache and metadata across restarts):
#   docker run --rm -p 127.0.0.1:8765:8765 \
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
FROM node:26-alpine@sha256:40cd011b55415bc894e45b02fdd5a5dfde056921b7d23a64c74f80497af48797 AS frontend-builder
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY frontend/ ./
RUN npm run build

# =============================================================================
# Stage 1b: Backend Builder (uv)
# =============================================================================
FROM ${UV_IMAGE} AS builder
ENV UV_PYTHON=3.13

# Copy only what the backend wheel needs, so frontend edits keep the build cache.
WORKDIR /build
COPY LICENSE README.md pyproject.toml uv.lock ./
COPY src ./src
# vestigo is the ledger strata depends on; strata-client and
# strata-pool are workspace members copied only so the workspace resolves.
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
#   --extra mcp: without it, STRATA_MCP_ENABLED only logs that /mcp is off.
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
      --extra mcp \
      --format requirements.txt \
      --output-file dist/runtime-requirements.txt

# Build the root package and the ledger (pure Python, py3-none-any wheels).
# --no-emit-workspace left vestigo out of the requirements above.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir dist && \
    uv build --wheel --package vestigo --out-dir dist

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
