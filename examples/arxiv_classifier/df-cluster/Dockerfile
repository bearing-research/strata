# Template: Strata worker image for Fly.io (CPU).
#
# Builds an image that runs `strata-worker`, an HTTP endpoint that
# accepts cell execution requests. Cells run in this container's Python
# environment, so pip-install whatever your cells import (pandas,
# datafusion, scikit-learn, ...).
#
# Deploy:
#   flyctl apps create strata-df-cluster   # first time only
#   flyctl secrets set STRATA_WORKER_TOKEN=<token>
#   flyctl deploy --remote-only
#
# The image refuses to start without STRATA_WORKER_TOKEN. Export the same
# token as STRATA_DF_CLUSTER_TOKEN wherever strata-notebook runs:
# notebook.toml's df-cluster worker names it as its token_env.

# Digest-pinned for reproducible builds (Scorecard Pinned-Dependencies).
# Bump tag + digest together via `docker buildx imagetools inspect`.
FROM python:3.14-slim@sha256:cea0e6040540fb2b965b6e7fb5ffa00871e632eef63719f0ea54bca189ce14a6

# The worker must be the same release as the server that dispatches to it.
# Plain pip, not a uv venv: the uv-runtime guard gates strata-notebook but
# not the worker entry point.
ARG STRATA_VERSION=0.8.0
RUN pip install --no-cache-dir \
    "strata-notebook[notebook]==${STRATA_VERSION}" \
    "datafusion>=42" \
    "pandas>=2" \
    "pyarrow>=18"

# Cells run arbitrary user code, so not as root.
RUN useradd --create-home --shell /bin/bash worker
USER worker
WORKDIR /home/worker

EXPOSE 8080

# Binding 0.0.0.0 without a token would leave /execute open to anyone.
CMD ["sh", "-c", "\
if [ -z \"${STRATA_WORKER_TOKEN:-}\" ]; then \
  echo 'refusing to start: STRATA_WORKER_TOKEN is unset, and this image binds' >&2; \
  echo '0.0.0.0, so an unauthenticated /execute is remote code execution.' >&2; \
  exit 1; \
fi; \
exec strata-worker --host 0.0.0.0 --port 8080"]
