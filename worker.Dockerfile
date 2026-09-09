# A Strata worker image that satisfies the strata-pool worker contract.
#
# Deliberately not a stage in the root Dockerfile: that one is built with no
# ``--target``, so Docker builds its *last* stage. Appending a worker stage
# there would silently change what ``docker build .`` produces and break the
# server image everyone else builds.
#
# At the repo root rather than in a ``docker/`` directory: a top-level
# directory named ``docker`` makes ruff's isort classify the ``docker`` PyPI
# package as first-party, which silently reorders imports in the integration
# tests that use it.
#
#   docker build -f worker.Dockerfile -t strata-worker:latest .
#
# The pool does not pull images, so build it on the host that will run it (or
# push it to a registry the host has already pulled from).

FROM python:3.14-slim

# Installed with plain pip, not into a uv venv. ``strata-notebook`` refuses to
# start outside one (src/strata/_uv_runtime.py), but the *worker* entry point
# is deliberately not gated by that guard -- which is what lets it run on a
# stock Python image here, and on Modal's standard image stack.
#
# The ``notebook`` extra is not optional for a worker: cells execute through
# harness.py, which needs orjson / cloudpickle / pandas / numpy to serialize
# what they produce. Without it the machine boots, answers /health, accepts a
# job, and fails at the point of doing the work.
# Requires 0.7.0 or newer: ``POST /execute``, the path the pool dispatches to,
# ships in that release. Pinned rather than floating so a machine's worker
# cannot drift from the server that issued its manifest.
#
# This installs from PyPI and uses nothing from the build context, so building
# it inside a checkout does NOT pick up local changes to the worker -- you get
# the published version. To test unreleased worker code, build a wheel
# (``uv build``) and install that instead of this line.
ARG STRATA_VERSION=0.7.0
RUN pip install --no-cache-dir "strata-notebook[notebook]==${STRATA_VERSION}"

# Cells execute arbitrary user code, so do not run them as root. The server
# image does the same (see the root Dockerfile).
RUN useradd --create-home --shell /bin/bash worker
USER worker
WORKDIR /home/worker

# Add whatever your cells import on top of this image:
#   FROM strata-worker:latest
#   RUN pip install --no-cache-dir torch transformers

# 8080 because that is DockerBackend's default ``worker_port``. The worker's
# own default is 9000, so this is set explicitly rather than left to agree by
# luck; change both together or neither.
EXPOSE 8080

# /health is unauthenticated by design -- it is polled before the machine is
# trusted with anything and reveals nothing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()"

# STRATA_WORKER_TOKEN is minted per machine by the pool and injected into the
# environment at boot; the worker reads it from there. Never bake one in.
#
# The library treats an unset token as "auth disabled" for backward
# compatibility, which is defensible for a loopback-bound process and not for
# this image: it binds 0.0.0.0 and publishes 8080, so starting without a token
# would stand up an unauthenticated remote-code-execution endpoint. The image
# therefore refuses to start rather than inheriting that default.
CMD ["sh", "-c", "\
if [ -z \"${STRATA_WORKER_TOKEN:-}\" ]; then \
  echo 'refusing to start: STRATA_WORKER_TOKEN is unset, and this image binds' >&2; \
  echo '0.0.0.0 -- an unauthenticated /execute is remote code execution.' >&2; \
  echo 'The pool mints one per machine; set it yourself to run by hand.' >&2; \
  exit 1; \
fi; \
exec strata-worker --host 0.0.0.0 --port 8080"]
