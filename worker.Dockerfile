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

FROM python:3.13-slim

# Installed with plain pip, not into a uv venv. ``strata-notebook`` refuses to
# start outside one (src/strata/_uv_runtime.py), but the *worker* entry point
# is deliberately not gated by that guard -- which is what lets it run on a
# stock Python image here, and on Modal's standard image stack.
#
# The ``notebook`` extra is not optional for a worker: cells execute through
# harness.py, which needs orjson / cloudpickle / pandas / numpy to serialize
# what they produce. Without it the machine boots, answers /health, accepts a
# job, and fails at the point of doing the work.
ARG STRATA_VERSION=0.7.0
RUN pip install --no-cache-dir "strata-notebook[notebook]==${STRATA_VERSION}"

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
CMD ["strata-worker", "--host", "0.0.0.0", "--port", "8080"]
