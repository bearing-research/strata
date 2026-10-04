# A Strata worker image that satisfies the strata-pool worker contract.
#
# Not a stage in the root Dockerfile: that one builds with no ``--target``, so
# a new last stage would change what ``docker build .`` produces.
#
# Not under ``docker/``: a top-level ``docker`` directory makes ruff's isort
# treat the ``docker`` PyPI package as first-party and reorder test imports.
#
#   docker build -f worker.Dockerfile -t strata-worker:latest .
#
# The pool does not pull images, so build it on the host that will run it (or
# push it to a registry the host has already pulled from).
#
# CI builds this and checks that it refuses to start without a token, does not
# run as root, serves /health on 8080, and requires the token on /execute.

FROM python:3.14-slim

# Plain pip, not a uv venv: the uv-runtime guard (src/strata/_uv_runtime.py)
# gates ``strata-notebook`` but not the worker entry point, so this runs on a
# stock Python image (and Modal's).
#
# The ``notebook`` extra is required: harness.py needs orjson / cloudpickle /
# pandas / numpy to serialize results. Without it the worker boots and answers
# /health, then fails every job.
#
# Needs 0.7.0+ (``POST /execute``). Pinned so the worker cannot drift from the
# server that issued its manifest. It installs from PyPI, so a build in a
# checkout does NOT pick up local worker changes; for unreleased code, install
# a ``uv build`` wheel instead. Bump the pin only *after* a release: CI builds
# this file, and a version PyPI lacks fails the build. Until then a fresh image
# runs a worker one version behind the server.
#
# ``uv`` is required too: a notebook with a uv.lock (every ``strata new``
# notebook) runs in a locked environment built by ``uv sync --frozen``
# (worker_env.py). Without uv the worker reports
# ``locked_environments: false`` and cells run in this image, ignoring the
# notebook's pins.
ARG STRATA_VERSION=0.8.0
RUN pip install --no-cache-dir "strata-notebook[notebook]==${STRATA_VERSION}" uv

# R cells, with --build-arg WITH_R=true:
#
#   docker build -f worker.Dockerfile --build-arg WITH_R=true -t strata-worker:r .
#
# harness.R needs jsonlite and arrow. Debian does not package arrow and
# compiling it takes about an hour, so both come as binaries from Posit's
# package manager. renv restores a notebook's renv.lock; without it the worker
# reports ``locked_r_environments: false`` and R cells use this library.
ARG WITH_R=false
RUN if [ "$WITH_R" = "true" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends r-base-core \
      && rm -rf /var/lib/apt/lists/* \
      && Rscript -e 'options(repos = c(CRAN = "https://packagemanager.posit.co/cran/__linux__/trixie/latest"), HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version["platform"], R.version["arch"], R.version["os"]))); install.packages(c("jsonlite", "arrow", "renv")); for (p in c("jsonlite", "arrow", "renv")) if (!requireNamespace(p, quietly = TRUE)) stop(p, " did not install")'; \
    fi

# Cells run arbitrary user code, so not as root.
RUN useradd --create-home --shell /bin/bash worker
USER worker
WORKDIR /home/worker

# Add whatever your cells import on top of this image:
#   FROM strata-worker:latest
#   RUN pip install --no-cache-dir torch transformers

# 8080 matches DockerBackend's default ``worker_port`` (the worker's own default
# is 9000); change both together or neither.
EXPOSE 8080

# /health is unauthenticated by design: it is polled before the machine is
# trusted and reveals nothing.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health').read()"

# The pool mints STRATA_WORKER_TOKEN per machine and injects it at boot. Never
# bake one in.
#
# The library treats an unset token as "auth disabled", fine on loopback but
# not here: this image binds 0.0.0.0, so no token would mean unauthenticated
# remote code execution. Refuse to start instead.
CMD ["sh", "-c", "\
if [ -z \"${STRATA_WORKER_TOKEN:-}\" ]; then \
  echo 'refusing to start: STRATA_WORKER_TOKEN is unset, and this image binds' >&2; \
  echo '0.0.0.0 -- an unauthenticated /execute is remote code execution.' >&2; \
  echo 'The pool mints one per machine; set it yourself to run by hand.' >&2; \
  exit 1; \
fi; \
exec strata-worker --host 0.0.0.0 --port 8080"]
