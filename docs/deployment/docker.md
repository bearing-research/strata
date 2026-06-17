# Docker Deployment

## Prerequisites

- Docker Engine ≥ 24 with Compose v2 (`docker compose`, not the
  legacy `docker-compose` script). [Docker Desktop](https://www.docker.com/products/docker-desktop/)
  bundles both; on Linux install Docker Engine + the
  [Compose plugin](https://docs.docker.com/compose/install/linux/).
- ~1.5 GB of disk for the first build; subsequent rebuilds reuse the
  layer cache.

## Quick Start

```bash
docker compose up -d --build
```

Open [http://localhost:8765](http://localhost:8765).

To stop:

```bash
docker compose down
```

## What's included

The `docker-compose.yml` runs Strata in **personal mode** with:

- Frontend built and served by the backend
- Persistent notebook storage via a named volume
- Persistent cache and metadata via a named volume
- Health check on `/health`

## Volumes

| Volume | Mount point | Purpose |
|--------|------------|---------|
| `strata-state` | `/home/strata/.strata` | Cache, metadata DB, artifacts |
| `strata-notebooks` | `/tmp/strata-notebooks` | Notebook directories |

Data persists across `docker compose down/up` cycles. To reset completely:

```bash
docker compose down -v  # removes volumes
```

### Edit notebooks from your host (bind mount)

Named volumes live inside Docker's storage area and aren't visible to
your file manager or editor on the host. To open the notebook
directories in VS Code, IntelliJ, or any local editor, swap the
named volume for a **bind mount** that maps a host path into the
container.

In `docker-compose.yml`, replace:

```yaml
    volumes:
      - strata-state:/home/strata/.strata
      - strata-notebooks:/tmp/strata-notebooks
```

…with:

```yaml
    volumes:
      - strata-state:/home/strata/.strata
      - ./notebooks:/tmp/strata-notebooks       # ← bind mount: ./notebooks is on your host
```

…and drop the matching `strata-notebooks` entry from the top-level
`volumes:` block (it's no longer used).

```bash
mkdir -p ./notebooks
docker compose up -d --build
# Notebook files now live in ./notebooks/ on your host, editable
# in VS Code / vim / Finder.
```

**Permissions.** The container runs as a non-root `strata` user (UID
1000 in the Docker image). If your host user has a different UID,
files the container writes will be owned by UID 1000 from the host's
perspective. On Linux, `chown -R 1000:1000 ./notebooks` once, or run
the container with `user: "${UID}:${GID}"` in compose to align IDs.
On macOS Docker Desktop this is handled transparently by the file-
sharing layer; no action needed.

**Why `/tmp/strata-notebooks` inside the container?** That's the path
the compose file pins via `STRATA_NOTEBOOK_STORAGE_DIR`. The path is
arbitrary as long as the env var and the volume mount agree — feel
free to change both to `/data/notebooks` or whatever fits your
mental model.

## Environment variables

Override defaults in `docker-compose.yml` or via `.env` file:

```yaml
environment:
  - STRATA_HOST=0.0.0.0
  - STRATA_PORT=8765
  - STRATA_DEPLOYMENT_MODE=personal
  - STRATA_CACHE_DIR=/home/strata/.strata/cache
```

See [Configuration Reference](../reference/configuration.md) for all options.

## Building the image manually

```bash
docker build -t strata .
docker run --rm -p 8765:8765 \
  -v strata_state:/home/strata/.strata \
  strata
```

The multi-stage Dockerfile:

1. **Frontend builder** (Node 26) builds the Vue.js UI
2. **Backend builder** (Python + Rust) builds the wheel with native extension
3. **Runtime**: minimal image with the wheel and frontend dist
