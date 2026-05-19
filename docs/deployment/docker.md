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

1. **Frontend builder** (Node 25) builds the Vue.js UI
2. **Backend builder** (Python + Rust) builds the wheel with native extension
3. **Runtime**: minimal image with the wheel and frontend dist
