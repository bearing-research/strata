# Fly.io Deployment

Strata's hosted preview runs on [Fly.io](https://fly.io) at [strata-notebook.fly.dev](https://strata-notebook.fly.dev).

## Prerequisites

- [Fly CLI](https://fly.io/docs/flyctl/install/) installed
- `fly auth login` completed

## Deploy

```bash
# First time
fly apps create strata-notebook
fly deploy

# Subsequent deploys
fly deploy
```

The volume defined in `[[mounts]]` (`strata_data`, 5 GB auto-extending to
20 GB) is created automatically on first deploy, no separate
`fly volumes create` step needed.

## Configuration

The `fly.toml` at the repo root configures:

- **VM size**: `shared-cpu-4x` with 2GB RAM
- **Auto-scaling**: machines suspend when idle, auto-start on requests
- **Persistent storage**: 5GB volume at `/home/strata/.strata` with auto-extend
- **Health check**: HTTP on `/health` every 15s

## Key environment variables

```toml
[env]
  STRATA_DEPLOYMENT_MODE = "personal"
  STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL = "true"
  STRATA_NOTEBOOK_PYTHON_VERSIONS = '["3.12","3.13"]'
  UV_PYTHON_DOWNLOADS = "automatic"
```

## Monitoring

```bash
fly logs          # Stream logs
fly status        # Machine status
fly ssh console   # SSH into the machine
```
