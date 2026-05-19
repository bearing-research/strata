# Fly.io Deployment

Strata's hosted preview runs on [Fly.io](https://fly.io) at [strata-notebook.fly.dev](https://strata-notebook.fly.dev). This page describes how to replicate that setup for your own single-tenant deployment.

## Trust model

!!! warning "Read before deploying to a public URL"
    The `fly.toml` in this repo deploys Strata in **personal mode** with
    `STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL = "true"`. Personal mode
    has no authentication and enables write endpoints (create / delete
    notebooks, upload artifacts). Anyone who reaches the Fly app's URL
    can use it.

    This is fine for: a personal scratch instance behind a URL you
    don't share, a hosted demo (like Strata's own preview), or a
    deployment fronted by an authenticating proxy (Cloudflare Access,
    Pomerium, etc.).

    This is **not** appropriate for: shared team deployments without
    an auth proxy, anything with sensitive data, anything you'd be
    upset about a stranger writing to. For those, use
    [service mode](service-mode.md) with the trusted-proxy auth
    pattern instead. See [Deployment Modes](modes.md) for the full
    comparison.

## Prerequisites

- [Fly CLI](https://fly.io/docs/flyctl/install/) installed (`brew install flyctl` on macOS).
- `fly auth login` completed (opens a browser; one-time).
- A Fly.io account with a payment method on file. The default
  `shared-cpu-4x` VM costs ~$5/mo if always-on, less if you let it
  scale to zero.

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

## Verify

After `fly deploy` reports success:

```bash
curl https://<your-app-name>.fly.dev/health
```

Expected response: `{"status":"ok"}` (plus details). If you get a
504 or connection error, run `fly logs` to inspect the startup —
the most common cause is a cold-start delay on the first request.

## Configuration

The `fly.toml` at the repo root configures:

- **VM size**: `shared-cpu-4x` with 2 GB RAM
- **Auto-scaling**: machines suspend when idle, auto-start on requests
- **Persistent storage**: 5 GB volume at `/home/strata/.strata` with auto-extend
- **Health check**: HTTP on `/health` every 15 s

## Key environment variables

```toml
[env]
  STRATA_DEPLOYMENT_MODE = "personal"
  STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL = "true"
  STRATA_NOTEBOOK_PYTHON_VERSIONS = '["3.12","3.13"]'
  UV_PYTHON_DOWNLOADS = "automatic"
```

`STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL` is what lets personal-mode
bind to `0.0.0.0` instead of loopback only — without it, Strata
refuses to start on a Fly machine because the Fly proxy can't reach
a loopback bind. Setting this is the explicit acknowledgment that
you understand the personal-mode trust model (see the warning above).

## Monitoring

```bash
fly logs          # Stream logs
fly status        # Machine status
fly ssh console   # SSH into the machine
```
