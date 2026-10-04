# Fly.io Deployment

This page deploys Strata on [Fly.io](https://fly.io) as a single-tenant, personal-mode server, from the template `fly.example.toml` at the repository root.

## Trust model

!!! warning "Read before deploying to a public URL"
    `fly.example.toml` deploys Strata in **personal mode** with
    `STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL = "true"`. Personal mode
    has no authentication and enables write endpoints (create / delete
    notebooks, upload artifacts). Anyone who reaches the Fly app's URL
    can use it.

    This is fine for: a personal scratch instance behind a URL you
    don't share, a hosted demo, or a deployment fronted by an
    authenticating proxy (Cloudflare Access, Pomerium, etc.).

    This is **not** appropriate for: shared team deployments without
    an auth proxy, anything with sensitive data, anything you'd be
    upset about a stranger writing to. For those, use
    [service mode](service-mode.md) with the trusted-proxy auth
    pattern instead. See [Deployment Modes](modes.md) for the full
    comparison.

## Prerequisites

- [Fly CLI](https://fly.io/docs/flyctl/install/) installed (`brew install flyctl` on macOS).
- `fly auth login` completed (opens a browser; one-time).
- A Fly.io account with a payment method on file. The template runs
  one `shared-cpu-4x` VM with 2 GB of memory. It suspends idle machines
  but keeps one running, so it does not scale to zero unless you change
  `min_machines_running`. See
  [Fly pricing](https://fly.io/docs/about/pricing/) for what that costs.

## Deploy

From the repository root, copy the template to `fly.toml` (gitignored,
so your values stay out of commits):

```bash
cp fly.example.toml fly.toml
```

Replace its placeholders:

- `app`: your app name. App names are global on Fly, so pick one nobody
  has taken.
- `primary_region`: the region the machine and its volume live in
  (`fly platform regions` lists the codes).
- `STRATA_ALLOWED_HOSTS`: `<your-app-name>.fly.dev`, plus any custom
  domain.

Then:

```bash
# First time
fly apps create <your-app-name>   # add --org <org> for an organization other than your personal one
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

Expected response: exactly `{"status":"ok"}`. The detailed views are `/health/dependencies` and `/health/ready`. If you get a
504 or connection error, run `fly logs` to inspect the startup -
the most common cause is a cold-start delay on the first request.

## Configuration

The template configures:

- **VM size**: `shared-cpu-4x` with 2 GB RAM
- **Auto-scaling**: machines suspend when idle, auto-start on requests
- **Persistent storage**: 5 GB volume at `/home/strata/.strata` with auto-extend
- **Health check**: HTTP on `/health` every 15 s

## Key environment variables

```toml
[env]
  STRATA_DEPLOYMENT_MODE = "personal"
  STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL = "true"
  STRATA_ALLOWED_HOSTS = "<your-app-name>.fly.dev"
  STRATA_NOTEBOOK_PYTHON_VERSIONS = '["3.12","3.13"]'
  UV_PYTHON_DOWNLOADS = "automatic"
```

`STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL` is what lets personal-mode
bind to `0.0.0.0` instead of loopback only - without it, Strata
refuses to start on a Fly machine because the Fly proxy can't reach
a loopback bind. Setting this is the explicit acknowledgment that
you understand the personal-mode trust model (see the warning above).

`STRATA_ALLOWED_HOSTS` names the host browsers reach the app on.
Personal mode answers only to loopback names, IP literals and the names
listed here, which keeps DNS-rebinding pages out; a request for any
other `Host` gets a 400. Set it to `<your-app-name>.fly.dev` and add
any custom domain, comma-separated.

## Monitoring

```bash
fly logs          # Stream logs
fly status        # Machine status
fly ssh console   # SSH into the machine
```
