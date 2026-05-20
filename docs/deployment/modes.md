# Deployment Modes

Strata's `deployment_mode` is binary — `personal` or `service` — but
`personal` supports two flavors depending on whether
`STRATA_PERSONAL_MODE_USER_HEADER` is set. In practice you're choosing
between three shapes.

## Decision matrix

| | **Personal** | **Personal + auth proxy** | **Service** |
| --- | --- | --- | --- |
| **Best for** | One developer, laptop, single notebook open | 5–20 trusted users behind Cloudflare Access / Pomerium / etc. | Multi-tenant team or customer-facing |
| **Writes** | Enabled | Enabled (per-user filter on discover/delete) | Server-side transforms only |
| **Auth** | None | Identity from `STRATA_PERSONAL_MODE_USER_HEADER` (proxy injects) | `X-Strata-Principal` + `X-Strata-Proxy-Token` from a trusted proxy |
| **Identity scoping** | No scoping (single user) | Per-user filter on `discover` / `delete`; shared artifact store | Per-tenant cache keys, storage dirs, QoS pools |
| **Multi-tenancy** | n/a | n/a | Optional (`multi_tenant_enabled=true`) |
| **ACLs** | Not evaluated | Not evaluated | Deny-first (`acl_config`) |
| **Default artifact dir** | `~/.strata/artifacts` | `~/.strata/artifacts` | Must be set explicitly (or blob backend) |
| **Network binding** | Loopback only | Non-loopback with `allow_remote_clients_in_personal=true` | Unrestricted |
| **Used by Strata's hosted preview** | – | ✓ (Fly.io + Cloudflare Access) | – |
| **Use in production for sharing?** | ❌ Anyone on the network can write | ⚠️ Only behind a real auth proxy, small trusted group | ✓ |

The rows that drive the choice are typically **Writes** (does anyone
who reaches the URL get to mutate?), **Auth** (who decides who's
allowed?), and **Identity scoping** (what's isolated and what's
shared?). The flags that follow are the consequences.

## Choosing a shape

- **Personal**: running the notebook on your own machine. Fast to
  start, nothing to configure, writes land in your home directory.
  This is the default for Docker Compose and the "from source"
  instructions.
- **Personal + auth proxy**: small team sharing one Strata instance
  with the proxy handling auth. Cheap to operate, no per-user
  isolation (artifact store is shared), but each user gets their
  own list in "Open existing." See [Sharing personal mode with a
  small group](#sharing-personal-mode-with-a-small-group) below.
  This is what Strata's own hosted preview at
  [strata-notebook.fly.dev](https://strata-notebook.fly.dev) runs.
  See [Fly.io deployment](fly.md) for the deployment recipe.
- **Service**: hosting Strata for users you can't fully trust, or
  with sensitive data, or with multi-tenant isolation requirements.
  Writes go through server-side transforms so the platform controls
  what gets materialized and by whom. See [Service Mode](service-mode.md).

## Setting the mode

```bash
export STRATA_DEPLOYMENT_MODE=personal   # or "service"
```

Or in `pyproject.toml`:

```toml
[tool.strata]
deployment_mode = "personal"
```

**Default is `personal`**: the common case. A first-time
`uv run strata-notebook` boots into a single-user, loopback-only
deployment that just works.

Service mode is explicit opt-in. The coherence checker fires clear
errors at startup if you set `deployment_mode=service` without
matching auth / artifact configuration, so misconfigured production
deploys fail fast rather than silently exposing write endpoints.

## Personal mode

```bash
STRATA_DEPLOYMENT_MODE=personal uv run strata-notebook
```

The server binds to `127.0.0.1` by default and refuses non-loopback
addresses unless you opt in:

```bash
STRATA_DEPLOYMENT_MODE=personal \
  STRATA_HOST=0.0.0.0 \
  STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL=true \
  uv run strata-notebook
```

Opt in only if you have separate protection (firewall, VPN, private
network) personal mode exposes write endpoints with no authentication.

Artifacts persist to `~/.strata/artifacts` unless `STRATA_ARTIFACT_DIR`
is set. Notebook deletion and session discovery/reconnect APIs are
personal-mode-only.

## Service mode

```bash
STRATA_DEPLOYMENT_MODE=service \
  STRATA_AUTH_MODE=trusted_proxy \
  STRATA_PROXY_TOKEN=<shared-secret> \
  uv run strata-notebook
```

Short version: an upstream proxy authenticates the caller, injects
identity headers (`X-Strata-Principal`, tenant header,
`X-Strata-Scopes`, `X-Strata-Proxy-Token`), and is the only ingress
path. Strata trusts the proxy and refuses to do its own auth.

See [Service Mode](service-mode.md) for the full story:

- The trusted-proxy header contract.
- The shipped demo stack (`docker-compose.service.yml` + nginx
  proxy injecting two demo identities on ports 8865/8866).
- Production reference architecture.
- Multi-tenancy, ACLs, server-side transforms.
- Migration path from personal mode.

## Sharing personal mode with a small group

A common deployment shape is "personal mode behind an authenticating proxy", for example, Cloudflare Access in front of a Fly.io app, sharing the
notebook UI with a handful of trusted users. This isn't full multi-tenancy
(no per-user storage, no per-user QoS, no artifact isolation), but Strata
provides a thin per-user filter so each invitee sees their own work in the
"Open existing" list.

Set:

```bash
STRATA_DEPLOYMENT_MODE=personal
STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL=true
STRATA_PERSONAL_MODE_USER_HEADER=Cf-Access-Authenticated-User-Email
```

The header value is whatever your proxy injects after authenticating the
caller. Strata treats the value as opaque, email, GitHub login, internal
ID, anything stable.

What changes when the header is set:

- `POST /v1/notebooks/create` stamps the caller's identity into
  `notebook.toml` as `owner`.
- `GET /v1/notebooks/discover` returns only notebooks where
  `owner == caller` or `owner is None`.
- `DELETE /v1/notebooks/{id}` and `POST /v1/notebooks/delete-by-path`
  return 404 if a non-owner tries to delete an owned notebook.
- Direct-URL access stays open: anyone with a notebook ID can `POST /open`
  and view it. This is intentional, it preserves "share a link with a
  collaborator" while preventing accidents in the discovery list.

What does *not* change:
- Concurrent edits to the same notebook still race (no per-user sessions).
- The artifact store is shared; provenance hashes don't include the caller.
- The AI API key pool (`STRATA_AI_*`) is shared across all users.
- Unowned (legacy) notebooks remain visible and deletable by everyone.

This shape is the right answer for a 5–20 person trusted group. For untrusted
or paid users, migrate to service mode for proper tenant isolation.

## Coherence enforcement

Strata rejects incoherent mode combinations at startup. These combos
raise `ValueError` during config load:

| Combination | Why it's rejected |
|---|---|
| `deployment_mode=personal` + `auth_mode=trusted_proxy` | Personal mode has no upstream proxy; identity headers would come from the loopback client |
| `deployment_mode=personal` + `multi_tenant_enabled=True` | Personal mode is single-user; there are no tenants to isolate |
| `deployment_mode=personal` + `require_tenant_header=True` | Same reason, no tenant dimension in personal mode |
| `deployment_mode=service` + `personal_mode_user_header` | Service mode uses `X-Strata-Principal` via trusted-proxy auth; the personal-mode shim is for proxy-fronted personal deployments only |

If you see one of these errors, you almost certainly pulled flags from a
service-mode config into a personal-mode deployment. Remove the
service-specific flags or switch to `deployment_mode=service`.

## Mode-independent settings

These apply identically in either mode and can be tuned freely:

- `rate_limit_*`, token-bucket rate limiting
- `acl_config`, deny/allow rules (only effective when `auth_mode` is set)
- `artifact_blob_backend`, local / s3 / gcs / azure
- Tracing, logging, S3 / GCS / Azure credentials
- Cache size, cache directory, metadata DB path
