# Deployment Modes

Strata runs in one of two deployment modes. The mode decides whether the
server accepts writes, who it trusts, and how strict it is about network
binding.

## Summary

| | Personal | Service |
|---|---|---|
| **Audience** | Single developer on a laptop | Multi-user hosted deployment |
| **Writes** | Enabled (`writes_enabled=True`) | Disabled at the surface; server transforms only |
| **Auth** | None (`auth_mode="none"`) | Trusted proxy injects identity headers |
| **Multi-tenancy** | Disabled | Optional (`multi_tenant_enabled=True`) |
| **Default artifact dir** | `~/.strata/artifacts` | Must be set explicitly (or use S3/GCS/Azure) |
| **Network binding** | Loopback only by default | Unrestricted |

## Choosing a mode

- **Personal** — running the notebook on your own machine. Fast to start,
  nothing to configure, writes land in your home directory. This is the
  default for Docker Compose and the "from source" instructions.
- **Service** — hosting Strata for a team, behind an ingress proxy that
  authenticates users. Writes must go through server-side transforms so
  the platform controls what gets materialized and by whom.

## Setting the mode

```bash
export STRATA_DEPLOYMENT_MODE=personal   # or "service"
```

Or in `pyproject.toml`:

```toml
[tool.strata]
deployment_mode = "personal"
```

**Default is `personal`** — the common case. A first-time
`uv run strata-server` boots into a single-user, loopback-only
deployment that just works.

Service mode is explicit opt-in. The coherence checker fires clear
errors at startup if you set `deployment_mode=service` without
matching auth / artifact configuration, so misconfigured production
deploys fail fast rather than silently exposing write endpoints.

## Personal mode

```bash
STRATA_DEPLOYMENT_MODE=personal uv run strata-server
```

The server binds to `127.0.0.1` by default and refuses non-loopback
addresses unless you opt in:

```bash
STRATA_DEPLOYMENT_MODE=personal \
  STRATA_HOST=0.0.0.0 \
  STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL=true \
  uv run strata-server
```

Opt in only if you have separate protection (firewall, VPN, private
network) — personal mode exposes write endpoints with no authentication.

Artifacts persist to `~/.strata/artifacts` unless `STRATA_ARTIFACT_DIR`
is set. Notebook deletion and session discovery/reconnect APIs are
personal-mode-only.

## Service mode

```bash
STRATA_DEPLOYMENT_MODE=service \
  STRATA_AUTH_MODE=trusted_proxy \
  STRATA_PROXY_TOKEN=<shared-secret> \
  uv run strata-server
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

A common deployment shape is "personal mode behind an authenticating proxy"
— for example, Cloudflare Access in front of a Fly.io app, sharing the
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
caller. Strata treats the value as opaque — email, GitHub login, internal
ID, anything stable.

What changes when the header is set:

- `POST /v1/notebooks/create` stamps the caller's identity into
  `notebook.toml` as `owner`.
- `GET /v1/notebooks/discover` returns only notebooks where
  `owner == caller` or `owner is None`.
- `DELETE /v1/notebooks/{id}` and `POST /v1/notebooks/delete-by-path`
  return 404 if a non-owner tries to delete an owned notebook.
- Direct-URL access stays open: anyone with a notebook ID can `POST /open`
  and view it. This is intentional — it preserves "share a link with a
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
| `deployment_mode=personal` + `require_tenant_header=True` | Same reason — no tenant dimension in personal mode |
| `deployment_mode=service` + `personal_mode_user_header` | Service mode uses `X-Strata-Principal` via trusted-proxy auth; the personal-mode shim is for proxy-fronted personal deployments only |

If you see one of these errors, you almost certainly pulled flags from a
service-mode config into a personal-mode deployment. Remove the
service-specific flags or switch to `deployment_mode=service`.

## Mode-independent settings

These apply identically in either mode and can be tuned freely:

- `rate_limit_*` — token-bucket rate limiting
- `acl_config` — deny/allow rules (only effective when `auth_mode` is set)
- `artifact_blob_backend` — local / s3 / gcs / azure
- Tracing, logging, S3 / GCS / Azure credentials
- Cache size, cache directory, metadata DB path
