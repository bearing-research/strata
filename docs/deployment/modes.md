# Deployment Modes

Strata's `deployment_mode` is `personal` or `service`. A personal server
has exactly one user; a service-mode server serves many, each with their
own identity.

## Decision matrix

| | **Personal** | **Service** |
| --- | --- | --- |
| **Best for** | One person: a laptop, or a server of your own | A team or customer-facing deployment |
| **Writes** | Enabled | Off by default (server-side transforms); opt-in client write-back via `service_writes_enabled` |
| **Auth** | None | `X-Strata-Principal` + `X-Strata-Proxy-Token` from a trusted proxy, or an API key |
| **Identity scoping** | None (one user) | Per-tenant cache keys, cache dirs, QoS pools; artifacts filtered by tenant |
| **Multi-tenancy** | n/a | Optional (`multi_tenant_enabled=true`) |
| **ACLs** | Not evaluated | Deny-first (`acl_config`) |
| **Default artifact dir** | `~/.strata/artifacts` | None; set `STRATA_ARTIFACT_DIR` explicitly (required with a blob backend too) |
| **Network binding** | Loopback only, or non-loopback with `allow_remote_clients_in_personal=true` | Unrestricted |
| **Use in production for sharing?** | No: anyone who reaches it can write | Yes |

The rows that drive the choice are typically **Writes** (does anyone
who reaches the URL get to mutate?), **Auth** (who decides who's
allowed?), and **Identity scoping** (what's isolated and what's
shared?). The flags that follow are the consequences.

## Choosing a mode

- **Personal**: running the notebook for yourself. Fast to start,
  nothing to configure, writes land in your home directory. This is
  the default for Docker Compose and the "from source" instructions.
  A personal server you reach over the network is still one person's;
  see [Fly.io deployment](fly.md) for a recipe that hosts one.
- **Service**: one server for more than one person, users you can't
  fully trust, sensitive data, or multi-tenant isolation requirements.
  Reads are tenant-scoped and ACL-gated; writes are off by default
  (routed through server-side transforms), or authenticated clients
  can publish directly to a shared store with `service_writes_enabled`.
  See [Service Mode](service-mode.md).

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
network): personal mode exposes write endpoints with no authentication.

Loopback binding alone does not keep web pages out: the browser runs on
the same machine. Personal mode refuses a cross-origin page's writes
and its notebook WebSocket, and answers only to the `Host` names it
expects, so a DNS-rebinding page that resolves its own name to
`127.0.0.1` gets a 400. The expected
names are `localhost`, `127.0.0.1`, `[::1]`, any IP literal,
`STRATA_HOST`, and whatever you list in `STRATA_ALLOWED_HOSTS`. When
you reach a remote personal-mode server by name (`devbox.lan`,
`strata.example.com`), list that name:

```bash
STRATA_ALLOWED_HOSTS=devbox.lan,strata.example.com
```

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
path. Strata trusts the proxy rather than authenticating users itself;
machine callers outside the proxy can use an API key instead
(`STRATA_AUTH_MODE=api_key`).

See [Service Mode](service-mode.md) for the full story:

- The trusted-proxy header contract.
- The shipped demo stack (`docker-compose.service.yml` + nginx
  proxy injecting two demo identities on ports 8865/8866).
- Production reference architecture.
- Multi-tenancy, ACLs, server-side transforms.
- Migration path from personal mode.

## Sharing with a team

Personal mode has no notion of more than one user, so a team has two
shapes:

- **A personal server per member.** Each member runs their own (a laptop,
  or one container each behind your proxy) and shares results through a
  service-mode store; see
  [Connecting a notebook to the shared store](service-mode.md#connecting-a-notebook-to-the-shared-store).
- **One shared server in service mode**, where every request carries the
  caller's identity and scopes. See [Service Mode](service-mode.md).

## Coherence enforcement

Strata rejects incoherent mode combinations at startup. These combos
raise `ValueError` during config load:

| Combination | Why it's rejected |
|---|---|
| `deployment_mode=personal` + `auth_mode=trusted_proxy` | Personal mode has no upstream proxy; identity headers would come from the loopback client |
| `deployment_mode=personal` + `multi_tenant_enabled=True` | Personal mode is single-user; there are no tenants to isolate |
| `deployment_mode=personal` + `require_tenant_header=True` | Same reason, no tenant dimension in personal mode |
| `deployment_mode=personal` + `auth_mode=api_key` | Personal mode is single-user on loopback; authenticating yourself to your own machine buys nothing |
| `deployment_mode=service` + `multi_tenant_enabled` or `acl_config` rules or `mcp_enabled`, without `trusted_proxy` / `api_key` auth | The tenant header would be spoofable, ACL rules are only evaluated for an authenticated caller, and MCP would have no caller to check |
| `deployment_mode=service` + `auth_mode=trusted_proxy` without `proxy_token` | Every request would be accepted and the identity headers could be spoofed |
| `deployment_mode=service` + `service_writes_enabled` without `auth_mode=trusted_proxy` | Writes are stamped with the caller's principal and tenant |
| `deployment_mode=service` + `artifact_metadata_dsn` with `artifact_blob_backend=local` | Another node would resolve an artifact from the shared database and then find no bytes |
| `deployment_mode=service` + an artifact store without `artifact_dir` (`artifact_metadata_dsn`, a non-local blob backend, `service_writes_enabled`, `auth_mode=api_key`, or transforms enabled) | The store is only created when `artifact_dir` is set, so every artifact route would fail |

If you see one of the personal-mode errors, you almost certainly pulled
flags from a service-mode config into a personal-mode deployment. Remove
the service-specific flags or switch to `deployment_mode=service`. The
service-mode errors name the setting to add.

Other settings are checked at startup in either mode: a SQL catalog in object
storage needs a `uri`, the team cache needs `STRATA_NOTEBOOK_REMOTE_STORE_URL`
and that URL must not be this server, and adaptive concurrency needs its slot
counts inside its range and no multi-tenancy.

## Mode-independent settings

These apply identically in either mode and can be tuned freely:

- `rate_limit_*`, token-bucket rate limiting
- `acl_config`, deny/allow rules (only evaluated under `trusted_proxy` or `api_key` auth, which only service mode allows)
- `artifact_blob_backend`, local / s3 / gcs / azure
- Tracing, logging, S3 / GCS / Azure credentials
- Cache size, cache directory, metadata DB path
- `public_base_url` and `public_base_path`, the address readers reach the server on (below)

## Serving under a path

A reverse proxy can serve Strata at a path such as
`https://app.example.com/o/acme/lab/` instead of a hostname of its own. Tell
Strata the path:

```bash
STRATA_PUBLIC_BASE_PATH=/o/acme/lab
STRATA_PUBLIC_BASE_URL=https://app.example.com   # origin only; the path comes from above
```

The proxy may strip the prefix before forwarding or pass it through; both
reach the same routes. The notebook UI, its WebSocket, the API docs, signed
build URLs and every link on a [publication](../notebook/publishing.md) page
carry the prefix. Open the UI with the trailing slash
(`https://app.example.com/o/acme/lab/`), since its assets load relative to
the page. The prefix's first segment must not be one of Strata's own
(`v1`, `p`, `assets`, `docs`, `redoc`, `openapi.json`, `health`, `metrics`,
`oembed`, `mcp`), or a stripped request is read as
already carrying it.

Clients take the full URL, path included (`https://app.example.com/o/acme/lab`):
the CLI's `--server`, the TUI and `strata_client` all append their routes to it.
