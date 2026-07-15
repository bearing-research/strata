# Service Mode

Service mode is what you run when more than one person uses Strata
through a network. It's the right mode when:

- Multiple users are sharing one Strata instance with their own
  identities (not just "everyone is logged in as me").
- The server is reachable beyond a loopback interface.
- Multi-tenancy matters, separate caches, separate QoS, separate
  metrics per tenant.
- You want the platform to mediate writes - either routing them
  through server-side transforms, or letting authenticated clients
  publish directly to a shared store with `service_writes_enabled`
  (the [shared research store](#authenticated-write-back-the-shared-research-store)).

For a single developer running on a laptop, use
[personal mode](modes.md#personal-mode). Personal-mode-behind-a-proxy
also covers small-team sharing (~5–20 trusted users) see
[Sharing personal mode with a small group](modes.md#sharing-personal-mode-with-a-small-group).

## Switching from the default

`STRATA_DEPLOYMENT_MODE` defaults to `personal`. To run in service
mode you set the mode explicitly *and* fill in the matching auth /
artifact configuration:

```bash
STRATA_DEPLOYMENT_MODE=service
STRATA_AUTH_MODE=trusted_proxy
STRATA_PROXY_TOKEN=<shared-secret>
STRATA_ARTIFACT_DIR=/path/to/persistent/dir   # or a blob backend
```

A [startup coherence check](https://github.com/bearing-research/strata/blob/main/src/strata/config.py) fires clear `ValueError`s on boot if anything's missing, a sloppy service-mode deploy refuses to start rather than silently exposing write endpoints.

## The trusted-proxy contract

Strata does not authenticate users itself. It trusts an upstream
proxy that:

1. **Terminates auth**: JWT, OIDC, mTLS, Cloudflare Access, SAML,
   whatever. Strata doesn't care which.
2. **Injects identity headers** on every request:

   | Header | Value | Required |
   |---|---|---|
   | `X-Strata-Principal` | Stable user identifier (email, sub claim, etc.) | Yes |
   | `X-Tenant-ID` | Tenant the user belongs to, when multi-tenant is on. Header name is configurable via `tenant_header`. | When `multi_tenant_enabled=true` |
   | `X-Strata-Scopes` | Space-separated capability set (e.g. `notebook:read notebook:write artifacts:write admin:cache`) | For scope-gated endpoints |
   | `X-Strata-Proxy-Token` | Shared secret matching `STRATA_PROXY_TOKEN` | Yes, proves the request came from the proxy, not a direct connection |

3. **Is the only path to Strata.** Strata is on a private network /
   VPC / Kubernetes namespace; the proxy is the only ingress.
   Without that, anything that can reach Strata directly can forge
   the headers above and impersonate any user.

The proxy-token check is a backstop, not the security boundary:
the network-level isolation is. If an attacker can reach Strata's
IP directly, they can read the token from any leaked config and
forge headers. Treat the token as defense-in-depth.

## Running the demo stack

The repo ships a complete service-mode demo: Strata + a notebook
executor sidecar + an nginx proxy that injects two pre-baked
identities for testing.

```bash
docker compose -f docker-compose.service.yml up --build
```

The proxy exposes Strata on two ports, same server, different
synthesized callers:

| URL | What nginx injects | Use for |
|---|---|---|
| `http://localhost:8865` | `Principal: demo-user`, scopes: `notebook:read notebook:write notebook:execute` | Normal-user view |
| `http://localhost:8866` | `Principal: demo-admin`, scopes: `admin:* notebook:*` | Admin-only operations |

Both URLs route to the same Strata instance. Tenant header
(`X-Tenant-ID: demo-team`) is injected on both. Open either in a
browser to use the notebook UI; hit `/v1/...` endpoints with curl
to exercise the REST surface.

The configuration is in `.docker/service-mode/`:

- `pyproject.toml`, Strata's service-mode config (mounted into the
  container as `/home/strata/pyproject.toml`). Includes the proxy
  token, multi-tenancy on, tenant header name, and a sample
  worker-catalog entry pointing at the executor sidecar.
- `nginx.conf`, the two `server {}` blocks that inject the demo
  identities.

To experiment with new identities, edit `nginx.conf` and restart
the proxy container (`docker compose -f docker-compose.service.yml
restart proxy`).

## Production reference architecture

```
┌────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Your auth     │     │                 │     │              │
│  provider      │◄────┤  Auth proxy     │     │  Strata      │
│  (OIDC/SAML/   │     │  (nginx,        │────►│  (service    │
│   Cloudflare   │     │   Envoy,        │     │   mode)      │
│   Access, …)   │     │   Cloud Run     │     │              │
│                │     │   ingress, …)   │     │              │
└────────────────┘     │                 │     └──────┬───────┘
                       │  Adds headers:  │            │
                       │  Principal      │            ▼
                       │  Tenant         │     ┌──────────────┐
                       │  Scopes         │     │ Notebook     │
                       │  Proxy-Token    │     │ executors    │
                       └─────────────────┘     │ (sidecars or │
                                               │  remote)     │
                                               └──────┬───────┘
                                                      │
                                                      ▼
                                               ┌──────────────┐
                                               │ Artifact     │
                                               │ store (S3 /  │
                                               │  GCS / Azure │
                                               │  / local)    │
                                               └──────────────┘
```

Strata sits on a private network, only the auth proxy can reach it.
Artifacts persist to a blob backend (S3, GCS, Azure) rather than a
local volume so it survives container churn and is shared across
replicas. Notebook execution dispatches to executors; the demo
stack runs one locally, production typically uses HTTP executors
on dedicated nodes or remote backends like Modal / Fly Machines.

## Minimum service-mode env vars

```bash
# Required
STRATA_DEPLOYMENT_MODE=service
STRATA_AUTH_MODE=trusted_proxy
STRATA_PROXY_TOKEN=<shared-secret-with-proxy>
STRATA_ARTIFACT_DIR=/path/to/persistent/dir  # or use STRATA_ARTIFACT_BLOB_BACKEND

# Multi-tenancy (optional but recommended for >1 team)
STRATA_MULTI_TENANT_ENABLED=true
STRATA_REQUIRE_TENANT_HEADER=true
STRATA_TENANT_HEADER=X-Tenant-ID  # match what your proxy injects
```

Run the server normally:

```bash
uv run python -m strata
# or
uv run strata-notebook
```

## What service mode changes

Compared to personal mode:

- **No default artifact dir.** You must set `STRATA_ARTIFACT_DIR`
  explicitly (or configure a blob backend via
  `STRATA_ARTIFACT_BLOB_BACKEND=s3` etc.). Service mode refuses to
  start without a persistent target.
- **Reads work; direct writes are off by default.** Clients can read
  results - scan/stream a table, fetch an artifact's data
  (`GET /v1/artifacts/{id}/v/{n}/data`), and resolve a dataset by name
  (`GET /v1/names/{name}`) - all tenant-scoped and ACL-gated. Direct
  *write* endpoints (`put`, `set_name`, …) are disabled at the surface
  by default; the platform decides what gets materialized via
  server-side transforms (`transforms_config` /
  `[tool.strata.transforms]`). To let authenticated clients publish
  directly - the shared-research-store pattern - opt in with
  `service_writes_enabled` (see
  [below](#authenticated-write-back-the-shared-research-store)).
- **ACLs apply.** `acl_config` deny / allow rules gate
  `POST /v1/materialize`, `GET /v1/streams/{id}`, and admin endpoints
  like `POST /v1/cache/clear`. Deny rules cannot be bypassed by
  allow rules, deny-first evaluation.
- **Per-tenant resources** when multi-tenancy is on. Each tenant
  gets its own QoS limiter pool, its own metric labels, and its own
  cache keying, bulk queries from tenant A can't starve tenant B's
  dashboards.

## Multi-tenancy

`STRATA_MULTI_TENANT_ENABLED=true` activates per-tenant isolation.
With `STRATA_REQUIRE_TENANT_HEADER=true`, requests without a tenant
header are rejected. The tenant ID is validated as 1–64
alphanumeric / `_` / `-` characters and hashed into:

- **Cache keys**: tenant A and tenant B can scan the same Iceberg
  table and never see each other's row-group cache entries.
- **Storage directories**: per-tenant subdirs under the artifact
  store.
- **QoS limiters**: interactive + bulk semaphores per tenant.
- **Metric labels**: Prometheus output carries a `tenant` label so
  you can dashboard per-tenant usage.

A tenant registry tracks active tenants (LRU-bounded). [Implementation details are in the source tree](https://github.com/bearing-research/strata/tree/main/src/strata) if you need to extend the tenant-scoping behavior.

## ACLs

`acl_config` is a `pyproject.toml` block:

```toml
[tool.strata.acl_config]
# deny rules evaluate first; explicit denies cannot be bypassed.
[[tool.strata.acl_config.deny]]
principal = "guest@example.com"
resource = "tables/internal/*"

[[tool.strata.acl_config.allow]]
principal = "analyst@example.com"
resource = "tables/marketing/*"
scope = "notebook:write"

# default action when no rule matches (default: "deny")
default = "deny"
```

Evaluation: deny rules → allow rules → default. [Wildcard, principal, and scope matching semantics are documented in source](https://github.com/bearing-research/strata/tree/main/src/strata) for anyone extending the ACL engine.

### Scope-gated endpoints

A few operations require a specific scope under trusted-proxy auth
(`admin:*` satisfies any of them):

| Scope | Gates |
|---|---|
| `admin:cache` | `POST /v1/cache/clear` |
| `admin:registry` | `POST /v1/registry/pending/approve` and `.../reject` - deciding protected-alias changes |
| `artifacts:write` | Publishing in service mode (`put` / `set_name` / `set_alias` / tags) when `service_writes_enabled=true`. See [below](#authenticated-write-back-the-shared-research-store). |

Registry **approval** additionally enforces separation of duty: the
principal who requested a protected-alias move cannot approve it
themselves unless they hold `admin:*`. The registry **audit** read
(`GET /v1/registry/audit`) is tenant-scoped - a principal sees only its
own tenant's history; `admin:*` sees the whole store.

## Authenticated write-back: the shared research store

!!! warning "Preview"
    Authenticated write-back is a **preview** feature - it deliberately re-opens
    writes in service mode, which is security-sensitive. It's off by default,
    auth-required, scope-gated, tenant-scoped, and audited, but the surface is
    new and may change. Evaluate it before relying on it in production. The
    server logs a notice at startup when it's enabled.

By default service mode is read-only to clients, computation goes through
server-side transforms. But a common deployment wants the inverse: a team of
researchers, each driving their own notebook, who **publish** processed datasets
to one central store so a dataset computed once is available to the whole team.

`service_writes_enabled` opts into that. It lets authenticated clients write
directly - `put`, `set_name`, `set_alias`, tags - under a strict contract:

- **Opt-in and auth-required.** Off by default; setting it requires
  `auth_mode=trusted_proxy` (enforced at startup), so every write is
  attributable.
- **Scope-gated.** Publishing requires the `artifacts:write` scope in the
  proxy-issued token (`admin:*` also satisfies it). Members without it stay
  read-only.
- **Tenant-scoped (team = tenant).** A write lands in the caller's tenant and
  can't target another, so teammates share a namespace and other teams are
  isolated. The publishing principal is recorded in the registry audit.

```bash
STRATA_DEPLOYMENT_MODE=service
STRATA_AUTH_MODE=trusted_proxy
STRATA_PROXY_TOKEN=<shared-secret-with-proxy>
STRATA_ARTIFACT_DIR=/path/to/persistent/dir   # or a blob backend
STRATA_MULTI_TENANT_ENABLED=true              # team = tenant
STRATA_SERVICE_WRITES_ENABLED=true            # the opt-in
```

The proxy injects `X-Strata-Scopes: artifacts:write` for principals allowed to
publish. The publish → consume loop then looks like:

```python
# Researcher A (team-a, artifacts:write) publishes a processed dataset:
strata.put(inputs=[], transform={"ref": "clean@v1"}, data=cleaned,
           name="team/cleaned-events")

# Any teammate (team-a) resolves the name to its current artifact and reads it:
info = strata.resolve_name("team/cleaned-events")   # {artifact_uri, version, …}

# Other-team principals (team-b) cannot resolve team-a's name - tenant isolation.
```

### Connecting a notebook to the shared store

Each researcher runs their own notebook, which **computes locally** but points
its ambient `strata` client at the central store via
`STRATA_NOTEBOOK_REMOTE_STORE_URL`. The notebook's own cell outputs and
provenance stay local; only what a cell explicitly publishes
(`strata.put(name=…)`) goes to the shared store.

```bash
# On each researcher's notebook server:
STRATA_NOTEBOOK_REMOTE_STORE_URL=https://store.team.example
# Auth the remote store needs (set via env, not committed config):
STRATA_NOTEBOOK_REMOTE_STORE_HEADERS='{"X-Strata-Proxy-Token":"…","X-Strata-Principal":"alice@team","X-Tenant-ID":"team-a","X-Strata-Scopes":"artifacts:write"}'
```

In a fully proxy-fronted setup the notebook's requests instead flow through the
same auth proxy, which injects identity, and `notebook_remote_store_headers`
can be omitted.

## Migrating from personal mode

If you've been running personal mode and want to grow into service:

1. **Decide on the auth boundary.** Anything from "Cloudflare Access
   in front of a Fly app" to "OIDC behind an enterprise ingress":
   the only requirement is that the proxy can inject the four
   headers above and that Strata is otherwise unreachable.

2. **Pick an artifact backend.** Local-disk artifacts don't survive
   container restarts cleanly in multi-replica setups. Configure
   one of `STRATA_ARTIFACT_BLOB_BACKEND=s3|gcs|azure` and the
   matching credentials. See
   [Artifact Storage](../reference/configuration.md#artifact-storage).

3. **Flip the mode** in env or `pyproject.toml`:
   ```bash
   STRATA_DEPLOYMENT_MODE=service
   STRATA_AUTH_MODE=trusted_proxy
   STRATA_PROXY_TOKEN=<your-shared-secret>
   ```
   Boot will fail with a clear error if anything's missing, that's
   the fail-closed property at work.

4. **(Optional) Add multi-tenancy.** Once multiple teams are using
   the same instance and you want isolation, flip
   `STRATA_MULTI_TENANT_ENABLED=true` and start injecting
   `X-Tenant-ID` from the proxy.

5. **(Optional) Add server-side transforms.** Configure the
   `transforms_config` block in `pyproject.toml` to expose the
   computations you want the platform to run on the client's
   behalf. The notebook executor in the demo stack is one example.

The demo compose stack is a working starting point you can fork:
swap `nginx.conf` for your real auth proxy config, point the
artifact dir at S3, and you have most of what production needs.
