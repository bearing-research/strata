# Service Mode

Service mode is what you run when more than one person uses Strata
through a network. It's the right mode when:

- Multiple users are sharing one Strata instance with their own
  identities (not just "everyone is logged in as me").
- The server is reachable beyond a loopback interface.
- Multi-tenancy matters, separate caches, separate QoS, separate
  metrics per tenant.
- You want the platform to control which transforms get to run
  (writes go through server-side transforms, not raw write endpoints).

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
   | `X-Tenant-ID` | Tenant the user belongs to, when multi-tenant is on | When `multi_tenant_enabled=true` |
   | `X-Strata-Scopes` | Space-separated capability set (e.g. `notebook:read notebook:write admin:cache`) | For scope-gated endpoints |
   | `X-Tenant-ID` | Tenant header, same role as `X-Strata-Tenant`, alternate name (configurable via `tenant_header`) | Either, depending on which header you configured |
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
uv run strata-server
```

## What service mode changes

Compared to personal mode:

- **No default artifact dir.** You must set `STRATA_ARTIFACT_DIR`
  explicitly (or configure a blob backend via
  `STRATA_ARTIFACT_BLOB_BACKEND=s3` etc.). Service mode refuses to
  start without a persistent target.
- **Writes go through server-side transforms.** Direct write
  endpoints are disabled at the surface, the platform decides what
  gets materialized, not the client. See `transforms_config` in
  `pyproject.toml` and `[tool.strata.transforms]`.
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
