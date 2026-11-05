# Service mode

Service mode is what you run when more than one person uses Strata
through a network. It's the right mode when:

- Multiple users are sharing one Strata instance with their own
  identities (not just "everyone is logged in as me").
- The server is reachable beyond a loopback interface.
- Multi-tenancy matters: each tenant gets its own cache keying and its
  own QoS limiter pool, and `GET /v1/admin/tenants` reports usage per
  tenant.
- You want the platform to mediate writes - either routing them
  through server-side transforms, or letting authenticated clients
  publish directly to a shared store with `service_writes_enabled`
  (the [shared research store](#authenticated-write-back-the-shared-research-store)).

For one person, use [personal mode](modes.md#personal-mode). A team can
also give each member a personal server and share results through a
service-mode store; see
[Connecting a notebook to the shared store](../guides/share/team-store.md#connecting-a-notebook-to-the-shared-store).

## Switching from the default

`STRATA_DEPLOYMENT_MODE` defaults to `personal`. To run in service
mode you set the mode explicitly *and* fill in the matching auth /
artifact configuration:

```bash
STRATA_DEPLOYMENT_MODE=service
STRATA_AUTH_MODE=trusted_proxy
STRATA_PROXY_TOKEN=<shared-secret>
STRATA_ARTIFACT_DIR=/path/to/dir   # required with any artifact store, blob backend or not
```

A [startup coherence check](https://github.com/bearing-research/strata/blob/main/src/strata/config.py) fires clear `ValueError`s on boot if anything's missing, a sloppy service-mode deploy refuses to start rather than silently exposing write endpoints.

## The trusted-proxy contract

Strata does not authenticate users itself. It trusts an upstream
proxy that:

1. **Terminates auth**: JWT, OIDC, mTLS, Cloudflare Access, SAML,
   whatever. Strata doesn't care which.
2. **Sets every identity header on every request**, replacing whatever
   the client sent:

   | Header | Value |
   |---|---|
   | `X-Strata-Principal` | Stable user identifier (email, sub claim, etc.) |
   | `X-Tenant-ID` | Tenant the user belongs to. Header name is configurable via `tenant_header`. Empty when multi-tenancy is off. |
   | `X-Strata-Scopes` | Space-separated capability set (e.g. `notebook:read notebook:write artifacts:write admin:cache`). Empty for a user with no scopes. |
   | `X-Strata-Proxy-Token` | Shared secret matching `STRATA_PROXY_TOKEN`; proves the request came through the proxy |

   Strata cannot tell a header the proxy set from one the client sent
   and the proxy passed along. A header the proxy leaves alone reaches
   Strata from the client: a proxy that sets no `X-Strata-Scopes` for a
   user lets that user send `X-Strata-Scopes: admin:*` and become an
   admin. So the proxy sets all four on every request, and sets an
   empty value when the user has none (nginx drops a header set to
   `""`, which is what you want). A minimal nginx block:

   ```nginx
   map $http_upgrade $connection_upgrade {  # in the http {} block
       default upgrade;
       ''      close;
   }

   location / {
       proxy_pass http://strata:8765;
       proxy_http_version 1.1;
       proxy_set_header Upgrade $http_upgrade;
       proxy_set_header Connection $connection_upgrade;
       proxy_set_header Host $http_host;

       # Every identity header, on every request, from the auth layer only.
       proxy_set_header X-Strata-Proxy-Token "<shared-secret>";
       proxy_set_header X-Strata-Principal   $authenticated_user;
       proxy_set_header X-Tenant-ID          $authenticated_tenant;  # "" if multi-tenancy is off
       proxy_set_header X-Strata-Scopes      $authenticated_scopes;  # "" for no scopes
   }
   ```

   `$authenticated_*` stand for whatever your auth layer provides (for
   example `auth_request_set` from an `auth_request` subrequest). If you
   renamed a header with `STRATA_PRINCIPAL_HEADER`, `STRATA_TENANT_HEADER`,
   `STRATA_SCOPES_HEADER` or `STRATA_PROXY_TOKEN_HEADER`, set the renamed
   one. Check it from outside
   the proxy: a request carrying `X-Strata-Scopes: admin:*` for an
   ordinary user must still get 403 from `GET /v1/logs`. The demo
   stack's `.docker/service-mode/nginx.conf` sets all four.

API keys are a separate auth mode (`STRATA_AUTH_MODE=api_key`) for a
server no proxy fronts: `strata apikey create` issues one, and the key
carries its own principal, tenant and scopes. A trusted-proxy server
ignores them, so machine callers (a CI job, an ETL service) of a proxied
server go through the proxy like everyone else. See
[Configuration → API keys](../reference/configuration.md#api-key-authentication).

3. **Is the only path to Strata.** Strata is on a private network /
   VPC / Kubernetes namespace; the proxy is the only ingress.
   Without that, anything that can reach Strata directly can forge
   the headers above and impersonate any user.

The proxy-token check is a backstop, not the security boundary:
the network-level isolation is. If an attacker can reach Strata's
IP directly, they can read the token from any leaked config and
forge headers. Treat the token as defense-in-depth.

The proxy must pass the browser's `Host` header through unchanged,
port included (nginx: `proxy_set_header Host $http_host;`, not
`$host`, which drops the port). Strata accepts a browser write or a
notebook WebSocket only when its `Origin` matches that `Host`; a proxy
that rewrites `Host` has to list the public origin in
`STRATA_CORS_ALLOW_ORIGINS` instead.

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
| `http://localhost:8866` | `Principal: demo-admin`, scopes: `admin:* notebook:read notebook:write notebook:execute` | Admin-only operations |

Both URLs route to the same Strata instance. Tenant header
(`X-Tenant-ID: demo-team`) is injected on both. Open either in a
browser to use the notebook UI; hit `/v1/...` endpoints with curl
to exercise the REST surface.

Cells run on the demo's executor, not on the Strata container: assign a
cell or notebook the `gpu-http` worker. A cell left on `local` is refused,
because a service-mode server does not run cell code on its own host (see
[What a cell can read](#what-a-cell-can-read)).

The configuration is in `.docker/service-mode/`:

- `pyproject.toml`, Strata's service-mode config (mounted into the
  container as `/home/strata/pyproject.toml`). Includes the proxy
  token, multi-tenancy on, tenant header name, and two worker-catalog
  entries pointing at the executor sidecar: `gpu-http` (direct transport)
  and `gpu-http-signed` (signed-URL transport).
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
                                        ┌─────────────────────────┐
                                        │ Artifact store          │
                                        │                         │
                                        │  metadata → Postgres    │
                                        │  blobs    → S3 / GCS /  │
                                        │             Azure       │
                                        └─────────────────────────┘
```

Strata sits on a private network, only the auth proxy can reach it.
Notebook execution dispatches to executors; the demo stack runs one
locally, production typically uses HTTP executors on dedicated nodes
or remote backends like Modal / Fly Machines. Cells do not run on the
Strata host itself unless you set `STRATA_NOTEBOOK_HARNESS_USER` (see
[What a cell can read](#what-a-cell-can-read)).

**The artifact store is two halves, and both have to be shared before
you can run a second replica.** Blobs go to S3, GCS or Azure rather
than a local volume, so they survive container churn. Metadata is a
SQLite file under `STRATA_ARTIFACT_DIR` by default, which is local to
one machine; set `STRATA_ARTIFACT_METADATA_DSN` to put it on Postgres
instead. A DSN with `STRATA_ARTIFACT_BLOB_BACKEND=local` is rejected at
startup for the same reason: shared metadata pointing at blobs only one
node can read is worse than either alone. See
[Configuration → artifact metadata](../reference/configuration.md#sharing-one-artifact-store-across-nodes)
for the settings and `strata migrate` for moving an existing store
across, and [Running more than one replica](#running-more-than-one-replica)
for what else the nodes must share.

## Minimum service-mode env vars

```bash
# Required
STRATA_DEPLOYMENT_MODE=service
STRATA_AUTH_MODE=trusted_proxy
STRATA_PROXY_TOKEN=<shared-secret-with-proxy>
STRATA_ARTIFACT_DIR=/path/to/dir  # required with any artifact store, blob backend or not
STRATA_HOST=0.0.0.0               # default 127.0.0.1; bind wider when the proxy runs on another host
STRATA_TRANSFORM_SIGNING_SECRET=<stable-secret>  # unset: a random one per process, so signed URLs die on restart and differ across replicas

# Multi-tenancy (optional but recommended for >1 team)
STRATA_MULTI_TENANT_ENABLED=true
STRATA_REQUIRE_TENANT_HEADER=true
STRATA_TENANT_HEADER=X-Tenant-ID  # match what your proxy injects
```

Run the server normally:

```bash
strata-notebook   # or `uv run strata-notebook` in a checkout
```

### Rotating the proxy token and the signing secret

`STRATA_PROXY_TOKEN` holds one value, with no overlap window: change it on
the proxy and the server together, and requests that arrive between the two
changes get a `401`. Changing `STRATA_TRANSFORM_SIGNING_SECRET` invalidates
every signed URL already issued, so a build in flight on a `signed` worker
fails; rotate it between runs and give every replica the same value.

## What service mode changes

Compared to personal mode:

- **No default artifact dir.** The artifact store exists only when
  `STRATA_ARTIFACT_DIR` is set, even when its metadata and blobs live
  elsewhere. Service mode refuses to start with a DSN, a non-local blob
  backend or `STRATA_SERVICE_WRITES_ENABLED` and no
  `STRATA_ARTIFACT_DIR`. Without any of those it runs scan-only, with no
  artifact store.
- **What is durable, and where.** The *metadata store* (Postgres with
  `STRATA_ARTIFACT_METADATA_DSN`, otherwise `artifacts.sqlite` in the
  artifact directory) holds artifact metadata, names, aliases,
  publications, pins, API keys and the server-managed
  [worker registry](../notebook/workers.md#server-managed-workers).
  The *blob store* holds artifact bytes: the configured bucket, or
  `blobs/` in the artifact directory with the local backend. Notebooks
  live in `STRATA_NOTEBOOK_STORAGE_DIR`. Back up those three. The
  artifact directory itself is node-local working space, and the SQLite
  file and local blobs are the only durable things in it, so with a DSN
  and a bucket it holds nothing that needs a backup. The row-group cache
  (`STRATA_CACHE_DIR`) is a cache and can always be thrown away.
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
- **ACLs apply.** `acl_config` deny / allow rules gate every table a
  request reads or writes: a scan through `POST /v1/materialize`, an
  artifact built from a table, cache warming, and writing an artifact
  into a table (`POST /v1/artifacts/{id}/v/{n}/export`). Deny rules
  cannot be bypassed by allow rules, deny-first evaluation. A stream
  (`GET /v1/streams/{id}`) is readable only by the principal that
  started it, in the tenant it started it in, and admin endpoints such as `POST /v1/cache/clear` need
  their [scope](#scope-gated-endpoints).
- **Per-tenant resources** when multi-tenancy is on. Each tenant
  gets its own QoS limiter pool and its own cache keying, so bulk
  queries from tenant A can't starve tenant B's dashboards. Per-tenant
  usage comes from `GET /v1/admin/tenants`, not Prometheus labels (see
  [Multi-tenancy](#multi-tenancy)).

## Running more than one replica

Several service-mode nodes can sit behind one proxy when they share
everything durable:

- **One metadata store and one blob store.** Every node sets the same
  `STRATA_ARTIFACT_METADATA_DSN` and the same bucket (see
  [above](#production-reference-architecture)). The worker registry
  lives in the metadata store, so a worker added through
  `/v1/admin/notebook-workers` on one node is offered by every node from
  its next request. Each node can hold up to 41 connections to the
  database, so size its `max_connections` for every node at once (see
  [Sharing one artifact store](../reference/configuration.md#sharing-one-artifact-store-across-nodes)).
- **One notebook storage.** Every node mounts the same
  `STRATA_NOTEBOOK_STORAGE_DIR` (NFS, EFS, Filestore or similar), so any
  node can open any notebook. A notebook's environment is built inside
  its directory (or in `STRATA_NOTEBOOK_SHARED_ENV_DIR`, which then has
  to be shared too), so run the same image on every node.
- **The same secrets.** `STRATA_PROXY_TOKEN` and
  `STRATA_TRANSFORM_SIGNING_SECRET` must match on every node, or a URL
  signed on one node fails on another. Set `STRATA_NODE_ADVERTISED_URL`
  per node to its own address, so stream fetches and a remote cell's
  console reach the right node (see
  [Running several nodes](../reference/configuration.md#running-several-nodes-behind-one-address)).
- **Session-affine routing.** A notebook session, which
  `POST /v1/notebooks/open` or `/create` returns, lives in the memory of
  the node that opened it, together with its WebSocket and running
  cells. Every later request for it must reach that node; any other node
  answers `404`. Route each user to one node, with a sticky cookie or a
  hash of the principal header, so the open and everything after it land
  together.

Not supported:

- **Moving a live session.** When a node goes away its sessions go with
  it. Opening the notebook again on another node starts a new session;
  cells whose inputs did not change are cache hits when they run again.
- **One notebook open on two nodes at once.** Each node keeps its own
  session and writes the same files, with nothing coordinating them.
- **Cluster-wide views of per-node state.** Cache warm jobs
  (`/v1/cache/warm/jobs*`), QoS limits, rate limits and metrics are per
  node; sum the metrics in your dashboards.

## Multi-tenancy

`STRATA_MULTI_TENANT_ENABLED=true` activates per-tenant isolation.
With `STRATA_REQUIRE_TENANT_HEADER=true`, requests without a tenant
header are rejected. Under `api_key` auth the tenant is the key's own
and an `X-Tenant-ID` the caller sends is ignored, so a key cannot pick
another tenant's limiters or log attribution; under `trusted_proxy` it
is the header the proxy sets. The tenant ID is validated as 1–64
alphanumeric / `_` / `-` characters, starting with a letter or digit, and
hashed into:

- **Cache keys**: tenant A and tenant B can scan the same Iceberg
  table and never see each other's row-group cache entries.
- **Cache directories**: per-tenant subdirs under the row-group
  cache dir. Artifacts are not split by directory; each row records
  its tenant, and reads are filtered by it.
- **QoS limiters**: interactive + bulk semaphores per tenant.
- **Notebooks**: each tenant's notebooks live under
  `<notebook_storage_dir>/<tenant>/`, and `/discover`, `/open`, `/create`
  and the imports stay inside it. A live session belongs to the tenant
  that opened it; to any other tenant its REST routes, WebSocket and MCP
  tools answer as for an unknown session. A caller the proxy sends
  without a tenant header is the default tenant (`_default`) for both:
  its notebooks live under `<notebook_storage_dir>/_default/` and its
  sessions are visible to that tenant only. `admin:*` sees every tenant's
  notebooks and sessions. This keeps tenants out of each other's
  notebooks, but cells still run on one host: it is a convenience for
  small, trusted tenants, not an isolation boundary (see
  [`STRATA_NOTEBOOK_HARNESS_USER`](#who-a-cell-is-strata_notebook_harness_user)
  and remote workers for what a cell can reach).
- **Cache warm jobs**: a background job from `POST /v1/cache/warm/async`
  is listed, read and cancelled (`/v1/cache/warm/jobs*`) only by its
  own tenant; `admin:*` sees every tenant's.
- **Metric labels**: under principal auth Prometheus output names no
  tenants, tables or principals: `/metrics/prometheus` is unauthenticated
  for scrapers, so the per-tenant series (`strata_tenant_*`,
  `strata_build_tenant_*`) and the per-table series are left out, and the
  AI usage series (`strata_ai_*`) are per-model totals with no `tenant` or
  `principal` label. Read per-tenant usage from `GET /v1/admin/tenants`
  (`admin:tenants`) and per-table usage from `GET /metrics/tables`
  (`admin:*`).

A tenant registry tracks active tenants (LRU-bounded; only a tenant with nothing in flight is evicted). [Implementation details are in the source tree](https://github.com/bearing-research/strata/tree/main/src/strata) if you need to extend the tenant-scoping behavior.

## ACLs

`acl_config` is a `pyproject.toml` block:

```toml
[tool.strata.acl_config]
# Action when no rule matches. Defaults to "allow" -- set it to "deny"
# if you want an allowlist, or every table not named below stays readable.
default = "deny"

# Deny rules evaluate first; explicit denies cannot be bypassed.
# "*:" covers every address of a table (see the configuration reference).
[[tool.strata.acl_config.deny]]
principal = "guest@example.com"
tables = ["*:internal.*"]

[[tool.strata.acl_config.allow]]
principal = "analyst@example.com"
tenant = "marketing"          # optional; omit to match any tenant
tables = ["file:marketing.*", "file:public.*"]
```

A rule matches when the principal matches (`*` matches any, and is the
default), the tenant matches if the rule names one, and **at least one** table
pattern matches. `tables` is required: a rule with no patterns can never match
anything, so the server refuses to start rather than loading a rule that does
nothing. A key Strata does not recognize (`tenants` for `tenant`, say) is
refused at startup too, rather than dropped from a rule that is then
wider than it reads.

Evaluation: deny rules → allow rules → default. [Wildcard and principal
matching semantics are documented in source](https://github.com/bearing-research/strata/tree/main/src/strata)
for anyone extending the ACL engine.

Under principal auth a denied table, whether scanned or used as a transform
input, is refused before planning with `404 Table not found`, so the answer
does not reveal that the table exists. Set
`STRATA_HIDE_FORBIDDEN_AS_NOT_FOUND=false` to get `403` instead.

The cache is shared, so the same rules gate reading a stored result: its data,
metadata, lineage and dependents, publishing it, and exporting it into a table
(`POST /v1/artifacts/{id}/v/{n}/export`). An export copies the result into a
table the caller can then scan, so it is refused both when the caller is denied
the target table and when the caller is denied a table the result was read from.
Rules have no separate write permission: a table a caller may read, it may export
into, unless another tenant [owns it](../guides/share/team-store.md#promoting-into-an-iceberg-table).

A result is gated by every table in its lineage, not only the ones it read
directly. A transform over a cached scan of a denied table is denied too, as is
anything computed from that result, and the same holds when such a result is
used as a transform input. The check follows the store's recorded input edges up
to 100 steps back; a result with a deeper lineage is refused with `403`, since a
table past that point cannot be checked. An ancestor that retention has already
collected cannot be checked either, so a deny rule only reaches results whose
lineage is still in the store.

### Scope-gated endpoints

A few operations require a specific scope under trusted-proxy or API-key
auth (`admin:*` satisfies any of them):

| Scope | Gates |
|---|---|
| `admin:cache` | `POST /v1/cache/clear`, `GET /v1/cache/stats`, `GET /v1/cache/entries`, `GET /v1/cache/histogram`, `GET /v1/debug/cache/inspect`, `POST /v1/metadata/cleanup` |
| `admin:tenants` | `GET /v1/admin/tenants` and `GET /v1/admin/tenants/{tenant_id}` |
| `admin:notebook-workers` | The server-managed worker registry, every `/v1/admin/notebook-workers*` route (list, replace, add, update, delete, refresh, reload); a personal server, which has no principal auth, serves them to whoever can reach it |
| `admin:notebook-credentials` | `GET` and `PUT /v1/admin/notebook-credentials`, a running server's named credentials and catalogs (see [Changing them while the server runs](../reference/notebook-toml.md#changing-them-while-the-server-runs)) |
| `admin:notebooks` | Quiescing a notebook or project and releasing it (`POST /v1/notebooks/{id}/quiesce` and `/release`, `POST /v1/projects/{path}/quiesce` and `/release`) |
| `admin:*` | Garbage collection over the whole store, another tenant or the tenantless rows (`POST /v1/artifacts/gc`), reading another tenant's `GET /v1/artifacts/usage` / `stats`, the server-wide log buffer (`GET /v1/logs`, `GET /v1/logs/stream`), per-table metrics (`GET /metrics/tables*`), and the process diagnostics under `/v1/debug/*` (latency, GC pauses, pools, connections, memory, rate limits) |
| `admin:registry` | `POST /v1/registry/pending/approve` and `.../reject` - deciding protected-alias changes |
| `artifacts:gc` | Garbage collection of the caller's own tenant (`POST /v1/artifacts/gc`); the tenant header is required, and tenantless rows are left alone |
| `artifacts:pin` | Pinning and unpinning a version against garbage collection (`POST` / `DELETE /v1/artifacts/{id}/v/{n}/pin`) |
| `artifacts:publish` | Minting, editing and withdrawing a publication (`POST /v1/artifacts/{id}/v/{n}/publish`, `PATCH` / `DELETE /v1/publications/{token}`) |
| `artifacts:write` | Publishing in service mode (`put` / `set_name` / `set_alias` / tags, and `name` on `POST /v1/materialize`) when `service_writes_enabled=true`. See [below](#authenticated-write-back-the-shared-research-store). |
| `notebook:read` | Every notebook `GET` over REST, and observing a notebook over its WebSocket (sync, previews, profiling) |
| `notebook:write` | Changing a notebook without running anything: creating, editing, reordering and deleting cells, and setting mounts, connections, workers, env, timeout, name and variants; also opening, creating, importing, closing and deleting notebooks, a cell's test source, promote, and recents validation. REST and WebSocket alike. |
| `notebook:execute` | Running code or changing its environment: executing a cell or its tests, run-all, dependency changes and environment sync, requirements imports, the Python version, SSH workers, the inspect REPL and widget updates. REST and WebSocket alike. |

The notebook scopes are checked against one table for both transports
(`strata.notebook.scopes`), so a viewer holding only `notebook:read` can't run a
cell over REST any more than over the socket. An operation the table doesn't
classify requires `notebook:execute`.

Registry **approval** additionally enforces separation of duty: the
principal who requested a protected-alias move cannot approve it
themselves unless they hold `admin:*`. The registry **audit** read
(`GET /v1/registry/audit`) and the events feed that follows it
(`GET /v1/events`) are tenant-scoped - a principal sees only its own tenant's
history; `admin:*` sees the whole store.

## Authenticated write-back: the shared research store

This has its own page now: [Team store](../guides/share/team-store.md).

## What a cell can read

A cell is arbitrary Python, spawned by default with the server's whole
environment. On a laptop that is right and there is nothing to protect. On a
shared server it means every member who can run a cell can read
`STRATA_NOTEBOOK_REMOTE_STORE_HEADERS`, `STRATA_PROXY_TOKEN`, worker tokens and
every data-source credential the server holds - from `os.environ`, from
`/proc/<pid>/environ`, or from any file the server process can open.

Two settings close this, and they only work together.

### Who a cell is: `STRATA_NOTEBOOK_HARNESS_USER`

A service-mode server **refuses to run cell code on its own host** unless you
say how that is safe. There are two answers:

1. **Run cells on another machine.** Assign cells a server-managed worker (a
   pool machine or a remote worker). This is the recommended answer: a
   different host is isolation that needs nothing arranged on this one.
2. **Run cells as a separate OS user.** Set `STRATA_NOTEBOOK_HARNESS_USER` to a
   user that exists on the server host. Cells then cannot read the server's
   environment through `/proc`, nor state the server keeps owner-only (below).

Without either, a cell that would run on the server host fails with a message
naming both. Cache hits are still served, since a hit starts no cell code.
Personal mode is unaffected. An `embedded://` executor worker counts as this
host: it runs the harness in place.

Switching users needs the privilege to do it, so **the server runs as root**
and drops to the harness user for every process that runs cell code: the cold
and R harnesses, the batch harness behind Run All, the warm pool workers, the
inspect REPL, cell tests and the R package restore. It is POSIX only. The
published Docker image runs as the non-root `strata` user, which refuses the
setting: run that container as root (`--user root`, or `user: root` in Compose)
and under an init (below). What the harness user needs:

| Path | Access |
| --- | --- |
| notebook directories (and, with multi-tenancy, each tenant's directory above them) | read, or traverse for the tenant directory |
| each notebook's `.venv` | read and execute |
| the Python interpreter behind the venvs | read and execute |
| each notebook's `.strata/` | traverse |

The interpreter row is the one that catches people. uv installs managed Pythons
under the installing user's home, which for a root server is `/root` and
unreadable by anyone else, so every cell fails with `Permission denied` on the
venv's `python`. Install them somewhere world-readable with
`UV_PYTHON_INSTALL_DIR=/opt/uv-python`. Nothing else needs arranging: the
per-run directories a cell writes into are handed to the harness user, as is
each notebook's `work/`, where its cells run. The notebook's directory stays the
server's, so a cell cannot change `notebook.toml`, its cells or its lockfiles,
or put a file beside them. Strata makes `work/` itself, refuses one that is a
link, and changes only that directory's owner, never that of the files an
earlier run left in it. A hand-over changes each entry's owner through a handle
opened on it, never by name, and leaves a symlink, or a file with another hard
link, as it is. Also handed over is
a remote `rw` mount's staging copy (which the server then uploads without
following any link the cell left in it, so a link to a server file is skipped,
not synced), the harness runs the venv's interpreter directly rather than
through `uv run`, and the artifact store is never read by the cell. The server, in turn, never
unpickles what a cell stored: prompt and SQL cells do not load a pickled value,
and cell tests load their inputs in the test process.

A cell run this way still shares the host's kernel and sees what any local user
can. A process a cell leaves running is stopped before the next run is handed a
directory, so it cannot read or change what later runs are given. On Linux,
before each hand-over Strata kills (`SIGKILL`) every process of the harness user
that no run holds. A run holds what descends from the server, and what is still
in the session of a running cell. The process that runs a cell adopts what the
cell orphans (it is a child subreaper), so a daemon a cell starts with `setsid`,
such as `ray start`'s, stays under it. So that daemon, or `cmd &` from a cell's
shell, is held while the process that ran its cell lives (a cold or warm harness
runs one cell, the batch harness behind Run All several), and once that process
exits it is stopped at the next hand-over. What started the server holds nothing,
so a supervisor that adopts orphans (`systemd --user`, `tini -s`) does not keep
what an ended run left. Every leftover is paused (`SIGSTOP`) before any is killed,
so one that keeps forking cannot outrun the sweep; if one still does after 20
passes, the server logs how many are left. Two things follow:

- **The harness user is this server's alone.** Any other process of that user,
  another server's cells or a login, is stopped too.
- **The server must not be PID 1.** There, every orphan is the server's child,
  so nothing is stopped and the server logs a warning. In a container, run it
  under an init (`docker run --init`, or `init: true` in Compose).

Until then the process runs as the harness user, so the server never writes by
path into a directory it has handed over. A notebook that needs more isolation
than that wants a worker on another machine.

A cell's ambient `strata` client reaches this server with a token the server
hands each run on the harness's stdin (see
[the ambient client](../guides/cells/python.md#the-ambient-strata-client)), which
stands in for the proxy token (or, under `api_key`, the key) on the client's
routes, plus promoting into its own session, and stops working when the run
ends. A request with it acts as whoever started the run, without their
`admin:` scopes. Every cell runs as the one harness user, so keep the host's
`kernel.yama.ptrace_scope` at 1 or higher: at 0, a cell can read the memory of
another running cell, its token included.

**One harness user serves every tenant, so it is not a tenant isolation
boundary.** It keeps cells away from the server; it does not keep one tenant's
cells away from another's. Every cell runs as the same user, and that user has
to read every notebook directory, so a cell can read any tenant's notebook
source and data files, and the bytes another notebook fetched (`.strata/fetch`)
or mounted (a view in `STRATA_NOTEBOOK_MOUNT_CACHE_DIR`), given the path. The
mount cache's `views/` and `objects/` directories cannot be listed, and a
view's name hashes the storage options that reach the store. Tenants that must not see
each other need separate servers (or containers, each with its own storage),
or every cell on workers that serve one tenant each.

What Strata keeps from the harness user, and from any other account on the
host, is its own state. It creates these owner-only, and narrows any an earlier
release left wider when it next opens them:

| State | Mode |
| --- | --- |
| `STRATA_ARTIFACT_DIR`, its `artifacts.sqlite` (with `-wal` / `-shm`) and `blobs/` | directories `0700`, files `0600` |
| `STRATA_CACHE_DIR` | `0700` |
| each notebook's `.strata/` | `0711`: the harness user passes through to its per-run directories and inputs, but cannot list it |
| in it, `artifacts/`, `console/`, `runtime.json` and `environment_jobs.json` | `0700` / `0600` |
| each tenant's notebook directory under the storage root | `0711` |

Strata only removes permission bits, so a stricter umask stands. Publication
tokens are stored as their SHA-256, so even a copy of `artifacts.sqlite` holds no
working link. Not covered, and yours to make owner-only: the server's config file
(a `pyproject.toml` holding `proxy_token`), the server's home, and
`STRATA_METADATA_DB`.

SQL cells are different: their queries run inside the server process, not as
the harness user, so the server checks which database file a SQLite or DuckDB
connection opens. After resolving `../` and following links, its `path` must be
inside the notebook's own directory, or else pass the mount-root rule below and
not be inside the server's state (the artifact directory, the cache directory,
the metadata database's directory, the notebook storage directory or the
server's home). The server opens the resolved file, checked again just before
opening, so a link changed after the first check is not followed. A SQLite
connection may not use `uri`, whose parameters can name
any file. Saving such a connection fails with a 400, and one already in
`notebook.toml` fails the cell and the connection's schema listing, naming it.
The keys the server sets on a connection it opens (`confine_to`,
`mount_sources`, `catalog_properties`) are dropped from a connection block.

A DuckDB SQL cell is also confined. Once its connection is
set up, it can reach only its own database, the roots of the mounts it reads,
and the locations of the catalog tables it reads; file access outside those is
refused, and the cell cannot turn it back on. So `read_text` of a server file,
`COPY ... TO`, and `ATTACH` of another path all fail. A catalog table whose data
files live outside its own location cannot be read this way.

A mount's root is therefore as much of the disk as the cell may read, as the
server, so a SQL cell refuses a local (`file://`) mount root, after following
links, that is `/` or has fewer than two path components (`/data`); that holds
or is inside the server's state (the artifact directory, the cache directory,
the metadata database's directory, the notebook storage directory or the
server's home), other than the notebook's own directory; or that is inside the
server's home, `/proc`, `/sys` or `/dev`. The cell fails
naming the mount. Python cells mount the same roots unchecked: they run as the
harness user, whose own file permissions decide what a root exposes. A SQLite cell also
runs in the server process: in a write cell `ATTACH`, `DETACH` and `VACUUM` are
refused, since they reach other files (a read cell runs only reads already).
Postgres, Snowflake and BigQuery cells send the query to their database server.
The server reads a BigQuery connection's `credentials_path` and
`write_credentials_path` key files itself, so each must be a file in the
notebook's own directory, after following links; one elsewhere fails the cell,
the schema listing and saving the connection. A BigQuery connection must name
`credentials_path`: without a key file the driver would use the server's own
Google credentials (its application default credentials), so one with none is
refused in the same three places.
A `${VAR}` in a connection's `auth` reads the notebook's env, never the server's
environment, which would otherwise go to whatever host the notebook names; a
secret the server holds reaches a connection only as a named `credential`.
`STRATA_NOTEBOOK_CREDENTIAL_PATHS` keeps a named credential to the notebooks
under some directories, and `STRATA_NOTEBOOK_CREDENTIAL_WRITERS` says who may
mount one read-write (see [Configuration](../reference/configuration.md#notebook)
and [Named credentials](../reference/notebook-toml.md#named-credentials)).

Prompt cells call their model from the server process too, and show the author
what came back. A `base_url` from the notebook's `[ai]` section is checked like
an `@fetch` URL: http(s) only, and a host on a loopback, private or link-local
address (the cloud metadata address among them) is refused unless it is named in
`STRATA_NOTEBOOK_FETCH_ALLOWED_HOSTS`, the same list `@fetch` uses. The
connection goes only to an address that passed the check, so it ignores proxy
settings. `STRATA_AI_BASE_URL` is yours and is not checked, nor is a notebook
naming that same URL or a provider's default one. Your `STRATA_AI_API_KEY` is
sent only to those URLs; a notebook naming another host must bring its own key.

Installing a notebook's Python packages runs as the server's user too, and building
a package from a source distribution runs that package's build backend, code
from wherever the package came from, with the server's environment. In service mode every `uv` command
a notebook runs (`sync`, `add`, `lock`, the `uv run` that starts a cell)
installs wheels only: a dependency with no wheel for the notebook's Python
fails to resolve, and uv's message says a wheel is required. Personal mode
still builds from source.

R has no wheels-only switch: `renv::restore()` builds CRAN packages from source
on Linux, running their configure scripts, and every `Rscript` started in a
notebook directory sources the notebook's `.Rprofile`. So in service mode that
code runs as the harness user, never as the server. The restore on notebook
open runs as the harness user with the filtered environment below, and the
notebook's `renv/` directory (and, with the shared environment backend, the
shared R library and package cache) is handed to that user to write. The R
package listing in the environment panel runs as the harness user too. With no
harness user nothing is restored, and the server log says why. Installing an R
package from the notebook (`renv::init`, adding a package) is refused, since it
writes `renv.lock` and `.Rprofile` into the notebook directory: commit the
package in `renv.lock` where the notebook is authored, and the server restores it
when the notebook opens. Personal mode is unchanged.

### What a cell is given: `STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST`

The allowlist narrows the environment a cell receives:

```bash
# Only what cells actually need; everything else stays with the server.
STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST=AWS_*,HF_TOKEN
```

Entries are exact names or a prefix with a trailing `*`. The essentials a
subprocess cannot start without are always included: `PATH`, `HOME`, the temp
and locale variables, and `UV_*`, `PYTHON*`, `VIRTUAL_ENV`, `R_*` and
`RSTUDIO_*`, except names that look like credentials (containing `KEY`,
`SECRET`, `TOKEN`, `PASSWORD` or `CREDENTIAL`), `UV_PUBLISH_*`,
`UV_INDEX_*_USERNAME`, and the index URLs (`UV_INDEX`, `UV_INDEX_URL`,
`UV_EXTRA_INDEX_URL`, `UV_DEFAULT_INDEX`, `UV_INDEX_*_URL`), which can carry a
login as `https://user:password@host/simple`. So a private index's login, which
the server's `uv sync` needs, does not reach cells. `STRATA_*` is dropped
unless named exactly - a prefix rule broad enough to catch a credential by
accident is the failure the setting exists to prevent. A name you list exactly
always passes. It applies to every process that runs cell code, the same list
as above.

The list stays short because a cell's own configuration does not come through
the process environment. `[env]` in `notebook.toml` and mount credentials
travel in the cell manifest and are applied inside the harness.

On its own the allowlist is not isolation. A cell running as the server's user
can still read the server's whole environment from `/proc/<server pid>/environ`,
because it *is* that user. The allowlist decides what a cell is handed; the
harness user is what stops it taking the rest.

### A notebook's secret manager

A notebook's `[secret_manager]` block names where its secrets come from, and the
server fetches them with its own Infisical credentials (`INFISICAL_CLIENT_ID` /
`INFISICAL_CLIENT_SECRET` or `INFISICAL_TOKEN`). Two consequences on a shared
server:

- **The host is yours.** The server logs in only at `INFISICAL_HOST` (or the
  public `https://app.infisical.com` when unset). A notebook whose `base_url`
  names another host fails to fetch, with a message naming `base_url`, and no
  login is attempted: otherwise the author could have the server send its
  credentials anywhere.
- **The project is the author's.** `project_id`, `environment` and `path` come
  from the notebook, and Strata has no list of projects a notebook may read. Any
  notebook author can read every secret the server's machine identity can
  read. Scope that identity to what every author on this server may see, or run
  a separate server (with its own identity) per group that may see more.

## Migrating from personal mode

If you've been running personal mode and want to grow into service:

1. **Decide on the auth boundary.** Anything from "Cloudflare Access
   in front of a Fly app" to "OIDC behind an enterprise ingress":
   the only requirement is that the proxy can inject the four
   headers above and that Strata is otherwise unreachable.

2. **Pick an artifact backend.** Local-disk artifacts don't survive
   container restarts cleanly in multi-replica setups. Configure
   one of `STRATA_ARTIFACT_BLOB_BACKEND=s3|gcs|azure` and the
   matching credentials, and keep `STRATA_ARTIFACT_DIR` set: service
   mode refuses a non-local blob backend without it. See
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
   `[tool.strata.transforms]` block in `pyproject.toml` to expose the
   computations you want the platform to run on the client's
   behalf. The block is an allowlist: `enabled = true` (or
   `STRATA_TRANSFORMS_ENABLED=true`) alone refuses every transform, so
   list each one. The built-in SQL transform runs in the server process:

   ```toml
   [tool.strata.transforms]
   enabled = true

   [[tool.strata.transforms.registry]]
   ref = "duckdb_sql@v1"
   executor_url = "embedded://local"
   ```

   The notebook executor in the demo stack is another example.

The demo compose stack is a working starting point you can fork:
swap `nginx.conf` for your real auth proxy config, move the
artifact blobs to S3, and you have most of what production needs.
