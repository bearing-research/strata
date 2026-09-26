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
also covers small-team sharing (~5–20 trusted users); see
[Sharing personal mode with a small group](modes.md#sharing-personal-mode-with-a-small-group).

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
2. **Injects identity headers** on every request:

   | Header | Value | Required |
   |---|---|---|
   | `X-Strata-Principal` | Stable user identifier (email, sub claim, etc.) | Yes |
   | `X-Tenant-ID` | Tenant the user belongs to, when multi-tenant is on. Header name is configurable via `tenant_header`. | When `multi_tenant_enabled=true` |
   | `X-Strata-Scopes` | Space-separated capability set (e.g. `notebook:read notebook:write artifacts:write admin:cache`) | For scope-gated endpoints |
   | `X-Strata-Proxy-Token` | Shared secret matching `STRATA_PROXY_TOKEN` | Yes, proves the request came from the proxy, not a direct connection |

Machine callers that do not sit behind the proxy (a CI job, an ETL
service) authenticate with an API key instead: `strata apikey create`
issues one, and the key carries its own principal, tenant and scopes.
See [Configuration → API keys](../reference/configuration.md#api-key-authentication).

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
across.

## Minimum service-mode env vars

```bash
# Required
STRATA_DEPLOYMENT_MODE=service
STRATA_AUTH_MODE=trusted_proxy
STRATA_PROXY_TOKEN=<shared-secret-with-proxy>
STRATA_ARTIFACT_DIR=/path/to/dir  # required with any artifact store, blob backend or not

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

- **No default artifact dir.** The artifact store exists only when
  `STRATA_ARTIFACT_DIR` is set, including when
  `STRATA_ARTIFACT_METADATA_DSN` and a blob backend such as
  `STRATA_ARTIFACT_BLOB_BACKEND=s3` hold everything durable; the
  directory then keeps nothing that needs a backup. Service mode refuses
  to start with a DSN, a non-local blob backend or
  `STRATA_SERVICE_WRITES_ENABLED` and no `STRATA_ARTIFACT_DIR`. Without
  any of those it runs scan-only, with no artifact store.
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
  started it, and admin endpoints such as `POST /v1/cache/clear` need
  their [scope](#scope-gated-endpoints).
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
- **Cache directories**: per-tenant subdirs under the row-group
  cache dir. Artifacts are not split by directory; each row records
  its tenant, and reads are filtered by it.
- **QoS limiters**: interactive + bulk semaphores per tenant.
- **Metric labels**: Prometheus output carries a `tenant` label so
  you can dashboard per-tenant usage.

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

### Scope-gated endpoints

A few operations require a specific scope under trusted-proxy or API-key
auth (`admin:*` satisfies any of them):

| Scope | Gates |
|---|---|
| `admin:cache` | `POST /v1/cache/clear`, `GET /v1/cache/entries`, `GET /v1/debug/cache/inspect` |
| `admin:tenants` | `GET /v1/admin/tenants` and `GET /v1/admin/tenants/{tenant_id}` |
| `admin:notebooks` | Quiescing a notebook or project (`POST /v1/notebooks/{id}/quiesce`, `POST /v1/projects/{path}/quiesce`) |
| `admin:*` | Garbage collection (`POST /v1/artifacts/gc`, still limited to the caller's tenant), and reading another tenant's `GET /v1/artifacts/usage` / `stats` |
| `admin:registry` | `POST /v1/registry/pending/approve` and `.../reject` - deciding protected-alias changes |
| `artifacts:pin` | Pinning and unpinning a version against garbage collection (`POST` / `DELETE /v1/artifacts/{id}/v/{n}/pin`) |
| `artifacts:publish` | Minting, editing and withdrawing a publication (`POST /v1/artifacts/{id}/v/{n}/publish`, `PATCH` / `DELETE /v1/publications/{token}`) |
| `artifacts:write` | Publishing in service mode (`put` / `set_name` / `set_alias` / tags) when `service_writes_enabled=true`. See [below](#authenticated-write-back-the-shared-research-store). |
| `notebook:read` | Every notebook `GET` over REST, and observing a notebook over its WebSocket (sync, previews, profiling) |
| `notebook:write` | Changing a notebook without running anything: creating, editing, reordering and deleting cells, and setting mounts, connections, workers, env, timeout, name and variants. REST and WebSocket alike. |
| `notebook:execute` | Running code or changing its environment: executing a cell or its tests, run-all, dependency changes and environment sync, requirements imports, the Python version, SSH workers, the inspect REPL, widget updates and the assistant. REST and WebSocket alike. |

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
STRATA_ARTIFACT_DIR=/path/to/dir   # required with any artifact store, blob backend or not
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

On a **shared** notebook server several members run cells, and the static
headers name one identity for all of them. When a request carries a principal,
the server sends that caller's id as `X-Strata-Principal` to the remote store,
replacing the one in the static headers. This covers results offered to the
team cache, promotions, registry reads and approvals from the Registry tab, and
a cell's own ambient client. The team cache's "computed by" and the registry
audit then name the member. The remote store still has to accept that principal:
the static headers are what authenticate the server to it. Set
`STRATA_NOTEBOOK_REMOTE_STORE_FORWARD_PRINCIPAL=false` for a store that expects
one fixed service identity. Personal mode has no principal and is unaffected.

### The team cache: sharing results nobody named

Everything above is **explicit publish** - a researcher decides a dataset is
worth sharing and names it. That leaves out the expensive middle of a pipeline,
because nobody names their intermediate results, and that is exactly where the
recomputation is.

`STRATA_NOTEBOOK_TEAM_CACHE_ENABLED=true` closes that gap. On a **local cache
miss** the notebook asks the shared store whether anyone has already run this
exact computation, and on a hit it fetches the result instead of running the
cell. After a cell does run, its outputs are offered back to the store so the
next teammate hits.

```bash
# On each researcher's notebook server, alongside the two settings above:
STRATA_NOTEBOOK_TEAM_CACHE_ENABLED=true
```

It works because a cell's provenance key -
`sha256(sorted_input_hashes + source_hash + env_hash)` - contains no notebook id
and no cell id. Two people running the same source over the same inputs in the
same environment already arrive at the same hash; the store just had no way to
be asked.

Lineage answers the follow-up. Opening a result's lineage (from the cell's
artifact strip, or the registry panel) shows each step with **who computed it,
on what, and what it cost** - `alice@lab · cpython-3.14-linux-x86_64 · 38s`.
Each step also carries its **environment identity** - `env:3d731494` (which
package set) alongside the platform (on what). Those two together are what
answers the question a shared cache generates on day one: *"you got a hit and I
didn't - why?"* The answer is almost always that the identities differ, and
until they were readable there was no way to see it.

The author column fills in only for steps that came *from* the store - a cell
you ran yourself has no authenticated identity to attribute, so on a solo
notebook it stays blank throughout. The platform and environment identity are
recorded on every run, so what a shared store changes is not that they appear
but that they stop being the same value on every row. A graph where one step
ran on someone else's machine, in a different environment, is the case these
columns exist to make visible.

All of it is best-effort and recent: artifacts produced before these fields
existed show blanks, and re-running the cell is what fills them in.

The notebook's profiling panel splits the savings once a team hit happens:
"Cache savings ~12m" alongside "From your team ~8m (4 hits, alice, bob)". The
team figure is priced by what the *publisher's* run actually cost, carried on
the artifact - whoever gets the hit never ran the cell, so their own history
holds no comparable number and a locally-derived estimate would credit zero for
exactly the case the shared store exists to create.

What to know before switching it on:

- **It is a separate opt-in from the URL, deliberately.** Wanting a shared store
  to publish *to* is not the same as wanting one to silently source results
  *from*. Enabling it without `STRATA_NOTEBOOK_REMOTE_STORE_URL` is rejected at
  startup rather than left silently inert.
- **Publishing needs `artifacts:write`.** Members without the scope still get
  team cache *hits*; they just do not contribute. That is a supported
  configuration, and the notebook logs a warning when a publish is refused so
  it does not look like the cache is simply empty.
- **It never fails a cell.** A store that is unreachable, refusing, or missing
  one of a cell's outputs ends in "run it locally".
- **A notebook whose environment does not match its lockfile will not publish.**
  A failed `uv sync` keeps the previous venv and stays usable on purpose, so a
  transient network failure does not lock you out of your own notebook - but
  provenance is computed from `uv.lock`, which has moved on, so anything
  produced afterwards would be stamped with an environment it was not built in.
  Locally that is your own problem; published it is permanent and everyone's,
  because the store keeps the first writer's result. The cell still runs and
  its result is still cached locally; only the publish is refused, with a
  warning naming the cell. Re-run the environment sync to resume publishing.
- **Point it at an authenticated store.** Without auth there is no tenant, so
  every team shares one flat namespace, and no principal, so every result
  arrives authored by nobody. The server warns at startup if the team cache is
  on with no `notebook_remote_store_headers`.
- **The environment key does not include the platform, but the artifact
  records it.** `uv.lock` resolves to different wheels on macOS-arm64 and
  Linux-x86_64, and the env hash covers the lockfile only. That is deliberate, since
  hashing the platform would drop cross-machine hit rate to roughly zero and
  delete the feature in order to protect it. So a hit can legitimately cross
  machines, and every artifact carries the interpreter and hardware that
  produced it (`cpython-3.14-linux-x86_64`), reported by the process that
  actually ran the cell: the notebook venv locally, or the worker's
  interpreter for a remote cell. A team hit says which platform it came from
  rather than crossing one silently. In practice a shared cache handing the
  whole team one number is the more reproducible outcome, but if your work
  depends on hardware-identical results, leave the team cache off for now.

### Sharing on purpose: `promoted` and `strata artifact promote`

Offering every downstream-consumed variable of every successful cell is right
for a server whose whole purpose is a shared cache. It is wrong for a personal
server: there, it means every intermediate a researcher ever computed lands in
the team's store, whether or not they meant to share it.

`STRATA_NOTEBOOK_TEAM_CACHE_PUBLISH` is the setting between "everything" and
"nothing":

| Value | Offers outward | Pulls |
| --- | --- | --- |
| `all` (default) | every consumed variable of every successful cell | yes |
| `promoted` | nothing automatically | yes |
| `off` | nothing | no |

Under `promoted`, a result reaches the team when someone says so:

```bash
strata artifact promote nb_taxi_cell_c2_var_model \
  --to https://store.example --name taxi/model --alias champion
```

The chain travels with it, and has to. The cache is keyed by provenance, so
each ancestor that arrives is a hit for the next person whose cell computes the
same thing - promoting the result alone would share the answer and none of the
work. Each row the promotion writes is stamped with its name (an `nb_promotion`
tag, kept out of the registry's tag lists), so a colleague's hit on any of it
says which promotion it came from: their profiling panel lists it under "Via
promotions". A row the store already held, from a cache publish or an earlier
promotion, keeps whatever it had. A protected alias (`registry_protected_aliases`) queues for approval
rather than moving, and the command says so.

The Registry tab and the per-cell strip describe that store too. They read the
local one until a team store is configured, then forward to it with the server's
remote-store headers: the notebook names things there, so describing the local
store would show an empty registry. Every registry route forwards the same way
(names, aliases, tags, lineage and approvals), so a promotion from the tab lands
where the tab reads, and a protected alias filed from one person's server gets
approved from another's.

A cell can promote one of its own upstream results without leaving the
notebook, naming it the way the cell reads it:

```python
strata.promote("rows", name="taxi/rows", alias="champion")
```

`off` is the whole feature off without unsetting
`STRATA_NOTEBOOK_REMOTE_STORE_URL`, which a cell's ambient `strata` client
still needs.

### Promoting into an Iceberg table

A tabular result can also become a table in the team's warehouse, so tools
outside Strata read it as a table and a notebook reads it with `@table`:

```bash
strata artifact promote nb_taxi_cell_c1_var_features \
  --to https://store.example --name taxi/features --alias champion \
  --table "s3://lake/warehouse#taxi.features"
```

The team store does the writing, with its own catalog settings, since that is
where the bytes and the warehouse credentials are. `--table` takes the same
forms `@table` reads: `<warehouse>#namespace.table`, or `namespace.table` in the
store's configured catalog.

- The first promotion creates the table and appends. Later promotions overwrite
  it, so the current snapshot is always one version. A table Strata did not
  write is refused rather than replaced. A notebook's `@table` on
  it goes stale when the next version is written, as for any other table.
- Each snapshot's summary names what it holds: `strata.artifact_id`,
  `strata.version`, `strata.provenance_hash`, and `strata.promoted_by` when the
  store knows the caller. (An overwrite commits a delete and then an append; the
  append is the snapshot that holds the version.)
- A new version may add columns or widen a type. One whose schema the table
  cannot evolve to is refused before anything is written.
- The alias is an Iceberg tag on the snapshot. Moving the alias later, including
  approving a protected one, moves the tag, as long as that version was written
  to the table.

The same write is `strata artifact export --table <table> <ref>` against a local
store, and `POST /v1/artifacts/{id}/v/{version}/export` with
`{"table": ..., "alias": ...}` for a platform that exports once a promotion
lands. Only Arrow tables can be written; a JSON value, a pickle, an array or a
scalar is refused.

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
   environment through `/proc`, its config, or other notebooks' files.

Without either, a cell that would run on the server host fails with a message
naming both. Cache hits are still served, since a hit starts no cell code.
Personal mode is unaffected. An `embedded://` executor worker counts as this
host: it runs the harness in place.

Switching users needs the privilege to do it, so **the server runs as root**
and drops to the harness user for every process that runs cell code: the cold
and R harnesses, the batch harness behind Run All, the warm pool workers, the
inspect REPL and cell tests. It is POSIX only. What the harness user needs:

| Path | Access |
| --- | --- |
| notebook directories | read |
| each notebook's `.venv` | read and execute |
| the Python interpreter behind the venvs | read and execute |
| each notebook's `.strata/` | traverse |

The interpreter row is the one that catches people. uv installs managed Pythons
under the installing user's home, which for a root server is `/root` and
unreadable by anyone else, so every cell fails with `Permission denied` on the
venv's `python`. Install them somewhere world-readable with
`UV_PYTHON_INSTALL_DIR=/opt/uv-python`. Nothing else needs arranging: the
per-run directories a cell writes into are handed to the harness user, the
harness runs the venv's interpreter directly rather than through `uv run`, and
the artifact store is never read by the cell.

A cell run this way still shares the host's kernel and sees what any local user
can. A notebook that needs more isolation than that wants a worker on another
machine.

### What a cell is given: `STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST`

The allowlist narrows the environment a cell receives:

```bash
# Only what cells actually need; everything else stays with the server.
STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST=AWS_*,HF_TOKEN
```

Entries are exact names or a prefix with a trailing `*`. The essentials a
subprocess cannot start without are always included, and `STRATA_*` is dropped
unless named exactly - a prefix rule broad enough to catch a credential by
accident is the failure the setting exists to prevent. It applies to every
process that runs cell code, the same list as above.

The list stays short because a cell's own configuration does not come through
the process environment. `[env]` in `notebook.toml` and mount credentials
travel in the cell manifest and are applied inside the harness.

On its own the allowlist is not isolation. A cell running as the server's user
can still read the server's whole environment from `/proc/<server pid>/environ`,
because it *is* that user. The allowlist decides what a cell is handed; the
harness user is what stops it taking the rest.

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
   `transforms_config` block in `pyproject.toml` to expose the
   computations you want the platform to run on the client's
   behalf. The notebook executor in the demo stack is one example.

The demo compose stack is a working starting point you can fork:
swap `nginx.conf` for your real auth proxy config, move the
artifact blobs to S3, and you have most of what production needs.
