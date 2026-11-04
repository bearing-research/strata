# Team store

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
  isolated. A name, alias or tag aimed at another tenant's artifact answers
  `404 Artifact not found`, the same as a missing artifact or a read of it.
  An upload or import that names an artifact id another tenant already holds,
  at any version, answers `409`.
  The publishing principal is recorded in the registry audit.

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
strata.put(inputs=[], transform={"ref": "clean@v1"}, data=cleaned, name="team/cleaned-events")

# Any teammate (team-a) resolves the name to its current artifact and reads it:
info = strata.resolve_name("team/cleaned-events")  # {artifact_uri, version, …}

# Other-team principals (team-b) cannot resolve team-a's name - tenant isolation.
```

## Connecting a notebook to the shared store

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

## The team cache: sharing results nobody named

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
and no cell id, and `env_hash` leaves out the notebook's own project name in
`uv.lock`. Two people running the same source over the same inputs in the
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

On a shared notebook server the author column names the member who ran each
cell: every artifact the run stores, variables and displays alike, records the
principal of the request that started it (a WebSocket run, the REST execute
route, or an MCP tool call), even when another member is watching the same
notebook. Promoting a result keeps those authors, so the team store says who
computed each step, not who promoted it. In personal mode there is no
authenticated identity, so the column fills in only for steps that came *from*
the store and stays blank throughout on a solo notebook. The platform and
environment identity are recorded on every run, so what a shared store changes
is not that they appear
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
- **A [`# @output_dir`](https://bearingresearch.com/strata/dev/reference/annotations/python/#output_dir) folder travels
  with its files.** A publish offers each file the folder names, one blob each
  under its own provenance key (a file the store already holds is not sent
  again), then the manifest naming where each landed. A pull fetches every file
  and checks its sha256 and size against the manifest before anything lands;
  the files land under the pulling notebook's own ids, then the manifest. A
  missing or altered file lands nothing and the cell runs locally. A stored
  folder whose files are gone is a miss, and the next publish of it replaces it.
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

## Sharing on purpose: `promoted` and `strata artifact promote`

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

`--tag KEY=VALUE` (repeatable) tags the promoted version, `--header 'Name:
value'` (repeatable) adds a header for `--to` beyond the `STRATA_STORE_TOKEN`
bearer, and `--max-depth` (default 10) bounds how far the chain is walked.

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

## Promoting into an Iceberg table

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
- A table belongs to the tenant whose export last wrote it. Another tenant's
  export to it is refused like a denied table (`404 Table not found`, or `403`
  with `STRATA_HIDE_FORBIDDEN_AS_NOT_FOUND=false`), even when the ACL lets that
  tenant read it. An `admin:*` export is not tenant-scoped, and leaves a table
  only another unscoped export can replace.
- Each snapshot's summary names what it holds: `strata.artifact_id`,
  `strata.version`, `strata.provenance_hash`, `strata.promoted_by` when the
  store knows the caller, and `strata.tenant` when the export was
  tenant-scoped. (An overwrite commits a delete and then an append; the
  append is the snapshot that holds the version.)
- A new version may add columns or widen a type. One whose schema the table
  cannot evolve to is refused before anything is written.
- The alias is an Iceberg tag on the snapshot. Moving the alias later, including
  approving a protected one, moves the tag, as long as that version was written
  to the table. When two moves overlap, the tag ends on the version the alias
  ends on.

The same write is `strata artifact export --table <table> <ref>` against a local
store (`--alias` tags the new snapshot, `--by` is recorded as
`strata.promoted_by`), and `POST /v1/artifacts/{id}/v/{version}/export` with
`{"table": ..., "alias": ...}` for a platform that exports once a promotion
lands. Only Arrow tables can be written; a JSON value, a pickle, an array or a
scalar is refused.
