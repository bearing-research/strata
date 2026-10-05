# Privacy & Sharing Model

A personal server has one user, so notebook access there is
**URL-based**, similar to Google Docs: notebook IDs are unguessable,
and anyone who can reach the server and has the ID can open and
execute the notebook. Service mode gates the routes by scope and
tenant. This page lays out the model honestly so you can pick the
deployment shape that matches your trust boundary.

## What's shared, what isn't

### Iceberg scan cache, shared by design

The original Strata value prop: identical scans hit the same cache.
The key is content-addressed:

```
hash(tenant | table_identity | snapshot_id | file_path | row_group_id | projection_fingerprint)
```

If Alice ran a scan an hour ago and Bob runs the same scan now, Bob
hits Alice's cached result. **This is intentional**: it's the
performance win that makes Strata interesting. In multi-tenant
service mode, the `tenant` term in the hash isolates one tenant
from another.

### Notebook artifacts, not actually shared

Per-cell variable outputs are stored in each notebook's own
`.strata/artifacts/` store as
`nb_{notebook_id}_cell_{cell_id}_var_{name}`. Two users with
identical notebook code but different notebooks each produce their
own artifacts under different `notebook_id`s, they don't dedupe
across notebooks. So sharing a Strata instance with a teammate
does **not** mean your cell outputs cross-pollinate. The exception
is opt-in: the [team cache](service-mode.md#the-team-cache-sharing-results-nobody-named)
(`STRATA_NOTEBOOK_TEAM_CACHE_ENABLED`) offers results to a shared
store by provenance, so identical cells in different notebooks hit
each other there.

### Notebook access

Notebook IDs are full UUIDs (8-char prefix for display, full UUID
for the actual ID). They're not in any global enumeration.

On a personal server there is no identity to check against: `discover`
lists every notebook under the storage root, and anyone who can reach
the port can open anything. That is the single-user shape, and it is why
personal mode binds to loopback by default. Use
[publishing](../notebook/publishing.md) to hand someone a result, or
service mode for a genuinely shared deployment.

## Trust boundaries, pick a shape

### Single developer (default)

`STRATA_DEPLOYMENT_MODE=personal` on localhost, no proxy. The
caller is you, every notebook is yours, sharing isn't on the table.
Use this shape unless something else applies.

### A team

Give each member a personal server (their own machine, or one container
each behind your proxy) and share results through a service-mode store:
see [Connecting a notebook to the shared store](service-mode.md#connecting-a-notebook-to-the-shared-store).
Share results by [publishing](../notebook/publishing.md) them rather
than by sending a notebook link. For one server the whole team uses,
run service mode (next section).

### Multi-tenant or hard-isolation requirements

`STRATA_DEPLOYMENT_MODE=service` + multi-tenancy. Each tenant gets
its own QoS pools, cache namespacing, and metric labels. Notebook routes
are scope gated (`notebook:read`, `notebook:write`, `notebook:execute`),
so what a principal can reach depends on the scopes the proxy asserts
for it.

Notes:

- **Notebooks are tenant-scoped, not per-principal.** A session
  records the tenant of the caller who opened it and looks missing to
  other tenants, and a multi-tenant server gives each tenant its own
  subdirectory of the storage root. Within a tenant there are no
  per-notebook ACLs: if Alice's notebook ID leaks to Bob in her tenant
  and Bob holds `notebook:read`, he can open it.
- **Notebook deletion is personal-mode only.** Service mode refuses
  `DELETE /v1/notebooks/{id}` and `delete-by-path`.

### Per-user isolation (every user truly private)

**Not supported in a single instance.** If you need every user to
have a notebook namespace nobody else can ever access, run one
Strata instance per user. The deployment cost is real (one process,
one venv, one storage volume per user) but the isolation is total.
Per-user containers behind a routing proxy is a reasonable
implementation.

This is the same answer you'd get from JupyterHub, Marimo Cloud, or
any other notebook tool: hard-private = separate instances.

## Future direction: per-notebook ACLs

Real per-notebook permissions (a `read_principals` / `write_principals`
list in `notebook.toml` checked on every endpoint) would sit between
the two shapes that exist today: a personal server has one user, and
in service mode anyone in the tenant who has the ID and the scope can
open a notebook. Nothing in between is implemented:

- [Publishing](../notebook/publishing.md) already hands a result to
  someone without handing them the notebook.
- The handful of users who need stronger isolation already have
  the "separate instance" escape hatch.

If you have a concrete use case that needs per-notebook ACLs,
file an issue describing the workflow, that's the right way to
move this off the future-work list.
