# Privacy & Sharing Model

Strata's sharing model is **URL-based**, similar to Google Docs:
notebook IDs are unguessable, and anyone with the ID can open and
execute the notebook. Some endpoints (delete, rename) are owner-
gated; the rest are not. This page lays out the model honestly so
you can pick the deployment shape that matches your trust
boundary.

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

Per-cell variable outputs are stored as
`nb_{notebook_id}_cell_{cell_id}_var_{name}.arrow`. Two users with
identical notebook code but different notebooks each produce their
own artifacts under different `notebook_id`s, they don't dedupe
across notebooks. So sharing a Strata instance with a teammate
does **not** mean your cell outputs cross-pollinate.

### Notebook access, URL-based

Notebook IDs are full UUIDs (8-char prefix for display, full UUID
for the actual ID). They're not in any global enumeration and
they're not in `discover`'s output unless the caller owns them.
What happens next depends on whether `STRATA_PERSONAL_MODE_USER_HEADER`
is configured.

**With per-user scoping on**, knowing the id is not enough. Every route
that takes a session id resolves it through one dependency,
`get_notebook_session`, which checks the notebook's recorded owner against
the caller's identity and answers `404` on a mismatch. The WebSocket upgrade
does the same and closes with `1008`. A missing header is denied too, so a
caller who sends nothing is not treated as everyone.

| Endpoint | Owner check? | How |
|---|---|---|
| `GET /v1/notebooks/{id}/cells` | **Yes** | `get_notebook_session`, 404 to non-owners |
| `POST /v1/notebooks/{id}/cells/{cell_id}/execute` | **Yes** | Same dependency |
| `GET /v1/notebooks/{id}/dag` | **Yes** | Same dependency |
| `WS /v1/notebooks/ws/{id}` | **Yes** | Upgrade closes with 1008 |
| `GET /v1/notebooks/discover` | **Yes** | Filters by owner, and scans only the caller's own storage root |
| `POST /v1/notebooks/open` | **Yes** | The path must lie inside the caller's own storage root |
| `DELETE /v1/notebooks/{id}` | **Yes** | 404 to non-owners |
| `POST /v1/notebooks/delete-by-path` | **Yes** | 404 to non-owners |
| `PUT /v1/notebooks/{id}/name` | **Yes** | 404 to non-owners |

The generic `404` is deliberate: a `403` would confirm that a notebook with
that id exists.

**With the header unset** there is no identity to check against, every
notebook is unowned, and anyone who can reach the port can open anything.
That is the single-user shape, and it is why personal mode binds to loopback
by default.

Sharing a link with a teammate therefore does not work under per-user
scoping, because their path resolution is confined to their own root. Use
[publishing](../notebook/publishing.md) to hand someone a result, or service
mode for a genuinely shared deployment.

## How notebook ownership gets stamped

The `owner` field on `notebook.toml` is set when:

- **Personal-mode-with-proxy** (`STRATA_PERSONAL_MODE_USER_HEADER`
  set) `POST /create` and `POST /import` stamp the caller's
  identity from the configured header (typically
  `Cf-Access-Authenticated-User-Email`, `X-Forwarded-Email`, etc.).
- **Service mode**: `_caller_identity(request)` resolves from
  `X-Strata-Principal` and stamps the same way.

When `personal_mode_user_header` is unset and the caller has no
identity, `owner` stays `None`, all notebooks are unowned and the
single-user pattern applies. This is the default for a developer
running on localhost.

Unowned notebooks (`owner is None`) remain accessible to any
caller. Migrating an unowned notebook to ownership requires
manually editing `notebook.toml`.

## Trust boundaries, pick a shape

### Single developer (default)

`STRATA_DEPLOYMENT_MODE=personal` on localhost, no proxy. The
caller is you, every notebook is yours, sharing isn't on the table.
Use this shape unless something else applies.

### Small trusted team (5–20 people)

`STRATA_DEPLOYMENT_MODE=personal` + an authenticating proxy
(Cloudflare Access, Pomerium, corporate SSO) +
`STRATA_PERSONAL_MODE_USER_HEADER=Cf-Access-Authenticated-User-Email`
(or whatever your proxy injects).

Every notebook is stamped with its creator's identity. `discover`
filters to your notebooks. Direct URL access still works for
collaboration. Delete / rename are owner-only.

**This is the right shape for most teams.** Trust the URL boundary,
share by sending the link, accept that anyone with the link can
look at the notebook. If you wouldn't put it in a shared Google Doc,
don't put it in a shared Strata notebook.

### Multi-tenant or hard-isolation requirements

`STRATA_DEPLOYMENT_MODE=service` + multi-tenancy. Each tenant gets
its own QoS pools, cache namespacing, and metric labels. Notebook routes
are scope gated, so what a principal can reach depends on the scopes the
proxy asserts for it.

Notes:

- **The URL-access property still holds within a tenant.** Multi-
  tenancy isolates one tenant from another; it does NOT add per-
  notebook ACLs within a tenant. If Alice and Bob are on the same
  tenant and Alice's notebook ID leaks to Bob, he can open it.
- **Cross-tenant isolation is hard.** Tenant A and tenant B can't
  see each other's cache, artifacts, or notebooks, the tenant
  dimension is hashed into every key.

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
list in `notebook.toml` checked on every endpoint) would close the
"anyone-with-URL" gap. It's not implemented today; the URL boundary
is the deliberate choice because:

- Collaboration via shared URLs is the dominant pattern in
  notebook workflows.
- Locking down opens / cell execution introduces friction that
  doesn't match how teams typically share work.
- The handful of users who need stronger isolation already have
  the "separate instance" escape hatch.

If you have a concrete use case that needs per-notebook ACLs,
file an issue describing the workflow, that's the right way to
move this off the future-work list.
