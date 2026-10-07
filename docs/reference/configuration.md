# Configuration Reference

Strata is configured via environment variables (prefixed with `STRATA_`) or a `[tool.strata]` section in `pyproject.toml`.

**Precedence**: defaults < pyproject.toml < environment variables < programmatic overrides

The `[tool.strata]` block accepts most of the env vars listed below with the
`STRATA_` prefix dropped and the name lowercased (e.g. `STRATA_HOST` →
`host`, `STRATA_CACHE_DIR` → `cache_dir`, `STRATA_S3_REGION` →
`s3_region`). Values are typed by `StrataConfig` in `src/strata/config.py`;
strings, numbers, booleans, and TOML arrays all work as expected.

Some settings are environment-only and have no `[tool.strata]` equivalent,
because they are read directly by the process that uses them rather than being
fields of `StrataConfig`: everything in the **Worker** section (the
`strata-worker` process), the TUI variables, logging, tracing and metrics, and
the fast-IO tuning. Those rows say so.

```toml
# pyproject.toml
[tool.strata]
host = "0.0.0.0"
port = 8765
cache_dir = "/var/cache/strata"
ai_model = "claude-sonnet-4-6"
```

Multi-tenancy is an access-control boundary, so it is refused without
authentication. This is a startup error, not a warning:

```toml
[tool.strata]
deployment_mode = "service"
artifact_dir = "/var/lib/strata/artifacts"
multi_tenant_enabled = true
auth_mode = "trusted_proxy"
proxy_token = "…"           # or STRATA_PROXY_TOKEN
```

## Server

| Variable                                  | Default     | Description                                  |
| ----------------------------------------- | ----------- | -------------------------------------------- |
| `STRATA_HOST`                             | `127.0.0.1` | Server bind address                          |
| `STRATA_PORT`                             | `8765`      | Server port                                  |
| `STRATA_PUBLIC_BASE_URL`                  | request     | Origin readers reach this server on; set it behind a reverse proxy so published-artifact embed URLs are the public ones |
| `STRATA_PUBLIC_BASE_PATH`                 | _(empty)_   | Path a reverse proxy serves this server under (`/o/acme/lab`), whether or not the proxy strips it. The UI, its WebSocket, the API docs and publication links carry it. Empty means the root. See [Serving under a path](../deployment/modes.md#serving-under-a-path) |
| `STRATA_DEPLOYMENT_MODE`                  | `personal`  | `personal` or `service`                      |
| `STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL` | `false`     | Allow non-localhost clients in personal mode |
| `STRATA_CORS_ALLOW_ORIGINS`               | _(empty)_   | Origins allowed to call the API from a browser, as a JSON array (`["http://localhost:5173"]`); a comma-separated value fails at startup. Empty means no cross-origin access. Personal mode has no auth, so any page allowed here can author and run cells |
| `STRATA_ALLOWED_HOSTS`                    | _(empty)_   | Host names the server answers to, beyond `localhost`, `127.0.0.1`, `[::1]` (any port), IP literals and `STRATA_HOST`. Comma-separated (not a JSON array, which is read as literal names); a leading dot matches a suffix (`.example.com`); `*` turns the check off. Any other `Host` gets 400 (HTTP) or a closed socket (WebSocket), which stops DNS-rebinding pages. Always checked in personal mode; service mode checks only when this is set. List the public hostname of a personal-mode server reached by name (Fly, a LAN name, a Codespace) |
| `STRATA_EMBED_FRAME_ANCESTORS`            | _(empty)_   | Origins allowed to embed a notebook's app view in an `<iframe>` (sets `Content-Security-Policy: frame-ancestors`). Empty means same-origin only. JSON array or comma-separated; `*` allows any host |
| `STRATA_MCP_ENABLED`                      | `false`     | Mount the MCP server at `/mcp` so a coding agent can drive the live session. In service mode it requires principal auth (`trusted_proxy` or `api_key`); each tool call then runs as its caller and is checked against the notebook scopes. Requires the `[mcp]` extra. See [Notebook → MCP](../notebook/mcp.md) |
| `STRATA_ARROW_MEMORY_POOL`                | `None`      | Arrow allocator: `default`, `system`, `jemalloc`, or `mimalloc`. Unset leaves the PyArrow default |

## Cache

| Variable                      | Default                | Description                           |
| ----------------------------- | ---------------------- | ------------------------------------- |
| `STRATA_CACHE_DIR`            | `~/.strata/cache`      | Disk cache location                   |
| `STRATA_MAX_CACHE_SIZE_BYTES` | `10737418240` (10 GB)  | Max cache size                        |
| `STRATA_CACHE_GRANULARITY`    | `row_group_projection` | `row_group_projection` or `row_group` |

## Fetcher

| Variable                       | Default | Description                     |
| ------------------------------ | ------- | ------------------------------- |
| `STRATA_BATCH_SIZE`            | `65536` | Rows per batch                  |
| `STRATA_FETCH_PARALLELISM`     | `4`     | Max concurrent fetches per scan |
| `STRATA_MAX_FETCH_WORKERS`     | `32`    | Max threads in fetch pool       |
| `STRATA_FETCH_TIMEOUT_SECONDS` | `60.0`  | Per-fetch timeout               |
| `STRATA_FAST_CONCAT`           | `rust` when the extension is built, else `pyarrow` | Arrow IPC concat implementation. `pyarrow` parses (slower, handles more edge cases). Environment only |
| `STRATA_MMAP_MIN_BYTES`        | `4194304` (4 MiB) | Cache reads at or above this size go through the Rust mmap path; `0` forces it always. Environment only |

## Resource Limits

| Variable                      | Default              | Description                         |
| ----------------------------- | -------------------- | ----------------------------------- |
| `STRATA_MAX_CONCURRENT_SCANS` | `100`                | Max concurrent scans                |
| `STRATA_MAX_TASKS_PER_SCAN`   | `1000`               | Max row groups per scan             |
| `STRATA_PLAN_TIMEOUT_SECONDS` | `30.0`               | Planning timeout                    |
| `STRATA_SCAN_TIMEOUT_SECONDS` | `300.0`              | Scan streaming timeout              |
| `STRATA_MAX_RESPONSE_BYTES`   | `536870912` (512 MB) | Max response size (413 if exceeded) |
| `STRATA_MAX_EQUALITY_DELETE_ROWS` | `10000000` | Iceberg equality delete rows a row group may need in memory; a scan over it is refused, pointing at compaction |
| `STRATA_STREAM_STATE_TTL_SECONDS` | `300.0`          | How long a completed/abandoned stream's state lingers before cleanup. A `mode="stream"` miss whose stream is never fetched in that time is marked `failed`, and the same request later computes it again |

## QoS (Two-Tier Admission)

| Variable                         | Default            | Description                                |
| -------------------------------- | ------------------ | ------------------------------------------ |
| `STRATA_INTERACTIVE_SLOTS`       | `32`               | Interactive tier concurrency               |
| `STRATA_BULK_SLOTS`              | `8`                | Bulk tier concurrency                      |
| `STRATA_INTERACTIVE_MAX_BYTES`   | `10485760` (10 MB) | Max bytes for interactive classification   |
| `STRATA_INTERACTIVE_MAX_COLUMNS` | `10`               | Max columns for interactive classification |
| `STRATA_INTERACTIVE_QUEUE_TIMEOUT` | `10.0`           | Queue wait for an interactive slot (seconds); exceeding it returns 429 with `Retry-After` |
| `STRATA_BULK_QUEUE_TIMEOUT`      | `30.0`             | Queue wait for a bulk slot (seconds); exceeding it returns 429 with `Retry-After` |
| `STRATA_PER_CLIENT_INTERACTIVE`  | `2`                | Per-client interactive slots; `0` disables per-client caps |
| `STRATA_PER_CLIENT_BULK`         | `1`                | Per-client bulk slots; `0` disables per-client caps |

### Adaptive concurrency

Off by default. When enabled, a background loop resizes the QoS slot counts
from what stream admission observes: queue wait when a scan acquires its tier
slot, and slot-held duration when it releases. Latency over the target walks
the tier down; latency well under it *plus* real queue pressure walks it up.
Hysteresis (`STRATA_ADAPTIVE_HYSTERESIS` consecutive readings in the same
direction) keeps it from flapping, and samples older than 60 seconds stop
counting so an idle tier is not steered by a burst that is over.

`STRATA_ADAPTIVE_TARGET_P95_MS` is compared against how long a scan holds its
slot, which for a bulk scan is the whole build. Set it from observed scan
duration on your deployment, not from a dashboard SLO, or the loop will read
normal work as overload and walk the tier down to its floor.

Two constraints are enforced at startup rather than papered over at runtime:

- `STRATA_INTERACTIVE_SLOTS` and `STRATA_BULK_SLOTS` must fall inside their
  adaptive `[min, max]` range. The controller starts from those counts, so a
  value outside the range means the first adjustment jumps to a bound.
- Adaptive control cannot be combined with `STRATA_MULTI_TENANT_ENABLED`. It
  steers the default tenant's limiters, which under multi-tenancy is a tier no
  request acquires.

| Variable                           | Default | Description                                          |
| ---------------------------------- | ------- | ---------------------------------------------------- |
| `STRATA_ADAPTIVE_ENABLED`          | `false` | Enable the adaptive concurrency controller (single-tenant only) |
| `STRATA_ADAPTIVE_INTERVAL_SECONDS` | `5.0`   | How often the controller evaluates                   |
| `STRATA_ADAPTIVE_TARGET_P95_MS`    | `500.0` | Latency target the controller aims to hold           |
| `STRATA_ADAPTIVE_MIN_INTERACTIVE`  | `4`     | Floor for interactive slots                          |
| `STRATA_ADAPTIVE_MAX_INTERACTIVE`  | `64`    | Ceiling for interactive slots                        |
| `STRATA_ADAPTIVE_MIN_BULK`         | `2`     | Floor for bulk slots                                 |
| `STRATA_ADAPTIVE_MAX_BULK`         | `32`    | Ceiling for bulk slots                               |
| `STRATA_ADAPTIVE_HYSTERESIS`       | `3`     | Consecutive same-direction readings before adjusting |

## Metadata

| Variable             | Default | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `STRATA_METADATA_DB` | `~/.strata/meta.sqlite` | SQLite catalog a personal server uses for an object-store warehouse URI (`s3://`, `gs://`, `abfs://`, `abfss://`) when no catalog `uri` is set. It is on this server's disk, so other readers of the bucket do not see it. A service refuses such a URI instead (see [Catalog](#catalog)) |

## Catalog

| Variable                     | Default   | Description                                                                                       |
| ---------------------------- | --------- | ------------------------------------------------------------------------------------------------- |
| `STRATA_CATALOG_NAME`        | `default` | Iceberg catalog name                                                                              |
| `STRATA_CATALOG_PROPERTIES`  | `{}`      | PyIceberg catalog properties (JSON object via env; `[tool.strata.catalog_properties]` in pyproject) |
| `STRATA_CATALOGS`            | `{}`      | Named catalogs: a JSON object of name to PyIceberg catalog properties (`[tool.strata.catalogs.<name>]` in pyproject), e.g. `{"lake": {"type": "rest", "uri": "https://catalog.example"}}`. A table in one is `<name>:<namespace>.<table>`, for `@table` and scans alike. Credentials a REST catalog vends for a table are used to read that table's files. An entry's `credential` names one in `STRATA_NOTEBOOK_CREDENTIALS` whose fields fill its properties, so no secret is written here (see [Named credentials](notebook-toml.md#named-credentials)) |
| `STRATA_CATALOG_URI`         | `None`    | Catalog database URI. Merged into `catalog_properties.uri`, so it does not replace sibling keys set in pyproject. Environment only |

A SQL catalog keeps its tables under the catalog's name, so
`STRATA_CATALOG_NAME` (or the `STRATA_CATALOGS` key) must match the name
the tables were created under; otherwise a bare `ns.table` (or
`<name>:ns.table`) is "Table not found". A warehouse URI
(`file:///wh#ns.table`, `s3://bucket/wh#ns.table`) always reads its SQL
catalog under the name `strata`.

A SQL catalog (no `type`, or `type = "sql"`) whose `warehouse` is in object
storage (`s3://`, `gs://`, `abfss://`, any scheme but `file://`) needs a `uri`,
in `catalog_properties`, in a `STRATA_CATALOGS` entry, or in the named
credential that entry names. Without one its tables
would be in a SQLite file on this server's disk, so the server refuses to start
and names the setting.

A warehouse named only in a request's table URI (`s3://bucket/wh#ns.table`)
is not known at startup. With no catalog `uri` set, a service
(`STRATA_DEPLOYMENT_MODE=service`) refuses such a URI in any object store with
a 400 naming `STRATA_CATALOG_URI`, on scans, transform inputs, cache warming
and export to a table. A personal server keeps the catalog in `STRATA_METADATA_DB`, so it can
write its own tables there; nothing else reading the bucket sees them.

Such a warehouse's catalog takes `uri` and the storage keys from
`STRATA_CATALOG_PROPERTIES`, but not its `warehouse`: a table an export creates
is written under the warehouse the request names. When the configured catalog
is a REST or Hive one (a `type` other than `sql`, or a `uri` starting with
`http` or `thrift`), a table URI cannot name a warehouse at all; the request is
a 400, and the table is addressed as `<namespace>.<table>` in that catalog.

## S3 Storage

| Variable                 | Default | Description                                      |
| ------------------------ | ------- | ------------------------------------------------ |
| `STRATA_S3_REGION`       | `None`  | AWS region (falls back to AWS_REGION)            |
| `STRATA_S3_ENDPOINT_URL` | `None`  | Custom endpoint (MinIO, LocalStack)              |
| `STRATA_S3_ACCESS_KEY`   | `None`  | Access key (falls back to AWS_ACCESS_KEY_ID)     |
| `STRATA_S3_SECRET_KEY`   | `None`  | Secret key (falls back to AWS_SECRET_ACCESS_KEY) |
| `STRATA_S3_ANONYMOUS`    | `false` | Use anonymous access                             |

## GCS Storage

Credentials for the GCS blob backend (`STRATA_ARTIFACT_BLOB_BACKEND=gcs`).
Unset credentials fall back to Application Default Credentials.

The same settings read lake tables on GCS. A table named by a `gs://`
warehouse URI (`gs://bucket/wh#ns.table`, see [Catalog](#catalog)) has its
metadata read with the bucket location, endpoint and service-account key,
passed to its catalog as `gcs.*` properties. PyIceberg has no anonymous GCS access, so
`STRATA_GCS_ANONYMOUS` covers data files only; set `gcs.oauth2.token` in
`STRATA_CATALOG_PROPERTIES` if the catalog needs a token. A key set there
overrides the one Strata derives.

| Variable                        | Default | Description                                                      |
| ------------------------------- | ------- | ---------------------------------------------------------------- |
| `STRATA_GCS_DEFAULT_BUCKET_LOCATION` | `None` | GCS location new buckets default to (`US`, `europe-west1`). `STRATA_GCS_PROJECT_ID` is still accepted for it and warns: it never set a project, since `GcsFileSystem` has no project parameter |
| `STRATA_GCS_CREDENTIALS_JSON`   | `None`  | Service-account key, as either a path to the JSON file or the JSON itself. Inline key material is written to a private temp file, because `GOOGLE_APPLICATION_CREDENTIALS` only resolves paths. Falls back to `GOOGLE_APPLICATION_CREDENTIALS` |
| `STRATA_GCS_ANONYMOUS`          | `false` | Use anonymous access (public buckets, emulators)                 |
| `STRATA_GCS_ENDPOINT_OVERRIDE`  | `None`  | Custom endpoint (fake-gcs-server and similar)                    |

## Azure Storage

Credentials for the Azure blob backend (`STRATA_ARTIFACT_BLOB_BACKEND=azure`).
Supply at least one of connection string, account key, SAS token, or default
credential; with several set, the connection string wins, then the default
credential, then the SAS token, then the account key.

The same settings read lake tables on Azure. A table named by an `abfs://` or
`abfss://` warehouse URI (see [Catalog](#catalog)) has its metadata read with the account name, key, SAS token, connection string and
endpoint that are set, passed to its catalog as `adls.*` properties; with no
secret set it uses `DefaultAzureCredential`. A key set in
`STRATA_CATALOG_PROPERTIES` overrides the one Strata derives.

`STRATA_AZURE_ENDPOINT_URL` names the blob host (`http://127.0.0.1:10000` for
Azurite), with the account as the first path segment. The artifact blob store
addresses the account the same way, as `<url>/<name>` (a URL that already ends
in `/<name>` is used as given); without an account name it uses the URL as
given. `adlfs` takes such an
endpoint only from a connection string, so with an account name and key set and
no connection string, Strata derives one for the catalog
(`DefaultEndpointsProtocol=<scheme>;AccountName=<name>;AccountKey=<key>;BlobEndpoint=<url>/<name>`).
PyArrow reads the endpoint from `adls.blob-storage-authority` and
`adls.blob-storage-scheme` instead, which are always set; with only a SAS
token or the default credential, the endpoint reaches PyArrow alone.

PyIceberg reads and writes that metadata with `adlfs`, which the `azure` extra
installs. `STRATA_CATALOG_PROPERTIES` can set `py-io-impl` to
`pyiceberg.io.pyarrow.PyArrowFileIO` instead; PyArrow ignores the connection string and
handles only locations of the form `abfs[s]://<container>/<path>`, with the
account from `STRATA_AZURE_ACCOUNT_NAME`. In a location of the form
`abfs[s]://<container>@<account>.dfs.core.windows.net/<path>` it takes
`<container>@<account>.dfs.core.windows.net` for the container, and the request
fails. PyIceberg follows the metadata and manifest locations the table's metadata
records, not the warehouse URI in the request, so naming the warehouse in the
other form changes nothing: a table written under `<container>@<account>`
locations needs `adlfs`. One written under `abfs[s]://<container>/<path>` reads
with either.

| Variable                              | Default | Description                                        |
| ------------------------------------- | ------- | -------------------------------------------------- |
| `STRATA_AZURE_ACCOUNT_NAME`           | `None`  | Storage account name                               |
| `STRATA_AZURE_ACCOUNT_KEY`            | `None`  | Storage account key                                |
| `STRATA_AZURE_CONNECTION_STRING`      | `None`  | Full connection string                             |
| `STRATA_AZURE_SAS_TOKEN`              | `None`  | SAS token                                          |
| `STRATA_AZURE_USE_DEFAULT_CREDENTIAL` | `false` | Use `DefaultAzureCredential` (managed identity)    |
| `STRATA_AZURE_ENDPOINT_URL`           | `None`  | Custom endpoint (Azurite emulator)                 |

## Artifact Storage

| Variable                          | Default     | Description                      |
| --------------------------------- | ----------- | -------------------------------- |
| `STRATA_ARTIFACT_DIR`             | `~/.strata/artifacts` in personal mode, unset in service mode | Artifact store directory. In service mode the store exists only when this is set, even when the metadata DSN and a blob backend hold everything, and startup refuses a DSN, a non-local blob backend or service writes without it |
| `STRATA_ARTIFACT_ZOMBIE_BUILD_TIMEOUT_SECONDS` | `3600.0` | Builds stuck in `building` longer than this are demoted to `failed` at startup |
| `STRATA_ARTIFACT_GC_INTERVAL_SECONDS` | `3600` in personal mode, unset (off) in service mode | How often the server sweeps its artifact store. `0` turns the sweep off. A sweep collects only what nothing holds: nothing named, aliased, pinned or published, nothing those or a running build depend on, and not the current value of an id somebody chose (a notebook's cell outputs). An unnamed `materialize` result is a cache entry; name or pin it to keep it. See [Cleaning up the Core artifact store](../deployment/lifecycle.md#cleaning-up-the-core-artifact-store). |
| `STRATA_ARTIFACT_GC_MAX_BYTES` | `21474836480` (20 GiB) in personal mode, unset in service mode | When the store holds more than this, a sweep collects the least recently used until it is at 80% of it. `0` means no cap. |
| `STRATA_ARTIFACT_GC_MAX_IDLE_DAYS` | `30.0` | A sweep collects what has not been used (a cache hit or a read) for this long, whatever the store's size. `0` means no idle limit. |
| `STRATA_ARTIFACT_GC_MIN_IDLE_SECONDS` | `3600.0` | A sweep never collects anything used more recently than this, so a result just handed to a reader stays. |
| `STRATA_REGISTRY_PROTECTED_ALIASES` | _(empty)_ | Comma-separated alias names (e.g. `champion,production`) whose moves/deletes queue for approval instead of applying |
| `STRATA_ARTIFACT_BLOB_BACKEND`    | `local`     | `local`, `s3`, `gcs`, or `azure` |
| `STRATA_ARTIFACT_S3_BUCKET`       | `None`      | S3 bucket for artifacts          |
| `STRATA_ARTIFACT_S3_PREFIX`       | `artifacts` | S3 key prefix                    |
| `STRATA_ARTIFACT_GCS_BUCKET`      | `None`      | GCS bucket for artifacts         |
| `STRATA_ARTIFACT_GCS_PREFIX`      | `artifacts` | GCS prefix                       |
| `STRATA_ARTIFACT_AZURE_CONTAINER` | `None`      | Azure container                  |
| `STRATA_ARTIFACT_AZURE_PREFIX`    | `artifacts` | Azure prefix                     |
| `STRATA_ARTIFACT_METADATA_DSN`    | `None`      | `postgresql://` URL for the artifact store's metadata. Unset keeps SQLite under `STRATA_ARTIFACT_DIR` |
| `STRATA_STORE_TOKEN`              | `None`      | Bearer token `strata artifact publish --to <url>` presents to the remote store. Read from the environment so it stays out of shell history; `--header` covers anything else a proxy wants |
| `STRATA_NODE_ADVERTISED_URL`      | `None`      | URL that reaches this node. Set only in multi-node deployments; enables stream redirects instead of 404s |

### Sharing one artifact store across nodes

By default the artifact store keeps its metadata in a SQLite file under
`STRATA_ARTIFACT_DIR`, which is local to one machine. Setting
`STRATA_ARTIFACT_METADATA_DSN` moves that metadata to Postgres so several
Strata nodes can share one store.

Requires the `postgres` extra:

```bash
uv pip install 'strata-notebook[postgres]'
export STRATA_ARTIFACT_METADATA_DSN='postgresql://user:pass@db:5432/strata'
export STRATA_ARTIFACT_BLOB_BACKEND=s3
export STRATA_ARTIFACT_S3_BUCKET=my-strata-artifacts
```

**Blobs must be shared too.** In service mode, a DSN with
`STRATA_ARTIFACT_BLOB_BACKEND=local` is rejected at startup: the metadata
would be shared while the bytes stayed on one node's disk, so another node
would resolve an artifact and then fail to read it, at fetch time, long
after the request that created it appeared to succeed.

Build state follows the same backend, since build rows live in the artifact
store's database. That is what makes a build started on one node visible to
`GET /v1/builds/{id}` on another.

The schema is created on first connection. **A DSN starts an empty store**:
existing SQLite metadata is not carried over automatically. To move one:

```bash
# 1. Boot once against the target so the stores create their schema.
STRATA_ARTIFACT_METADATA_DSN='postgresql://...' python -m strata   # then stop it

# 2. See what would move.
strata migrate --to-dsn 'postgresql://...' --dry-run

# 3. Move it.
strata migrate --to-dsn 'postgresql://...'
```

`--artifact-dir` names the source store when it is not `~/.strata/artifacts`.
The copy is idempotent: rows already in the target are skipped, so an
interrupted run can be repeated with `--allow-nonempty-target`. It refuses a
populated target otherwise, because merging two different stores is not
recoverable.

Rows the target refuses are reported and the run continues; `strata migrate`
then **exits non-zero**, so `strata migrate && cut-over` will not switch traffic
to a target that is missing rows. The usual cause is a build row referencing an
artifact version that `garbage_collect` or `delete_artifact` removed, so a
dangling reference SQLite tolerated and Postgres does not.

`--dry-run` makes no schema or data changes. It does open the source with
Strata's normal SQLite settings, which sets `journal_mode=WAL` on the file, the
same as starting the server against it.

**Blobs are not copied.** Point the target deployment at the same blob backend,
or its metadata will resolve to bytes it cannot read. Live stream-ownership
rows are also skipped, since they describe streams that do not survive the
move.

`artifact_builds` carries a foreign key to `artifact_versions`, and both
backends enforce it (Strata's SQLite connections enable `PRAGMA foreign_keys`):
a build row for an artifact version that does not exist is rejected. Strata's
own flow creates the artifact version first, so this only affects callers
writing build rows directly.

## Authentication

| Variable                             | Default              | Description                          |
| ------------------------------------ | -------------------- | ------------------------------------ |
| `STRATA_AUTH_MODE`                   | `none`               | `none`, `trusted_proxy`, or `api_key` |
| `STRATA_PROXY_TOKEN`                 | `None`               | Shared secret for proxy verification |
| `STRATA_PROXY_TOKEN_HEADER`          | `X-Strata-Proxy-Token` | Header carrying the proxy token     |
| `STRATA_PRINCIPAL_HEADER`            | `X-Strata-Principal` | Header for user identity             |
| `STRATA_SCOPES_HEADER`               | `X-Strata-Scopes`    | Header for permission scopes         |
| `STRATA_HIDE_FORBIDDEN_AS_NOT_FOUND` | `true`               | Return 404 instead of 403            |
| `STRATA_SERVICE_WRITES_ENABLED`      | `false`              | **Preview.** Opt-in: let authenticated clients write/publish in service mode (`put`, `set_name`, `set_alias`, tags), scoped to the caller's tenant and gated by the `artifacts:write` scope. Requires `trusted_proxy` auth (enforced at startup). Default keeps service mode read-only. See [Service Mode → shared research store](../deployment/service-mode.md#authenticated-write-back-the-shared-research-store). |

### Access control rules

| Variable            | Default | Description                                       |
| ------------------- | ------- | ------------------------------------------------- |
| `STRATA_ACL_CONFIG` | _(none)_ | Deny/allow rules, as a JSON object. Also settable as `[tool.strata.acl]` |

Evaluation is deny-first: deny rules, then allow rules, then `default`.

```toml
[tool.strata.acl]
default = "deny"

deny = [
  { principal = "*", tables = ["*:finance.*"] },
]

allow = [
  { principal = "bi-dashboard", tables = ["file:analytics.*"] },
  { tenant = "data-platform", tables = ["file:analytics.*"] },
]
```

The same shape as JSON in the env var:

```bash
export STRATA_ACL_CONFIG='{"default":"deny","allow":[{"principal":"bi","tables":["file:analytics.*"]}]}'
```

**A table pattern names the address a table was requested under, not the
table itself.** The prefix is `s3:`, `gs:` or `az:` for a warehouse URI in
object storage, the catalog's own name for a table in a configured catalog
(`STRATA_CATALOGS`, addressed as `<name>:<namespace>.<table>`), and `file:`
for everything else: a `file://` warehouse, any other path, and a bare
`<namespace>.<table>`.

So one table can have several names. A table reachable as
`s3://bucket/wh#finance.ledger` and as `lake:finance.ledger` is
`s3:finance.ledger` under the first and `lake:finance.ledger` under the
second. **Write deny rules with a `*` prefix** so they cover every name:

```toml
deny = [
  { principal = "*", tables = ["*:finance.*"] },
]
```

With a SQL catalog (`catalog_properties` with a `uri`), Strata knows which
names are one table and a deny rule covers all of them. Every warehouse URI
reads that one catalog whatever comes before `#`, so the table behind
`s3://bucket/wh#finance.ledger` is also the one behind a path that doesn't
exist, and behind a bare `finance.ledger` when `catalog_name` is `strata`. A
deny on `s3:finance.*` refuses it under `file:`, `gs:` and `az:` too. Strata
cannot tell that a named catalog holds the same data as a warehouse, which is
what the `*` prefix is for.

An allow rule matches only the name it was written for, even where a
deny would cover the others: an address it misses falls through to
`default`. **If you added a named catalog, or a GCS or Azure warehouse,
check your deny rules**: before this release every warehouse table matched
`file:` whatever store held it, so a rule written then covers less than it
used to.

`principal` and each `tables` entry are glob patterns; `tenant` is an exact
match. Every rule must list at least one table pattern; a rule with none can
never match, so it is rejected at startup rather than sitting inert. An unknown
key is rejected for the same reason, at the top of the block and inside a rule:
a mistyped `deny` would leave an ACL that boots clean and enforces nothing, and
`tenants = "acme"` (the plural) would leave `tenant` unset, so the rule would
apply to every tenant instead of one.

Rules are enforced only when the caller is authenticated. Service mode rejects
configured rules under any other auth mode, so they cannot sit inert. **Personal
mode does not**: `auth_mode` is always `none` there, so rules are accepted at
startup and never evaluated. Do not rely on ACL for a personal deployment.

### API key authentication

`STRATA_AUTH_MODE=api_key` is the mode where Strata authenticates callers
itself, rather than trusting a proxy to have done it. Clients present a key as
a bearer token:

```
Authorization: Bearer strata_<key_id>_<secret>
```

A key resolves to the same principal a proxy header would have produced, so ACL
rules, tenant scoping, and scope checks behave identically across both modes.
The tenant is always the key's: an `X-Tenant-ID` header the caller sends is
ignored, for QoS limiters and log attribution as well as for data access.

Keys are stored in the artifact store's database, which means they follow
whichever metadata backend it uses and are shared across nodes automatically.
Service mode with `auth_mode='api_key'` therefore requires `artifact_dir`;
personal mode rejects the mode outright, since authenticating to your own
loopback-bound single-user server buys nothing.

Mint the first key from the CLI, with no running server needed, which is what
makes bootstrapping possible:

```bash
strata apikey create svc-etl \
  --tenant acme \
  --scope artifacts:write \
  --description "ETL pipeline"

strata apikey list
strata apikey revoke <key_id>
```

The secret is printed once. Only a SHA-256 of it is stored, so it cannot be
shown again, by you or by us, and a database disclosure yields no usable
credentials. The command opens the store the server is configured with
(`[tool.strata]` and `STRATA_*`: its `artifact_dir` and metadata DSN), so it
writes where the server reads; `--artifact-dir` opens one local SQLite store
instead, and `--dsn` names a Postgres metadata store directly. `create
--expires-in-days N` sets an expiry (none by default); `list --principal`
filters to one principal and `--format json` prints JSON.

**Revocation is immediate**: verification reads the row on each request, so a
revoked key stops working at once rather than after a cache expiry. A notebook
WebSocket opened with the key re-checks it on every frame that edits or runs
something and closes on the first one after revocation.

### Running several nodes behind one address

Streams cannot move between nodes. A stream holds a live task and an in-memory
read plan, so only the node that planned it can serve it. A request for it
that lands elsewhere used to return a bare `404`, indistinguishable from an
expired stream.

Set `STRATA_NODE_ADVERTISED_URL` to the address that reaches *this* node:

```bash
export STRATA_NODE_ADVERTISED_URL='https://strata-3.internal:8765'
```

Each node then records which streams it is serving, in the artifact store's
database, and a node asked for a sibling's stream answers `307` pointing at the
owner instead of `404`. This removes the need for session affinity on the
stream-fetch step; the URL must be one clients can actually reach.

Leave it unset for a single-node deployment: nothing is written or read, so
there is no cost. Claims expire with `STRATA_STREAM_STATE_TTL_SECONDS`, which
is what stops a node that died from being advertised indefinitely.

Note this makes stream *fetches* routable, not stream *survival*. A node lost
mid-stream still ends that stream; the artifact behind it is durable and its
build is reclaimable by another node, so the client re-requests and gets a
cache hit or joins the in-flight build.

`STRATA_NODE_ADVERTISED_URL` also carries a remote cell's live console
between nodes. A worker posts console chunks to the server's public address,
so a chunk can land on a node other than the one holding the notebook session
that dispatched the cell.
That node leaves the chunk in the shared build store, and the dispatching node
polls the store every half second while the cell runs and sends the chunks to
the notebook's sockets in order. Single-node deployments neither write nor
poll; there the console goes straight from the log route to the sockets.

## Multi-Tenancy

| Variable                       | Default       | Description                           |
| ------------------------------ | ------------- | ------------------------------------- |
| `STRATA_MULTI_TENANT_ENABLED`  | `false`       | Enable multi-tenant mode              |
| `STRATA_TENANT_HEADER`         | `X-Tenant-ID` | Header for tenant identification      |
| `STRATA_REQUIRE_TENANT_HEADER` | `false`       | Require tenant header on all requests |

## Transforms & Builds

Server-side transform execution and the async build runner (service mode / the
artifact build pipeline). Transforms are also configured via the
`[tool.strata.transforms]` block in `pyproject.toml`; `STRATA_TRANSFORMS_ENABLED`
toggles `enabled` there.

The v2-pull signed-URL routes (build manifest, signed download / upload, and
`finalize`) have no on/off switch. They are served whenever the deployment can
issue and honor them at all (personal mode, or service mode with transforms
enabled), so `STRATA_TRANSFORM_SIGNING_SECRET` below matters in every such
deployment, not only in one that opted in to something. In service mode the
build manifest is issued only under `STRATA_AUTH_MODE=trusted_proxy`, since it
carries upload and finalize capabilities; under any other auth mode it returns
404.

| Variable                                | Default | Description                                                                                     |
| --------------------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `STRATA_TRANSFORM_MODE`                 | `embedded` | Accepted but **not currently wired up**: the registry is always built in embedded mode, so setting `registry` has no effect. Configure transforms through `transforms_config` instead. |
| `STRATA_TRANSFORMS_CONFIG`              | `{}`    | The whole transforms block as a JSON object (`enabled`, `registry`, …). Normally written as `[tool.strata.transforms]` instead; `STRATA_TRANSFORMS_ENABLED` merges into it rather than replacing it. |
| `STRATA_SIGNED_URL_EXPIRY_SECONDS`      | `600`   | Validity window for pull-model signed build URLs. For a notebook cell on a `signed` worker it is a floor: the URLs last at least the cell's timeout plus `STRATA_WORKER_PROVISIONING_TIMEOUT_SECONDS` plus 5 minutes, so a long cell can still upload its result. |
| `STRATA_ARTIFACT_PRESIGNED_URLS` | `false` | Put presigned object-store URLs in build manifests where the blob store can sign them, so a worker's inputs and output bypass the server: S3 with an access key pair or a role (the `s3` extra), GCS with a service-account key or workload identity (the `gcs` extra), Azure with the account key or a managed identity. The output becomes a form upload (`output.fields`) or, on Azure, a `PUT` (`output.method`), which older workers don't send, so enable it once the workers are upgraded. The [executor protocol](executor-protocol.md) says what each store signs with. |
| `STRATA_WORKER_PROVISIONING_TIMEOUT_SECONDS` | `600` | For a worker that answers a dispatch (direct or signed) with 202: how long the job may take to start running. The cell's own timeout starts once it runs. See [Executor protocol](executor-protocol.md). |
| `STRATA_TRANSFORM_SIGNING_SECRET`       | `None`  | HMAC secret signing pull-model build URLs. Unset → a random per-process secret (signed URLs break on restart and differ across replicas); set a stable value for multi-replica / restart-surviving deployments. |
| `STRATA_BUILD_RUNNER_POLL_INTERVAL_MS`  | `500`   | How often the embedded build runner polls for pending builds.                                   |
| `STRATA_BUILD_RUNNER_MAX_CONCURRENT`    | `10`    | Max concurrent builds across the runner.                                                        |
| `STRATA_BUILD_RUNNER_MAX_PER_TENANT`    | `3`     | Max concurrent builds per tenant.                                                               |
| `STRATA_BUILD_RUNNER_DEFAULT_TIMEOUT`   | `300`   | Default per-build timeout (seconds).                                                            |
| `STRATA_BUILD_RUNNER_DEFAULT_MAX_OUTPUT`| `1 GiB` | Default per-build output-size cap (bytes).                                                       |
| `STRATA_BUILD_QOS_INTERACTIVE_SLOTS`    | `16`    | Global interactive build slots.                                                                 |
| `STRATA_BUILD_QOS_BULK_SLOTS`           | `8`     | Global bulk build slots.                                                                         |
| `STRATA_BUILD_QOS_PER_TENANT_INTERACTIVE` | `4`   | Per-tenant interactive build slots.                                                             |
| `STRATA_BUILD_QOS_PER_TENANT_BULK`      | `2`     | Per-tenant bulk build slots.                                                                     |
| `STRATA_BUILD_QOS_INTERACTIVE_TIMEOUT`  | `5`     | Queue wait for an interactive build slot (seconds).                                              |
| `STRATA_BUILD_QOS_BULK_TIMEOUT`         | `15`    | Queue wait for a bulk build slot (seconds).                                                      |
| `STRATA_BUILD_QOS_PER_TENANT_TIMEOUT`   | `1`     | Queue wait for a per-tenant slot (seconds).                                                      |
| `STRATA_BUILD_QOS_BYTES_PER_DAY`        | `None`  | Per-tenant daily output-bytes quota (unset = unlimited).                                         |
| `STRATA_BUILD_QOS_BULK_BYTES_THRESHOLD` | `100 MiB` | Estimated output above this classifies a build as bulk.                                        |
| `STRATA_BUILD_QOS_BULK_INPUTS_THRESHOLD`| `5`     | Input count above this classifies a build as bulk.                                              |

## Notebook

| Variable                            | Default                     | Description                                                    |
| ----------------------------------- | --------------------------- | -------------------------------------------------------------- |
| `STRATA_NOTEBOOK_STORAGE_DIR`       | `~/.strata/notebooks`       | Default notebook storage directory. (Pre-2026-05 default was `/tmp/strata-notebooks`; see [Operations & Lifecycle](../deployment/lifecycle.md#notebook-storage-location) for the migration note.) |
| `STRATA_NOTEBOOK_PYTHON_VERSIONS`   | every uv-installed minor matching Strata's `requires-python` | Available Python versions for new notebooks (JSON array or comma-separated list). Falls back to the server's own minor when uv is unavailable. |
| `STRATA_NOTEBOOK_ENV_BACKEND`       | `uv`                        | How notebook Python environments are kept. `uv`: each notebook has its own `.venv`. `shared`: notebooks with the same `uv.lock` and interpreter build share one environment, and each notebook's `.venv` is a symlink to it, so a second notebook with that lock installs nothing. Adding or removing a package moves only that notebook to another environment. R libraries are shared the same way, one per `renv.lock` and R build, with `renv/library` a symlink. POSIX only. See [Shared environments](../notebook/environment.md#shared-environments). |
| `STRATA_NOTEBOOK_SHARED_ENV_DIR`    | `envs` beside `STRATA_NOTEBOOK_STORAGE_DIR` | Where shared environments live, one directory per lockfile and interpreter. |
| `STRATA_NOTEBOOK_SHARED_ENV_TTL_DAYS` | `7.0`                     | A shared environment no notebook links to is removed once unused this long, by an hourly sweep in the server or `strata env gc`. One a notebook links to is never removed. |
| `STRATA_NOTEBOOK_OBJECT_CODEC` | `cloudpickle` | How a cell's value that is neither Arrow nor JSON is pickled when it is handed to another cell: `cloudpickle` (stdlib `pickle` if cloudpickle is not installed) or `pickle`. Any other value fails the cell's serialization. Environment only. |
| `STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS` | `3` | How many earlier values of each cell output a notebook's own store keeps beside the current one, so reverting a recent edit is still a cache hit. Older ones are pruned each time the server opens the notebook, an already-open one included. `0` turns pruning off and keeps every value. |
| `STRATA_NOTEBOOK_REMOTE_STORE_URL`  | `None`                      | Point the ambient `strata` client injected into cells at a remote shared store instead of this local notebook server, so a team publishes/consumes against one central deployment. Also what the Registry tab and the per-cell strip describe: with this set they forward there, so the dashboard shows the store the notebook actually names things in. Unset → both target the local server. Naming this server's own host and port is rejected at startup, because the registry routes would forward to themselves. See [Service Mode → shared research store](../deployment/service-mode.md#authenticated-write-back-the-shared-research-store). |
| `STRATA_NOTEBOOK_REMOTE_STORE_HEADERS` | `{}`                     | Auth headers the ambient client attaches when pointed at a remote store (e.g. the trusted-proxy identity/token). JSON object; set via env so secrets stay out of committed config. |
| `STRATA_NOTEBOOK_REMOTE_STORE_FORWARD_PRINCIPAL` | `true` | With a caller's principal in context (service mode), send its id as `X-Strata-Principal` to the remote store in place of the one in the static headers, so team-cache attribution, promotions and registry approvals from a shared server name the member. `false` keeps the static identity for every request. |
| `STRATA_NOTEBOOK_TEAM_CACHE_ENABLED` | `false`                    | Consult the remote store on a **local cache miss**, so a colleague's expensive cell becomes your instant result. Distinct from the URL above, which only redirects a cell's ambient client (explicit publish). Opt-in because it is a behaviour change, not only a performance one: it puts bytes another machine produced into your local store. Requires `STRATA_NOTEBOOK_REMOTE_STORE_URL`; enabling it without one is rejected at startup rather than left silently inert. |
| `STRATA_NOTEBOOK_TEAM_CACHE_PUBLISH` | `all` | What the cache offers *outward*: `all` (every downstream-consumed variable of every successful cell), `promoted` (nothing automatically: `strata artifact promote` is how a result reaches the team; pulls are unchanged), or `off` (no offers and no pulls, without unsetting the URL a cell's ambient client still needs). Use `promoted` on a personal server, where offering everything means every intermediate a researcher computes lands in the team's store whether or not they meant to share it. |
| `STRATA_NOTEBOOK_CREDENTIALS` | `{}` | Named credentials as a JSON object, `{name: {field: value}}`. A mount or connection in `notebook.toml`, or a catalog in `STRATA_CATALOGS`, references one with `credential = "<name>"`, so no secret is committed. Values are usually `${VAR}`, resolved against the notebook's environment (where a secret manager puts fetched secrets) and then the server's. A mount's fields become fsspec storage options, a connection's become driver auth, a catalog's become pyiceberg catalog properties. The name is part of provenance; the values are not, so rotation invalidates nothing. See [Named credentials](notebook-toml.md#named-credentials). |
| `STRATA_NOTEBOOK_MOUNT_CREDENTIALS` | `{}` | A default credential per mount URI scheme as JSON, e.g. `{"s3": "org-bucket"}`, applied to mounts that name none. |
| `STRATA_NOTEBOOK_FETCH_ALLOWED_HOSTS` | `[]` | Hosts `@fetch`, and a prompt cell's `[ai] base_url` from `notebook.toml`, may reach even on a private address (an internal data server or model server). Public hosts need no entry; in service mode, private, loopback and link-local addresses are refused unless listed. Personal mode allows them, so `localhost` works on a laptop. Comma-separated (not a JSON array) exact names, or a leading dot for a suffix (`.internal`). Same rule as `STRATA_WORKER_ALLOWED_HOSTS`, and applied to every redirect hop. |
| `STRATA_NOTEBOOK_CELL_LOCK_SECONDS` | `5.0` | How long the last person to change a notebook cell holds it. An edit by someone else inside the window is refused with `cell_locked` and the holder's name, over the WebSocket and over REST, unless it sends `force`. One identity never contends with itself, so a single user in several tabs is unaffected. `0` turns the soft lock off. See the [client protocol](notebook-protocol.md#presence-and-soft-locks). |
| `STRATA_NOTEBOOK_WARM_POOL_SIZE` | `2` | Pre-spawned processes each open notebook keeps ready, so a cell starts with its imports loaded. Applies to the Python pool and, for a notebook with R cells, the R pool. Each holds memory for as long as the notebook is open. `0` turns the pools off: cells start cold. |
| `STRATA_NOTEBOOK_SESSION_TTL_SECONDS` | `14400` (4 h) | An open notebook nobody has edited, run or focused for this long is closed, with a tab connected or not, and its warm processes stop. Syncs, previews and keep-alives do not count; a running cell does. Its results stay on disk and reopening restores them; a connected tab gets `session_closed` and offers to reopen. Checked every minute. See [Session lifetime](notebook-protocol.md#session-lifetime-and-session_closed). |
| `STRATA_NOTEBOOK_MAX_SESSIONS` | `50` | Most notebooks open at once. Beyond it the least recently used is closed, with a tab connected or not. |
| `STRATA_NOTEBOOK_SESSION_MIN_AVAILABLE_MB` | unset (off) | When the host's available memory (`MemAvailable` in `/proc/meminfo`) is below this many MB, the least recently used open notebook is closed, one at a time, until it is above or nothing idle is left. Checked every minute and before each open. Linux only: elsewhere it logs a warning once and does nothing. |
| `STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST` | `[]` (everything) | Which of the server's environment variables a cell subprocess is given. A cell is arbitrary Python, so when it is unset (the default, and right on a laptop) every member who can run a cell can read `STRATA_NOTEBOOK_REMOTE_STORE_HEADERS`, `STRATA_PROXY_TOKEN`, worker tokens and every data-source credential the server holds. Entries are exact names or a prefix written with a trailing `*`; the essentials a subprocess cannot start without (`PATH`, `HOME`, `TMPDIR`, `UV_*`, …) are always included, and `STRATA_*` is dropped unless named exactly. A cell's own `[env]` and mount credentials travel in the manifest rather than the process environment, so the list stays short. Applies to every process that runs cell code: the cold, R and batch harnesses, the warm pool worker, the inspect REPL and cell tests. Not isolation on its own: a cell running as the server's user can read `/proc/<server pid>/environ`; see `STRATA_NOTEBOOK_HARNESS_USER`. |
| `STRATA_NOTEBOOK_HARNESS_USER` | `None` | The OS user cell code runs as. In **service mode**, a cell that would run on the server's own host is refused unless this is set, or the cell is assigned a server-managed worker on another machine. Cache hits are still served. Setting it needs the server to run as root, so it can switch users, and the harness user must be able to read the notebook directories, their `.venv`s and the interpreter behind them (install uv-managed Pythons outside `/root`, e.g. `UV_PYTHON_INSTALL_DIR=/opt/uv-python`). POSIX only. In personal mode nothing is refused, and the user applies only when set. See [Service Mode → What a cell can read](../deployment/service-mode.md#what-a-cell-can-read). |
| `STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES` | `2147483648` (2 GiB) | Per-file cap when reading a remote worker's output bundle; a larger member fails the cell rather than being read into memory. Environment only. Values that don't parse, or are `<= 0`, fall back to the default. |

## TUI

Defaults for the `strata-notebook-tui` client; each is also a command-line
flag, and the flag wins. Environment only.

| Variable                       | Default                 | Description                                                       |
| ------------------------------ | ----------------------- | ----------------------------------------------------------------- |
| `STRATA_TUI_SERVER`            | `http://localhost:8765` | Server the TUI connects to                                        |

## Client

Read by `StrataClient` (the `strata-client` package) when it is constructed
without a URL or a config, to find the server. The server does not read
`STRATA_SERVER_URL`.

| Variable            | Default | Description |
| ------------------- | ------- | ----------- |
| `STRATA_SERVER_URL` | `None`  | The server's full URL, e.g. `https://strata.internal:8765`. Wins over `STRATA_HOST` / `STRATA_PORT`, which win over `[tool.strata]` `host` / `port` in the nearest `pyproject.toml`; with none of them set the client uses `http://127.0.0.1:8765`. |

## Worker

These are read by `strata-worker`, not the main server. They have no effect on a `strata-notebook` process.

| Variable                          | Default              | Description                                                                                  |
| --------------------------------- | -------------------- | -------------------------------------------------------------------------------------------- |
| `STRATA_WORKER_TOKEN`             | `None`               | Optional bearer token. When set, the worker's `/v1/*` execution endpoints require `Authorization: Bearer <token>`. `/health` stays open. See [Workers § Authentication](../notebook/workers.md#authentication). |
| `STRATA_WORKER_LAUNCH_ID`         | `None`               | Reported as `launch_id` in `/health`. Strata sets it, over stdin, on a worker it [launches over SSH](../notebook/workers.md#run-cells-on-a-machine-you-can-ssh-to) and connects only when the worker answering through the tunnel reports it; there is no reason to set it yourself. |
| `STRATA_WORKER_CONNECT_TOKEN`     | `None`               | The token `strata-worker --connect` presents to the relay as `Authorization: Bearer <token>` on its WebSocket handshake. Read once at startup and removed from the environment. Unrelated to `STRATA_WORKER_TOKEN`, which still guards each request. See [Worker Relay Protocol](worker-connect.md). |
| `STRATA_WORKER_MAX_INPUT_BYTES`   | `2147483648` (2 GiB) | Per-input download cap for the pull-model (`/v1/execute-manifest`). Reject inputs larger than this with 413, as soon as the count passes it. Inputs stream to disk, so this bounds disk use, not memory. |
| `STRATA_WORKER_ALLOWED_HOSTS`     | _(empty)_            | Comma-separated hosts whose manifest URLs skip the private-address check, e.g. `build.internal,.svc.cluster.local`. A leading dot is a suffix (anchored on the dot, so `.example.com` does not match `evil-example.com`); anything else must match exactly. Matched on the name, so listing a host is trust in whoever controls its DNS. Prefer this over `STRATA_WORKER_ALLOW_LOCAL_HOSTS` in production. |
| `STRATA_WORKER_ALLOW_LOCAL_HOSTS` | `false`              | Bypass the private-address check for **every** host. For tests and local dev with 127.0.0.1 build servers; in production name the hosts with `STRATA_WORKER_ALLOWED_HOSTS` instead. Setting both gives the wholesale bypass. |
| `STRATA_WORKER_MAX_CONCURRENT`    | unlimited            | Executions the worker runs at once; one more is refused with 503 and `Retry-After`. Same as `--max-concurrent`, which wins. |
| `STRATA_WORKER_GPU_SLOTS`         | none                 | GPUs to hand out, one per execution: the worker sets `CUDA_VISIBLE_DEVICES` for each cell itself. Same as `--gpu-slots`, which wins. |
| `STRATA_WORKER_ENV_ROOT`          | `~/.strata/worker-envs` | Where the worker keeps one locked environment per notebook `uv.lock` and interpreter build, and under `r/` one R library per `renv.lock` and R build. See [the `environment` block](executor-protocol.md#the-environment-block). |
| `STRATA_WORKER_ENV_REGISTRY_URL`  | `None`               | Fetch a missing locked environment from `<url>/<key>/<interpreter>/<platform>` as a `.tar.gz` instead of building it with `uv sync`, and a missing R library from `<url>/r/<key>/<R version>/<platform>` instead of restoring it with renv; a `404` is built locally. See [the `environment` block](executor-protocol.md#the-environment-block). |

The worker's input downloads, result uploads and log forwarding connect only to an address that passed the private-address check, and they connect directly: `HTTPS_PROXY` and the other proxy variables are ignored, because through a proxy the proxy would resolve the name and the check would say nothing about where it connects. A worker that can reach its store only through a proxy needs `STRATA_WORKER_ALLOW_LOCAL_HOSTS`, under which nothing is checked and the proxy variables apply. See the SSRF defenses in the [executor protocol](executor-protocol.md).

## Rate Limiting

| Variable                       | Default  | Description                    |
| ------------------------------ | -------- | ------------------------------ |
| `STRATA_RATE_LIMIT_ENABLED`    | `true`   | Enable rate limiting           |
| `STRATA_RATE_LIMIT_GLOBAL_RPS` | `1000.0` | Global requests per second     |
| `STRATA_RATE_LIMIT_CLIENT_RPS` | `100.0`  | Per-client requests per second |
| `STRATA_RATE_LIMIT_SCAN_RPS`   | `50.0`   | Scan endpoint rate limit       |
| `STRATA_RATE_LIMIT_WARM_RPS`   | `10.0`   | Cache-warm endpoint rate limit |
| `STRATA_RATE_LIMIT_GLOBAL_BURST` | `100.0` | Global token-bucket burst     |
| `STRATA_RATE_LIMIT_CLIENT_BURST` | `20.0` | Per-client token-bucket burst  |

## Observability

| Variable                      | Default  | Description             |
| ----------------------------- | -------- | ----------------------- |
| `STRATA_LOG_LEVEL`            | `INFO`   | Log level. Environment only |
| `STRATA_LOG_FORMAT`           | `json`   | `json` or `text`. Environment only |
| `STRATA_TRACING_ENABLED`      | `true`   | A kill switch, not an opt-in: set `false` to disable tracing. No effect unless the `[otel]` extra is installed, which is what keeps it off by default. Environment only. |
| `STRATA_METRICS_ENABLED`      | `true`   | Set `false` to stop collecting request/cache metrics. Environment only |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `None`   | OTLP collector endpoint |
| `OTEL_SERVICE_NAME`           | `strata` | Service name for traces |

## AI (prompt cells)

The `STRATA_AI_*` variables are the server's defaults, read at startup. The
provider keys are read from the **notebook's** environment (the Runtime panel,
`[env]` in `notebook.toml`, or a secret manager), not from the server's shell,
so a key exported where the server starts does not reach every notebook.

| Variable                       | Default  | Description                                                  |
| ------------------------------ | -------- | ------------------------------------------------------------ |
| `STRATA_AI_BASE_URL`           | `None`   | OpenAI-compatible API base URL                               |
| `STRATA_AI_MODEL`              | `None`   | Model identifier (e.g. `claude-sonnet-4-6`, `gpt-5.4`)       |
| `STRATA_AI_API_KEY`            | `None`   | API key (generic, works with any provider). Also read from the notebook's environment, where it overrides the server's key (a provider key there wins over it) |
| `STRATA_AI_MAX_OUTPUT_TOKENS`  | `4096`   | Max output tokens requested                                  |
| `STRATA_AI_TIMEOUT_SECONDS`    | `60.0`   | AI request timeout                                           |
| `ANTHROPIC_API_KEY`            | `None`   | Anthropic API key, from the notebook's environment (auto-sets base URL + model) |
| `OPENAI_API_KEY`               | `None`   | OpenAI API key, from the notebook's environment (auto-sets base URL + model) |
| `GEMINI_API_KEY`               | `None`   | Google Gemini API key, from the notebook's environment (auto-sets base URL + model) |
| `MISTRAL_API_KEY`              | `None`   | Mistral API key, from the notebook's environment (auto-sets base URL + model) |

Precedence, highest first: `[ai]` in `notebook.toml`, then a provider key in
the notebook's environment, then the server's `STRATA_AI_*`. A provider key
also sets its provider's `base_url` and `model`, so it replaces the server's
`STRATA_AI_BASE_URL` and `STRATA_AI_MODEL` too. In service mode a `base_url` from
`notebook.toml` is checked like an `@fetch` URL: a private, loopback or
link-local address is refused unless its host is in
`STRATA_NOTEBOOK_FETCH_ALLOWED_HOSTS`. `STRATA_AI_BASE_URL` is the operator's
and is not checked, nor is a notebook naming that same URL or a provider's
default. In every mode the server's `STRATA_AI_API_KEY` goes only to those
trusted URLs: a notebook whose `[ai] base_url` names another host needs its own
key (the Runtime panel, or `[ai] api_key`).

```toml
[ai]
api_key = ""              # prefer the Runtime panel; writing here commits the key
base_url = "http://localhost:11434/v1"
model = "llama3"
max_output_tokens = 4096
timeout_seconds = 60.0
```

All fields are optional, set only the ones you want to override.

## Timeouts

| Variable                            | Default | Description            |
| ----------------------------------- | ------- | ---------------------- |
| `STRATA_S3_CONNECT_TIMEOUT_SECONDS` | `10.0`  | S3 connection timeout  |
| `STRATA_S3_REQUEST_TIMEOUT_SECONDS` | `30.0`  | S3 request timeout     |
| `STRATA_PLAN_TIMEOUT_SECONDS`       | `30.0`  | Planning phase timeout |
| `STRATA_SCAN_TIMEOUT_SECONDS`       | `300.0` | Scan streaming timeout |
| `STRATA_FETCH_TIMEOUT_SECONDS`      | `60.0`  | Per-fetch timeout      |
