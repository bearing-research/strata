# Configuration Reference

Strata is configured via environment variables (prefixed with `STRATA_`) or a `[tool.strata]` section in `pyproject.toml`.

**Precedence**: defaults < pyproject.toml < environment variables < programmatic overrides

The `[tool.strata]` block accepts every env var listed below with the
`STRATA_` prefix dropped and the name lowercased (e.g. `STRATA_HOST` →
`host`, `STRATA_CACHE_DIR` → `cache_dir`, `STRATA_S3_REGION` →
`s3_region`). Values are typed by `StrataConfig` in `src/strata/config.py`;
strings, numbers, booleans, and TOML arrays all work as expected.

```toml
# pyproject.toml
[tool.strata]
host = "0.0.0.0"
port = 8765
deployment_mode = "service"
cache_dir = "/var/cache/strata"
multi_tenant_enabled = true
ai_model = "claude-sonnet-4-6"
```

## Server

| Variable                                  | Default     | Description                                  |
| ----------------------------------------- | ----------- | -------------------------------------------- |
| `STRATA_HOST`                             | `127.0.0.1` | Server bind address                          |
| `STRATA_PORT`                             | `8765`      | Server port                                  |
| `STRATA_DEPLOYMENT_MODE`                  | `personal`  | `personal` or `service`                      |
| `STRATA_ALLOW_REMOTE_CLIENTS_IN_PERSONAL` | `false`     | Allow non-localhost clients in personal mode |
| `STRATA_CORS_ALLOW_ORIGINS`               | _(empty)_   | Origins allowed to call the API from a browser. Empty means no cross-origin access — personal mode has no auth, so any page allowed here can author and run cells |
| `STRATA_EMBED_FRAME_ANCESTORS`            | _(empty)_   | Origins allowed to embed a notebook's app view in an `<iframe>` (sets `Content-Security-Policy: frame-ancestors`). Empty means same-origin only. JSON array or comma-separated; `*` allows any host |
| `STRATA_MCP_ENABLED`                      | `false`     | Mount the MCP server at `/mcp` so a coding agent can drive the live session. **Personal mode only** (rejected at startup in service mode) and requires the `[mcp]` extra. See [Notebook → MCP](../notebook/mcp.md) |
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
| `STRATA_FAST_CONCAT`           | `rust` when the extension is built, else `pyarrow` | Arrow IPC concat implementation. `pyarrow` parses (slower, handles more edge cases) |
| `STRATA_MMAP_MIN_BYTES`        | `4194304` (4 MiB) | Cache reads at or above this size go through the Rust mmap path; `0` forces it always |

## Resource Limits

| Variable                      | Default              | Description                         |
| ----------------------------- | -------------------- | ----------------------------------- |
| `STRATA_MAX_CONCURRENT_SCANS` | `100`                | Max concurrent scans                |
| `STRATA_MAX_TASKS_PER_SCAN`   | `1000`               | Max row groups per scan             |
| `STRATA_PLAN_TIMEOUT_SECONDS` | `30.0`               | Planning timeout                    |
| `STRATA_SCAN_TIMEOUT_SECONDS` | `300.0`              | Scan streaming timeout              |
| `STRATA_MAX_RESPONSE_BYTES`   | `536870912` (512 MB) | Max response size (413 if exceeded) |
| `STRATA_STREAM_STATE_TTL_SECONDS` | `300.0`          | How long a completed/abandoned stream's state lingers before cleanup |

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
| `STRATA_METADATA_DB` | `None`  | SQLite path for metadata persistence |

## Catalog

| Variable                     | Default   | Description                                                                                       |
| ---------------------------- | --------- | ------------------------------------------------------------------------------------------------- |
| `STRATA_CATALOG_NAME`        | `default` | Iceberg catalog name                                                                              |
| `STRATA_CATALOG_PROPERTIES`  | `{}`      | PyIceberg catalog properties (JSON object via env; `[tool.strata.catalog_properties]` in pyproject) |
| `STRATA_CATALOG_URI`         | `None`    | Catalog database URI. Merged into `catalog_properties.uri`, so it does not replace sibling keys set in pyproject |

## S3 Storage

| Variable                 | Default | Description                                      |
| ------------------------ | ------- | ------------------------------------------------ |
| `STRATA_S3_REGION`       | `None`  | AWS region                                       |
| `STRATA_S3_ENDPOINT_URL` | `None`  | Custom endpoint (MinIO, LocalStack)              |
| `STRATA_S3_ACCESS_KEY`   | `None`  | Access key (falls back to AWS_ACCESS_KEY_ID)     |
| `STRATA_S3_SECRET_KEY`   | `None`  | Secret key (falls back to AWS_SECRET_ACCESS_KEY) |
| `STRATA_S3_ANONYMOUS`    | `false` | Use anonymous access                             |

## GCS Storage

Credentials for the GCS blob backend (`STRATA_ARTIFACT_BLOB_BACKEND=gcs`).
Unset credentials fall back to Application Default Credentials.

| Variable                        | Default | Description                                                      |
| ------------------------------- | ------- | ---------------------------------------------------------------- |
| `STRATA_GCS_PROJECT_ID`         | `None`  | Despite the name, this is passed to PyArrow as the **default bucket location** (`US`, `europe-west1`), not a project id ([#604](https://github.com/bearing-research/strata/issues/604)) |
| `STRATA_GCS_CREDENTIALS_JSON`   | `None`  | **Path** to a service-account key file (it is assigned to `GOOGLE_APPLICATION_CREDENTIALS`, which the Google client resolves as a path). Inline JSON does not work ([#605](https://github.com/bearing-research/strata/issues/605)) |
| `STRATA_GCS_ANONYMOUS`          | `false` | Use anonymous access (public buckets, emulators)                 |
| `STRATA_GCS_ENDPOINT_OVERRIDE`  | `None`  | Custom endpoint (fake-gcs-server and similar)                    |

## Azure Storage

Credentials for the Azure blob backend (`STRATA_ARTIFACT_BLOB_BACKEND=azure`).
Supply exactly one of connection string, account key, SAS token, or default
credential.

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
| `STRATA_ARTIFACT_DIR`             | `None`      | Artifact store directory         |
| `STRATA_ARTIFACT_ZOMBIE_BUILD_TIMEOUT_SECONDS` | `3600.0` | Builds stuck in `building` longer than this are demoted to `failed` at startup |
| `STRATA_REGISTRY_PROTECTED_ALIASES` | _(empty)_ | Comma-separated alias names (e.g. `champion,production`) whose moves/deletes queue for approval instead of applying |
| `STRATA_ARTIFACT_BLOB_BACKEND`    | `local`     | `local`, `s3`, `gcs`, or `azure` |
| `STRATA_ARTIFACT_S3_BUCKET`       | `None`      | S3 bucket for artifacts          |
| `STRATA_ARTIFACT_S3_PREFIX`       | `artifacts` | S3 key prefix                    |
| `STRATA_ARTIFACT_GCS_BUCKET`      | `None`      | GCS bucket for artifacts         |
| `STRATA_ARTIFACT_GCS_PREFIX`      | `artifacts` | GCS prefix                       |
| `STRATA_ARTIFACT_AZURE_CONTAINER` | `None`      | Azure container                  |
| `STRATA_ARTIFACT_AZURE_PREFIX`    | `artifacts` | Azure prefix                     |
| `STRATA_ARTIFACT_METADATA_DSN`    | `None`      | `postgresql://` URL for the artifact store's metadata. Unset keeps SQLite under `STRATA_ARTIFACT_DIR` |
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
would resolve an artifact and then fail to read it — at fetch time, long
after the request that created it appeared to succeed.

Build state follows the same backend, since build rows live in the artifact
store's database. That is what makes a build started on one node visible to
`GET /v1/builds/{id}` on another.

The schema is created on first connection. **A DSN starts an empty store** —
existing SQLite metadata is not carried over automatically. To move one:

```bash
# 1. Boot once against the target so the stores create their schema.
STRATA_ARTIFACT_METADATA_DSN='postgresql://...' python -m strata   # then stop it

# 2. See what would move.
strata migrate --to-dsn 'postgresql://...' --dry-run

# 3. Move it.
strata migrate --to-dsn 'postgresql://...'
```

The copy is idempotent — rows already in the target are skipped, so an
interrupted run can be repeated with `--allow-nonempty-target`. It refuses a
populated target otherwise, because merging two different stores is not
recoverable.

Rows the target refuses are reported and the run continues; `strata migrate`
then **exits non-zero**, so `strata migrate && cut-over` will not switch traffic
to a target that is missing rows. The usual cause is a build row referencing an
artifact version that `garbage_collect` or `delete_artifact` removed — a
dangling reference SQLite tolerated and Postgres does not.

`--dry-run` makes no schema or data changes. It does open the source with
Strata's normal SQLite settings, which sets `journal_mode=WAL` on the file, the
same as starting the server against it.

**Blobs are not copied.** Point the target deployment at the same blob backend,
or its metadata will resolve to bytes it cannot read. Live stream-ownership
rows are also skipped, since they describe streams that do not survive the
move.

One behavioral difference worth knowing: `artifact_builds` carries a foreign
key to `artifact_versions`, and **Postgres enforces it while SQLite does not**
(Strata never enables `PRAGMA foreign_keys`). A build row for an artifact
version that does not exist is rejected on Postgres and silently accepted on
SQLite. Strata's own flow creates the artifact version first, so this only
affects callers writing build rows directly.

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
  { principal = "*", tables = ["file:finance.*"] },
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

`principal` and each `tables` entry are glob patterns; `tenant` is an exact
match. Every rule must list at least one table pattern — a rule with none can
never match, so it is rejected at startup rather than sitting inert. Unknown
keys are rejected for the same reason: a mistyped `deny` would otherwise leave
an ACL that boots clean and enforces nothing.

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

Keys are stored in the artifact store's database, which means they follow
whichever metadata backend it uses and are shared across nodes automatically.
Service mode with `auth_mode='api_key'` therefore requires `artifact_dir`;
personal mode rejects the mode outright, since authenticating to your own
loopback-bound single-user server buys nothing.

Mint the first key from the CLI — no running server needed, which is what
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
shown again — by you or by us — and a database disclosure yields no usable
credentials. Pass `--dsn` (or set `STRATA_ARTIFACT_METADATA_DSN`) when the
metadata lives on Postgres, so the CLI writes where the server reads.

**Revocation is immediate**: verification reads the row on each request, so a
revoked key stops working at once rather than after a cache expiry.

### Running several nodes behind one address

Streams cannot move between nodes. A stream holds a live task and an in-memory
read plan, so only the node that planned it can serve it — a request for it
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
issue and honor them at all — personal mode, or service mode with transforms
enabled — so `STRATA_TRANSFORM_SIGNING_SECRET` below matters in every such
deployment, not only in one that opted in to something.

| Variable                                | Default | Description                                                                                     |
| --------------------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `STRATA_TRANSFORM_MODE`                 | `embedded` | `embedded` (common transforms like `duckdb_sql@v1` run in-process) or `registry` (only transforms configured in `transforms_config`, via external executors). |
| `STRATA_TRANSFORMS_CONFIG`              | `{}`    | The whole transforms block as a JSON object (`enabled`, `registry`, …). Normally written as `[tool.strata.transforms]` instead; `STRATA_TRANSFORMS_ENABLED` merges into it rather than replacing it. |
| `STRATA_SIGNED_URL_EXPIRY_SECONDS`      | `600`   | Validity window for pull-model signed build URLs.                                               |
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
| `STRATA_NOTEBOOK_PYTHON_VERSIONS`   | current server Python minor | Available Python versions (JSON array or comma-separated list) |
| `STRATA_PERSONAL_MODE_USER_HEADER`  | `None`                      | Request header carrying caller identity. When set in personal mode, notebooks are stamped with the caller's identity on create and `discover`/`delete` scope to it. Intended for proxy-fronted personal deployments. |
| `STRATA_NOTEBOOK_REMOTE_STORE_URL`  | `None`                      | Point the ambient `strata` client injected into cells at a remote shared store instead of this local notebook server, so a team publishes/consumes against one central deployment. Unset → the ambient client targets the local server. See [Service Mode → shared research store](../deployment/service-mode.md#authenticated-write-back-the-shared-research-store). |
| `STRATA_NOTEBOOK_REMOTE_STORE_HEADERS` | `{}`                     | Auth headers the ambient client attaches when pointed at a remote store (e.g. the trusted-proxy identity/token). JSON object; set via env so secrets stay out of committed config. |
| `STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES` | `2147483648` (2 GiB) | Per-file cap when packing a notebook bundle for a remote worker. Values that don't parse, or are `<= 0`, fall back to the default. |

## TUI

Defaults for the `strata tui` client; each is also a command-line flag, and the
flag wins.

| Variable                       | Default                 | Description                                                       |
| ------------------------------ | ----------------------- | ----------------------------------------------------------------- |
| `STRATA_TUI_SERVER`            | `http://localhost:8765` | Server the TUI connects to                                        |
| `STRATA_TUI_USER`              | `None`                  | Caller identity to send, for a personal deployment behind a proxy |
| `STRATA_TUI_USER_HEADER_NAME`  | `None`                  | Header carrying that identity (matches `STRATA_PERSONAL_MODE_USER_HEADER` on the server) |

## Worker

These are read by `strata-worker`, not the main server. They have no effect on a `strata-notebook` process.

| Variable                          | Default              | Description                                                                                  |
| --------------------------------- | -------------------- | -------------------------------------------------------------------------------------------- |
| `STRATA_WORKER_TOKEN`             | `None`               | Optional bearer token. When set, the worker's `/v1/*` execution endpoints require `Authorization: Bearer <token>`. `/health` stays open. See [Workers § Authentication](../notebook/workers.md#authentication). |
| `STRATA_WORKER_MAX_INPUT_BYTES`   | `2147483648` (2 GiB) | Per-input download cap for the pull-model (`/v1/execute-manifest`). Reject inputs larger than this with 413. |
| `STRATA_WORKER_ALLOW_LOCAL_HOSTS` | `false`              | Bypass the SSRF defense that rejects manifest URLs resolving to private / loopback IPs. Only set for tests or local dev with 127.0.0.1 build servers; production deployments leave it unset. |

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
| `STRATA_LOG_LEVEL`            | `INFO`   | Log level               |
| `STRATA_LOG_FORMAT`           | `json`   | `json` or `text`        |
| `STRATA_TRACING_ENABLED`      | `false`  | Enable OpenTelemetry    |
| `STRATA_METRICS_ENABLED`      | `true`   | Set `false` to stop collecting request/cache metrics |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `None`   | OTLP collector endpoint |
| `OTEL_SERVICE_NAME`           | `strata` | Service name for traces |

## AI Assistant

| Variable                       | Default  | Description                                                  |
| ------------------------------ | -------- | ------------------------------------------------------------ |
| `STRATA_AI_BASE_URL`           | `None`   | OpenAI-compatible API base URL                               |
| `STRATA_AI_MODEL`              | `None`   | Model identifier (e.g. `claude-sonnet-4-6`, `gpt-5.4`)       |
| `STRATA_AI_API_KEY`            | `None`   | API key (generic, works with any provider)                   |
| `STRATA_AI_MAX_CONTEXT_TOKENS` | `100000` | Max context tokens sent to the model                         |
| `STRATA_AI_MAX_OUTPUT_TOKENS`  | `4096`   | Max output tokens requested                                  |
| `STRATA_AI_TIMEOUT_SECONDS`    | `60.0`   | AI request timeout                                           |
| `STRATA_AI_APPROVAL_TIMEOUT_SECONDS` | `120.0` | Agent confirm-prompt timeout; expiry counts as a decline |
| `ANTHROPIC_API_KEY`            | `None`   | Anthropic API key (auto-sets base URL + model)               |
| `OPENAI_API_KEY`               | `None`   | OpenAI API key (auto-sets base URL + model)                  |
| `GEMINI_API_KEY`               | `None`   | Google Gemini API key (auto-sets base URL + model)           |
| `MISTRAL_API_KEY`              | `None`   | Mistral API key (auto-sets base URL + model)                 |

Provider-specific keys auto-configure `base_url` and `model` defaults.
`STRATA_AI_*` variables override provider defaults. Notebook-level `[ai]`
config in `notebook.toml` overrides both.

```toml
[ai]
api_key = ""              # prefer the Runtime panel; writing here commits the key
base_url = "http://localhost:11434/v1"
model = "llama3"
max_context_tokens = 100000
max_output_tokens = 4096
timeout_seconds = 60.0
approval_timeout_seconds = 120.0
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
