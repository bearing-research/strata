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

## Metadata

| Variable             | Default | Description                          |
| -------------------- | ------- | ------------------------------------ |
| `STRATA_METADATA_DB` | `None`  | SQLite path for metadata persistence |

## S3 Storage

| Variable                 | Default | Description                                      |
| ------------------------ | ------- | ------------------------------------------------ |
| `STRATA_S3_REGION`       | `None`  | AWS region                                       |
| `STRATA_S3_ENDPOINT_URL` | `None`  | Custom endpoint (MinIO, LocalStack)              |
| `STRATA_S3_ACCESS_KEY`   | `None`  | Access key (falls back to AWS_ACCESS_KEY_ID)     |
| `STRATA_S3_SECRET_KEY`   | `None`  | Secret key (falls back to AWS_SECRET_ACCESS_KEY) |
| `STRATA_S3_ANONYMOUS`    | `false` | Use anonymous access                             |

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

## Authentication

| Variable                             | Default              | Description                          |
| ------------------------------------ | -------------------- | ------------------------------------ |
| `STRATA_AUTH_MODE`                   | `none`               | `none` or `trusted_proxy`            |
| `STRATA_PROXY_TOKEN`                 | `None`               | Shared secret for proxy verification |
| `STRATA_PRINCIPAL_HEADER`            | `X-Strata-Principal` | Header for user identity             |
| `STRATA_SCOPES_HEADER`               | `X-Strata-Scopes`    | Header for permission scopes         |
| `STRATA_HIDE_FORBIDDEN_AS_NOT_FOUND` | `true`               | Return 404 instead of 403            |
| `STRATA_SERVICE_WRITES_ENABLED`      | `false`              | **Preview.** Opt-in: let authenticated clients write/publish in service mode (`put`, `set_name`, `set_alias`, tags), scoped to the caller's tenant and gated by the `artifacts:write` scope. Requires `trusted_proxy` auth (enforced at startup). Default keeps service mode read-only. See [Service Mode → shared research store](../deployment/service-mode.md#authenticated-write-back-the-shared-research-store). |

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

| Variable                                | Default | Description                                                                                     |
| --------------------------------------- | ------- | ----------------------------------------------------------------------------------------------- |
| `STRATA_TRANSFORM_MODE`                 | `embedded` | `embedded` (common transforms like `duckdb_sql@v1` run in-process) or `registry` (only transforms configured in `transforms_config`, via external executors). |
| `STRATA_PULL_MODEL_ENABLED`             | `false` | Enable the v2-pull executor protocol (executors fetch inputs / upload outputs via signed URLs). |
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

## Worker

These are read by `strata-worker`, not the main server. They have no effect on a `strata-notebook` process.

| Variable                          | Default              | Description                                                                                  |
| --------------------------------- | -------------------- | -------------------------------------------------------------------------------------------- |
| `STRATA_WORKER_TOKEN`             | `None`               | Optional bearer token. When set, the worker's `/v1/*` execution endpoints require `Authorization: Bearer <token>`. `/health` stays open. See [Workers § Authentication](../notebook/workers.md#authentication). |
| `STRATA_WORKER_MAX_INPUT_BYTES`   | `268435456` (256 MB) | Per-input download cap for the pull-model (`/v1/execute-manifest`). Reject inputs larger than this with 413. |
| `STRATA_WORKER_ALLOW_LOCAL_HOSTS` | `false`              | Bypass the SSRF defense that rejects manifest URLs resolving to private / loopback IPs. Only set for tests or local dev with 127.0.0.1 build servers; production deployments leave it unset. |

## Rate Limiting

| Variable                       | Default  | Description                    |
| ------------------------------ | -------- | ------------------------------ |
| `STRATA_RATE_LIMIT_ENABLED`    | `true`   | Enable rate limiting           |
| `STRATA_RATE_LIMIT_GLOBAL_RPS` | `1000.0` | Global requests per second     |
| `STRATA_RATE_LIMIT_CLIENT_RPS` | `100.0`  | Per-client requests per second |
| `STRATA_RATE_LIMIT_SCAN_RPS`   | `50.0`   | Scan endpoint rate limit       |

## Observability

| Variable                      | Default  | Description             |
| ----------------------------- | -------- | ----------------------- |
| `STRATA_LOG_LEVEL`            | `INFO`   | Log level               |
| `STRATA_LOG_FORMAT`           | `json`   | `json` or `text`        |
| `STRATA_TRACING_ENABLED`      | `false`  | Enable OpenTelemetry    |
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
