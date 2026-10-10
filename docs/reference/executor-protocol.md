# Executor Protocol

The executor protocol is the HTTP contract between Strata's notebook orchestrator and remote workers. Anyone can implement it to bring custom compute (GPUs, alternative engines, sandboxed environments). The reference implementation is `strata-worker` from `strata.notebook.remote_executor`.

This page is the canonical specification. The [Distributed Workers](../notebook/workers.md) page covers deployment and registration; this one covers wire format.

## Versioning

| Constant | Value | Source |
| --- | --- | --- |
| Executor protocol version | `v1` | `EXECUTOR_PROTOCOL_VERSION` |
| Notebook-cell protocol | `notebook-cell-v1` | `NOTEBOOK_EXECUTOR_PROTOCOL_VERSION` |
| Notebook-cell transform ref | `notebook_cell@v1` | `NOTEBOOK_EXECUTOR_TRANSFORM_REF` |
| Manifest format | `notebook-build-manifest@v1` | `NOTEBOOK_EXECUTOR_MANIFEST_VERSION` |
| Output bundle | `notebook-output-bundle@v1` | `schema_version` in the bundle's `manifest.json` |

Workers reject mismatched protocol versions with `400 Bad Request`.

## Authentication

Optional. Set `STRATA_WORKER_TOKEN=<opaque>` on the worker process; clients must then send `Authorization: Bearer <opaque>` on `/v1/*` endpoints. `/health` is always open so platform probes work without the secret.

Unauthenticated requests against a token-gated worker return:

```http
HTTP/1.1 401 Unauthorized
Content-Type: application/json

{"detail": "Missing or malformed Authorization header (expected Bearer token)"}
```

A wrong token returns `401 Unauthorized` with `{"detail": "Invalid worker token"}`. Comparison is constant-time.

## `GET /health`

Liveness + capabilities probe. No auth.

**Response (200)**:

```json
{
  "status": "healthy",
  "capabilities": {
    "protocol_versions": ["v1"],
    "transform_refs": ["notebook_cell@v1"],
    "features": {
      "notebook_protocol_version": "notebook-cell-v1",
      "output_format": "notebook-output-bundle@v1",
      "pull_model": true,
      "cancel": true,
      "locked_environments": true,
      "locked_r_environments": true,
      "languages": ["python", "r"]
    }
  },
  "version": "1.0.0",
  "uptime_seconds": 42.5,
  "launch_id": null,
  "active_executions": 0,
  "max_concurrent": 2,
  "gpu_slots": 2,
  "free_gpu_slots": 2,
  "hardware": {
    "cpus": 32,
    "memory_mb": 257000,
    "accelerators": [{"name": "NVIDIA A100-SXM4-80GB", "memory_mb": 81920, "driver": "535.104.05"}],
    "cuda": "12.2"
  }
}
```

`active_executions` is the count of in-flight `/v1/*` calls - useful for autoscaler signals. `max_concurrent` and `gpu_slots` are the worker's limits (`null` when unset), and `free_gpu_slots` how many GPUs are unassigned (`null` without `gpu_slots`), so a caller can plan rather than discover the limit by being refused. `launch_id` is `STRATA_WORKER_LAUNCH_ID`, set only on a worker Strata launched over SSH, which checks it to tell its worker from another process on the port; otherwise `null`. `hardware` is what the machine reports about itself: `cpus` (those this process may use) and `memory_mb` from the OS, and `accelerators` and `cuda` from `nvidia-smi` when it is on the worker's `PATH`. It lets a caller check a provider's machine against the class it was sold as without submitting a job. A field that could not be read is omitted, so a missing `accelerators` means unknown, not "no GPU". The notebook UI polls this and shows the worker badge red if `/health` fails or times out.

`locked_environments: true` says the worker runs a cell in the notebook's own locked environment when the request carries one (below). Strata sends that block only to a worker that advertises it; any other gets requests exactly as before. Answer it honestly: building that environment is a `uv sync --frozen`, so the reference worker reports it by probing for `uv` on its own `PATH` rather than claiming it unconditionally. A worker that claims it without `uv` is sent work it will refuse, and every notebook has a lockfile.

A cell runs with the worker's environment minus the worker's own secrets: `strata-worker` takes its token and credentials out of the process environment at startup and holds them in memory, so a cell cannot read them from its own environment or through `/proc`. `STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST` narrows the rest, as [the server's allowlist](../deployment/service-mode.md#what-a-cell-can-read) narrows a cell there. A cell gets what its manifest carries.

`languages` lists the cell languages the worker can run: `r` when `Rscript` is on its `PATH`. An R cell's request says `"language": "r"`, in `transform.params.language` on `POST /v1/execute`, `language` in `POST /v1/notebook-execute` metadata, and `params.language` in a manifest; a Python cell's request carries no `language`. The worker runs `harness.R` under `Rscript` with the same manifest a Python cell's harness gets, and answers an R cell with `500` and `Rscript is not installed on this worker` when it has no R, or `400` for a language it does not know.

`locked_r_environments: true` says the worker restores an R cell's lock, the notebook's `renv.lock`, when the request carries one ([below](#the-r-environment-block)). It works like `locked_environments`: Strata sends the R block only to a worker that advertises it, and the reference worker reports it only when its `Rscript` can load `renv`.

### The `environment` block

A request to a worker that advertises `locked_environments` carries the notebook's lock, in `transform.params.environment` on `POST /v1/execute`, `environment` in `POST /v1/notebook-execute` metadata, and `params.environment` in a manifest:

```json
{
  "key": "<uv_lock_key of uv.lock>",
  "python": "3.13",
  "lockfile": "<the notebook's uv.lock>",
  "pyproject": "<the notebook's pyproject.toml>"
}
```

`key` names what the lock installs, not which notebook it belongs to, so notebooks with the same resolved dependencies share one key. It is `strata.notebook.env.uv_lock_key(lockfile)`: the lock parsed as TOML; in the notebook's own `package` entry (the one whose `source` is `{ virtual = "." }` or `{ editable = "." }`), `name`, `version`, `source` and `metadata` dropped; any other `name` field equal to that project's name, and that name in `manifest.members`, replaced by `"<project>"`; `package` sorted by each entry's JSON with sorted keys and `manifest.members` sorted; and the SHA-256 hex digest of the result as compact JSON with sorted keys (Python's `json.dumps(data, sort_keys=True, separators=(",", ":"))`). An environment registry addressed by `key` should compute it the same way, ideally by calling that function. The worker installs with `--no-install-project`, so the environment holds only the lock's packages.

The worker runs the cell's harness with the interpreter of that environment:

- It keeps one environment per `key` and interpreter build under `STRATA_WORKER_ENV_ROOT` (default `~/.strata/worker-envs`). An environment already there is reused, so a second cell with the same lock installs nothing.
- A missing one is fetched from `STRATA_WORKER_ENV_REGISTRY_URL/<key>/<interpreter>/<platform>` as a `.tar.gz` of the environment directory when that is set. A registry that answers `404` has none, and the worker builds the environment itself; any other failure fails the cell. Without a registry the worker builds it, with `uv sync --frozen` from the lock. The worker needs `uv` on its `PATH` for that.
- A lock whose key is not `key`, or that cannot be installed, fails the cell with the reason (`500`).

The two last segments of a registry path name the interpreter the worker resolved for `python` (`uv python find --system <python>`), since an environment's contents depend on the build it was made with:

- `<interpreter>` is `f"{sys.implementation.name}-{platform.python_version()}{sys.abiflags}"`, for example `cpython-3.13.1`, or `cpython-3.13.1t` for a free-threaded build.
- `<platform>` is `sysconfig.get_platform()`, for example `linux-x86_64`, `linux-aarch64` or `macosx-14.0-arm64`.

So one registry serves every Python build and platform, and a registry that prebuilds for a worker image runs those two expressions with the image's interpreter.

### The R `environment` block

An R cell sent to a worker that advertises `locked_r_environments` carries the notebook's `renv.lock` in the same places, when the notebook has one:

```json
{
  "key": "<renv_lock_key of renv.lock>",
  "lockfile": "<the notebook's renv.lock>"
}
```

`key` is `strata.notebook.env.renv_lock_key(lockfile)`: the SHA-256 hex digest of the lock's UTF-8 bytes, the same digest the [shared environment backend](../notebook/environment.md#shared-environments) keys an R library by. An `renv.lock` names no project, so nothing is left out. The worker then:

- Keeps one R library per `key` and R build under `STRATA_WORKER_ENV_ROOT/r`, and reuses it, so a second R cell with the same lock restores nothing.
- Fetches a missing one from `STRATA_WORKER_ENV_REGISTRY_URL/r/<key>/<R version>/<platform>` when that is set: a `.tar.gz` whose top-level directories are the packages (`jsonlite/DESCRIPTION`, ...). A `404` is restored locally; any other failure fails the cell. Without a registry, or on a `404`, it runs `renv::restore(lockfile = ..., library = <that directory>)`, with renv's package cache in `STRATA_WORKER_ENV_ROOT/r/cache`.
- Keeps a library only when every package in the lock's `Packages`, `renv` itself aside, has a `DESCRIPTION` in it; otherwise the cell fails and the next one fetches or restores again.
- Runs `harness.R` with that library on `R_LIBS`, so it comes first and the image's libraries serve only what the lock does not list, such as the `jsonlite` and `arrow` the harness loads.
- Fails the cell with the reason (`500`) for a lock whose key is not `key`, that is not JSON, or that cannot be restored.

The two last segments of a registry path name the R the worker runs:

- `<R version>` is `paste0("R-", R.version$major, ".", R.version$minor)`, for example `R-4.4.1`.
- `<platform>` is `R.version$platform`, for example `x86_64-pc-linux-gnu` or `aarch64-apple-darwin20`.

An archive must hold the packages themselves, not renv's links into a package cache, which unpacking refuses. R packages built from source link the system libraries of the machine that built them, which the platform does not name, so a registry should build an image's archives on that image.

**`503 Service Unavailable`** from any execution route means the worker is full: `max_concurrent` executions are in flight, or every GPU slot is taken. It carries `Retry-After` in seconds and is refused before any input is downloaded, so retrying costs the worker nothing.

## `POST /v1/execute` (push model - recommended)

The standard executor v1 envelope. Cells and inputs are pushed inline; the worker returns the output bundle directly in the response.

**Content-Type**: `multipart/form-data`.

**Form fields:**

| Field | Type | Description |
| --- | --- | --- |
| `metadata` | JSON file part | The execution envelope (schema below) |
| `<input_name>` | file part | One field per input variable, content per input descriptor's `format` |

**`metadata` JSON:**

```json
{
  "protocol_version": "v1",
  "build_id": "notebook-3f2a9c1b7e04",
  "transform": {
    "ref": "notebook_cell@v1",
    "params": {
      "source": "result = df.sum()",
      "timeout_seconds": 300,
      "mounts": [
        {
          "name": "data",
          "uri": "s3://bucket/prefix",
          "mode": "ro",
          "options": {"anon": true}
        }
      ],
      "env": {
        "MODEL_PATH": "/models/bge-large"
      }
    }
  },
  "inputs": [
    {"name": "df", "format": "arrow/ipc"},
    {"name": "weights", "format": "pickle/object"}
  ]
}
```

| Field | Type | Description |
| --- | --- | --- |
| `protocol_version` | string (required) | Must be `"v1"`. |
| `transform.ref` | string (required) | Must be `"notebook_cell@v1"` for the notebook executor. |
| `transform.params.source` | string (required) | The cell Python source to execute. |
| `transform.params.timeout_seconds` | float | Execution timeout (default 30). |
| `transform.params.mounts` | array of MountSpec | Filesystem mounts injected as `Path` variables (see [notebook.toml schema](notebook-toml.md#mounts-filesystem-mounts)). |
| `transform.params.env` | object | Env vars set in the cell subprocess. |
| `transform.params.mutation_defines` | array of string | Present only when non-empty. Variables the cell changes in place without rebinding. The harness re-serializes exactly these: without the list it sees an unchanged `id()` and stores nothing, and the downstream cell reads the value from before, under a provenance hash that says otherwise. |
| `transform.params.tables` | object | Present only when non-empty. `{name: {uri, snapshot_id}}` for each `@table` the cell declares. Injected as `<name>` and `<name>_snapshot`; no catalog access is needed at the worker to read them. |
| `inputs` | array of `{name, format}` | Each entry references a multipart field with the same `name`. `format` is the content type - `arrow/ipc`, `pickle/object`, `json/object`, `module/import`, `module/cell`, `module/cell-instance`, or `file/path` for an `@fetch`'s bytes, which the harness injects as a `pathlib.Path` to the written file instead of loading. |

**Response (200)**:

```http
HTTP/1.1 200 OK
Content-Type: application/x-tar
X-Strata-Executor-Protocol: v1
X-Strata-Notebook-Executor-Protocol: notebook-cell-v1

<tar bundle - see "Output bundle" below>
```

A cell that raises still answers `200`: the bundle's manifest says `"success": false` and carries the error. The server refuses a response whose protocol headers name a version other than these.

**Asynchronous execution (202).** A worker, or a dispatcher in front of one
that first has to boot a machine, may answer `202 Accepted` with
`{"job_url": "/v1/jobs/..."}`, as the [pull model](#post-v1execute-manifest-pull-model)
does. The server polls `GET {job_url}` with the same headers it sent the
request with (`Authorization` and `X-Strata-Executor-Protocol`), under the
same rules: `job_url` must resolve to the host and port the request went to,
and the job answers `{"state": "queued" | "provisioning" | "starting" | "running"}`
while it waits or runs. When the job finishes, the job URL answers exactly
what a synchronous worker would have: `200` with `Content-Type: application/x-tar`,
the two protocol headers and the output bundle, which the server reads and
checks as it reads the `200` above. A job that fails answers
`{"state": "failed", "status_code": 502, "error": "..."}`, and the cell fails
with that error (`status_code` `408` fails it as a timeout). A `finished`
state in JSON is refused, since a direct job's result is its bundle.

The deadlines are the pull model's: waiting before `running` is bounded by
`STRATA_WORKER_PROVISIONING_TIMEOUT_SECONDS` (default 600), and the cell's
`timeout_seconds` starts only when the job first reports `running`, so a
machine's boot does not spend the cell's budget. If either deadline passes,
or the cell is cancelled while the server polls, the server calls
`/v1/executions/{build_id}/cancel` with the `build_id` from `metadata`.
A worker that answers `200` directly needs no change.

**Errors:**

| Status | When |
| --- | --- |
| `400` | Missing/invalid `metadata`, unsupported `protocol_version`, unsupported `transform.ref`, malformed input descriptor, unknown cell `language` |
| `401` | Token gate failed |
| `408` | Cell execution exceeded `timeout_seconds` |
| `409` | The execution was cancelled (`/v1/executions/{build_id}/cancel`) before its harness started |
| `413` | Pull model only: an input exceeds `STRATA_WORKER_MAX_INPUT_BYTES` (default 2 GiB) |
| `500` | The harness could not run: the locked environment failed to build, `Rscript` is missing for an R cell, or the subprocess crashed |
| `502` | Pull model only: downloading an input, uploading the bundle, or finalizing failed |
| `503` | The worker is full (see above) |

## `POST /v1/notebook-execute` (notebook-specific envelope)

A legacy/alternative entry point that takes a flatter envelope (no `transform.ref` wrapper). Functionally equivalent to `/v1/execute` for the notebook-cell case. Same multipart shape; the `metadata` JSON differs:

```json
{
  "protocol_version": "notebook-cell-v1",
  "source": "result = df.sum()",
  "timeout_seconds": 300,
  "inputs": {
    "df": {"content_type": "arrow/ipc", "file": "df.arrow"},
    "weights": {"content_type": "pickle/object", "file": "weights.pickle"}
  },
  "mounts": [],
  "env": {}
}
```

New workers should prefer `/v1/execute` for forward compatibility. `/v1/notebook-execute` exists for backwards compatibility with notebook deployments that pre-date the unified v1 envelope; the notebook client will route to whichever endpoint the `workers.config.url` points at.

## `POST /v1/execute-manifest` (pull model)

For workloads where streaming inputs through Strata is a bandwidth bottleneck (large artifacts, geo-distant workers), the orchestrator can hand the worker **signed URLs** and let it fetch inputs and upload the result directly to blob storage.

**Content-Type**: `application/json`.

**Request body:**

```json
{
  "build_id": "01HZJV...",
  "metadata": {
    "build_id": "01HZJV...",
    "artifact_id": "nb_remote_9c7e22e3-8ed7-452c-885c-49574d7aa02f_01HZJV...",
    "version": 1,
    "executor_ref": "notebook_cell@v1",
    "params": {
      "source": "result = big_df.summarize()",
      "timeout_seconds": 600,
      "input_specs": {
        "big_df": {"uri": "strata://artifact/abc123@v=4"}
      },
      "mounts": [],
      "env": {}
    },
    "principal": "alice",
    "tenant": "team-a",
    "notebook_id": "9c7e22e3-8ed7-452c-885c-49574d7aa02f",
    "cell_id": "ba3b7451",
    "cell_provenance_hash": "5f2c…",
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
  },
  "inputs": [
    {
      "artifact_id": "abc123",
      "version": 4,
      "url": "https://strata.example.com/v1/artifacts/download?...&signature=...",
      "expires_at": 1789455608.2
    }
  ],
  "output": {
    "url": "https://strata.example.com/v1/artifacts/upload?...&signature=...",
    "max_bytes": 1073741824,
    "expires_at": 1789455608.2,
    "method": "POST"
  },
  "finalize_url": "https://strata.example.com/v1/builds/01HZJV/finalize?...",
  "log_url": "https://strata.example.com/v1/builds/01HZJV/log?..."
}
```

Each URL is a signed capability that expires at `expires_at`; `output.max_bytes`
caps the upload. Fetching a build's manifest again (`GET
/v1/builds/{build_id}/manifest`) renews its lease and retires the upload and
finalize URLs of every earlier manifest, so an executor still holding an old
one can no longer publish. If the lease lapsed and the build runner took the
build back, the fetch answers `409` instead. Each manifest's upload lands under its own key, and
bytes that are never finalized are removed by the server's build runner once
the build is over and the upload URL has expired.

With `STRATA_ARTIFACT_PRESIGNED_URLS` on and a blob store the server can sign
for, the input URLs and `output.url` point straight at the object store.
`finalize_url` and `log_url` stay Strata routes, and finalize checks the
output's size against `output.max_bytes` before publishing.

| Store | Inputs | Output | Signs with |
| --- | --- | --- | --- |
| S3 | SigV4 query URLs | POST policy: `fields`, bounded by `content-length-range` | An access key pair (config or `AWS_ACCESS_KEY_ID`), else a role through botocore (instance profile, ECS task role, web identity; the `s3` extra). The role's session token goes in the URL, and a URL stops working when that token expires. |
| GCS | V4 signed URLs | V4 POST policy: `fields`, bounded by `content-length-range` | A service-account key (`STRATA_GCS_CREDENTIALS_JSON` or `GOOGLE_APPLICATION_CREDENTIALS`), else, on GCE or GKE workload identity, the IAM `signBlob` API as the attached service account, which needs `roles/iam.serviceAccountTokenCreator` on itself. Needs the `gcs` extra. |
| Azure | Read-only blob SAS | Create/write blob SAS: `method` `PUT` with `headers` | The account key (`STRATA_AZURE_ACCOUNT_KEY` or a connection string that holds it), else a user delegation key for `STRATA_AZURE_USE_DEFAULT_CREDENTIAL`, which needs the Storage Blob Delegator role. A PUT cannot bound its body, so only finalize enforces `max_bytes`. |

Without the setting, or when the store cannot sign (a local disk, anonymous
access, a configured `STRATA_AZURE_SAS_TOKEN`, which is never handed to a
worker, or GCS user credentials with no service account), every URL is a Strata
route and `output` has neither `fields` nor `headers`.

`principal`, `tenant`, `notebook_id`, `cell_id` and `cell_provenance_hash` say
who ran the cell and which cell of which notebook the build is for, so a
dispatcher can attribute and match a job from the manifest alone rather than
calling `GET /v1/builds/{build_id}`. `principal` and `tenant` are `null` when no
authenticated caller dispatched the cell, as on a personal server.
`cell_provenance_hash` is the cell's own cache key: two submissions of the same
computation share it. They sit outside `params` deliberately, since `params` is
hashed into the build's transport provenance and who ran a cell must not change
what is cached. A worker can ignore them.

`traceparent` and `tracestate` are the W3C trace context of the server's
`notebook.dispatch` span, present when the server has tracing on. The same
values go in the request's headers. A worker opens its `worker.execute` span as
a child of the header context if there is one, and otherwise of the manifest's.
The header wins because a dispatcher in between, such as a pool, forwards its
own span there. A dispatcher that queues the manifest and sends only the body
still leaves the manifest's copy to link the trace. The direct transport
carries the context in the request headers only.

**Worker behavior:**

1. For each entry in `metadata.params.input_specs`, look up its `uri` in `inputs[]` and stream-download from the signed URL to the input file, so an input is bounded by the worker's disk rather than its memory. Inputs that exceed `STRATA_WORKER_MAX_INPUT_BYTES` (declared via `Content-Length` or measured during stream) are rejected with `413`.
2. Run the cell in a subprocess (same as `/v1/execute`). While it runs, `POST` each chunk of console output as the raw body to `log_url` with `&stream=stdout` or `&stream=stderr` and `&seq=N` appended, so the notebook shows it live. `seq` numbers each stream's chunks from 0, counting every chunk the cell printed: a worker that skips one (its queue was full, a post failed) leaves a gap, and the notebook then shows nothing after the gap until the cell finishes, so what it showed is always the start of the console. A repeated `seq` is ignored, so a chunk can be retried. The route answers `202`. `log_url` is optional: a worker that ignores it still delivers the console in the bundle.
3. Upload the resulting output bundle to `output.url`. With `output.fields`, it is a presigned form upload (S3, GCS): `POST` a multipart form containing each field plus the bundle as the `file` part (S3 and GCS answer `204`). With `output.method` `PUT` (Azure), `PUT` the bundle as the raw body with `output.headers` and `Content-Length` (Azure answers `201`). Otherwise `POST` the bundle as the raw body with `Content-Type: application/x-tar` (a Strata route).
4. `POST {"output_format": "notebook-output-bundle@v1"}` to `finalize_url`.
5. Return the `finalize` response body to the caller.

**Response (200):**

```json
{
  "success": true,
  "build_id": "01HZJV...",
  "byte_size": 1048576,
  "protocol_version": "notebook-build-manifest@v1",
  "finalize": { "...orchestrator's finalize response..." }
}
```

**Asynchronous execution (202).** A worker, or a pool or dispatcher in front of
one, may instead answer `202 Accepted` right away:

```json
{"job_url": "/v1/jobs/01HZJV"}
```

`job_url` may be relative to the manifest URL, and must resolve to the same host and port the manifest went to: the server refuses one pointing anywhere else rather than poll a host of the worker's choosing with the worker's token. The server then polls
`GET {job_url}`, with the same `Authorization` header, for:

```json
{"state": "provisioning"}
{"state": "running"}
{"state": "finished", "status_code": 200, "result": { "...the 200 body above..." }}
{"state": "failed", "status_code": 502, "error": "..."}
```

Before the cell runs, the state is `queued`, `provisioning` or `starting`.
That wait is bounded by `STRATA_WORKER_PROVISIONING_TIMEOUT_SECONDS` (default
600). The cell's own timeout starts only when the job first reports `running`,
so a machine that takes 45 s to boot doesn't spend a 60 s cell's budget. At
`finished` or `failed`, `status_code` and `result` (or `error`) stand for the
response a synchronous worker would have given, and are handled the same way.
If either deadline passes, the server cancels the job through
`/v1/executions/{build_id}/cancel` and fails the cell with
`PROVISIONING_TIMEOUT` or `TIMEOUT`. While the job is being provisioned, the
cell's badge reads "starting". A worker that answers synchronously needs no
change.

**SSRF defenses on signed URLs:** Before fetching/posting, the worker validates each URL:

- **Scheme allowlist**: only `http://` and `https://`. Blocks `file://`, `data:`, `javascript:`, etc.
- **IP blocklist**: the URL's hostname is resolved (via `getaddrinfo`); every returned address must be public. Loopback / link-local (incl. cloud metadata `169.254.169.254` / `fd00:ec2::254`) / private / multicast / reserved / unspecified addresses, and any other non-global range such as shared address space `100.64.0.0/10`, are rejected with `400`. IPv4-mapped IPv6 is unmapped before checking.
- **Connection pinning**: the fetch itself resolves the host once more, checks every address the same way, and connects only to one of those addresses, so a name cannot pass the check with a public address and then answer the connection with `127.0.0.1` (DNS rebinding). TLS still verifies the certificate against the hostname. These connections go direct: `HTTPS_PROXY` and the other proxy variables are ignored, because through a proxy the proxy would resolve the name. With `STRATA_WORKER_ALLOW_LOCAL_HOSTS=1` nothing is pinned and the proxy variables apply.

`STRATA_WORKER_ALLOWED_HOSTS` names hosts that pass the IP check anyway (comma-separated; a leading dot is a suffix), for a server on a private address. Set `STRATA_WORKER_ALLOW_LOCAL_HOSTS=1` to bypass the IP check for every host (tests and local-dev with 127.0.0.1 servers only).

## `POST /v1/executions/{build_id}/cancel`

Stops the harness running `build_id`, and every process it started, if it is still running. A cancel that arrives while the worker is still fetching inputs or building the locked environment stops the run before the harness starts; that execution then answers 409. Answers `{"build_id": "...", "cancelled": true}`, or `"cancelled": false` when nothing by that id is running, which is a normal answer rather than an error: the cell may have finished before the cancel arrived. The server calls it when a remote cell is cancelled or times out, and `strata-pool` calls it with the machine's token when a running job is cancelled through `POST /v1/jobs/{id}/cancel`.

## `POST /execute` (worker-pool alias)

The same handler as `/v1/execute-manifest`, at the path `strata-pool` dispatches to. The pool forwards the job body verbatim with no content type, and a build manifest is self-describing, so a manifest that arrives through the pool and one pushed directly are validated identically.

## Output bundle (`notebook-output-bundle@v1`)

An uncompressed tar archive containing:

```
manifest.json           - execution result + index of the files below
stdout.txt              - the cell's stdout
stderr.txt              - the cell's stderr
files/                  - one file per serialized variable
  result.arrow
  log.json
  __display__0.png      - the cell's last display, as variable "_"
```

**`manifest.json`:**

```json
{
  "schema_version": "notebook-output-bundle@v1",
  "success": true,
  "variables": {
    "result": {"content_type": "arrow/ipc", "file": "files/result.arrow", "...": "..."},
    "log": {"content_type": "json/object", "file": "files/log.json", "...": "..."},
    "_": {"content_type": "image/png", "file": "files/__display__0.png", "...": "..."}
  },
  "displays": [
    {"content_type": "image/png", "file": "files/__display__0.png", "...": "..."}
  ],
  "stdout_file": "stdout.txt",
  "stderr_file": "stderr.txt",
  "mutation_warnings": [],
  "error": null,
  "traceback": null,
  "build_env": "cpython-3.13-linux-x86_64",
  "hardware": {"cpus": 32, "memory_mb": 257000, "accelerators": ["..."], "cuda": "12.2"}
}
```

Each entry in `variables` is the serializer's metadata for that value, with
`file` pointing inside the bundle. A variable that could not be serialized has
`{"error", "type"}` and no file. `displays` lists every display the cell
emitted, in order, in the same shape (a display that failed to serialize has
`error` and no file); the last one also travels as the variable `_`. It must
be a list, possibly empty.

`hardware` is the same object `/health` returns, echoed from the worker that ran the job. The notebook records it on each stored artifact's transform spec, beside `build_env`, and does not hash it: the machine type a cell asked for is part of its identity, while the exact accelerator and driver are kept for the record. That way identical machines of one class still share a cache.

When the cell raises, the bundle is still returned with `"success": false`,
`error` and `traceback` set, and `variables` usually empty.

## Error envelope

Protocol and transport failures (`400`, `401`, `413`, `502`, `503`) use FastAPI's JSON shape:

```json
{"detail": "<human-readable error message>"}
```

A failure to run the cell at all (`408`, `409`, `500`, and `400` for an unknown language) answers:

```json
{"success": false, "error": "<human-readable error message>"}
```

Workers do not return structured error codes - the HTTP status is the machine-readable signal. Production deployments behind an authenticating proxy should not surface worker error messages to end users verbatim, since they may include path fragments or internal hostnames.

## Implementing a custom worker

The minimum surface is `POST /v1/execute` + `GET /health`. The reference Python implementation is `create_notebook_executor_app()` in `src/strata/notebook/remote_executor.py` and is the canonical specification when in doubt.

A custom worker doesn't have to run Python - it just has to accept the `notebook_cell@v1` envelope, execute the source somehow, and return the bundle. In practice almost all workers wrap a Python interpreter (since cells are Python) and the `strata-worker` script is the path of least resistance.

A worker the server cannot reach can dial out instead: `strata-worker --connect` carries these same requests over one outbound WebSocket to a relay, which presents the worker at an ordinary URL. The [Worker Relay Protocol](worker-connect.md) specifies the framing a relay implements.

## Core transform executors

Everything above is the notebook-cell contract. The Core build runner, which
executes `POST /v1/materialize` transforms other than `scan@v1`, uses a simpler
one. Each transform ref is matched against the registry in
`[tool.strata.transforms]`:

```toml
[tool.strata.transforms]
enabled = true

[[tool.strata.transforms.registry]]
ref = "pandas_script@*"
executor_url = "http://executor:8080"
timeout_seconds = 300
max_output_bytes = 1073741824
max_input_bytes = 0
requires_scope = "transforms:pandas"
```

| Key | Type | Description |
| --- | --- | --- |
| `ref` | string (required) | Glob matched against the transform ref, e.g. `duckdb_sql@v1` or `pandas_script@*`. The first matching entry wins. |
| `executor_url` | string | Base URL; the runner posts to `{executor_url}/v1/execute`. Empty or `embedded://local` runs a built-in transform in-process. |
| `timeout_seconds` | float | Per-build timeout. Default `300`; `0` uses `STRATA_BUILD_RUNNER_DEFAULT_TIMEOUT`. |
| `max_output_bytes` | integer | Output cap, enforced while the response streams in. Default `0`, which uses `STRATA_BUILD_RUNNER_DEFAULT_MAX_OUTPUT`. |
| `max_input_bytes` | integer | Cap on the total size of the inputs; the build fails before the call when they exceed it. Default `0` (no cap). |
| `requires_scope` | string | Scope a principal needs to materialize this transform, checked under principal auth. |

A personal-mode server always holds the in-process `duckdb_sql@v1`, plus the
listed entries when `[tool.strata.transforms]` has `enabled = true`. In service
mode the registry holds only the listed entries, and only when
`enabled = true`; list `ref = "duckdb_sql@v1"` with
`executor_url = "embedded://local"` to offer the built-in. A ref no entry matches is refused at
`POST /v1/materialize` (`403` in service mode, `400` in personal mode).

The in-process `duckdb_sql@v1` runs in the server process, so its DuckDB reads
only the inputs registered as `input0`, `input1`, ...: file access (`read_csv`,
`COPY ... TO`, `ATTACH`), extension loading and network reads are off, and the
query cannot turn them back on or change any other setting. A query that needs
those runs on an HTTP executor instead.

### `POST {executor_url}/v1/execute`

`multipart/form-data` with the header `X-Strata-Executor-Protocol: v1`. Strata
sends no credentials, so keep the executor where only Strata can reach it.

| Part | Content type | Content |
| --- | --- | --- |
| `metadata` | `application/json` | The envelope below |
| `input0`, `input1`, … | `application/vnd.apache.arrow.stream` | One Arrow IPC stream per input, in the order the transform lists them. Artifacts, names and table scans all arrive as Arrow |

```json
{
  "protocol_version": "v1",
  "build_id": "…",
  "tenant": "team-a",
  "principal": "alice",
  "provenance_hash": "…",
  "transform": {
    "ref": "pandas_script@v1",
    "code_hash": "<first 16 hex of sha256 of the transform spec>",
    "params": {}
  },
  "inputs": [{"name": "input0", "format": "arrow_ipc_stream"}]
}
```

`tenant` and `principal` come from the caller's principal, and are `null` without principal auth.

A success is `200` with an Arrow IPC stream as the body, which becomes the
artifact. An optional `X-Strata-Logs` header carries the executor's logs,
base64-encoded. A failure is a `4xx` or `5xx` with a JSON body:

```json
{"success": false, "error_code": "…", "error_message": "…"}
```

Any error status fails the build, and the build's error carries `error_message` (or `detail`, or the raw body), cut to 500 characters.

### Pull model

An executor can pull a core build instead: `GET /v1/builds/{build_id}/manifest`
returns the manifest shape shown under
[`POST /v1/execute-manifest`](#post-v1execute-manifest-pull-model), with
`metadata` holding only `build_id`, `artifact_id`, `version`, `executor_ref`
and `params`. Download each input from its URL, upload the result as an Arrow
IPC stream to `output.url`, then `POST {}` to `finalize_url`. `output.max_bytes`
is the server-wide `STRATA_BUILD_RUNNER_DEFAULT_MAX_OUTPUT`, not the registry
entry's cap. Fetching the manifest claims a pending build; a build the local
runner already claimed, or reclaimed after the lease lapsed, answers `409`. In service mode the manifest is issued
only under `STRATA_AUTH_MODE=trusted_proxy`.
