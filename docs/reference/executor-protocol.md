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
      "languages": ["python", "r"]
    }
  },
  "version": "1.0.0",
  "uptime_seconds": 42.5,
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

`active_executions` is the count of in-flight `/v1/*` calls - useful for autoscaler signals. `max_concurrent` and `gpu_slots` are the worker's limits (`null` when unset), and `free_gpu_slots` how many GPUs are unassigned, so a caller can plan rather than discover the limit by being refused. `hardware` is what the machine reports about itself: `cpus` (those this process may use) and `memory_mb` from the OS, and `accelerators` and `cuda` from `nvidia-smi` when it is on the worker's `PATH`. It lets a caller check a provider's machine against the class it was sold as without submitting a job. A field that could not be read is omitted, so a missing `accelerators` means unknown, not "no GPU". The notebook UI polls this and shows the worker badge red if `/health` fails or times out.

`locked_environments: true` says the worker runs a cell in the notebook's own locked environment when the request carries one (below). Strata sends that block only to a worker that advertises it; any other gets requests exactly as before. Answer it honestly: building that environment is a `uv sync --frozen`, so the reference worker reports it by probing for `uv` on its own `PATH` rather than claiming it unconditionally. A worker that claims it without `uv` is sent work it will refuse, and every notebook has a lockfile.

A cell runs with the worker's environment minus the worker's own secrets: `strata-worker` takes its token and credentials out of the process environment at startup and holds them in memory, so a cell cannot read them from its own environment or through `/proc`. `STRATA_NOTEBOOK_HARNESS_ENV_ALLOWLIST` narrows the rest, as [the server's allowlist](../deployment/service-mode.md#what-a-cell-can-read) narrows a cell there. A cell gets what its manifest carries.

`languages` lists the cell languages the worker can run: `r` when `Rscript` is on its `PATH`. An R cell's request says `"language": "r"`, in `transform.params.language` on `POST /v1/execute`, `language` in `POST /v1/notebook-execute` metadata, and `params.language` in a manifest; a Python cell's request carries no `language`. The worker runs `harness.R` under `Rscript` with the same manifest a Python cell's harness gets, and answers an R cell with `500` and `Rscript is not installed on this worker` when it has no R, or `400` for a language it does not know. An R cell carries no `environment` block.

### The `environment` block

A request to a worker that advertises `locked_environments` carries the notebook's lock, in `transform.params.environment` on `POST /v1/execute`, `environment` in `POST /v1/notebook-execute` metadata, and `params.environment` in a manifest:

```json
{
  "key": "<sha256 of uv.lock>",
  "python": "3.13",
  "lockfile": "<the notebook's uv.lock>",
  "pyproject": "<the notebook's pyproject.toml>"
}
```

The worker runs the cell's harness with the interpreter of that environment:

- It keeps one environment per `key` and interpreter build under `STRATA_WORKER_ENV_ROOT` (default `~/.strata/worker-envs`). An environment already there is reused, so a second cell with the same lock installs nothing.
- A missing one is fetched from `STRATA_WORKER_ENV_REGISTRY_URL/<key>` as a `.tar.gz` of the environment directory when that is set, and otherwise built with `uv sync --frozen` from the lock. The worker needs `uv` on its `PATH` for that.
- A lock that does not hash to `key`, or that cannot be installed, fails the cell with the reason (`500`).

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

**Errors:**

| Status | When |
| --- | --- |
| `400` | Missing/invalid `metadata`, unsupported `protocol_version`, unsupported `transform.ref`, malformed input descriptor, unknown cell `language` |
| `401` | Token gate failed |
| `408` | Cell execution exceeded `timeout_seconds` |
| `413` | Input exceeds `STRATA_WORKER_MAX_INPUT_BYTES` (default 2 GiB) |
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
    "expires_at": 1789455608.2
  },
  "finalize_url": "https://strata.example.com/v1/builds/01HZJV/finalize?...",
  "log_url": "https://strata.example.com/v1/builds/01HZJV/log?..."
}
```

Each URL is a signed capability that expires at `expires_at`; `output.max_bytes`
caps the upload. Fetching a build's manifest again (`GET
/v1/builds/{build_id}/manifest`) renews its lease and retires the upload and
finalize URLs of every earlier manifest, so an executor still holding an old
one can no longer publish. Each manifest's upload lands under its own key, and
bytes that are never finalized are removed by the server's build runner once
the build is over and the upload URL has expired.

With `STRATA_ARTIFACT_PRESIGNED_URLS` on and an S3 blob store the server can sign
for, the input URLs and `output.url` point straight at the object store: SigV4
query URLs for inputs, and for the output a POST policy URL with the form
`fields` to send. The policy bounds the body with `content-length-range`, so S3
refuses an oversized upload itself, and finalize checks the size again before
publishing. `finalize_url` and `log_url` stay Strata routes. Without it, or when
the store cannot sign (a local disk, or S3 credentials held only by an instance
role inside PyArrow), every URL is a Strata route and `output` has no `fields`.

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
2. Run the cell in a subprocess (same as `/v1/execute`). While it runs, `POST` each chunk of console output as the raw body to `log_url` with `&stream=stdout` or `&stream=stderr` appended, so the notebook shows it live. `log_url` is optional: a worker that ignores it still delivers the console in the bundle.
3. Upload the resulting output bundle to `output.url`. Without `output.fields`, `POST` the bundle as the raw body with `Content-Type: application/x-tar` (a Strata route). With `output.fields`, it is a presigned object-store upload: `POST` a multipart form containing each field plus the bundle as the `file` part (S3 answers `204`).
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
- **IP blocklist**: the URL's hostname is resolved (via `getaddrinfo`); every returned address must be public. Loopback / link-local (incl. cloud metadata `169.254.169.254` / `fd00:ec2::254`) / private / multicast / reserved / unspecified addresses are rejected with `400`. IPv4-mapped IPv6 is unmapped before checking.

`STRATA_WORKER_ALLOWED_HOSTS` names hosts that pass the IP check anyway (comma-separated; a leading dot is a suffix), for a server on a private address. Set `STRATA_WORKER_ALLOW_LOCAL_HOSTS=1` to bypass the IP check for every host (tests and local-dev with 127.0.0.1 servers only).

## `POST /v1/executions/{build_id}/cancel`

Stops the harness running `build_id`, and every process it started, if it is still running. Answers `{"build_id": "...", "cancelled": true}`, or `"cancelled": false` when nothing by that id is running, which is a normal answer rather than an error: the cell may have finished before the cancel arrived. The server calls it when a remote cell is cancelled or times out.

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
`{"error", "type"}` and no file.

`hardware` is the same object `/health` returns, echoed from the worker that ran the job. The notebook records it on each stored artifact's transform spec, beside `build_env`, and does not hash it: the machine type a cell asked for is part of its identity, while the exact accelerator and driver are kept for the record. That way identical machines of one class still share a cache.

When the cell raises, the bundle is still returned with `"success": false`,
`error` and `traceback` set, and `variables` usually empty.

## Error envelope

Protocol and transport failures (`400`, `401`, `413`, `502`, `503`) use FastAPI's JSON shape:

```json
{"detail": "<human-readable error message>"}
```

A failure to run the cell at all (`408`, `500`, and `400` for an unknown language) answers:

```json
{"success": false, "error": "<human-readable error message>"}
```

Workers do not return structured error codes - the HTTP status is the machine-readable signal. Production deployments behind an authenticating proxy should not surface worker error messages to end users verbatim, since they may include path fragments or internal hostnames.

## Implementing a custom worker

The minimum surface is `POST /v1/execute` + `GET /health`. The reference Python implementation is `create_notebook_executor_app()` in `src/strata/notebook/remote_executor.py` and is the canonical specification when in doubt.

A custom worker doesn't have to run Python - it just has to accept the `notebook_cell@v1` envelope, execute the source somehow, and return the bundle. In practice almost all workers wrap a Python interpreter (since cells are Python) and the `strata-worker` script is the path of least resistance.
