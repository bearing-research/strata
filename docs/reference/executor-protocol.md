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
| Output bundle | `notebook-output-bundle@v1` | (response header) |

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
      "pull_model": true
    }
  },
  "version": "1.0.0",
  "uptime_seconds": 42.5,
  "active_executions": 0
}
```

`active_executions` is the count of in-flight `/v1/*` calls - useful for autoscaler signals. The notebook UI polls this and shows the worker badge red if `/health` fails or times out.

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
| `inputs` | array of `{name, format}` | Each entry references a multipart field with the same `name`. `format` is the content type - `arrow/ipc`, `pickle/object`, `json/object`, `module/import`, `module/cell`, `module/cell-instance`. |

**Response (200)**:

```http
HTTP/1.1 200 OK
Content-Type: application/x-tar
X-Strata-Output-Format: notebook-output-bundle@v1

<gzipped tar bundle - see "Output bundle" below>
```

**Errors:**

| Status | When |
| --- | --- |
| `400` | Missing/invalid `metadata`, unsupported `protocol_version`, unsupported `transform.ref`, malformed input descriptor |
| `401` | Token gate failed |
| `413` | Input exceeds `STRATA_WORKER_MAX_INPUT_BYTES` (default 2 GiB) |
| `502` | Internal subprocess crash or timeout fetching from an upstream URL |
| `504` | Cell execution exceeded `timeout_seconds` |

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
    "executor_ref": "notebook_cell@v1",
    "params": {
      "source": "result = big_df.summarize()",
      "timeout_seconds": 600,
      "input_specs": {
        "big_df": {"uri": "strata://artifact/abc123@v=4"}
      },
      "mounts": [],
      "env": {}
    }
  },
  "inputs": [
    {
      "artifact_id": "abc123",
      "version": 4,
      "url": "https://s3.amazonaws.com/strata-artifacts/abc123-v4.arrow?X-Amz-Signature=..."
    }
  ],
  "output": {
    "url": "https://s3.amazonaws.com/strata-artifacts/build-01HZJV.tar?X-Amz-Signature=..."
  },
  "finalize_url": "https://strata.example.com/v1/builds/01HZJV/finalize"
}
```

**Worker behavior:**

1. For each entry in `metadata.params.input_specs`, look up its `uri` in `inputs[]` and stream-download from the signed URL. Inputs that exceed `STRATA_WORKER_MAX_INPUT_BYTES` (declared via `Content-Length` or measured during stream) are rejected with `413`.
2. Run the cell in a subprocess (same as `/v1/execute`).
3. Stream the resulting output bundle to `output.url` via `POST` with `Content-Type: application/x-tar`.
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

**SSRF defenses on signed URLs:** Before fetching/posting, the worker validates each URL:

- **Scheme allowlist**: only `http://` and `https://`. Blocks `file://`, `data:`, `javascript:`, etc.
- **IP blocklist**: the URL's hostname is resolved (via `getaddrinfo`); every returned address must be public. Loopback / link-local (incl. cloud metadata `169.254.169.254` / `fd00:ec2::254`) / private / multicast / reserved / unspecified addresses are rejected with `400`. IPv4-mapped IPv6 is unmapped before checking.

Set `STRATA_WORKER_ALLOW_LOCAL_HOSTS=1` to bypass the IP check (tests and local-dev with 127.0.0.1 servers only).

## Output bundle (`notebook-output-bundle@v1`)

A gzipped tar archive containing:

```
manifest.json           - execution metadata + index of files below
outputs/                - each defined variable as one file
  result.arrow          - content_type "arrow/ipc"
  log.json              - content_type "json/object"
  model.pickle          - content_type "pickle/object"
display/                - display-only outputs (figures, markdown blobs)
  cell_default.png      - content_type "image/png"
console.json            - { "stdout": "...", "stderr": "..." }
error.json              - present only on failure
```

**`manifest.json`:**

```json
{
  "protocol_version": "notebook-output-bundle@v1",
  "executor_ref": "notebook_cell@v1",
  "duration_ms": 4280,
  "outputs": [
    {"name": "result", "content_type": "arrow/ipc", "file": "outputs/result.arrow", "bytes": 32812},
    {"name": "log", "content_type": "json/object", "file": "outputs/log.json", "bytes": 412}
  ],
  "display": [
    {"file": "display/cell_default.png", "content_type": "image/png", "bytes": 18234}
  ],
  "console": "console.json",
  "error": null
}
```

On cell errors, `error.json` is populated and `outputs` may be empty:

```json
{
  "type": "RuntimeError",
  "message": "model not loaded",
  "traceback": "Traceback (most recent call last):\n  ...",
  "exit_code": 1
}
```

## Error envelope

All `4xx` and `5xx` responses use FastAPI's default JSON shape:

```json
{"detail": "<human-readable error message>"}
```

Workers do not return structured error codes - the HTTP status is the machine-readable signal. Production deployments behind an authenticating proxy should not surface worker error messages to end users verbatim, since they may include path fragments or internal hostnames.

## Implementing a custom worker

The minimum surface is `POST /v1/execute` + `GET /health`. The reference Python implementation is `create_notebook_executor_app()` in `src/strata/notebook/remote_executor.py` (~750 LOC) and is the canonical specification when in doubt.

A custom worker doesn't have to run Python - it just has to accept the `notebook_cell@v1` envelope, execute the source somehow, and return the bundle. In practice almost all workers wrap a Python interpreter (since cells are Python) and the `strata-worker` script is the path of least resistance.
