# Annotations

<div class="annotation-index" markdown>

| Annotation | What it does |
| --- | --- |
| [`@name`](https://bearingresearch.com/strata/dev/reference/annotations/python/#name) | A display name for the cell. |
| [`@worker`](https://bearingresearch.com/strata/dev/reference/annotations/python/#worker) | Run the cell on a named worker. |
| [`@timeout`](https://bearingresearch.com/strata/dev/reference/annotations/python/#timeout) | A time limit for this cell, in seconds. |
| [`@env`](https://bearingresearch.com/strata/dev/reference/annotations/python/#env) | An environment variable for this cell only. |
| [`@mount`](https://bearingresearch.com/strata/dev/reference/annotations/python/#mount) | Attach a local or remote folder as a path. |
| [`@fetch`](https://bearingresearch.com/strata/dev/reference/annotations/python/#fetch) | Bytes from a URL, as a recorded input. |
| [`@dataset`](https://bearingresearch.com/strata/dev/reference/annotations/python/#dataset) | A registry name, as a recorded input. |
| [`@table`](https://bearingresearch.com/strata/dev/reference/annotations/python/#table) | An Iceberg table, as a versioned input. |
| [`@write_table`](https://bearingresearch.com/strata/dev/reference/annotations/python/#write_table) | Write a variable into an Iceberg table. |
| [`@output_dir`](https://bearingresearch.com/strata/dev/reference/annotations/python/#output_dir) | A folder whose files are an output. |
| [`@each`](https://bearingresearch.com/strata/dev/reference/annotations/python/#each) | Run once per file or row, each cached on its own. |
| [`@nocache`](https://bearingresearch.com/strata/dev/reference/annotations/python/#nocache) | Always run; never serve a cached result. |
| [`@stochastic`](https://bearingresearch.com/strata/dev/reference/annotations/python/#stochastic) | Every run is a sample: it always runs and keeps its outputs. |
| [`@after`](https://bearingresearch.com/strata/dev/reference/annotations/python/#after) | An ordering-only edge from another cell. |
| [`@sql`](https://bearingresearch.com/strata/dev/reference/annotations/sql/#sql) | Mark a SQL cell and name its connection. |
| [`@cache`](https://bearingresearch.com/strata/dev/reference/annotations/sql/#cache) | Choose a SQL cell's cache policy. |
| [`@model`](https://bearingresearch.com/strata/dev/reference/annotations/prompt/#model) | The model a prompt cell calls. |
| [`@temperature`](https://bearingresearch.com/strata/dev/reference/annotations/prompt/#temperature) | Sampling temperature. |
| [`@max_tokens`](https://bearingresearch.com/strata/dev/reference/annotations/prompt/#max_tokens) | The most tokens the response may use. |
| [`@system`](https://bearingresearch.com/strata/dev/reference/annotations/prompt/#system) | The system prompt. |
| [`@output`](https://bearingresearch.com/strata/dev/reference/annotations/prompt/#output) | Parse the response as JSON. |
| [`@output_schema`](https://bearingresearch.com/strata/dev/reference/annotations/prompt/#output_schema) | Validate the response against a JSON Schema. |
| [`@validate_retries`](https://bearingresearch.com/strata/dev/reference/annotations/prompt/#validate_retries) | How often to retry a response that fails validation. |
| [`@loop`](https://bearingresearch.com/strata/dev/reference/annotations/loops-and-variants/#loop) | Iterate the cell, carrying a variable between steps. |
| [`@loop_until`](https://bearingresearch.com/strata/dev/reference/annotations/loops-and-variants/#loop_until) | Stop a loop when an expression is true. |
| [`@variant`](https://bearingresearch.com/strata/dev/reference/annotations/loops-and-variants/#variant) | Mark the cell as one alternative in a group. |
| [`@per_variant`](https://bearingresearch.com/strata/dev/reference/annotations/loops-and-variants/#per_variant) | Run the cell once per variant of its input. |
| [`@live`](https://bearingresearch.com/strata/dev/reference/annotations/widget/#live) | Let a control's change run the cheap cells below it. |
| [`@app hide`](https://bearingresearch.com/strata/dev/reference/annotations/widget/#app-hide) | Keep the cell out of the app view. |

</div>

R cells take a subset of these: see [R annotations](https://bearingresearch.com/strata/dev/reference/annotations/r/). SQL cells take six of them,
written as SQL comments (`-- @key value`): see [What a SQL cell reads](https://bearingresearch.com/strata/dev/reference/annotations/sql/#what-a-sql-cell-reads).

## How annotations are read

Annotations are metadata directives written in the leading comment block of a cell. They control display names, execution routing, timeouts, environment variables, filesystem mounts, and cache behavior, without separate configuration UI. The cell source is the single source of truth.

```python
# @name Train Classifier
# @worker my-gpu
# @timeout 120
# @env CUDA_VISIBLE_DEVICES=0
x = embeddings
y = labels
classifier.fit(x, y)
```

Annotations are parsed from the **first contiguous block** of `#`-prefixed lines (`--` in a [SQL cell](https://bearingresearch.com/strata/dev/reference/annotations/sql/#writing-them)). Once a non-comment, non-blank line is encountered, parsing stops. Each annotation is one line in the format `# @key value`. A key that is not an
annotation, such as a misspelled `# @wroker gpu`, does nothing; validation
warns about it (`annotation_unknown`, in the cell header and `strata validate`)
and suggests the annotation it is closest to.

A known annotation whose value cannot be read, such as `# @timeout soon`,
`# @mount data` with no URI or `# @nocache maybe`, stops the cell: it does not
run, on any path (the editor, Run All, `strata run`, the API), and fails with an
error that names the line and why, such as `timeout_not_numeric` or
`mount_unreadable`. The cells that depend on it are skipped too. Running it
anyway would compute, and cache, something other than what the source declares.
`strata validate` reports the same errors before you run.

In the editor you type them directly into the cell. From a script or agent you can either write them into the source you pass to `strata cell add/edit`, or splice a single directive without touching the body: `strata cell annotate <notebook_dir> <cell_id> --set worker=gpu-box --unset timeout` (see the [CLI](https://bearingresearch.com/strata/dev/notebook/cli/#authoring-cells-and-dependencies-cell-addeditrmmv-dep)).

## Precedence rules

When the same setting is configured at multiple levels, the most specific wins:

| Setting | Annotation | Cell config (notebook.toml) | Notebook default |
|---------|-----------|---------------------------|-----------------|
| **Worker** | `# @worker X` | `cell.worker` field | `notebook.worker` field |
| **Timeout** | `# @timeout N` | `cell.timeout` field | `notebook.timeout` field, else 300 seconds |
| **Env vars** | `# @env K=V` | `cell.env` overrides | `notebook.env` defaults |
| **Mounts** | `# @mount ...` | `cell.mounts` overrides | `notebook.mounts` defaults |
| **SQL connection** | `-- @sql connection=X` | (annotation only) | none; required for SQL cells |
| **Cache policy** | `-- @cache <policy>` | (annotation only) | `fingerprint` (read), `session` (write) |
| **Cache opt-out** | `# @nocache` | (annotation only) | cache on (serve provenance hits) |

Annotations always take priority. This lets you override per-cell behavior without editing `notebook.toml`.
