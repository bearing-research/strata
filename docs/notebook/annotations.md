# Cell Annotations & Environment

Annotations are metadata directives written in the leading comment block of a cell. They control display names, execution routing, timeouts, environment variables, and filesystem mounts, without separate configuration UI. The cell source is the single source of truth.

```python
# @name Train Classifier
# @worker my-gpu
# @timeout 120
# @env CUDA_VISIBLE_DEVICES=0
x = embeddings
y = labels
classifier.fit(x, y)
```

Annotations are parsed from the **first contiguous block** of `#`-prefixed lines. Once a non-comment, non-blank line is encountered, parsing stops. Each annotation is one line in the format `# @key value`.

---

## @name

Set a human-readable display name for the cell. Shown in the DAG view and as a badge in the cell editor.

```python
# @name Load arXiv Papers
import pandas as pd
papers = pd.read_parquet("https://...")
```

For **Python cells**, any non-empty string is accepted, spaces, parentheses, and special characters are fine.

For **prompt cells**, `@name` also sets the output variable name and must be a valid Python identifier:

```
# @name research_themes
Given these paper counts: {{ category_stats }}
Identify 3 research themes.
```

If no `@name` is set, the DAG view falls back to showing the cell's defined variable names, then the cell ID.

!!! warning "`@name` is a display label, not a referenceable ID"
    `@name` sets what shows in the DAG view. It is **not** what
    `@after` or `@loop start_from=` references. Those resolve against
    the cell's `id` in `notebook.toml`, which is a separate field —
    see [Cell IDs](#cell-ids) below.

---

## Cell IDs

Every cell has an `id` in `notebook.toml`. That ID is what `@after`
and `@loop start_from=` reference across cells. The defaults:

- **Created via the UI or REST**: the backend generates an 8-character
  UUID prefix like `a1b2c3d4`. This is what you'll see in fresh
  notebooks.
- **Hand-written notebook.toml**: any string is accepted, as long as
  it's unique within the notebook. Friendly IDs like `seed`,
  `threshold`, or `top-orders` are common in example notebooks where
  the author wants `@after seed` to read like prose.

`notebook.toml` example:

```toml
cells = [
    { id = "seed",      file = "seed_database.py", language = "sql",    order = 0 },
    { id = "threshold", file = "threshold.py",     language = "python", order = 1 },
    { id = "top-orders", file = "top_orders.py",   language = "sql",    order = 2 },
]
```

…then in `cells/top_orders.py`:

```sql
# @sql connection=warehouse
# @after seed
SELECT category, COUNT(*) FROM products GROUP BY category
```

The `@after seed` resolves to the cell with `id = "seed"`. If you'd
rather not hand-edit, opening the notebook in the UI also lets you
rename cell IDs via the cell's metadata panel.

`@name` and `id` are independent: a cell can have `id = "seed"` and
`@name "Seed database"` simultaneously. The DAG view shows the
`@name`; `@after` and `@loop start_from=` use the `id`.

---

## @worker

Route the cell's execution to a named worker instead of the local machine.

```python
# @worker df-cluster
category_stats = ctx.sql("SELECT topic, COUNT(*) FROM papers GROUP BY topic").to_pandas()
```

Workers are HTTP endpoints that implement the Strata executor protocol. Register them via the **Workers panel** in the sidebar, the persisted result lands in `notebook.toml` as:

```toml
[[workers]]
name = "df-cluster"
backend = "executor"
runtime_id = "df-cluster"

[workers.config]
url = "https://my-datafusion-worker.fly.dev/v1/execute"
transport = "http"
```

During execution, the UI shows a pulsing "dispatching → df-cluster" badge on the cell. After completion, the worker name appears in the cell's metadata.

Workers can be anything that speaks HTTP: a GPU box on RunPod, a DataFusion cluster on Fly, a beefy EC2 instance, or a local process on a different port. The built-in `remote_executor.py` provides a reference implementation:

```bash
python -m strata.notebook.remote_executor --port 9000
```

If no `@worker` is set, the cell runs locally in the notebook's Python environment.

---

## @timeout

Override the execution timeout for a single cell, in seconds. The default
is 300 seconds (5 minutes); the value must satisfy `0 < t ≤ 86400` (one day max).

```python
# @timeout 300
# @worker my-gpu
embeddings = model.encode(abstracts, batch_size=256)
```

Useful for cells that download data, train models, or call slow external APIs. The timeout applies to the full execution including any remote worker round-trip.

!!! warning "Prompt-cell timeout vs AI API timeout"
    Prompt cells have two timeouts that can collide silently. The
    cell-level `# @timeout` (default **300 s**) wraps the whole cell;
    the AI API call inside has its own timeout from
    `STRATA_AI_TIMEOUT_SECONDS` / `[ai] timeout_seconds` in
    `notebook.toml` (default **60 s**).

    With defaults, the cell-level wrap fires first and you get a
    cell timeout while the API call would have succeeded eventually.
    Set `# @timeout` on prompt cells to at least
    `STRATA_AI_TIMEOUT_SECONDS + a few seconds of slack` (e.g.
    `# @timeout 90`), or lower `STRATA_AI_TIMEOUT_SECONDS` to match
    the cell budget.

---

## @env

Set an environment variable for this cell only, overriding the notebook-level value.

```python
# @env CUDA_VISIBLE_DEVICES=0
# @env OMP_NUM_THREADS=4
import torch
model = torch.nn.Linear(384, 10).cuda()
```

Format: `# @env KEY=value`. Multiple `@env` lines are supported. The variable is available in `os.environ` during cell execution.

!!! warning "Don't put secrets in `# @env`"
    `# @env` values live in **committed cell source** (`cells/*.py`).
    The sensitive-key blanking that the notebook.toml `[env]` writer
    applies (KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL values blanked
    before commit) does **not** apply to `# @env` — the literal value
    you type goes straight to git.

    For API keys and other secrets, use the notebook's `[env]` block
    in `notebook.toml` (blanked automatically) or the Runtime panel
    in the UI (kept in the server process only). See
    [notebook.toml `[env]`](../reference/notebook-toml.md#env-notebook-environment-variables).
    `# @env` is for non-sensitive overrides like `CUDA_VISIBLE_DEVICES`
    or `OMP_NUM_THREADS`.

---

## @mount

Attach a filesystem mount to the cell. Mounts provide read or read-write access to external storage (S3, local paths) during execution.

```python
# @mount raw_data s3://my-bucket/dataset ro
# @mount scratch file:///tmp/work rw
df = pd.read_parquet(raw_data / "events.parquet")
scratch / "summary.txt"  # → Path("/tmp/strata/mounts/.../summary.txt")
```

**The mount name becomes a `pathlib.Path` variable in the cell's namespace.** No
`/mnt/<name>` directory convention — the variable directly references the
resolved local path (the cached mirror for remote URIs, the URI's local
filesystem path for `file://` URIs). Use standard `Path` operations: `/` for
joining, `.read_text()`, `.iterdir()`, etc.

Format: `# @mount <name> <uri> [ro|rw]`. Defaults to `ro` (read-only) if the mode is omitted. The mount name must be a valid Python identifier (it's an injected variable).

---

## @table

Declare an Iceberg table input. The table's current snapshot id becomes part
of the cell's provenance: **when new data lands in the table, the cell goes
stale and the normal cascade machinery re-runs it** — no manual data-version
bookkeeping. For an end-to-end walkthrough (build a warehouse, scan it,
retrain on new data, pin a snapshot), see
[Lake-Aware Cells](lake-aware-cells.md).

```python
# @table trips file:///data/warehouse#nyc.trips
art = client.materialize(
    inputs=[trips],
    transform={"executor": "scan@v1", "params": {"snapshot_id": trips_snapshot}},
)
```

**Two variables are injected into the cell's namespace**: `<name>` — the table
URI string — and `<name>_snapshot` — the snapshot id resolved when the cell's
provenance was computed. Passing `<name>_snapshot` to the scan makes the cell
fully deterministic: it reads exactly the snapshot its provenance recorded.

Format: `# @table <name> <uri> [snapshot=<id>]`. The URI is
`<warehouse>#<namespace>.<table>` — the same format `client.materialize`
accepts. The name must be a valid Python identifier.

`snapshot=<id>` pins the table: the cell reads that snapshot forever and never
goes stale on new data (the lake-side analog of a mount `pin`). Without a pin,
the snapshot is re-resolved every time staleness is evaluated.

Like mount variables, the injected names live only in the declaring cell's
namespace — they are not cell *defines* and do not flow to downstream cells.
To use the snapshot id downstream, export it as a real variable:
`scanned_snapshot = trips_snapshot`.

If the catalog is unreachable when provenance is computed, the cell is
conservatively treated as stale; if it is still unreachable at execution time,
the run fails with a clear error.

---

## Prompt Cell Annotations

Prompt cells (language `prompt`) accept an additional set of annotations that
configure the AI call.

### `@model`

Override the notebook-level AI model for this cell only.

```
# @model claude-sonnet-4-6
Summarize {{ df }} in one paragraph.
```

### `@temperature`

Sampling temperature. Defaults to `0.0`.

```
# @temperature 0.3
```

### `@max_tokens`

Ceiling on output tokens for this call.

```
# @max_tokens 1024
```

### `@system`

System prompt prepended to the conversation.

```
# @system You are a terse data analyst. Answer in bullet points.
```

Multiple `@system` lines are concatenated with newlines.

### `@output`

Force the response format.

```
# @output json
```

Accepts `json` (the response is parsed/coerced as JSON) or `text`
(free-form text — the default). Auto-set to `json` when
`@output_schema` is present, so the schema and `# @output json` don't
need to appear together.

### `@output_schema`

Inline JSON Schema pinning the response shape. When provided, Strata
dispatches to provider-native structured output (OpenAI's `json_schema` with
strict mode; Anthropic's native tool-use) so the response comes back as
validated JSON rather than free-form text. Providers without schema support
fall back to `json_object`, valid JSON, shape not enforced, and the
`@validate_retries` loop catches shape violations.

```
# @output_schema {"type": "object", "properties": {"themes": {"type": "array", "items": {"type": "string"}}}, "required": ["themes"]}
```

Editing the schema invalidates the cell's cache, the schema is part of the
provenance hash.

### `@validate_retries`

Total attempts for the validate-and-retry loop (1 initial call + N-1 retries).
Defaults to 3. Only takes effect when `@output_schema` is set; each failed
validation feeds the prior response and path-addressed errors back as a retry
turn.

```
# @validate_retries 5
```

---

## Loop Cell Annotations

A Python cell carrying `@loop` is executed iteratively. The body runs once per
iteration and the `carry` variable threads state between them.

### `@loop`

```python
# @loop max_iter=50 carry=state
# @loop_until state["converged"]
state = state if "state" in dir() else initial
state = step(state)
```

Key/value parameters:

- `max_iter=<N>`, hard upper bound on iterations.
- `carry=<var>`, the variable threaded between iterations.
- `start_from=<cell-id>@iter=<k>`, (optional) resume from another loop cell's
  stored iteration `k`. Useful for forking a converged run to explore a
  variant. `<cell-id>` is the upstream loop cell's `id` in
  `notebook.toml` (not its `@name`) — see [Cell IDs](#cell-ids).

### `@loop_until`

Python expression evaluated after each iteration in the cell's namespace. When
it returns truthy, the loop exits early.

```python
# @loop max_iter=100 carry=acc
# @loop_until acc["loss"] < 0.05
```

Each iteration's carry state is stored as `…@iter=k` artifacts; the final
iteration becomes the cell's canonical artifact. Progress is broadcast over
WebSocket as `cell_iteration_progress` messages.

---

## SQL Cell Annotations

A cell with `language = "sql"` runs a query through a declared connection.
See [SQL Cells](cells.md#sql-cells) for the full feature walkthrough; this
section is the per-annotation reference.

### `@sql`

Marks the cell as SQL and binds it to a named connection.

```sql
# @sql connection=warehouse
SELECT * FROM orders WHERE amount > :min_amount
```

Key/value parameters:

- `connection=<name>`, required. Must reference an entry under
  `[connections.<name>]` in `notebook.toml`. Manage these via the
  **Connections panel** in the sidebar; you don't need to edit the file
  directly.
- `write=true`, opt the cell into writable execution. Without this flag,
  the connection is opened in enforced read-only mode (SQLite `mode=ro` +
  `PRAGMA query_only=ON`; PostgreSQL `SET default_transaction_read_only =
  on`) and any DDL/DML errors before mutating the database. With it, the
  cell can run setup scripts (`CREATE TABLE`, `INSERT`, `DROP`). The flag
  is per-cell, read cells using the same connection stay read-only.

```sql
# @sql connection=warehouse write=true
DROP TABLE IF EXISTS events;
CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT NOT NULL);
INSERT INTO events VALUES (1, 'alpha'), (2, 'beta');
```

Write cells split the body into individual statements via sqlglot, run each
in sequence, and emit a per-statement status table (`stmt`, `kind`,
`rows_affected`). Default cache policy for write cells is `session`;
`fingerprint` and `snapshot` error early because probe-based invalidation
has no anchor when the cell mutates state.

### `@cache`

Override the default `fingerprint` cache policy on a SQL cell.

| Policy            | Behavior                                                     |
| ----------------- | ------------------------------------------------------------ |
| `fingerprint`     | Default. Probe-derived freshness token + schema fingerprint folded into the hash. |
| `forever`         | Static salt; never invalidates from DB-side state.           |
| `session`         | Session-unique salt; invalidates across sessions.            |
| `ttl=<seconds>`   | `floor(now / ttl)` bucketed time-based salt.                 |
| `snapshot`        | Probe MUST return a durable snapshot ID. Errors at execute time when the driver can't (SQLite/Postgres can't; Iceberg-via-engine can). |

```sql
# @sql connection=warehouse
# @cache forever
SELECT * FROM dim_country
```

`# @cache snapshot` requires `AdapterCapabilities.supports_snapshot = True`
on the driver; otherwise the resolver fails fast before any connection is
opened. Per-driver freshness probe details are in
[SQL Cells](cells.md#per-driver-freshness).

### `@name`

For SQL cells, `@name` sets the output variable name (default `result`),
the same way it does for prompt cells.

```sql
# @sql connection=warehouse
# @name top_customers
SELECT customer, SUM(amount) AS total
FROM orders GROUP BY customer ORDER BY total DESC LIMIT 5
```

A downstream Python cell can then reference `top_customers` directly as a
pandas DataFrame.

---

## Variant Cells

Variant cells let you keep multiple alternative implementations of the
same DAG slot side by side and switch between them. The canonical use
case is "we want to try three models for this experiment": three
training cells all produce a `model` variable, and downstream cells
reference `model` without caring which variant produced it.

```python
# @variant classifier logreg
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000).fit(X_train, y_train)
```

```python
# @variant classifier rf
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=200).fit(X_train, y_train)
```

Both cells declare `# @variant <group> <name>` with the same group
(`classifier`) and different names (`logreg`, `rf`). At any given time
exactly one variant is **active**; only the active variant participates
in the DAG, so downstream cells see one producer for `model`. In the UI
the group renders as a tab strip, clicking a tab switches the active
variant, and the cell editor shows that variant's source.

### Switching variants

The active variant per group is persisted in `notebook.toml`:

```toml
[[variant_group]]
group = "classifier"
active = "rf"
```

Switching is a one-line diff. Each variant carries its own provenance
hash, so re-running a variant you've already trained is a cache hit:
flip-flopping between two variants is free after each has run once.
Downstream cells go stale on switch (their input artifact comes from
a different upstream cell) but become cache hits on the way back.

If `notebook.toml` doesn't pin a selection, or pins a name no cell
provides, the DAG falls back to the **first variant in source order**.
A `variant_active_unknown` diagnostic surfaces in the UI when the
selection drifts (e.g. you renamed a variant in source without updating
the toml entry).

### Defines contract

All variants in a group must produce the same set of top-level
bindings. The validator compares each variant's `defines` against its
siblings and flags `variant_contract_mismatch` on any outlier, if
`logreg` exposes only `model` and `rf` exposes `model + feature_importance`,
downstream cells that depend on the missing name would break under one
selection but not the other.

Imports don't count toward the contract, they're scaffolding, not
interface. The variants above each bring in a different sklearn class,
which is fine; only the *values* the cells produce need to match.

### Mixing cell kinds

A variant group can mix any cell kinds. A Python variant and a prompt
variant can sit in the same group as long as they both produce the
contract names, e.g. one variant calls a deterministic regex
classifier, another asks an AI model to classify.

### Adding and removing variants

The variant tab strip carries a `+` button that clones the active
variant as a sibling. The new cell starts as a copy of the active
body with the `# @variant` line rewritten to an auto-generated name
(`<active>_copy`, then `_copy2`, `_copy3`, …). Rename happens by
editing the annotation line in source, the standard
annotation-as-truth pattern, no separate rename UI.

Deleting a variant tab removes only that variant. If you delete the
active one, the next variant in source order auto-promotes. Deleting
the last variant in a group removes the cell entirely *and* drops the
`[[variant_group]]` entry, the group dissolves.

### Bootstrapping

The first variant of a new group is created by typing the annotation:
add `# @variant <new_group> <variant_name>` to any existing cell, save
it, then use the `+` tab to add siblings. (There's no UI affordance for
the bootstrap step, source is the only place a group comes into
existence, which keeps `notebook.toml` honest about *which* groups
exist.)

---

## Cross-Cell Ordering

### `@after`

Add an ordering-only DAG edge from another cell to this one. Useful when the
dependency is on a side effect, e.g. a SQL `seed` cell creates the database
state that subsequent SQL cells query, and no Python variable flows
between them.

```sql
# @sql connection=warehouse
# @after seed
SELECT category, COUNT(*) FROM products GROUP BY category
```

Multiple `@after` lines stack; each cell ID adds one edge. Whitespace-
separated IDs on a single line work too: `# @after seed migrate`. Self-
references and unknown cell IDs are silently dropped at the DAG layer
(annotation_validation surfaces them as a diagnostic for the user).

`<cell-id>` is the `id` field in `notebook.toml`, **not** the cell's
`@name`. See [Cell IDs](#cell-ids) for how to set friendly IDs like
`seed` or `migrate`.

The edge participates in upstream/downstream wiring and the topological
order, but contributes no variable to `consumed_variables`, so it
doesn't affect per-variable provenance hashes.

---

## Precedence Rules

When the same setting is configured at multiple levels, the most specific wins:

| Setting | Annotation | Cell config (notebook.toml) | Notebook default |
|---------|-----------|---------------------------|-----------------|
| **Worker** | `# @worker X` | `cell.worker` field | `notebook.worker` field |
| **Timeout** | `# @timeout N` | `cell.timeout` field | 300 seconds |
| **Env vars** | `# @env K=V` | `cell.env` overrides | `notebook.env` defaults |
| **Mounts** | `# @mount ...` | `cell.mounts` overrides | `notebook.mounts` defaults |
| **SQL connection** | `# @sql connection=X` |, | none, required for SQL cells |
| **Cache policy** | `# @cache <policy>` |, | `fingerprint` (read), `session` (write) |

Annotations always take priority. This lets you override per-cell behavior without editing `notebook.toml`.

---

## Notebook-Level Environment (Runtime Panel)

Notebook-wide environment variables are set via the **Runtime panel** in the sidebar. These apply to all cells unless overridden by a cell-level `@env` annotation.

Common use cases:

- **API keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (for prompt cells and AI assistant)
- **Database URLs**: `DATABASE_URL`, `REDIS_URL`
- **Feature flags**: `DEBUG=true`, `LOG_LEVEL=info`

!!! note "Sensitive values are not persisted to disk"
    Environment variables with names containing KEY, SECRET, TOKEN, PASSWORD, or CREDENTIAL have their values blanked from `notebook.toml` when saving. The key names are preserved as a "which vars are configured" reminder *only* when something real is configured alongside them. A notebook whose `[env]` would contain nothing but blanked sensitive slots is persisted without an `[env]` block at all, so typing an API key in the Runtime panel doesn't churn the committed notebook.

Notebook env vars are stored in the `[env]` section of `notebook.toml`:

```toml
[env]
DATABASE_URL = "postgres://localhost/mydb"
OPENAI_API_KEY = ""  # value blanked; name kept because DATABASE_URL above is real config
```
