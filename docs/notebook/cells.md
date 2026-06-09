# Cell Types

Strata Notebook has five cell kinds:

| Kind       | What it runs                            | Created by                                                    |
| ---------- | --------------------------------------- | ------------------------------------------------------------- |
| **Python** | Python source in the notebook's venv    | The default, pick **Python** from the **+ Add cell** menu    |
| **Prompt** | A text template sent to an AI model     | Pick **Prompt** from the **+ Add cell** menu                  |
| **SQL**    | A query against a connected database    | Pick **SQL** from the **+ Add cell** menu                     |
| **Loop**   | A Python cell executed N times in a row | Add a Python cell, then put a `# @loop` annotation at the top |
| **R**      | R source via the system `Rscript`       | Pick **R** from the **+ Add cell** menu                      |

All five participate in the DAG, cache by provenance hash, and can be routed to remote workers. Pick the kind that matches the shape of the computation, this page walks through each.

See [Concepts](concepts.md) for the execution model; see [Cell Annotations](annotations.md) for the full per-annotation reference.

---

## Python Cells

The default. A Python cell is just Python source, assignments at module scope become the cell's outputs, and free variables become inputs pulled from upstream cells.

### Writing a Python cell

```python
import pandas as pd

sales = pd.read_parquet("https://example.com/sales.parquet")
by_region = sales.groupby("region")["total"].sum()
```

This cell _defines_ `sales` and `by_region`. A downstream cell that references either name will automatically depend on this one.

```python
# downstream cell, reads by_region from upstream
top_region = by_region.idxmax()
print(f"Top region: {top_region}")
```

```text title="Output"
Top region: West
```

### Variable flow and the DAG

Strata analyzes each cell's AST to extract:

- **Defines**: top-level assignments (`x = 1`, `df = pd.read_csv(...)`)
- **References**: free variables used but not defined locally

The DAG builder links references back to the **last** cell that defined each name (shadowing is handled by order). Edges flow producer → consumer. When you edit an upstream cell, every downstream cell that depends on it becomes stale automatically.

Only variables that a downstream cell actually references get stored as artifacts. Intermediate scratch variables stay in the subprocess and are discarded when the cell finishes.

### The ambient `strata` client

Every **locally-executed** Python cell gets a ready `strata` client in its namespace, already bound to the running server — no import or construction needed:

```python
# no `from strata.client import StrataClient` / `StrataClient(base_url=...)`
art = strata.materialize(
    inputs=[trips],
    transform={"executor": "scan@v1", "params": {"snapshot_id": trips_snapshot}},
)
strata.set_alias("taxi/tip-model", "champion", art.artifact_id, art.version)
```

It's a lightweight client with the same API surface for the common operations (`materialize`, `put`, `set_alias`, `set_tag`, `resolve_alias`, …); explicit `from strata.client import StrataClient` still works if you want the full client. The ambient `strata` is created fresh per cell run and closed automatically — you don't manage its lifecycle. It's an injected runtime tool, not a cell variable: it doesn't flow downstream and isn't part of provenance (using it has no effect on staleness), exactly like a `@mount` path or an `@table` variable.

Cells routed to a **remote executor worker** (not local execution) don't get the ambient `strata` — import a client explicitly there.

When a cell publishes with a name — `strata.put(model, name="taxi/tip-model")` — the artifact appears in the [registry dashboard](../core/registry.md#in-the-notebook-the-registry-dashboard): a promote strip under the cell, and the Registry tab in the bottom drawer (promote, approve, lineage — all in the UI).

> Calls through `strata` are **side effects**. On a cache hit the cell body doesn't re-run, so a `strata.set_alias(...)` won't re-fire — fine for idempotent calls (setting an alias to the version it already points at is a no-op), and side-effect-only cells (no stored output) re-run every time anyway.

### Library cells (cross-cell defs and classes)

Top-level `def` and `class` definitions are shared across cells via a synthetic Python module, write a helper once, call it anywhere.

```python
import math

CIRCLE_PRECISION = 4

def area(r):
    return round(math.pi * r * r, CIRCLE_PRECISION)

def perimeter(r):
    return round(2 * math.pi * r, CIRCLE_PRECISION)
```

Downstream cells reference `area(7.5)`, `perimeter(7.5)`, and `CIRCLE_PRECISION` directly.

#### How sharing works (slicing)

Defs and classes don't pickle reliably across the subprocess boundary, so they round-trip via **source reconstitution**: Strata writes a slice of the cell's source to disk, re-executes it in a fresh module on the consumer side, and hands the downstream cell the resulting attribute. The slice must be side-effect-free.

The slice keeps the module docstring, `import` / `from import` (no `from X import *`), `def` / `async def`, `class`, and assignments whose RHS is a **literal constant**: numbers, strings, bools, `None`, bytes, negations of literals, and nested tuples/lists/sets/dicts of literals. Everything else (non-literal assignments, augmented assigns, expression statements, control flow, bare annotations) is dropped from the slice but stays in the cell's runtime execution.

A single cell can therefore mix runtime work and library code:

```python
# Runtime, dropped from the slice; flows through the artifact path.
raw_min = round(-math.tau * 7, 2)
raw_max = round(math.tau * 16, 2)
print(f"loaded raw bounds: [{raw_min}, {raw_max}]")

# Library, kept in the slice, exported as a synthetic module.
CLAMP_MIN = 0.0
CLAMP_MAX = 100.0

def clamp(value):
    return max(CLAMP_MIN, min(CLAMP_MAX, value))
```

```text title="Output"
loaded raw bounds: [-43.98, 100.53]
```

A downstream cell can call `clamp(raw_max)`, `clamp` and `CLAMP_MIN/MAX` come from the synthetic module, `raw_max` from the artifact path.

#### When the slice isn't self-contained

Every name a kept def or class references must be bound by something else in the slice (or a Python builtin). When it isn't, Strata blocks the export with a `module_export_blocked` diagnostic, surfaced pre-flight, not just at run time.

```python
runtime_threshold = math.sqrt(9)   # dropped, non-literal RHS

def is_outlier(value):
    return value > runtime_threshold
```

> `is_outlier` references names not defined or imported in this cell: runtime_threshold

Other shapes that block on the same principle:

- **Decorators / default values / base classes** evaluated at module load: `@my_decorator` where `my_decorator` isn't imported in the same cell, or `class Child(Parent)` where `Parent` is computed at runtime.
- **Divergence**: a name kept by the slice is also reassigned by dropped runtime code, so the synthetic module's value would differ from the cell's final state. `def f(): ...; f = wrap(f)` exports the unwrapped `f`.
- **Lambda assignments**: `add = lambda x: x + 1`, even though `cloudpickle` could serialize the value, the synthetic-module path is reserved for source-backed library code.

The fix is usually one of: move the runtime line into its own cell, add the missing import to the same cell as the def, or take the dependency as a function argument.

#### Single-cell scope

The synthetic module is built from one cell's source only, no transitive composition across cells. A def can't reach a name imported or defined in a different cell; each cell that hosts library code carries its own imports.

One concession: annotations that reference names outside the slice would normally block, but adding `from __future__ import annotations` relaxes this. PEP 563 stringifies annotations and the free-variable check drops them, so cross-cell type hints "just work" with the future import.

Walked through end-to-end in the [`library_cells`](../examples/library_cells.md) example notebook.

### Mutation warnings

If a cell mutates a value it received from an upstream cell (e.g. `df.drop(columns=[...], inplace=True)`), Strata raises a **mutation warning**: the upstream artifact was supposed to be immutable, and subsequent cells that reuse the cached artifact will see the mutated version.

The fix is to copy before mutating:

```python
df = upstream_df.copy()    # make a private copy
df.drop(columns=[...], inplace=True)
```

Warnings surface as a pill on the cell and a structured entry in the execution log.

### Python-cell annotations

| Annotation         | What it does                                     |
| ------------------ | ------------------------------------------------ |
| `# @name X`        | Display name for the DAG view                    |
| `# @worker X`      | Route execution to a named remote worker         |
| `# @timeout 60`    | Override execution timeout (seconds, default 300) |
| `# @env KEY=value` | Set an env var for this cell only — non-sensitive values only; literal lands in committed source. For secrets use `notebook.toml [env]` or the Runtime panel. |
| `# @mount …`       | Attach a filesystem mount (see [Annotations][a]) |
| `# @loop …`        | Turn the cell into a [loop cell](#loop-cells)    |

See [Cell Annotations][a] for the full reference.

[a]: annotations.md

---

## Prompt Cells

A prompt cell is a text template that gets rendered with upstream variable values, sent to an AI model, and the response stored as an artifact. Prompt cells participate in the DAG and cache by provenance exactly like Python cells, same inputs + same template + same model config = cache hit, no API call.

Create a prompt cell with the **"Add Prompt Cell"** button in the UI, the same toolbar that adds a Python cell. You never need to touch `notebook.toml` directly; editing the cell's source, wiring it into the DAG, and persisting the result all happen through the UI.

### Basic syntax

```
# @name summary
Summarize this dataset and return the top 3 findings as a numbered list:

{{ df }}
```

```text title="Output (illustrative model response)"
1. Setosa is linearly separable from versicolor and virginica based on petal dimensions alone.
2. Versicolor and virginica overlap moderately on sepal width but separate well by petal length.
3. The dataset is balanced, 50 samples per species, no missing values.
```

- `{{ df }}` is replaced with a text representation of the upstream variable `df` before sending to the model.
- The model's response is stored as an artifact named `summary` (from `# @name`).
- Downstream cells can read `summary` like any other upstream variable.

### Template syntax

Variables are injected with `{{ expression }}`. The expression is resolved against upstream cell outputs and converted to text using type-specific rules:

| Upstream type     | Text representation                     |
| ----------------- | --------------------------------------- |
| pandas DataFrame  | Markdown table (first 20 rows)          |
| pandas Series     | String representation (first 20 values) |
| numpy ndarray     | Shape + dtype + first 10 elements       |
| dict / list       | JSON, indented                          |
| str / int / float | Direct string conversion                |

Each variable has a 2,000-token budget per template render. Oversized values are truncated with a `... (truncated)` marker.

**Attribute access** is supported for safe read-only operations:

```
{{ df.describe() }}     # OK, pandas describe() is allow-listed
{{ df.head() }}         # OK
{{ obj.attr }}          # OK, attribute access (non-callable)
{{ obj.mutate() }}      # blocked, unknown method, left as-is in the template
```

Only a small set of methods is permitted (`describe`, `head`, `tail` on pandas objects). Arbitrary method calls are blocked to keep template rendering side-effect-free.

### Prompt-cell annotations

| Annotation               | What it does                                                              | Default              |
| ------------------------ | ------------------------------------------------------------------------- | -------------------- |
| `# @name <identifier>`   | Output variable name; must be a Python identifier                         | `result`             |
| `# @model <model_id>`    | Override the notebook-level AI model                                      | From provider config |
| `# @temperature <float>` | Sampling temperature (0.0 = deterministic; see [Caching](#caching) below) | `0.0`                |
| `# @max_tokens <int>`    | Response token ceiling                                                    | `4096`               |
| `# @system <text>`       | System prompt prepended to the request                                    | None                 |
| `# @output json\|text`   | Coerce the response to JSON (or keep as free-form text)                   | `text`               |
| `# @output_schema {…}`   | Inline JSON Schema pinning the response shape                             | None                 |
| `# @validate_retries N`  | Total attempts for the validate-and-retry loop (1 initial + N−1 retries)  | `3`                  |

Example using several at once:

```
# @name classification
# @model gpt-5.4
# @temperature 0.0
# @max_tokens 1000
# @system You are a data scientist. Return only valid JSON.
Classify each paper by topic:

{{ sampled_papers }}

Return a JSON object mapping paper ID to topic.
```

### Schema-constrained output

`# @output_schema {...}` pins the shape of the model response to an inline JSON Schema. Strata picks the best provider-native path:

| Provider                             | Enforcement                                                                                                                                                                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI**                           | Native `response_format: {type: "json_schema"}`. `additionalProperties: false` is auto-injected at every `object` node; strict mode is used when the user's `required` list covers every property (otherwise relaxed to `strict: false`). |
| **Anthropic**                        | Native `/v1/messages` with tool-use: the schema is sent as a tool's `input_schema` and `tool_choice` is forced to that tool. The returned `tool_use.input` is extracted verbatim.                                                         |
| **Gemini / Mistral / Ollama / vLLM** | Fallback to `response_format: {type: "json_object"}`, valid JSON guaranteed, shape not enforced server-side. Client-side validation (see below) fills the gap.                                                                           |
| **Servers that reject the extensions** | Some OpenAI-compatible servers 400 on `response_format` or `stream_options` outright. Strata retries once without them, appending a schema-guidance system turn; the result is marked degraded and the validate-and-retry loop carries full enforcement. |

Setting `@output_schema` implies `@output json`; you don't need both.

Validation is lenient about packaging: when a provider wraps otherwise-valid
JSON in code fences or prose, Strata extracts the JSON document before
validating instead of burning a retry on the wrapper.

Example, triage each review into a structured record:

```
# @name triage
# @output_schema {"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"sentiment":{"type":"string","enum":["positive","negative","neutral"]},"priority":{"type":"string","enum":["low","medium","high"]},"tags":{"type":"array","items":{"type":"string"}}},"required":["sentiment","priority","tags"]}}},"required":["items"]}
Triage these customer reviews:

{{ reviews }}

For each review return sentiment, priority, and 1–3 short tags.
```

A downstream cell can then destructure without regex-wrangling:

```python
import pandas as pd
df = pd.DataFrame(triage["items"])
print(df["priority"].value_counts())
```

!!! note "Long schemas don't have an escape hatch yet"
    `# @output_schema` is parsed as a single JSON value from a single
    annotation line (`prompt_analyzer.py:143-154`). For real-world
    schemas this gets long fast and the resulting cell source isn't
    pretty.

    Two pragmatic workarounds while a multi-line / file-reference
    syntax is on the backlog:

    - **Generate the schema in an upstream Python cell** and write
      the schema-driven prompt cell by hand. The Python cell's
      `json.dumps(...)` keeps the schema readable in source; the
      prompt cell pastes the result inline.
    - **Keep the schema in a `schemas/`-style sibling file** inside
      the notebook directory, and have a setup cell load + cache it
      as a string. Downstream prompt cells can use a shorter inline
      schema for caching while documenting the rich version
      elsewhere.

    Track the multi-line annotation feature in the issue tracker if
    this is blocking you.

```text title="Output"
priority
high      4
medium    2
low       1
Name: count, dtype: int64
```

### Validate-and-retry

When `@output_schema` is set, Strata runs a **validate-and-retry loop** after every model call:

1. Parse the response as JSON and run it through `jsonschema`.
2. On success → store the artifact and return.
3. On failure → append the bad response as an `assistant` turn, feed the validator's path-addressed errors back as a `user` turn, and retry.
4. On retry exhaustion → surface a cell error with the last validator messages.

The default is 3 total attempts (1 initial + 2 retries). Override with `# @validate_retries N`. Cumulative input/output tokens across all attempts are recorded on the artifact so cost accounting is accurate. The retry count is surfaced on the cell result (`validation_retries`) the UI shows "validated after N retries" when non-zero.

Retries are mostly invisible on OpenAI-strict and Anthropic-native paths because the provider enforces the schema at decode time. They earn their keep on the `json_object` fallback path (Gemini, Mistral, Ollama) where the provider only guarantees _syntactic_ JSON.

### Caching

A prompt cell's provenance hash mixes together:

- The rendered template text (after `{{ var }}` injection)
- Model name
- Temperature
- System prompt
- Output type (`json` / `text`)
- Output schema fingerprint (when set)

Editing any of these invalidates the cache. In particular, tweaking `@output_schema` on a cached cell forces a fresh call, exactly what you want when iterating on the response shape.

!!! tip "Keep temperature at 0.0 for prompt cells"
With `temperature=0.0` the model is deterministic: same inputs → same output, and cache behavior is intuitive. Bumping temperature makes the first response "sticky" in the cache, future runs return the stored stochastic sample rather than re-sampling.

See [AI Integration](ai.md) for provider configuration and the conversational AI assistant.

---

## SQL Cells

A SQL cell sends a query to a connected database via ADBC and stores the result as an Arrow Table artifact. Like Python and prompt cells, SQL cells participate in the DAG, cache by provenance hash, and surface their output to downstream cells.

```sql
# @sql connection=warehouse
SELECT customer, SUM(amount) AS total
FROM orders
WHERE amount > :min_amount
GROUP BY customer
ORDER BY total DESC
```

```text title="Output (illustrative result rows)"
       customer     total
0      acme_inc  18420.50
1   north_winds   9112.75
2  bright_solar   6033.10
```

The cell above pulls `min_amount` from an upstream Python cell, sends a parameterized query through the `warehouse` connection, and stores the resulting rows as an Arrow Table that any downstream cell can consume as a pandas DataFrame.

### Connections

A SQL cell references a **named connection**. Connections live in `notebook.toml` under `[connections.<name>]`, but you don't need to edit that file by hand. Open the **Connections panel** in the right sidebar, click `+ Add connection`, and fill in the form. The driver dropdown switches the field layout per backend (path for SQLite; URI + auth + role + search path for PostgreSQL; account / warehouse / database / schema for Snowflake; project / dataset / credentials for BigQuery).

```toml
[connections.warehouse]
driver = "sqlite"
path = "analytics.db"

[connections.prod]
driver = "postgresql"
uri = "postgresql://localhost:5432/prod"

[connections.prod.auth]
user = "${PGUSER}"
password = "${PGPASS}"
```

Notes:

- **Driver-specific extras** (e.g. `options.search_path`, `options.warehouse` for Snowflake, future driver-specific keys) round-trip through the editor unchanged. The form editorializes the keys it knows; everything else is preserved.
- **Auth values use `${VAR}` indirection.** Literal credentials get blanked when `notebook.toml` is saved, so committing the file never leaks secrets. The form shows a warning border on a literal value so you know to switch it to a variable reference.
- **Relative `path` values are notebook-local.** `path = "analytics.db"` resolves against the notebook directory at execution time. The on-disk value stays relative so a notebook moves cleanly between machines.
- **Currently shipped drivers**: DuckDB, SQLite, PostgreSQL, Snowflake, and BigQuery. DuckDB uses the native DuckDB DBAPI; the other four are ADBC-backed (`adbc-driver-sqlite`, `adbc-driver-postgresql`, `adbc-driver-snowflake`, `adbc-driver-bigquery`). For Snowflake, read cells use `role`; `write=true` cells switch to `write_role` when configured, otherwise they reuse `role`. For BigQuery, read cells use `credentials_path`; `write=true` cells switch to `write_credentials_path` when configured, otherwise they reuse `credentials_path`. Snowflake and BigQuery do not have a session-level read-only flag like PostgreSQL's `SET default_transaction_read_only = on`, so the safety boundary is the grants on the configured role or service account.

### Schema discovery

The **Schema panel** in the sidebar shows the tables and columns visible through each declared connection. Click a connection to lazy-load its schema; click a table to expand its columns. The `↻` button re-fetches when the underlying database has changed externally. No SQL cell needs to run for this, the panel talks directly to each driver's catalog query surface (`sqlite_master` for SQLite, `information_schema.tables JOIN columns` for PostgreSQL, and the driver-specific catalog queries for Snowflake and BigQuery).

### Bind parameters

`:name` placeholders resolve against upstream cell variables. Strata coerces a strict allowlist of Python types (`int`, `float`, `str`, `bytes`, `bool`, `None`, `Decimal`, `UUID`, `datetime`/`date`/`time`) into ADBC bind values; anything else (a list, a numpy scalar, a custom object) is rejected with a clear error. **No string substitution ever**: values flow through ADBC's prepared-statement layer, so adversarial strings (`'; DROP TABLE …`) round-trip as data, not SQL.

```python
# upstream Python cell
min_amount = 100
```

```sql
# @sql connection=warehouse
SELECT * FROM orders WHERE amount > :min_amount
```

```text title="Output (illustrative result rows)"
   order_id  customer_id  amount       placed_at
0      1042            7   245.99  2026-04-12
1      1093           14   180.50  2026-04-13
2      1117            3   299.00  2026-04-14
```

The DAG links the SQL cell to the Python cell automatically, same edge logic Strata uses for Python free variables.

### Cache policies

A SQL cell's **provenance hash** folds together:

- The query text (sqlglot-normalized so whitespace and comment edits don't churn the cache).
- The bind parameters (type-tagged: `True` ≠ `1`).
- The connection's identity (host / DB / user / role / search_path, never the password).
- The hashes of every upstream artifact referenced via `:name`.
- The driver's **freshness probe** result for the touched tables.
- The driver's **schema fingerprint** for the touched tables.
- A salt derived from the `# @cache` policy below.

`# @cache` controls how DB-side state factors in. Default is `fingerprint`.

| Policy          | Behavior                                                                                                                  | When to use                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `fingerprint`   | Default. Probe-derived freshness token + schema fingerprint folded in.                                                    | Most queries.                      |
| `forever`       | Static salt; never invalidates from DB-side state.                                                                        | True reference data. User asserts. |
| `session`       | Session-unique salt; invalidates across sessions.                                                                         | Always-fresh queries / dashboards. |
| `ttl=<seconds>` | `floor(now / ttl)` in the salt; bucketed time-based invalidation.                                                         | Stale-tolerant aggregations.       |
| `snapshot`      | Probe MUST return a durable snapshot ID. Errors at execute time if the driver can't (SQLite/Postgres can't; Iceberg can). | Reproducibility-critical reads.    |

```sql
# @sql connection=warehouse
# @cache forever
SELECT * FROM dim_country
```

### Per-driver freshness

`fingerprint` correctness depends on what the driver can probe.

| Driver     | Probe                                           | Granularity | Notes                                                                                                                                                                                                                                                                                                                                                        |
| ---------- | ----------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PostgreSQL | `pg_stat_user_tables` + `pg_class.relfilenode`  | per-table   | Up to ~500 ms stats-collector lag.                                                                                                                                                                                                                                                                                                                           |
| SQLite     | `PRAGMA data_version` + `PRAGMA schema_version` | **DB-wide** | DML cross-process needs the probe connection open across the write, `data_version` resets on a fresh connection. DDL (schema change) invalidates cleanly.                                                                                                                                                                                                   |
| Snowflake  | `INFORMATION_SCHEMA.TABLES.LAST_ALTERED`        | per-table   | Per-database scoping (one query per touched database). Bills cloud-services credits but each query is small. `LAST_ALTERED` updates even on 0-row DML, safe direction (over-invalidates, never under).                                                                                                                                                      |
| BigQuery   | `__TABLES__.last_modified_time`                 | per-table   | Per-dataset scoping. `__TABLES__` is the legacy-but-stable view; `INFORMATION_SCHEMA.TABLES` doesn't expose `last_modified_time`. **Streaming-buffer caveat**: tables receiving streaming inserts have `last_modified_time` lag by minutes-to-90-min until the buffer flushes, pin `# @cache session` on those queries. Permissions: `bigquery.tables.get`. |

The schema fingerprint catches metadata-only changes (`ADD COLUMN`, type changes, nullability flips) that the freshness token would miss.

### Read-only by default

SQL cells are **read-only by default**, but the enforcement mechanism depends on the backend:

- **SQLite**: `mode=ro` plus `PRAGMA query_only=ON`
- **PostgreSQL**: `SET default_transaction_read_only = on`
- **Snowflake**: the configured `role` must be read-only
- **BigQuery**: the configured `credentials_path` must point at a read-only service account

For SQLite and PostgreSQL, Strata enforces read-only at the connection/session level. For Snowflake and BigQuery, Strata selects the read-scoped role or credentials, and the cloud platform's grants are the actual boundary. In all cases, the default path is “read unless you explicitly opt into `write=true`.”

### Write cells

Setup, seeding, and migration scripts opt into writable execution per cell:

```sql
# @sql connection=warehouse write=true
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer TEXT NOT NULL,
    amount REAL
);
INSERT INTO orders VALUES (1, 'alice', 25.50), (2, 'bob', 199.99);
```

- The body is split into individual statements via sqlglot (ADBC's cursor runs only the first statement otherwise).
- `:name` bind placeholders work the same as in read cells.
- The default cache policy is `session` (one execution per session; same body in the same session is a cache hit).
- `# @cache fingerprint` and `# @cache snapshot` error early on write cells, probe-based invalidation has no anchor when the cell mutates state.
- The cell still produces an Arrow artifact: a per-statement status table with `stmt`, `kind` (`CREATE TABLE`, `INSERT`, …), and `rows_affected` (nullable; `null` for DDL).
- Read cells using the same connection stay on the read path, the override is per-cell.

### `# @name` and downstream consumption

A SQL cell's output variable name defaults to `result`; override with `# @name <identifier>`. Downstream cells access the result as a pandas DataFrame (Arrow IPC artifacts deserialize through the standard notebook serializer):

```sql
# @sql connection=warehouse
# @name top_customers
SELECT customer, SUM(amount) AS total
FROM orders GROUP BY customer ORDER BY total DESC LIMIT 5
```

```python
# downstream Python cell
print(top_customers.shape)            # (5, 2)
print(top_customers["total"].sum())   # ndarray sum, etc
```

### `# @after` for setup-then-query pipelines

A read SQL cell that depends on a write SQL cell's side effects (the underlying database state) can declare an explicit ordering edge:

```sql
# @sql connection=warehouse write=true
CREATE TABLE products (sku TEXT PRIMARY KEY, category TEXT);
INSERT INTO products VALUES ('A', 'widgets'), ('B', 'gadgets');
```

```sql
# @sql connection=warehouse
# @after seed
SELECT category, COUNT(*) FROM products GROUP BY category
```

`# @after seed` adds a DAG edge from the `seed` cell to this one even though no Python variable flows between them, the dependency is on a side effect (the SQLite file). This is what cascade execution and staleness recompute use to ensure the right ordering.

### Worked example

The [`sql_orders_report`](../examples/sql_orders_report.md) example notebook walks through all of this end-to-end: a SQL `seed` cell, a Python `threshold` cell, two parameterized SQL queries, and a Python report cell, five cells, two languages, with both `fingerprint` and `forever` cache policies side by side.

### SQL-cell annotations

| Annotation                              | What it does                                          |
| --------------------------------------- | ----------------------------------------------------- |
| `# @sql connection=<name> [write=true]` | Mark the cell as SQL; reference a declared connection |
| `# @cache <policy>`                     | Override the default `fingerprint` cache policy       |
| `# @name <identifier>`                  | Name the output variable (default: `result`)          |
| `# @after <cell-id>`                    | Add an ordering-only DAG edge to an upstream cell     |

See [Cell Annotations][a] for the full reference.

---

## Markdown Cells

Plain prose between cells, rendered with `markdown-it` + `DOMPurify` for safe HTML output. Useful for section headings, methodology notes, and annotating decision points in a notebook. Markdown cells are **not** part of the DAG — they don't produce artifacts, don't participate in cascade execution, and don't have an `id` / variable that downstream cells can reference. They survive saves and exports verbatim.

```markdown
## Stage 1: Load + Clean

The next two cells pull last quarter's events and drop rows with
missing timestamps. The clean DataFrame `events_clean` is what
everything downstream reads.
```

Create a markdown cell via the **+ Add cell** menu (pick "Markdown") or by setting `language = "markdown"` on a cell entry in `notebook.toml`. The cell source lives in `cells/<id>.md` rather than `.py`.

See [Markdown showcase](../examples/markdown_showcase.md) for what the supported renderers handle (headings, lists, tables, code fences, fenced HTML attributes) and what's stripped by the security guard.

---

## Artifact URIs

Cells expose their outputs as artifacts under a stable URI scheme:

```
strata://artifact/<artifact_id>@v=<version>
```

For loop cells, each iteration gets a suffix:

```
strata://artifact/<artifact_id>@v=<version>@iter=<k>
```

The `<artifact_id>` is content-addressed (derived from the provenance hash); same code + same inputs + same env = same artifact ID across machines and runs. The `@v=N` version increments only when the same name pointer is re-bound to a new content hash — see [Library usage](../getting-started/core.md) for how named artifacts work in the Core SDK.

Notebook cell outputs follow the pattern `nb_<notebook_id>_cell_<cell_id>_var_<variable>@v=N` for the variable-level artifacts a cell produces. You don't usually need to construct these by hand; the inspect panel surfaces them and `# @loop start_from=<cell-id>@iter=k` references them by cell ID + iteration index, not the full URI.

---

## Loop Cells

A loop cell is a regular Python cell with a `# @loop` annotation. The body runs N times, with a **carry variable** threaded between iterations. Each iteration's state is stored as its own artifact, so you can inspect any intermediate step.

Use loop cells for iterative refinement (hill climbing, MCMC, training loops with checkpoints), simulations, and anything where you'd want to pause and inspect intermediate states, or fork a new run from a promising one.

### Minimal example

Two cells: a seed and a loop.

```python
# seed cell, initial carry state
state = {"x": 0.0, "best_score": float("inf"), "iter": 0}
```

```python
# loop cell
# @loop max_iter=40 carry=state
# @loop_until state["best_score"] < 1e-3
import random

# Each iteration: read `state`, compute the next step, rebind `state`.
candidate = state["x"] + random.uniform(-0.1, 0.1)
score = candidate ** 2   # some objective
if score < state["best_score"]:
    state = {**state, "x": candidate, "best_score": score, "iter": state["iter"] + 1}
else:
    state = {**state, "iter": state["iter"] + 1}
```

After execution, `state` holds the final iteration's value and every intermediate iteration is queryable.

### Required directives

| Directive            | What it does                                                                                                                                         |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `# @loop max_iter=N` | Hard cap on iterations. Required, the safety bound on the loop.                                                                                     |
| `# @loop carry=VAR`  | The variable threaded between iterations. Required. Must be re-bound by the cell body each iteration, and seeded by an upstream cell on iteration 0. |

These can be on the same line: `# @loop max_iter=40 carry=state`.

### Optional directives

| Directive                          | What it does                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| `# @loop_until <expr>`             | Early termination when `<expr>` is truthy (evaluated against the current `state`)   |
| `# @loop start_from=<cell>@iter=k` | Seed iteration 0 from a specific prior iteration's artifact, used for forking runs |

### Per-iteration artifacts

Every iteration's carry value becomes its own artifact with an `@iter=k` suffix:

```
strata://artifact/nb_..._cell_<loop_id>_var_state@v=1@iter=0
strata://artifact/nb_..._cell_<loop_id>_var_state@v=1@iter=1
...
```

The inspect panel shows an iteration picker so you can scrub through the intermediate states. The **final** iteration's artifact is also the cell's canonical output (no `@iter` suffix) downstream cells read it via the normal DAG path.

### Forking a loop

Intermediate iterations are first-class artifacts, so you can branch a new
run from any step of an old one without re-running the expensive prefix.

**Scenario.** You ran a hill-climbing search for 50 iterations. Glancing at
the inspect panel, iteration 17 looked like it was about to find a better
local optimum before the sampler drifted away. You want to explore what
happens if you push harder from that exact state with a different step size.

1. Open the loop cell's **Inspect** panel, scrub to iteration 17, copy its
   artifact URI. It'll look like
   `strata://artifact/nb_..._cell_hill_climb_var_state@v=1@iter=17`.
2. Add a new loop cell below. Reference the original cell's ID (not the full
   URI) in `start_from`:

   ```python
   # new loop cell, continues from iteration 17 of the previous run
   # @loop max_iter=20 carry=state start_from=hill_climb@iter=17
   state["step_size"] *= 0.5  # smaller steps from here on
   state = sample_and_score(state)
   ```

3. Run the new cell. It reads iteration 17's carry value as its seed, runs up
   to 20 more iterations under the modified strategy, and stores those
   iterations as its own artifact chain, the original run stays untouched.

You now have two parallel forks materialized in the artifact store. Either
one can be forked further, and the inspect panel shows both chains.

This is the escape hatch for "that intermediate state looked promising, let
me explore from there", the thing that's hard to do in a plain for-loop
once you've thrown away the intermediates.

### When not to use a loop cell

- Tight `for` loops over short collections, a regular Python cell with a `for` loop is simpler and the extra per-iteration artifact overhead isn't worth it.
- Loops where intermediate state is genuinely disposable, store only the final answer in a regular Python cell.
- Anything that needs to branch out into multiple parallel runs, loop cells are sequential by design. Use separate cells, or model the fan-out in Python.

Reach for loop cells when **being able to inspect or fork from iteration k matters**. That's the feature you're paying for.

---

## R Cells *(0.2.0)*

R is a first-class notebook language alongside Python. R cells run R source via the system `Rscript`, with the same provenance + caching + Arrow-IPC artifact pipeline as Python cells. Data crosses the language boundary as `arrow/ipc`: a `pandas.DataFrame` produced by a Python cell becomes a `data.frame` in the next R cell; an R `data.frame` (or `tibble`) becomes a `pandas.DataFrame` in the next Python cell. Non-tabular R values (S3 objects, lists with classes, environments) are stored as RDS and tagged `r_only=true` — a downstream Python cell that consumes one fails with a structured `StrataRArtifactError` rather than a confusing `NameError`.

### R environments

R environments are managed from the **Environment** panel, at parity with Python's uv-backed venv:

- **System R by default.** A notebook with no `renv.lock` runs R cells against your system R library. The R card reads **System R** and shows the detected R version — cells work immediately as long as `arrow` + `jsonlite` are available system-wide.
- **One-click renv bootstrap.** **Initialize renv** creates a project-scoped, reproducible environment: it installs `renv` if missing, inits a bare project library, installs `jsonlite` + `arrow`, and snapshots to `renv.lock`. Progress streams live on the card (the first run compiles `arrow` from source, which takes a few minutes). When it finishes the card flips to **In sync**.
- **Per-package install.** A missing-package error in an R cell surfaces a structured hint with an **Install** button that runs `renv::install()` + `renv::snapshot()` for the named package.
- **Automatic restore on open.** Opening a notebook that has an `renv.lock` restores the project library automatically — the `uv sync` analogue for R. Editing `renv.lock` invalidates R cells' cache the same way editing `uv.lock` invalidates Python cells'.

`renv.lock` is committed config (like `uv.lock`); the built `renv/library/` is gitignored.

### Plots

R cells display plots inline, like a Python cell's matplotlib figure. Base graphics (`plot()`, `hist()`, …) and grid-based plots (ggplot2, lattice) are captured to PNG and rendered in the cell output. A bare trailing plot object auto-renders — a last-line `p` where `p <- ggplot(...)` shows the plot without an explicit `print(p)`, mirroring the R console. A cell that draws several plots produces an ordered list of image outputs.

### What's still ahead

**Warm Rscript pool.** Notebooks containing R cells keep a small pool of
pre-spawned R workers (R startup, `.Rprofile`/renv activation, and
`jsonlite`/`arrow` loads already paid), so an R cell run skips the ~1–2s
interpreter cold-start. Workers are single-shot — one cell each, then
replaced — preserving per-cell isolation; editing `renv.lock` drains and
respawns the pool. Pure-Python notebooks and machines without `Rscript`
never start one.

R execution and display are complete; remaining R polish is tracked on GitHub: [#83](https://github.com/bearing-research/strata/issues/83) (R version matrix on CI), [#84](https://github.com/bearing-research/strata/issues/84) (cross-language run-all batching).

### What you need today

- R `>= 4.2` on `PATH` (Strata calls `Rscript`).
- The `arrow` and `jsonlite` R packages — either available in the system R library, or installed into a project library via **Initialize renv** in the Environment panel (which installs both for you).

### Example

See [`examples/r_lm_vs_sklearn/`](https://github.com/bearing-research/strata/tree/main/examples/r_lm_vs_sklearn) for a working notebook: a Python cell synthesises a housing dataset, an R cell fits `lm(price ~ sqft + bedrooms + age + location, data = housing_train)` and returns tidy coefficient + model-stats + prediction `data.frame`s, a Python cell fits the same model with `LinearRegression` and prints a side-by-side comparison. The R formula's auto-factor-encoding of `location` is the one-line stats-edge moment.

### R-cell annotations

The same annotation parser handles both Python and R cells (`#`-prefixed comments at the top of the source). Supported on R cells:

| Annotation                 | Effect                                                                  |
| -------------------------- | ----------------------------------------------------------------------- |
| `# @name <text>`           | Cell display name in the UI.                                            |
| `# @env KEY=value`         | Sets `Sys.getenv("KEY")` for the cell's process.                        |
| `# @mount data file:///x`  | Binds `data` inside the R cell to the mount's local path (a character string — R has no `pathlib.Path`).  |
| `# @timeout 60`            | Per-cell execution timeout in seconds.                                  |

Loop annotations (`@loop`, `@loop_until`) and prompt-cell annotations (`@output_schema` etc.) do not apply to R cells.

### What if Rscript isn't installed?

An R cell whose harness can't find `Rscript` on `PATH` returns a clean cell-level failure with the message `Rscript not found on PATH. Install R (https://cran.r-project.org/) and reopen the notebook.` No server crash, no other cells affected.

---

## Choosing between kinds

| Reach for a… | When you want…                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| Python cell  | Ordinary computation. Default.                                                                         |
| Prompt cell  | An AI response as a first-class, cached, DAG-participating artifact.                                   |
| SQL cell     | A query against a connected database, with bind parameters, schema discovery, and probe-based caching. |
| Loop cell    | Iterative refinement where pausing or forking from an intermediate state matters.                      |
| R cell       | A computation where R's stats / formula syntax / domain packages are the right tool, while keeping the rest of the pipeline in Python. |

Mixing is encouraged, a typical pipeline might be a SQL cell for extraction → Python cells for transformation → an R cell for the linear model → a prompt cell for narrative summarisation.

Any kind of cell can also live inside a [variant group](annotations.md#variant-cells) a tabbed slot where multiple cells share one place in the DAG and you switch between them without forking the notebook.
