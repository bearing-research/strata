# Cell Types

Strata Notebook has four cell kinds:

| Kind       | What it runs                            | Created by                                                    |
| ---------- | --------------------------------------- | ------------------------------------------------------------- |
| **Python** | Python source in the notebook's venv    | The default — pick **Python** from the **+ Add cell** menu    |
| **Prompt** | A text template sent to an LLM          | Pick **Prompt** from the **+ Add cell** menu                  |
| **SQL**    | A query against a connected database    | Pick **SQL** from the **+ Add cell** menu                     |
| **Loop**   | A Python cell executed N times in a row | Add a Python cell, then put a `# @loop` annotation at the top |

All four participate in the DAG, cache by provenance hash, and can be routed to remote workers. Pick the kind that matches the shape of the computation — this page walks through each.

See [Concepts](concepts.md) for the execution model; see [Cell Annotations](annotations.md) for the full per-annotation reference.

---

## Python Cells

The default. A Python cell is just Python source — assignments at module scope become the cell's outputs, and free variables become inputs pulled from upstream cells.

### Writing a Python cell

```python
import pandas as pd

sales = pd.read_parquet("https://example.com/sales.parquet")
by_region = sales.groupby("region")["total"].sum()
```

This cell _defines_ `sales` and `by_region`. A downstream cell that references either name will automatically depend on this one.

```python
# downstream cell — reads by_region from upstream
top_region = by_region.idxmax()
print(f"Top region: {top_region}")
```

### Variable flow and the DAG

Strata analyzes each cell's AST to extract:

- **Defines** — top-level assignments (`x = 1`, `df = pd.read_csv(...)`)
- **References** — free variables used but not defined locally

The DAG builder links references back to the **last** cell that defined each name (shadowing is handled by order). Edges flow producer → consumer. When you edit an upstream cell, every downstream cell that depends on it becomes stale automatically.

Only variables that a downstream cell actually references get stored as artifacts. Intermediate scratch variables stay in the subprocess and are discarded when the cell finishes.

### Library cells (cross-cell defs and classes)

Top-level `def` and `class` definitions are shared across cells via a synthetic Python module — write a helper once, call it anywhere.

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

The slice keeps the module docstring, `import` / `from import` (no `from X import *`), `def` / `async def`, `class`, and assignments whose RHS is a **literal constant** — numbers, strings, bools, `None`, bytes, negations of literals, and nested tuples/lists/sets/dicts of literals. Everything else (non-literal assignments, augmented assigns, expression statements, control flow, bare annotations) is dropped from the slice but stays in the cell's runtime execution.

A single cell can therefore mix runtime work and library code:

```python
# Runtime — dropped from the slice; flows through the artifact path.
raw_min = round(-math.tau * 7, 2)
raw_max = round(math.tau * 16, 2)
print(f"loaded raw bounds: [{raw_min}, {raw_max}]")

# Library — kept in the slice, exported as a synthetic module.
CLAMP_MIN = 0.0
CLAMP_MAX = 100.0

def clamp(value):
    return max(CLAMP_MIN, min(CLAMP_MAX, value))
```

A downstream cell can call `clamp(raw_max)` — `clamp` and `CLAMP_MIN/MAX` come from the synthetic module, `raw_max` from the artifact path.

#### When the slice isn't self-contained

Every name a kept def or class references must be bound by something else in the slice (or a Python builtin). When it isn't, Strata blocks the export with a `module_export_blocked` diagnostic — surfaced pre-flight, not just at run time.

```python
runtime_threshold = math.sqrt(9)   # dropped — non-literal RHS

def is_outlier(value):
    return value > runtime_threshold
```

> `is_outlier` references names not defined or imported in this cell: runtime_threshold

Other shapes that block on the same principle:

- **Decorators / default values / base classes** evaluated at module load: `@my_decorator` where `my_decorator` isn't imported in the same cell, or `class Child(Parent)` where `Parent` is computed at runtime.
- **Divergence**: a name kept by the slice is also reassigned by dropped runtime code, so the synthetic module's value would differ from the cell's final state. `def f(): ...; f = wrap(f)` exports the unwrapped `f`.
- **Lambda assignments**: `add = lambda x: x + 1` — even though `cloudpickle` could serialize the value, the synthetic-module path is reserved for source-backed library code.

The fix is usually one of: move the runtime line into its own cell, add the missing import to the same cell as the def, or take the dependency as a function argument.

#### Single-cell scope

The synthetic module is built from one cell's source only — no transitive composition across cells. A def can't reach a name imported or defined in a different cell; each cell that hosts library code carries its own imports.

One concession: annotations that reference names outside the slice would normally block, but adding `from __future__ import annotations` relaxes this. PEP 563 stringifies annotations and the free-variable check drops them, so cross-cell type hints "just work" with the future import.

Walked through end-to-end in the [`library_cells`](../../examples/library_cells) example notebook.

### Mutation warnings

If a cell mutates a value it received from an upstream cell (e.g. `df.drop(columns=[...], inplace=True)`), Strata raises a **mutation warning** — the upstream artifact was supposed to be immutable, and subsequent cells that reuse the cached artifact will see the mutated version.

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
| `# @timeout 60`    | Override execution timeout (seconds, default 30) |
| `# @env KEY=value` | Set an env var for this cell only                |
| `# @mount …`       | Attach a filesystem mount (see [Annotations][a]) |
| `# @loop …`        | Turn the cell into a [loop cell](#loop-cells)    |

See [Cell Annotations][a] for the full reference.

[a]: annotations.md

---

## Prompt Cells

A prompt cell is a text template that gets rendered with upstream variable values, sent to an LLM, and the response stored as an artifact. Prompt cells participate in the DAG and cache by provenance exactly like Python cells — same inputs + same template + same model config = cache hit, no LLM call.

Create a prompt cell with the **"Add Prompt Cell"** button in the UI — the same toolbar that adds a Python cell. You never need to touch `notebook.toml` directly; editing the cell's source, wiring it into the DAG, and persisting the result all happen through the UI.

### Basic syntax

```
# @name summary
Summarize this dataset and return the top 3 findings as a numbered list:

{{ df }}
```

- `{{ df }}` is replaced with a text representation of the upstream variable `df` before sending to the LLM.
- The LLM's response is stored as an artifact named `summary` (from `# @name`).
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
{{ df.describe() }}     # OK — pandas describe() is allow-listed
{{ df.head() }}         # OK
{{ obj.attr }}          # OK — attribute access (non-callable)
{{ obj.mutate() }}      # blocked — unknown method, left as-is in the template
```

Only a small set of methods is permitted (`describe`, `head`, `tail` on pandas objects). Arbitrary method calls are blocked to keep template rendering side-effect-free.

### Prompt-cell annotations

| Annotation               | What it does                                                              | Default              |
| ------------------------ | ------------------------------------------------------------------------- | -------------------- |
| `# @name <identifier>`   | Output variable name; must be a Python identifier                         | `result`             |
| `# @model <model_id>`    | Override the notebook-level LLM model                                     | From provider config |
| `# @temperature <float>` | Sampling temperature (0.0 = deterministic; see [Caching](#caching) below) | `0.0`                |
| `# @max_tokens <int>`    | Response token ceiling                                                    | `4096`               |
| `# @system <text>`       | System prompt prepended to the request                                    | None                 |
| `# @output json\|text`   | Coerce the response to JSON (or keep as free-form text)                   | `text`               |
| `# @output_schema {…}`   | Inline JSON Schema pinning the response shape                             | None                 |
| `# @validate_retries N`  | Total attempts for the validate-and-retry loop (1 initial + N−1 retries)  | `3`                  |

Example using several at once:

```
# @name classification
# @model gpt-4o
# @temperature 0.0
# @max_tokens 1000
# @system You are a data scientist. Return only valid JSON.
Classify each paper by topic:

{{ sampled_papers }}

Return a JSON object mapping paper ID to topic.
```

### Schema-constrained output

`# @output_schema {...}` pins the shape of the LLM response to an inline JSON Schema. Strata picks the best provider-native path:

| Provider                             | Enforcement                                                                                                                                                                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **OpenAI**                           | Native `response_format: {type: "json_schema"}`. `additionalProperties: false` is auto-injected at every `object` node; strict mode is used when the user's `required` list covers every property (otherwise relaxed to `strict: false`). |
| **Anthropic**                        | Native `/v1/messages` with tool-use: the schema is sent as a tool's `input_schema` and `tool_choice` is forced to that tool. The returned `tool_use.input` is extracted verbatim.                                                         |
| **Gemini / Mistral / Ollama / vLLM** | Fallback to `response_format: {type: "json_object"}` — valid JSON guaranteed, shape not enforced server-side. Client-side validation (see below) fills the gap.                                                                           |

Setting `@output_schema` implies `@output json`; you don't need both.

Example — triage each review into a structured record:

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

### Validate-and-retry

When `@output_schema` is set, Strata runs a **validate-and-retry loop** after every LLM call:

1. Parse the response as JSON and run it through `jsonschema`.
2. On success → store the artifact and return.
3. On failure → append the bad response as an `assistant` turn, feed the validator's path-addressed errors back as a `user` turn, and retry.
4. On retry exhaustion → surface a cell error with the last validator messages.

The default is 3 total attempts (1 initial + 2 retries). Override with `# @validate_retries N`. Cumulative input/output tokens across all attempts are recorded on the artifact so cost accounting is accurate. The retry count is surfaced on the cell result (`validation_retries`) — the UI shows "validated after N retries" when non-zero.

Retries are mostly invisible on OpenAI-strict and Anthropic-native paths because the provider enforces the schema at decode time. They earn their keep on the `json_object` fallback path (Gemini, Mistral, Ollama) where the provider only guarantees _syntactic_ JSON.

### Caching

A prompt cell's provenance hash mixes together:

- The rendered template text (after `{{ var }}` injection)
- Model name
- Temperature
- System prompt
- Output type (`json` / `text`)
- Output schema fingerprint (when set)

Editing any of these invalidates the cache. In particular, tweaking `@output_schema` on a cached cell forces a fresh call — exactly what you want when iterating on the response shape.

!!! tip "Keep temperature at 0.0 for prompt cells"
With `temperature=0.0` the model is deterministic: same inputs → same output, and cache behavior is intuitive. Bumping temperature makes the first response "sticky" in the cache — future runs return the stored stochastic sample rather than re-sampling.

See [LLM Integration](llm.md) for provider configuration and the conversational AI assistant.

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
- **Currently shipped drivers**: SQLite, PostgreSQL, Snowflake, and BigQuery. All are ADBC-backed (`adbc-driver-sqlite`, `adbc-driver-postgresql`, `adbc-driver-snowflake`, `adbc-driver-bigquery`). For Snowflake, read cells use `role`; `write=true` cells switch to `write_role` when configured, otherwise they reuse `role`. For BigQuery, read cells use `credentials_path`; `write=true` cells switch to `write_credentials_path` when configured, otherwise they reuse `credentials_path`. Snowflake and BigQuery do not have a session-level read-only flag like PostgreSQL's `SET default_transaction_read_only = on`, so the safety boundary is the grants on the configured role or service account.

### Schema discovery

The **Schema panel** in the sidebar shows the tables and columns visible through each declared connection. Click a connection to lazy-load its schema; click a table to expand its columns. The `↻` button re-fetches when the underlying database has changed externally. No SQL cell needs to run for this — the panel talks directly to each driver's catalog query surface (`sqlite_master` for SQLite, `information_schema.tables JOIN columns` for PostgreSQL, and the driver-specific catalog queries for Snowflake and BigQuery).

### Bind parameters

`:name` placeholders resolve against upstream cell variables. Strata coerces a strict allowlist of Python types (`int`, `float`, `str`, `bytes`, `bool`, `None`, `Decimal`, `UUID`, `datetime`/`date`/`time`) into ADBC bind values; anything else (a list, a numpy scalar, a custom object) is rejected with a clear error. **No string substitution ever** — values flow through ADBC's prepared-statement layer, so adversarial strings (`'; DROP TABLE …`) round-trip as data, not SQL.

```python
# upstream Python cell
min_amount = 100
```

```sql
# @sql connection=warehouse
SELECT * FROM orders WHERE amount > :min_amount
```

The DAG links the SQL cell to the Python cell automatically — same edge logic Strata uses for Python free variables.

### Cache policies

A SQL cell's **provenance hash** folds together:

- The query text (sqlglot-normalized so whitespace and comment edits don't churn the cache).
- The bind parameters (type-tagged: `True` ≠ `1`).
- The connection's identity (host / DB / user / role / search_path — never the password).
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
| SQLite     | `PRAGMA data_version` + `PRAGMA schema_version` | **DB-wide** | DML cross-process needs the probe connection open across the write — `data_version` resets on a fresh connection. DDL (schema change) invalidates cleanly.                                                                                                                                                                                                   |
| Snowflake  | `INFORMATION_SCHEMA.TABLES.LAST_ALTERED`        | per-table   | Per-database scoping (one query per touched database). Bills cloud-services credits but each query is small. `LAST_ALTERED` updates even on 0-row DML — safe direction (over-invalidates, never under).                                                                                                                                                      |
| BigQuery   | `__TABLES__.last_modified_time`                 | per-table   | Per-dataset scoping. `__TABLES__` is the legacy-but-stable view; `INFORMATION_SCHEMA.TABLES` doesn't expose `last_modified_time`. **Streaming-buffer caveat**: tables receiving streaming inserts have `last_modified_time` lag by minutes-to-90-min until the buffer flushes — pin `# @cache session` on those queries. Permissions: `bigquery.tables.get`. |

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
- `# @cache fingerprint` and `# @cache snapshot` error early on write cells — probe-based invalidation has no anchor when the cell mutates state.
- The cell still produces an Arrow artifact: a per-statement status table with `stmt`, `kind` (`CREATE TABLE`, `INSERT`, …), and `rows_affected` (nullable; `null` for DDL).
- Read cells using the same connection stay on the read path — the override is per-cell.

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

`# @after seed` adds a DAG edge from the `seed` cell to this one even though no Python variable flows between them — the dependency is on a side effect (the SQLite file). This is what cascade execution and staleness recompute use to ensure the right ordering.

### Worked example

The [`sql_orders_report`](../../examples/sql_orders_report) example notebook walks through all of this end-to-end: a SQL `seed` cell, a Python `threshold` cell, two parameterized SQL queries, and a Python report cell — five cells, two languages, with both `fingerprint` and `forever` cache policies side by side.

### SQL-cell annotations

| Annotation                              | What it does                                          |
| --------------------------------------- | ----------------------------------------------------- |
| `# @sql connection=<name> [write=true]` | Mark the cell as SQL; reference a declared connection |
| `# @cache <policy>`                     | Override the default `fingerprint` cache policy       |
| `# @name <identifier>`                  | Name the output variable (default: `result`)          |
| `# @after <cell-id>`                    | Add an ordering-only DAG edge to an upstream cell     |

See [Cell Annotations][a] for the full reference.

---

## Loop Cells

A loop cell is a regular Python cell with a `# @loop` annotation. The body runs N times, with a **carry variable** threaded between iterations. Each iteration's state is stored as its own artifact, so you can inspect any intermediate step.

Use loop cells for iterative refinement (hill climbing, MCMC, training loops with checkpoints), simulations, and anything where you'd want to pause and inspect intermediate states — or fork a new run from a promising one.

### Minimal example

Two cells: a seed and a loop.

```python
# seed cell — initial carry state
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
| `# @loop max_iter=N` | Hard cap on iterations. Required — the safety bound on the loop.                                                                                     |
| `# @loop carry=VAR`  | The variable threaded between iterations. Required. Must be re-bound by the cell body each iteration, and seeded by an upstream cell on iteration 0. |

These can be on the same line: `# @loop max_iter=40 carry=state`.

### Optional directives

| Directive                          | What it does                                                                        |
| ---------------------------------- | ----------------------------------------------------------------------------------- |
| `# @loop_until <expr>`             | Early termination when `<expr>` is truthy (evaluated against the current `state`)   |
| `# @loop start_from=<cell>@iter=k` | Seed iteration 0 from a specific prior iteration's artifact — used for forking runs |

### Per-iteration artifacts

Every iteration's carry value becomes its own artifact with an `@iter=k` suffix:

```
strata://artifact/nb_..._cell_<loop_id>_var_state@v=1@iter=0
strata://artifact/nb_..._cell_<loop_id>_var_state@v=1@iter=1
...
```

The inspect panel shows an iteration picker so you can scrub through the intermediate states. The **final** iteration's artifact is also the cell's canonical output (no `@iter` suffix) — downstream cells read it via the normal DAG path.

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
   # new loop cell — continues from iteration 17 of the previous run
   # @loop max_iter=20 carry=state start_from=hill_climb@iter=17
   state["step_size"] *= 0.5  # smaller steps from here on
   state = sample_and_score(state)
   ```

3. Run the new cell. It reads iteration 17's carry value as its seed, runs up
   to 20 more iterations under the modified strategy, and stores those
   iterations as its own artifact chain — the original run stays untouched.

You now have two parallel forks materialized in the artifact store. Either
one can be forked further, and the inspect panel shows both chains.

This is the escape hatch for "that intermediate state looked promising, let
me explore from there" — the thing that's hard to do in a plain for-loop
once you've thrown away the intermediates.

### When not to use a loop cell

- Tight `for` loops over short collections — a regular Python cell with a `for` loop is simpler and the extra per-iteration artifact overhead isn't worth it.
- Loops where intermediate state is genuinely disposable — store only the final answer in a regular Python cell.
- Anything that needs to branch out into multiple parallel runs — loop cells are sequential by design. Use separate cells, or model the fan-out in Python.

Reach for loop cells when **being able to inspect or fork from iteration k matters**. That's the feature you're paying for.

---

## Choosing between kinds

| Reach for a… | When you want…                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------ |
| Python cell  | Ordinary computation. Default.                                                                         |
| Prompt cell  | An LLM response as a first-class, cached, DAG-participating artifact.                                  |
| SQL cell     | A query against a connected database, with bind parameters, schema discovery, and probe-based caching. |
| Loop cell    | Iterative refinement where pausing or forking from an intermediate state matters.                      |

Mixing is encouraged — a typical pipeline might be a SQL cell for extraction → Python cells for transformation → a prompt cell for narrative summarization.

Any kind of cell can also live inside a [variant group](annotations.md#variant-cells) — a tabbed slot where multiple cells share one place in the DAG and you switch between them without forking the notebook.
