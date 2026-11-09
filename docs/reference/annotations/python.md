# Python annotations

These apply to Python cells, and most to other compute cells too.

## @name

Set a human-readable display name for the cell. Shown in the DAG view and as a badge in the cell editor.

```python
# @name Load arXiv Papers
import pandas as pd

papers = pd.read_parquet("https://...")
```

For **Python cells**, any non-empty string is accepted: spaces, parentheses, and special characters are fine.

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
    the cell's `id` in `notebook.toml`, which is a separate field -
    see [Cell IDs](#cell-ids) below.

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
    { id = "seed",       file = "seed_database.sql", language = "sql",    order = 0 },
    { id = "threshold",  file = "threshold.py",      language = "python", order = 1 },
    { id = "top-orders", file = "top_orders.sql",    language = "sql",    order = 2 },
]
```

…then in `cells/top_orders.sql`:

```sql
-- @sql connection=warehouse
-- @after seed
SELECT category, COUNT(*) FROM products GROUP BY category
```

The `@after seed` resolves to the cell with `id = "seed"`. There is no
UI for renaming a cell ID; a friendly ID is set by editing
`notebook.toml`.

`@name` and `id` are independent: a cell can have `id = "seed"` and
`@name "Seed database"` simultaneously. The DAG view shows the
`@name`; `@after` and `@loop start_from=` use the `id`.

## @worker

Route the cell's execution to a named worker instead of the local machine.

```python
# @worker df-cluster
category_stats = ctx.sql("SELECT topic, COUNT(*) FROM papers GROUP BY topic").to_pandas()
```

Workers are HTTP endpoints that implement the Strata executor protocol. Register them via the **Workers panel** in the sidebar; the persisted result lands in `notebook.toml` as:

```toml
[[workers]]
name = "df-cluster"
backend = "executor"
runtime_id = "df-cluster"

[workers.config]
url = "https://my-datafusion-worker.fly.dev/v1/execute"
transport = "direct"
```

A server may hide the Workers panel (`STRATA_NOTEBOOK_WORKERS_PANEL=hidden`) or run remote cells only on its own registry's workers, ignoring a notebook's `[[workers]]` (`STRATA_NOTEBOOK_WORKERS_EDITABLE=false`); see [Server-managed workers](../../notebook/workers.md#server-managed-workers).

During execution, the UI shows a pulsing "dispatching → df-cluster" badge on the cell. After completion, the worker name appears in the cell's metadata.

Workers can be anything that speaks HTTP: a GPU box on RunPod, a DataFusion cluster on Fly, a beefy EC2 instance, or a local process on a different port. The built-in `remote_executor.py` provides a reference implementation:

```bash
strata-worker --host 127.0.0.1 --port 9000
```

If no `@worker` is set, the cell runs locally in the notebook's Python environment.

## @timeout

Override the execution timeout for a single cell, in seconds. The default
is 300 seconds (5 minutes); the value must be a positive number.

```python
# @timeout 300
# @worker my-gpu
embeddings = model.encode(abstracts, batch_size=256)
```

Useful for cells that download data, train models, or call slow external APIs. The timeout applies to the full execution including any remote worker round-trip.

`@timeout` does not apply to prompt or SQL cells. A prompt cell's model call
is bounded by `[ai] timeout_seconds` in `notebook.toml`, or the server's
`STRATA_AI_TIMEOUT_SECONDS` (default 60 s); a SQL cell's deadline is set at
the connection.

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
    `# @env` values live in **committed cell source** (the files under `cells/`).
    The sensitive-key blanking that the notebook.toml `[env]` writer
    applies (KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL values blanked
    before commit) does **not** apply to `# @env` - the literal value
    you type goes straight to git.

    For API keys and other secrets, use the notebook's `[env]` block
    in `notebook.toml` (blanked automatically) or the Runtime panel
    in the UI (kept in the server process only). See
    [notebook.toml `[env]`](../notebook-toml.md#env-notebook-environment-variables).
    `# @env` is for non-sensitive overrides like `CUDA_VISIBLE_DEVICES`
    or `OMP_NUM_THREADS`.

## @mount

Attach a filesystem mount to the cell. Mounts provide read or read-write access to external storage (local paths, S3, GCS, Azure) during execution.

```python
# @mount raw_data s3://my-bucket/dataset ro
# @mount scratch file:///tmp/work rw
df = pd.read_parquet(raw_data / "events.parquet")
scratch / "summary.txt"  # → Path("/tmp/work/summary.txt")
```

The full guide: [Mounts](../../guides/data/mounts.md).

## @fetch

Declare a URL the cell reads. The bytes are downloaded into the notebook's
cache and **their digest becomes part of the cell's provenance**, so the cell
and its downstream cells go stale when the URL starts
serving something else - instead of the URL
string being in the source hash and the bytes being in nothing.

```python
# @fetch zones https://example.org/taxi_zones.csv
import pandas as pd

lookup = pd.read_csv(zones)
```

The full guide: [Fetch and datasets](../../guides/data/fetch-and-datasets.md).

## @dataset

Declare a registry name the cell reads. The name resolves to one artifact
version, and **that version becomes part of the cell's provenance**, so when
`champion` moves, a cell that reads the team's champion model and its downstream
cells go stale. Resolving
the name with the `strata` client inside the cell records nothing: the name is
in the source hash and the version is in nothing.

```python
# @dataset model taxi/model@champion
predictions = model.predict(features)
```

The full guide: [Fetch and datasets](../../guides/data/fetch-and-datasets.md).

## @table

Declare an Iceberg table input. The table's current snapshot id becomes part
of the cell's provenance: **when new data lands in the table, the cell and its
downstream cells go stale, and the normal cascade
machinery re-runs them** - no manual data-version
bookkeeping. For an end-to-end walkthrough (build a warehouse, scan it,
retrain on new data, pin a snapshot), see
[Lake and Iceberg](../../notebook/lake-aware-cells.md).

```python
# @table trips file:///data/warehouse#nyc.trips
art = strata.materialize(
    inputs=[trips],
    transform={"executor": "scan@v1", "params": {"snapshot_id": trips_snapshot}},
)
```

**Two variables are injected into the cell's namespace**: `<name>` - the table
URI string - and `<name>_snapshot` - the snapshot id resolved when the cell's
provenance was computed. Passing `<name>_snapshot` to the scan makes the cell
fully deterministic: it reads exactly the snapshot its provenance recorded.
For a table with no snapshots yet, `<name>_snapshot` is `None` (a scan reads
it as zero rows) and the cell is never served from the cache, so it sees the
table's first write.

Format: `# @table <name> <uri> [snapshot=<id>]`. The URI is one of the forms
`client.materialize` accepts:

- `<catalog>:<namespace>.<table>`, a table in a catalog the server names under
  `STRATA_CATALOGS` (REST, Glue, SQL or any other PyIceberg catalog type), e.g.
  `# @table orders lake:sales.orders`;
- `<warehouse>#<namespace>.<table>`, a table in a SQL catalog kept in that
  warehouse; a warehouse starting `./` or `../` is a folder from the notebook's
  directory, as in `# @table orders ./warehouse#sales.orders`;
- `<namespace>.<table>`, a table in the server's default catalog.

The warehouse may be local or on S3, GCS (`gs://`) or Azure (`abfs://`,
`abfss://`; pyiceberg has no `az://` catalog support); files on GCS and Azure are read with the same
`STRATA_GCS_*` and `STRATA_AZURE_*` settings the artifact blob store uses. An
Azure table's metadata is read with `adlfs`, or with PyArrow only when the table's
locations have no `@<account>` host
([Azure Storage](../configuration.md#azure-storage)). The name must be a
valid Python identifier.

A warehouse in the notebook's folder suits a notebook that builds its own tables: a
`@write_table` creates it on first write, and a fresh clone builds its own. Add it
to the notebook's `.gitignore`: its catalog records where its files are as absolute
paths, so it does not survive being moved or cloned. Delete it and rerun the cells
that write it (`strata run --force`): they stay cached otherwise, and a cached cell
writes nothing.

Catalogs are named on the server, not in `notebook.toml`, because the scan the
cell runs happens in the server and has to resolve the same name. (A DuckDB
connection, which attaches its catalog itself, can also name one the notebook
defines on a personal server; see
[notebook-level catalogs](../notebook-toml.md#notebook-catalogs).)

`snapshot=<id>` pins the table: the cell reads that snapshot forever and never
goes out of date on new data (the lake-side analog of a mount `pin`). Without a pin,
the snapshot is re-resolved every time staleness is evaluated.

Like mount variables, the injected names live only in the declaring cell's
namespace - they are not cell *defines* and do not flow to downstream cells.
To use the snapshot id downstream, export it as a real variable:
`scanned_snapshot = trips_snapshot`. A `@loop` cell gets them in every
iteration. An R cell cannot declare `@table`; read the table in an upstream
Python cell.

If the catalog is unreachable when provenance is computed, the cell reads
idle, since nothing says its result is wrong; if it is still unreachable at execution time,
the run fails with a clear error.

## @write_table

Write one of the cell's variables into an Iceberg table when the run succeeds:
a cleaning cell that replaces a table, an ingest cell that appends each day's
data, a cell that merges corrections by key, a cell whose table should hold
exactly what it produces.

```python
# @write_table readings lake:lab.readings append
readings = load_today()

# @write_table fixes lake:lab.readings merge on=sample_id
fixes = corrections()

# @write_table current lake:lab.current sync on=sample_id
current = latest_per_sample()
```

The full guide: [Write back](../../guides/data/write-back.md).

## @output_dir

Give the cell an empty folder whose contents are an output: checkpoints, a saved
model, a set of figures.

```python
# @output_dir ckpt
for epoch in range(3):
    train(model)
    torch.save(model.state_dict(), ckpt / f"epoch{epoch}.pt")
```

Format: `# @output_dir <name>`, repeatable. `<name>` is a `pathlib.Path` to an
empty, writable folder in the run's directory; everything in it when the run
ends is the output. Each file is stored once, by its sha256, in the notebook's
store (a file that did not change between runs is not stored again), and the
output itself is a manifest of paths, sizes and digests with the run's
provenance and inputs, so lineage and cache hits work as for any variable. The
folder is kept even when no cell reads it.

A cell that references `<name>` gets the folder rebuilt as a local `Path`: a
copy, so writing to it changes nothing upstream.

**The cell shows what the folder holds**: its file count, total size, and a
table of paths and sizes (the first 20 files, then a count of the rest), ahead
of the cell's own output. It is a markdown display like any other, so a cache
hit, a reopened notebook and an export show it too.

![A cell with # @output_dir ckpt that saves three checkpoint files and a metrics file. Below its source: "ckpt: 4 files, 3.0 MiB" and a table of each file with its size.](../../assets/output-dir-light.png#only-light)
![A cell with # @output_dir ckpt that saves three checkpoint files and a metrics file. Below its source: "ckpt: 4 files, 3.0 MiB" and a table of each file with its size.](../../assets/output-dir-dark.png#only-dark)

**A failed run keeps what it wrote.** Its files are stored and its manifest is
kept as a failed version: never a cache hit and never an input, but lineage
shows it, and its files are listed and downloaded over
[REST](../rest-api.md#read-a-cells-output-folder) like any
version's. A link in the folder, or a file with more than one hard link, is
left out with a warning.

**Files go with the last version that names them.** Folder versions are pruned
like any cell output's earlier values
([`STRATA_NOTEBOOK_KEEP_SUPERSEDED_VERSIONS`](../configuration.md#notebook)),
except that a failed run's version stays until a later run of the cell stores
a new successful version, and at least as long as the successful version it
follows, so a crash's checkpoints can still be downloaded. A cache hit (the
source put back to an earlier good version) stores nothing new, so the failed
version stays.
A file no kept version names is removed in the same pass, so old checkpoints
do not pile up. Pin a folder version to keep its files: a store's garbage
collection keeps every file version a published, named or pinned folder names.

**On a remote worker** the folder comes back with the run's result, each file
streamed, so a checkpoint is not held in memory and is not limited by
`STRATA_NOTEBOOK_MAX_BUNDLE_MEMBER_BYTES`. A failed run on a worker keeps its
files the same way. A worker too old to return folders is refused before the
cell runs.

**On the `signed` transport each file reaches the server while the cell runs**,
once its size and mtime stop changing (checked every
`STRATA_WORKER_OUTPUT_DIR_POLL_SECONDS`, 5 by default), and again if it changes.
The server stores it at once, so a worker that is lost, or a run that times out,
still leaves every file that arrived, kept as a failed version. A training loop
that writes one checkpoint per epoch keeps every finished epoch. The result then
names those files instead of carrying them, and a file the cell deleted before it
ended is not part of the folder. On the `direct` transport the worker cannot
reach the server, so files come back only with the result.

**A cell on a remote worker can read another cell's folder.** The server sends
the folder's files as one tar input, and the worker's harness unpacks it to a
local `Path` with the same name it has on the server before the cell runs. A
worker too old to take one is refused before the cell runs.

**In a `# @loop` cell the folder lasts the whole loop**, and each iteration's
contents are stored as its `@iter=k` version ([loop cells](../../guides/cells/loops-and-variants.md#output-folders)).

**A [`# @per_variant`](../../guides/cells/loops-and-variants.md#fan-out-run-a-downstream-cell-once-per-variant) cell
keeps a folder per variant**, a failed variant's too, and a plain downstream
cell reads them as a `{variant: Path}` dict. Each variant's folder is listed and
downloaded over REST under its name.

**A folder leaves the notebook with its files.** Promoting or publishing it, or
a result made from it (the notebook's promote route, `strata artifact promote`,
`strata artifact publish` with or without `--to`, and the MCP tools), copies each file the
folder's version names before its manifest; a file the other store already
holds is not sent again. When a file lands under another version or id there
(another tenant held its id), the copied manifest names where it landed. Read
from that store as [`@dataset`](#dataset), the folder arrives as a `Path` with
its files; `strata artifact pull <id>@v=<n> --to <dir>` on that store's server
rebuilds them in a new directory, and from anywhere else strata-client's
`client.download_folder(<id>, <n>, <dir>)` does, over HTTP, checking each
file's size and sha256. A [snapshot](../../notebook/export.md#snapshots) carries the file versions each
manifest names, and a [publication's archive](../../notebook/publishing.md#archiving-the-copy-that-needs-no-server)
holds the files under `folder/`.

The [team cache](../../guides/share/team-store.md#the-team-cache-sharing-results-nobody-named)
offers and serves a folder with its files: a teammate's pull lands every file
under their notebook's own ids before the manifest, or nothing.

For now run-all runs a cell with a folder on its own rather than in a batch.
An R cell gets the folder as a path string and its folder is kept, but cannot
read another cell's folder yet.

## @each

Runs a cell once per file or row, and caches each item on its own: change one file
and one instance runs.

```python
# @mount raw s3://lab/incoming ro
# @each path in raw glob=*.csv
import pandas as pd

frame = pd.read_csv(path)
summary = frame.describe()
```

Format: `# @each <name> in <source> [key=<col>] [glob=<pattern>]`. `<name>` is bound
to one item in each instance. What `<source>` is decides what an item is:

| Source                      | One item per | Named by         | Runs again when         |
| --------------------------- | ------------ | ---------------- | ----------------------- |
| a mount, with an optional `glob=` | file   | its relative path | its content changes (a remote file: its listed version) |
| a dict from another cell    | entry        | its key          | its value changes       |
| a list from another cell    | value        | the value itself | an edit is a new item   |
| a table from another cell, with `key=` | row | the `key=` column | the row changes       |

With `glob=`, a remote mount downloads only the files the pattern names, and the
input weights and the large-read prompt count only those.
The [`each_pipeline`](../../examples/each_pipeline.md) example runs over 287 files from a
public bucket and syncs a table in a REST catalog.

A file is bound as a `pathlib.Path`, and a link to a file counts as a file. A mount
that is itself one file is refused. A row is bound as a dict of JSON-safe values, with
a null as `None`. A list
item is named by its content, so inserting one does not run the ones after it, and a
repeated value runs once. A table's key must not be null or repeat. The collection
has to be a dict, a list or a table as stored: a pickled one is refused, because the
server does not unpickle.

**What a cell downstream sees.** Each output variable reads as a `{item: value}` dict
of the items now in the source:

```python
# In a later cell: {"a.csv": <DataFrame>, "b.csv": <DataFrame>}
report = {name: s.loc["mean"] for name, s in summary.items()}
```

An item removed from the source leaves the dict. When every instance has succeeded,
the cell stores one manifest per output, naming the value each item produced; a run
with a failed item stores none, so downstream cells keep reading the last good one,
and the items that succeeded stay cached for the next run.

**What runs again.** Nothing changed is one lookup of the cell's key: no item is
listed. A new or changed item runs alone. Any other input of the cell, or its source,
runs every item. Going back to earlier inputs finds the values already stored. Rerun
(↻) runs every item.

**What the cell shows.** Each item's displays (a figure, `display(...)`, a trailing
expression) are shown one after another in the order the items ran, each under its
item's key, as its console prefixes each item's lines with `[variant=<key>]`. An item
that ran shows what it displayed now, and a cached one what it displayed when it ran,
on a hit that lists nothing too. If any item fails, the cell shows no displays.

**Writing a table.** `# @write_table ... sync on=<col>` is the only mode an `@each`
cell takes. The table is written once per run, in one commit, over the rows of every
item, the cached ones too, so a removed file's rows are deleted though nothing ran. A
failed item writes nothing, since its missing rows would read as deletions. A source
with no items refuses the run rather than delete every row. `append`, `merge` and
`overwrite` are refused: one commit per item, or only the last item's rows.

**Limits.** Python cells on the local worker, read by Python cells. Instances run one at a time, each in
a process of its own. In the UI, `strata run` and MCP that is a warm process, started ahead
with the notebook's dependencies imported, when one is ready: about a tenth of a second
before the cell's own work, or about a fifth when no warm process is ready. Plan on that
for a first run of thousands of items. An `@each` cell cannot also be a `@loop`, `@per_variant` or
`@variant` cell, or fan out over a sweep. Cell tests, branching from a release,
`@dataset`, `strata artifact pull` and a publication bundle over its output are not
supported yet and are refused; a cell that reads the output can do the last three.
A snapshot carries the output and every item it names.

A prune keeps each instance a kept manifest names, and what it displayed, and drops
the rest, so a removed item's value goes once no manifest left names it. A copy to
another store carries the values, not the displays.

## @nocache

Force the cell to **always re-execute**: never serve a provenance cache hit.

```python
# @nocache
import urllib.request

latest = urllib.request.urlopen("https://api.example.com/now").read()
print(latest)
```

By default every cell is cached by provenance
(`sha256(inputs + source + env)`), including a leaf cell that only `print`s,
an unchanged rerun replays its output instantly instead of recomputing. That is
exactly what you want for deterministic work, and it makes the notebook a good
**scratchpad** (see [Driving a notebook with an agent](../../notebook/agent.md)). But it is
wrong for a cell whose *point* is a side effect or fresh value:

- writes a file, sends a request, mutates external state: replaying the cached
  result would **skip the effect**;
- reads the clock, `random`, a live API: you want a **fresh value** each run.

`# @nocache` opts such a cell out. `# @nocache off` re-enables caching (useful to
override a default you set elsewhere). It applies to every language (on a SQL
cell it wins over any `-- @cache` policy, `forever` included) and sits alongside
the read-write-`@mount` rule, which already forces re-execution for the same
reason.

A cell that reads a `@nocache` (or read-write-mount) cell's value is keyed on
the bytes that value stored rather than on the producer's provenance, which is
the same on every run. A new value re-runs the consumer; a run that produces the
same bytes is still a cache hit for it. Within one run (a cascade, Run All,
headless `strata run`) each cell executes once, so two consumers of a
`@nocache` cell see the same value.

## @stochastic

Mark a cell whose runs are **samples**: every run executes (as `@nocache`) and
every run's outputs are **kept**, each under its own run rather than under the
cell's provenance key.

```python
# @stochastic
import numpy as np

rng = np.random.default_rng()
draw = rng.normal(size=1000)
```

A `@nocache` cell re-executes but each run's outputs replace the last under the
same key. A `@stochastic` cell's outputs are keyed by the run that made them,
so the ledger holds every sample side by side, each tracing back to the same
inputs and source, and none of them is ever served as a cache hit. Use it for a
seedless simulation, a random search, anything where the point is the spread
of runs rather than the one answer.

`@stochastic` implies `@nocache`, so an explicit `# @nocache off` beside it, in
either order, contradicts it: the cell does not run, and fails with
`stochastic_nocache_conflict` on the `@nocache off` line. As with every
annotation, the last line of each key is the one read, so `# @stochastic off`
or a later `# @nocache` settles it.

## Cross-cell ordering

### `@after`

Add an ordering-only DAG edge from another cell to this one. Useful when the
dependency is on a side effect, e.g. a SQL `seed` cell creates the database
state that subsequent SQL cells query, and no Python variable flows
between them.

```sql
-- @sql connection=warehouse
-- @after seed
SELECT category, COUNT(*) FROM products GROUP BY category
```

Multiple `@after` lines stack; each cell ID adds one edge. Whitespace-
separated IDs on a single line work too: `# @after seed migrate`. An ID
that names no cell (`after_unknown_cell`), or a variant that is not active
and so never runs (`after_inactive_variant`), is an error: the ordering it
asks for would be lost, so the cell does not run until it is fixed. An ID
that names the cell itself adds no edge and is a warning (`after_self`).

`<cell-id>` is the `id` field in `notebook.toml`, **not** the cell's
`@name`. See [Cell IDs](#cell-ids) for how to set friendly IDs like
`seed` or `migrate`.

The edge participates in upstream/downstream wiring and the topological
order, but contributes no variable to `consumed_variables`, so it
doesn't affect per-variable provenance hashes.

## Notebook-level environment (runtime panel)

Notebook-wide environment variables are set via the **Runtime panel** in the sidebar. These apply to all cells unless overridden by a cell-level `@env` annotation.

Common use cases:

- **API keys**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` (for prompt cells)
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
