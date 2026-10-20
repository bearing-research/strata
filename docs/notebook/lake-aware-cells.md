# Lake-Aware Cells

A **lake-aware cell** is a notebook cell that takes an Iceberg table as a
versioned input via the [`@table`](annotations.md#table) annotation. The
table's current snapshot id is folded into the cell's provenance, so **new
data landing in the lake puts the cell out of date, its downstream cells go
stale, and the normal cascade re-runs them** - no manual data-version
bookkeeping, no re-pointing paths.

This page is the end-to-end walkthrough: build a tiny warehouse, scan it from
a cell, retrain when new data arrives, and pin a snapshot for reproducibility.
For the bare syntax, see the [`@table` reference](annotations.md#table).

## When to use it

Reach for `@table` when a cell's input is a table that **grows or changes over
time** and you want re-runs to track those changes automatically:

- Feature engineering or model training over an evolving fact table.
- Any pipeline where "the data moved" should invalidate downstream results the
  same way "the code changed" does.

If your input is a fixed file, a plain mount (`# @mount`) or a hard-coded path
is simpler. `@table` earns its keep precisely when the snapshot can move.

To query catalog tables in SQL rather than scan them from Python, a DuckDB
[SQL cell over the lake](cells.md#duckdb-over-the-lake) pins every table it
reads the same way, with no `@table` line.

## Prerequisites

- A running Strata server: `strata-notebook` (personal mode, the default)
  serves the notebook UI on `http://localhost:8765`.
- `pyiceberg` for the two setup scripts below. It is a core Strata dependency,
  so `uv run python` in the Strata project has it. The notebook's own
  environment does not need it: the server resolves the snapshot and runs the
  scan.

## Step 1 - Build a warehouse

Any Iceberg catalog works (local, S3, GCS, Azure). For this walkthrough, a
local SQLite-catalog warehouse with one table. Run this once, outside the
notebook. The catalog must be named `strata`: a warehouse URI always reads its
SQL catalog under that name (see
[Catalog](../reference/configuration.md#catalog)).

```python
# setup_warehouse.py
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField

WAREHOUSE = "/tmp/strata-demo/warehouse"

# Strata reads a warehouse URI's SQL catalog under the name "strata".
catalog = SqlCatalog(
    "strata",
    uri=f"sqlite:///{WAREHOUSE}/catalog.db",
    warehouse=WAREHOUSE,
)
catalog.create_namespace("shop")
schema = Schema(
    NestedField(1, "order_id", LongType(), required=False),
    NestedField(2, "amount", LongType(), required=False),
)
table = catalog.create_table("shop.orders", schema)

# Month 1 → snapshot S1
table.append(pa.table({"order_id": [1, 2, 3], "amount": [10, 20, 30]}))
print("table URI:", f"file://{WAREHOUSE}#shop.orders")
print("snapshot S1:", table.current_snapshot().snapshot_id)
```

```bash
mkdir -p /tmp/strata-demo/warehouse
uv run python setup_warehouse.py
```

The **table URI** is `<warehouse>#<namespace>.<table>` - here
`file:///tmp/strata-demo/warehouse#shop.orders`. This is the same URI format
`client.materialize` accepts. A table in a catalog the server names under
`STRATA_CATALOGS` is written `<catalog>:<namespace>.<table>` instead (see the
[`@table` reference](annotations.md#table)).

Strata finds a local warehouse's tables in `<warehouse>/catalog.db`. An object
store (`s3://`, `gs://`, `abfs://`) has no such file, so set the catalog database
with `STRATA_CATALOG_URI`. Without it, a personal server keeps the catalog in
SQLite at `STRATA_METADATA_DB` on its own disk, which no other reader of the
bucket sees, and a service refuses the table, naming the setting. Either
catalog reads the table's metadata with the server's own S3, GCS or Azure
settings ([GCS](../reference/configuration.md#gcs-storage),
[Azure](../reference/configuration.md#azure-storage), which also says when an
Azure table needs `adlfs`). A configured
warehouse (`warehouse` in `catalog_properties` or in a named SQL catalog) in
object storage with no `uri` stops the server at startup
([Catalog settings](../reference/configuration.md#catalog)).

## Step 2 - Declare a lake-aware cell

In a notebook cell, declare the table and scan it with the cell's
[ambient `strata` client](cells.md#the-ambient-strata-client), which is already
bound to the server. The `@table` annotation injects two variables: `orders`
(the table URI) and `orders_snapshot` (the resolved snapshot id).

```python
# @table orders file:///tmp/strata-demo/warehouse#shop.orders
scan = strata.materialize(
    inputs=[orders],
    transform={"executor": "scan@v1", "params": {"snapshot_id": orders_snapshot}},
    name="shop/orders-raw",
)
df = scan.to_pandas()

# Re-export the snapshot as a real variable so downstream cells can use it
# (injected @table vars live only in this cell - see "Gotchas" below).
orders_snapshot_value = orders_snapshot

total = int(df["amount"].sum())
print(f"scanned {len(df)} rows at snapshot {orders_snapshot} - total={total}")
```

Run it (Shift+Enter). Passing `orders_snapshot` to the scan makes the cell
**deterministic**: it reads exactly the snapshot its provenance recorded.

## Step 3 - The staleness loop

Add a downstream cell that depends on the scan:

```python
report = f"orders total at snapshot {orders_snapshot_value}: {total}"
report
```

Run all cells - both go green. Now **land new data** in the lake:

```python
# append_month2.py
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog

catalog = SqlCatalog(
    "strata",
    uri="sqlite:////tmp/strata-demo/warehouse/catalog.db",
    warehouse="/tmp/strata-demo/warehouse",
)
table = catalog.load_table("shop.orders")
table.append(pa.table({"order_id": [4, 5], "amount": [40, 50]}))  # snapshot S2
print("snapshot S2:", table.current_snapshot().snapshot_id)
```

```bash
uv run python append_month2.py
```

Back in the notebook, the `@table` cell is now **out of date (idle)** - its
snapshot id moved from S1 to S2, so its provenance changed and no stored result
matches it. The cells downstream of it that ran before read **stale**. The
badges do not update live: the notebook looks up the current snapshot only when
it recomputes staleness, which happens when you reopen the notebook, edit a
cell, or run one. Refreshing the browser tab alone keeps the old badge. A plain
**Run** (no force) of the scan cell recomputes it against S2; running a
downstream cell instead **cascades** the scan first, then runs that cell.
Nothing changed in your code; the data moved, and Strata treated that exactly
like a code change.

Run again without appending and the cell is a **cache hit** - same snapshot,
same provenance, instant.

## Step 4 - Pin a snapshot for reproducibility

To freeze a cell to one snapshot forever (e.g. to reproduce a past result),
add `snapshot=<id>`:

```python
# @table orders file:///tmp/strata-demo/warehouse#shop.orders snapshot=1292033279574548405
```

A pinned cell reads that snapshot regardless of new data and **never goes
out of date** on appends - the lake-side analog of a mount `pin`. Drop the
`snapshot=` to return to tracking the current snapshot.

## How it works

The snapshot id is part of the cell's **provenance hash**, alongside the
source hash, environment hash, and input hashes:

```
provenance = hash(input_hashes + mount_fingerprints + table_fingerprints
                  + fetch_fingerprints + dataset_fingerprints,
                  source_hash, env_hash)
```

A table fingerprint is `"<name>:table:<uri>:<snapshot_id>"`. Because the
snapshot id is immutable and content-addressed, a cached result for a given
provenance is valid forever - and a moved snapshot is a different provenance,
hence a different (missing) cache entry, hence a recompute. This is the same
provenance machinery that puts ordinary cells out of date when their source or
inputs change; `@table` adds the lake snapshot to the mix.

## Which cells read a table

`.strata/tables/index.json` lists, for each cell, the tables it read on its
last successful run (a cache hit included) and the snapshot of each. A SQL
cell's catalog tables are listed too. A tool that shows a table's readers
reads this file rather than parsing cell sources:

```json
{
  "c1a2b3c4": [
    {
      "uri": "file:///tmp/strata-demo/warehouse#shop.orders",
      "catalog": null,
      "table": "shop.orders",
      "snapshot_id": 1292033279574548405
    }
  ]
}
```

`uri` is as the cell wrote it; a SQL cell's table appears as
`<catalog>:<namespace>.<table>`. `catalog` is the configured catalog the URI
names, or `null` for a warehouse path or the server's default catalog
(`STRATA_CATALOG_NAME`). `snapshot_id` is `null` for a table with no snapshots
yet. A failed run leaves the cell's entry as it was, and a
cell that stops reading tables, or is deleted, drops out.

## What a run moved

`.strata/reads/index.json` lists, for each cell, the bytes its last successful
run moved onto the machine for each `@mount`, `@table` and `@fetch` input, and
the machine it ran on (`local` for the server itself), so a host can count the
data a notebook moved:

```json
{
  "c1a2b3c4": {
    "machine": "local",
    "ran_at": 1760889600000,
    "inputs": [
      {"kind": "mount", "name": "data", "uri": "s3://bucket/data", "credential": null, "bytes_moved": 5368709120},
      {"kind": "table", "name": "orders", "uri": "file:///tmp/strata-demo/warehouse#shop.orders", "credential": null, "bytes_moved": 41943040},
      {"kind": "fetch", "name": "zones", "uri": "https://example.com/zones.csv", "credential": null, "bytes_moved": 0}
    ]
  }
}
```

`bytes_moved` is 0 for an input already on the machine, a cache hit included,
and `null` where it is not known: a mount on a worker that does not report what
it downloaded (a remote worker reports it in its result). A table counts what
the cell's scans through the ambient `strata` client read from storage, the
compressed Parquet column chunks: 0 when the row-group cache or a stored scan
answered, and `null` when the cell read the table some other way or not at all,
since the server only sees its own scans. On a
worker reached through signed URLs, a forced rerun of a build the server already
stored is answered with that stored result, so its entry repeats the first run's
figures. It follows the table index's rules: a failed run leaves the
entry, and a cell that stops reading these inputs, or is deleted, drops out.

With a [read threshold](#asking-before-a-large-read) set, each entry also has
`confirmation`: `under_threshold`, `approved` (someone said Run, now or for the
same inputs on that machine before), or `not_asked` (a run that could not ask,
such as a headless `strata run`, went ahead over it).

## Asking before a large read

A host can have the notebook ask before a run moves a lot of data onto a
machine. Before a run, each cell about to run is weighed as its badge is, on
the machine it will run on, except that a `@fetch` is asked with a `HEAD` rather
than downloaded. What would move is what the machine does not hold yet, of the
sizes that are known; a table counts its snapshot's files, an upper bound. When
that passes `STRATA_NOTEBOOK_CONFIRM_READ_BYTES`, or its cost at
`STRATA_NOTEBOOK_CREDENTIAL_READ_COST`'s rate per GB passes
`STRATA_NOTEBOOK_CONFIRM_READ_COST`, nothing starts: the notebook lists those
cells, their inputs, sizes and cost, and the host's
`STRATA_NOTEBOOK_CONFIRM_READ_MESSAGE`, with Run and Cancel. One question covers
the whole run, Run All included. A ready cell is left out, since its run is a
cache hit.

A Run is kept per cell, machine and inputs (the mount listings, table snapshots
and fetch URLs), so a rerun on the same machine does not ask again, while new
data, or another machine, does. A run that cannot ask goes ahead and its reads
index entry says `not_asked`: a headless `strata run`, the CLI, and REST or MCP
runs that do not pass `confirm_reads` (see
[Execute Cell](../reference/rest-api.md#execute-cell-rest) and
[MCP](mcp.md)). Nothing is ever refused for its size.

The question comes before a cell's cache is checked, so a stale cell whose
result turns out to be cached may still ask; asking is the safe direction.

## Gotchas

- **Injected vars don't flow downstream.** `orders` and `orders_snapshot` live
  only in the *declaring* cell's namespace - they are injections, not cell
  *defines*, so downstream cells can't reference them directly. Re-export what
  you need as a real assignment (`orders_snapshot_value = orders_snapshot`),
  exactly as in Step 2. This mirrors how mount variables behave.
- **The name must be a valid Python identifier.**
- **Schema-only changes do not restale the cell.** The fingerprint is the
  snapshot id, and changing a table's schema (adding or renaming a column)
  commits no new snapshot. The cell stays a cache hit, and a scan that passes
  `orders_snapshot` keeps reading that snapshot's schema. The new schema shows
  up once new data lands in a new snapshot.
- **Unreachable catalog → conservatively out of date.** If the catalog can't
  be reached when provenance is computed (which also happens on notebook open),
  the cell is treated as out of date rather than crashing; if it's still unreachable
  at execution time, the run fails with a clear error.
- **The embedded scan runs in-process.** `scan@v1` is handled by the server
  itself in both modes: it is resolved before any executor dispatch, so
  scanning a table needs no registered executor and none is consulted.
- **Merge-on-read deletes are applied.** When Spark, Flink or DuckDB
  deletes from a table without rewriting its data files, the scan drops the
  rows the snapshot's positional delete files (format v2) or deletion vectors
  (format v3) name, and caches the row groups without them. Equality deletes,
  which Flink and CDC sinks write for upserts, are applied too: every older
  row whose key a delete names is dropped, null matching null. Applying them
  holds the delete keys in memory, so a scan in which a row group would need
  more than `max_equality_delete_rows` of them (10 million by default) is
  refused while planning, with a message saying so, rather than read
  partially; compact the table (`rewrite_data_files`) to bring the count
  down. An equality delete file in ORC or Avro, or keyed on a struct column,
  is refused the same way.
- **Schema changes are read the way Iceberg defines them.** Columns are
  matched by field id, not by name, so an older data file reads as the table's
  schema: an added column is null in it (or its default, for a column a v3
  table added with one), a renamed column comes back under its new name, a
  column dropped and added again is null rather than the old values, a
  widened type (int to long, float to double, a wider decimal) comes back
  wide, and a required column made optional reads across both kinds of file,
  including for the fields inside a struct, list or map column. Nanosecond
  timestamps in a v1 or v2 table are read at the table's microsecond unit,
  truncating, as pyiceberg does. A scan of the current table reads the
  current schema; one that names a snapshot reads that snapshot's.

## See also

- [`@table` annotation reference](annotations.md#table) - the syntax surface.
- [Cell Annotations](annotations.md) - all per-cell annotations.
- [Core Quickstart](../getting-started/core.md) - `client.materialize` and
  `scan@v1` from the SDK directly, without the notebook.
