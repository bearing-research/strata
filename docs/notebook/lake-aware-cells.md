# Lake-Aware Cells

A **lake-aware cell** is a notebook cell that takes an Iceberg table as a
versioned input via the [`@table`](annotations.md#table) annotation. The
table's current snapshot id is folded into the cell's provenance, so **new
data landing in the lake makes the cell stale and the normal cascade re-runs
it** - no manual data-version bookkeeping, no re-pointing paths.

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

- A running Strata server: `uv run python -m strata` (personal mode, the
  default) serves the notebook UI on `http://localhost:8765`.
- `pyiceberg` for the two setup scripts below. It is a core Strata dependency,
  so `uv run python` in the Strata project has it. The notebook's own
  environment does not need it: the server resolves the snapshot and runs the
  scan.

## Step 1 - Build a warehouse

Any Iceberg catalog works (local, S3, GCS, Azure). For this walkthrough, a
local SQLite-catalog warehouse with one table. Run this once, outside the
notebook:

```python
# setup_warehouse.py
import pyarrow as pa
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import LongType, NestedField

WAREHOUSE = "/tmp/strata-demo/warehouse"

catalog = SqlCatalog(
    "demo",
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
[Azure](../reference/configuration.md#azure-storage)). A configured
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
    "demo",
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

Back in the notebook, the `@table` cell is now **stale** - its snapshot id
moved from S1 to S2, so its provenance changed. The badge does not update live:
the notebook looks up the current snapshot only when it recomputes staleness,
which happens when you reopen the notebook, edit a cell, or run one. Refreshing
the browser tab alone keeps the old badge. A plain **Run** (no force)
recomputes the scan against S2 and **cascades** the rebuild to every
downstream cell. Nothing changed in your code; the data moved, and Strata
treated that exactly like a code change.

Run again without appending and the cell is a **cache hit** - same snapshot,
same provenance, instant.

## Step 4 - Pin a snapshot for reproducibility

To freeze a cell to one snapshot forever (e.g. to reproduce a past result),
add `snapshot=<id>`:

```python
# @table orders file:///tmp/strata-demo/warehouse#shop.orders snapshot=1292033279574548405
```

A pinned cell reads that snapshot regardless of new data and **never goes
stale** on appends - the lake-side analog of a mount `pin`. Drop the
`snapshot=` to return to tracking the current snapshot.

## How it works

The snapshot id is part of the cell's **provenance hash**, alongside the
source hash, environment hash, and input hashes:

```
provenance = hash(input_hashes + mount_fingerprints + table_fingerprints,
                  source_hash, env_hash)
```

A table fingerprint is `"<name>:table:<uri>:<snapshot_id>"`. Because the
snapshot id is immutable and content-addressed, a cached result for a given
provenance is valid forever - and a moved snapshot is a different provenance,
hence a different (missing) cache entry, hence a recompute. This is the same
provenance machinery that makes ordinary cells stale when their source or
inputs change; `@table` adds the lake snapshot to the mix.

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
- **Unreachable catalog → conservatively stale.** If the catalog can't be
  reached when provenance is computed (which also happens on notebook open),
  the cell is treated as stale rather than crashing; if it's still unreachable
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
