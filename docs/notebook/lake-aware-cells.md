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

## Prerequisites

- A running Strata server in **personal mode** (the embedded `scan@v1`
  transform runs there): `uv run python -m strata` serves the notebook UI on
  `http://localhost:8765`.
- `pyiceberg` available in your notebook environment (it ships with the
  `[notebook]` extra).

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
`client.materialize` accepts.

## Step 2 - Declare a lake-aware cell

In a notebook cell, declare the table and scan it. The `@table` annotation
injects two variables: `orders` (the table URI) and `orders_snapshot` (the
resolved snapshot id).

```python
# @table orders file:///tmp/strata-demo/warehouse#shop.orders
from strata_client import StrataClient

client = StrataClient(base_url="http://127.0.0.1:8765")

scan = client.materialize(
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

catalog = SqlCatalog("demo", uri="sqlite:////tmp/strata-demo/warehouse/catalog.db",
                     warehouse="/tmp/strata-demo/warehouse")
table = catalog.load_table("shop.orders")
table.append(pa.table({"order_id": [4, 5], "amount": [40, 50]}))  # snapshot S2
print("snapshot S2:", table.current_snapshot().snapshot_id)
```

```bash
uv run python append_month2.py
```

Back in the notebook, the `@table` cell now shows **stale** - its snapshot id
moved from S1 to S2, so its provenance changed. A plain **Run** (no force)
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
inputs change; `@table` simply adds the lake snapshot to the mix.

## Gotchas

- **Injected vars don't flow downstream.** `orders` and `orders_snapshot` live
  only in the *declaring* cell's namespace - they are injections, not cell
  *defines*, so downstream cells can't reference them directly. Re-export what
  you need as a real assignment (`orders_snapshot_value = orders_snapshot`),
  exactly as in Step 2. This mirrors how mount variables behave.
- **The name must be a valid Python identifier.**
- **Unreachable catalog → conservatively stale.** If the catalog can't be
  reached when provenance is computed (which also happens on notebook open),
  the cell is treated as stale rather than crashing; if it's still unreachable
  at execution time, the run fails with a clear error.
- **Personal mode for the embedded scan.** `scan@v1` runs as a built-in
  transform in personal mode. In service mode, scanning goes through a
  registered executor.

## See also

- [`@table` annotation reference](annotations.md#table) - the syntax surface.
- [Cell Annotations](annotations.md) - all per-cell annotations.
- [Core Quickstart](../getting-started/core.md) - `client.materialize` and
  `scan@v1` from the SDK directly, without the notebook.
