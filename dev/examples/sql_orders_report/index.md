# SQL Orders Report

A small five-cell notebook that demonstrates Strata's SQL cell support
against a local SQLite warehouse. The cells span three languages
(Python → SQL → Python) and exercise both `-- @cache fingerprint` (data-
sensitive) and `-- @cache forever` (reference-data) policies.

## What it shows

- **Connection config in `notebook.toml`.** The `[connections.warehouse]`
  block declares a SQLite driver with a relative path. SQL cells
  reference it by name via `-- @sql connection=warehouse`.
- **Bind parameters from upstream Python.** `top_orders` resolves
  `:min_amount` from the `threshold` cell's variable. No string
  substitution happens, the value flows through ADBC's parameter
  binding API, so adversarial strings round-trip as data.
- **Two cache policies side by side.** `top_orders` uses `fingerprint`
  (folds SQLite's `PRAGMA data_version` + `schema_version` into the
  hash) so a DDL change to the warehouse re-executes it. `category_summary`
  uses `forever` because the user's asserting the catalog is reference
  data; only edits to the SQL body itself invalidate it.
- **Read-only by default, opt-in writes.** SQL cells open the
  connection with `mode=ro` plus `PRAGMA query_only = ON` so a
  stray `INSERT` can't mutate the DB. The seed cell explicitly
  opts into writable execution via `-- @sql connection=warehouse
  write=true`: that's the per-cell escape hatch for setup
  scripts. Other cells in the notebook stay read-only.
- **Cross-language pipeline.** The two SQL results flow back into the
  `report` Python cell as `pyarrow.Table`s, which it turns into pandas
  DataFrames with `.to_pandas()`.

## Cells

| Cell | Language | What it does |
|---|---|---|
| `seed` | SQL (`write=true`) | Drops + creates `products` and `orders` and inserts ten rows. Default `session` cache policy means a re-run inside the same session cache-hits. |
| `threshold` | Python | Defines `min_amount = 50`. Edit and rerun to vary the threshold. |
| `top_orders` | SQL | `WHERE amount > :min_amount`, joined to the product catalog, top 5. Declares `-- @after seed`. |
| `category_summary` | SQL | Revenue by category with `-- @cache forever`. Declares `-- @after seed`. |
| `report` | Python | Stitches the two SQL outputs into a markdown report. |

## DAG dependencies

`top_orders` and `category_summary` depend on the SQLite file that
`seed` produces, but no Python variable flows between them, the
dependency is on a side effect. The `-- @after seed` annotation adds
an ordering-only edge to the DAG so cascade execution and staleness
both see the link.

## Running

From the project root:

```bash
uv run strata-notebook
```

Then open `examples/sql_orders_report` from the Strata home page. Run
`seed` first to create the SQLite file, then run the other cells in
order (or hit "Run all").

## Try this

- **Change the threshold.** Edit `threshold` to `min_amount = 100`.
  `top_orders` re-executes (different bind value → different
  provenance hash). Run `top_orders` again without changing
  anything, it cache-hits, because the canonical artifact's
  provenance still matches the current threshold.
- **Mutate the schema.** From a separate shell:

  ```bash
  sqlite3 analytics.db 'ALTER TABLE orders ADD COLUMN region TEXT'
  ```

  Re-run `top_orders`. SQLite's `PRAGMA schema_version` bumps, the
  freshness token changes, and the cell re-executes with a new schema.
- **Try a write from a SQL cell.** Add a cell with `INSERT INTO orders
  VALUES (...)`. The executor returns an error and the database row
  count is unchanged, read-only mode is the security boundary.

## seed

<sub>**kind** sql</sub>

```sql
-- @sql connection=warehouse write=true
-- Seed analytics.db with a small orders dataset.
--
-- ``write=true`` opts this cell into writable execution: the
-- adapter opens the connection without the read-only enforcement
-- (mode=ro / PRAGMA query_only=ON) so the DROP / CREATE / INSERT
-- statements below can run. Other SQL cells in the notebook stay
-- read-only by default: the override is per-cell, not per-
-- connection.
--
-- The default cache policy for a write cell is `session`, so this
-- cell runs once per session and dedup's no-op re-runs caused by
-- editing other cells. Edits to this body invalidate the session
-- cache and re-seed.

DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;

CREATE TABLE products (
    sku      TEXT PRIMARY KEY,
    category TEXT NOT NULL
);

CREATE TABLE orders (
    id         INTEGER PRIMARY KEY,
    customer   TEXT NOT NULL,
    sku        TEXT NOT NULL REFERENCES products(sku),
    amount     REAL NOT NULL,
    ordered_at TEXT NOT NULL
);

INSERT INTO products (sku, category) VALUES
    ('WIDGET-A',    'widgets'),
    ('WIDGET-B',    'widgets'),
    ('GADGET-A',    'gadgets'),
    ('GADGET-B',    'gadgets'),
    ('THINGAMAJIG', 'misc');

INSERT INTO orders (id, customer, sku, amount, ordered_at) VALUES
    (1,  'alice', 'WIDGET-A',    25.50,  '2026-04-01'),
    (2,  'alice', 'GADGET-A',    199.99, '2026-04-02'),
    (3,  'bob',   'WIDGET-B',    75.00,  '2026-04-02'),
    (4,  'bob',   'GADGET-B',    350.00, '2026-04-05'),
    (5,  'carol', 'WIDGET-A',    42.00,  '2026-04-08'),
    (6,  'carol', 'THINGAMAJIG', 500.00, '2026-04-10'),
    (7,  'dave',  'WIDGET-B',    18.00,  '2026-04-12'),
    (8,  'alice', 'GADGET-A',    199.99, '2026-04-15'),
    (9,  'bob',   'WIDGET-A',    25.50,  '2026-04-18'),
    (10, 'carol', 'GADGET-B',    350.00, '2026-04-20');
```

## min_amount

<sub>**kind** python</sub>

```python
# @name min_amount
"""Threshold for the orders SQL query.

Edit this number and the downstream SQL cell will re-execute on the
next run. ``top_orders`` folds ``min_amount`` into its provenance
hash, so 50 → 100 produces a different artifact; running again with
the same value cache-hits.
"""

min_amount = 50
min_amount
```

## top_orders

<sub>**kind** sql</sub>

```sql
-- @sql connection=warehouse
-- @name top_orders
-- @cache fingerprint
-- @after seed
SELECT
    o.id,
    o.customer,
    o.sku,
    p.category,
    o.amount,
    o.ordered_at
FROM orders AS o
JOIN products AS p USING (sku)
WHERE o.amount > :min_amount
ORDER BY o.amount DESC
LIMIT 5
```

## category_summary

<sub>**kind** sql</sub>

```sql
-- @sql connection=warehouse
-- @name category_summary
-- @cache forever
-- @after seed
-- The product catalog is reference data: it changes rarely and
-- the user's asserting "treat this as static". `forever` skips the
-- freshness probe; only an edit to the SQL body itself invalidates
-- the cache.
SELECT
    p.category,
    COUNT(DISTINCT p.sku) AS sku_count,
    COUNT(o.id)           AS order_count,
    ROUND(SUM(o.amount), 2) AS total_revenue
FROM products AS p
LEFT JOIN orders AS o USING (sku)
GROUP BY p.category
ORDER BY total_revenue DESC NULLS LAST
```

## report

<sub>**kind** python</sub>

```python
# @name report
"""Combine the two SQL results into a markdown report.

The SQL cells return Arrow Tables, which reach this cell as
``pyarrow.Table``; ``.to_pandas()`` gives the familiar ``.iterrows()``.
"""

lines = [
    "# Orders report",
    "",
    f"## Top {len(top_orders)} orders above ${min_amount}",
    "",
    "| customer | sku | category | amount | ordered |",
    "|---|---|---|---:|---|",
]
for _, row in top_orders.to_pandas().iterrows():
    lines.append(
        f"| {row['customer']} | {row['sku']} | {row['category']} | "
        f"${row['amount']:.2f} | {row['ordered_at']} |"
    )

lines += [
    "",
    "## Revenue by product category",
    "",
    "| category | SKUs | orders | revenue |",
    "|---|---:|---:|---:|",
]
for _, row in category_summary.to_pandas().iterrows():
    revenue = row["total_revenue"]
    revenue_str = f"${revenue:.2f}" if revenue is not None else "n/a"
    lines.append(
        f"| {row['category']} | {int(row['sku_count'])} | "
        f"{int(row['order_count'])} | {revenue_str} |"
    )

report = "\n".join(lines)
print(report)
report
```
