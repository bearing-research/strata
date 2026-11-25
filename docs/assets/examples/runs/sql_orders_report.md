<!-- strata example run, source a9b003c5cfbb4d71 -->
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
-- With no `-- @cache`, a write cell is cached by its body, in its
-- session and after, while the database is as its run left it: a
-- reopen probes analytics.db and finds this cell, and the cells after
-- it, ready. An edit to this body, or a database deleted or changed
-- since, re-seeds.

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

6 rows × 3 cols
| stmt | kind | rows_affected |
| --- | --- | --- |
| 1 | DROP TABLE | NULL |
| 2 | DROP TABLE | NULL |
| 3 | CREATE TABLE | NULL |
| 4 | CREATE TABLE | NULL |
| 5 | INSERT | 5 |
| 6 | INSERT | 10 |

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

```json title="Output"
50
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

5 rows × 6 cols
| id | customer | sku | category | amount | ordered_at |
| --- | --- | --- | --- | --- | --- |
| 6 | carol | THINGAMAJIG | misc | 500.0 | 2026-04-10 |
| 4 | bob | GADGET-B | gadgets | 350.0 | 2026-04-05 |
| 10 | carol | GADGET-B | gadgets | 350.0 | 2026-04-20 |
| 2 | alice | GADGET-A | gadgets | 199.99 | 2026-04-02 |
| 8 | alice | GADGET-A | gadgets | 199.99 | 2026-04-15 |

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

3 rows × 4 cols
| category | sku_count | order_count | total_revenue |
| --- | --- | --- | --- |
| gadgets | 2 | 4 | 1099.98 |
| misc | 1 | 1 | 500.0 |
| widgets | 2 | 5 | 186.0 |

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
Markdown(report)
```

# Orders report

## Top 5 orders above $50

| customer | sku | category | amount | ordered |
|---|---|---|---:|---|
| carol | THINGAMAJIG | misc | $500.00 | 2026-04-10 |
| bob | GADGET-B | gadgets | $350.00 | 2026-04-05 |
| carol | GADGET-B | gadgets | $350.00 | 2026-04-20 |
| alice | GADGET-A | gadgets | $199.99 | 2026-04-02 |
| alice | GADGET-A | gadgets | $199.99 | 2026-04-15 |

## Revenue by product category

| category | SKUs | orders | revenue |
|---|---:|---:|---:|
| gadgets | 2 | 4 | $1099.98 |
| misc | 1 | 1 | $500.00 |
| widgets | 2 | 5 | $186.00 |

```text title="stdout"
# Orders report

## Top 5 orders above $50

| customer | sku | category | amount | ordered |
|---|---|---|---:|---|
| carol | THINGAMAJIG | misc | $500.00 | 2026-04-10 |
| bob | GADGET-B | gadgets | $350.00 | 2026-04-05 |
| carol | GADGET-B | gadgets | $350.00 | 2026-04-20 |
| alice | GADGET-A | gadgets | $199.99 | 2026-04-02 |
| alice | GADGET-A | gadgets | $199.99 | 2026-04-15 |

## Revenue by product category

| category | SKUs | orders | revenue |
|---|---:|---:|---:|
| gadgets | 2 | 4 | $1099.98 |
| misc | 1 | 1 | $500.00 |
| widgets | 2 | 5 | $186.00 |
```
