# @sql connection=warehouse write=true
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
