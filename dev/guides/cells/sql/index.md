# SQL cells

A SQL cell sends a query to a connected database via ADBC (DuckDB through its own driver) and stores the result as an Arrow Table artifact. Like Python and prompt cells, SQL cells participate in the DAG, cache by provenance hash, and surface their output to downstream cells.

```sql
-- @sql connection=warehouse
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

The cell above pulls `min_amount` from an upstream Python cell, sends a parameterized query through the `warehouse` connection, and stores the resulting rows as an Arrow Table, which downstream cells receive as a `pyarrow.Table` (`.to_pandas()` for a DataFrame).

## Connections

A SQL cell references a **named connection**. Connections live in `notebook.toml` under `[connections.<name>]`, but you don't need to edit that file by hand. Open **Notebook settings** in the right sidebar, then the **Connections panel**, click `+ Add connection`, and fill in the form. The driver dropdown switches the field layout per backend (path for DuckDB and SQLite; URI + auth + role + search path for PostgreSQL; account / warehouse / database / schema for Snowflake; project / dataset / credentials for BigQuery).

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
- **Auth values use `${VAR}` indirection.** Literal credentials get blanked when `notebook.toml` is saved, so committing the file never leaks secrets. The form shows a warning border on a literal value so you know to switch it to a variable reference. A personal server reads the variable from its own environment; a server in [service mode](https://bearingresearch.com/strata/dev/deployment/service-mode/) reads it from the notebook's env.
- **Relative `path` values are notebook-local.** `path = "analytics.db"` resolves against the notebook directory at execution time. The on-disk value stays relative so a notebook moves cleanly between machines.
- **Currently shipped drivers**: DuckDB, SQLite, PostgreSQL, Snowflake, and BigQuery. DuckDB uses the native DuckDB DBAPI; the other four are ADBC-backed (`adbc-driver-sqlite`, `adbc-driver-postgresql`, `adbc-driver-snowflake`, `adbc-driver-bigquery`). For Snowflake, read cells use `role`; `write=true` cells switch to `write_role` when configured, otherwise they reuse `role`. For BigQuery, read cells use `credentials_path`; `write=true` cells switch to `write_credentials_path` when configured, otherwise they reuse `credentials_path`. Snowflake and BigQuery do not have a session-level read-only flag like PostgreSQL's `SET default_transaction_read_only = on`, so the safety boundary is the grants on the configured role or service account.

## DuckDB over the lake

A DuckDB connection can read the organization's Iceberg catalog and the notebook's mounts. `catalog` names a REST or AWS Glue catalog the server configures (`[tool.strata.catalogs.<name>]`) or, on a personal server, one the notebook defines in [`[catalogs.<name>]`](https://bearingresearch.com/strata/dev/reference/notebook-toml/#notebook-catalogs). `mounts` names mounts declared for the cell, in `[[mounts]]` or with `-- @mount`:

```toml
[connections.lake]
driver = "duckdb"
path = ":memory:"
catalog = "lake"
mounts = ["raw"]
```

```sql
-- @sql connection=lake
SELECT t.zone, count(*) AS trips
FROM lake.taxi.trips t JOIN raw USING (zone)
GROUP BY t.zone
```

Each mount is a view by its name over its Parquet, CSV or JSON files, read with the mount's storage options. `file`, `s3`, `gs` and `az` mounts work; GCS and Azure ones are read through the same fsspec filesystem (gcsfs, adlfs) a Python cell's mount uses, so a connection's mounts on one of those schemes must share their storage options. Every catalog table the query reads is an input the way an `@table` declaration is: the cell's provenance folds its current snapshot, the query reads that snapshot, and a new snapshot makes the cell idle (out of date) and the cells downstream of it stale. With [`-- @cache snapshot`](https://bearingresearch.com/strata/dev/reference/annotations/sql/#cache) the cell instead keeps reading the snapshots its first run read until it is rerun. Each mount's fingerprint is folded too, so a new file does the same. The catalog is attached read-only and the mounts are views; a `write` cell on the connection writes only its `path` database. A REST catalog's `s3.*` keys apply only under an `s3://` warehouse; otherwise its tables are read with the credentials the catalog vends. A catalog that vends nothing (its store has no STS) sets `header.X-Iceberg-Access-Delegation` to something other than `vended-credentials`, usually empty: DuckDB then asks it for no credentials, and its `s3.*` keys read the tables, under its `s3://` warehouse or, when the warehouse is a name, under the locations of the tables the query names. A table named only at run time is outside those, so its files are read without the keys. So are data files outside a table's location (added with `add_files` from another prefix, or a `write.data.path` elsewhere): in personal mode they are read without the keys and fail with the store's 403, and service mode refuses them already. A Glue catalog is attached through Glue's Iceberg REST endpoint in its `glue.region` for the account `glue.id`, both required; its `glue.*` keys (or `client.*`, or `s3.*`) sign the catalog's requests and read its tables' files, and without keys the AWS credential chain (environment, profile, instance role) does.

A read cell runs reads: `SELECT`, set operations, `VALUES`, `TABLE`, `SUMMARIZE`, `PIVOT`, `DESCRIBE`, `SHOW` and plain `EXPLAIN`. Anything else (DDL, DML, `COPY`, `ATTACH`, `USE`, `CALL`, `SET`, `PRAGMA`, and `EXPLAIN ANALYZE`, which runs the statement it describes) is refused before the cell reaches the driver, naming the statement, because the read-only transaction the driver opens is one a `COMMIT` in the body can end. This holds on every driver, not only DuckDB. Use `-- @sql connection=<name> write=true` for a cell that changes a database.

## Schema discovery

The **Schema panel**, under **Notebook settings** in the sidebar, shows the tables and columns visible through each declared connection. Click a connection to lazy-load its schema; click a table to expand its columns. The `↻` button re-fetches when the underlying database has changed externally. No SQL cell needs to run for this: the panel talks directly to each driver's catalog query surface (`sqlite_master` for SQLite, `information_schema.tables JOIN columns` for PostgreSQL, and the driver-specific catalog queries for Snowflake and BigQuery).

## Bind parameters

`:name` placeholders resolve against upstream cell variables. One inside a string literal or a comment is not a placeholder, and strings end where the connection's dialect ends them (on Snowflake and BigQuery, `\'` does not). Strata coerces a strict allowlist of Python types (`int`, `float`, `str`, `bytes`, `bool`, `None`, `Decimal`, `UUID`, `datetime`/`date`/`time`) into ADBC bind values; anything else (a list, a table, a custom object) is rejected with a clear error. An upstream numpy scalar is stored as the Python number it holds, so it binds as that number. A value stored as a Python pickle is rejected without being loaded: the server never unpickles a cell's output. **No string substitution ever**: values flow through ADBC's prepared-statement layer, so adversarial strings (`'; DROP TABLE …`) round-trip as data, not SQL.

```python
# upstream Python cell
min_amount = 100
```

```sql
-- @sql connection=warehouse
SELECT * FROM orders WHERE amount > :min_amount
```

```text title="Output (illustrative result rows)"
   order_id  customer_id  amount       placed_at
0      1042            7   245.99  2026-04-12
1      1093           14   180.50  2026-04-13
2      1117            3   299.00  2026-04-14
```

The DAG links the SQL cell to the Python cell automatically, with the same edge logic Strata uses for Python free variables. The run records the values it bound, so the lineage behind a result names the value it ran at.

## Cache policies

A SQL cell's **provenance hash** folds together:

- The query's own tokens (so whitespace, comment and keyword-case edits don't churn the cache, and any other edit runs the query again).
- The bind parameters (type-tagged: `True` ≠ `1`).
- The connection's identity (host / DB / user / role / search_path, never the password).
- The hashes of every upstream artifact referenced via `:name`.
- The driver's **freshness probe** result for the touched tables.
- The driver's **schema fingerprint** for the touched tables.
- A salt derived from the `-- @cache` policy below.

`-- @cache` controls how DB-side state factors in. Default is `fingerprint`.

| Policy          | Behavior                                                                                                                  | When to use                        |
| --------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `fingerprint`   | Default. Probe-derived freshness token + schema fingerprint folded in.                                                    | Most queries.                      |
| `forever`       | Static salt; never invalidates from DB-side state.                                                                        | True reference data. User asserts. |
| `session`       | Session-unique salt; invalidates across sessions.                                                                         | Always-fresh queries / dashboards. |
| `ttl=<seconds>` | `floor(now / ttl)` in the salt; bucketed time-based invalidation.                                                         | Stale-tolerant aggregations.       |
| `snapshot`      | Pins the query to one point in time through the warehouse's time travel (Snowflake, BigQuery), or to the Iceberg snapshots of a DuckDB connection's catalog tables. Refused on other drivers before a connection opens. See [`@cache`](https://bearingresearch.com/strata/dev/reference/annotations/sql/#cache). | Reproducibility-critical reads.    |

**On reopen.** `fingerprint` and `snapshot` open idle. Both say, in the cell's
own annotation, that the cached rows are good only while the source still
agrees, and asking the source is a query, not something to do while opening a
notebook. `forever`, `session` and `ttl` can come back ready, because each has
already said what it depends on and that is settled locally: a `session` cell
matches only within the session that ran it, and a `ttl` cell only inside the
same time bucket, so one whose window has passed comes back idle anyway.

The connection counts too. Repointing `[connections.<name>]` at another
database, or at a different credential, means the cell is no longer ready about
what it was ready about, so it opens idle rather than green with the previous
database's rows.

```sql
-- @sql connection=warehouse
-- @cache forever
SELECT * FROM dim_country
```

## Per-driver freshness

`fingerprint` correctness depends on what the driver can probe.

| Driver     | Probe                                           | Granularity | Notes                                                                                                                                                                                                                                                                                                                                                        |
| ---------- | ----------------------------------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PostgreSQL | `pg_stat_user_tables` + `pg_class.relfilenode`  | per-table   | Up to ~500 ms stats-collector lag.                                                                                                                                                                                                                                                                                                                           |
| SQLite     | The database file's size, mtime and header change counter, its `-wal` sidecar, plus `PRAGMA data_version` / `schema_version` | **DB-wide** | Any commit to the file, from any process, changes the token, so a write to one table invalidates cells reading another. A `:memory:` database has only the pragmas.                                                                                                                                                                              |
| DuckDB     | `PRAGMA database_size`                          | **DB-wide** | Advances when DuckDB writes blocks; two row states between block-aligned flushes can share a token. Pin `-- @cache session` on a database that is being written to while you work.                                                                                                                                                                         |
| Snowflake  | `INFORMATION_SCHEMA.TABLES.LAST_ALTERED`        | per-table   | Per-database scoping (one query per touched database). Bills cloud-services credits but each query is small. `LAST_ALTERED` updates even on 0-row DML, safe direction (over-invalidates, never under).                                                                                                                                                      |
| BigQuery   | `__TABLES__.last_modified_time`                 | per-table   | Per-dataset scoping. `__TABLES__` is the legacy-but-stable view; `INFORMATION_SCHEMA.TABLES` doesn't expose `last_modified_time`. **Streaming-buffer caveat**: tables receiving streaming inserts have `last_modified_time` lag by minutes-to-90-min until the buffer flushes; pin `-- @cache session` on those queries. Permissions: `bigquery.tables.get`. |

The schema fingerprint catches metadata-only changes (`ADD COLUMN`, type changes, nullability flips) that the freshness token would miss.

A probe can only ask about a table the query names. A table named at run time (Snowflake's `IDENTIFIER($tbl)` or `TABLE($tbl)`), a table function (`read_parquet(...)`, `query_table(...)`, a Postgres set-returning function) or a file read in place of a table (DuckDB's `FROM 'events.parquet'`) is not fingerprinted, so under `fingerprint` such a cell re-runs every time, and the header says so. Its result's content is folded into its provenance, so a downstream cell re-runs only when the rows actually changed. Declare `-- @cache session` or `-- @cache ttl=...` to reuse results instead, or read files through a mount on a [DuckDB lake connection](#duckdb-over-the-lake), whose fingerprint the cache does track.

## Read-only by default

SQL cells are **read-only by default**, but the enforcement mechanism depends on the backend:

- **SQLite**: `mode=ro` plus `PRAGMA query_only=ON`
- **DuckDB**: a file opens with `read_only=True`, and every read runs inside `BEGIN TRANSACTION READ ONLY` (which also covers `:memory:`)
- **PostgreSQL**: `SET default_transaction_read_only = on`
- **Snowflake**: the configured `role` must be read-only
- **BigQuery**: the configured `credentials_path` must point at a read-only service account

For SQLite, DuckDB and PostgreSQL, Strata enforces read-only at the connection/session level. For Snowflake and BigQuery, Strata selects the read-scoped role or credentials, and the cloud platform's grants are the actual boundary. In all cases, the default path is “read unless you explicitly opt into `write=true`,” and the statement check above refuses anything but a read before the driver sees it.

In service mode a DuckDB or SQLite cell, which runs inside the server process, is also confined: its database file must be in the notebook's directory or outside the server's state (a SQLite `uri` is refused), a DuckDB connection reaches only its own database and the mounts and catalog tables it reads, and a SQLite write cell cannot `ATTACH`, `DETACH` or `VACUUM`. A BigQuery connection there must name a `credentials_path` key file in the notebook's directory, so it never runs as the server's own Google credentials. See [Service mode](https://bearingresearch.com/strata/dev/deployment/service-mode/).

## Write cells

Setup, seeding, and migration scripts opt into writable execution per cell:

```sql
-- @sql connection=warehouse write=true
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer TEXT NOT NULL,
    amount REAL
);
INSERT INTO orders VALUES (1, 'alice', 25.50), (2, 'bob', 199.99);
```

- The body is split into individual statements on sqlglot's tokenizer (ADBC's cursor runs only the first statement otherwise). Each statement runs as written in the cell, and a trailing comment is not a statement.
- `:name` bind placeholders work the same as in read cells.
- The default cache policy is `session` (one execution per session; same body in the same session is a cache hit).
- `-- @cache fingerprint` and `-- @cache snapshot` error early on write cells: probe-based invalidation has no anchor when the cell mutates state.
- The cell still produces an Arrow artifact: a per-statement status table with `stmt`, `kind` (`CREATE TABLE`, `INSERT`, …), and `rows_affected` (nullable; `null` for DDL).
- Read cells using the same connection stay on the read path; the override is per-cell.

A query runs off the server's event loop, so a cancel lands at once: the run is abandoned and publishes nothing. The query itself may keep running in the database, because ADBC drivers differ on whether they can cancel one.

## `-- @name` and downstream consumption

A SQL cell's output variable name defaults to `result`; override with `-- @name <identifier>`. Downstream cells receive the result as a `pyarrow.Table` (Arrow IPC artifacts deserialize through the standard notebook serializer); call `.to_pandas()` for a pandas DataFrame:

```sql
-- @sql connection=warehouse
-- @name top_customers
SELECT customer, SUM(amount) AS total
FROM orders GROUP BY customer ORDER BY total DESC LIMIT 5
```

```python
# downstream Python cell
print(top_customers.shape)  # (5, 2)
print(top_customers.to_pandas()["total"].sum())
```

## `-- @after` for setup-then-query pipelines

A read SQL cell that depends on a write SQL cell's side effects (the underlying database state) can declare an explicit ordering edge:

```sql
-- @sql connection=warehouse write=true
CREATE TABLE products (sku TEXT PRIMARY KEY, category TEXT);
INSERT INTO products VALUES ('A', 'widgets'), ('B', 'gadgets');
```

```sql
-- @sql connection=warehouse
-- @after seed
SELECT category, COUNT(*) FROM products GROUP BY category
```

`-- @after seed` adds a DAG edge from the `seed` cell to this one even though no Python variable flows between them: the dependency is on a side effect (the SQLite file). This is what cascade execution and staleness recompute use to ensure the right ordering.

## Worked example

The [`sql_orders_report`](https://bearingresearch.com/strata/dev/examples/sql_orders_report/) example notebook walks through all of this end-to-end: a SQL `seed` cell, a Python `threshold` cell, two parameterized SQL queries, and a Python report cell: five cells, two languages, with both `fingerprint` and `forever` cache policies side by side.

## SQL-cell annotations

| Annotation                              | What it does                                          |
| --------------------------------------- | ----------------------------------------------------- |
| `-- @sql connection=<name> [write=true]` | Mark the cell as SQL; reference a declared connection |
| `-- @cache <policy>`                     | Override the default `fingerprint` cache policy       |
| `-- @nocache`                            | Always re-execute; wins over any `-- @cache` policy    |
| `-- @name <identifier>`                  | Name the output variable (default: `result`)          |
| `-- @after <cell-id>`                    | Add an ordering-only DAG edge to an upstream cell     |
| `-- @mount <name> <uri> ro`              | A mount a DuckDB connection's `mounts` can name       |

These are the only annotations a SQL cell acts on; validation warns about any other.
They are SQL comments, so a `#` line is not one (see [Writing them](https://bearingresearch.com/strata/dev/reference/annotations/sql/#writing-them)).

See [Cell Annotations][a] for the full reference.
