# SQL annotations

A cell with `language = "sql"` runs a query through a declared connection.
See [SQL cells](https://bearingresearch.com/strata/dev/guides/cells/sql/#sql-cells) for the full feature walkthrough; this
section is the per-annotation reference.

## Writing them

A SQL cell's annotations are SQL comments: `-- @key value`, in the leading block of
`--` lines at the top of the cell. Blank lines and other `--` comments may sit in the
block, and the query starts at the first line that is neither. The file stays valid
SQL, so an editor or linter reads the `.sql` file as it is.

```sql
-- @name top_customers
-- @sql connection=warehouse
-- @cache forever
-- Reference data: it changes rarely.
SELECT customer, SUM(amount) AS total FROM orders GROUP BY customer
```

`#` is not a comment in SQL, so a `# @sql ...` line in a SQL cell is not an
annotation: it would reach the database as SQL. Validation reports it as the error
`sql_annotation_hash`, naming the line, and the cell does not run.

## What a SQL cell reads

| Annotation | What it does |
| --- | --- |
| [`-- @sql connection=<name> [write=true]`](#sql) | The connection the query runs on (required), and whether it may write. |
| [`-- @cache <policy>`](#cache) | The cache policy: `fingerprint` (the default), `forever`, `session`, `ttl=<seconds>` or `snapshot`. |
| [`-- @nocache`](https://bearingresearch.com/strata/dev/reference/annotations/python/#nocache) | Always run; wins over any `@cache` policy. |
| [`-- @name <identifier>`](#name) | The output variable (default `result`). |
| [`-- @after <cell-id>`](https://bearingresearch.com/strata/dev/reference/annotations/python/#after) | An ordering-only edge from another cell, such as the one that writes the database this cell reads. |
| [`-- @mount <name> <uri> ro`](https://bearingresearch.com/strata/dev/reference/annotations/python/#mount) | A mount a DuckDB connection's `mounts` can name, read-only. |

These six are the only ones a SQL cell acts on. Any other annotation, such as
`-- @worker` or `-- @timeout`, does nothing on a SQL cell, and validation warns
about it (`annotation_not_for_sql`). A SQL cell runs in the server, so it has no
worker, timeout or environment of its own.

## `@sql`

Marks the cell as SQL and binds it to a named connection.

```sql
-- @sql connection=warehouse
SELECT * FROM orders WHERE amount > :min_amount
```

Key/value parameters:

- `connection=<name>`, required. Must reference an entry under
  `[connections.<name>]` in `notebook.toml`. Manage these via the
  **Connections panel**, under **Notebook settings** in the sidebar; you don't need to edit the file
  directly.
- `write=true`, opt the cell into writable execution. Without this flag the
  cell may only read: `SELECT`, set operations, `VALUES`, `TABLE`,
  `SUMMARIZE`, `PIVOT`, `DESCRIBE`, `SHOW` and plain `EXPLAIN`. Anything else
  (DDL, DML, `COPY`, `ATTACH`, `USE`, `CALL`, `SET`, `PRAGMA`, and `EXPLAIN
  ANALYZE`, which runs the statement it describes) is refused before the
  statement reaches the driver, naming it. The connection is still opened
  read-only, but that is a second line rather than the boundary: the
  read-only transaction a driver opens is one a `COMMIT` in the cell body
  can end. With the flag the cell can run setup scripts (`CREATE TABLE`,
  `INSERT`, `DROP`). It is per-cell; read cells on the same connection stay
  read-only.

```sql
-- @sql connection=warehouse write=true
DROP TABLE IF EXISTS events;
CREATE TABLE events (id INTEGER PRIMARY KEY, label TEXT NOT NULL);
INSERT INTO events VALUES (1, 'alpha'), (2, 'beta');
```

Write cells split the body into individual statements via sqlglot, run each
in sequence, and emit a per-statement status table (`stmt`, `kind`,
`rows_affected`). Default cache policy for write cells is `session`;
`fingerprint` and `snapshot` error early because probe-based invalidation
has no anchor when the cell mutates state.

## `@cache`

Override the default `fingerprint` cache policy on a SQL cell.

| Policy            | Behavior                                                     |
| ----------------- | ------------------------------------------------------------ |
| `fingerprint`     | Default. Probe-derived freshness token + schema fingerprint folded into the hash. A reopen probes a SQLite or DuckDB file again and opens the cell ready if it is unchanged, stale if it changed; on other drivers the cell opens idle, because trusting the cache means asking the source. |
| `forever`         | Static salt; never invalidates from DB-side state.           |
| `session`         | Session-unique salt; invalidates across sessions.            |
| `ttl=<seconds>`   | `floor(now / ttl)` bucketed time-based salt.                 |
| `snapshot`        | The result is pinned to one queryable state of the warehouse. Snowflake, BigQuery, and DuckDB over a catalog; refused on other drivers. |

```sql
-- @sql connection=warehouse
-- @cache forever
SELECT * FROM dim_country
```

`-- @cache snapshot` needs a driver whose warehouse can query a table as it
stood at a moment in time. **Snowflake** (`AT (TIMESTAMP => ...)`) and
**BigQuery** (`FOR SYSTEM_TIME AS OF ...`) can, and so can a **DuckDB**
connection with a `catalog` (see below). SQLite, PostgreSQL and DuckDB without
a catalog cannot, and the cell is refused before any connection is opened.

```sql
-- @sql connection=warehouse
-- @cache snapshot
SELECT region, SUM(amount) AS revenue FROM orders GROUP BY region
```

The first run reads the warehouse clock and runs the query with every table
pinned to that moment. The timestamp is the cell's cache key and is recorded on
its artifact. Running the cell again finds it there and hits the cache, even
after new rows land, because the cell still names the same state. The
timestamp moves only when the cell changes (its query, binds, connection or
upstream inputs) or when you rerun it (`↻`), which pins to now.

Time travel only reaches back so far, so each run reports how long its state
stays queryable:

```
State as of 2026-09-15T10:05:00+00:00; queryable until 2026-09-16T10:05:00+00:00.
```

On Snowflake the horizon is the shortest `DATA_RETENTION_TIME_IN_DAYS` among
the tables the query reads. On BigQuery it is 48 hours, the shortest time travel
window a dataset can be configured with, since a dataset's own setting lives in
a region-scoped view the connection does not name. After the horizon, a rerun
works, but the recorded state can no longer be queried again. The horizon is
also on the [publication page](https://bearingresearch.com/strata/dev/notebook/publishing/) of a result built from the cell.

On a DuckDB connection over a [catalog](https://bearingresearch.com/strata/dev/guides/cells/sql/#duckdb-over-the-lake), the
state is the Iceberg snapshot of each catalog table the query reads. The first
run records those snapshot ids on its artifact and reads each table `AT
(VERSION => id)`; later runs read the same snapshots and hit the cache without
asking the catalog, so a hit works with the catalog unreachable, and a new
snapshot of a table does not make the cell out of date. Only a rerun (`↻`)
or a change to the cell moves it to the current snapshots. The run says which
snapshots it shows:

```
State of lake.taxi.trips at snapshot 5739646342007261951.
```

Every table the query reads must be one of the catalog's, since only those have
snapshots to read again: a query that reads a mount or a table in the
connection's own database file is refused, naming it. An Iceberg snapshot stays
queryable until the catalog's maintenance expires it, which Strata cannot see,
so no horizon is recorded. Per-driver
freshness probe details are in [SQL cells](https://bearingresearch.com/strata/dev/guides/cells/sql/#per-driver-freshness).

## `@name`

For SQL cells, `@name` sets the output variable name (default `result`),
the same way it does for prompt cells.

```sql
-- @sql connection=warehouse
-- @name top_customers
SELECT customer, SUM(amount) AS total
FROM orders GROUP BY customer ORDER BY total DESC LIMIT 5
```

A downstream Python cell can then reference `top_customers` directly as a
`pyarrow.Table`; call `.to_pandas()` for a pandas DataFrame.
