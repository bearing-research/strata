# Write back

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

Format: `# @write_table <var> <table> append|overwrite|merge|sync [on=<col>,...]`.
The table is one of the forms `@table` takes, written with the server's
catalogs and credentials (a catalog's `strata_credential` resolves against the
notebook's environment first). The mode is required:

- `append` adds the rows as a new snapshot;
- `overwrite` replaces the table's contents;
- `merge` upserts on the `on` columns: rows whose keys match are replaced, the
  rest are added;
- `sync` merges on the `on` columns and deletes every row whose key the variable
  no longer holds, so the table ends up holding what `overwrite` would leave, but
  data files with no changed row stay. A table Strata did not write is refused,
  as for `overwrite`.

For `merge` and `sync` each row needs one key of its own, so a null or a repeated
key fails the write. The change commits at once (as more than one snapshot, each
naming the run), and nothing is committed when nothing differs.

The variable must be an Arrow-representable table (a pandas, polars or pyarrow
frame). A new table is created from its schema; an existing one gains new
columns and widened types, and anything else is refused before writing.

**The write rides the run.** The server writes after the cell succeeds and
before its outputs are kept, so a write that fails fails the cell and keeps
nothing: the next run is not a cache hit and writes again. A cache hit writes
nothing, so running the same inputs again never appends twice; an explicit
rerun (↻) writes again. The exception is a table that no longer exists: the
hit writes the cell's stored output into it, without running the cell, and
leaves the tables still there alone. If that write fails, the cell fails with
the reason, and the next run tries it again. A catalog that can't be reached
leaves the hit standing and writes nothing. The target and the mode are part of the cell's
provenance, so changing either runs the cell again. Each snapshot's summary
names the run: the output's artifact id and version and its provenance hash,
as an export's does. A cell reading the table with `@table` goes out of date
after each write. With several `@write_table` lines the tables are written one
after another; if a later one fails, earlier ones keep their new snapshot and
the next run writes them again.

**Going back to earlier source** (or inputs) is a cache hit for an `append`
cell, which writes nothing. An `overwrite`, `merge` or `sync` cell runs again
instead, since the later run replaced, merged or deleted rows that only a write
puts back; the table then holds what the cell shows.

A `# @loop` cell writes its final values once, after the last iteration, and
not each iteration's. A `# @per_variant` cell writes once per variant into the
same table, so `overwrite` and `sync` are refused there (each variant would
remove the rows of the one before it); append, or merge on a column that tells
the variants apart. A [`# @each`](https://bearingresearch.com/strata/dev/reference/annotations/python/#each) cell takes only `sync`, and writes once.

A cell should not also read the table it writes with `@table`: its own write
changes the snapshot it read, so it would show out of date at once.

A table Strata did not create (another writer's) is never overwritten, and is
appended to or merged into only when the server sets
`STRATA_NOTEBOOK_TABLE_WRITES_FOREIGN=true`. Under principal auth the table ACL
applies, as for an export, and a table last written for another tenant is
refused. In service mode a write also needs what an export needs:
`STRATA_SERVICE_WRITES_ENABLED=true` and the `artifacts:write` scope. A catalog
whose `strata_credential` the server's
[`STRATA_NOTEBOOK_CREDENTIAL_WRITERS`](https://bearingresearch.com/strata/dev/reference/notebook-toml/#named-credentials)
does not let the run's principal write through is refused too. Each of these
fails the cell, naming why, before it runs (in a Run All batch, before its
outputs are kept).
