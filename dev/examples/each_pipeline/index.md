# Pipeline over files: `@each` and a synced table

A cell that runs once per file, caches each file separately, and keeps an Iceberg
table in step with the results. It reads a few hundred real files from a public
bucket and writes to a real warehouse: an Iceberg REST catalog over an object
store, running on your machine.

The data is NOAA's daily weather for 2024, one CSV per station. The bucket has
12,159 files, and the notebook reads the 287 that `glob=725*.csv` names.

## What it shows

- `# @each path in gsod glob=725*.csv` runs the cell once per file, and a remote
  mount downloads only the files the pattern names.
- Each file's result is cached on its own, so a changed file runs alone.
- `# @write_table summary lake:gsod.stations sync on=station` makes the table hold
  what the cell produced, in one commit per run. A file that goes away takes its
  row with it, and a run that changes nothing commits nothing.
- A file the cell can't read fails that item alone. The other items stay cached, and
  the table stays as it was, since a missing row would read as a deletion.
- A SQL cell reads the table back through the catalog, at the snapshot the cell's
  key holds. It never touches the bucket: a notebook-level mount reaches only the
  cells that read it.

## Running

You need Docker, and [uv](https://docs.astral.sh/uv/). From the project root:

```bash
docker compose -f examples/each_pipeline/docker-compose.yml up -d --wait catalog
source examples/each_pipeline/lake.env
uv run strata run examples/each_pipeline
```

The first run downloads the 287 files, about 25 MB, and takes a few minutes. `strata
run` says where it is as it goes. Run it again and nothing runs: the summary cell is
one lookup, after listing the bucket to see that nothing changed.

Strata reads the bucket in its own process, so an installed Strata needs the `s3`
extra (`s3fs`): `uv tool install "strata-notebook[s3]"`. The development environment
already has it.

## Try it

Change `glob=725*.csv` to `glob=7250*.csv` in `cells/summarize.py` and run again. The
table goes from 287 rows to 41 in one new snapshot, and the SQL cell runs again because
the table changed. Change it back and all 287 return. No item runs either time.

## Using your own warehouse

The notebook names the table `lake:gsod.stations`, and `lake` is defined on the
server, by `STRATA_CATALOGS`. Nothing in the notebook changes when `lake` is your
organization's catalog instead of the one in `docker-compose.yml`:

```bash
export STRATA_CATALOGS='{"lake": {"type": "rest", "uri": "https://catalog.example", "warehouse": "prod", "token": "..."}}'
```

Glue and SQL catalogs work the same way for the write. To keep the keys out of the
setting, see
[named credentials](https://bearingresearch.com/strata/latest/reference/notebook-toml/#named-credentials).
`sync` refuses a table Strata did not write, so point it at a new table or one this
notebook owns.

To stop the warehouse and delete its data:

```bash
docker compose -f examples/each_pipeline/docker-compose.yml down -v
```

## Summarize each station

<sub>**kind** python</sub>

```python
# @name Summarize each station
# @each path in gsod glob=725*.csv
# @write_table summary lake:gsod.stations sync on=station
import pandas as pd
import pyarrow as pa

df = pd.read_csv(path, dtype={"STATION": str})
# GSOD writes a missing reading as 9999.9.
measured = df[df["MAX"] < 999]
summary = pa.table(
    {
        "station": [df["STATION"].iloc[0]],
        "name": [str(df["NAME"].iloc[0])],
        "days": [len(df)],
        "mean_temp_f": [float(df["TEMP"].mean())],
        "hottest_f": [float(measured["MAX"].max())],
    }
)
```

## Hottest stations

<sub>**kind** sql</sub>

```sql
-- @name Hottest stations
-- @sql connection=lake
-- @after summarize
SELECT station, name, days, hottest_f
FROM lake.gsod.stations
ORDER BY hottest_f DESC
LIMIT 10
```
