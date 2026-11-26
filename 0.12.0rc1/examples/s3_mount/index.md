# Read a NOAA weather file from a public S3 bucket

Demonstrates Strata's **mount** feature: a notebook cell can read a
remote filesystem path as if it were local, with no custom code inside
the cell.

The example mounts a single file from NOAA's GSOD (Global Surface
Summary of Day) dataset, a public, anonymous-readable S3 bucket, and
runs a small pandas analysis over it.

---

## Load JFK weather from S3 { .nb-cell }

```python
# @name Load JFK weather from S3
import pandas as pd

weather = pd.read_csv(jfk_weather / "72503014732.csv")
weather["DATE"] = pd.to_datetime(weather["DATE"])
weather = weather[["DATE", "NAME", "TEMP", "MAX", "MIN", "PRCP"]]
weather
```

**Output**

| DATE | NAME | TEMP | MAX | MIN | PRCP |
| --- | --- | --- | --- | --- | --- |
| 2024-01-01 00:00:00 | LAGUARDIA AIRPORT, NY US | 43 | 46.9 | 39.9 | 0 |
| 2024-01-02 00:00:00 | LAGUARDIA AIRPORT, NY US | 36.7 | 46.9 | 30 | 0.01 |
| 2024-01-03 00:00:00 | LAGUARDIA AIRPORT, NY US | 39.3 | 43 | 30 | 0 |
| 2024-01-04 00:00:00 | LAGUARDIA AIRPORT, NY US | 40.7 | 45 | 35.1 | 0 |
| 2024-01-05 00:00:00 | LAGUARDIA AIRPORT, NY US | 32.1 | 45 | 27 | 0 |
| 2024-01-06 00:00:00 | LAGUARDIA AIRPORT, NY US | 34.8 | 37.9 | 27 | 0 |
| 2024-01-07 00:00:00 | LAGUARDIA AIRPORT, NY US | 36.9 | 39.2 | 34 | 0.39 |
| 2024-01-08 00:00:00 | LAGUARDIA AIRPORT, NY US | 39.3 | 45 | 35.1 | 0.12 |

*…showing 8 of 366 rows*

## Monthly weather summary { .nb-cell }

```python
# @name Monthly weather summary
monthly = (
    weather.assign(month=weather["DATE"].dt.to_period("M").astype(str))
    .groupby("month", as_index=False)
    .agg(
        avg_temp=("TEMP", "mean"),
        max_temp=("MAX", "max"),
        min_temp=("MIN", "min"),
        total_precip=("PRCP", "sum"),
    )
    .round(2)
)
monthly
```

**Output**

| month | avg_temp | max_temp | min_temp | total_precip |
| --- | --- | --- | --- | --- |
| 2024-01 | 37.07 | 60.1 | 19 | 6.2 |
| 2024-02 | 39.3 | 60.1 | 24.1 | 1.92 |
| 2024-03 | 46.47 | 72 | 28.9 | 9.74 |
| 2024-04 | 52.9 | 84.9 | 37 | 3.82 |
| 2024-05 | 63.79 | 86 | 48.9 | 2.97 |
| 2024-06 | 75.44 | 100.9 | 61 | 1.8 |
| 2024-07 | 79.2 | 97 | 64.9 | 3.9 |
| 2024-08 | 76.12 | 96.1 | 59 | 7.92 |

*…showing 8 of 12 rows*

---

## About this example

### What it shows

- Declaring a mount in `notebook.toml` under `[[mounts]]`.
- Using fsspec `options` to authenticate (here, `anon = true` for a
  public bucket).
- Accessing the mounted path as a `pathlib.Path` inside a cell.
- Zero AWS credentials required, `anon = true` tells fsspec to skip
  the credential chain.

### Mount declaration

```toml
[[mounts]]
name = "jfk_weather"
uri = "s3://noaa-gsod-pds/2024/72503014732.csv"
mode = "ro"
options = { anon = true }
```

Inside a cell, `jfk_weather` is a `pathlib.Path` to a local directory
that holds the mounted file, so `load` reads
`jfk_weather / "72503014732.csv"`; the CSV is materialized locally by
fsspec on first read.

### Cells

| Cell | What it does |
|---|---|
| `load` | Reads the weather CSV into a DataFrame, parses dates, keeps the columns we need. |
| `summary` | Groups by month and aggregates avg / max / min / total-precip. |

### Running

From the project root:

```bash
uv run strata-notebook --notebook-dir examples
```

Then open **S3 Mount (Public Bucket)** from the notebook list on the Strata home page.

Strata reads the bucket in its own process, so an installed Strata
needs the `s3` extra (`s3fs`): `uv tool install "strata-notebook[s3]"`.
The development environment already has it.

### Swapping in a private bucket

Drop the `options = { anon = true }` line and configure AWS credentials
the normal way (`aws configure`, `AWS_PROFILE`, IAM role, etc.). fsspec
will pick them up automatically.
