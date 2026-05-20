# S3 Mount — reading a public bucket from a notebook

Demonstrates Strata's **mount** feature: a notebook cell can read a
remote filesystem path as if it were local, with no custom code inside
the cell.

The example mounts a single file from NOAA's GSOD (Global Surface
Summary of Day) dataset — a public, anonymous-readable S3 bucket — and
runs a small pandas analysis over it.

## What it shows

- Declaring a mount in `notebook.toml` under `[[mounts]]`.
- Using fsspec `options` to authenticate (here, `anon = true` for a
  public bucket).
- Accessing the mounted path as a `pathlib.Path` inside a cell.
- Zero AWS credentials required — `anon = true` tells fsspec to skip
  the credential chain.

## Mount declaration

```toml
[[mounts]]
name = "jfk_weather"
uri = "s3://noaa-gsod-pds/2024/72503014732.csv"
mode = "ro"
options = { anon = true }
```

Inside a cell, `jfk_weather` is a `pathlib.Path`; the CSV is
materialized locally by fsspec on first read.

## Cells

| Cell | What it does |
|---|---|
| `load` | Reads the weather CSV into a DataFrame, parses dates, keeps the columns we need. |
| `summary` | Groups by month and aggregates avg / max / min / total-precip. |

## Running

From the project root:

```bash
uv run strata-notebook --host 127.0.0.1 --port 8765
```

Then open `examples/s3_mount` from the Strata home page.

## Swapping in a private bucket

Drop the `options = { anon = true }` line and configure AWS credentials
the normal way (`aws configure`, `AWS_PROFILE`, IAM role, etc.). fsspec
will pick them up automatically.
