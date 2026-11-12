"""The seeded project the uptake tasks run in: ``harbor``, a uv workspace with an editable
sibling (``packages/fleet``) and a 1M-row parquet of order events.

The events are generated from a seed, not committed. Their shape gives each question
one answer: weekday traffic peaks on Tuesday, the west's amounts run highest and the
east's latency slowest, errors spike on 2026-02-17, one customer (4242) orders far more
than any other, and product popularity falls off as a power law.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

TEMPLATE = Path(__file__).parent / "template"
REGIONS = ("north", "south", "east", "west", "central")
PRODUCTS = tuple(f"p{i:02d}" for i in range(50))
HEAVY_CUSTOMER = 4242
SPIKE_DAY = "2026-02-17"


def events(rows: int = 1_000_000, seed: int = 0) -> pd.DataFrame:
    """The order events, the same for the same *rows* and *seed*."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2026-01-01", "2026-03-31", freq="D")
    weekday = np.array([1.0, 1.3, 1.05, 1.0, 0.95, 0.7, 0.6])[days.day_of_week]
    day = rng.choice(len(days), size=rows, p=weekday / weekday.sum())
    region = rng.choice(len(REGIONS), size=rows, p=[0.24, 0.2, 0.22, 0.18, 0.16])
    popularity = 1 / np.arange(1, len(PRODUCTS) + 1) ** 1.1
    product = rng.choice(len(PRODUCTS), size=rows, p=popularity / popularity.sum())
    customer = rng.integers(0, 50_000, size=rows)
    customer[rng.random(rows) < 0.002] = HEAVY_CUSTOMER
    amount_shift = np.array([0.0, -0.1, 0.05, 0.25, 0.0])[region]
    amount = np.round(rng.lognormal(3.0 + amount_shift, 0.6), 2)
    latency_scale = np.array([120.0, 120.0, 190.0, 120.0, 110.0])[region]
    latency = rng.gamma(2.0, latency_scale).astype(np.int64)
    spike = days[day] == pd.Timestamp(SPIKE_DAY)
    draw = rng.random(rows)
    error_p = np.where(spike, 0.12, 0.04)
    status = np.where(draw < error_p, "error", np.where(draw < error_p + 0.01, "timeout", "ok"))
    return pd.DataFrame(
        {
            "event_id": np.arange(rows, dtype=np.int64),
            "day": days[day].date,
            "region": np.asarray(REGIONS)[region],
            "product": np.asarray(PRODUCTS)[product],
            "customer_id": customer,
            "amount": amount,
            "latency_ms": latency,
            "status": status,
        }
    )


def build(dest: Path, *, rows: int = 1_000_000, seed: int = 0) -> Path:
    """A fresh copy of the project at *dest*, with its events written."""
    shutil.copytree(TEMPLATE, dest, ignore=shutil.ignore_patterns("__pycache__"))
    table = pa.Table.from_pandas(events(rows, seed), preserve_index=False)
    pq.write_table(table, dest / "data" / "events.parquet")
    return dest


def read_events(workspace: Path) -> pd.DataFrame:
    return pd.read_parquet(workspace / "data" / "events.parquet")
