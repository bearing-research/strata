"""The order events."""

import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

EVENTS = Path(__file__).resolve().parents[2] / "data" / "events.parquet"


def load_events() -> pd.DataFrame:
    """Every order event: event_id, day, region, product, customer_id, amount,
    latency_ms, status."""
    started = time.monotonic()
    frame = pd.read_parquet(EVENTS)
    _record()
    # The warehouse export this stands in for takes as long, every time.
    time.sleep(
        max(0.0, float(os.environ.get("HARBOR_LOAD_SECONDS", "20")) - (time.monotonic() - started))
    )
    return frame


def _record() -> None:
    log = os.environ.get("HARBOR_LOAD_LOG")
    if not log:
        return
    line = {
        "at": time.time(),
        "pid": os.getpid(),
        "cell": os.environ.get("STRATA_CELL_ID"),
        "argv": sys.argv[:1],
    }
    with open(log, "a") as out:
        out.write(json.dumps(line) + "\n")
