"""Notebooks that ship in the wheel, so a first run needs no clone.

``examples/`` is the source of truth for ``samples/``, and the docs gallery's cards for
``sample_thumbnails/``; tests keep them equal.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any

from strata.notebook.writer import _apply_notebook_toml_update

SAMPLES_DIR = Path(__file__).parent / "samples"
# ``<name>-<theme>.webp``, the docs gallery's card for each sample, written by
# ``scripts/capture_docs_shots.py``.
THUMBNAILS_DIR = Path(__file__).parent / "sample_thumbnails"
THEMES = ("light", "dark")


@dataclass(frozen=True)
class Sample:
    """A bundled notebook. ``warm_imports`` is what its first run would spend time importing.

    ``first_run_seconds`` is how long Run All takes on a fresh copy, once it is open:
    measured, rounded up, on a 2-core Linux machine.
    """

    name: str
    title: str
    description: str
    warm_imports: tuple[str, ...]
    first_run_seconds: int


SAMPLES = (
    Sample(
        name="caching",
        title="Tutorial: what the cache saves",
        description="Two minutes. Run it twice, edit a cell, see what runs again.",
        warm_imports=("pyarrow", "pandas", "matplotlib.pyplot"),
        first_run_seconds=4,
    ),
    Sample(
        name="keeling_curve",
        title="The Keeling curve, with a receipt",
        description="CO₂ at Mauna Loa since 1958, and a published figure that shows its sources.",
        warm_imports=("pyarrow", "pandas", "matplotlib.pyplot"),
        first_run_seconds=2,
    ),
    Sample(
        name="pandas_basics",
        title="Pandas Basics",
        description="Create, filter, group and merge DataFrames.",
        warm_imports=("pyarrow", "pandas"),
        first_run_seconds=1,
    ),
)


def get_sample(name: str) -> Sample | None:
    return next((sample for sample in SAMPLES if sample.name == name), None)


def thumbnail_path(sample: Sample, theme: str) -> Path:
    return THUMBNAILS_DIR / f"{sample.name}-{theme}.webp"


def copy_sample(sample: Sample, root: Path) -> Path:
    """Copy ``sample`` into a new directory under ``root``, as a notebook of its own."""
    root.mkdir(parents=True, exist_ok=True)
    for number in count(1):
        suffix = "" if number == 1 else f"_{number}"
        dest = root / f"{sample.name}{suffix}"
        try:
            shutil.copytree(
                SAMPLES_DIR / sample.name, dest, ignore=shutil.ignore_patterns("__pycache__")
            )
        except FileExistsError:
            continue

        def own_identity(toml_data: dict[str, Any], number: int = number) -> bool:
            # Two copies with one id would collide in a shared store.
            toml_data["notebook_id"] = str(uuid.uuid4())
            if number > 1:
                toml_data["name"] = f"{sample.title} {number}"
            return True

        _apply_notebook_toml_update(dest, own_identity)
        return dest
    raise AssertionError("unreachable")
