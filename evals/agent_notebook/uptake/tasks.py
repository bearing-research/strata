"""The uptake task bank (design-agent-uptake §5): 29 un-primed tasks in the ``harbor``
project, of five kinds, and the holdout split tuning never sees.

A prompt never says notebook, cell or scratchpad. Each question carries how its answer
is checked: a number or a word in the agent's reply, computed from the workspace's own
data and code, or a check of the files the agent left. An iteration task's turns are
sent one at a time, each after the previous answer.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date
from functools import cached_property
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from . import workspace as ws

ITERATION, PROBING, PLOT, NEUTRAL, MUST_NOT = "iteration", "probing", "plot", "neutral", "must-not"
KINDS = (ITERATION, PROBING, PLOT, NEUTRAL, MUST_NOT)
# Should-use kinds: the scratchpad is the better tool. A must-not task's run is a false
# positive; a neutral one may go either way.
SHOULD_USE = (ITERATION, PROBING, PLOT)


class Context:
    """What an answer is computed from: one workspace's events, and harbor's code as
    the template has it (an agent may have edited the copy)."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    @cached_property
    def events(self) -> pd.DataFrame:
        return ws.read_events(self.workspace)

    @staticmethod
    def module(name: str) -> ModuleType:
        for path in (ws.TEMPLATE / "src", ws.TEMPLATE / "packages" / "fleet" / "src"):
            if str(path) not in sys.path:
                sys.path.append(str(path))
        return importlib.import_module(name)


# --- how a reply is checked ---------------------------------------------------------

_NUMBER = re.compile(
    r"(?<![\w.])(-?\d[\d,]*(?:\.\d+)?(?:[eE]-?\d+)?)"
    r"(?:\s*(million|thousand|billion|bn|[MkKB])\b)?"
)
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "m": 1e6, "billion": 1e9, "b": 1e9, "bn": 1e9}


def numbers_in(text: str) -> list[float]:
    """Every number in *text*, with "8.58 million" or "12k" scaled; each also as written."""
    found = []
    for digits, scale in _NUMBER.findall(text):
        try:
            value = float(digits.replace(",", ""))
        except ValueError:
            continue
        found.append(value)
        if scale:
            found.append(value * _SCALE[scale.lower()])
    return found


@dataclass(frozen=True)
class Number:
    """A number in the reply within tolerance; a *share* may also be given in percent."""

    value: float
    rel: float = 0.01
    abs: float = 0.0
    share: bool = False

    def passes(self, text: str, workspace: Path) -> bool:
        targets = [self.value] + ([self.value * 100] if self.share else [])
        return any(
            math.isclose(n, t, rel_tol=self.rel, abs_tol=self.abs * (100 if i else 1))
            for n in numbers_in(text)
            for i, t in enumerate(targets)
        )


@dataclass(frozen=True)
class Word:
    """Any of *options* in the reply, as a whole word, ignoring case; with *every*, all.
    A word may carry an ordinal or plural ending: "17th", "Tuesdays"."""

    options: tuple[str, ...]
    every: bool = False

    def passes(self, text: str, workspace: Path) -> bool:
        def said(word: str) -> bool:
            pattern = rf"(?<![\w-]){re.escape(word)}(?:st|nd|rd|th|s|'s)?(?![\w-])"
            return re.search(pattern, text, re.I) is not None

        return (all if self.every else any)(said(w) for w in self.options)


@dataclass(frozen=True)
class Files:
    """What the agent left in the workspace passes *check*."""

    check: Callable[[Path], bool]

    def passes(self, text: str, workspace: Path) -> bool:
        return self.check(workspace)


Expected = Number | Word | Files


@dataclass(frozen=True)
class Question:
    prompt: str
    # The expected answer, computed for a workspace.
    expect: Callable[[Context], Expected]


@dataclass(frozen=True)
class Task:
    id: str
    kind: str
    turns: tuple[Question, ...]
    # Whether the task's slow step is ``load_events()``: its trial pays off when that ran once.
    loads: bool = field(default=False)

    def expected(self, workspace: Path) -> list[Expected]:
        context = Context(workspace)
        return [question.expect(context) for question in self.turns]


def _project_python(workspace: Path, *args: str) -> bool:
    """Whether a command in the workspace's code (eval interpreter, its sources on the
    path) exits 0."""
    sources = [workspace / "src", workspace / "packages" / "fleet" / "src"]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, sources))}
    done = subprocess.run(
        [sys.executable, *args], cwd=workspace, env=env, capture_output=True, timeout=300
    )
    return done.returncode == 0


def _pytest(workspace: Path, *paths: str) -> bool:
    return _project_python(workspace, "-m", "pytest", "-q", "-p", "no:cacheprovider", *paths)


def _config(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / "data" / "config.json").read_text())


# --- the events ---------------------------------------------------------------------


def _month(frame: pd.DataFrame, month: int) -> pd.DataFrame:
    return frame[pd.to_datetime(frame["day"]).dt.month == month]


def _date_words(day: date) -> tuple[str, ...]:
    return (
        day.isoformat(),
        f"{day:%B} {day.day}",
        f"{day:%b} {day.day}",
        f"{day.day} {day:%B}",
        f"{day.day} {day:%b}",
    )


def _p95(values: pd.Series) -> float:
    return float(np.percentile(values, 95))


def _heavy(c: Context) -> int:
    return int(c.events["customer_id"].value_counts().idxmax())


def _top_product(c: Context) -> str:
    return str(c.events["product"].value_counts().idxmax())


def _weekday(c: Context) -> str:
    counts = pd.to_datetime(c.events["day"]).dt.day_name().value_counts()
    return str(counts.idxmax())


LOAD = "Load the order events with `harbor.data.load_events()`"

ITERATION_TASKS = (
    Task(
        "iter_west",
        ITERATION,
        (
            Question(
                f"{LOAD} and tell me how many events were in the west region.",
                lambda c: Number(int((c.events["region"] == "west").sum()), rel=0),
            ),
            Question(
                "Of the west's events with status error, what is the median latency_ms?",
                lambda c: Number(
                    float(
                        c.events.query("region == 'west' and status == 'error'")[
                            "latency_ms"
                        ].median()
                    ),
                    rel=0.001,
                ),
            ),
            Question(
                "Which product had the highest total amount in February?",
                lambda c: Word(
                    (str(_month(c.events, 2).groupby("product")["amount"].sum().idxmax()),)
                ),
            ),
            Question(
                "What share of customers placed more than 30 events?",
                lambda c: Number(
                    float((c.events["customer_id"].value_counts() > 30).mean()),
                    rel=0.02,
                    share=True,
                ),
            ),
        ),
        loads=True,
    ),
    Task(
        "iter_january",
        ITERATION,
        (
            Question(
                f"{LOAD}. What was the total amount in January?",
                lambda c: Number(float(_month(c.events, 1)["amount"].sum()), rel=0.001),
            ),
            Question(
                "Which date had the most events?",
                lambda c: Word(
                    _date_words(pd.Timestamp(c.events["day"].value_counts().idxmax()).date())
                ),
            ),
            Question(
                "What fraction of central's events have status error?",
                lambda c: Number(
                    float((c.events.query("region == 'central'")["status"] == "error").mean()),
                    rel=0.02,
                    share=True,
                ),
            ),
            Question(
                "What is the mean amount of the events that timed out?",
                lambda c: Number(
                    float(c.events.query("status == 'timeout'")["amount"].mean()), rel=0.002
                ),
            ),
        ),
        loads=True,
    ),
    Task(
        "iter_customers",
        ITERATION,
        (
            Question(
                f"{LOAD}. How many distinct customers are there?",
                lambda c: Number(int(c.events["customer_id"].nunique()), rel=0),
            ),
            Question(
                "Which customer placed the most events?",
                lambda c: Number(_heavy(c), rel=0),
            ),
            Question(
                "What is that customer's total amount?",
                lambda c: Number(
                    float(c.events[c.events["customer_id"] == _heavy(c)]["amount"].sum()),
                    rel=0.001,
                ),
            ),
            Question(
                "And which product did they order most often?",
                lambda c: Word(
                    (
                        str(
                            c.events[c.events["customer_id"] == _heavy(c)]["product"]
                            .value_counts()
                            .idxmax()
                        ),
                    )
                ),
            ),
        ),
        loads=True,
    ),
    Task(
        "iter_latency",
        ITERATION,
        (
            Question(
                f"{LOAD}. What is the 95th percentile of latency_ms over all events?",
                lambda c: Number(_p95(c.events["latency_ms"]), rel=0.01),
            ),
            Question(
                "And in the east region?",
                lambda c: Number(_p95(c.events.query("region == 'east'")["latency_ms"]), rel=0.01),
            ),
            Question(
                "Which region has the highest 95th percentile latency?",
                lambda c: Word(
                    (str(c.events.groupby("region")["latency_ms"].quantile(0.95).idxmax()),)
                ),
            ),
            Question(
                "How many events took longer than 1000 ms?",
                lambda c: Number(int((c.events["latency_ms"] > 1000).sum()), rel=0),
            ),
        ),
        loads=True,
    ),
    Task(
        "iter_top_product",
        ITERATION,
        (
            Question(
                f"{LOAD}. Which product has the most events?",
                lambda c: Word((_top_product(c),)),
            ),
            Question(
                "What is that product's median amount?",
                lambda c: Number(
                    float(c.events[c.events["product"] == _top_product(c)]["amount"].median()),
                    rel=0.002,
                ),
            ),
            Question(
                "How many of its events have status error?",
                lambda c: Number(
                    int(
                        c.events.query(
                            f"product == '{_top_product(c)}' and status == 'error'"
                        ).shape[0]
                    ),
                    rel=0,
                ),
            ),
            Question(
                "In which region does it have the most events?",
                lambda c: Word(
                    (
                        str(
                            c.events[c.events["product"] == _top_product(c)]["region"]
                            .value_counts()
                            .idxmax()
                        ),
                    )
                ),
            ),
        ),
        loads=True,
    ),
    Task(
        "iter_calendar",
        ITERATION,
        (
            Question(
                f"{LOAD}. How many events were on 2026-03-01?",
                lambda c: Number(int((c.events["day"] == date(2026, 3, 1)).sum()), rel=0),
            ),
            Question("Which day of the week is busiest overall?", lambda c: Word((_weekday(c),))),
            Question(
                "What was the average daily total amount in March?",
                lambda c: Number(float(_month(c.events, 3)["amount"].sum()) / 31, rel=0.002),
            ),
            Question(
                "On how many days were there more than 12,000 events?",
                lambda c: Number(int((c.events["day"].value_counts() > 12_000).sum()), rel=0),
            ),
        ),
        loads=True,
    ),
)


def _cheapest_region(c: Context) -> Word:
    quote = c.module("harbor.pricing").quote
    return Word((min(ws.REGIONS, key=lambda region: quote("p07", 10, region)),))


def _riskiest(c: Context) -> Number:
    score = c.module("harbor.scoring").risk_score
    return Number(max(range(100, 121), key=score), rel=0)


def _first_over(c: Context) -> Number:
    quote = c.module("harbor.pricing").quote
    return Number(next(q for q in range(1, 10_000) if quote("p03", q, "north") > 500), rel=0)


def _farthest(c: Context) -> Word:
    distance = c.module("fleet.geo").distance
    pairs = [(a, b) for i, a in enumerate(ws.REGIONS) for b in ws.REGIONS[i + 1 :]]
    return Word(max(pairs, key=lambda pair: distance(*pair)), every=True)


def _widest_gap(c: Context) -> Word:
    quote = c.module("harbor.pricing").quote
    products = [f"p{i:02d}" for i in range(10)]
    return Word((max(products, key=lambda p: quote(p, 5, "west") - quote(p, 5, "east")),))


PROBING_TASKS = (
    Task(
        "probe_cheapest_region",
        PROBING,
        (
            Question(
                "Using `harbor.pricing.quote`, which region is cheapest for 10 units of p07?",
                _cheapest_region,
            ),
        ),
    ),
    Task(
        "probe_riskiest_customer",
        PROBING,
        (
            Question(
                "Among customers 100 to 120, which has the highest `harbor.scoring.risk_score`?",
                _riskiest,
            ),
        ),
    ),
    Task(
        "probe_quote_threshold",
        PROBING,
        (
            Question(
                "What is the smallest quantity of p03 whose `harbor.pricing.quote` in the north"
                " is over 500?",
                _first_over,
            ),
        ),
    ),
    Task(
        "probe_farthest_regions",
        PROBING,
        (Question("Which two regions are farthest apart, by `fleet.geo.distance`?", _farthest),),
    ),
    Task(
        "probe_quote_gap",
        PROBING,
        (
            Question(
                "Of products p00 to p09, which has the largest gap between its west and east"
                " `harbor.pricing.quote` for 5 units?",
                _widest_gap,
            ),
        ),
    ),
)


def _latency_peak(c: Context) -> Number:
    counts, edges = np.histogram(c.events["latency_ms"], bins=np.arange(0, 3000, 25))
    peak = int(np.argmax(counts))
    return Number(float(edges[peak] + 12.5), rel=0, abs=75)


def _worst_error_day(c: Context) -> Word:
    rate = (c.events["status"] == "error").groupby(c.events["day"]).mean()
    return Word(_date_words(pd.Timestamp(rate.idxmax()).date()))


PLOT_TASKS = (
    Task(
        "plot_daily_events",
        PLOT,
        (
            Question(
                f"{LOAD}, plot the number of events per day, and tell me which day of the week"
                " is busiest.",
                lambda c: Word((_weekday(c),)),
            ),
        ),
        loads=True,
    ),
    Task(
        "plot_latency_histogram",
        PLOT,
        (
            Question(
                f"{LOAD} and plot a histogram of latency_ms. Roughly where does it peak, in ms?",
                _latency_peak,
            ),
        ),
        loads=True,
    ),
    Task(
        "plot_amount_by_region",
        PLOT,
        (
            Question(
                f"{LOAD} and draw a box plot of amount by region. Which region has the highest"
                " median amount?",
                lambda c: Word((str(c.events.groupby("region")["amount"].median().idxmax()),)),
            ),
        ),
        loads=True,
    ),
    Task(
        "plot_error_rate",
        PLOT,
        (
            Question(
                f"{LOAD} and plot the daily error rate. On which date was it highest?",
                _worst_error_day,
            ),
        ),
        loads=True,
    ),
)

NEUTRAL_TASKS = (
    Task(
        "neutral_base_price",
        NEUTRAL,
        (
            Question(
                "What is `harbor.pricing.BASE_PRICES['p05']`?",
                lambda c: Number(c.module("harbor.pricing").BASE_PRICES["p05"], rel=0),
            ),
        ),
    ),
    Task(
        "neutral_region_count",
        NEUTRAL,
        (
            Question(
                "How many regions does `fleet.rates` define a multiplier for?",
                lambda c: Number(len(c.module("fleet.rates").MULTIPLIERS), rel=0),
            ),
        ),
    ),
    Task(
        "neutral_weekday",
        NEUTRAL,
        (
            Question(
                "What day of the week was 2026-03-14?",
                lambda c: Word((date(2026, 3, 14).strftime("%A"),)),
            ),
        ),
    ),
    Task(
        "neutral_eta",
        NEUTRAL,
        (
            Question(
                "What does `harbor.shipping.eta_days('north', 'south')` return?",
                lambda c: Number(c.module("harbor.shipping").eta_days("north", "south"), rel=0),
            ),
        ),
    ),
    Task(
        "neutral_modpow",
        NEUTRAL,
        (Question("What is 3**100 mod 1009?", lambda c: Number(pow(3, 100, 1009), rel=0)),),
    ),
    Task(
        "neutral_distance",
        NEUTRAL,
        (
            Question(
                "What does `fleet.geo.distance('east', 'west')` return?",
                lambda c: Number(c.module("fleet.geo").distance("east", "west"), rel=0, abs=0.05),
            ),
        ),
    ),
)


def _renamed(workspace: Path) -> bool:
    code = (
        "from harbor.shipping import delivery_days; import harbor.shipping as s;"
        " assert not hasattr(s, 'eta_days')"
    )
    # An alias left behind fails the hasattr check; a caller left unrenamed, the test.
    return _project_python(workspace, "-c", code) and _pytest(workspace, "tests/test_shipping.py")


def _km_to_miles(workspace: Path) -> bool:
    code = "from fleet.units import km_to_miles; assert abs(km_to_miles(10) - 6.21371) < 1e-3"
    return _project_python(workspace, "-c", code)


def _timeout_30(workspace: Path) -> bool:
    config = _config(workspace)
    return config.get("timeout_s") == 30 and config.get("owner") == "logistics-platform"


def _version_bumped(workspace: Path) -> bool:
    meta = tomllib.loads((workspace / "pyproject.toml").read_text())
    return meta["project"]["version"] == "0.2.0"


def _geo_test(workspace: Path) -> bool:
    test = workspace / "tests" / "test_geo.py"
    return test.is_file() and "distance" in test.read_text() and _pytest(workspace, str(test))


MUST_NOT_TASKS = (
    Task(
        "edit_fix_slugify",
        MUST_NOT,
        (
            Question(
                "The test in tests/test_text.py fails. Fix `harbor.text.slugify` so it passes.",
                lambda c: Files(lambda w: _pytest(w, "tests/test_text.py")),
            ),
        ),
    ),
    Task(
        "edit_rename_eta",
        MUST_NOT,
        (
            Question(
                "Rename `harbor.shipping.eta_days` to `delivery_days`, everywhere it is used.",
                lambda c: Files(_renamed),
            ),
        ),
    ),
    Task(
        "edit_add_km_to_miles",
        MUST_NOT,
        (
            Question(
                "Add a `km_to_miles(km)` function to `fleet.units`.",
                lambda c: Files(_km_to_miles),
            ),
        ),
    ),
    Task(
        "json_config_regions",
        MUST_NOT,
        (
            Question(
                "Which regions are listed in data/config.json?",
                lambda c: Word(tuple(_config(c.workspace)["regions"]), every=True),
            ),
        ),
    ),
    Task(
        "edit_config_timeout",
        MUST_NOT,
        (Question("Set `timeout_s` in data/config.json to 30.", lambda c: Files(_timeout_30)),),
    ),
    Task(
        "edit_bump_version",
        MUST_NOT,
        (
            Question(
                "Bump harbor's version in pyproject.toml to 0.2.0.",
                lambda c: Files(_version_bumped),
            ),
        ),
    ),
    Task(
        "edit_geo_test",
        MUST_NOT,
        (
            Question(
                "Write a test in tests/test_geo.py checking that `fleet.geo.distance` is"
                " symmetric.",
                lambda c: Files(_geo_test),
            ),
        ),
    ),
    Task(
        "json_alert_threshold",
        MUST_NOT,
        (
            Question(
                "What is the error_rate alert threshold in data/config.json?",
                lambda c: Number(_config(c.workspace)["alerts"]["error_rate"], rel=0, share=True),
            ),
        ),
    ),
)

TASKS = ITERATION_TASKS + PROBING_TASKS + PLOT_TASKS + NEUTRAL_TASKS + MUST_NOT_TASKS
TASKS_BY_ID = {task.id: task for task in TASKS}

# One third of each kind, never seen by tuning (§5). Committed before any arm-C run; a
# test checks it is the seeded split below.
HOLDOUT_SEED = 20261112
HOLDOUT_COUNTS = {ITERATION: 2, PROBING: 2, PLOT: 1, NEUTRAL: 2, MUST_NOT: 3}
HOLDOUT = frozenset(
    {
        "iter_customers",
        "iter_top_product",
        "probe_farthest_regions",
        "probe_quote_threshold",
        "plot_daily_events",
        "neutral_base_price",
        "neutral_region_count",
        "edit_config_timeout",
        "edit_fix_slugify",
        "json_config_regions",
    }
)


def seeded_holdout(seed: int = HOLDOUT_SEED) -> frozenset[str]:
    """The split :data:`HOLDOUT` was drawn as."""
    rng = random.Random(seed)
    chosen: set[str] = set()
    for kind in KINDS:
        ids = sorted(task.id for task in TASKS if task.kind == kind)
        chosen.update(rng.sample(ids, HOLDOUT_COUNTS[kind]))
    return frozenset(chosen)


def split(holdout: bool) -> tuple[Task, ...]:
    return tuple(task for task in TASKS if (task.id in HOLDOUT) == holdout)
