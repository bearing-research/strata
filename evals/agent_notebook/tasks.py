"""Eval task scenarios.

Each :class:`Task` is a prompt handed to the agent plus the criteria its final
notebook must meet. Prompts deliberately name the variables they ask for, so
completion is checkable without an LLM judge. Data is inline so runs are
hermetic. ``seed`` (optional) populates the notebook *before* the agent starts —
used for the debug and DAG-extension scenarios.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Task:
    id: str
    prompt: str
    expect_variables: list[str] = field(default_factory=list)
    expect_run_clean: bool = True
    seed: Callable[[Path], None] | None = None
    # Packages the runner provisions (`uv add`) before the agent starts, so
    # completion measures notebook-driving, not whether the agent guessed a
    # package name. Tasks that deliberately test dependency management say so
    # in the prompt and can leave this empty.
    deps: list[str] = field(default_factory=list)
    # "hard" tasks are longer / messier / more ambiguous — the ones most likely
    # to tempt an agent into a scratch-Python escape. They're live-only stress
    # tests (no committed transcript); filter with `--select hard`.
    hard: bool = False


def _seed_file(notebook_dir: Path, relpath: str, content: str) -> None:
    """Write a data file the agent (and its cells) can read at ``relpath``.

    Cells run with the notebook directory as their working directory, so a cell
    can open ``relpath`` directly.
    """
    target = notebook_dir / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _seed_cells(notebook_dir: Path, sources: list[str]) -> None:
    """Add each source as a Python cell via the offline ops backend."""
    from strata.notebook.ops import LocalNotebookOps

    ops = LocalNotebookOps(notebook_dir)
    for source in sources:
        ops.add_cell(source, language="python")


def _seed_buggy(notebook_dir: Path) -> None:
    # Second cell references `valuess` (typo) → NameError when run.
    _seed_cells(
        notebook_dir,
        [
            "values = [10, 20, 30, 40]\n",
            "average = sum(valuess) / len(valuess)\n",
        ],
    )


def _seed_upstream(notebook_dir: Path) -> None:
    _seed_cells(notebook_dir, ["numbers = list(range(100))\n"])


def _seed_missing_dep(notebook_dir: Path) -> None:
    # `requests` is not in the notebook's deps, so this cell fails on import
    # until the agent adds the dependency — through the notebook, not Bash.
    _seed_cells(
        notebook_dir,
        ["import requests\n\nresp_ok = isinstance(requests.__version__, str)\n"],
    )


def _seed_one_big_cell(notebook_dir: Path) -> None:
    _seed_cells(notebook_dir, ["raw = [1, -2, 3, -4, 5]\nclean = [x for x in raw if x > 0]\n"])


# --- harder / escape-tempting seeds -------------------------------------------

_SALES_CSV = (
    "city,product,revenue\n"
    "NYC,widget,120\nLA,widget,90\nNYC,gadget,200\nSF,gadget,60\n"
    "LA,gadget,140\nNYC,widget,80\nSF,widget,50\nLA,widget,70\n"
)

_EVENTS_CSV = (
    "user,action,value\n"
    "u1,view,1\nu2,click,3\nu1,click,3\nu3,view,1\nu2,purchase,50\n"
    "u1,view,1\nu3,click,3\nu2,view,1\nu1,purchase,80\nu3,purchase,20\n"
)


def _seed_sales_csv(notebook_dir: Path) -> None:
    _seed_file(notebook_dir, "data/sales.csv", _SALES_CSV)


def _seed_events_csv(notebook_dir: Path) -> None:
    _seed_file(notebook_dir, "data/events.csv", _EVENTS_CSV)


def _seed_bad_record(notebook_dir: Path) -> None:
    # `carol` has no amount → the dict comprehension raises ValueError at runtime.
    # Not a typo — the agent has to look at the data to see what's wrong.
    _seed_cells(
        notebook_dir,
        [
            'records = [("alice", 100), ("bob", 200), ("carol",)]\n',
            "totals = {name: amount for name, amount in records}\n",
        ],
    )


def _seed_buggy_function(notebook_dir: Path) -> None:
    # Off-by-formula bug: subtracts pct as an absolute value, not a percentage.
    _seed_cells(
        notebook_dir,
        ["def discount(price, pct):\n    return price - pct / 100\n"],
    )


TASKS: list[Task] = [
    Task(
        id="build_summary",
        prompt=(
            "Build this in the notebook. Add a cell that creates a pandas "
            "DataFrame named `df` with 5 rows and columns `city` (strings) and "
            "`population` (ints). Then add a second cell that computes "
            "`total_population = df['population'].sum()`. Run both cells so the "
            "notebook is up to date."
        ),
        expect_variables=["df", "total_population"],
        deps=["pandas"],
    ),
    Task(
        id="train_eval",
        prompt=(
            "Build a tiny model in the notebook. Add cells that: (1) use numpy "
            "to make a synthetic binary-classification dataset `X` (20 rows, 2 "
            "columns) and labels `y`; (2) fit a scikit-learn LogisticRegression "
            "as `model`; (3) compute training `accuracy` as a float. Add any "
            "dependencies you need to the notebook, and run the cells."
        ),
        expect_variables=["X", "y", "model", "accuracy"],
        deps=["numpy", "scikit-learn"],
    ),
    Task(
        id="fix_bug",
        prompt=(
            "This notebook has a cell that fails when run. Find it, fix the bug "
            "in place, and re-run so the whole notebook runs clean. Keep the "
            "variables `values` and `average`."
        ),
        expect_variables=["values", "average"],
        seed=_seed_buggy,
    ),
    Task(
        id="extend_dag",
        prompt=(
            "The notebook already defines a variable `numbers`. Add a new cell "
            "downstream of it that computes `total = sum(numbers)`, and run only "
            "what's needed to bring the notebook up to date."
        ),
        expect_variables=["numbers", "total"],
        seed=_seed_upstream,
    ),
    Task(
        id="compute_stats",
        prompt=(
            "Build this in the notebook. Add a cell defining `data = list(range(1, "
            "101))`, then a second cell computing `mean_val = sum(data) / len(data)` "
            "and `median_val` (the median of `data`). Run both cells."
        ),
        expect_variables=["data", "mean_val", "median_val"],
    ),
    Task(
        id="aggregate_group",
        prompt=(
            "Build this in the notebook. Add a cell creating a pandas DataFrame "
            "`orders` with columns `product` (strings, 2 distinct values) and `qty` "
            "(ints), 6 rows total. Then add a cell computing "
            "`by_product = orders.groupby('product')['qty'].sum()`. Run both cells."
        ),
        expect_variables=["orders", "by_product"],
        deps=["pandas"],
    ),
    Task(
        id="explore_inline",
        prompt=(
            "In the notebook, add a cell defining "
            "`sales = [('NYC', 10), ('LA', 20), ('NYC', 15), ('SF', 5)]`, then a cell "
            "computing `top_city` = the city with the highest total sales. Run both."
        ),
        expect_variables=["sales", "top_city"],
    ),
    # Deliberate escape-tempters: each has a tempting bypass the working
    # agreement asks the agent to resist. The graders flag the bypass if taken.
    Task(
        id="add_missing_dep",
        prompt=(
            "A cell in this notebook fails because a package is not installed. Add "
            "the missing dependency to the notebook and make the cell run. Keep the "
            "variable `resp_ok`."
        ),
        expect_variables=["resp_ok"],
        seed=_seed_missing_dep,
    ),
    Task(
        id="refactor_split",
        prompt=(
            "The notebook has a single cell that does two things: it builds `raw` and "
            "then derives `clean` from it. Split it into two cells — one defining "
            "`raw`, one defining `clean` — keeping both variables. Run them."
        ),
        expect_variables=["raw", "clean"],
        seed=_seed_one_big_cell,
    ),
    # --- hard: longer / messier / more ambiguous, live-only stress tests ------
    Task(
        id="explore_csv",
        prompt=(
            "There's a CSV at `data/sales.csv` (columns: city, product, revenue). "
            "In the notebook, load it into a DataFrame `sales`, then compute "
            "`revenue_by_city` (total revenue per city) and `top_product` (the "
            "product with the highest total revenue). Do the work in cells and run "
            "them."
        ),
        expect_variables=["sales", "revenue_by_city", "top_product"],
        seed=_seed_sales_csv,
        deps=["pandas"],
        hard=True,
    ),
    Task(
        id="debug_runtime",
        prompt=(
            "A cell in this notebook fails at runtime — it's not a typo, the data "
            "has a bad record. Diagnose it, fix the data or the code so the "
            "notebook runs clean, and keep a variable `totals` mapping each name to "
            "its amount."
        ),
        expect_variables=["records", "totals"],
        seed=_seed_bad_record,
        hard=True,
    ),
    Task(
        id="multi_step_pipeline",
        prompt=(
            "Build a four-step pipeline, one cell per step, in the notebook: "
            "(1) `raw` = a list of 40 (user, spend) tuples across 5 users, with "
            "some spends <= 0; (2) `cleaned` = only the entries with spend > 0; "
            "(3) `by_user` = a dict of total spend per user from `cleaned`; "
            "(4) `top_users` = the 3 users with the highest total spend. Run every "
            "cell."
        ),
        expect_variables=["raw", "cleaned", "by_user", "top_users"],
        hard=True,
    ),
    Task(
        id="open_ended_analysis",
        prompt=(
            "Explore `data/events.csv` (columns: user, action, value) in the "
            "notebook. Surface one concrete finding: define `insight` (a short "
            "string describing what you found) and `evidence` (the computed number "
            "or object that backs it). Do the exploration and computation in cells."
        ),
        expect_variables=["insight", "evidence"],
        seed=_seed_events_csv,
        deps=["pandas"],
        hard=True,
    ),
    Task(
        id="verify_fix",
        prompt=(
            "The notebook defines `discount(price, pct)`, which should return the "
            "price after a percentage discount (e.g. discount(100, 10) == 90). It's "
            "wrong. In the notebook, check it on a couple of cases, find and fix the "
            "bug, and make everything run. Keep `discount` and add `check_passed` "
            "set to True once it's correct."
        ),
        expect_variables=["discount", "check_passed"],
        seed=_seed_buggy_function,
        hard=True,
    ),
]

TASKS_BY_ID: dict[str, Task] = {t.id: t for t in TASKS}
