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
]

TASKS_BY_ID: dict[str, Task] = {t.id: t for t in TASKS}
