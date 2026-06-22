"""Keep the pandas_basics example's cell unit tests green.

The example doubles as documentation for the cell-test feature, so CI runs the
shipped ``cells/<id>.test.py`` files through the real cell-test runner with the
same upstream inputs the DAG would supply.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from strata.notebook.cell_test_runner import run_cell_tests_in_dir

_CELLS = Path(__file__).parents[2] / "examples" / "pandas_basics" / "cells"


def _src(file_stem: str) -> str:
    return (_CELLS / f"{file_stem}.py").read_text()


@pytest.fixture(scope="module")
def upstream_sales():
    """Build the `sales` frame as create-data → add-columns would."""
    ns0: dict = {}
    exec(_src("create_data"), ns0)  # noqa: S102 — trusted example source
    sales_raw = ns0["sales"]
    ns1 = {"sales": sales_raw.copy()}
    exec(_src("add_columns"), ns1)  # noqa: S102 — trusted example source
    return sales_raw, ns1["sales"]


def test_pandas_basics_cell_tests_pass(upstream_sales, tmp_path):
    sales_raw, sales_with_revenue = upstream_sales

    # (cell-id, cell-file, inputs) — inputs mirror each cell's upstream edge.
    plan = [
        ("create-data", "create_data", {}),
        ("add-columns", "add_columns", {"sales": sales_raw.copy()}),
        ("select-filter", "select_filter", {"sales": sales_raw.copy()}),
        ("summary", "summary", {"sales": sales_with_revenue.copy()}),
    ]

    for cell_id, cell_file, inputs in plan:
        test_path = _CELLS / f"{cell_id}.test.py"
        assert test_path.exists(), f"missing example test file: {test_path}"

        result = run_cell_tests_in_dir(
            rundir=tmp_path / cell_id,
            venv_python=Path(sys.executable),
            cell_source=_src(cell_file),
            test_source=test_path.read_text(),
            inputs=inputs,
        )

        assert result["passed"] >= 1, f"{cell_id}: expected passing tests, got {result}"
        assert result["failed"] == 0, f"{cell_id}: {result['tests']}"
        assert result["errored"] == 0, f"{cell_id}: {result['tests']}"
