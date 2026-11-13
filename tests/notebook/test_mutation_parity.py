"""A cell's mutation warnings are the same whether it runs alone or in a Run All batch.

The warm pool's half of this is ``test_warm_execution_reports_same_mutation_warnings_as_cold``
in ``test_executor.py``; the batch harness is a third copy of the same check.
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from strata.notebook import ws
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

pytestmark = pytest.mark.asyncio

CASES = [
    pytest.param("x = 1", "del x\ny = 2", "z = y", ["x"], id="a deleted input"),
    # Exported, so downstream gets the mutated list: nothing to warn.
    pytest.param("xs = [1, 2]", "xs[0] = 9", "y = xs[0]", [], id="an exported mutated input"),
    pytest.param(
        "x = 1",
        "params = [1.0, 2.0]\ntrainer = {'params': params}",
        "z = (params, trainer)",
        ["params"],
        id="two outputs sharing a list",
    ),
]


def _session(path: Path, cells: dict[str, str]) -> NotebookSession:
    notebook_dir = create_notebook(path, "Mutations")
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(notebook_dir, cell_id, after)
        write_cell(notebook_dir, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


def _warned(warnings: list) -> list[str]:
    return [w["var_name"] if isinstance(w, dict) else w.var_name for w in warnings]


@pytest.mark.parametrize(("up", "mid", "down", "warned"), CASES)
async def test_run_all_batch_warns_as_a_single_run_does(tmp_path, up, mid, down, warned):
    cells = {"up": up, "mid": mid, "down": down}

    single = _session(tmp_path / "single", cells)
    executor = CellExecutor(single)
    assert (await executor.execute_cell("up", up)).success
    alone = await executor.execute_cell("mid", mid)
    assert alone.success, alone.error
    assert _warned(alone.mutation_warnings) == warned

    batched = _session(tmp_path / "batch", cells)
    results: dict[str, object] = {}
    real_apply = batched.apply_execution_result_metadata

    def apply(cell_id: str, result, **kwargs) -> None:
        results.setdefault(cell_id, result)
        real_apply(cell_id, result, **kwargs)

    try:
        with mock.patch.object(batched, "apply_execution_result_metadata", apply):
            await ws._execute_run_all(
                None,
                batched,
                ["up", "mid", "down"],
                ws._ensure_execution_state(batched.id),
                batched.id,
            )
    finally:
        ws.forget_notebook_execution_state(batched.id)

    in_batch = results["mid"]
    assert in_batch.success, in_batch.error
    assert in_batch.execution_method == "batch"
    assert _warned(in_batch.mutation_warnings) == warned
