"""An annotation that cannot be read keeps its cell, and what depends on it, from running."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from strata.notebook import ws
from strata.notebook.annotation_validation import blocking_diagnostic
from strata.notebook.annotations import ANNOTATION_KEYS
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

# One malformed example per key, with the code it is refused under.
MALFORMED = {
    "after": ("# @after", "after_unreadable"),
    "app": ("# @app show", "app_unreadable"),
    "cache": ("# @cache forevr", "cache_policy_unknown"),
    "dataset": ("# @dataset model", "dataset_unreadable"),
    "env": ("# @env NOEQUALS", "env_malformed"),
    "fetch": ("# @fetch zones", "fetch_unreadable"),
    "live": ("# @live maybe", "live_unreadable"),
    "loop": ("# @loop max_iter=abc carry=x", "loop_unreadable"),
    "loop_until": ("# @loop_until", "loop_until_unreadable"),
    "max_tokens": ("# @max_tokens many", "max_tokens_unreadable"),
    "model": ("# @model", "model_unreadable"),
    "mount": ("# @mount data", "mount_unreadable"),
    "name": ("# @name", "name_unreadable"),
    "nocache": ("# @nocache maybe", "nocache_unreadable"),
    "output": ("# @output yaml", "output_unreadable"),
    "output_dir": ("# @output_dir 1bad", "output_dir_unreadable"),
    "output_schema": ("# @output_schema", "output_schema_unreadable"),
    "per_variant": ("# @per_variant a b", "per_variant_unreadable"),
    "sql": ("# @sql connection", "sql_unreadable"),
    "stochastic": ("# @stochastic sometimes", "stochastic_unreadable"),
    "system": ("# @system", "system_unreadable"),
    "table": ("# @table orders", "table_unreadable"),
    "temperature": ("# @temperature hot", "temperature_unreadable"),
    "timeout": ("# @timeout soon", "timeout_not_numeric"),
    "validate_retries": ("# @validate_retries 0", "validate_retries_unreadable"),
    "variant": ("# @variant onlygroup", "variant_malformed"),
    "worker": ("# @worker", "worker_unreadable"),
    "write_table": ("# @write_table t lake:a.b upsert", "write_table_unreadable"),
}


def _session(tmp_path: Path, cells: dict[str, str]) -> NotebookSession:
    nb = create_notebook(tmp_path, "Unreadable", initialize_environment=False)
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.venv_python = Path(sys.executable)
    return session


def test_every_annotation_key_has_a_malformed_example():
    assert set(MALFORMED) == ANNOTATION_KEYS


@pytest.mark.parametrize("key", sorted(MALFORMED))
def test_a_malformed_annotation_is_an_error_and_refuses_the_run(tmp_path, key):
    directive, code = MALFORMED[key]
    source = f"{directive}\nx = 1\n"
    session = _session(tmp_path, {"c1": source})
    cell = session.notebook_state.get_cell("c1")

    diagnostic = blocking_diagnostic(cell, session.notebook_state, session.dag)
    assert diagnostic is not None
    assert diagnostic.code == code

    result = asyncio.run(CellExecutor(session).execute_cell("c1", source))
    assert not result.success
    assert result.error_code == code
    assert result.error.endswith(f"[{code}]")
    assert session.get_artifact_manager().artifact_store.list_artifacts() == []


@pytest.mark.parametrize("mode", ["execute_cell", "execute_cell_force", "execute_cell_rerun"])
def test_every_run_mode_is_refused(tmp_path, mode):
    source = "# @timeout soon\nx = 1\n"
    session = _session(tmp_path, {"c1": source})

    result = asyncio.run(getattr(CellExecutor(session), mode)("c1", source))

    assert not result.success
    assert result.error_code == "timeout_not_numeric"
    assert result.error.startswith("line 1: `@timeout soon` could not be read")


@pytest.mark.parametrize(
    "value",
    ["", "on", "off", "true", "false", "yes", "no", "1", "0", "OFF"],
)
def test_a_flag_reads_each_documented_value(tmp_path, value):
    session = _session(tmp_path, {"c1": f"# @nocache {value}\nx = 1\n"})
    cell = session.notebook_state.get_cell("c1")
    assert blocking_diagnostic(cell, session.notebook_state, session.dag) is None


@pytest.mark.parametrize(
    "directive",
    [
        "# @mount data file:///tmp ro credential=lab",
        "# @table trips file:///wh#nyc.trips snapshot=12",
        "# @loop max_iter=3 carry=x until=x>2",
        "# @sql connection=db write=true",
        "# @cache ttl=60",
        "# @app hide",
        "# @per_variant model",
        "# @temperature 0",
        "# @output json",
    ],
)
def test_a_well_formed_annotation_is_not_refused(tmp_path, directive):
    session = _session(tmp_path, {"c1": f"{directive}\nx = 1\n"})
    cell = session.notebook_state.get_cell("c1")
    assert blocking_diagnostic(cell, session.notebook_state, session.dag) is None


BLOCKED = "# @timeout soon\nx = 1\n"


def _run_all(session: NotebookSession, monkeypatch) -> list[str]:
    """Run All over every cell; the ids it marked blocked by a failure."""
    blocked: list[str] = []
    original = ws._mark_blocked_by_failure

    async def record(session, notebook_id, cell_id):
        blocked.append(cell_id)
        await original(session, notebook_id, cell_id)

    monkeypatch.setattr(ws, "_mark_blocked_by_failure", record)
    asyncio.run(
        ws._execute_run_all(
            None,
            session,
            [c.id for c in session.notebook_state.cells],
            ws.NotebookExecutionState(),
            session.id,
        )
    )
    return blocked


def test_run_all_skips_a_dependent_through_a_variable_edge(tmp_path, monkeypatch):
    session = _session(tmp_path, {"c1": BLOCKED, "c2": "y = x + 1\n"})

    assert _run_all(session, monkeypatch) == ["c2"]
    assert session.notebook_state.get_cell("c1").status == "error"


def test_run_all_skips_a_dependent_through_an_after_edge(tmp_path, monkeypatch):
    session = _session(tmp_path, {"c1": BLOCKED, "c2": "# @after c1\ny = 1\n"})

    assert _run_all(session, monkeypatch) == ["c2"]


def test_run_all_skips_a_fork_through_its_seed_edge(tmp_path, monkeypatch):
    seed = "# @loop max_iter=2 carry=s\n# @timeout soon\ns = 1\n"
    fork = "# @loop max_iter=2 carry=s start_from=seed@iter=0\ns = s + 1\n"
    session = _session(tmp_path, {"init": "s = 0\n", "seed": seed, "fork": fork})
    assert session.dag.cell_upstream["fork"] == ["seed"]

    assert _run_all(session, monkeypatch) == ["fork"]


def test_a_dependent_run_on_its_own_stops_at_the_blocked_cell(tmp_path):
    session = _session(tmp_path, {"c1": BLOCKED, "c2": "y = x + 1\n"})
    executor = CellExecutor(session)

    result = asyncio.run(executor.execute_cell("c2", "y = x + 1\n"))

    assert not result.success
    assert "Failed to materialise upstream cell c1" in result.error
    assert "timeout_not_numeric" in result.error
    assert session.get_artifact_manager().artifact_store.list_artifacts() == []


def test_a_cascade_stops_at_the_blocked_cell(tmp_path):
    from strata.notebook.cascade import CascadePlanner

    session = _session(tmp_path, {"c1": BLOCKED, "c2": "y = x + 1\n"})
    plan = CascadePlanner(session).plan("c2")
    assert plan is not None
    assert [step.cell_id for step in plan.steps] == ["c1", "c2"]

    asyncio.run(ws._execute_cascade(None, session, plan, ws.NotebookExecutionState(), session.id))

    assert session.notebook_state.get_cell("c1").status == "error"
    assert session.notebook_state.get_cell("c2").status != "ready"
    assert session.get_artifact_manager().artifact_store.list_artifacts() == []


def test_the_headless_run_refuses_the_cell(tmp_path, capsys):
    from strata.cli import main
    from tests.notebook.test_cli import _build_notebook, _mk_fake_venv

    nb = _build_notebook(tmp_path, cells=[("c1", BLOCKED, None), ("c2", "y = x + 1\n", "c1")])
    _mk_fake_venv(nb)

    assert main(["run", str(nb), "--no-sync", "--format", "json"]) != 0
    cells = {cell["id"]: cell for cell in json.loads(capsys.readouterr().out)["cells"]}

    assert cells["c1"]["status"] == "error"
    assert "timeout_not_numeric" in cells["c1"]["error"]
    assert cells["c2"]["status"] != "ok"


def test_run_all_skips_dependents_that_would_run_as_one_batch(tmp_path, monkeypatch):
    session = _session(tmp_path, {"c1": BLOCKED, "c2": "y = x + 1\n", "c3": "z = y + 1\n"})

    assert _run_all(session, monkeypatch) == ["c2", "c3"]
