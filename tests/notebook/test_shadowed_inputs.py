"""A name several upstream cells define is read from the producer the DAG wired.

``c0`` defines ``state`` and one more name, ``c1`` redefines ``state``, ``c2`` reads
both: ``state`` must come from ``c1`` (the last definer before ``c2``), never from
``c0``, on every path that loads a cell's inputs and in the provenance that keys it.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

# References are sorted, and ``upstream_ids`` follow them: with ``a`` c0 comes first, with
# ``z`` c1 does. The old loaders kept the first (batch) or the last (single cell, SQL,
# prompt) definer they met, so each order broke one of them.
_OTHER = ["a", "z"]


def _upstream(other: str) -> list[tuple[str, str]]:
    return [("c0", f"state = 1\n{other} = 10\n"), ("c1", "state = state + 1\n")]


def _session(tmp_path: Path, other: str, language: str, reader: str) -> NotebookSession:
    notebook_dir = create_notebook(tmp_path, "Shadowed")
    previous: str | None = None
    cells = [(cid, "python", src) for cid, src in _upstream(other)] + [("c2", language, reader)]
    for cell_id, cell_language, source in cells:
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=previous, language=cell_language)
        write_cell(notebook_dir, cell_id, source)
        previous = cell_id
    db_path = tmp_path / "empty.db"
    sqlite3.connect(db_path).close()
    toml_path = notebook_dir / "notebook.toml"
    toml_path.write_text(
        toml_path.read_text() + f'\n[connections.db]\ndriver = "sqlite"\npath = "{db_path}"\n'
    )
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    return session


async def _run_upstreams(session: NotebookSession, other: str) -> None:
    executor = CellExecutor(session)
    for cell_id, source in _upstream(other):
        result = await executor.execute_cell(cell_id, source)
        assert result.success, result.error


def _c1_state_hash(session: NotebookSession) -> str:
    manager = session.get_artifact_manager()
    artifact = manager.artifact_store.get_latest_version(manager.cell_artifact_id("c1", "state"))
    assert artifact is not None
    return artifact.provenance_hash


@pytest.mark.parametrize("other", _OTHER)
def test_the_dag_wires_each_name_from_its_last_definer(tmp_path: Path, other: str):
    session = _session(tmp_path, other, "python", f"out = state + {other}\n")
    assert session.dag is not None
    assert session.dag.wired_variables("c2", "c1") == {"state"}
    assert session.dag.wired_variables("c2", "c0") == {other}
    assert session.dag.wired_variables("c1", "c0") == {"state"}


@pytest.mark.asyncio
@pytest.mark.parametrize("other", _OTHER)
async def test_single_cell_reads_the_wired_producer(tmp_path: Path, other: str):
    reader = f"out = state * 100 + {other}\nprint(out)\n"
    session = _session(tmp_path, other, "python", reader)
    result = await CellExecutor(session).execute_cell("c2", reader)
    assert result.success, result.error
    assert result.stdout == "210\n"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.warm_pool
@pytest.mark.parametrize("other", _OTHER)
async def test_warm_pool_reads_the_wired_producer(tmp_path: Path, other: str):
    """The pool worker gets the same manifest inputs as the cold harness."""
    from strata.notebook.pool import WarmProcessPool

    reader = f"out = state * 100 + {other}\nprint(out)\n"
    session = _session(tmp_path, other, "python", reader)
    await _run_upstreams(session, other)
    session.ensure_venv_synced()
    pool = WarmProcessPool(
        session.path, pool_size=1, python_executable=session.venv_python or Path("python")
    )
    await pool.start()
    try:
        result = await CellExecutor(session, pool).execute_cell("c2", reader)
    finally:
        await pool.drain()
    assert result.success, result.error
    assert result.execution_method == "warm"
    assert result.stdout == "210\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("other", _OTHER)
async def test_run_all_batch_reads_the_wired_producer(tmp_path: Path, other: str):
    """Upstreams outside the batch seed one shared namespace; the shadowed one must not."""
    reader = f"out = state * 100 + {other}\nprint(out)\n"
    session = _session(tmp_path, other, "python", reader)
    await _run_upstreams(session, other)
    spec = {
        "cell_id": "c2",
        "source": reader,
        "consumed_vars": [],
        "env": {},
        "mount_manifest": {},
        "source_hash": "",
        "env_hash": "",
    }
    batch = await CellExecutor(session).execute_batch([spec])
    assert batch.completed, batch.end_reason
    (result,) = batch.cell_results
    assert result.status == "ok", result
    assert result.stdout == "210\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("other", _OTHER)
async def test_sql_binds_the_wired_producer(tmp_path: Path, other: str):
    from strata.notebook.sql.cell_executor import _load_upstream_variables

    source = (
        "# @sql connection=db\n"
        f"SELECT CAST(:state AS INTEGER) * 100 + CAST(:{other} AS INTEGER) AS out\n"
    )
    session = _session(tmp_path, other, "sql", source)
    await _run_upstreams(session, other)

    cell = session.notebook_state.get_cell("c2")
    namespace, hashes, _ = _load_upstream_variables(session, "c2", list(cell.references))
    assert namespace == {"state": 2, other: 10}
    assert hashes["state"] == _c1_state_hash(session)


@pytest.mark.asyncio
@pytest.mark.parametrize("other", _OTHER)
async def test_prompt_renders_the_wired_producer(tmp_path: Path, other: str):
    from strata.notebook.prompt_executor import _load_upstream_variables

    session = _session(tmp_path, other, "prompt", f"{{{{ state }}}} {{{{ {other} }}}}")
    await _run_upstreams(session, other)

    variables, hashes = _load_upstream_variables(session, "c2")
    assert variables == {"state": 2, other: 10}
    assert hashes["state"] == _c1_state_hash(session)


@pytest.mark.asyncio
async def test_provenance_follows_the_middle_cell(tmp_path: Path):
    """``out``'s inputs are c1's ``state`` and c0's ``z``: editing c1 moves its hash."""
    reader = "out = state * 100 + z\nprint(out)\n"
    session = _session(tmp_path, "z", "python", reader)
    executor = CellExecutor(session)
    first = await executor.execute_cell("c2", reader)
    assert first.stdout == "210\n"
    before = session.notebook_state.get_cell("c2").last_provenance_hash

    session.notebook_state.get_cell("c1").source = "state = state + 5\n"
    for cell in session.notebook_state.cells:
        session.re_analyze_cell(cell.id)
    second = await executor.execute_cell("c2", reader)
    assert second.cache_hit is False
    assert second.stdout == "610\n"
    after = session.notebook_state.get_cell("c2").last_provenance_hash
    assert after != before

    # Staleness recomputes the hash the run recorded.
    session.compute_staleness()
    assert session.notebook_state.get_cell("c2").status == "ready"


def _var_hash(session: NotebookSession, cell_id: str, var: str) -> str:
    manager = session.get_artifact_manager()
    artifact = manager.artifact_store.get_latest_version(manager.cell_artifact_id(cell_id, var))
    assert artifact is not None
    return artifact.provenance_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("other", _OTHER)
async def test_a_shadowed_read_keys_on_its_wired_producer(tmp_path: Path, other: str):
    """A cell that once cached the shadowed value gets a new key, so it runs once more."""
    session = _session(tmp_path, other, "python", f"out = state * 100 + {other}\n")
    await _run_upstreams(session, other)

    hashes = session._collect_input_hashes("c2")
    assert [h for h in hashes if h.startswith("wired:")] == [
        f"wired:state={_var_hash(session, 'c1', 'state')}"
    ]
    # c1 reads state from its only definer: no record, the key it always had.
    assert sorted(session._collect_input_hashes("c1")) == sorted(
        [_var_hash(session, "c0", "state"), _var_hash(session, "c0", other)]
    )


@pytest.mark.asyncio
async def test_a_plain_chain_keeps_its_provenance(tmp_path: Path):
    """No shadowed name: the key is still sha256(input artifact hashes, source, env)."""
    from strata.notebook.env import compute_execution_env_hash
    from strata.notebook.provenance import compute_provenance_hash, compute_source_hash

    notebook_dir = create_notebook(tmp_path, "Plain")
    add_cell_to_notebook(notebook_dir, "c0")
    write_cell(notebook_dir, "c0", "x = 1\n")
    add_cell_to_notebook(notebook_dir, "c1", after_cell_id="c0")
    write_cell(notebook_dir, "c1", "y = x + 1\n")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    executor = CellExecutor(session)
    for cell_id, source in (("c0", "x = 1\n"), ("c1", "y = x + 1\n")):
        assert (await executor.execute_cell(cell_id, source)).success

    x_hash = _var_hash(session, "c0", "x")
    assert session._collect_input_hashes("c1") == [x_hash]
    cell = session.notebook_state.get_cell("c1")
    env_hash = compute_execution_env_hash(
        session.path,
        session._collect_runtime_env(cell),
        runtime_identity=session._effective_worker_runtime_identity(cell),
    )
    assert cell.last_provenance_hash == compute_provenance_hash(
        [x_hash], compute_source_hash("y = x + 1\n"), env_hash
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("other", _OTHER)
async def test_staleness_agrees_with_the_run_for_a_shadowed_read(tmp_path: Path, other: str):
    """Ready after the run, stale once the middle cell changes, ready again after a rerun."""
    reader = f"out = state * 100 + {other}\n"
    session = _session(tmp_path, other, "python", reader)
    executor = CellExecutor(session)
    await _run_upstreams(session, other)
    assert (await executor.execute_cell("c2", reader)).success
    session.compute_staleness()
    assert session.notebook_state.get_cell("c2").status == "ready"

    session.notebook_state.get_cell("c1").source = "state = state + 5\n"
    for cell in session.notebook_state.cells:
        session.re_analyze_cell(cell.id)
    session.compute_staleness()
    assert session.notebook_state.get_cell("c2").status == "stale"

    rerun = await executor.execute_cell("c2", reader)
    assert rerun.success and rerun.cache_hit is False
    session.compute_staleness()
    assert session.notebook_state.get_cell("c2").status == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize("other", _OTHER)
async def test_lineage_records_only_the_wired_producer(tmp_path: Path, other: str):
    import json

    reader = f"out = state * 100 + {other}\n"
    session = _session(tmp_path, other, "python", reader)
    await _run_upstreams(session, other)
    assert (await CellExecutor(session).execute_cell("c2", reader)).success

    manager = session.get_artifact_manager()
    console = manager.artifact_store.get_latest_version(
        manager.cell_artifact_id("c2", "__console__")
    )
    assert console is not None
    recorded = set(json.loads(console.input_versions))
    c0 = session.notebook_state.get_cell("c0").artifact_uris
    c1 = session.notebook_state.get_cell("c1").artifact_uris
    assert recorded == {c0[other], c1["state"]}
