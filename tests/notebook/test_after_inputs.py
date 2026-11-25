"""A cell's `@after` upstream is an input of its key: the run it follows.

What the upstream left (a database, files) is what the cell reads, so a cell after one
that ran again is stale, and runs again, in every language and on every path.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _setup(text: str, *, nocache: bool = False) -> str:
    """Writes a file and stores nothing."""
    return ("# @nocache\n" if nocache else "") + (
        f"from pathlib import Path\nPath('../marker.txt').write_text({text!r})\nprint('set')\n"
    )


READ = "# @after setup\nfrom pathlib import Path\nseen = Path('../marker.txt').read_text()\n"
USE = "out = seen\n"


def _notebook(tmp_path: Path, setup: str) -> Path:
    nb_dir = create_notebook(tmp_path, "after_inputs")
    previous = None
    for cell_id, source in (("setup", setup), ("read", READ), ("use", USE)):
        add_cell_to_notebook(nb_dir, cell_id, after_cell_id=previous)
        write_cell(nb_dir, cell_id, source)
        previous = cell_id
    return nb_dir


def _open(nb_dir: Path) -> NotebookSession:
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    session.refresh_environment_runtime()
    return session


async def _run(session: NotebookSession, cell_id: str, *, rerun: bool = False):
    cell = session.notebook_state.get_cell(cell_id)
    assert cell is not None
    executor = CellExecutor(session)
    run = executor.execute_cell_rerun if rerun else executor.execute_cell
    result = await run(cell_id, cell.source)
    assert result.success, result.error
    return result


def _edit(session: NotebookSession, nb_dir: Path, cell_id: str, source: str) -> None:
    """As an editor's source update does."""
    write_cell(nb_dir, cell_id, source)
    cell = session.notebook_state.get_cell(cell_id)
    assert cell is not None
    cell.source = source
    session.re_analyze_cell(cell_id)


def _seen(session: NotebookSession) -> str:
    manager = session.get_artifact_manager()
    latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id("read", "seen"))
    assert latest is not None
    return manager.load_artifact_data(latest.id, latest.version).decode()


@pytest.mark.asyncio
async def test_a_python_cell_after_a_new_setup_runs_again(tmp_path):
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    assert (await _run(session, "read")).cache_hit is False
    assert (await _run(session, "read")).cache_hit is True

    _edit(session, nb_dir, "setup", _setup("two"))
    await _run(session, "setup")

    assert (await _run(session, "read")).cache_hit is False
    assert "two" in _seen(session)


@pytest.mark.asyncio
async def test_a_cell_after_a_setup_that_ran_again_is_stale(tmp_path):
    """And the explanation names the setup."""
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    await _run(session, "read")
    assert session.compute_staleness()["read"].status == CellStatus.READY

    await _run(session, "setup", rerun=True)
    staleness = session.compute_staleness()

    assert staleness["read"].status == CellStatus.STALE
    details = session.causality_map["read"].details
    assert [(d.type.value, d.cell_id) for d in details] == [("input_changed", "setup")]


@pytest.mark.asyncio
async def test_a_setup_that_stores_nothing_is_followed_by_its_runs(tmp_path):
    """A `@nocache` setup with no output still records each run, and only a run."""
    from strata.notebook.keys import after_run

    nb_dir = _notebook(tmp_path, _setup("one", nocache=True))
    session = _open(nb_dir)
    await _run(session, "setup")
    first = after_run(session, "read", "setup")
    session.compute_staleness()
    session.compute_staleness()
    assert after_run(session, "read", "setup") == first

    await _run(session, "setup")

    assert after_run(session, "read", "setup") not in (first, "never")


@pytest.mark.asyncio
async def test_a_setup_that_stopped_storing_is_followed_still(tmp_path):
    """Its record from a run that stored one is no run of the setup's now."""
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    await _run(session, "read")

    _edit(session, nb_dir, "setup", _setup("one", nocache=True))
    await _run(session, "setup")

    cell = session.notebook_state.get_cell("read")
    assert cell is not None
    result = await CellExecutor(session).execute_cell_force("read", cell.source)
    assert result.cache_hit is False


@pytest.mark.asyncio
async def test_a_reopen_keeps_the_run_a_cell_followed(tmp_path):
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    await _run(session, "read")

    reopened = _open(nb_dir)

    assert reopened.compute_staleness()["read"].status == CellStatus.READY
    assert (await _run(reopened, "read")).cache_hit is True


async def _run_all(session: NotebookSession) -> None:
    """As a Run All over the WebSocket does: one batch, keyed before it starts."""
    from strata.notebook import ws

    notebook_id = session.notebook_state.id
    batch = await ws._run_partition_batch(
        session=session,
        executor=CellExecutor(session),
        cells_in_run=list(session.notebook_state.cells),
        notebook_id=notebook_id,
        force=False,
        execution_state=ws._ensure_execution_state(notebook_id),
    )
    assert batch.completed, batch.cell_results


@pytest.mark.asyncio
async def test_run_all_reads_the_run_of_a_setup_in_the_same_batch(tmp_path):
    """The cell after it is checked once the setup ran, so it runs again and ends ready."""
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run_all(session)
    _edit(session, nb_dir, "setup", _setup("two"))

    await _run_all(session)

    assert "two" in _seen(session)
    staleness = session.compute_staleness()
    assert {cell_id: staleness[cell_id].status for cell_id in ("setup", "read", "use")} == {
        "setup": CellStatus.READY,
        "read": CellStatus.READY,
        "use": CellStatus.READY,
    }


@pytest.mark.asyncio
async def test_run_all_again_unchanged_reuses_the_cell_after_a_setup(tmp_path):
    """A batched cache hit runs no code, so the cell after it keeps its key."""
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run_all(session)
    first = _token(nb_dir)

    await _run_all(session)

    assert _token(nb_dir) == first
    assert session.compute_staleness()["read"].status == CellStatus.READY


# A write after a seed: what the seed rebuilt drops what the write added.


def _sql(source: str) -> str:
    return "-- @sql connection=db write=true\n" + source


SEED = _sql("DROP TABLE IF EXISTS t;\nCREATE TABLE t (n INTEGER);\nINSERT INTO t VALUES (1);\n")
ADD = _sql("-- @after seed\nINSERT INTO t VALUES (2);\n")


def _sql_notebook(tmp_path: Path) -> Path:
    nb_dir = create_notebook(tmp_path, "after_writes", initialize_environment=False)
    add_cell_to_notebook(nb_dir, "seed", language="sql")
    write_cell(nb_dir, "seed", SEED)
    add_cell_to_notebook(nb_dir, "add", after_cell_id="seed", language="sql")
    write_cell(nb_dir, "add", ADD)
    toml = nb_dir / "notebook.toml"
    toml.write_text(toml.read_text() + '\n[connections.db]\ndriver = "sqlite"\npath = "w.db"\n')
    return nb_dir


def _rows(nb_dir: Path) -> list[int]:
    with closing(sqlite3.connect(nb_dir / "w.db")) as conn:
        return sorted(n for (n,) in conn.execute("SELECT n FROM t"))


@pytest.mark.asyncio
async def test_a_write_after_a_seed_run_again_writes_again(tmp_path):
    """The seed's run again is the same computation, and still drops the write's row."""
    pytest.importorskip("adbc_driver_sqlite")
    nb_dir = _sql_notebook(tmp_path)
    session = _open(nb_dir)
    await _run(session, "seed")
    await _run(session, "add")
    assert _rows(nb_dir) == [1, 2]

    await _run(session, "seed", rerun=True)
    assert _rows(nb_dir) == [1]
    assert session.compute_staleness()["add"].status == CellStatus.STALE

    assert (await _run(session, "add")).cache_hit is False
    assert _rows(nb_dir) == [1, 2]


# What counts as a run of the upstream.


def _token(nb_dir: Path, cell_id: str = "setup") -> str | None:
    from strata.notebook import runs

    return runs.read(nb_dir).get(cell_id)


@pytest.mark.parametrize(
    "body",
    [
        "x = 1\n",
        "raise RuntimeError('partway')\n",  # it may have changed the world first
        "# @timeout 1\nimport time\ntime.sleep(30)\n",  # as a stop or a dead harness
    ],
    ids=["ran", "failed", "timed_out"],
)
@pytest.mark.asyncio
async def test_a_run_of_the_upstream_is_minted_as_its_code_starts(tmp_path, body):
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    first = _token(nb_dir)
    _edit(session, nb_dir, "setup", body)
    cell = session.notebook_state.get_cell("setup")
    assert cell is not None

    await CellExecutor(session).execute_cell("setup", cell.source)

    assert _token(nb_dir) not in (None, first)


@pytest.mark.asyncio
async def test_what_runs_no_code_mints_nothing(tmp_path):
    """A cache hit, and a refusal before the run (an annotation it cannot read)."""
    nb_dir = _notebook(tmp_path, "x = 1\n")
    session = _open(nb_dir)
    await _run(session, "setup")
    first = _token(nb_dir)
    assert (await _run(session, "setup")).cache_hit is True
    assert _token(nb_dir) == first

    _edit(session, nb_dir, "setup", "# @timeout soon\nx = 1\n")
    cell = session.notebook_state.get_cell("setup")
    assert cell is not None
    refused = await CellExecutor(session).execute_cell("setup", cell.source)

    assert not refused.success
    assert _token(nb_dir) == first


def test_an_upstream_that_never_ran_here_is_followed_by_this_checkouts_token(tmp_path):
    """A markdown cell, or one not run yet: stable here, and no other checkout's."""
    from strata.notebook.keys import after_run

    def checkout(name: str) -> NotebookSession:
        nb_dir = _notebook(tmp_path / name, _setup("one"))
        add_cell_to_notebook(nb_dir, "intro", language="markdown")
        write_cell(nb_dir, "intro", "# Notes\n")
        return _open(nb_dir)

    mine, theirs = checkout("mine"), checkout("theirs")

    assert after_run(mine, "read", "intro") == after_run(mine, "read", "intro")
    assert after_run(mine, "read", "setup") == after_run(mine, "read", "setup")
    assert after_run(mine, "read", "setup") != after_run(theirs, "read", "setup")


@pytest.mark.asyncio
async def test_a_cells_tests_are_stale_after_its_upstream_ran_again(tmp_path):
    """The Tests tab keys its inputs as the cell is keyed."""
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    await _run(session, "read")
    session.compute_staleness()
    before = session._cell_test_inputs["read"]

    await _run(session, "setup", rerun=True)
    session.compute_staleness()

    assert session._cell_test_inputs["read"] != before


@pytest.mark.asyncio
async def test_a_prompts_after_only_orders_it(tmp_path, monkeypatch):
    """It reads only its template's variables, so a rerun of the upstream is not news."""
    from strata.notebook import prompt_executor
    from strata.notebook.llm import LlmConfig

    # Nothing listens there: each run fails after it is keyed, which is all this reads.
    config = LlmConfig(base_url="http://127.0.0.1:9", api_key="k", model="m")
    nb_dir = _notebook(tmp_path, _setup("one"))
    add_cell_to_notebook(nb_dir, "ask", after_cell_id="use", language="prompt")
    write_cell(nb_dir, "ask", "# @after setup\nSay hi.\n")
    session = _open(nb_dir)
    keys: list[str] = []
    real = prompt_executor.compute_prompt_provenance_hash

    def spy(**kwargs):
        keys.append(real(**kwargs))
        return keys[-1]

    monkeypatch.setattr(prompt_executor, "compute_prompt_provenance_hash", spy)
    await _run(session, "setup")
    cell = session.notebook_state.get_cell("ask")
    assert cell is not None
    await prompt_executor.execute_prompt_cell(session, "ask", cell.source, config)
    await _run(session, "setup", rerun=True)
    await prompt_executor.execute_prompt_cell(session, "ask", cell.source, config)

    assert len(keys) == 2 and keys[0] == keys[1]
    # Nor does its staleness key read the upstream's run.
    from strata.notebook.keys import after_fingerprints

    assert session.analysis is not None
    assert after_fingerprints(session, session.analysis, "ask") == []


def test_an_after_change_is_blamed_on_the_upstream_only_when_it_ran(tmp_path):
    """Not on a record from before after edges were keyed."""
    import cellgraph as cg

    from strata.notebook import staleness_explanations

    session = _open(_notebook(tmp_path, _setup("one")))

    def part(fingerprint: str) -> list[str]:
        resolved = cg.Resolved(identity=None, fingerprint=fingerprint)
        return ["after", "setup", cg.input_component("after", "setup", resolved)]

    def changes(before: list[list[str]], now: list[list[str]]):
        return [
            (detail.type.value, detail.cell_id, detail.input_kind)
            for detail in staleness_explanations._input_changes(
                session, {"inputs": before}, {"inputs": now}, False, set()
            )
        ]

    assert changes([part("setup:after:1")], [part("setup:after:2")]) == [
        ("input_changed", "setup", None)
    ]
    assert changes([], [part("setup:after:2")]) == []


def test_a_run_in_another_process_is_read(tmp_path):
    """It writes the notebook's runs file, which the open session reads."""
    from strata.notebook import runs
    from strata.notebook.keys import after_run

    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    first = runs.record(nb_dir, "setup")
    assert after_run(session, "read", "setup") == first

    second = runs.record(nb_dir, "setup")

    assert after_run(session, "read", "setup") == second != first


def test_another_cells_run_keeps_this_ones(tmp_path):
    from strata.notebook import runs

    token = runs.record(tmp_path, "setup")
    runs.record(tmp_path, "other")

    assert runs.read(tmp_path)["setup"] == token


@pytest.mark.asyncio
async def test_an_imported_snapshot_keeps_the_run_a_cell_followed(tmp_path):
    import zipfile

    from strata.notebook.snapshot import write_committed_files, write_snapshot
    from strata.notebook.snapshot_import import import_snapshot

    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    await _run(session, "read")
    bundle = tmp_path / "snap.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        write_committed_files(session, archive)
        write_snapshot(session, archive, include="all")

    imported = _open(import_snapshot(bundle, tmp_path / "dst").notebook_dir)

    assert imported.compute_staleness()["read"].status == CellStatus.READY


@pytest.mark.asyncio
async def test_a_batched_cell_whose_mount_failed_did_not_run(tmp_path):
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    first = _token(nb_dir)
    missing = (tmp_path / "no-such-dir").as_uri()
    _edit(session, nb_dir, "setup", f"# @mount data {missing} ro\n" + _setup("one"))

    from strata.notebook import ws

    notebook_id = session.notebook_state.id
    setup = session.notebook_state.get_cell("setup")
    batch = await ws._run_partition_batch(
        session=session,
        executor=CellExecutor(session),
        cells_in_run=[setup],
        notebook_id=notebook_id,
        force=False,
        execution_state=ws._ensure_execution_state(notebook_id),
    )

    assert [r.status for r in batch.cell_results] == ["cell_error"]
    assert _token(nb_dir) == first


@pytest.mark.asyncio
async def test_a_read_only_sql_cell_changes_nothing_and_mints_nothing(tmp_path):
    pytest.importorskip("adbc_driver_sqlite")
    nb_dir = _sql_notebook(tmp_path)
    session = _open(nb_dir)
    await _run(session, "seed")
    add_cell_to_notebook(nb_dir, "look", after_cell_id="add", language="sql")
    write_cell(nb_dir, "look", "-- @sql connection=db\nSELECT n FROM t\n")
    session = _open(nb_dir)
    await _run(session, "look")

    assert _token(nb_dir, "look") is None
    assert _token(nb_dir, "seed") is not None


@pytest.mark.asyncio
async def test_a_loop_upstream_is_minted_as_its_iterations_start(tmp_path):
    nb_dir = create_notebook(tmp_path, "after_loop")
    previous = None
    for cell_id, source in (
        ("seed", "state = 0\n"),
        ("setup", "# @loop max_iter=2 carry=state\nstate = state + 1\n"),
        ("read", "# @after setup\nx = 1\n"),
    ):
        add_cell_to_notebook(nb_dir, cell_id, after_cell_id=previous)
        write_cell(nb_dir, cell_id, source)
        previous = cell_id
    session = _open(nb_dir)
    await _run(session, "setup")
    first = _token(nb_dir)
    assert first is not None

    await _run(session, "setup", rerun=True)

    assert _token(nb_dir) not in (None, first)


@pytest.mark.asyncio
async def test_what_is_keyed_while_the_upstream_runs_is_stale_once_it_ends(tmp_path):
    from strata.notebook import runs
    from strata.notebook.keys import after_run

    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    async with runs.running(nb_dir, "setup"):
        during = after_run(session, "read", "setup")

    assert after_run(session, "read", "setup") not in (during, "never")


@pytest.mark.asyncio
async def test_a_stopped_run_still_ends(tmp_path):
    import asyncio

    from strata.notebook import runs

    async def run() -> None:
        async with runs.running(tmp_path, "setup"):
            await asyncio.sleep(30)

    task = asyncio.create_task(run())
    while "setup" not in runs.read(tmp_path):
        await asyncio.sleep(0.01)
    during = runs.read(tmp_path)["setup"]
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert runs.read(tmp_path)["setup"] != during


@pytest.mark.asyncio
async def test_a_batched_run_ends_before_its_result_is_applied(tmp_path, monkeypatch):
    """So a cell keyed while it ran is stale once Run All applies its result."""
    from strata.notebook import runs

    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    seen: list[str | None] = []
    real = session.apply_execution_result_metadata

    def apply(cell_id, result, **kwargs):
        if cell_id == "setup":
            seen.append(runs.read(nb_dir).get("setup"))
        return real(cell_id, result, **kwargs)

    session.apply_execution_result_metadata = apply  # type: ignore[method-assign]
    started: list[str] = []
    mint = runs._record

    def spy(notebook_dir, cell_id):
        token = mint(notebook_dir, cell_id)
        if cell_id == "setup":
            started.append(token)
        return token

    monkeypatch.setattr(runs, "_record", spy)
    await _run_all(session)

    assert len(started) == 2 and seen == [started[1]]


@pytest.mark.asyncio
async def test_a_cells_tests_are_a_run_of_its_code(tmp_path):
    """They run its body again, side effects and all."""
    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    first = _token(nb_dir)

    executor = CellExecutor(session)
    await executor.run_cell_tests("setup", "def test_x():\n    assert True\n")

    assert _token(nb_dir) not in (None, first)


@pytest.mark.asyncio
async def test_a_tests_click_without_pytest_is_no_run(tmp_path, monkeypatch):
    """Its code never starts, so a cell after it keeps its result."""
    from strata.notebook import cell_test_runner, dependencies
    from strata.notebook.dependencies import DependencyChangeResult

    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    await _run(session, "setup")
    first = _token(nb_dir)
    monkeypatch.setattr(cell_test_runner, "_pytest_available", lambda _python: False)
    monkeypatch.setattr(
        dependencies,
        "ensure_dev_tool",
        lambda _dir, tool, **_kw: DependencyChangeResult(
            success=False, package=tool, action="add", error="offline"
        ),
    )

    result = await CellExecutor(session).run_cell_tests("setup", "def test_x():\n    assert True\n")

    assert result.pytest_unavailable is True
    assert _token(nb_dir) == first


@pytest.mark.asyncio
async def test_a_run_whose_end_cannot_be_recorded_keeps_its_outcome(tmp_path, monkeypatch, caplog):
    from strata.notebook import runs

    calls: list[str] = []

    def record(notebook_dir, cell_id):
        calls.append(cell_id)
        if len(calls) == 2:
            raise OSError("disk full")
        return "token"

    monkeypatch.setattr(runs, "_record", record)
    async with runs.running(tmp_path, "setup"):
        outcome = "finished"

    assert outcome == "finished" and calls == ["setup", "setup"]
    assert "ended unrecorded" in caplog.text


def test_keying_a_cell_after_an_upstream_that_never_ran_writes_nothing(tmp_path):
    """A key is worked out on the event loop, and while the notebook is held."""
    from strata.notebook import runs
    from strata.notebook.keys import after_run

    nb_dir = _notebook(tmp_path, _setup("one"))
    session = _open(nb_dir)
    after_run(session, "read", "setup")

    assert not runs.runs_path(nb_dir).exists()


@pytest.mark.asyncio
async def test_a_run_that_ends_while_the_notebook_is_held_still_ends(tmp_path):
    """A run that drained past its window: what was keyed while it ran is stale."""
    from strata.notebook import quiesce, runs

    hold = None
    try:
        async with runs.running(tmp_path, "setup"):
            during = runs.read(tmp_path)["setup"]
            hold = quiesce.begin(tmp_path.resolve(), 60)
            quiesce.settle(hold)
    finally:
        quiesce.reset()

    assert runs.read(tmp_path)["setup"] != during
