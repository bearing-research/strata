"""What each cell path does after a run, in what order, and when a step fails.

The orders differ and stay different: Python and R push before displays and reject a false
store before either; batch stores displays first, rejects a false store after them, and pushes
last; a loop has no ``_store_outputs`` and no push; ``@each`` does its instances by the Python
path and then writes its manifests. Each failure is pinned here.
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strata.notebook import executor as executor_module
from strata.notebook.cell_paths import fanout
from strata.notebook.executor import (
    CellExecutionResult,
    CellExecutor,
    TableWriteError,
    _mount_sync_failure_result,
    _store_failure_result,
)
from strata.notebook.harness_user import UnsafeRunFile
from strata.notebook.models import WorkerBackendType, WorkerSpec
from strata.notebook.mounts import MountResolver
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.notebook.conftest import skip_if_no_r

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX cell execution")

PYTHON = ("python", 'value = 1\ndisplay(Markdown("d"))\n', "seen = value\n", "")
R = ("r", "value <- 1\n", "seen <- value\n", "R ")
LANGUAGES = [
    pytest.param(*PYTHON, id="python"),
    pytest.param(*R, id="r", marks=skip_if_no_r),
]
STEPS = (
    "_store_outputs",
    "_close_open_run",
    "_push_to_team_store",
    "_store_display_outputs",
    "_store_console_outputs",
    "_store_each_manifests",
)
STORE_FAILED = "Cell executed successfully but failed to store output artifacts. Check server logs."


def _owner(name: str):
    """Where a step lives: the executor, or the moved path it is a function of."""
    return fanout if name == "_store_each_manifests" else CellExecutor


def _session(tmp_path: Path, *cells: tuple[str, str, str]) -> NotebookSession:
    """A notebook of ``(cell id, language, source)``."""
    nb = create_notebook(tmp_path / "nb", "After")
    after = None
    for cell_id, language, source in cells:
        add_cell_to_notebook(nb, cell_id, after, language=language)
        write_cell(nb, cell_id, source)
        after = cell_id
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _latest(session: NotebookSession, cell_id: str, label: str) -> int | None:
    manager = session.get_artifact_manager()
    latest = manager.artifact_store.get_latest_version(manager.cell_artifact_id(cell_id, label))
    return latest.version if latest is not None else None


def _after_run(steps: list[str]) -> list[str]:
    """*steps* without closing the run, which every dispatch does at its end."""
    return [step for step in steps if step != "_close_open_run"]


@pytest.fixture
def steps(monkeypatch) -> list[str]:
    """The after-run steps, as they are called, each still doing its work."""
    calls: list[str] = []

    def spy(name: str):
        original = getattr(_owner(name), name)
        if inspect.iscoroutinefunction(original):

            async def recorded(self, *args, **kwargs):
                calls.append(name)
                return await original(self, *args, **kwargs)
        else:

            def recorded(self, *args, **kwargs):
                calls.append(name)
                return original(self, *args, **kwargs)

        return recorded

    for name in STEPS:
        monkeypatch.setattr(_owner(name), name, spy(name))
    return calls


def _fail(monkeypatch, name: str, outcome: Exception | bool, calls: list[str]) -> None:
    """*name* records its call, then raises *outcome* or returns it."""
    if inspect.iscoroutinefunction(getattr(_owner(name), name)):

        async def failing(self, *args, **kwargs):
            calls.append(name)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
    else:

        def failing(self, *args, **kwargs):
            calls.append(name)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    monkeypatch.setattr(_owner(name), name, failing)


WARNING = {"var_name": "x", "message": "x was mutated", "suggestion": "rebind it"}


def _team_store_breaks(monkeypatch, error: BaseException = RuntimeError("no team")) -> list[str]:
    """The team cache is on, and publishing raises *error*. Returns the cells offered."""
    offered: list[str] = []

    async def no_pull(self, **_kwargs):
        return None

    async def publish(_store, _manager, *, cell_id, **_kwargs):
        offered.append(cell_id)
        raise error

    real = CellExecutor._lake_config

    def lake_config(self):
        config = real(self)
        return SimpleNamespace(
            **{
                **vars(config),
                "notebook_team_cache_enabled": True,
                "notebook_team_cache_publish": "all",
                "notebook_remote_store_url": "http://team",
            }
        )

    monkeypatch.setattr(CellExecutor, "_lake_config", lake_config)
    monkeypatch.setattr(CellExecutor, "_pull_from_team_store", no_pull)
    monkeypatch.setattr(executor_module, "publish_cell_outputs", publish)
    monkeypatch.setattr(NotebookSession, "environment_attestation_error", lambda self: None)
    return offered


class TestAFinishedRunThatFails:
    """What Python's and R's shared failure results keep of the run."""

    @staticmethod
    def _run() -> CellExecutionResult:
        return CellExecutionResult(
            cell_id="c",
            success=True,
            stdout="out",
            stderr="err",
            outputs={"x": {"preview": 1}},
            duration_ms=5.0,
            execution_method="warm",
            mutation_warnings=[{"variable": "x"}],
        )

    def _assert_kept(self, failed: CellExecutionResult) -> None:
        run = self._run()
        assert failed.success is False
        assert (failed.cell_id, failed.stdout, failed.stderr) == ("c", "out", "err")
        assert failed.outputs == run.outputs
        assert failed.duration_ms == run.duration_ms
        assert failed.execution_method == run.execution_method

    def test_a_store_failure_keeps_the_run_and_its_mutation_warnings(self):
        failed = _store_failure_result(self._run(), "no table")

        self._assert_kept(failed)
        assert failed.error == "no table"
        assert failed.mutation_warnings == [{"variable": "x"}]

    def test_a_store_failure_with_no_reason_says_so(self):
        assert _store_failure_result(self._run(), None).error == STORE_FAILED

    def test_a_mount_sync_failure_keeps_the_run_and_its_mutation_warnings(self):
        failed = _mount_sync_failure_result(self._run(), OSError("read only"))

        self._assert_kept(failed)
        assert failed.error == (
            "Cell executed successfully but failed to sync read-write mounts: read only"
        )
        assert failed.mutation_warnings == [{"variable": "x"}]


class TestPythonAndR:
    """Store, push, displays; a false store is rejected before the push."""

    async def _run(self, tmp_path, language, source, reader):
        session = _session(tmp_path, ("v", language, source), ("w", language, reader))
        return session, await CellExecutor(session).execute_cell("v", source)

    @pytest.mark.parametrize(("language", "source", "reader", "prefix"), LANGUAGES)
    async def test_the_order(self, tmp_path, steps, language, source, reader, prefix):
        _, result = await self._run(tmp_path, language, source, reader)

        assert result.success, result.error
        expected = [
            "_store_outputs",
            "_push_to_team_store",
            "_store_display_outputs",
            "_store_console_outputs",
        ]
        if language == "python":
            # Its run closes after the store. The one at the end is the dispatcher's, for all.
            expected.insert(1, "_close_open_run")
        assert steps == [*expected, "_close_open_run"]

    async def test_a_leaf_stores_its_console_after_displays_in_python(self, tmp_path, steps):
        session = _session(tmp_path, ("v", "python", PYTHON[1]))

        await CellExecutor(session).execute_cell("v", PYTHON[1])

        assert _after_run(steps)[-2:] == ["_store_display_outputs", "_store_console_outputs"]

    @pytest.mark.parametrize(("language", "source", "reader", "prefix"), LANGUAGES)
    async def test_a_false_store_fails_before_the_push_and_displays(
        self, tmp_path, monkeypatch, steps, language, source, reader, prefix
    ):
        _fail(monkeypatch, "_store_outputs", False, steps)

        session, result = await self._run(tmp_path, language, source, reader)

        assert not result.success
        assert result.error == STORE_FAILED
        assert "_push_to_team_store" not in steps
        assert "_store_display_outputs" not in steps
        assert _latest(session, "v", "__display__0") is None

    @pytest.mark.parametrize(("language", "source", "reader", "prefix"), LANGUAGES)
    async def test_a_table_write_error_is_the_cells_error(
        self, tmp_path, monkeypatch, steps, language, source, reader, prefix
    ):
        _fail(monkeypatch, "_store_outputs", TableWriteError("no table"), steps)

        _, result = await self._run(tmp_path, language, source, reader)

        assert not result.success
        assert result.error == "no table"
        assert "_push_to_team_store" not in steps

    @pytest.mark.parametrize(("language", "source", "reader", "prefix"), LANGUAGES)
    async def test_an_unsafe_run_file_is_this_cells_error(
        self, tmp_path, monkeypatch, steps, language, source, reader, prefix
    ):
        _fail(monkeypatch, "_store_outputs", UnsafeRunFile("a link"), steps)

        _, result = await self._run(tmp_path, language, source, reader)

        assert not result.success
        assert result.error == "a link"
        assert "_push_to_team_store" not in steps

    @pytest.mark.parametrize(("language", "source", "reader", "prefix"), LANGUAGES)
    async def test_a_display_failure_fails_a_cell_whose_outputs_stay(
        self, tmp_path, monkeypatch, steps, language, source, reader, prefix
    ):
        _fail(monkeypatch, "_store_display_outputs", RuntimeError("no display"), steps)

        session, result = await self._run(tmp_path, language, source, reader)

        assert not result.success
        assert result.error == f"{prefix}{'E' if prefix == '' else 'e'}xecution failed: no display"
        # Already stored and offered to the team by then.
        assert steps.index("_push_to_team_store") < steps.index("_store_display_outputs")
        assert _latest(session, "v", "value") == 1

    @pytest.mark.parametrize(("language", "source", "reader", "prefix"), LANGUAGES)
    async def test_a_push_that_raises_leaves_the_cell_finished(
        self, tmp_path, monkeypatch, steps, language, source, reader, prefix
    ):
        offered = _team_store_breaks(monkeypatch)

        session, result = await self._run(tmp_path, language, source, reader)

        assert offered == ["v"]
        assert result.success, result.error
        assert "_store_display_outputs" in steps
        assert _latest(session, "v", "value") == 1

    async def test_a_cancel_during_the_push_is_still_a_cancel(self, tmp_path, monkeypatch, steps):
        _team_store_breaks(monkeypatch, asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await self._run(tmp_path, "python", PYTHON[1], PYTHON[2])

    @pytest.mark.parametrize(
        ("language", "assign"),
        [
            pytest.param("python", "x = 1\n", id="python"),
            pytest.param("r", "x <- 1\n", id="r", marks=skip_if_no_r),
        ],
    )
    async def test_a_mount_that_cannot_sync_back_fails_the_cell(
        self, tmp_path, monkeypatch, language, assign
    ):
        data = tmp_path / "data"
        data.mkdir()
        source = f"# @mount out {data.as_uri()} rw\n{assign}"
        session = _session(tmp_path, ("v", language, source))

        async def no_sync(self, resolved):
            raise OSError("read only")

        monkeypatch.setattr(MountResolver, "sync_back", no_sync)

        result = await CellExecutor(session).execute_cell("v", source)

        assert not result.success
        assert (
            result.error
            == "Cell executed successfully but failed to sync read-write mounts: read only"
        )


class TestOnAWorker:
    @pytest.mark.parametrize(
        ("language", "source"),
        [
            pytest.param("python", "x = 1\n", id="python"),
            pytest.param("r", "x <- 1\n", id="r", marks=skip_if_no_r),
        ],
    )
    async def test_a_false_store_names_its_worker(
        self, tmp_path, monkeypatch, steps, notebook_executor_server, language, source
    ):
        session = _session(tmp_path, ("v", language, source))
        session.notebook_state.workers = [
            WorkerSpec(
                name="remote",
                backend=WorkerBackendType.EXECUTOR,
                config={"url": notebook_executor_server["execute_url"]},
            )
        ]
        session.notebook_state.get_cell("v").worker = "remote"
        _fail(monkeypatch, "_store_outputs", False, steps)

        result = await CellExecutor(session).execute_cell("v", source)

        assert result.error == STORE_FAILED
        assert result.execution_method == "executor"
        assert result.remote_worker == "remote"

    async def test_an_embedded_workers_mount_sync_failure_names_it(self, tmp_path, monkeypatch):
        data = tmp_path / "data"
        data.mkdir()
        source = f"# @mount out {data.as_uri()} rw\nx = 1\n"
        session = _session(tmp_path, ("v", "python", source))
        session.notebook_state.workers = [
            WorkerSpec(
                name="embedded",
                backend=WorkerBackendType.EXECUTOR,
                config={"url": "embedded://local"},
            )
        ]
        session.notebook_state.get_cell("v").worker = "embedded"

        async def no_sync(self, resolved):
            raise OSError("read only")

        monkeypatch.setattr(MountResolver, "sync_back", no_sync)

        result = await CellExecutor(session).execute_cell("v", source)

        assert "failed to sync read-write mounts: read only" in result.error
        assert result.remote_worker == "embedded"


class TestBatch:
    """Store, reject a false store, displays, console, push last."""

    SOURCE = PYTHON[1]
    READER = PYTHON[2]

    async def _run(self, tmp_path):
        session = _session(tmp_path, ("v", "python", self.SOURCE), ("w", "python", self.READER))
        specs = [
            {
                "cell_id": cell_id,
                "source": source,
                "consumed_vars": sorted(session.dag.consumed_variables.get(cell_id, set())),
                "env": {},
                "mount_manifest": {},
                "source_hash": "",
                "env_hash": "",
            }
            for cell_id, source in (("v", self.SOURCE), ("w", self.READER))
        ]
        return session, CellExecutor(session).execute_batch(specs)

    def _first(self, result):
        return next(cell for cell in result.cell_results if cell.cell_id == "v")

    async def test_the_order(self, tmp_path, steps):
        session, running = await self._run(tmp_path)

        result = await running

        assert result.completed
        assert steps[:4] == [
            "_store_outputs",
            "_store_display_outputs",
            "_store_console_outputs",
            "_push_to_team_store",
        ]
        # w has no displays to store.
        assert steps[4:] == ["_store_outputs", "_store_console_outputs", "_push_to_team_store"]
        assert _latest(session, "v", "__display__0") == 1

    async def test_a_false_store_is_rejected_before_the_displays(
        self, tmp_path, monkeypatch, steps
    ):
        _fail(monkeypatch, "_store_outputs", False, steps)
        session, running = await self._run(tmp_path)

        result = await running

        assert result.end_reason == "persist_failed"
        assert result.failed_cell_id == "v"
        assert self._first(result).error == STORE_FAILED
        assert steps == ["_store_outputs"]
        # No display outlives the cell's outputs.
        assert _latest(session, "v", "__display__0") is None
        assert _latest(session, "v", "value") is None

    @pytest.mark.parametrize(
        "error", [UnsafeRunFile("a link"), TableWriteError("no table")], ids=["unsafe", "table"]
    )
    async def test_a_store_that_raises_is_this_cells_error(
        self, tmp_path, monkeypatch, steps, error
    ):
        _fail(monkeypatch, "_store_outputs", error, steps)
        session, running = await self._run(tmp_path)

        result = await running

        assert result.end_reason == "persist_failed"
        assert self._first(result).error == str(error)
        assert steps == ["_store_outputs"]
        assert _latest(session, "v", "__display__0") is None

    async def test_a_display_failure_is_logged_and_the_cell_goes_on(
        self, tmp_path, monkeypatch, steps
    ):
        _fail(monkeypatch, "_store_display_outputs", RuntimeError("no display"), steps)
        session, running = await self._run(tmp_path)

        result = await running

        assert result.completed
        assert self._first(result).status == "ok"
        assert steps.count("_push_to_team_store") == 2
        assert _latest(session, "v", "value") == 1
        assert _latest(session, "v", "__display__0") is None

    async def test_a_push_that_raises_leaves_the_batch_running(self, tmp_path, monkeypatch, steps):
        offered = _team_store_breaks(monkeypatch)
        session, running = await self._run(tmp_path)

        result = await running

        assert result.completed
        assert self._first(result).status == "ok"
        assert offered
        assert _latest(session, "v", "value") == 1
        # The cell after it ran too.
        assert steps.count("_store_outputs") == 2


class TestLoop:
    """Its own staging and finalize, then displays, then a leaf's console. No push."""

    LOOP = '# @loop max_iter=2 carry=state\nstate = state + 1\ndisplay(Markdown("d"))\n'
    SEED = "state = 0\n"

    async def _session(self, tmp_path, *after):
        session = _session(tmp_path, ("s", "python", self.SEED), ("l", "python", self.LOOP), *after)
        await CellExecutor(session).execute_cell("s", self.SEED)
        return session

    async def test_the_order(self, tmp_path, steps):
        session = await self._session(tmp_path)
        steps.clear()

        result = await CellExecutor(session).execute_cell("l", self.LOOP)

        assert result.success, result.error
        assert _after_run(steps) == ["_store_display_outputs", "_store_console_outputs"]

    async def test_a_consumed_carry_stores_its_console_too(self, tmp_path, steps):
        session = await self._session(tmp_path, ("w", "python", "seen = state\n"))
        steps.clear()

        await CellExecutor(session).execute_cell("l", self.LOOP)

        assert _after_run(steps) == ["_store_display_outputs", "_store_console_outputs"]

    @staticmethod
    def _warn_each_iteration(monkeypatch, *, then=None):
        """The harness reports a mutation warning each run; *then* is raised from run 2 on."""
        real = CellExecutor._run_harness
        runs = []

        async def warning(self, *args, **kwargs):
            runs.append(True)
            if then is not None and len(runs) > 1:
                raise then
            result = await real(self, *args, **kwargs)
            result["mutation_warnings"] = [WARNING]
            return result

        monkeypatch.setattr(CellExecutor, "_run_harness", warning)

    async def test_a_display_failure_is_a_failed_result_with_its_console(
        self, tmp_path, monkeypatch, steps
    ):
        loop = (
            "# @loop max_iter=2 carry=state\n"
            "state = state + 1\n"
            'print("said")\n'
            'display(Markdown("d"))\n'
        )
        session = _session(tmp_path, ("s", "python", self.SEED), ("l", "python", loop))
        await CellExecutor(session).execute_cell("s", self.SEED)
        _fail(monkeypatch, "_store_display_outputs", RuntimeError("no display"), steps)
        self._warn_each_iteration(monkeypatch)

        result = await CellExecutor(session).execute_cell("l", loop)

        assert not result.success
        assert result.error == "Loop execution failed: no display"
        assert result.stdout.count("said") == 2
        assert result.mutation_warnings == [WARNING, WARNING]
        # Finalized before the displays, so the outputs stay current.
        assert _latest(session, "l", "state") == 1

    async def test_folders_of_a_finished_loop_are_not_kept_as_a_failed_run(
        self, tmp_path, monkeypatch, steps
    ):
        loop = (
            "# @loop max_iter=1 carry=state\n"
            "# @output_dir out\n"
            "state = state + 1\n"
            "(out / 'a').write_text('a')\n"
            'display(Markdown("d"))\n'
        )
        session = _session(tmp_path, ("s", "python", self.SEED), ("l", "python", loop))
        await CellExecutor(session).execute_cell("s", self.SEED)
        _fail(monkeypatch, "_store_display_outputs", RuntimeError("no display"), steps)
        kept = []
        monkeypatch.setattr(
            CellExecutor, "_keep_failed_output_dirs", lambda self, *args: kept.append(args)
        )

        result = await CellExecutor(session).execute_cell("l", loop)

        assert result.error == "Loop execution failed: no display"
        assert kept == []

    async def test_a_failure_mid_loop_keeps_its_folders_and_console(
        self, tmp_path, monkeypatch, steps
    ):
        loop = (
            "# @loop max_iter=3 carry=state\n"
            "# @output_dir out\n"
            "state = state + 1\n"
            "(out / 'a').write_text('a')\n"
            'print("said")\n'
        )
        session = _session(tmp_path, ("s", "python", self.SEED), ("l", "python", loop))
        await CellExecutor(session).execute_cell("s", self.SEED)
        self._warn_each_iteration(monkeypatch, then=RuntimeError("no process"))
        kept = []
        monkeypatch.setattr(
            CellExecutor, "_keep_failed_output_dirs", lambda self, *args: kept.append(args)
        )

        result = await CellExecutor(session).execute_cell("l", loop)

        assert result.error == "Loop execution failed: no process"
        assert result.stdout.count("said") == 1
        assert result.mutation_warnings == [WARNING]
        assert len(kept) == 1

    async def test_a_cancel_is_still_a_cancel(self, tmp_path, monkeypatch, steps):
        session = await self._session(tmp_path)
        self._warn_each_iteration(monkeypatch, then=asyncio.CancelledError())

        with pytest.raises(asyncio.CancelledError):
            await CellExecutor(session).execute_cell("l", self.LOOP)

    async def test_a_timeout_keeps_the_mutation_warnings(self, tmp_path, monkeypatch, steps):
        session = await self._session(tmp_path)
        self._warn_each_iteration(monkeypatch, then=TimeoutError())

        result = await CellExecutor(session).execute_cell("l", self.LOOP)

        assert "timed out" in result.error
        assert result.mutation_warnings == [WARNING]

    async def test_a_failed_table_write_discards_what_it_staged(self, tmp_path, monkeypatch, steps):
        seed = "import pyarrow as pa\nrows = pa.table({'id': [0]})\n"
        loop = (
            "# @loop max_iter=1 carry=rows\n"
            f"# @write_table rows {(tmp_path / 'wh').as_uri()}#a.t append\n"
            "rows = rows\n"
        )
        session = _session(tmp_path, ("s", "python", seed), ("l", "python", loop))
        await CellExecutor(session).execute_cell("s", seed)
        _fail(monkeypatch, "_write_cell_tables", TableWriteError("no table"), steps)
        self._warn_each_iteration(monkeypatch)

        result = await CellExecutor(session).execute_cell("l", loop)

        assert result.error == "no table"
        assert result.mutation_warnings == [WARNING]
        assert _latest(session, "l", "rows") is None
        manager = session.get_artifact_manager()
        staged = manager.artifact_store.get_artifact(manager.cell_artifact_id("l", "rows"), 1)
        assert staged.state == "failed"


class TestEach:
    """Each instance by the Python path; then the manifests. No push of a manifest."""

    SOURCE = "# @each item in items\nout = item\n"

    async def _session(self, tmp_path):
        items = 'items = {"x": 1, "y": 2}\n'
        session = _session(
            tmp_path,
            ("a", "python", items),
            ("e", "python", self.SOURCE),
            ("w", "python", "seen = out\n"),
        )
        await CellExecutor(session).execute_cell("a", items)
        return session

    async def test_the_order(self, tmp_path, steps):
        session = await self._session(tmp_path)
        steps.clear()

        result = await CellExecutor(session).execute_cell("e", self.SOURCE)

        assert result.success, result.error
        instance = [
            "_store_outputs",
            "_push_to_team_store",
            "_store_display_outputs",
            "_store_console_outputs",
        ]
        assert _after_run(steps) == [
            *instance,
            *instance,
            "_store_each_manifests",
            "_store_console_outputs",
        ]

    @pytest.mark.parametrize(
        ("step", "outcome", "error"),
        [
            ("_store_outputs", False, STORE_FAILED),
            ("_store_outputs", TableWriteError("no table"), "no table"),
            ("_store_outputs", UnsafeRunFile("a link"), "a link"),
            ("_store_display_outputs", RuntimeError("no display"), "Execution failed: no display"),
        ],
        ids=["false", "table", "unsafe", "display"],
    )
    async def test_an_item_that_fails_stops_the_manifests(
        self, tmp_path, monkeypatch, steps, step, outcome, error
    ):
        session = await self._session(tmp_path)
        steps.clear()
        _fail(monkeypatch, step, outcome, steps)

        result = await CellExecutor(session).execute_cell("e", self.SOURCE)

        assert not result.success
        assert result.error == f"Item 'x' failed: {error}"
        assert "_store_each_manifests" not in steps
        assert _latest(session, "e", "out") is None

    async def test_a_push_that_raises_leaves_the_item_good(self, tmp_path, monkeypatch, steps):
        session = await self._session(tmp_path)
        steps.clear()
        offered = _team_store_breaks(monkeypatch)

        result = await CellExecutor(session).execute_cell("e", self.SOURCE)

        assert result.success, result.error
        assert offered
        assert "_store_each_manifests" in steps

    @pytest.mark.parametrize(
        ("error", "reported"),
        [
            (TableWriteError("no table"), "no table"),
            (RuntimeError("no disk"), "Execution failed: no disk"),
        ],
        ids=["table", "other"],
    )
    async def test_a_manifest_failure_is_the_cells(
        self, tmp_path, monkeypatch, steps, error, reported
    ):
        session = await self._session(tmp_path)
        steps.clear()
        _fail(monkeypatch, "_store_each_manifests", error, steps)

        result = await CellExecutor(session).execute_cell("e", self.SOURCE)

        assert not result.success
        assert result.error == reported
        assert _latest(session, "e", "out") is None


class TestARefusedRun:
    @pytest.mark.parametrize(
        ("language", "shows", "edited"),
        [
            pytest.param("python", 'display(Markdown("d"))\n', 'display(Markdown("e"))\n', id="py"),
            pytest.param("r", "plot(1)\n", "plot(2)\n", id="r", marks=skip_if_no_r),
        ],
    )
    async def test_it_takes_the_old_displays_down_and_its_error_outlives_the_session(
        self, tmp_path, monkeypatch, language, shows, edited
    ):
        session = _session(tmp_path, ("c", language, shows))
        first = await CellExecutor(session).execute_cell("c", shows)
        assert len(first.display_outputs) == 1
        monkeypatch.setattr(CellExecutor, "_table_write_refusal", lambda _self, _source: "refused")

        session.notebook_state.get_cell("c").source = edited
        result = await CellExecutor(session).execute_cell("c", edited)

        assert result.error == "refused"
        assert session.notebook_state.get_cell("c").display_outputs == []
        reopened = NotebookSession(parse_notebook(session.path), session.path)
        assert reopened.notebook_state.get_cell("c").display_outputs == []
        assert "refused" in reopened.notebook_state.get_cell("c").error
