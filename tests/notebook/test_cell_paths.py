"""An extracted cell path takes the session as a host and a context, never the executor."""

from __future__ import annotations

import ast
import asyncio
import hashlib
import json
import os
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from vestigo.provenance import derive_subkey

import strata.notebook.cell_paths
from strata.notebook import each
from strata.notebook.annotations import LoopAnnotation
from strata.notebook.cell_paths import batch, cell_tests, fanout, loop, python, r
from strata.notebook.cell_paths.cell_tests import CellTestsContext, run_cell_tests
from strata.notebook.cell_paths.context import (
    STORE_FAILED,
    PathContext,
    RemoteExecutionError,
    artifact_content_type,
)
from strata.notebook.cell_paths.fanout import FanoutContext, run_each_cell, run_fanout_cell
from strata.notebook.cell_paths.loop import LoopContext, run_loop_cell
from strata.notebook.cell_paths.module_export import write_module_export_outputs
from strata.notebook.cell_paths.prompt import run_prompt_cell
from strata.notebook.cell_paths.python import (
    LocalDispatchContext,
    PythonContext,
    dispatch_local,
    run_python_cell,
)
from strata.notebook.cell_paths.r import RContext, run_r_cell
from strata.notebook.cell_paths.sql import run_sql_cell
from strata.notebook.cell_paths.widget import run_widget_cell
from strata.notebook.execution_result import CellExecutionResult
from strata.notebook.executor import CellExecutor
from strata.notebook.harness_user import LocalExecutionRefused, UnsafeRunFile
from strata.notebook.local_runs import RunFenced, RunRefused
from strata.notebook.models import CellOutput, MountMode, MountSpec, WorkerBackendType
from strata.notebook.mounts import MountWrite, ResolvedMount
from strata.notebook.parser import parse_notebook
from strata.notebook.provenance import safe_filename_stem
from strata.notebook.session import NotebookSession
from strata.notebook.table_writes import TableWriteError
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

PATHS = sorted(Path(strata.notebook.cell_paths.__file__).parent.glob("*.py"))


def _outside_type_checking(tree: ast.Module) -> list[ast.stmt]:
    return [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        )
    ]


def executor_uses(source: str) -> list[str]:
    """Each run-time import of the executor module, and each mention of ``CellExecutor``."""
    tree = ast.parse(source)
    found = []
    for statement in _outside_type_checking(tree):
        for node in ast.walk(statement):
            if isinstance(node, ast.ImportFrom) and node.module == "strata.notebook.executor":
                found.append("import")
            elif isinstance(node, ast.Import) and any(
                alias.name == "strata.notebook.executor" for alias in node.names
            ):
                found.append("import")
            elif isinstance(node, ast.Name) and node.id == "CellExecutor":
                found.append("CellExecutor")
            elif isinstance(node, ast.Constant) and node.value == "CellExecutor":
                found.append("CellExecutor")
    return found


def test_the_guard_sees_the_executor():
    assert executor_uses("from strata.notebook.executor import CellExecutor\n") == ["import"]
    assert executor_uses("def f():\n    from strata.notebook.executor import x\n") == ["import"]
    assert executor_uses("import strata.notebook.executor\n") == ["import"]
    assert executor_uses("def f(e: 'CellExecutor'): ...\n") == ["CellExecutor"]
    assert executor_uses("def f():\n    return CellExecutor\n") == ["CellExecutor"]
    typed = "if TYPE_CHECKING:\n    from strata.notebook.executor import _CellProvenance\n"
    assert executor_uses(typed) == []


def test_there_are_paths_to_check():
    assert "widget.py" in {path.name for path in PATHS}


@pytest.mark.parametrize("path", PATHS, ids=lambda path: path.stem)
def test_a_path_does_not_get_the_executor(path):
    assert executor_uses(path.read_text()) == []


async def _nothing_gone(_cell_id, _source):
    return None


async def _no_gone_tables(_source):
    return []


def _raises(error: Exception):
    def raises(*_args, **_kwargs):
        raise error

    return raises


PROVENANCE = SimpleNamespace(
    provenance_hash="prov",
    source_hash="source",
    env_hash="env",
    table_reads=[{"table": "t"}],
    key_parts={"key": "parts"},
    reads=lambda mounts, tables: {"mounts": mounts, "tables": tables},
)
RECORDED = (
    ("c1", "prov", "source", "env"),
    {
        "table_reads": [{"table": "t"}],
        "reads": {"mounts": {"data": 3}, "tables": {"file:///t": 4}},
        "key_parts": {"key": "parts"},
    },
)


def _fakes(cell: SimpleNamespace, failure: CellExecutionResult | None = None):
    """A host and a context with nothing behind them, and a record of what they were asked."""
    calls = SimpleNamespace(recorded=[], materialized=[], cell=cell, persisted=[], applied=[])
    calls.stored_displays = []
    calls.store_inline = []

    def store_inline_display_outputs(cell_id, provenance_hash, displays, **kwargs):
        calls.store_inline.append((cell_id, provenance_hash, displays, kwargs))
        return calls.stored_displays

    async def compute_provenance(cell_id, source):
        return PROVENANCE

    async def materialize_upstreams_or_failure(cell_id, start_time, method):
        calls.materialized.append((cell_id, start_time, method))
        return failure

    host = SimpleNamespace(
        notebook_state=SimpleNamespace(get_cell=lambda cell_id: cell, env={}),
        record_successful_execution_provenance=lambda *args, **kwargs: calls.recorded.append(
            (args, kwargs)
        ),
        persist_display_outputs=lambda cell_id, displays: calls.persisted.append(
            (cell_id, displays)
        ),
        apply_execution_result_metadata=lambda cell_id, result: calls.applied.append(
            (cell_id, result)
        ),
    )
    context = PathContext(
        compute_provenance=compute_provenance,
        moved=lambda cell_id: ({"data": 3}, {"file:///t": 4}),
        materialize_upstreams_or_failure=materialize_upstreams_or_failure,
        store_inline_display_outputs=store_inline_display_outputs,
    )
    return host, context, calls


class TestTheExecutorsContext:
    @pytest.fixture
    def executor(self, tmp_path) -> CellExecutor:
        nb = create_notebook(tmp_path / "nb", "Context")
        add_cell_to_notebook(nb, "c1", None)
        return CellExecutor(NotebookSession(parse_notebook(nb), nb))

    def test_moved_is_what_the_cell_moved(self, executor):
        executor._mounts_moved["c1"] = {"data": 3}
        executor._tables_moved["c1"] = {"file:///t": 4}

        assert executor._path_context().moved("c1") == ({"data": 3}, {"file:///t": 4})

    def test_upstream_materialization_is_the_executors(self, executor):
        context = executor._path_context()

        assert (
            context.materialize_upstreams_or_failure == executor._materialize_upstreams_or_failure
        )

    def test_inline_displays_are_the_executors(self, executor):
        context = executor._path_context()

        assert context.store_inline_display_outputs == executor._store_inline_display_outputs

    def test_a_replaced_method_still_holds(self, executor, monkeypatch):
        async def replaced(cell_id, source):
            return "replaced"

        monkeypatch.setattr(executor, "_compute_cell_provenance", replaced)

        assert executor._path_context().compute_provenance is replaced


class TestWidget:
    """The path on its own: a fake host, a fake context, a stubbed widget executor."""

    @pytest.fixture
    def path(self, monkeypatch):
        """Runs the path with the widget executor answering *answer*; ``calls`` holds the rest."""
        host, context, calls = _fakes(SimpleNamespace(artifact_uris={"old": "uri"}))
        calls.use_cache = []

        async def run(answer, *, use_cache=True):
            def stub(session, cell_id, source, *, use_cache):
                calls.use_cache.append(use_cache)
                return answer

            monkeypatch.setattr("strata.notebook.widget_executor.execute_widget_cell", stub)
            return await run_widget_cell(host, context, "c1", "src", 0.0, use_cache=use_cache)

        run.calls = calls
        return run

    async def test_a_run_publishes_and_records(self, path):
        result = await path(
            {
                "success": True,
                "outputs": {"a": {"artifact_uri": "strata://a"}, "b": {}, "c": "plain"},
                "display_outputs": [{"markdown": "shown"}],
                "cache_hit": True,
                "artifact_uri": "strata://a",
                "execution_method": "stored",
            }
        )

        assert path.calls.cell.artifact_uris == {"a": "strata://a"}
        assert path.calls.recorded == [RECORDED]
        assert result.success is True
        assert result.outputs["a"] == {"artifact_uri": "strata://a"}
        assert result.display_outputs == [{"markdown": "shown"}]
        assert result.cache_hit is True
        assert result.artifact_uri == "strata://a"
        assert result.execution_method == "stored"
        assert result.duration_ms > 0

    async def test_a_failure_records_nothing(self, path):
        result = await path({"success": False, "error": "bad control"})

        assert path.calls.recorded == []
        assert path.calls.cell.artifact_uris == {"old": "uri"}
        assert result.success is False
        assert result.error == "bad control"
        assert result.outputs == {}
        assert result.execution_method == "widget"

    @pytest.mark.parametrize("use_cache", [True, False])
    async def test_use_cache_reaches_the_widget_executor(self, path, use_cache):
        await path({"success": False}, use_cache=use_cache)

        assert path.calls.use_cache == [use_cache]


class TestPrompt:
    """The path on its own: a fake host and context, a stubbed LLM config and prompt executor."""

    @pytest.fixture
    def path(self, monkeypatch):
        def build(*, failure=None, config="llm", answer=None):
            host, context, calls = _fakes(SimpleNamespace(), failure)
            calls.asked = []

            async def stub(session, cell_id, source, llm_config, *, use_cache, on_delta):
                calls.asked.append((session, cell_id, source, llm_config, use_cache, on_delta))
                return answer

            monkeypatch.setattr(
                "strata.notebook.llm.config.llm_config_for_session", lambda session: config
            )
            monkeypatch.setattr("strata.notebook.prompt_executor.execute_prompt_cell", stub)

            async def run(*, materialize_upstreams=False, use_cache=True, on_delta=None):
                return await run_prompt_cell(
                    host,
                    context,
                    "c1",
                    "src",
                    0.0,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                    on_delta=on_delta,
                )

            return run, host, calls

        return build

    async def test_an_answer_is_recorded_under_the_standard_key(self, path):
        answer = {
            "success": True,
            "outputs": {"reply": {"preview": "hi"}},
            "display_outputs": [{"markdown": "hi"}],
            "display_output": {"markdown": "hi"},
            "stdout": "out",
            "stderr": "err",
            "cache_hit": True,
            "duration_ms": 7,
            "execution_method": "cached",
            "artifact_uri": "strata://reply",
            "mutation_warnings": [{"variable": "x"}],
            "validation_retries": 2,
        }
        run, _, calls = path(answer=answer)

        result = await run()

        assert calls.recorded == [RECORDED]
        assert result.success is True
        for field in (
            "outputs",
            "display_outputs",
            "display_output",
            "stdout",
            "stderr",
            "cache_hit",
            "duration_ms",
            "execution_method",
            "artifact_uri",
            "mutation_warnings",
            "validation_retries",
        ):
            assert getattr(result, field) == answer[field], field

    async def test_a_failure_records_nothing(self, path):
        run, _, calls = path(answer={"success": False, "outputs": {}, "error": "no reply"})

        result = await run()

        assert calls.recorded == []
        assert result.error == "no reply"
        assert result.execution_method == "llm"
        assert (result.stdout, result.stderr, result.duration_ms) == ("", "", 0)

    async def test_it_is_asked_with_what_it_was_given(self, path):
        run, host, calls = path(answer={"success": False, "outputs": {}})

        async def on_delta(payload):
            return None

        await run(use_cache=False, on_delta=on_delta)

        assert calls.asked == [(host, "c1", "src", "llm", False, on_delta)]

    async def test_no_llm_config_says_so_without_asking(self, path):
        run, _, calls = path(config=None)

        result = await run()

        assert calls.asked == []
        assert result.success is False
        assert result.error.startswith("LLM not configured.")
        assert result.execution_method == "llm"
        assert result.duration_ms > 0

    async def test_upstreams_run_first_when_asked(self, path):
        run, _, calls = path(answer={"success": False, "outputs": {}})

        await run(materialize_upstreams=True)

        assert calls.materialized == [("c1", 0.0, "prompt")]

    async def test_upstreams_do_not_run_when_not_asked(self, path):
        run, _, calls = path(answer={"success": False, "outputs": {}})

        await run(materialize_upstreams=False)

        assert calls.materialized == []

    async def test_an_upstream_failure_is_the_result(self, path):
        failure = CellExecutionResult(cell_id="c1", success=False, error="upstream")
        run, _, calls = path(failure=failure)

        assert await run(materialize_upstreams=True) is failure
        assert calls.asked == []


async def test_the_executor_hands_the_sql_path_what_it_was_asked(tmp_path, monkeypatch):
    nb = create_notebook(tmp_path / "nb", "Sql")
    add_cell_to_notebook(nb, "c1", None)
    executor = CellExecutor(NotebookSession(parse_notebook(nb), nb))
    seen = {}

    async def stub(host, context, cell_id, source, start_time, **kwargs):
        seen.update(kwargs, host=host)
        return "ran"

    monkeypatch.setattr("strata.notebook.executor.run_sql_cell", stub)

    ran = await executor._execute_sql_cell(
        "c1", "src", 0.0, materialize_upstreams=True, use_cache=False
    )

    assert ran == "ran"
    assert seen == {"host": executor.session, "materialize_upstreams": True, "use_cache": False}


async def test_the_executor_hands_its_delta_callback_to_the_prompt_path(tmp_path, monkeypatch):
    nb = create_notebook(tmp_path / "nb", "Delta")
    add_cell_to_notebook(nb, "c1", None)
    executor = CellExecutor(NotebookSession(parse_notebook(nb), nb))
    seen = {}

    async def stub(host, context, cell_id, source, start_time, **kwargs):
        seen.update(kwargs, host=host)
        return "ran"

    async def on_delta(payload):
        return None

    executor.on_prompt_delta = on_delta
    monkeypatch.setattr("strata.notebook.executor.run_prompt_cell", stub)

    ran = await executor._execute_prompt_cell(
        "c1", "src", 0.0, materialize_upstreams=True, use_cache=False
    )

    assert ran == "ran"
    assert seen == {
        "host": executor.session,
        "materialize_upstreams": True,
        "use_cache": False,
        "on_delta": on_delta,
    }


class TestSql:
    """The path on its own: a fake host and context, a stubbed SQL executor."""

    DISPLAY = {"markdown": "| a |"}
    STORED = {"markdown": "| a |", "artifact_uri": "strata://display"}

    @pytest.fixture
    def path(self, monkeypatch):
        def build(answer=None, *, failure=None):
            host, context, calls = _fakes(SimpleNamespace(cache_hit=None), failure)
            calls.asked = []

            async def stub(session, cell_id, source, *, use_cache):
                calls.asked.append((session, cell_id, source, use_cache))
                return answer

            monkeypatch.setattr("strata.notebook.sql.cell_executor.execute_sql_cell", stub)

            async def run(*, materialize_upstreams=False, use_cache=True):
                return await run_sql_cell(
                    host,
                    context,
                    "c1",
                    "src",
                    0.0,
                    materialize_upstreams=materialize_upstreams,
                    use_cache=use_cache,
                )

            return run, host, calls

        return build

    async def test_a_success_records_the_generic_key_and_its_lake_reads(self, path):
        answer = {
            "success": True,
            "outputs": {"t": {"preview": 1}},
            "table_reads": [{"table": "lake"}],
            "display_outputs": [self.DISPLAY],
            "stdout": "out",
            "stderr": "err",
            "cache_hit": True,
            "execution_method": "cached",
            "artifact_uri": "strata://t",
            "mutation_warnings": [{"variable": "x"}],
        }
        run, _, calls = path(answer)
        calls.stored_displays = [self.STORED]

        result = await run()

        args, kwargs = RECORDED
        assert calls.recorded == [
            (args, {**kwargs, "table_reads": [{"table": "t"}, {"table": "lake"}]})
        ]
        assert calls.cell.cache_hit is True
        assert calls.store_inline == [
            (
                "c1",
                "prov",
                [self.DISPLAY],
                {"source_hash": "source", "source": "src", "env_hash": "env"},
            )
        ]
        assert calls.persisted == [("c1", [self.STORED])]
        assert calls.applied == [("c1", result)]
        assert result.display_outputs == [self.STORED]
        assert result.display_output == self.STORED
        assert result.outputs == answer["outputs"]
        assert (result.stdout, result.stderr) == ("out", "err")
        assert result.cache_hit is True
        assert result.execution_method == "cached"
        assert result.artifact_uri == "strata://t"
        assert result.mutation_warnings == [{"variable": "x"}]
        assert result.duration_ms > 0
        assert isinstance(result.duration_ms, int)

    async def test_nothing_stored_keeps_what_the_run_showed(self, path):
        run, _, calls = path({"success": True, "outputs": {}, "display_outputs": [self.DISPLAY]})

        result = await run()

        assert result.display_outputs == [self.DISPLAY]
        assert calls.persisted == [("c1", [self.DISPLAY])]

    async def test_no_displays_persist_as_none(self, path):
        run, _, calls = path({"success": True, "outputs": {}})

        await run()

        assert calls.persisted == [("c1", None)]

    async def test_a_failure_clears_what_it_showed_and_is_applied(self, path):
        run, _, calls = path({"success": False, "outputs": {}, "error": "bad query"})

        result = await run()

        assert calls.recorded == []
        assert calls.store_inline == []
        assert calls.cell.cache_hit is None
        assert calls.persisted == [("c1", None)]
        assert calls.applied == [("c1", result)]
        assert result.error == "bad query"
        assert result.execution_method == "sql"

    @pytest.mark.parametrize("use_cache", [True, False])
    async def test_it_is_asked_with_what_it_was_given(self, path, use_cache):
        run, host, calls = path({"success": False, "outputs": {}})

        await run(use_cache=use_cache)

        assert calls.asked == [(host, "c1", "src", use_cache)]

    async def test_upstreams_run_first_when_asked(self, path):
        run, _, calls = path({"success": False, "outputs": {}})

        await run(materialize_upstreams=True)

        assert calls.materialized == [("c1", 0.0, "sql")]

    async def test_upstreams_do_not_run_when_not_asked(self, path):
        run, _, calls = path({"success": False, "outputs": {}})

        await run(materialize_upstreams=False)

        assert calls.materialized == []

    async def test_an_upstream_failure_is_the_result(self, path):
        failure = CellExecutionResult(cell_id="c1", success=False, error="upstream")
        run, _, calls = path(failure=failure)

        assert await run(materialize_upstreams=True) is failure
        assert calls.asked == []
        assert calls.applied == []


class TestModuleExport:
    """The step on its own: a fake host, a real export plan, files in a temp directory."""

    DEFS = "def f():\n    return 1\n\n\ndef g():\n    return 2\n"
    CLOSES_OVER_UPSTREAM = "def f():\n    return base + 1\n"
    CLOSES_OVER_RUNTIME = "import os\nroot = os.getcwd()\n\n\ndef f():\n    return root\n"

    @staticmethod
    def _host(consumed, *, references=(), producer=None, cells=(), stored=None):
        cell = SimpleNamespace(id="c1", references=list(references), builtin_references=[])

        def store_cell_output(**kwargs):
            stored.append(kwargs)
            return SimpleNamespace(id="stored", version=3)

        return SimpleNamespace(
            dag=SimpleNamespace(
                consumed_variables={"c1": set(consumed)}, variable_producer=producer or {}
            ),
            notebook_state=SimpleNamespace(id="nb", cells=[cell, *cells]),
            get_artifact_manager=lambda: SimpleNamespace(store_cell_output=store_cell_output),
        )

    def test_no_dag_writes_nothing(self, tmp_path):
        outputs = {}
        host = SimpleNamespace(dag=None)

        assert write_module_export_outputs(host, "c1", self.DEFS, tmp_path, "p", outputs) is None
        assert outputs == {}

    def test_nothing_consumed_writes_nothing(self, tmp_path):
        outputs = {}
        host = self._host([])

        assert write_module_export_outputs(host, "c1", self.DEFS, tmp_path, "p", outputs) is None
        assert outputs == {}
        assert list(tmp_path.iterdir()) == []

    def test_only_what_is_consumed_is_exported(self, tmp_path):
        outputs = {}
        host = self._host(["f"])

        assert write_module_export_outputs(host, "c1", self.DEFS, tmp_path, "p", outputs) is None

        assert set(outputs) == {"f"}
        descriptor = json.loads((tmp_path / outputs["f"]["file"]).read_text())
        assert descriptor["symbol_name"] == "f"
        assert descriptor["provenance_hash"] == derive_subkey("p", "f")
        assert descriptor["injected"] == {}

    def test_an_upstream_value_with_no_artifact_blocks_the_export(self, tmp_path):
        producer_cell = SimpleNamespace(id="up", artifact_uris={})
        host = self._host(
            ["f"], references=["base"], producer={"base": "up"}, cells=[producer_cell]
        )
        outputs = {}

        error = write_module_export_outputs(
            host, "c1", self.CLOSES_OVER_UPSTREAM, tmp_path, "p", outputs
        )

        assert "a value it closes over (`base`)" in error
        assert outputs == {}

    def test_the_injected_reference_names_the_module(self, tmp_path):
        names = []
        for uri in ("strata://artifact/a@v=1", "strata://artifact/a@v=2"):
            producer_cell = SimpleNamespace(id="up", artifact_uris={"base": uri})
            host = self._host(
                ["f"], references=["base"], producer={"base": "up"}, cells=[producer_cell]
            )
            outputs = {}
            assert (
                write_module_export_outputs(
                    host, "c1", self.CLOSES_OVER_UPSTREAM, tmp_path, "p", outputs
                )
                is None
            )
            descriptor = json.loads((tmp_path / outputs["f"]["file"]).read_text())
            assert descriptor["injected"] == {"base": uri}
            names.append(descriptor["module_name"])

        assert names[0] != names[1]

    def test_a_same_cell_value_is_stored_and_pinned(self, tmp_path):
        (tmp_path / "root.json").write_bytes(b'"here"')
        outputs = {"root": {"file": "root.json", "content_type": "json/object"}}
        stored = []
        host = self._host(["f"], stored=stored)

        assert (
            write_module_export_outputs(
                host, "c1", self.CLOSES_OVER_RUNTIME, tmp_path, "p", outputs
            )
            is None
        )

        assert stored[0]["provenance_hash"] == derive_subkey("p", "root")
        assert stored[0]["blob_data"] == b'"here"'
        descriptor = json.loads((tmp_path / outputs["f"]["file"]).read_text())
        assert descriptor["injected"] == {"root": "strata://artifact/stored@v=3"}

    @pytest.mark.parametrize(
        "spec",
        [
            pytest.param({"file": "root.json", "error": "too big"}, id="errored"),
            pytest.param({"file": "gone.json"}, id="gone"),
            pytest.param({"error": "no file"}, id="no-file"),
        ],
    )
    def test_a_same_cell_value_that_cannot_be_stored_blocks_the_export(self, tmp_path, spec):
        (tmp_path / "root.json").write_bytes(b'"here"')
        stored = []
        host = self._host(["f"], stored=stored)

        error = write_module_export_outputs(
            host, "c1", self.CLOSES_OVER_RUNTIME, tmp_path, "p", {"root": spec}
        )

        assert "a value it closes over (`root`)" in error
        assert stored == []


class TestCellTests:
    """The path on its own: a fake host and context, a stubbed pytest run."""

    RAW = {"passed": 1, "failed": 0, "errored": 0, "skipped": 0, "tests": []}

    @staticmethod
    def _provenance(**overrides):
        fields = {
            "fetch_error": None,
            "dataset_error": None,
            "fetched": {},
            "datasets": [],
            "table_snapshots": {},
            "input_hashes": ["input"],
            "mount_fingerprints": ["mount"],
            "table_fingerprints": ["table"],
            "fetch_fingerprints": ["fetch"],
            "dataset_fingerprints": ["dataset"],
        }
        fields.update(overrides)

        def listed(listing):
            return SimpleNamespace(
                **fields,
                over_mounts=lambda mounts: listed("as read"),
                reads=lambda mounts, tables: {"listing": listing, "mounts": mounts},
            )

        return listed("as keyed")

    @pytest.fixture
    def path(self, tmp_path, monkeypatch):
        calls = SimpleNamespace(order=[], handed=[], recorded=[], fingerprint=[], ran=[])

        def build(prov):
            cell = SimpleNamespace(id="c1", source="x = 1\n", mounts=[])
            host = SimpleNamespace(
                notebook_state=SimpleNamespace(
                    get_cell=lambda cell_id: cell if cell_id == "c1" else None, cells=[cell]
                ),
                venv_python=Path("python"),
                path=tmp_path,
                # Not analyzed: nothing to say the cell is an inactive variant.
                analysis=None,
                record_cell_test_inputs=lambda cell_id, inputs, env: (
                    calls.fingerprint.append(inputs) or "fingerprint"
                ),
                cell_env_hash=lambda cell: "env",
                cell_test_fingerprint=lambda *args: ("source", "tests", None),
            )

            async def materialize_upstreams(cell_id):
                calls.order.append("upstreams")

            async def compute_provenance(cell_id, source, **kwargs):
                calls.order.append("provenance")
                return prov

            async def prepare_mounts(specs, cell_id):
                return {}

            context = CellTestsContext(
                materialize_upstreams=materialize_upstreams,
                forget_moved=lambda cell_id: calls.order.append("forget"),
                compute_provenance=compute_provenance,
                moved=lambda cell_id: ({"data": 3}, {}),
                manifest_tables=lambda tables, snapshots: {},
                prepare_mounts=prepare_mounts,
                resolve_runtime_env=lambda cell_id, env: {},
                fanout_variants=lambda cell_id: (),
                load_input_blobs=lambda cell_id, directory, fanout_variant: {},
                add_dataset_inputs=lambda specs, datasets, directory: None,
                harness_env=lambda env: {},
            )

            def stub(**kwargs):
                calls.ran.append(kwargs)
                return self.RAW

            monkeypatch.setattr(cell_tests, "resolve_harness_user", lambda: None)
            monkeypatch.setattr(cell_tests, "prepare_work_dir", lambda path, user: None)
            monkeypatch.setattr(cell_tests, "let_through", lambda path, user: None)
            monkeypatch.setattr(
                cell_tests, "hand_over", lambda path, user: calls.handed.append(path.name)
            )
            monkeypatch.setattr("strata.notebook.cell_test_runner.run_cell_tests_in_dir", stub)
            monkeypatch.setattr(
                "strata.notebook.runtime_state.persist_cell_test_result", lambda *args: None
            )
            monkeypatch.setattr(
                "strata.notebook.reads_index.record",
                lambda path, cell_id, reads, live: calls.recorded.append((cell_id, reads, live)),
            )

            async def run(cell_id="c1"):
                return await run_cell_tests(host, context, cell_id, "def test_x(cell): ...\n")

            return run

        build.calls = calls
        return build

    async def test_an_unknown_cell_is_not_found(self, path):
        with pytest.raises(FileNotFoundError, match="Cell gone not found"):
            await path(self._provenance())("gone")

    async def test_the_last_runs_moved_bytes_go_before_the_inputs_are_read(self, path):
        await path(self._provenance())()

        assert path.calls.order == ["upstreams", "forget", "provenance"]

    async def test_a_run_records_what_it_read_from_the_listing_it_read(self, path):
        result = await path(self._provenance())()

        assert result.passed == 1
        assert path.calls.recorded == [
            ("c1", {"listing": "as read", "mounts": {"data": 3}}, ["c1"])
        ]

    async def test_the_fingerprint_counts_every_input(self, path):
        result = await path(self._provenance())()

        assert path.calls.fingerprint == [["input", "mount", "table", "fetch", "dataset"]]
        assert result.input_fingerprint == "fingerprint"

    async def test_the_inputs_are_handed_to_the_harness_user(self, path):
        await path(self._provenance())()

        assert path.calls.handed == ["inputs-0"]

    @pytest.mark.parametrize(
        "problem",
        [
            pytest.param({"fetch_error": "no fetch"}, id="fetch"),
            pytest.param({"dataset_error": "no dataset"}, id="dataset"),
        ],
    )
    async def test_an_input_it_cannot_read_is_refused_without_recording_reads(self, path, problem):
        result = await path(self._provenance(**problem))()

        assert path.calls.ran == []
        assert path.calls.recorded == []
        assert result.errored == 1
        assert result.tests[0].name == "<refused>"
        assert result.tests[0].message == next(iter(problem.values()))


class TestLoop:
    """The moved path against fakes; the harness-backed behavior is in ``test_executor.py``."""

    LOOP = LoopAnnotation(max_iter=2, carry="state")

    @staticmethod
    def _host(tmp_path):
        cell = SimpleNamespace(worker=None, mutation_defines=[], display_outputs=[])
        return SimpleNamespace(
            notebook_state=SimpleNamespace(worker=None, get_cell=lambda _id: cell),
            path=tmp_path,
            dag=None,
            venv_python=None,
            get_artifact_manager=lambda: None,
        )

    @staticmethod
    def _context(**overrides):
        """A context whose run reaches the first iteration's harness, which times out."""

        async def nothing(*_args, **_kwargs):
            return None

        async def no_rw_mount(_specs):
            return [], False

        async def no_mounts(*_args, **_kwargs):
            return {}

        prov = SimpleNamespace(
            provenance_hash="p",
            env_hash="e",
            source_hash="s",
            table_snapshots={},
        )
        prov.over_mounts = lambda _mounts: prov

        async def provenance(*_args, **_kwargs):
            return prov

        async def harness_times_out(*_args, **_kwargs):
            raise TimeoutError

        @contextmanager
        def run_token(_timeout):
            yield "token"

        unused = lambda *_args, **_kwargs: pytest.fail("not reached")  # noqa: E731
        fields = {
            "materialize_upstreams_or_failure": nothing,
            "resolve_cell_mount_specs": lambda *_args: [],
            "fingerprint_mounts": no_rw_mount,
            "resolve_runtime_env": lambda *_args: {},
            "compute_provenance": provenance,
            "manifest_tables": lambda *_args: {},
            "moved": unused,
            "promote_reverted_outputs": unused,
            "restore_gone_tables": _nothing_gone,
            "canonical_output": unused,
            "table_write_refusal": lambda _source: None,
            "prepare_mounts": no_mounts,
            "load_input_blobs": lambda *_args: {},
            "keep_failed_output_dirs": lambda *_args: None,
            "write_manifest": lambda *_args, **_kwargs: Path("manifest.json"),
            "run_token": run_token,
            "run_harness": harness_times_out,
            "input_refs": unused,
            "write_cell_tables": unused,
            "store_display_outputs": unused,
            "store_console_outputs": unused,
            "folder_listings": {},
            "on_iteration_complete": None,
        }
        return LoopContext(**{**fields, **overrides})

    @staticmethod
    def _reach_the_harness(monkeypatch):
        monkeypatch.setattr(loop, "_resolve_loop_seed", lambda *_args: (("seed", 1), "json/object"))
        monkeypatch.setattr(loop, "_write_carry", lambda *_args: "digest")
        monkeypatch.setattr(
            "strata.notebook.runtime_state.persist_cell_loop_run", lambda *_args: None
        )

    async def _run(
        self, tmp_path, context, source="# @loop max_iter=2 carry=state\n", folders=None
    ):
        return await run_loop_cell(
            self._host(tmp_path),
            context,
            "c",
            source,
            self.LOOP,
            1.0,
            0.0,
            materialize_upstreams=False,
            use_cache=False,
            folders=folders,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("variables", "message"),
        [
            ({"state": {"content_type": "json/object"}}, "without a 'file' entry"),
            (
                {"state": {"content_type": "json/object", "file": "gone.json"}},
                "carry file not produced by harness: gone.json",
            ),
        ],
    )
    async def test_a_carry_with_no_file_is_a_failed_result(
        self, tmp_path, monkeypatch, variables, message
    ):
        self._reach_the_harness(monkeypatch)

        async def harness(*_args, **_kwargs):
            return {
                "success": True,
                "stdout": "said",
                "stderr": "warned",
                "variables": variables,
                "mutation_warnings": ["careful"],
            }

        kept = []
        folders = tmp_path / "folders"
        folders.mkdir()
        result = await self._run(
            tmp_path,
            self._context(
                run_harness=harness,
                keep_failed_output_dirs=lambda *args: kept.append(args),
            ),
            folders=folders,
        )

        assert result.success is False
        assert message in result.error
        assert result.error.startswith("Loop cell iter 0 ")
        assert (result.stdout, result.stderr) == ("said", "warned")
        assert result.mutation_warnings == ["careful"]
        assert result.execution_method == "loop"
        assert len(kept) == 1

    @pytest.mark.asyncio
    async def test_an_rw_mount_is_refused_before_anything_is_keyed(self, tmp_path):
        async def rw_mount(_specs):
            return [], True

        async def not_asked(*_args, **_kwargs):
            pytest.fail("a refused loop computes no provenance")

        result = await self._run(
            tmp_path, self._context(fingerprint_mounts=rw_mount, compute_provenance=not_asked)
        )
        assert result.success is False
        assert "rw mounts" in result.error
        assert result.execution_method == "loop"

    @pytest.mark.asyncio
    async def test_a_refused_table_write_runs_nothing(self, tmp_path):
        async def not_run(*_args, **_kwargs):
            pytest.fail("a refused write runs no iteration")

        result = await self._run(
            tmp_path,
            self._context(table_write_refusal=lambda _source: "no writes", run_harness=not_run),
        )
        assert result.success is False
        assert result.error == "no writes"
        assert result.execution_method == "loop"

    @pytest.mark.asyncio
    async def test_a_run_drops_the_folder_listings_of_the_last(self, tmp_path, monkeypatch):
        self._reach_the_harness(monkeypatch)
        listings = {"c": {"old": []}, "other": {"kept": []}}
        await self._run(tmp_path, self._context(folder_listings=listings))
        assert listings == {"other": {"kept": []}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("refused", [False, True])
    async def test_folders_go_to_the_user_cells_run_as(self, tmp_path, monkeypatch, refused):
        self._reach_the_harness(monkeypatch)
        user = object()

        def resolve_user():
            if refused:
                raise LocalExecutionRefused("no user")
            return user

        handed = []
        monkeypatch.setattr(loop, "resolve_harness_user", resolve_user)
        monkeypatch.setattr(loop, "hand_over", lambda path, to: handed.append((path, to)))
        folders = tmp_path / "folders"
        folders.mkdir()
        await self._run(
            tmp_path,
            self._context(),
            source="# @loop max_iter=2 carry=state\n# @output_dir out\n",
            folders=folders,
        )
        # A refused run still hands over; the refusal is reported where the cell would start.
        assert handed == [(folders, None if refused else user)]

    @pytest.mark.parametrize(
        ("params", "expected"),
        [
            ({"content_type": "arrow/ipc"}, "arrow/ipc"),
            ({"content_type": ""}, "pickle/object"),
            ({}, "pickle/object"),
        ],
    )
    def test_an_artifact_without_a_content_type_reads_as_pickle(self, params, expected):
        assert artifact_content_type(SimpleNamespace(params=params)) == expected

    def test_the_seed_comes_from_the_upstream_wired_for_the_carry(self):
        """An upstream that once defined the carry keeps a row; the wiring names the definer."""
        cells = {
            "c": SimpleNamespace(upstream_ids=["shadowed", "definer"]),
            "shadowed": SimpleNamespace(),
            "definer": SimpleNamespace(),
        }
        store = SimpleNamespace(
            get_latest_version=lambda artifact_id: SimpleNamespace(
                state="ready", version=1, params={"content_type": "json/object"}
            ),
            blob_exists=lambda *_args: True,
        )
        manager = SimpleNamespace(artifact_store=store)
        host = SimpleNamespace(
            notebook_state=SimpleNamespace(id="n", get_cell=cells.get),
            get_artifact_manager=lambda: manager,
            wired_variables=lambda _cell, upstream: {"state"} if upstream == "definer" else set(),
        )
        carry, content_type = loop._resolve_loop_seed(host, "c", self.LOOP)
        assert carry == ("nb_n_cell_definer_var_state", 1)
        assert content_type == "json/object"

    @pytest.fixture
    def seeded(self, tmp_path):
        def make(loop_source):
            notebook = create_notebook(tmp_path, "loop")
            add_cell_to_notebook(notebook, "seed", None)
            write_cell(notebook, "seed", "state = {'n': 0}\n")
            add_cell_to_notebook(notebook, "lp", after_cell_id="seed")
            write_cell(notebook, "lp", loop_source)
            session = NotebookSession(parse_notebook(notebook), notebook)
            session.refresh_environment_runtime()
            return session

        return make

    @pytest.mark.asyncio
    async def test_a_carry_is_copied_and_read_whole_only_to_store_the_final_state(
        self, seeded, monkeypatch
    ):
        source = "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\n"
        session = seeded(source)
        assert (await CellExecutor(session).execute_cell("seed", "state = {'n': 0}\n")).success
        store = session.get_artifact_manager().artifact_store
        read = store.read_blob
        whole: list[str] = []

        def recording(artifact_id, *args, **kwargs):
            whole.append(artifact_id)
            return read(artifact_id, *args, **kwargs)

        monkeypatch.setattr(store, "read_blob", recording)

        result = await CellExecutor(session).execute_cell("lp", source)

        assert result.success, result.error
        carries = [name for name in whole if "_var_state" in name]
        assert carries == [session.get_artifact_manager().cell_artifact_id("lp", "state", 2)]
        final = session.get_artifact_manager().cell_artifact_id("lp", "state")
        latest = store.get_latest_version(final)
        assert json.loads(read(final, latest.version)) == {"n": 3}

    def test_a_seed_with_no_blob_is_named(self):
        store = SimpleNamespace(
            get_latest_version=lambda _id: SimpleNamespace(state="ready", version=2, params={}),
            blob_exists=lambda *_args: False,
        )
        manager = SimpleNamespace(artifact_store=store)
        cells = {"c": SimpleNamespace(upstream_ids=["d"]), "d": SimpleNamespace()}
        host = SimpleNamespace(
            notebook_state=SimpleNamespace(id="n", get_cell=cells.get),
            get_artifact_manager=lambda: manager,
            wired_variables=lambda *_args: {"state"},
        )

        with pytest.raises(
            ValueError, match="Loop seed blob missing for nb_n_cell_d_var_state@v=2"
        ):
            loop._resolve_loop_seed(host, "c", self.LOOP)

    def test_a_carrys_digest_is_of_its_bytes(self, tmp_path):
        def copy_ready_blob(_id, _version, path):
            path.write_bytes(b"bytes")

        manager = SimpleNamespace(copy_ready_blob=copy_ready_blob)

        digest = loop._write_carry(manager, ("a", 1), tmp_path / "x", None)

        assert digest == hashlib.sha256(b"bytes").hexdigest()

    def test_a_known_digest_is_not_read_again(self, tmp_path, monkeypatch):
        def copy_ready_blob(_id, _version, path):
            path.write_bytes(b"bytes")

        def no_digest(*_args):
            raise AssertionError("read the carry to digest it")

        monkeypatch.setattr(loop.hashlib, "file_digest", no_digest)
        manager = SimpleNamespace(copy_ready_blob=copy_ready_blob)

        assert loop._write_carry(manager, ("a", 1), tmp_path / "x", "known") == "known"

    def test_a_carry_replaces_the_seed_already_there(self, tmp_path):
        """A clone cannot replace a file: left there, the carry is always a full copy."""
        path = tmp_path / "x"
        path.write_bytes(b"the seed")

        def copy_ready_blob(_id, _version, target):
            assert not target.exists()
            target.write_bytes(b"the carry")

        manager = SimpleNamespace(copy_ready_blob=copy_ready_blob)

        loop._write_carry(manager, ("a", 1), path, None)

        assert path.read_bytes() == b"the carry"

    @pytest.mark.asyncio
    async def test_each_iterations_key_chains_on_the_last_ones_bytes(self, seeded, monkeypatch):
        source = "# @loop max_iter=3 carry=state\nstate = {'n': state['n'] + 1}\n"
        session = seeded(source)
        assert (await CellExecutor(session).execute_cell("seed", "state = {'n': 0}\n")).success
        chained: dict[int, str] = {}
        derive = loop.derive_subkey

        def spying(*parts):
            if len(parts) == 3 and parts[2].startswith("iter="):
                chained[int(parts[2].removeprefix("iter="))] = parts[1]
            return derive(*parts)

        monkeypatch.setattr(loop, "derive_subkey", spying)
        manager = session.get_artifact_manager()
        stored: dict[int, bytes] = {}
        stage = manager.store_cell_output

        def store(**kwargs):
            if kwargs["variable_name"] == "state":
                stored[kwargs["iteration"]] = kwargs["blob_data"]
            return stage(**kwargs)

        monkeypatch.setattr(manager, "store_cell_output", store)

        result = await CellExecutor(session).execute_cell("lp", source)

        assert result.success, result.error
        assert sorted(stored) == [0, 1, 2]
        assert chained[1] == hashlib.sha256(stored[0]).hexdigest()
        assert chained[2] == hashlib.sha256(stored[1]).hexdigest()

    @pytest.mark.asyncio
    async def test_a_carry_gone_before_the_next_iteration_fails_the_cell(self, seeded, monkeypatch):
        source = "# @loop max_iter=2 carry=state\nstate = {'n': state['n'] + 1}\n"
        session = seeded(source)
        assert (await CellExecutor(session).execute_cell("seed", "state = {'n': 0}\n")).success
        manager = session.get_artifact_manager()
        monkeypatch.setattr(manager, "copy_blob", lambda *_args: False)

        result = await CellExecutor(session).execute_cell("lp", source)

        assert not result.success
        assert "Blob data not found" in result.error

    @pytest.mark.asyncio
    async def test_the_final_carry_records_what_it_read(self, seeded):
        source = "# @loop max_iter=2 carry=state\nstate = {'n': state['n'] + 1}\n"
        session = seeded(source)
        executor = CellExecutor(session)
        assert (await executor.execute_cell("lp", source)).success
        manager = session.get_artifact_manager()
        final = manager.artifact_store.get_latest_version(manager.cell_artifact_id("lp", "state"))
        read = json.loads(final.input_versions)
        assert [uri for uri in read if "_cell_seed_var_state@" in uri] == list(read)
        assert read

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("annotation", "stored"), [("", 1), ("# @nocache\n", 0)])
    async def test_a_leaf_keeps_its_console_unless_nocache(self, seeded, annotation, stored):
        source = f"{annotation}# @loop max_iter=1 carry=state\nstate = dict(state)\nprint('hi')\n"
        session = seeded(source)
        executor = CellExecutor(session)
        calls = []
        real = executor._store_console_outputs

        def spy(*args, **kwargs):
            calls.append(args[0])
            return real(*args, **kwargs)

        executor._store_console_outputs = spy
        result = await executor.execute_cell("lp", source)
        assert result.success, result.error
        assert calls.count("lp") == stored


@pytest.mark.parametrize(
    ("method", "runner", "extra"),
    [
        ("_execute_fanout_cell", "run_fanout_cell", {"variant_names": ("a",)}),
        ("_execute_each_cell", "run_each_cell", {}),
        ("_execute_r_cell", "run_r_cell", {}),
    ],
)
async def test_the_executor_hands_a_path_what_it_was_asked(
    tmp_path, monkeypatch, method, runner, extra
):
    nb = create_notebook(tmp_path / "nb", "Fanout")
    add_cell_to_notebook(nb, "c1", None)
    executor = CellExecutor(NotebookSession(parse_notebook(nb), nb))
    seen = {}

    async def stub(host, context, cell_id, source, timeout, start_time, **kwargs):
        seen.update(kwargs, host=host)
        return "ran"

    monkeypatch.setattr(f"strata.notebook.executor.{runner}", stub)

    ran = await getattr(executor, method)(
        "c1", "src", 1.0, 0.0, materialize_upstreams=True, use_cache=False, **extra
    )

    assert ran == "ran"
    assert seen == {
        "host": executor.session,
        "materialize_upstreams": True,
        "use_cache": False,
        **extra,
    }


class TestFanout:
    """The moved path against fakes; the harness-backed behavior is in ``test_each.py``."""

    @staticmethod
    def _base(**overrides):
        fields = {
            "provenance_hash": "prov",
            "source_hash": "source",
            "env_hash": "env",
            "table_reads": [],
            "key_parts": {},
            "effective_worker": None,
            "has_rw_mount": False,
            "mount_specs": [],
            "reads": lambda *_args: {},
            "over_mounts": lambda _prepared: pytest.fail("not over a mount"),
        }
        return SimpleNamespace(**{**fields, **overrides})

    @staticmethod
    def _fakes(base, *, consumed=(), latest=None, finalized=(), **overrides):
        """A host and a context with nothing behind them, and a record of what they were asked."""
        calls = SimpleNamespace(
            recorded=[], applied=[], instances=[], staged=[], discarded=[], persisted=[]
        )
        cell = SimpleNamespace(
            artifact_uris={}, artifact_uri=None, cache_hit=False, last_provenance_hash=None
        )
        calls.cell = cell

        def stage_cell_output(**kwargs):
            calls.staged.append(kwargs)
            return f"staged-{kwargs['variable_name']}"

        manager = SimpleNamespace(
            artifact_store=SimpleNamespace(get_latest_version=lambda _id: latest),
            cell_artifact_id=lambda cell_id, var, variant=None: f"{cell_id}.{var}.{variant}",
            load_artifact_data=lambda _id, _version: each.encode_manifest("m", []),
            stage_cell_output=stage_cell_output,
            finalize_cell_outputs=lambda _staged: list(finalized),
            discard_cell_outputs=calls.discarded.append,
        )
        host = SimpleNamespace(
            notebook_state=SimpleNamespace(get_cell=lambda _id: cell),
            dag=SimpleNamespace(
                consumed_variables={"c": set(consumed)},
                read_producer=lambda _cell_id, _source: None,
            ),
            get_artifact_manager=lambda: manager,
            resolve_cached_console=lambda *_args, **_kwargs: None,
            record_successful_execution_provenance=lambda *args, **kwargs: calls.recorded.append(
                (args, kwargs)
            ),
            apply_execution_result_metadata=lambda cell_id, result: calls.applied.append(
                (cell_id, result)
            ),
            persist_display_outputs=lambda cell_id, displays: calls.persisted.append(
                (cell_id, displays)
            ),
            stored_display_outputs=lambda _artifacts: [],
        )

        async def compute_provenance(*_args, **_kwargs):
            return base

        async def execute_python_cell(cell_id, *_args, **kwargs):
            calls.instances.append(kwargs)
            return CellExecutionResult(cell_id=cell_id, success=True)

        async def nothing(*_args, **_kwargs):
            return None

        fields = {
            "materialize_upstreams": nothing,
            "compute_provenance": compute_provenance,
            "execute_python_cell": execute_python_cell,
            "moved": lambda _cell_id: ({}, {}),
            "promote_reverted_outputs": lambda *_args: None,
            "gone_tables": _no_gone_tables,
            "write_back": lambda *_args: None,
            "store_console_outputs": lambda *_args, **_kwargs: None,
            "prepare_mounts": nothing,
            "input_refs": lambda _cell_id: {"in": "strata://artifact/in@v=1"},
            "write_cell_tables": lambda *_args: None,
            "variant_writes": {},
            "each_items": {},
            "each_listed": {},
            "branch_inits": {},
            "on_variant_complete": None,
        }
        return host, FanoutContext(**{**fields, **overrides}), calls

    @staticmethod
    async def _run_fanout(host, context, source="", variants=("a",), use_cache=True):
        return await run_fanout_cell(
            host,
            context,
            "c",
            source,
            1.0,
            0.0,
            materialize_upstreams=False,
            use_cache=use_cache,
            variant_names=variants,
        )

    @pytest.fixture
    def local_worker(self, monkeypatch):
        monkeypatch.setattr(
            fanout,
            "resolve_worker_spec",
            lambda *_args: SimpleNamespace(backend=WorkerBackendType.LOCAL),
        )

    @staticmethod
    def _over_data(tmp_path, uri="file:///data", versions=None):
        """A provenance over a read-only ``data`` mount and a read-write ``out`` one."""
        specs = [
            SimpleNamespace(name="data", uri=uri, mode=MountMode.READ_ONLY),
            SimpleNamespace(name="out", uri="file:///out", mode=MountMode.READ_WRITE),
        ]
        prepared = {
            "data": SimpleNamespace(local_path=tmp_path, file_versions=versions, digests_path=None),
            "out": SimpleNamespace(local_path=tmp_path, file_versions=None, digests_path=None),
        }

        async def prepare_mounts(_specs, _cell_id, only=None):
            prepare_mounts.only = only
            return prepared

        return specs, prepared, prepare_mounts

    @pytest.mark.parametrize(("mode", "tail"), [("overwrite", ""), ("sync", " on=id")])
    async def test_a_write_that_replaces_runs_no_variant(self, mode, tail):
        async def not_asked(*_args, **_kwargs):
            pytest.fail("a refused cell runs nothing")

        host, context, _calls = self._fakes(self._base(), execute_python_cell=not_asked)

        result = await self._run_fanout(host, context, f"# @write_table out t {mode}{tail}\n")

        assert result.success is False
        assert f"@write_table t {mode} is not supported" in result.error

    async def test_an_instance_runs_under_the_cells_key_with_its_upstreams_done(self):
        base = self._base()
        host, context, calls = self._fakes(base)

        await self._run_fanout(host, context, variants=("a", "b"), use_cache=False)

        assert calls.instances == [
            {
                "materialize_upstreams": False,
                "use_cache": False,
                "fanout_variant": name,
                "cell_provenance": base,
            }
            for name in ("a", "b")
        ]

    async def test_writes_of_an_earlier_run_are_not_this_ones(self):
        stale = MountWrite(mount="out", uri="file:///out", files=[])
        host, context, calls = self._fakes(self._base(), variant_writes={"c": [stale]})

        await self._run_fanout(host, context)

        assert calls.recorded[0][1]["writes"] is None

    async def test_a_callback_that_raises_does_not_fail_the_cell(self):
        async def raises(_payload):
            raise RuntimeError("the socket went away")

        host, context, _calls = self._fakes(self._base(), on_variant_complete=raises)

        result = await self._run_fanout(host, context)

        assert result.success is True

    async def test_no_instances_is_not_a_hit(self):
        host, context, _calls = self._fakes(self._base())

        result = await self._run_fanout(host, context, variants=())

        assert result.success is True
        assert result.cache_hit is False

    async def test_the_cells_result_is_applied(self):
        host, context, calls = self._fakes(self._base())

        result = await self._run_fanout(host, context)

        assert calls.applied == [("c", result)]

    @staticmethod
    async def _run_each(host, context, source="# @each path in data\n"):
        return await run_each_cell(
            host, context, "c", source, 1.0, 0.0, materialize_upstreams=False, use_cache=True
        )

    @pytest.mark.usefixtures("local_worker")
    async def test_the_key_finding_its_manifests_is_a_hit(self):
        manifest = SimpleNamespace(id="m", version=3)
        host, context, calls = self._fakes(
            self._base(), consumed=["out"], latest=manifest, promote_reverted_outputs=lambda *_: 1
        )

        result = await self._run_each(host, context)

        assert result.execution_method == "cached"
        assert calls.cell.cache_hit is True
        assert calls.cell.artifact_uris == {"out": "strata://artifact/m@v=3"}
        assert calls.cell.artifact_uri == "strata://artifact/m@v=3"

    @pytest.mark.usefixtures("local_worker")
    async def test_a_source_it_cannot_list_is_the_result(self):
        host, context, _calls = self._fakes(self._base())

        result = await self._run_each(host, context)

        assert result.success is False
        assert "not a mount or a variable" in result.error
        assert result.execution_method == "fanout"

    @pytest.mark.usefixtures("local_worker")
    async def test_what_instances_read_lasts_as_long_as_they_do(self, tmp_path, monkeypatch):
        specs, _prepared, prepare_mounts = self._over_data(tmp_path)
        base = self._base(mount_specs=specs, over_mounts=lambda _prepared: base)
        during = {}

        async def execute_python_cell(cell_id, *_args, **_kwargs):
            during["items"] = set(context.each_items[cell_id])
            during["listed"] = set(context.each_listed[cell_id])
            raise RuntimeError("the instance died")

        host, context, _calls = self._fakes(
            base, prepare_mounts=prepare_mounts, execute_python_cell=execute_python_cell
        )
        monkeypatch.setattr(
            each, "items_of_mount", lambda *_args, **_kwargs: [each.EachItem("a.csv", "digest")]
        )

        with pytest.raises(RuntimeError, match="the instance died"):
            await self._run_each(host, context)

        assert during == {"items": {each.instance_name("a.csv")}, "listed": {"data"}}
        assert context.each_items == {}
        assert context.each_listed == {}

    @pytest.mark.parametrize("uri", ["s3://bucket/data", "file:///data"])
    async def test_a_listing_reads_only_the_glob_and_keys_items_as_the_mount_was_keyed(
        self, tmp_path, monkeypatch, uri
    ):
        specs, prepared, prepare_mounts = self._over_data(tmp_path, uri, versions={"a.csv": "v1"})
        over = self._base()
        over_mounts = []
        base = self._base(
            mount_specs=specs, over_mounts=lambda mounts: over_mounts.append(mounts) or over
        )
        listed = {}

        def items_of_mount(root, glob, **kwargs):
            listed.update(kwargs, root=root, glob=glob)
            return []

        host, context, _calls = self._fakes(base, prepare_mounts=prepare_mounts)
        monkeypatch.setattr(each, "items_of_mount", items_of_mount)

        items, keyed = await fanout._list_each_items(
            host, context, "c", {"source": "data", "glob": "*.csv"}, base
        )

        assert items == []
        assert prepare_mounts.only == {"data": "*.csv"}
        assert listed == {
            "root": tmp_path,
            "glob": "*.csv",
            "versions": {"a.csv": "v1"},
            "digests_path": None,
        }
        # A read-write mount gets its own staging per instance, so it is not handed on.
        assert context.each_listed == {"c": {"data": prepared["data"]}}
        assert over_mounts == [prepared]
        assert keyed is over

    def test_a_variable_nothing_stored_says_so(self):
        host, _context, _calls = self._fakes(self._base())

        with pytest.raises(each.EachError, match="its cell has not stored it"):
            fanout._each_items_of_variable(host, "producer", {"source": "rows"})

    def test_a_manifest_carries_the_cells_lineage(self):
        version = SimpleNamespace(id="v", version=2)
        finalized = SimpleNamespace(id="m", version=1, params={"variable_name": "out"})
        host, context, calls = self._fakes(self._base(), latest=version, finalized=[finalized])

        fanout._store_each_manifests(
            host, context, "c", "src", [each.EachItem("a", "digest")], self._base(), ["out"], [], []
        )

        stage = calls.staged[0]
        assert stage["content_type"] == each.CONTENT_TYPE
        assert stage["provenance_hash"] == derive_subkey("prov", "out")
        assert stage["computation_key"] == "prov"
        assert stage["input_versions"] == {"in": "strata://artifact/in@v=1"}
        assert calls.cell.artifact_uris == {"out": "strata://artifact/m@v=1"}

    def test_an_item_without_the_output_stores_no_manifest(self):
        host, context, calls = self._fakes(self._base(), latest=None)

        with pytest.raises(each.EachError, match="'a' did not produce `out`"):
            fanout._store_each_manifests(
                host,
                context,
                "c",
                "src",
                [each.EachItem("a", "digest")],
                self._base(),
                ["out"],
                [],
                [],
            )

        assert calls.staged == []
        assert calls.discarded == [[]]

    def test_a_write_that_fails_keeps_no_manifest(self):
        def fails(*_args):
            raise TableWriteError("the catalog is gone")

        version = SimpleNamespace(id="v", version=2)
        host, context, calls = self._fakes(self._base(), latest=version, write_cell_tables=fails)

        with pytest.raises(TableWriteError, match="the catalog is gone"):
            fanout._store_each_manifests(
                host,
                context,
                "c",
                "src",
                [each.EachItem("a", "digest")],
                self._base(),
                ["out"],
                [SimpleNamespace(variable="other")],
                [],
            )

        assert calls.discarded == [["staged-out"]]


async def test_the_executor_shares_its_run_state_with_the_r_path(tmp_path):
    nb = create_notebook(tmp_path / "nb", "R")
    add_cell_to_notebook(nb, "c1", None)
    executor = CellExecutor(NotebookSession(parse_notebook(nb), nb))

    context = executor._r_context()

    assert context.received_output_files is executor._received_output_files
    assert context.folder_listings is executor._folder_listings


class TestR:
    """The moved path against fakes; the harness-backed behavior is in ``test_r_cells.py``."""

    LOCAL = SimpleNamespace(name="here", backend=WorkerBackendType.LOCAL)

    @staticmethod
    def _worker(transport="signed"):
        return SimpleNamespace(name="w", backend=WorkerBackendType.EXECUTOR, transport=transport)

    @staticmethod
    def _prov(annotations=None, **overrides):
        fields = {"datasets": [], "tables": [], "nocache": False, "table_writes": []}
        prov = SimpleNamespace(
            source_hash="source",
            runtime_env={"K": "V"},
            env_hash="env",
            input_hashes=["input"],
            provenance_hash="prov",
            annotations=SimpleNamespace(**{**fields, **(annotations or {})}),
            fetch_error=None,
            fetch_error_code=None,
            effective_worker=None,
            fetched={},
            mount_specs=[],
            has_rw_mount=False,
            table_reads=[],
            key_parts={},
            reads=lambda *_args: {},
        )
        prov.over_mounts = lambda _mounts: prov
        for name, value in overrides.items():
            setattr(prov, name, value)
        return prov

    def _fakes(
        self,
        monkeypatch,
        prov=None,
        *,
        worker=None,
        consumed=(),
        hit=None,
        team=None,
        displays=(),
        console=None,
        pool=None,
        **overrides,
    ):
        """A host and a context with nothing behind them, and a record of what they were asked."""
        prov = prov or self._prov()
        calls = SimpleNamespace(
            materialized=[],
            lookups=[],
            replays=[],
            recorded=[],
            applied=[],
            persisted=[],
            kept=[],
            stored=[],
            pushed=[],
            consoles=[],
            good=[],
            dispatched=[],
            metadata=[],
            harness=[],
            prov=prov,
        )
        calls.cell = SimpleNamespace(
            display_outputs=[],
            display_output=None,
            artifact_uris={},
            artifact_uri=None,
            cache_hit=False,
            last_provenance_hash=None,
        )
        monkeypatch.setattr(r, "resolve_worker_spec", lambda *_args: worker)
        monkeypatch.setattr(
            r, "is_http_executor_worker", lambda w: w.backend == WorkerBackendType.EXECUTOR
        )
        monkeypatch.setattr(r, "worker_transport", lambda w: w.transport)

        def resolve_cached_display_outputs(*args):
            calls.replays.append(args)
            return list(displays)

        host = SimpleNamespace(
            notebook_state=SimpleNamespace(id="nb", get_cell=lambda _id: calls.cell),
            dag=SimpleNamespace(consumed_variables={"c": set(consumed)}),
            path=Path("notebook"),
            r_warm_pool=pool,
            get_artifact_manager=lambda: SimpleNamespace(
                cell_artifact_id=lambda cell_id, var: f"{cell_id}.{var}",
                artifact_store=SimpleNamespace(get_latest_version=lambda _id: hit),
            ),
            resolve_cached_display_outputs=resolve_cached_display_outputs,
            resolve_cached_console=lambda *_args: console,
            record_successful_execution_provenance=lambda *args, **kwargs: calls.recorded.append(
                (args, kwargs)
            ),
            apply_execution_result_metadata=lambda cell_id, result: calls.applied.append(
                (cell_id, result)
            ),
            persist_display_outputs=lambda cell_id, outputs: calls.persisted.append(
                (cell_id, outputs)
            ),
        )

        async def materialize_upstreams(cell_id):
            calls.materialized.append(cell_id)

        async def compute_provenance(*_args):
            return prov

        def remote_execution_metadata(spec, remote_build_id=None):
            calls.metadata.append((spec, remote_build_id))
            metadata = {"remote_worker": spec.name} if spec is not None else {}
            return {**metadata, **({"remote_build_id": remote_build_id} if remote_build_id else {})}

        async def find_or_pull_outputs(*args, **kwargs):
            calls.lookups.append((args, kwargs))
            return hit, team

        async def dispatch_http_executor(*args, **kwargs):
            calls.dispatched.append((args, kwargs))
            return {"success": True}, args[4], "executor", {}

        async def prepare_mounts(*_args, **_kwargs):
            return {}

        async def run_r_harness(manifest_path, timeout):
            calls.harness.append((manifest_path, timeout))
            return {"success": True}

        def parse_result(cell_id, result, duration_ms, method):
            return CellExecutionResult(
                cell_id=cell_id,
                success=result.get("success", True),
                error=result.get("error"),
                stdout="printed",
                stderr="warned",
                display_outputs=result.get("display_outputs", []),
                execution_method=method,
                mutation_warnings=[{"variable": "x"}],
            )

        async def collect(*_args):
            return "collected"

        def store_outputs(*args, **kwargs):
            calls.stored.append((args, kwargs))
            return True

        async def push_to_team_store(cell_id):
            calls.pushed.append(cell_id)

        async def sync_back(_mounts):
            return []

        fields = {
            "materialize_upstreams": materialize_upstreams,
            "compute_provenance": compute_provenance,
            "remote_execution_metadata": remote_execution_metadata,
            "find_or_pull_outputs": find_or_pull_outputs,
            "validate_output_hit": lambda *args: (args[5], args[7]),
            "promote_reverted_outputs": lambda *_args: None,
            "moved": lambda _cell_id: ({}, {}),
            "table_write_refusal": lambda _source: None,
            "restore_gone_tables": _nothing_gone,
            "load_input_blobs": lambda *_args: [],
            "dispatch_http_executor": dispatch_http_executor,
            "prepare_mounts": prepare_mounts,
            "write_manifest": lambda *_args: Path("manifest.json"),
            "run_r_harness": run_r_harness,
            "parse_result": parse_result,
            "collect_output_dirs_off_loop": collect,
            "keep_failed_output_dirs": lambda *args: calls.kept.append(args),
            "store_outputs": store_outputs,
            "push_to_team_store": push_to_team_store,
            "store_result_displays": lambda *_args, **_kwargs: None,
            "sync_back_mounts": sync_back,
            "store_console_outputs": lambda *args, **kwargs: calls.consoles.append((args, kwargs)),
            "store_mount_writes": lambda *_args, **_kwargs: None,
            "record_good_run": lambda *args: calls.good.append(args),
            "timeout_message": lambda _cell_id, seconds: f"timed out after {seconds}",
            "received_output_files": {},
            "folder_listings": {},
        }
        return host, RContext(**{**fields, **overrides}), calls

    @staticmethod
    async def _run(host, context, *, materialize_upstreams=False, use_cache=True):
        return await run_r_cell(
            host,
            context,
            "c",
            "src",
            7.0,
            0.0,
            materialize_upstreams=materialize_upstreams,
            use_cache=use_cache,
        )

    async def test_a_run_drops_what_the_last_one_left(self, monkeypatch):
        prov = self._prov({"datasets": ["d"]})
        host, context, _calls = self._fakes(
            monkeypatch, prov, received_output_files={"c": ["a"]}, folder_listings={"c": ["b"]}
        )

        await self._run(host, context)

        assert context.received_output_files == {}
        assert context.folder_listings == {}

    @pytest.mark.parametrize("asked", [True, False])
    async def test_upstreams_run_only_when_asked(self, monkeypatch, asked):
        host, context, calls = self._fakes(monkeypatch)

        await self._run(host, context, materialize_upstreams=asked)

        assert calls.materialized == (["c"] if asked else [])

    @pytest.mark.parametrize(
        ("prov_fields", "worker", "message"),
        [
            ({"annotations": {"datasets": ["d"]}}, None, "@dataset is not supported"),
            ({"annotations": {"tables": ["t"]}}, None, "@table is not supported"),
            ({"fetch_error": "no sum"}, None, "no sum"),
            (
                {"fetched": {"f": Path("f")}},
                SimpleNamespace(name="w", backend=WorkerBackendType.EXECUTOR, transport="signed"),
                "@fetch on an R cell",
            ),
            (
                {},
                SimpleNamespace(name="odd", backend="neither"),
                "worker 'odd' is neither",
            ),
        ],
    )
    async def test_what_r_cannot_do_is_refused_before_the_cache(
        self, monkeypatch, prov_fields, worker, message
    ):
        host, context, calls = self._fakes(monkeypatch, self._prov(**prov_fields), worker=worker)

        result = await self._run(host, context)

        assert result.success is False
        assert message in result.error
        assert result.execution_method == "error"
        assert calls.lookups == []
        assert calls.harness == []
        assert calls.dispatched == []

    @pytest.mark.parametrize(
        ("prov_fields", "worker", "overrides"),
        [
            pytest.param({"annotations": {"datasets": ["d"]}}, None, {}, id="dataset"),
            pytest.param({"annotations": {"tables": ["t"]}}, None, {}, id="table"),
            pytest.param({"fetch_error": "no sum"}, None, {}, id="fetch"),
            pytest.param(
                {"fetched": {"f": Path("f")}},
                SimpleNamespace(name="w", backend=WorkerBackendType.EXECUTOR, transport="signed"),
                {},
                id="remote-fetch",
            ),
            pytest.param({}, SimpleNamespace(name="odd", backend="neither"), {}, id="worker"),
            pytest.param(
                {}, None, {"table_write_refusal": lambda _source: "no catalog"}, id="table-write"
            ),
        ],
    )
    async def test_a_refusal_ends_as_any_failure_does(
        self, monkeypatch, prov_fields, worker, overrides
    ):
        host, context, calls = self._fakes(
            monkeypatch, self._prov(**prov_fields), worker=worker, **overrides
        )

        result = await self._run(host, context)

        assert result.success is False
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    async def test_a_fetch_that_failed_keeps_its_code(self, monkeypatch):
        prov = self._prov(fetch_error="no sum", fetch_error_code="fetch_unchecked")
        host, context, _calls = self._fakes(monkeypatch, prov)

        result = await self._run(host, context)

        assert result.error_code == "fetch_unchecked"

    @pytest.mark.parametrize(
        "prov_fields", [{"has_rw_mount": True}, {"annotations": {"nocache": True}}]
    )
    async def test_what_cannot_be_reused_looks_nothing_up(self, monkeypatch, prov_fields):
        host, context, calls = self._fakes(monkeypatch, self._prov(**prov_fields))

        await self._run(host, context)

        assert calls.lookups == []
        assert calls.replays == []
        assert len(calls.harness) == 1

    async def test_a_rerun_replays_no_display(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch)

        await self._run(host, context, use_cache=False)

        assert calls.replays == []

    async def test_a_hit_shows_what_it_found(self, monkeypatch):
        version = SimpleNamespace(id="a", version=2)
        shown = [
            SimpleNamespace(model_dump=lambda n=n: {"n": n}, artifact_uri=f"strata://{n}")
            for n in (1, 2)
        ]
        team = SimpleNamespace(principal="p", build_env="linux", saved_ms=5, promotion=None)
        console = SimpleNamespace(stdout="out", stderr="err", run_ms=9)
        host, context, calls = self._fakes(
            monkeypatch,
            worker=self.LOCAL,
            consumed=["out"],
            hit=version,
            team=team,
            displays=shown,
            console=console,
        )

        result = await self._run(host, context)

        cell = calls.cell
        assert result.execution_method == "cached"
        assert cell.cache_hit is True
        assert cell.display_outputs == shown
        assert cell.display_output is shown[-1]
        assert cell.artifact_uris == {"out": "strata://artifact/a@v=2"}
        assert cell.artifact_uri == "strata://artifact/a@v=2"
        assert result.display_outputs == [{"n": 1}, {"n": 2}]
        assert result.artifact_uri == "strata://artifact/a@v=2"
        assert result.stdout == "out"
        assert result.stderr == "err"
        assert result.replaces_console is True
        assert result.from_team_cache is True
        assert result.team_cache_principal == "p"
        assert result.team_cache_build_env == "linux"
        assert result.remote_worker == "here"
        assert calls.applied == [("c", result)]
        assert len(calls.recorded) == 1
        assert calls.harness == []

    async def test_a_refused_table_write_runs_nothing(self, monkeypatch):
        host, context, calls = self._fakes(
            monkeypatch, table_write_refusal=lambda _source: "no catalog"
        )

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "no catalog"
        assert calls.harness == []

    async def test_a_run_is_keyed_on_the_listing_it_read(self, monkeypatch):
        listed = self._prov(provenance_hash="listed")
        prov = self._prov(over_mounts=lambda _mounts: listed)
        host, context, calls = self._fakes(monkeypatch, prov, consumed=["out"])

        await self._run(host, context)

        assert calls.stored[0][0][2] == "listed"
        assert calls.good[0][1] == "listed"
        assert calls.good[0][4] is listed
        assert calls.consoles[0][0][1] == "listed"

    async def test_what_ran_it_says_is_stored(self, monkeypatch):
        reported = {"success": True, "build_env": "linux", "hardware": {"gpu": 1}}

        async def run_r_harness(*_args):
            return reported

        host, context, calls = self._fakes(monkeypatch, run_r_harness=run_r_harness)

        await self._run(host, context)

        stored = calls.stored[0][1]
        assert stored["build_env"] == "linux"
        assert stored["hardware"] == {"gpu": 1}
        assert stored["collected"] == "collected"

    @pytest.mark.parametrize(
        ("consumed", "prov_fields", "displays", "stored"),
        [
            (["out"], {}, [], True),
            ([], {}, [{"n": 1}], True),
            # A leaf with only a console is not cached, and a record would read it as ready.
            ([], {}, [], False),
            (["out"], {"has_rw_mount": True}, [], False),
            (["out"], {"annotations": {"nocache": True}}, [], False),
        ],
    )
    async def test_a_console_is_kept_where_a_hit_would_replay_it(
        self, monkeypatch, consumed, prov_fields, displays, stored
    ):
        async def run_r_harness(*_args):
            return {"success": True, "display_outputs": displays}

        host, context, calls = self._fakes(
            monkeypatch,
            self._prov(**prov_fields),
            consumed=consumed,
            run_r_harness=run_r_harness,
        )

        await self._run(host, context)

        assert bool(calls.consoles) is stored
        if stored:
            args, kwargs = calls.consoles[0]
            assert args[:3] == ("c", "prov", "printed")
            assert kwargs["display_count"] == len(displays)

    async def test_the_warm_pool_comes_first(self, monkeypatch):
        asked = []

        async def execute_with_pool(pool, manifest, path, timeout):
            asked.append((pool, manifest, path, timeout))
            return {"success": True}

        monkeypatch.setattr(
            "strata.notebook.pool.PooledCellExecutor",
            SimpleNamespace(execute_with_pool=execute_with_pool),
        )
        host, context, calls = self._fakes(monkeypatch, pool="pool")

        result = await self._run(host, context)

        assert asked == [("pool", Path("manifest.json"), Path("notebook"), 7.0)]
        assert calls.harness == []
        assert result.execution_method == "warm"

    async def test_a_pool_with_nothing_free_falls_back_to_a_cold_run(self, monkeypatch):
        async def execute_with_pool(*_args):
            return None

        monkeypatch.setattr(
            "strata.notebook.pool.PooledCellExecutor",
            SimpleNamespace(execute_with_pool=execute_with_pool),
        )
        host, context, calls = self._fakes(monkeypatch, pool="pool")

        result = await self._run(host, context)

        assert calls.harness == [(Path("manifest.json"), 7.0)]
        assert result.execution_method == "cold"

    async def test_a_failed_run_keeps_its_folders_and_no_displays(self, monkeypatch):
        async def run_r_harness(*_args):
            return {"success": False, "error": "boom", "display_outputs": [{"n": 1}]}

        host, context, calls = self._fakes(monkeypatch, run_r_harness=run_r_harness)

        result = await self._run(host, context)

        assert result.success is False
        cell_id, source, _directory, provenance, env, collected = calls.kept[0]
        assert (cell_id, source, provenance, env, collected) == (
            "c",
            "src",
            "prov",
            "env",
            "collected",
        )
        assert calls.persisted == [("c", None)]
        assert calls.stored == []
        assert calls.good == []

    @pytest.mark.parametrize(("transport", "built"), [("signed", True), ("direct", False)])
    async def test_a_worker_is_handed_the_run(self, monkeypatch, transport, built):
        host, context, calls = self._fakes(
            monkeypatch, worker=self._worker(transport), consumed=["out"]
        )

        result = await self._run(host, context)

        args, kwargs = calls.dispatched[0]
        assert args[1] == "src"
        assert args[5] == {"K": "V"}
        assert args[6] == 7.0
        assert kwargs["cell_id"] == "c"
        assert kwargs["cell_provenance_hash"] == "prov"
        assert kwargs["language"] == "r"
        assert kwargs["remote_build_id"] == result.remote_build_id
        assert bool(result.remote_build_id) is built
        assert calls.harness == []

    @pytest.mark.parametrize("uploaded", [True, False])
    async def test_a_lost_worker_keeps_what_it_uploaded(self, monkeypatch, uploaded):
        received = {}

        async def lost(*_args, **_kwargs):
            if uploaded:
                received["c"] = ["part"]
            raise RemoteExecutionError(
                "gone", remote_build_state="failed", remote_error_code="worker_lost"
            )

        host, context, calls = self._fakes(
            monkeypatch,
            worker=self._worker(),
            dispatch_http_executor=lost,
            received_output_files=received,
        )

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "gone"
        assert result.remote_build_state == "failed"
        assert result.remote_error_code == "worker_lost"
        assert calls.kept == ([("c", "src", None, "prov", "env")] if uploaded else [])
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    @pytest.mark.parametrize("uploaded", [True, False])
    async def test_a_cancelled_run_keeps_what_the_worker_uploaded(self, monkeypatch, uploaded):
        received = {}

        async def cancelled(*_args, **_kwargs):
            if uploaded:
                received["c"] = ["part"]
            raise asyncio.CancelledError

        host, context, calls = self._fakes(
            monkeypatch,
            worker=self._worker(),
            dispatch_http_executor=cancelled,
            received_output_files=received,
        )

        with pytest.raises(asyncio.CancelledError):
            await self._run(host, context)

        assert calls.kept == ([("c", "src", None, "prov", "env")] if uploaded else [])

    async def test_a_timeout_is_the_result(self, monkeypatch):
        async def times_out(*_args):
            raise TimeoutError

        host, context, calls = self._fakes(monkeypatch, worker=self.LOCAL, run_r_harness=times_out)

        result = await self._run(host, context)

        assert result.error == "timed out after 7.0"
        assert result.remote_worker == "here"
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    async def test_anything_else_is_the_result(self, monkeypatch):
        async def breaks(*_args):
            raise RuntimeError("boom")

        host, context, calls = self._fakes(monkeypatch, run_r_harness=breaks)

        result = await self._run(host, context)

        assert result.error == "R execution failed: boom"
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            pytest.param(TimeoutError(), "timed out after 7.0", id="timeout"),
            pytest.param(RuntimeError("boom"), "R execution failed: boom", id="other"),
        ],
    )
    @pytest.mark.parametrize("uploaded", [True, False])
    async def test_a_failed_dispatch_keeps_the_worker_and_its_uploads(
        self, monkeypatch, error, message, uploaded
    ):
        received = {}

        async def fails(*_args, **_kwargs):
            if uploaded:
                received["c"] = ["part"]
            raise error

        host, context, calls = self._fakes(
            monkeypatch,
            worker=self._worker(),
            dispatch_http_executor=fails,
            received_output_files=received,
        )

        result = await self._run(host, context)

        assert result.error == message
        assert result.remote_worker == "w"
        assert calls.kept == ([("c", "src", None, "prov", "env")] if uploaded else [])

    @pytest.mark.parametrize(
        ("store_outputs", "message"),
        [
            pytest.param(lambda *_a, **_k: False, STORE_FAILED, id="false"),
            pytest.param(_raises(TableWriteError("no table")), "no table", id="table"),
            pytest.param(_raises(UnsafeRunFile("a link")), "a link", id="unsafe"),
        ],
    )
    async def test_a_failed_store_keeps_the_worker(self, monkeypatch, store_outputs, message):
        host, context, _calls = self._fakes(
            monkeypatch, worker=self._worker(), store_outputs=store_outputs
        )

        result = await self._run(host, context)

        assert result.error == message
        assert result.stdout == "printed"
        assert result.remote_worker == "w"

    async def test_a_failed_mount_sync_keeps_the_worker(self, monkeypatch):
        async def dispatch(*args, **_kwargs):
            return {"success": True}, args[4], "executor", {"m": "mounted"}

        async def sync_back(_mounts):
            raise OSError("read only")

        host, context, _calls = self._fakes(
            monkeypatch,
            worker=self._worker(),
            dispatch_http_executor=dispatch,
            sync_back_mounts=sync_back,
        )

        result = await self._run(host, context)

        assert "failed to sync read-write mounts: read only" in result.error
        assert result.remote_worker == "w"

    async def test_a_step_after_the_run_keeps_what_it_printed(self, monkeypatch):
        host, context, _calls = self._fakes(
            monkeypatch,
            worker=self._worker(),
            store_result_displays=_raises(RuntimeError("no display")),
        )

        result = await self._run(host, context)

        assert result.error == "R execution failed: no display"
        assert (result.stdout, result.stderr) == ("printed", "warned")
        assert result.mutation_warnings == [{"variable": "x"}]
        assert result.remote_worker == "w"


class TestRHarness:
    """What the harness is started with, with no ``Rscript`` behind it."""

    USER = object()

    @pytest.fixture
    def spawned(self, monkeypatch):
        seen = SimpleNamespace(handed=[], terminated=[], spawn=None, outcome=(b"out", b"err"))

        async def communicate():
            if isinstance(seen.outcome, BaseException):
                raise seen.outcome
            return seen.outcome

        seen.proc = SimpleNamespace(pid=1, communicate=communicate)

        async def create_subprocess_exec(*cmd, **kwargs):
            seen.spawn = (cmd, kwargs)
            return seen.proc

        async def terminate(proc):
            seen.terminated.append(proc)

        monkeypatch.setattr(r.asyncio, "create_subprocess_exec", create_subprocess_exec)
        monkeypatch.setattr("strata.notebook.process_tree.terminate_subprocess_tree", terminate)
        monkeypatch.setattr(r.shutil, "which", lambda _name: "/bin/Rscript")
        monkeypatch.setattr(r, "resolve_harness_user", lambda: self.USER)
        monkeypatch.setattr(r, "hand_over", lambda path, user: seen.handed.append((path, user)))
        monkeypatch.setattr(r, "identity_env", lambda env, user: {"env": env, "user": user})
        monkeypatch.setattr(r, "spawn_kwargs", lambda user: {"spawned_as": user})
        monkeypatch.setattr(r, "lifeline_handoff", lambda: ([3], {"LIFE": "1"}))
        monkeypatch.setattr(r, "lifeline_command", lambda cmd: ["lifeline", *cmd])
        return seen

    @staticmethod
    async def _run(tmp_path):
        return await r.run_r_harness(
            Path("notebook"),
            Path("harness.R"),
            lambda extra: {"extra": extra},
            tmp_path / "manifest.json",
            7.0,
        )

    async def test_it_starts_as_the_harness_user(self, spawned, tmp_path):
        await self._run(tmp_path)

        cmd, kwargs = spawned.spawn
        assert cmd == (
            "lifeline",
            "/bin/Rscript",
            "harness.R",
            str(tmp_path / "manifest.json"),
        )
        assert spawned.handed == [(tmp_path, self.USER)]
        assert kwargs["env"] == {"env": {"extra": {"LIFE": "1"}}, "user": self.USER}
        assert kwargs["spawned_as"] is self.USER
        assert kwargs["pass_fds"] == [3]
        assert kwargs["cwd"] == "notebook"
        assert kwargs["start_new_session"] is True

    async def test_no_result_file_reports_what_it_printed(self, spawned, tmp_path):
        result = await self._run(tmp_path)

        assert result["success"] is False
        assert result["error"] == "err"
        assert result["stdout"] == "out"

    async def test_a_result_file_is_the_result(self, spawned, tmp_path):
        (tmp_path / "harness-result.json").write_text('{"success": true, "variables": {"x": 1}}')

        result = await self._run(tmp_path)

        assert result == {"success": True, "variables": {"x": 1}}

    async def test_a_timeout_ends_the_process_tree(self, spawned, tmp_path):
        spawned.outcome = TimeoutError()

        with pytest.raises(TimeoutError):
            await self._run(tmp_path)

        assert spawned.terminated == [spawned.proc]

    async def test_a_cancel_ends_the_process_tree(self, spawned, tmp_path):
        spawned.outcome = asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError):
            await self._run(tmp_path)

        assert spawned.terminated == [spawned.proc]


def _batch_prov(*, nocache=False):
    prov = SimpleNamespace(
        provenance_hash="prov",
        source_hash="source",
        env_hash="env",
        input_hashes=["input"],
        annotations=SimpleNamespace(nocache=nocache, table_writes=[]),
        table_reads=[],
        key_parts={},
        reads=lambda *_args: {},
    )
    prov.rekeyed = lambda: prov
    return prov


def _batch_art(artifact_id="art", version=3, content_type="json/object"):
    return SimpleNamespace(id=artifact_id, version=version, params={"content_type": content_type})


def _batch_fakes(tmp_path, *, consumed=(), displays=(), console=None, wired=("x",), **overrides):
    """A host and a context with nothing behind them, and a record of what they were asked."""
    seen = SimpleNamespace(
        recorded=[],
        stored=[],
        consoles=[],
        display_stores=[],
        persisted=[],
        pushed=[],
        written=[],
        tokens=[],
        outputs={},
        blobs={},
        latest={},
    )
    seen.cell = SimpleNamespace(
        source="edited",
        upstream_ids=[],
        artifact_uris={},
        artifact_uri=None,
        cache_hit=False,
        display_outputs=[],
        display_output=None,
        last_provenance_hash=None,
    )
    seen.cells = {"c": seen.cell}

    def record(cell_id, provenance_hash, *args, **kwargs):
        seen.recorded.append((cell_id, provenance_hash))
        seen.cell.last_provenance_hash = provenance_hash

    host = SimpleNamespace(
        path=tmp_path,
        venv_python="python",
        environment_execution_block_message=lambda: "not ready",
        notebook_state=SimpleNamespace(get_cell=seen.cells.get),
        dag=SimpleNamespace(consumed_variables={"c": set(consumed)}),
        wired_variables=lambda _cell, _upstream: set(wired),
        get_artifact_manager=lambda: SimpleNamespace(
            cell_artifact_id=lambda cell_id, var: f"{cell_id}.{var}",
            artifact_store=SimpleNamespace(
                read_blob=lambda artifact_id, _version: seen.blobs.get(artifact_id),
                get_latest_version=seen.latest.get,
                get_artifact=lambda artifact_id, _version: seen.latest.get(artifact_id),
            ),
        ),
        resolve_cached_display_outputs=lambda *_args: list(displays),
        resolve_cached_console=lambda *_args: console,
        record_successful_execution_provenance=record,
        persist_display_outputs=lambda cell_id, outputs: seen.persisted.append((cell_id, outputs)),
        parse_artifact_uri=NotebookSession.parse_artifact_uri,
    )

    @contextmanager
    def run_token(timeout):
        seen.tokens.append(timeout)
        yield "token"

    async def compute_provenance(*_args):
        return _batch_prov()

    async def pull_from_team_store(**_kwargs):
        return None

    async def push_to_team_store(cell_id):
        seen.pushed.append(cell_id)

    def store_outputs(*args, **kwargs):
        seen.stored.append((args, kwargs))
        return True

    def write_input_file(path, *_args):
        seen.written.append(path)
        return True

    fields = {
        "harness_path": Path("harness.py"),
        "harness_env": dict,
        "work_dir_fields": lambda: {"work_dir": "work"},
        "run_token": run_token,
        "compute_provenance": compute_provenance,
        "canonical_output": lambda _cell, var, _hash: seen.outputs.get(var),
        "input_refs": lambda _cell: {},
        "moved": lambda _cell: ({}, {}),
        "promote_reverted_outputs": lambda *_args: None,
        "restore_gone_tables": _nothing_gone,
        "promote_reverted_leaf": lambda *_args: None,
        "pull_from_team_store": pull_from_team_store,
        "push_to_team_store": push_to_team_store,
        "store_outputs": store_outputs,
        "store_console_outputs": lambda *args, **kwargs: seen.consoles.append((args, kwargs)),
        "store_display_outputs": lambda *_args, **_kwargs: [],
        "write_input_file": write_input_file,
    }
    return host, batch.BatchContext(**{**fields, **overrides}), seen


def _frame(frame_type, **payload):
    return {"type": frame_type, "payload": payload}


def _frame_bytes(*frames):
    return b"".join(
        frame if isinstance(frame, bytes) else (json.dumps(frame) + "\n").encode()
        for frame in frames
    )


class TestBatchServiceLoop:
    """The loop over the harness's frames, with the parent's answers faked."""

    @pytest.fixture
    def serve(self, monkeypatch, tmp_path):
        seen = SimpleNamespace(
            responses=[],
            results={},
            events=[],
            started=[],
            terminated=[],
            check={"cache_hit": False, "provenance_hash": "prov"},
            persist={"ok": True, "uri": "", "display_outputs": []},
        )

        async def check(*_args, **_kwargs):
            return dict(seen.check)

        async def persist(*_args, **_kwargs):
            return dict(seen.persist)

        async def terminate(proc):
            seen.terminated.append(proc)

        monkeypatch.setattr(batch, "batch_service_cache_check", check)
        monkeypatch.setattr(batch, "batch_service_persist", persist)
        monkeypatch.setattr(batch, "terminate_subprocess_tree", terminate)

        async def on_cell_event(result):
            seen.events.append(result)

        async def on_cell_start(cell_id):
            seen.started.append(cell_id)

        async def run(*frames, reader=None, watchdog_state=None):
            if reader is None:
                reader = asyncio.StreamReader()
                reader.feed_data(_frame_bytes(*frames))
                reader.feed_eof()
            host, context, _ = _batch_fakes(tmp_path)
            return await batch.batch_service_loop(
                host,
                context,
                reader,
                seen.responses.append,
                seen.results,
                tmp_path,
                executed_sources={},
                provenance={},
                use_cache=True,
                on_cell_event=on_cell_event,
                on_cell_start=on_cell_start,
                watchdog_state=watchdog_state,
            )

        seen.run = run
        return seen

    @staticmethod
    def _watchdog():
        return {
            "active_cell_id": None,
            "active_started_at": None,
            "active_timeout": 1000.0,
            "timed_out": False,
            "timeout_cell_id": None,
            "proc": None,
        }

    async def test_a_frame_it_cannot_read_is_skipped(self, serve):
        ended = await serve.run(b"not json\n", _frame("batch_end", reason="complete"))

        assert ended == ("complete", None)

    async def test_it_announces_a_cell_as_it_starts(self, serve):
        await serve.run(_frame("cell_start", cell_id="c"))

        assert serve.started == ["c"]

    async def test_a_good_persist_keeps_what_the_cell_took(self, serve):
        await serve.run(_frame("persist", cell_id="c", stdout="out", duration_ms=5.0))

        result = serve.results["c"]
        assert (result.status, result.stdout, result.duration_ms) == ("ok", "out", 5.0)
        assert serve.events == [result]

    async def test_a_display_that_was_not_stored_is_not_reported(self, serve):
        serve.persist = {"ok": True, "uri": "", "display_outputs": []}

        await serve.run(_frame("persist", cell_id="c", display_outputs=[{"file": "d.png"}]))

        assert serve.results["c"].status == "ok"
        assert serve.results["c"].display_outputs == []

    @pytest.mark.parametrize("expired", [True, False], ids=["expired", "while-waiting"])
    async def test_a_timeout_goes_on_when_ending_the_harness_fails(
        self, serve, monkeypatch, expired
    ):
        logged = []

        async def terminate(_proc):
            raise OSError("no such process")

        monkeypatch.setattr(batch, "terminate_subprocess_tree", terminate)
        monkeypatch.setattr(batch.logger, "exception", lambda *args: logged.append(args[0]))
        watchdog = self._watchdog()
        watchdog["active_cell_id"] = "c"
        watchdog["active_started_at"] = time.time() - (2000.0 if expired else 999.99)
        reader = asyncio.StreamReader()

        ended = await serve.run(reader=reader, watchdog_state=watchdog)

        assert watchdog["timed_out"] is True
        assert ended is not None
        assert logged == ["Failed terminating batch harness after a timeout"]

    async def test_a_failed_persist_keeps_what_the_cell_printed(self, serve):
        serve.persist = {"ok": False, "error": "no"}

        await serve.run(_frame("persist", cell_id="c", stdout="out", stderr="err", duration_ms=5.0))

        result = serve.results["c"]
        assert result.status == "persist_failed"
        assert (result.error, result.stdout, result.stderr) == ("no", "out", "err")
        assert result.duration_ms == 5.0

    @pytest.mark.parametrize(
        "ending",
        [
            _frame("persist", cell_id="c"),
            _frame("cell_output", cache_hit=True),
            _frame("cell_error", cell_id="c", error="no"),
        ],
        ids=["persist", "hit", "error"],
    )
    async def test_a_finished_cell_closes_the_watchdog_window(self, serve, ending):
        watchdog = self._watchdog()

        await serve.run(
            _frame("cell_start", cell_id="c"),
            _frame("cache_check", cell_id="c"),
            ending,
            watchdog_state=watchdog,
        )

        assert watchdog["active_cell_id"] is None
        assert watchdog["active_started_at"] is None

    async def test_a_hit_carries_its_team_pull_and_console_but_the_harness_does_not(self, serve):
        pull = object()
        serve.check = {"cache_hit": True, "team_pull": pull, "replaces_console": True}
        serve.check["hit_run_ms"] = 7

        await serve.run(
            _frame("cell_start", cell_id="c"),
            _frame("cache_check", cell_id="c"),
            _frame("cell_output", cache_hit=True),
        )

        result = serve.results["c"]
        assert result.team_pull is pull
        assert (result.replaces_console, result.hit_run_ms) == (True, 7)
        assert set(serve.responses[0]) == {"cache_hit", "input_uris"}

    async def test_an_oversized_frame_ends_the_harness(self, serve):
        reader = asyncio.StreamReader(limit=8)
        reader.feed_data(b"x" * 100 + b"\n")
        reader.feed_eof()
        proc = object()

        ended = await serve.run(reader=reader, watchdog_state={**self._watchdog(), "proc": proc})

        assert ended == ("subprocess_died", None)
        assert serve.terminated == [proc]


URI = "strata://artifact/art@v=3"


class TestBatchInputs:
    """What goes into the batch's directory for the harness to read."""

    async def test_it_stages_only_wired_variables_from_outside_the_batch(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path, wired=("x", "z"))
        seen.cell.upstream_ids = ["outside", "inside"]
        seen.cells["outside"] = SimpleNamespace(
            artifact_uris={"x": URI, "y": "strata://artifact/other@v=1"}
        )
        seen.cells["inside"] = SimpleNamespace(
            upstream_ids=[], artifact_uris={"z": "strata://artifact/inside@v=1"}
        )
        for artifact_id in ("art", "other", "inside"):
            seen.latest[artifact_id] = _batch_art(artifact_id)

        staged = batch.batch_resolve_upstream_inputs(
            host, context, [{"cell_id": "c"}, {"cell_id": "inside"}], tmp_path
        )

        assert staged == {"x": {"content_type": "json/object", "file": "__upstream__/x.json"}}

    async def test_a_failed_write_is_not_staged(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path, write_input_file=lambda *_args: False)
        seen.cell.upstream_ids = ["outside"]
        seen.cells["outside"] = SimpleNamespace(artifact_uris={"x": URI})
        seen.latest["art"] = _batch_art()

        staged = batch.batch_resolve_upstream_inputs(host, context, [{"cell_id": "c"}], tmp_path)

        assert staged == {}

    async def test_a_missing_artifact_is_not_written(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path)

        assert batch.materialize_artifact_to_dir(host, context, URI, tmp_path, "x") is None
        assert seen.written == []

    async def test_a_mixed_case_name_gets_its_collision_proof_file(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path)
        seen.latest["art"] = _batch_art()

        spec = batch.materialize_artifact_to_dir(host, context, URI, tmp_path, "Widget")

        assert spec["file"] == f"{safe_filename_stem('Widget')}.json"
        assert spec["file"] != "Widget.json"
        assert seen.written == [tmp_path / spec["file"]]

    async def test_a_cell_is_told_only_the_inputs_wired_to_it(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path)
        seen.cell.upstream_ids = ["outside"]
        seen.cells["outside"] = SimpleNamespace(
            artifact_uris={"x": URI, "y": "strata://artifact/other@v=1"}
        )

        assert batch.upstream_artifact_uris(host, context, "c") == {"x": {"uri": URI}}

    async def test_a_uri_that_is_not_a_versioned_artifact_is_refused(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path)
        seen.latest["art"] = _batch_art()

        with pytest.raises(ValueError, match="not a versioned artifact URI"):
            batch.materialize_artifact_to_dir(host, context, "strata://artifact/art", tmp_path, "x")

        assert seen.written == []

    async def test_a_fan_out_instances_uri_is_staged(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path)
        seen.latest["nb_cell_c_var_x@variant=triple"] = _batch_art()

        spec = batch.materialize_artifact_to_dir(
            host, context, "strata://artifact/nb_cell_c_var_x@variant=triple@v=2", tmp_path, "x"
        )

        assert spec is not None
        assert seen.written == [tmp_path / spec["file"]]


class TestBatchCacheCheck:
    """A cache check's answer, and what it leaves on the cell."""

    @staticmethod
    async def _check(host, context, tmp_path, *, provenance=None):
        return await batch.batch_service_cache_check(
            host,
            context,
            "c",
            tmp_path,
            executed_sources={"c": "ran"},
            provenance={"c": _batch_prov()} if provenance is None else provenance,
            use_cache=True,
        )

    async def test_a_hit_names_its_outputs_on_the_cell(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path, consumed=("x",))
        seen.outputs["x"] = _batch_art()
        seen.blobs["art"] = b"bytes"

        answer = await self._check(host, context, tmp_path)

        assert answer["cache_hit"] is True
        assert answer["cached_outputs"] == {"x": {"content_type": "json/object", "file": "x.json"}}
        assert (tmp_path / "c" / "x.json").read_bytes() == b"bytes"
        assert seen.cell.cache_hit is True
        assert seen.cell.artifact_uris == {"x": URI}
        assert seen.cell.artifact_uri == URI

    async def test_a_missing_blob_is_a_miss(self, tmp_path):
        host, context, seen = _batch_fakes(tmp_path, consumed=("x",))
        seen.outputs["x"] = _batch_art()

        answer = await self._check(host, context, tmp_path)

        assert answer == {"cache_hit": False, "provenance_hash": "prov"}
        assert seen.cell.cache_hit is False

    async def test_a_hit_replays_its_displays(self, tmp_path):
        models = [CellOutput(content_type="json/object", bytes=n) for n in (1, 2)]
        host, context, seen = _batch_fakes(tmp_path, consumed=("x",), displays=models)
        seen.outputs["x"] = _batch_art()
        for index in range(2):
            seen.latest[f"c.__display__{index}"] = _batch_art(f"d{index}")
            seen.blobs[f"d{index}"] = b"shown"
        seen.blobs["art"] = b"bytes"

        answer = await self._check(host, context, tmp_path)

        assert [d["file"] for d in answer["cached_displays"]] == [
            "__display__0.json",
            "__display__1.json",
        ]
        assert seen.cell.display_outputs == models
        assert seen.cell.display_output == models[1]

    async def test_a_display_with_no_blob_is_left_out(self, tmp_path):
        models = [CellOutput(content_type="json/object")]
        console = SimpleNamespace(stdout="", stderr="", run_ms=0)
        host, context, seen = _batch_fakes(tmp_path, displays=models, console=console)
        seen.latest["c.__display__0"] = _batch_art("d0")

        answer = await self._check(host, context, tmp_path)

        assert answer["cache_hit"] is True
        assert answer["cached_displays"] == []

    async def test_a_hit_answers_with_its_console(self, tmp_path):
        console = SimpleNamespace(stdout="out", stderr="err", run_ms=9)
        host, context, _ = _batch_fakes(tmp_path, console=console)

        answer = await self._check(host, context, tmp_path)

        assert (answer["stdout"], answer["stderr"]) == ("out", "err")
        assert (answer["replaces_console"], answer["hit_run_ms"]) == (True, 9)

    @pytest.mark.parametrize(("last", "replaces"), [("prov", False), ("other", True)])
    async def test_with_no_console_kept_a_hit_compares_the_key_before_recording(
        self, tmp_path, last, replaces
    ):
        host, context, seen = _batch_fakes(tmp_path, consumed=("x",))
        seen.outputs["x"] = _batch_art()
        seen.blobs["art"] = b"bytes"
        seen.cell.last_provenance_hash = last

        answer = await self._check(host, context, tmp_path)

        assert answer["replaces_console"] is replaces
        assert seen.recorded == [("c", "prov")]

    async def test_a_provenance_failure_is_a_miss(self, tmp_path):
        async def raises(*_args):
            raise RuntimeError("no")

        host, context, _ = _batch_fakes(tmp_path, compute_provenance=raises)

        answer = await self._check(host, context, tmp_path, provenance={})

        assert answer == {"cache_hit": False, "provenance_hash": ""}

    async def test_a_team_pull_is_read_back_as_a_hit(self, tmp_path):
        pull = object()

        async def pull_from_team_store(**_kwargs):
            seen.outputs["x"] = _batch_art()
            return pull

        host, context, seen = _batch_fakes(
            tmp_path, consumed=("x",), pull_from_team_store=pull_from_team_store
        )
        seen.blobs["art"] = b"bytes"

        answer = await self._check(host, context, tmp_path)

        assert answer["cache_hit"] is True
        assert answer["team_pull"] is pull
        assert "x" in answer["cached_outputs"]


class TestBatchPersist:
    """A persist's answer, and what it records."""

    @pytest.fixture
    def exported(self, monkeypatch):
        seen = []

        def export(host, cell_id, source, *_args):
            seen.append(source)

        monkeypatch.setattr(batch, "write_module_export_outputs", export)
        return seen

    @staticmethod
    async def _persist(host, context, tmp_path, *, provenance=None, **payload):
        (tmp_path / "c").mkdir(exist_ok=True)
        return await batch.batch_service_persist(
            host,
            context,
            {"cell_id": "c", **payload},
            tmp_path,
            executed_sources={"c": "ran"},
            provenance={"c": _batch_prov()} if provenance is None else provenance,
        )

    async def test_no_output_directory_is_refused(self, tmp_path, exported):
        host, context, _ = _batch_fakes(tmp_path)

        answer = await batch.batch_service_persist(
            host,
            context,
            {"cell_id": "c"},
            tmp_path,
            executed_sources={},
            provenance={"c": _batch_prov()},
        )

        assert answer == {"ok": False, "error": "output dir missing for c"}
        assert exported == []

    async def test_no_cache_check_is_refused(self, tmp_path, exported):
        host, context, seen = _batch_fakes(tmp_path)

        answer = await self._persist(host, context, tmp_path, provenance={})

        assert answer == {"ok": False, "error": "no cache check for c"}
        assert seen.stored == []

    async def test_an_export_that_raises_is_refused(self, tmp_path, monkeypatch):
        def export(*_args):
            raise RuntimeError("boom")

        monkeypatch.setattr(batch, "write_module_export_outputs", export)
        host, context, seen = _batch_fakes(tmp_path)

        answer = await self._persist(host, context, tmp_path)

        assert answer == {"ok": False, "error": "module export raised: boom"}
        assert seen.stored == []

    async def test_it_records_the_source_that_ran_not_an_edit(self, tmp_path, exported):
        host, context, seen = _batch_fakes(tmp_path)

        await self._persist(host, context, tmp_path)

        assert exported == ["ran"]
        assert seen.stored[0][1]["source"] == "ran"
        assert seen.consoles[0][1]["source"] == "ran"

    async def test_displays_record_the_source_that_ran_not_an_edit(self, tmp_path, exported):
        asked = []

        def store(*_args, **kwargs):
            asked.append(kwargs["source"])
            return []

        host, context, _seen = _batch_fakes(tmp_path, store_display_outputs=store)

        await self._persist(host, context, tmp_path, display_outputs=[{"file": "d"}])

        assert asked == ["ran"]

    async def test_a_display_store_that_fails_is_answered_with_no_displays(
        self, tmp_path, exported
    ):
        def store(*_args, **_kwargs):
            raise OSError("no display")

        host, context, seen = _batch_fakes(tmp_path, store_display_outputs=store)

        answer = await self._persist(host, context, tmp_path, display_outputs=[{"file": "d"}])

        assert answer["ok"] is True
        assert answer["display_outputs"] == []
        assert seen.pushed

    async def test_a_record_that_fails_is_logged_and_the_cell_stays_good(
        self, tmp_path, exported, monkeypatch
    ):
        logged = []

        def record(*_args, **_kwargs):
            raise OSError("no record")

        host, context, _seen = _batch_fakes(tmp_path)
        host.record_successful_execution_provenance = record
        monkeypatch.setattr(batch.logger, "exception", lambda *args: logged.append(args))

        answer = await self._persist(host, context, tmp_path)

        assert answer["ok"] is True
        assert logged == [("Batch could not record the run of cell %s", "c")]

    async def test_a_false_store_stores_no_displays(self, tmp_path, exported):
        asked = []

        host, context, _seen = _batch_fakes(
            tmp_path,
            store_outputs=lambda *_args, **_kwargs: False,
            store_display_outputs=lambda *_args, **_kwargs: asked.append(True) or [],
        )

        answer = await self._persist(host, context, tmp_path, display_outputs=[{"file": "d"}])

        assert answer["ok"] is False
        assert asked == []

    async def test_nocache_stores_no_console(self, tmp_path, exported):
        host, context, seen = _batch_fakes(tmp_path)

        answer = await self._persist(
            host, context, tmp_path, provenance={"c": _batch_prov(nocache=True)}
        )

        assert answer["ok"] is True
        assert seen.consoles == []
        assert len(seen.stored) == 1

    async def test_stored_displays_reach_the_cell_the_console_and_the_ack(self, tmp_path, exported):
        stored = [{"content_type": "json/object", "bytes": n} for n in (1, 2)]
        host, context, seen = _batch_fakes(
            tmp_path, store_display_outputs=lambda *_args, **_kwargs: stored
        )

        answer = await self._persist(
            host, context, tmp_path, display_outputs=[{"file": "d"}], duration_ms=5.0
        )

        assert answer["display_outputs"] == stored
        assert seen.consoles[0][1]["display_count"] == 2
        assert seen.cell.display_outputs == [CellOutput(**d) for d in stored]
        assert seen.cell.display_output == CellOutput(**stored[1])
        assert seen.persisted == [("c", stored)]

    async def test_no_displays_clears_what_was_persisted(self, tmp_path, exported):
        host, context, seen = _batch_fakes(tmp_path)

        await self._persist(host, context, tmp_path)

        assert seen.persisted == [("c", None)]


class TestRunBatch:
    """What the harness is started with, with a real frame pipe and no process behind it."""

    USER = object()

    @pytest.fixture
    def harness(self, monkeypatch):
        seen = SimpleNamespace(
            frames=[],
            hangs=False,
            spawned=[],
            handed=[],
            terminated=[],
            sent=[],
            writer=None,
        )

        def close_writer():
            if seen.writer is not None:
                os.close(seen.writer)
                seen.writer = None

        async def wait():
            # Lets the frame pipe's transport see its EOF and close before the test ends.
            for _ in range(5):
                await asyncio.sleep(0)

        async def create_subprocess_exec(*cmd, **kwargs):
            manifest = Path(cmd[3])
            seen.spawned.append((cmd, kwargs))
            seen.tmpdir = manifest.parent
            seen.manifest = json.loads(manifest.read_text())
            seen.writer = os.dup(kwargs["pass_fds"][0])
            os.write(seen.writer, _frame_bytes(*seen.frames))
            if not seen.hangs:
                close_writer()
            output = asyncio.StreamReader()
            output.feed_eof()
            seen.proc = SimpleNamespace(
                pid=1,
                stdin=SimpleNamespace(write=seen.sent.append, close=lambda: None),
                stdout=output,
                stderr=output,
                wait=wait,
            )
            return seen.proc

        async def terminate(proc):
            seen.terminated.append(proc)
            close_writer()

        monkeypatch.setattr(batch.asyncio, "create_subprocess_exec", create_subprocess_exec)
        monkeypatch.setattr(batch, "terminate_subprocess_tree", terminate)
        monkeypatch.setattr(batch, "resolve_harness_user", lambda: self.USER)
        monkeypatch.setattr(batch, "hand_over", lambda path, user: seen.handed.append((path, user)))
        monkeypatch.setattr(batch, "identity_env", lambda env, user: {"env": env, "user": user})
        monkeypatch.setattr(batch, "spawn_kwargs", lambda user: {"spawned_as": user})
        monkeypatch.setattr(batch, "lifeline_handoff", lambda: ((), {"LIFE": "1"}))
        yield seen
        close_writer()

    @staticmethod
    async def _run(tmp_path, *, host=None, batch_timeout=60.0, cell_timeout=60.0, **kwargs):
        (tmp_path / ".strata").mkdir(exist_ok=True)
        fake_host, context, seen = _batch_fakes(tmp_path)
        ran = await asyncio.wait_for(
            batch.run_batch(
                host or fake_host,
                context,
                [{"cell_id": "c", "source": "ran"}],
                use_cache=True,
                batch_timeout_seconds=batch_timeout,
                cell_timeout_seconds=cell_timeout,
                **kwargs,
            ),
            timeout=30,
        )
        return ran, seen

    async def test_no_interpreter_runs_nothing(self, tmp_path, harness):
        host, _, _ = _batch_fakes(tmp_path)
        host.venv_python = None

        ran, _ = await self._run(tmp_path, host=host)

        assert harness.spawned == []
        assert (ran.cell_results[0].status, ran.cell_results[0].error) == (
            "cell_error",
            "not ready",
        )
        assert (ran.failed_cell_id, ran.end_reason) == ("c", "cell_error")

    async def test_it_starts_the_harness_as_the_harness_user(self, tmp_path, harness):
        harness.frames = [_frame("batch_end", reason="complete")]

        ran, seen = await self._run(tmp_path, batch_timeout=9.0)

        assert ran.completed is True
        ((cmd, kwargs),) = harness.spawned
        assert cmd[1:3] == ("harness.py", "--batch")
        assert harness.tmpdir.parent == tmp_path / ".strata"
        assert harness.manifest["work_dir"] == "work"
        assert harness.manifest["run_token_on_stdin"] is True
        assert harness.handed == [(harness.tmpdir, self.USER)]
        assert kwargs["env"]["user"] is self.USER
        assert kwargs["env"]["env"]["LIFE"] == "1"
        assert kwargs["spawned_as"] is self.USER
        assert seen.tokens == [9.0]
        assert harness.sent == [b"token\n"]

    async def test_the_batch_directory_is_removed_after(self, tmp_path, harness):
        await self._run(tmp_path)

        assert harness.tmpdir.parent == tmp_path / ".strata"
        assert not harness.tmpdir.exists()

    async def test_it_announces_each_cell(self, tmp_path, harness):
        harness.frames = [_frame("cell_start", cell_id="c")]
        started = []

        async def on_cell_start(cell_id):
            started.append(cell_id)

        await self._run(tmp_path, on_cell_start=on_cell_start)

        assert started == ["c"]

    async def test_the_batch_timeout_ends_the_process_tree(self, tmp_path, harness):
        harness.hangs = True

        ran, _ = await self._run(tmp_path, batch_timeout=0.01)

        assert harness.terminated == [harness.proc]
        assert (ran.completed, ran.end_reason) == (False, "subprocess_died")

    async def test_a_cell_timeout_names_the_cell_and_reports_it(self, tmp_path, harness):
        harness.frames = [_frame("cell_start", cell_id="c")]
        harness.hangs = True
        events = []

        async def on_cell_event(result):
            events.append(result)

        ran, _ = await self._run(tmp_path, cell_timeout=0.01, on_cell_event=on_cell_event)

        assert harness.terminated == [harness.proc]
        assert (ran.end_reason, ran.failed_cell_id) == ("cell_timeout", "c")
        assert events == [ran.cell_results[0]]
        assert events[0].status == "cell_error"
        assert events[0].error == batch.cell_timeout_message(0.01)


async def test_the_executor_shares_its_run_state_with_the_python_path(tmp_path):
    nb = create_notebook(tmp_path / "nb", "Python")
    add_cell_to_notebook(nb, "c1", None)
    executor = CellExecutor(NotebookSession(parse_notebook(nb), nb))

    context = executor._python_context()

    assert context.received_output_files is executor._received_output_files
    assert context.folder_listings is executor._folder_listings
    assert context.each_items is executor._each_items
    assert context.each_listed is executor._each_listed
    assert context.branch_inits is executor._branch_inits
    assert context.open_runs is executor._open_runs
    assert context.variant_writes is executor._variant_writes


class TestPythonInputs:
    """What goes to a harness besides the source: fetched bytes and a restored release."""

    def test_two_fetches_of_one_file_name_stay_two_files(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "out").mkdir()
        (tmp_path / "a" / "data.csv").write_text("first")
        (tmp_path / "b" / "data.csv").write_text("second")
        specs = {}

        python._add_fetch_inputs(
            specs,
            {"a": tmp_path / "a" / "data.csv", "b": tmp_path / "b" / "data.csv"},
            tmp_path / "out",
        )

        assert (tmp_path / "out" / specs["a"]["file"]).read_text() == "first"
        assert (tmp_path / "out" / specs["b"]["file"]).read_text() == "second"
        assert specs["a"]["content_type"] == "file/path"

    def test_a_fetch_is_copied_where_it_cannot_be_linked(self, tmp_path, monkeypatch):
        (tmp_path / "out").mkdir()
        (tmp_path / "data.csv").write_text("bytes")

        def no_link(*_args):
            raise OSError("cross-device link")

        monkeypatch.setattr(python.os, "link", no_link)
        specs = {}

        python._add_fetch_inputs(specs, {"a": tmp_path / "data.csv"}, tmp_path / "out")

        assert (tmp_path / "out" / specs["a"]["file"]).read_text() == "bytes"

    def test_the_harness_user_skips_uv(self, monkeypatch):
        monkeypatch.setattr(python, "resolve_uv", lambda: "/bin/uv")

        command = python.python_harness_command(
            Path("harness.py"), Path("nb"), Path("m.json"), Path("venv/python"), object()
        )

        assert command == ["venv/python", "harness.py", "m.json"]

    def test_uv_runs_in_the_notebook(self, monkeypatch):
        monkeypatch.setattr(python, "resolve_uv", lambda: "/bin/uv")

        command = python.python_harness_command(
            Path("harness.py"), Path("nb"), Path("m.json"), Path("venv/python"), None
        )

        assert command == ["/bin/uv", "run", "--directory", "nb", "python", "harness.py", "m.json"]

    def test_without_uv_there_is_no_command(self, monkeypatch):
        monkeypatch.setattr(python, "resolve_uv", lambda: None)

        assert (
            python.python_harness_command(
                Path("harness.py"), Path("nb"), Path("m.json"), Path("venv/python"), None
            )
            is None
        )

    @staticmethod
    def _restore_host(state):
        version = SimpleNamespace(
            id="nb_n_cell_c_var_x", version=2, params={"content_type": "json/object"}
        )
        bundle = SimpleNamespace(
            release=SimpleNamespace(state=state, sequence=4), outputs=[version]
        )
        return SimpleNamespace(
            get_artifact_manager=lambda: SimpleNamespace(
                release_store=SimpleNamespace(bundle=lambda _id: bundle)
            )
        )

    def test_a_restore_names_where_each_part_came_from(self, tmp_path):
        run = SimpleNamespace(restores_from="r1", resumed_from="r1")

        restore = python.restore_manifest(
            self._restore_host("accepted"), lambda *_args: None, run, tmp_path
        )

        assert restore["parts"]["x"]["uri"] == "strata://artifact/nb_n_cell_c_var_x@v=2"
        assert restore["sequence"] == 4

    def test_a_checkpoint_file_is_copied_to_its_place_in_the_folder(self, tmp_path, monkeypatch):
        copied = []
        version = SimpleNamespace(id="file", version=3, params={"manifest": False})
        host = SimpleNamespace(
            get_artifact_manager=lambda: SimpleNamespace(
                release_store=SimpleNamespace(
                    bundle=lambda _id: SimpleNamespace(
                        release=SimpleNamespace(state="accepted", sequence=1), outputs=[]
                    )
                ),
                copy_ready_blob=lambda *args: copied.append(args),
            )
        )
        monkeypatch.setattr(
            python.checkpoint_files, "files_of", lambda _outputs: {"model/w.bin": version}
        )
        run = SimpleNamespace(restores_from="r1", resumed_from="r1")

        restore = python.restore_manifest(host, lambda *_args: None, run, tmp_path)

        target = tmp_path / restore["folder"] / "model" / "w.bin"
        assert copied == [("file", 3, target)]
        assert target.parent.is_dir()

    def test_a_release_that_was_not_accepted_is_not_restored(self, tmp_path):
        run = SimpleNamespace(restores_from="r1", resumed_from="r1")

        with pytest.raises(RuntimeError, match="not an accepted release"):
            python.restore_manifest(
                self._restore_host("rejected"), lambda *_args: None, run, tmp_path
            )


class TestPython:
    """The moved path against fakes; the harness-backed behavior is in ``test_executor.py``."""

    SOURCE = "x = 1\n"
    DISPLAY = SimpleNamespace(
        artifact_uri="strata://artifact/display@v=1",
        model_dump=lambda: {"kind": "display"},
    )
    CONSOLE = SimpleNamespace(stdout="old out", stderr="old err", run_ms=9)
    LOCAL = SimpleNamespace(
        name="local", backend=WorkerBackendType.LOCAL, config=SimpleNamespace(url="", transport="")
    )
    HTTP = SimpleNamespace(
        name="w",
        backend=WorkerBackendType.EXECUTOR,
        config=SimpleNamespace(url="http://w", transport="direct"),
    )
    SIGNED = SimpleNamespace(
        name="w",
        backend=WorkerBackendType.EXECUTOR,
        config=SimpleNamespace(url="http://w", transport="signed"),
    )
    EMBEDDED = SimpleNamespace(
        name="e",
        backend=WorkerBackendType.EXECUTOR,
        config=SimpleNamespace(url="embedded://local", transport="direct"),
    )

    @staticmethod
    def _prov(annotations=None, **overrides):
        fields = {"nocache": False, "tables": [], "table_writes": [], "stochastic": False}
        prov = SimpleNamespace(
            source_hash="source",
            runtime_env={"K": "V"},
            effective_worker=None,
            env_hash="env",
            input_hashes=["input"],
            mount_specs=[],
            mount_fingerprints=[],
            table_fingerprints=[],
            provenance_hash="prov",
            has_rw_mount=False,
            annotations=SimpleNamespace(**{**fields, **(annotations or {})}),
            fetch_error=None,
            fetch_error_code=None,
            dataset_error=None,
            fetched={},
            datasets=[],
            table_snapshots={},
            table_reads=[],
            key_parts={},
            reads=lambda *_args: {},
        )
        prov.over_mounts = lambda _mounts: prov
        prov.for_instance = lambda _instance: prov
        for name, value in overrides.items():
            setattr(prov, name, value)
        return prov

    def _fakes(
        self,
        monkeypatch,
        prov=None,
        *,
        worker=None,
        consumed=(),
        hit=None,
        displays=(),
        console=None,
        result=None,
        **overrides,
    ):
        """A host and a context with nothing behind them, and a record of what they were asked."""
        worker = worker or self.LOCAL
        prov = prov or self._prov()
        calls = SimpleNamespace(
            materialized=[],
            promoted=[],
            recorded=[],
            applied=[],
            persisted=[],
            kept=[],
            stored=[],
            pushed=[],
            consoles=[],
            writes=[],
            good=[],
            dispatched=[],
            prepared=[],
            opened=[],
            closed=[],
            loaded=[],
            cell=SimpleNamespace(
                display_outputs=[],
                display_output=None,
                artifact_uris={},
                artifact_uri=None,
                cache_hit=False,
                last_provenance_hash=None,
                mutation_defines=["defined"],
            ),
            export_error=None,
        )
        monkeypatch.setattr(python, "resolve_worker_spec", lambda *_args: worker)
        monkeypatch.setattr(
            python, "write_module_export_outputs", lambda *_args: calls.export_error
        )
        host = SimpleNamespace(
            notebook_state=SimpleNamespace(id="nb", get_cell=lambda _id: calls.cell),
            dag=SimpleNamespace(each_cells=set(), consumed_variables={"c": set(consumed)}),
            venv_python=Path("venv/python"),
            get_artifact_manager=lambda: SimpleNamespace(
                cell_artifact_id=lambda cell_id, var, variant=None: f"{cell_id}.{var}",
                artifact_store=SimpleNamespace(get_latest_version=lambda _id: hit),
            ),
            resolve_cached_display_outputs=lambda *_args: list(displays),
            resolve_cached_console=lambda *_args, **_kwargs: console,
            record_successful_execution_provenance=lambda *args, **kwargs: calls.recorded.append(
                (args, kwargs)
            ),
            apply_execution_result_metadata=lambda cell_id, res: calls.applied.append(
                (cell_id, res)
            ),
            persist_display_outputs=lambda cell_id, outputs: calls.persisted.append(
                (cell_id, outputs)
            ),
        )

        async def execute_each_cell(*_args, **_kwargs):
            raise AssertionError("not an each cell")

        async def execute_fanout_cell(*_args, **_kwargs):
            raise AssertionError("not a fan-out")

        async def materialize_upstreams(cell_id):
            calls.materialized.append(cell_id)

        async def compute_provenance(*_args):
            return prov

        def remote_execution_metadata(spec, remote_build_id=None):
            metadata = {
                "remote_worker": spec.name,
                "remote_transport": python.worker_transport(spec),
            }
            return {**metadata, **({"remote_build_id": remote_build_id} if remote_build_id else {})}

        async def find_or_pull_outputs(*_args, **_kwargs):
            return hit, None

        def promote_reverted_leaf(*args):
            calls.promoted.append(args)
            return False

        def load_input_blobs(cell_id, output_dir, fanout_variant=None):
            calls.loaded.append(fanout_variant)
            return {}

        async def prepare_mounts(specs, cell_id):
            calls.prepared.append(specs)
            return {}

        def open_local_run(**kwargs):
            calls.opened.append(kwargs)
            return SimpleNamespace(restores_from=None)

        local_runs = SimpleNamespace(open=open_local_run, start_renewing=lambda _run: "renewing")

        async def dispatch_execution(*args, **kwargs):
            calls.dispatched.append((args, kwargs))
            outcome = result if result is not None else {"success": True}
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome, args[4], "cold", {}

        def parse_result(cell_id, res, duration_ms, method):
            return CellExecutionResult(
                cell_id=cell_id,
                success=res.get("success", True),
                error=res.get("error"),
                stdout="printed",
                stderr="warned",
                duration_ms=duration_ms,
                display_outputs=res.get("display_outputs", []),
                mutation_warnings=res.get("mutation_warnings", []),
                execution_method=method,
            )

        async def collect(*_args):
            return "collected"

        def store_outputs(*args, **kwargs):
            calls.stored.append((args, kwargs))
            return True

        async def close_open_run(cell_id):
            calls.closed.append(cell_id)

        async def push_to_team_store(**kwargs):
            calls.pushed.append(kwargs)

        async def sync_back_mounts(_mounts):
            return []

        fields = {
            "execute_each_cell": execute_each_cell,
            "execute_fanout_cell": execute_fanout_cell,
            "fanout_variants": lambda _cell_id: None,
            "materialize_upstreams": materialize_upstreams,
            "compute_provenance": compute_provenance,
            "remote_execution_metadata": remote_execution_metadata,
            "manifest_tables": lambda *_args: {},
            "find_or_pull_outputs": find_or_pull_outputs,
            "validate_output_hit": lambda *args: (args[5], args[7]),
            "promote_reverted_leaf": promote_reverted_leaf,
            "moved": lambda _cell_id: ({}, {}),
            "table_write_refusal": lambda _source: None,
            "restore_gone_tables": _nothing_gone,
            "load_input_blobs": load_input_blobs,
            "add_dataset_inputs": lambda *_args: None,
            "prepare_mounts": prepare_mounts,
            "local_runs": lambda: local_runs,
            "input_refs": lambda *_args: {},
            "restore_manifest": lambda *_args: None,
            "dispatch_execution": dispatch_execution,
            "parse_result": parse_result,
            "collect_output_dirs_off_loop": collect,
            "keep_failed_output_dirs": lambda *args, **kwargs: calls.kept.append((args, kwargs)),
            "store_outputs": store_outputs,
            "close_open_run": close_open_run,
            "push_to_team_store": push_to_team_store,
            "store_result_displays": lambda *_args, **_kwargs: None,
            "store_console_outputs": lambda *args, **kwargs: calls.consoles.append((args, kwargs)),
            "store_mount_writes": lambda *args, **kwargs: calls.writes.append((args, kwargs)),
            "record_good_run": lambda *args: calls.good.append(args),
            "timeout_message": lambda _cell_id, seconds: f"timed out after {seconds}",
            "sync_back_mounts": sync_back_mounts,
            "received_output_files": {},
            "folder_listings": {},
            "each_items": {},
            "each_listed": {},
            "branch_inits": {},
            "open_runs": {},
            "variant_writes": {},
        }
        calls.context = PythonContext(**{**fields, **overrides})
        calls.host = host
        return host, calls.context, calls

    @staticmethod
    async def _run(host, context, *, source=None, use_cache=True, materialize=True, variant=None):
        return await run_python_cell(
            host,
            context,
            "c",
            TestPython.SOURCE if source is None else source,
            7.0,
            0.0,
            materialize_upstreams=materialize,
            use_cache=use_cache,
            fanout_variant=variant,
        )

    # A run's start

    async def test_a_run_drops_what_an_earlier_one_uploaded(self, monkeypatch):
        host, context, _calls = self._fakes(
            monkeypatch,
            received_output_files={"c": ["old"]},
            folder_listings={"c": ["old"]},
        )

        await self._run(host, context)

        assert "c" not in context.received_output_files
        assert "c" not in context.folder_listings

    async def test_upstreams_are_materialized_only_when_asked(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch)

        await self._run(host, context, materialize=False)
        assert calls.materialized == []

        await self._run(host, context, materialize=True)
        assert calls.materialized == ["c"]

    # Fetches

    @staticmethod
    def _fetched(tmp_path):
        path = tmp_path / "data.csv"
        path.write_text("bytes")
        return {"f": path}

    async def test_a_fetch_is_a_read_only_mount_here(self, monkeypatch, tmp_path):
        prov = self._prov(fetched=self._fetched(tmp_path))
        host, context, calls = self._fakes(monkeypatch, prov)

        await self._run(host, context)

        args, _kwargs = calls.dispatched[0]
        assert "f" not in args[2]
        assert [(m.name, m.mode) for m in args[3]] == [("f", MountMode.READ_ONLY)]

    async def test_a_fetch_travels_as_an_input_to_a_worker(self, monkeypatch, tmp_path):
        prov = self._prov(fetched=self._fetched(tmp_path))
        host, context, calls = self._fakes(monkeypatch, prov, worker=self.HTTP)

        await self._run(host, context)

        args, _kwargs = calls.dispatched[0]
        assert args[2]["f"]["content_type"] == "file/path"
        assert args[3] == []

    # The cache check

    async def test_a_consumer_does_not_hit_on_a_console_alone(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, consumed=["x"], console=self.CONSOLE)

        result = await self._run(host, context)

        assert result.cache_hit is False
        assert calls.dispatched

    async def test_a_leaf_is_not_promoted_when_the_cache_is_off(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch)

        await self._run(host, context, use_cache=False)

        assert calls.promoted == []

    async def test_a_leaf_hits_on_its_displays_alone(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, displays=[self.DISPLAY])

        result = await self._run(host, context)

        assert result.cache_hit is True
        assert result.execution_method == "cached"
        assert result.artifact_uri == self.DISPLAY.artifact_uri
        assert calls.dispatched == []

    async def test_a_hit_updates_the_cell(self, monkeypatch):
        other = SimpleNamespace(artifact_uri="u2", model_dump=lambda: {"kind": "last"})
        host, context, calls = self._fakes(monkeypatch, displays=[self.DISPLAY, other])

        result = await self._run(host, context)

        assert calls.cell.cache_hit is True
        assert calls.cell.display_outputs == [self.DISPLAY, other]
        assert calls.cell.display_output is other
        assert result.display_output == {"kind": "last"}

    async def test_a_hit_reports_how_long_the_check_took(self, monkeypatch):
        host, context, _calls = self._fakes(monkeypatch, displays=[self.DISPLAY])

        # Started at the epoch, so any elapsed time is far above zero.
        result = await self._run(host, context)

        assert result.duration_ms > 0

    async def test_a_signed_workers_hit_is_a_ready_build(self, monkeypatch):
        host, context, _calls = self._fakes(
            monkeypatch, worker=self.SIGNED, displays=[self.DISPLAY]
        )

        result = await self._run(host, context)

        assert result.remote_build_state == "ready"

    @pytest.mark.parametrize(
        ("variant", "last_hash", "replaces"),
        [("v", None, False), (None, "prov", False), (None, "older", True)],
    )
    async def test_a_hit_with_no_console_record_replaces_only_another_keys_console(
        self, monkeypatch, variant, last_hash, replaces
    ):
        host, context, calls = self._fakes(monkeypatch, displays=[self.DISPLAY])
        calls.cell.last_provenance_hash = last_hash

        result = await self._run(host, context, variant=variant)

        assert result.replaces_console is replaces

    async def test_a_variants_hit_is_left_for_its_fan_out_to_record(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, displays=[self.DISPLAY])

        await self._run(host, context, variant="v")
        assert calls.recorded == []

        await self._run(host, context)
        assert calls.recorded[0][0] == ("c", "prov", "source", "env")

    # Refusals before anything runs

    async def test_a_refused_table_write_does_not_run(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, table_write_refusal=lambda _s: "refused")

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "refused"
        assert result.execution_method == "error"
        assert calls.dispatched == []

    @staticmethod
    def _refuses(*_args, **_kwargs):
        raise RunRefused("no")

    @staticmethod
    def _no_snapshot(*_args):
        raise RuntimeError("no")

    @pytest.mark.parametrize(
        ("prov_fields", "overrides"),
        [
            pytest.param({"fetch_error": "no"}, {}, id="fetch"),
            pytest.param({"dataset_error": "no"}, {}, id="dataset"),
            pytest.param({}, {"manifest_tables": "_no_snapshot"}, id="tables"),
            pytest.param({}, {"table_write_refusal": lambda _s: "no"}, id="table-write"),
            pytest.param({}, {"local_runs": "_refuses"}, id="running"),
        ],
    )
    async def test_a_refusal_ends_as_any_failure_does(self, monkeypatch, prov_fields, overrides):
        if overrides.get("manifest_tables"):
            overrides = {"manifest_tables": self._no_snapshot}
        if overrides.get("local_runs"):
            overrides = {"local_runs": lambda: SimpleNamespace(open=self._refuses)}
        host, context, calls = self._fakes(monkeypatch, self._prov(**prov_fields), **overrides)

        result = await self._run(host, context)

        assert result.error == "no"
        assert calls.dispatched == []
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    async def test_a_second_run_of_a_running_build_is_refused(self, monkeypatch):
        def refuse(**_kwargs):
            raise RunRefused("already running")

        host, context, calls = self._fakes(
            monkeypatch,
            local_runs=lambda: SimpleNamespace(open=refuse),
        )

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "already running"
        assert result.execution_method == "error"
        assert calls.dispatched == []

    # Mounts

    async def test_an_embedded_worker_gets_mounts_this_server_prepared(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, worker=self.EMBEDDED)

        await self._run(host, context)

        assert calls.prepared == [[]]
        assert calls.dispatched[0][1]["resolved_mounts"] == {}

    async def test_an_http_worker_prepares_its_own(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, worker=self.HTTP)

        await self._run(host, context)

        assert calls.prepared == []
        assert calls.dispatched[0][1]["resolved_mounts"] is None

    @staticmethod
    def _listed_mount():
        spec = MountSpec(name="raw", uri="file:///raw")
        return spec, ResolvedMount(spec=spec, local_path=Path("/raw"), fingerprint="fp")

    async def test_a_plain_run_lists_its_mounts_itself(self, monkeypatch):
        spec, mount = self._listed_mount()
        prov = self._prov(mount_specs=[spec])
        host, context, calls = self._fakes(monkeypatch, prov, each_listed={"c": {"raw": mount}})

        await self._run(host, context)

        assert calls.prepared == [[spec]]

    async def test_an_item_binds_as_a_mount_of_its_own_file(self, monkeypatch, tmp_path):
        spec, mount = self._listed_mount()
        item = SimpleNamespace(instance="inst", path=tmp_path / "a.csv", digest="digest")
        prov = self._prov(mount_specs=[spec])
        host, context, calls = self._fakes(
            monkeypatch,
            prov,
            each_listed={"c": {"raw": mount}},
            each_items={"c": {"v": item}},
        )

        await self._run(
            host,
            context,
            source="# @mount raw file:///raw ro\n# @each path in raw\nx = 1\n",
            variant="v",
        )

        assert calls.prepared == [[]]
        bound = calls.dispatched[0][1]["resolved_mounts"]
        assert bound["raw"] is mount
        assert bound["path"].fingerprint == "digest"
        assert bound["path"].local_path == item.path
        assert bound["path"].spec.mode == MountMode.READ_ONLY

    # The run

    async def test_a_local_run_opens_under_the_cells_timeout(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch)

        await self._run(host, context)

        assert calls.opened[0]["timeout_seconds"] == 7.0

    async def test_a_run_is_told_what_the_cell_mutates(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch)

        await self._run(host, context)

        assert calls.dispatched[0][1]["mutation_defines"] == ["defined"]

    async def test_a_signed_build_is_ready_once_it_returns(self, monkeypatch):
        host, context, _calls = self._fakes(monkeypatch, worker=self.SIGNED)

        result = await self._run(host, context)

        assert result.remote_build_state == "ready"

    async def test_a_failed_run_keeps_its_output_dirs_under_its_variant(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, result={"success": False, "error": "boom"})

        await self._run(host, context, variant="v")

        assert calls.kept[0][0][3:5] == ("prov", "env")
        assert calls.kept[0][1] == {"variant": "v"}

    async def test_a_failed_run_keeps_no_displays(self, monkeypatch):
        host, context, calls = self._fakes(
            monkeypatch,
            result={"success": False, "error": "boom", "display_outputs": [{"kind": "x"}]},
        )

        await self._run(host, context)

        assert calls.persisted == [("c", None)]

    # After a run

    async def test_an_export_failure_keeps_what_the_run_reported(self, monkeypatch):
        host, context, calls = self._fakes(
            monkeypatch,
            worker=self.HTTP,
            result={"success": True, "mutation_warnings": ["careful"]},
        )
        calls.export_error = "cannot export"

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "cannot export"
        assert result.stdout == "printed"
        assert result.stderr == "warned"
        assert result.mutation_warnings == ["careful"]
        assert result.remote_worker == "w"
        assert calls.stored == []

    async def test_outputs_are_stored_with_what_the_run_reported(self, monkeypatch):
        prov = self._prov(annotations={"stochastic": True})
        host, context, calls = self._fakes(
            monkeypatch,
            prov,
            result={"success": True, "build_env": "linux", "hardware": {"gpu": "a"}},
        )

        await self._run(host, context, variant="v")

        _args, kwargs = calls.stored[0]
        assert kwargs["variant"] == "v"
        assert kwargs["build_env"] == "linux"
        assert kwargs["hardware"] == {"gpu": "a"}
        assert kwargs["collected"] == "collected"
        assert kwargs["stochastic"] is True
        assert calls.pushed == [{"cell_id": "c", "variant": "v"}]

    async def test_a_table_write_failure_is_named_and_not_pushed(self, monkeypatch):
        def store_outputs(*_args, **_kwargs):
            raise TableWriteError("no such warehouse")

        host, context, calls = self._fakes(monkeypatch, store_outputs=store_outputs)

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "no such warehouse"
        assert calls.closed == ["c"]
        assert calls.pushed == []

    async def test_a_false_store_fails_the_run_and_is_not_pushed(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, store_outputs=lambda *_a, **_k: False)

        result = await self._run(host, context)

        assert result.success is False
        assert calls.pushed == []
        assert calls.consoles == []

    async def test_only_a_cacheable_runs_console_is_stored(self, monkeypatch):
        prov = self._prov(annotations={"nocache": True})
        host, context, calls = self._fakes(monkeypatch, prov)

        await self._run(host, context)

        assert calls.consoles == []

    async def test_a_console_is_stored_with_its_display_count_and_variant(self, monkeypatch):
        host, context, calls = self._fakes(
            monkeypatch, result={"success": True, "display_outputs": [{"kind": "x"}]}
        )

        await self._run(host, context, variant="v")

        _args, kwargs = calls.consoles[0]
        assert kwargs["display_count"] == 1
        assert kwargs["variant"] == "v"
        assert calls.persisted == [("c", [{"kind": "x"}])]

    @staticmethod
    def _rw_write():
        return {"mount": "out", "uri": "file:///out", "files": []}

    async def test_what_a_worker_reports_writing_is_recorded(self, monkeypatch):
        spec = MountSpec(name="out", uri="file:///out", mode=MountMode.READ_WRITE)
        prov = self._prov(mount_specs=[spec])
        host, context, calls = self._fakes(
            monkeypatch,
            prov,
            worker=self.HTTP,
            result={"success": True, "mount_writes": [self._rw_write()]},
        )

        await self._run(host, context)

        written = calls.writes[0][0][2]
        assert [write.mount for write in written] == ["out"]
        assert calls.good[0][5] == written

    async def test_mounts_a_run_used_are_synced_back(self, monkeypatch):
        write = MountWrite(mount="out", uri="file:///out", files=[])
        synced = []

        async def sync_back_mounts(mounts):
            synced.append(mounts)
            return [write]

        host, context, calls = self._fakes(monkeypatch, sync_back_mounts=sync_back_mounts)

        async def dispatch_execution(*args, **_kwargs):
            return {"success": True}, args[4], "cold", {"out": "resolved"}

        context = replace(context, dispatch_execution=dispatch_execution)

        await self._run(host, context)

        assert synced == [{"out": "resolved"}]
        assert calls.writes[0][0][2] == [write]

    async def test_a_failed_sync_keeps_the_workers_metadata(self, monkeypatch):
        async def sync_back_mounts(_mounts):
            raise OSError("disk gone")

        async def dispatch_execution(*args, **_kwargs):
            return {"success": True}, args[4], "cold", {"out": "resolved"}

        host, context, calls = self._fakes(
            monkeypatch,
            worker=self.HTTP,
            sync_back_mounts=sync_back_mounts,
            dispatch_execution=dispatch_execution,
        )

        result = await self._run(host, context)

        assert result.success is False
        assert result.remote_worker == "w"
        assert calls.good == []

    async def test_a_variant_leaves_its_good_run_to_the_fan_out(self, monkeypatch):
        write = MountWrite(mount="out", uri="file:///out", files=[])

        async def sync_back_mounts(_mounts):
            return [write]

        async def dispatch_execution(*args, **_kwargs):
            return {"success": True}, args[4], "cold", {"out": "resolved"}

        host, context, calls = self._fakes(
            monkeypatch,
            sync_back_mounts=sync_back_mounts,
            dispatch_execution=dispatch_execution,
        )

        await self._run(host, context, variant="v")

        assert calls.good == []
        assert context.variant_writes == {"c": [write]}

    async def test_a_good_run_is_recorded(self, monkeypatch):
        prov = self._prov()
        host, context, calls = self._fakes(monkeypatch, prov)

        await self._run(host, context)

        assert calls.good == [("c", "prov", "source", "env", prov, [])]

    # Failures

    async def test_a_cancel_keeps_what_the_worker_uploaded(self, monkeypatch):
        host, context, calls = self._fakes(
            monkeypatch, result=asyncio.CancelledError(), received_output_files={}
        )

        async def dispatch_execution(*_args, **_kwargs):
            context.received_output_files["c"] = ["up"]
            raise asyncio.CancelledError()

        context = replace(context, dispatch_execution=dispatch_execution)

        with pytest.raises(asyncio.CancelledError):
            await self._run(host, context, variant="v")

        assert calls.kept[0][0][:3] == ("c", self.SOURCE, None)
        assert calls.kept[0][1] == {"variant": "v"}

    @staticmethod
    def _lost():
        return RemoteExecutionError("lost it", remote_build_state="failed", remote_error_code="E1")

    async def test_a_lost_worker_keeps_what_it_uploaded_and_says_so(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, worker=self.HTTP)

        async def dispatch_execution(*_args, **_kwargs):
            context.received_output_files["c"] = ["up"]
            raise self._lost()

        context = replace(context, dispatch_execution=dispatch_execution)

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "lost it"
        assert result.duration_ms > 0
        assert result.remote_worker == "w"
        assert result.remote_build_state == "failed"
        assert result.remote_error_code == "E1"
        assert calls.kept[0][0][:3] == ("c", self.SOURCE, None)
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    async def test_a_lost_worker_with_no_uploads_keeps_nothing(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, worker=self.HTTP, result=self._lost())

        await self._run(host, context)

        assert calls.kept == []

    async def test_a_timeout_names_the_limit_and_clears_the_displays(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, worker=self.HTTP, result=TimeoutError())

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "timed out after 7.0"
        assert result.remote_worker == "w"
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    async def test_any_other_failure_is_a_failed_result(self, monkeypatch):
        host, context, calls = self._fakes(monkeypatch, worker=self.HTTP, result=ValueError("bad"))

        result = await self._run(host, context)

        assert result.success is False
        assert result.error == "Execution failed: bad"
        assert result.remote_worker == "w"
        assert calls.persisted == [("c", None)]
        assert calls.applied == [("c", result)]

    @pytest.mark.parametrize(
        ("error", "message"),
        [
            pytest.param(TimeoutError(), "timed out after 7.0", id="timeout"),
            pytest.param(ValueError("bad"), "Execution failed: bad", id="other"),
        ],
    )
    @pytest.mark.parametrize("uploaded", [True, False])
    async def test_a_failed_dispatch_keeps_what_the_worker_uploaded(
        self, monkeypatch, error, message, uploaded
    ):
        host, context, calls = self._fakes(monkeypatch, worker=self.HTTP)

        async def dispatch_execution(*_args, **_kwargs):
            if uploaded:
                context.received_output_files["c"] = ["up"]
            raise error

        context = replace(context, dispatch_execution=dispatch_execution)

        result = await self._run(host, context, variant="v")

        assert result.error == message
        if uploaded:
            assert calls.kept[0][0][:3] == ("c", self.SOURCE, None)
            assert calls.kept[0][1] == {"variant": "v"}
        else:
            assert calls.kept == []

    async def test_an_unsafe_file_at_the_store_is_named_and_closes_the_run(self, monkeypatch):
        host, context, calls = self._fakes(
            monkeypatch, store_outputs=_raises(UnsafeRunFile("a link"))
        )

        result = await self._run(host, context)

        assert result.error == "a link"
        assert calls.closed == ["c"]
        assert calls.pushed == []

    async def test_a_step_after_the_run_keeps_what_it_printed(self, monkeypatch):
        host, context, _calls = self._fakes(
            monkeypatch,
            result={"success": True, "stdout": "printed", "mutation_warnings": [{"variable": "x"}]},
            store_result_displays=_raises(RuntimeError("no display")),
        )

        result = await self._run(host, context)

        assert result.error == "Execution failed: no display"
        assert (result.stdout, result.stderr) == ("printed", "warned")
        assert result.mutation_warnings == [{"variable": "x"}]


class TestPythonDispatchLocal:
    """A run on this machine: the warm pool first, then the cold harness."""

    @staticmethod
    def _fakes(monkeypatch, *, pool=None, pool_result=None, **overrides):
        seen = SimpleNamespace(manifests=[], harness=[], warm=[], tokens=[])

        @contextmanager
        def run_token(_seconds):
            token = f"token{len(seen.tokens)}"
            seen.tokens.append(token)
            yield token

        def write_manifest(*args, **kwargs):
            seen.manifests.append((args, kwargs))
            return Path("run/manifest.json")

        async def run_harness(*args, **kwargs):
            seen.harness.append((args, kwargs))
            return {"success": True, "from": "cold"}

        async def execute_with_pool(*args, **kwargs):
            seen.warm.append((args, kwargs))
            if isinstance(pool_result, BaseException):
                raise pool_result
            return pool_result

        monkeypatch.setattr(
            "strata.notebook.pool.PooledCellExecutor.execute_with_pool",
            staticmethod(execute_with_pool),
        )

        async def prepare_mounts(*_args):
            return {}

        fields = {
            "prepare_mounts": prepare_mounts,
            "local_runs": lambda: SimpleNamespace(event_handler=lambda *_args: "on event"),
            "run_token": run_token,
            "write_manifest": write_manifest,
            "run_harness": run_harness,
            "pool": pool,
        }
        return (
            SimpleNamespace(path=Path("nb")),
            LocalDispatchContext(**{**fields, **overrides}),
            seen,
        )

    @staticmethod
    async def _dispatch(host, context, *, venv=Path("venv/python"), local_run="run"):
        return await dispatch_local(
            host,
            context,
            "x = 1\n",
            {},
            [],
            Path("out"),
            venv,
            {},
            7.0,
            mutation_defines=["defined"],
            tables={"t": {}},
            cell_id="c",
            local_run=local_run,
            restore={"release_id": "r"},
            resolved_mounts={},
        )

    async def test_the_manifest_gets_all_of_the_run(self, monkeypatch):
        host, context, seen = self._fakes(monkeypatch)

        await self._dispatch(host, context)

        _args, kwargs = seen.manifests[0]
        assert kwargs == {
            "mutation_defines": ["defined"],
            "tables": {"t": {}},
            "cell_id": "c",
            "run_token": "token0",
            "restore": {"release_id": "r"},
        }

    async def test_the_cold_harness_gets_its_own_token_and_the_runs_events(self, monkeypatch):
        host, context, seen = self._fakes(monkeypatch)

        result, _output, method, _mounts = await self._dispatch(host, context)

        assert result["from"] == "cold"
        assert method == "cold"
        assert seen.harness[0][1] == {"run_token": "token1", "on_event": "on event"}

    async def test_a_run_outside_the_ledger_has_no_event_handler(self, monkeypatch):
        host, context, seen = self._fakes(monkeypatch)

        await self._dispatch(host, context, local_run=None)

        assert seen.harness[0][1]["on_event"] is None

    async def test_the_warm_pool_gets_the_token_and_the_runs_events(self, monkeypatch):
        host, context, seen = self._fakes(monkeypatch, pool="pool", pool_result={"success": True})

        _result, _output, method, _mounts = await self._dispatch(host, context)

        assert method == "warm"
        assert seen.warm[0][1] == {"run_token": "token0", "on_event": "on event"}
        assert seen.harness == []

    async def test_without_an_interpreter_the_pool_is_not_asked(self, monkeypatch):
        host, context, seen = self._fakes(monkeypatch, pool="pool", pool_result={"success": True})

        await self._dispatch(host, context, venv=None)

        assert seen.warm == []
        assert seen.harness

    async def test_a_fenced_warm_run_is_a_failed_result(self, monkeypatch):
        host, context, seen = self._fakes(
            monkeypatch, pool="pool", pool_result=RunFenced("lost the build")
        )

        result, _output, method, _mounts = await self._dispatch(host, context)

        assert result["success"] is False
        assert result["error"] == "lost the build"
        assert method == "warm"
        assert seen.harness == []


class TestPythonHarness:
    """What the cold harness is started with, with no process behind it."""

    USER = object()
    MESSAGE = "no interpreter here"

    @pytest.fixture
    def spawned(self, monkeypatch):
        seen = SimpleNamespace(
            handed=[],
            terminated=[],
            spawn=None,
            outcome=(b"out", b"err"),
            sent=[],
            timeouts=[],
            extra=[],
            relay=None,
            eof=None,
            relay_cancelled=False,
        )

        async def communicate(data=None):
            seen.sent.append(data)
            await asyncio.sleep(0)
            if isinstance(seen.outcome, BaseException):
                raise seen.outcome
            return seen.outcome

        seen.proc = SimpleNamespace(pid=1, communicate=communicate)

        async def create_subprocess_exec(*cmd, **kwargs):
            seen.spawn = (cmd, kwargs)
            return seen.proc

        async def terminate(proc):
            seen.terminated.append(proc)

        async def relay_events(read_fd, on_event, proc):
            # A non-blocking read sees EOF only if the executor closed its write end.
            os.set_blocking(read_fd, False)
            try:
                seen.eof = os.read(read_fd, 1) == b""
            except BlockingIOError:
                seen.eof = False
            os.close(read_fd)
            seen.relay = (on_event, proc)
            if seen.relay_hangs:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    seen.relay_cancelled = True
                    raise
            return seen.fence

        real_wait_for = asyncio.wait_for

        async def wait_for(awaitable, timeout):
            seen.timeouts.append(timeout)
            return await real_wait_for(awaitable, timeout)

        seen.relay_hangs = False
        seen.fence = None
        monkeypatch.setattr(python.asyncio, "create_subprocess_exec", create_subprocess_exec)
        monkeypatch.setattr(python.asyncio, "wait_for", wait_for)
        monkeypatch.setattr("strata.notebook.process_tree.terminate_subprocess_tree", terminate)
        monkeypatch.setattr(
            "strata.notebook.process_tree.subprocess_kwargs_for_new_group",
            lambda: {"new_group": True},
        )
        monkeypatch.setattr(python, "relay_events", relay_events)
        monkeypatch.setattr(python, "resolve_harness_user", lambda: self.USER)
        monkeypatch.setattr(
            python, "hand_over", lambda path, user: seen.handed.append((path, user))
        )
        monkeypatch.setattr(python, "identity_env", lambda env, user: {"env": env, "user": user})
        monkeypatch.setattr(python, "uv_env", lambda env: {"uv": env})
        monkeypatch.setattr(python, "spawn_kwargs", lambda user: {"spawned_as": user})
        monkeypatch.setattr(python, "lifeline_handoff", lambda: ((), {"LIFE": "1"}))
        return seen

    @staticmethod
    async def _run(
        tmp_path, *, command=("harness",), run_token="tok", on_event=None, venv=Path("py")
    ):
        return await python.run_harness(
            SimpleNamespace(
                path=Path("notebook"),
                environment_execution_block_message=lambda: TestPythonHarness.MESSAGE,
            ),
            lambda *_args: None if command is None else list(command),
            lambda extra: {"extra": extra},
            tmp_path / "manifest.json",
            venv,
            7.0,
            run_token=run_token,
            on_event=on_event,
        )

    @staticmethod
    async def _on_event(_event):
        return None

    async def test_it_starts_as_the_harness_user(self, spawned, tmp_path):
        await self._run(tmp_path)

        cmd, kwargs = spawned.spawn
        assert cmd == ("harness",)
        assert spawned.handed == [(tmp_path, self.USER)]
        assert kwargs["env"] == {"uv": {"env": {"extra": {"LIFE": "1"}}, "user": self.USER}}
        assert kwargs["spawned_as"] is self.USER
        assert kwargs["new_group"] is True
        assert kwargs["cwd"] == "notebook"

    async def test_the_token_goes_on_its_stdin(self, spawned, tmp_path):
        await self._run(tmp_path, run_token="tok")

        assert spawned.spawn[1]["stdin"] == asyncio.subprocess.PIPE
        assert spawned.sent == [b"tok\n"]

    async def test_no_token_means_no_stdin(self, spawned, tmp_path):
        await self._run(tmp_path, run_token=None)

        assert spawned.spawn[1]["stdin"] is None
        assert spawned.sent == [None]

    async def test_it_runs_under_the_cells_timeout(self, spawned, tmp_path):
        await self._run(tmp_path)

        assert spawned.timeouts == [7.0]

    async def test_no_interpreter_runs_nothing(self, spawned, tmp_path):
        result = await self._run(tmp_path, venv=None)

        assert result["success"] is False
        assert result["error"] == self.MESSAGE
        assert spawned.spawn is None

    async def test_a_refused_host_runs_nothing(self, spawned, tmp_path, monkeypatch):
        def refuse():
            raise LocalExecutionRefused("not here")

        monkeypatch.setattr(python, "resolve_harness_user", refuse)

        result = await self._run(tmp_path)

        assert result["success"] is False
        assert result["error"] == "not here"
        assert spawned.spawn is None

    async def test_no_uv_runs_nothing_and_says_so(self, spawned, tmp_path):
        result = await self._run(tmp_path, command=None)

        assert result["success"] is False
        assert result["error"] == python.UV_NOT_FOUND_MESSAGE
        assert spawned.spawn is None

    async def test_a_process_that_cannot_start_leaves_no_pipe_open(
        self, spawned, tmp_path, monkeypatch
    ):
        real_pipe = os.pipe
        ends = []

        def pipe():
            ends.extend(real_pipe())
            return ends[0], ends[1]

        async def cannot_start(*_cmd, **_kwargs):
            raise OSError("no such file")

        monkeypatch.setattr(python.os, "pipe", pipe)
        monkeypatch.setattr(python.asyncio, "create_subprocess_exec", cannot_start)

        with pytest.raises(OSError, match="no such file"):
            await self._run(tmp_path, on_event=self._on_event)

        assert len(ends) == 2
        for end in ends:
            with pytest.raises(OSError, match="Bad file descriptor"):
                os.fstat(end)

    async def test_the_event_pipe_reaches_the_harness_and_the_relay(self, spawned, tmp_path):
        await self._run(tmp_path, on_event=self._on_event)

        _cmd, kwargs = spawned.spawn
        write_fd = kwargs["pass_fds"][0]
        extra = kwargs["env"]["uv"]["env"]["extra"]
        assert extra[python.EVENTS_FD_ENV] == str(write_fd)
        assert spawned.relay == (self._on_event, spawned.proc)
        # Its own copy closed, or the relay would never see the harness exit.
        assert spawned.eof is True

    async def test_a_fenced_run_is_a_failed_result(self, spawned, tmp_path):
        spawned.fence = RunFenced("lost the build")

        result = await self._run(tmp_path, on_event=self._on_event)

        assert result["success"] is False
        assert result["error"] == "lost the build"

    async def test_a_pipe_a_child_holds_open_does_not_hang_the_run(
        self, spawned, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(python, "EVENTS_DRAIN_SECONDS", 0.01)
        spawned.relay_hangs = True
        (tmp_path / "harness-result.json").write_text('{"success": true}')

        result = await self._run(tmp_path, on_event=self._on_event)

        assert result == {"success": True}
        assert spawned.timeouts == [7.0, 0.01]

    async def test_no_result_file_reports_what_it_printed(self, spawned, tmp_path):
        result = await self._run(tmp_path)

        assert result["success"] is False
        assert result["error"] == "err"
        assert result["stderr"] == "err"
        assert result["stdout"] == "out"

    async def test_a_result_file_is_the_result(self, spawned, tmp_path):
        (tmp_path / "harness-result.json").write_text('{"success": true, "variables": {"x": 1}}')

        result = await self._run(tmp_path)

        assert result == {"success": True, "variables": {"x": 1}}

    async def test_a_result_file_that_is_a_link_is_not_followed(self, spawned, tmp_path):
        secret = tmp_path / "secret"
        secret.write_text('{"success": true}')
        (tmp_path / "harness-result.json").symlink_to(secret)

        with pytest.raises(UnsafeRunFile):
            await self._run(tmp_path)

    @pytest.mark.parametrize(
        ("outcome", "raised"),
        [(TimeoutError(), TimeoutError), (asyncio.CancelledError(), asyncio.CancelledError)],
    )
    async def test_an_interrupted_run_ends_the_tree_and_the_relay(
        self, spawned, tmp_path, outcome, raised
    ):
        spawned.outcome = outcome
        spawned.relay_hangs = True

        with pytest.raises(raised):
            await self._run(tmp_path, on_event=self._on_event)

        # The cancel is delivered on the loop's next pass.
        await asyncio.sleep(0)
        assert spawned.terminated == [spawned.proc]
        assert spawned.relay_cancelled is True
