"""An extracted cell path takes the session as a host and a context, never the executor."""

from __future__ import annotations

import ast
import asyncio
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from vestigo.provenance import derive_subkey

import strata.notebook.cell_paths
from strata.notebook import each
from strata.notebook.annotations import LoopAnnotation
from strata.notebook.cell_paths import cell_tests, fanout, loop, r
from strata.notebook.cell_paths.cell_tests import CellTestsContext, run_cell_tests
from strata.notebook.cell_paths.context import (
    PathContext,
    RemoteExecutionError,
    artifact_content_type,
)
from strata.notebook.cell_paths.fanout import FanoutContext, run_each_cell, run_fanout_cell
from strata.notebook.cell_paths.loop import LoopContext, run_loop_cell
from strata.notebook.cell_paths.module_export import write_module_export_outputs
from strata.notebook.cell_paths.prompt import run_prompt_cell
from strata.notebook.cell_paths.r import RContext, run_r_cell
from strata.notebook.cell_paths.sql import run_sql_cell
from strata.notebook.cell_paths.widget import run_widget_cell
from strata.notebook.execution_result import CellExecutionResult
from strata.notebook.executor import CellExecutor
from strata.notebook.harness_user import LocalExecutionRefused
from strata.notebook.models import MountMode, WorkerBackendType
from strata.notebook.mounts import MountWrite
from strata.notebook.parser import parse_notebook
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
        monkeypatch.setattr(loop, "_resolve_loop_seed", lambda *_args: (b"{}", "json/object"))
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
            )
        )
        manager = SimpleNamespace(
            artifact_store=store,
            load_artifact_data=lambda artifact_id, _version: artifact_id.encode(),
        )
        host = SimpleNamespace(
            notebook_state=SimpleNamespace(id="n", get_cell=cells.get),
            get_artifact_manager=lambda: manager,
            wired_variables=lambda _cell, upstream: {"state"} if upstream == "definer" else set(),
        )
        blob, content_type = loop._resolve_loop_seed(host, "c", self.LOOP)
        assert blob == b"nb_n_cell_definer_var_state"
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
        calls = SimpleNamespace(recorded=[], applied=[], instances=[], staged=[], discarded=[])
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
            "data": SimpleNamespace(local_path=tmp_path, file_versions=versions),
            "out": SimpleNamespace(local_path=tmp_path, file_versions=None),
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

    @pytest.mark.parametrize(
        ("uri", "by_content"), [("s3://bucket/data", True), ("file:///data", False)]
    )
    async def test_a_listing_reads_only_the_glob_and_keys_a_remote_mount_on_content(
        self, tmp_path, monkeypatch, uri, by_content
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
            "by_content": by_content,
            "versions": {"a.csv": "v1"},
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
            host, context, "c", "src", [each.EachItem("a", "digest")], self._base(), ["out"], []
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
                host, context, "c", "src", [each.EachItem("a", "digest")], self._base(), ["out"], []
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
                display_outputs=result.get("display_outputs", []),
                execution_method=method,
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

        assert result.error == "R cell execution timed out after 7.0s"
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
