"""An extracted cell path takes the session as a host and a context, never the executor."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from vestigo.provenance import derive_subkey

import strata.notebook.cell_paths
from strata.notebook.cell_paths import cell_tests
from strata.notebook.cell_paths.cell_tests import CellTestsContext, run_cell_tests
from strata.notebook.cell_paths.context import PathContext
from strata.notebook.cell_paths.module_export import write_module_export_outputs
from strata.notebook.cell_paths.prompt import run_prompt_cell
from strata.notebook.cell_paths.sql import run_sql_cell
from strata.notebook.cell_paths.widget import run_widget_cell
from strata.notebook.execution_result import CellExecutionResult
from strata.notebook.executor import CellExecutor
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook

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
