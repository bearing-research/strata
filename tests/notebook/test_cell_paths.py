"""An extracted cell path takes the session as a host and a context, never the executor."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import strata.notebook.cell_paths
from strata.notebook.cell_paths.context import PathContext
from strata.notebook.cell_paths.prompt import run_prompt_cell
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
    calls = SimpleNamespace(recorded=[], materialized=[], cell=cell)

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
    )
    context = PathContext(
        compute_provenance=compute_provenance,
        moved=lambda cell_id: ({"data": 3}, {"file:///t": 4}),
        materialize_upstreams_or_failure=materialize_upstreams_or_failure,
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
