"""An extracted cell path takes the session as a host and a context, never the executor."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import strata.notebook.cell_paths
from strata.notebook.cell_paths.context import PathContext
from strata.notebook.cell_paths.widget import run_widget_cell
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

    def test_a_replaced_method_still_holds(self, executor, monkeypatch):
        async def replaced(cell_id, source):
            return "replaced"

        monkeypatch.setattr(executor, "_compute_cell_provenance", replaced)

        assert executor._path_context().compute_provenance is replaced


class TestWidget:
    """The path on its own: a fake host, a fake context, a stubbed widget executor."""

    PROVENANCE = SimpleNamespace(
        provenance_hash="prov",
        source_hash="source",
        env_hash="env",
        table_reads=[{"table": "t"}],
        key_parts={"key": "parts"},
        reads=lambda mounts, tables: {"mounts": mounts, "tables": tables},
    )

    @pytest.fixture
    def path(self, monkeypatch):
        """Runs the path with the widget executor answering *answer*; ``calls`` holds the rest."""
        cell = SimpleNamespace(artifact_uris={"old": "uri"})
        calls = SimpleNamespace(recorded=[], use_cache=[], cell=cell)
        host = SimpleNamespace(
            notebook_state=SimpleNamespace(get_cell=lambda cell_id: cell),
            record_successful_execution_provenance=lambda *args, **kwargs: calls.recorded.append(
                (args, kwargs)
            ),
        )

        async def compute_provenance(cell_id, source):
            return self.PROVENANCE

        context = PathContext(
            compute_provenance=compute_provenance,
            moved=lambda cell_id: ({"data": 3}, {"file:///t": 4}),
        )

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
        assert path.calls.recorded == [
            (
                ("c1", "prov", "source", "env"),
                {
                    "table_reads": [{"table": "t"}],
                    "reads": {"mounts": {"data": 3}, "tables": {"file:///t": 4}},
                    "key_parts": {"key": "parts"},
                },
            )
        ]
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
