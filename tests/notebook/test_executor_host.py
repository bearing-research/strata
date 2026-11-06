"""``CellExecutor`` needs the session only through ``ExecutorHost``.

``ty check`` shows ``NotebookSession`` satisfies the protocol and that the typed paths use
only what it lists. The traces run with every executor's session behind a recording host
(``tests/contract/cellgraph/conftest.py``). Here: what no trace reaches, read from the code,
and a plain cell run on a fake with no ``NotebookSession`` behind it.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import pytest

import strata.notebook.executor
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.dag import NotebookDag
from strata.notebook.executor import CellExecutor
from strata.notebook.graph import analyze_notebook
from strata.notebook.parser import parse_notebook
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell
from tests.contract.cellgraph.recording_host import RecordingHost, host_members

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="POSIX cell execution")

NOTEBOOK = Path(strata.notebook.executor.__file__).parent
# The executor, and what it hands the session to.
USERS = (
    "executor.py",
    "keys.py",
    "prompt_executor.py",
    "widget_executor.py",
    "llm/config.py",
    "sql/cell_executor.py",
    "sql/lake.py",
)


def _session_uses(source: str) -> set[str]:
    """Each ``self.session.x``, ``session.x`` and ``host.x``, and each ``getattr`` of them."""

    def is_session(node: ast.expr) -> bool:
        if isinstance(node, ast.Name):
            return node.id in {"session", "host"}
        return isinstance(node, ast.Attribute) and node.attr == "session"

    uses = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute) and is_session(node.value):
            uses.add(node.attr)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"getattr", "hasattr", "setattr"}
            and is_session(node.args[0])
            and isinstance(node.args[1], ast.Constant)
        ):
            uses.add(node.args[1].value)
    return uses


def test_the_protocol_lists_exactly_what_the_code_uses():
    files = [NOTEBOOK / name for name in USERS] + list((NOTEBOOK / "cell_paths").glob("*.py"))
    uses = set().union(*(_session_uses(path.read_text()) for path in files))

    assert uses == host_members()


def test_each_way_of_reaching_the_session_counts_as_a_use():
    source = "getattr(self.session, 'x', None)\nself.session.y\nsession.z\nhost.w\n"
    assert _session_uses(source) == {"x", "y", "z", "w"}


class TestRecordingHost:
    def test_a_listed_read_passes_through_unremembered(self):
        strays: set[str] = set()
        session = type("Session", (), {"path": "here"})()

        assert RecordingHost(session, {"path"}, strays).path == "here"
        assert strays == set()

    def test_any_other_read_passes_through_remembered(self):
        strays: set[str] = set()
        session = type("Session", (), {"other": 1})()

        assert RecordingHost(session, {"path"}, strays).other == 1
        assert strays == {"read other"}

    def test_a_write_is_remembered_even_of_a_listed_name(self):
        strays: set[str] = set()
        session = type("Session", (), {"path": "here"})()

        RecordingHost(session, {"path"}, strays).path = "there"

        assert session.path == "there"
        assert strays == {"set path"}


class _FakeHost:
    """What a plain Python cell needs, with no ``NotebookSession`` behind it."""

    def __init__(self, path: Path) -> None:
        self.id = "fake"
        self.path = path
        self.notebook_state = parse_notebook(path)
        self.analysis = analyze_notebook(self.notebook_state)
        self.dag = NotebookDag.of(self.analysis.graph)
        self.venv_python = Path(sys.executable)
        self.r_warm_pool = None
        self.opened_by = None
        self.manager = NotebookArtifactManager(
            notebook_id=self.notebook_state.id, artifact_dir=path / ".strata" / "artifacts"
        )
        self.provenance: dict[str, str] = {}
        for cell in self.notebook_state.cells:
            analyzed = self.analysis.cells[cell.id]
            cell.defines = list(analyzed.defines)
            cell.references = list(analyzed.references)
            cell.upstream_ids = self.dag.cell_upstream.get(cell.id, [])
            cell.downstream_ids = self.dag.cell_downstream.get(cell.id, [])

    def get_artifact_manager(self) -> NotebookArtifactManager:
        return self.manager

    def lake_config(self) -> Any:
        from strata.config import StrataConfig

        return StrataConfig.load()

    def parse_artifact_uri(self, artifact_uri: str) -> tuple[str, int]:
        artifact_id, version = artifact_uri.removeprefix("strata://artifact/").rsplit("@v=", 1)
        return artifact_id, int(version)

    def environment_attestation_error(self) -> str | None:
        return None

    def cell_env_hash(self, cell: Any) -> str:
        return "fake-env"

    def wired_variables(self, cell_id: str, upstream_id: str) -> set[str]:
        return set()

    def collect_input_refs(self, cell_id: str, *, variant: str | None = None) -> dict[str, str]:
        return {}

    def resolve_cached_console(
        self, cell_id: str, provenance_hash: str, *, variant: str | None = None
    ) -> None:
        return None

    def resolve_cached_display_outputs(
        self, cell_id, provenance_hash, current_outputs, variant=None
    ):
        return current_outputs

    def apply_execution_result_metadata(self, cell_id: str, result: Any) -> None:
        return None

    def persist_display_outputs(self, cell_id: str, display_outputs: Any) -> None:
        return None

    def record_cell_inputs(self, cell_id: str, machine: str, weights: list[Any]) -> bool:
        return False

    def record_successful_execution_provenance(
        self, cell_id, provenance_hash, source_hash, env_hash, **kwargs
    ) -> None:
        self.provenance[cell_id] = provenance_hash


async def test_a_plain_python_cell_runs_on_a_fake_host(tmp_path):
    nb = create_notebook(tmp_path / "nb", "Fake host")
    add_cell_to_notebook(nb, "c1", None)
    write_cell(nb, "c1", "x = 40 + 2\nprint('hello')\n")
    add_cell_to_notebook(nb, "c2", "c1")
    write_cell(nb, "c2", "y = x\n")
    host = _FakeHost(nb)

    result = await CellExecutor(host).execute_cell("c1", "x = 40 + 2\nprint('hello')\n")

    assert result.success, result.error
    assert "hello" in result.stdout
    latest = host.manager.artifact_store.get_latest_version(
        host.manager.cell_artifact_id("c1", "x")
    )
    assert latest is not None
    assert latest.provenance_hash is not None
    assert "c1" in host.provenance
