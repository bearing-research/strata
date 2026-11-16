"""A notebook's stores are closed by whatever opened them, and only by it.

On SQLite a store may keep its database open between calls; until it is closed,
Windows refuses to delete or move the notebook.
"""

from __future__ import annotations

from vestigo.sql_backend import SqliteDialect

from strata.notebook import artifact_integration
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.export import _hydrate_output
from strata.notebook.models import CellOutput
from strata.notebook.ops import LocalNotebookOps
from strata.notebook.session import SessionManager
from strata.notebook.writer import create_notebook


def _count_closes(monkeypatch) -> list[NotebookArtifactManager]:
    closed: list[NotebookArtifactManager] = []
    close = NotebookArtifactManager.close

    def recorded(self):
        closed.append(self)
        close(self)

    monkeypatch.setattr(NotebookArtifactManager, "close", recorded)
    return closed


def test_a_manager_closes_the_dialect_it_made_and_not_one_passed_in(tmp_path, monkeypatch):
    own = NotebookArtifactManager("nb", tmp_path / "own")
    closes = []
    monkeypatch.setattr(own.artifact_store.dialect, "close", lambda: closes.append("own"))
    own.close()
    assert closes == ["own"]

    shared = SqliteDialect(tmp_path / "shared.sqlite")
    monkeypatch.setattr(shared, "close", lambda: closes.append("shared"))
    NotebookArtifactManager("nb", tmp_path / "theirs", dialect=shared).close()
    assert closes == ["own"]


def test_closing_a_session_closes_its_stores(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)
    sm = SessionManager()
    session = sm.open_notebook(create_notebook(tmp_path, "nb"))
    sm.close_session(session.id)
    assert closed == [session.artifact_manager]


def test_local_ops_close_the_session_they_opened_and_not_one_they_wrap(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)
    notebook_dir = create_notebook(tmp_path, "nb")
    with LocalNotebookOps(notebook_dir) as ops:
        ops.add_cell("x = 1", language="python")
    assert closed == [ops._session.artifact_manager]

    session = SessionManager().open_notebook(notebook_dir)
    LocalNotebookOps.from_session(session).close()
    assert closed == [ops._session.artifact_manager]


def test_export_closes_the_store_it_reads_an_output_from(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)
    monkeypatch.setattr(
        artifact_integration.NotebookArtifactManager,
        "load_artifact_data",
        lambda self, artifact_id, version: b"# shown",
    )
    output = CellOutput(content_type="text/markdown", artifact_uri="strata://artifact/a@v=1")
    hydrated = _hydrate_output(output, notebook_dir=tmp_path, notebook_id="nb")
    assert hydrated.markdown_text == "# shown"
    assert len(closed) == 1
