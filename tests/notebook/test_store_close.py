"""A notebook's stores are closed by whatever opened them, and only by it.

On SQLite a store may keep its database open between calls; until it is closed,
Windows refuses to delete or move the notebook.
"""

from __future__ import annotations

import pytest
from vestigo.sql_backend import SqliteDialect

from strata.notebook import session as session_module
from strata.notebook.artifact_integration import NotebookArtifactManager
from strata.notebook.export import _hydrate_output, _StoredOutputs
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
    closes = []
    own = NotebookArtifactManager("nb", tmp_path / "own")
    close = own.artifact_store.dialect.close

    def recorded():
        closes.append("own")
        close()

    monkeypatch.setattr(own.artifact_store.dialect, "close", recorded)
    own.close()
    assert closes == ["own"]

    shared = SqliteDialect(tmp_path / "shared.sqlite")
    try:
        monkeypatch.setattr(shared, "close", lambda: closes.append("shared"))
        NotebookArtifactManager("nb", tmp_path / "theirs", dialect=shared).close()
        assert closes == ["own"]
    finally:
        monkeypatch.undo()
        shared.close()


def test_closing_a_session_closes_its_stores(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)
    sm = SessionManager()
    session = sm.open_notebook(create_notebook(tmp_path, "nb"))
    sm.close_session(session.id)
    assert closed == [session.artifact_manager]


def test_a_session_that_fails_to_open_closes_its_stores(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)

    def broken(self, *args, **kwargs):
        raise RuntimeError("staleness failed")

    monkeypatch.setattr(session_module.NotebookSession, "compute_staleness", broken)
    with pytest.raises(RuntimeError, match="staleness failed"):
        SessionManager().open_notebook(create_notebook(tmp_path, "nb"))
    assert len(closed) == 1


def test_local_ops_close_the_session_they_opened_and_not_one_they_wrap(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)
    notebook_dir = create_notebook(tmp_path, "nb")
    with LocalNotebookOps(notebook_dir) as ops:
        ops.add_cell("x = 1", language="python")
    assert closed == [ops._session.artifact_manager]

    sm = SessionManager()
    session = sm.open_notebook(notebook_dir)
    LocalNotebookOps.from_session(session).close()
    assert closed == [ops._session.artifact_manager]
    sm.close_session(session.id)


@pytest.mark.asyncio
async def test_local_ops_aclose_closes_the_stores_too(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)
    ops = LocalNotebookOps(create_notebook(tmp_path, "nb"))
    await ops.aclose()
    assert closed == [ops._session.artifact_manager]


def test_an_export_opens_the_store_once_and_closes_it(tmp_path, monkeypatch):
    closed = _count_closes(monkeypatch)
    opened = []
    init = NotebookArtifactManager.__init__

    def counted(self, *args, **kwargs):
        opened.append(self)
        init(self, *args, **kwargs)

    monkeypatch.setattr(NotebookArtifactManager, "__init__", counted)
    monkeypatch.setattr(
        NotebookArtifactManager,
        "load_artifact_data",
        lambda self, artifact_id, version: b"# shown",
    )
    stored = _StoredOutputs(tmp_path, "nb")
    for version in (1, 2):
        output = CellOutput(
            content_type="text/markdown", artifact_uri=f"strata://artifact/a@v={version}"
        )
        assert _hydrate_output(output, stored=stored).markdown_text == "# shown"
    stored.close()
    assert len(opened) == 1
    assert closed == opened
