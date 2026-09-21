"""Getting a cell's display output out as a file.

An agent that runs a plotting cell gets back `content_type: image/png` and a
null preview. There is no text an image can be flattened into, so metadata
alone told it that *something* was drawn and nothing about what. The bytes
were in the artifact store the whole time with no way to ask for them.
"""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path

import httpx
import pytest

from strata.notebook.executor import CellExecutor
from strata.notebook.mcp_server import _save_cell_output
from strata.notebook.ops import LocalNotebookOps, NotebookOpsError, RemoteNotebookOps
from strata.notebook.parser import parse_notebook
from strata.notebook.scopes import (
    CLASSIFIED_TOOLS,
    NOTEBOOK_SCOPE_READ,
    required_scope_for_tool,
)
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

# A 1x1 blue PNG. Small enough to inline, real enough that a byte comparison
# means something.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

PLOT_CELL = f"""
class Figure:
    def _repr_png_(self):
        return {PNG!r}

Figure()
"""


def _notebook_with_a_plot(tmp_path: Path) -> tuple[Path, NotebookSession]:
    notebook_dir = create_notebook(tmp_path, "Plots", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "p")
    write_cell(notebook_dir, "p", PLOT_CELL)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    result = asyncio.run(CellExecutor(session).execute_cell("p", PLOT_CELL))
    assert result.success is True
    return notebook_dir, session


def test_the_view_says_what_the_output_is_and_how_big(tmp_path: Path):
    notebook_dir, _ = _notebook_with_a_plot(tmp_path)

    output = LocalNotebookOps(notebook_dir).get_cell("p").outputs[0]
    assert output.content_type == "image/png"
    assert output.preview is None  # an image has no text form; this is the point
    assert output.bytes == len(PNG)
    assert output.artifact_uri is not None
    assert output.artifact_uri.startswith("strata://artifact/")


def test_saving_the_output_writes_the_bytes_that_were_stored(tmp_path: Path):
    notebook_dir, _ = _notebook_with_a_plot(tmp_path)
    dest = tmp_path / "figure.png"

    saved = LocalNotebookOps(notebook_dir).save_output("p", dest)

    assert dest.read_bytes() == PNG
    assert saved.content_type == "image/png"
    assert saved.bytes == len(PNG)
    assert saved.index == 0  # -1 resolved against a single output
    assert saved.path == str(dest)


def test_saving_names_the_cell_and_the_index_it_cannot_find(tmp_path: Path):
    notebook_dir, _ = _notebook_with_a_plot(tmp_path)
    ops = LocalNotebookOps(notebook_dir)

    with pytest.raises(NotebookOpsError, match="no cell with id"):
        ops.save_output("ghost", tmp_path / "x.png")
    with pytest.raises(NotebookOpsError, match="no index 4"):
        ops.save_output("p", tmp_path / "x.png", index=4)


def test_a_cell_that_displayed_nothing_says_so(tmp_path: Path):
    notebook_dir = create_notebook(tmp_path, "Quiet", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "q")
    write_cell(notebook_dir, "q", "x = 1\n")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    asyncio.run(CellExecutor(session).execute_cell("q", "x = 1\n"))

    with pytest.raises(NotebookOpsError, match="no display output to save"):
        LocalNotebookOps(notebook_dir).save_output("q", tmp_path / "x.png")


def test_the_remote_backend_gets_the_same_bytes(tmp_path: Path):
    """The client writes the file; the server only hands over the blob."""
    notebook_dir, session = _notebook_with_a_plot(tmp_path)
    written: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        written["path"] = request.url.path
        cell = session.notebook_state.get_cell("p")
        blob = session.read_display_blob(cell.display_outputs[0])
        return httpx.Response(
            200,
            content=blob,
            headers={"content-type": "image/png", "X-Strata-Output-Index": "0"},
        )

    remote = RemoteNotebookOps(
        "http://test", "sess-1", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    dest = tmp_path / "remote.png"
    saved = remote.save_output("p", dest)

    assert written["path"] == "/v1/notebooks/sess-1/cells/p/outputs/-1/blob"
    assert dest.read_bytes() == PNG
    assert saved.content_type == "image/png"
    assert saved.index == 0


def test_the_mcp_tool_writes_inside_the_notebook(tmp_path: Path):
    """No caller-named path: the destination is the notebook's own runtime dir."""
    notebook_dir, session = _notebook_with_a_plot(tmp_path)

    class _Manager:
        def get_session(self, session_id: str):
            return session if session_id == "s" else None

        def list_sessions(self):
            return ["s"]

    saved = _save_cell_output(_Manager(), "s", "p")

    written = Path(saved["path"])
    assert written == notebook_dir / ".strata" / "outputs" / "p-0.png"
    assert written.read_bytes() == PNG


def test_the_route_serves_the_blob_as_itself(tmp_path: Path, monkeypatch):
    """What the remote backend talks to, driven for real rather than mocked."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from strata.notebook.routes import router

    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
    notebook_dir, _ = _notebook_with_a_plot(tmp_path)

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    sid = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)}).json()["session_id"]

    resp = client.get(f"/v1/notebooks/{sid}/cells/p/outputs/-1/blob")
    assert resp.status_code == 200
    assert resp.content == PNG
    assert resp.headers["content-type"] == "image/png"
    assert resp.headers["X-Strata-Output-Index"] == "0"

    missing = client.get(f"/v1/notebooks/{sid}/cells/p/outputs/7/blob")
    assert missing.status_code == 404
    assert "no index 7" in missing.json()["detail"]


def test_the_cli_writes_the_file_and_reports_it(tmp_path: Path, capsys):
    import json as _json

    from strata.cli import main

    notebook_dir, _ = _notebook_with_a_plot(tmp_path)
    dest = tmp_path / "cli.png"

    assert main(["cell", "output", str(notebook_dir), "p", "--out", str(dest)]) == 0
    payload = _json.loads(capsys.readouterr().out)

    assert dest.read_bytes() == PNG
    assert payload["content_type"] == "image/png"
    assert payload["bytes"] == len(PNG)


def test_the_cli_refuses_a_directory_that_is_not_there(tmp_path: Path, capsys):
    from strata.cli import main

    notebook_dir, _ = _notebook_with_a_plot(tmp_path)
    missing = tmp_path / "nope" / "cli.png"

    assert main(["cell", "output", str(notebook_dir), "p", "--out", str(missing)]) == 2
    assert "no such directory" in capsys.readouterr().err


def test_the_cli_says_so_when_the_target_is_not_writable(tmp_path: Path, capsys):
    """A parent that exists is not a path that can be written.

    `--out` naming an existing directory passed the parent check and then
    raised IsADirectoryError out of the command as a traceback.
    """
    from strata.cli import main

    notebook_dir, _ = _notebook_with_a_plot(tmp_path)
    a_directory = tmp_path / "already-a-dir"
    a_directory.mkdir()

    assert main(["cell", "output", str(notebook_dir), "p", "--out", str(a_directory)]) == 2
    assert "cannot write" in capsys.readouterr().err


MARKDOWN_CELL = """
class Note:
    def _repr_markdown_(self):
        return "# Heading"

Note()
"""


def test_local_and_remote_agree_on_a_text_content_type(tmp_path: Path, monkeypatch):
    """Starlette appends `; charset=utf-8` to text/* responses.

    The remote backend copied that header verbatim, so a markdown output was
    `text/markdown` locally and `text/markdown; charset=utf-8` remotely. The
    two backends are meant to be one view of the same notebook.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from strata.notebook.routes import router

    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
    notebook_dir = create_notebook(tmp_path, "Notes", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "m")
    write_cell(notebook_dir, "m", MARKDOWN_CELL)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    assert asyncio.run(CellExecutor(session).execute_cell("m", MARKDOWN_CELL)).success

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    sid = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)}).json()["session_id"]

    # Drive RemoteNotebookOps against the real route by routing its httpx
    # client through the TestClient's transport.
    remote = RemoteNotebookOps("http://testserver", sid, client=client)
    remote_saved = remote.save_output("m", tmp_path / "remote.md")
    local_saved = LocalNotebookOps(notebook_dir).save_output("m", tmp_path / "local.md")

    assert local_saved.content_type == "text/markdown"
    assert remote_saved.content_type == local_saved.content_type
    assert (tmp_path / "remote.md").read_bytes() == (tmp_path / "local.md").read_bytes()


def test_the_new_tool_is_classified(tmp_path: Path):
    assert "save_cell_output" in CLASSIFIED_TOOLS
    assert required_scope_for_tool("save_cell_output") == NOTEBOOK_SCOPE_READ
