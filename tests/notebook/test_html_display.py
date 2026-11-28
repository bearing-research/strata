"""A value that shows itself as HTML (plotly, altair, a styled table) is shown as that page.

The page runs its own script, so the notebook serves it sandboxed, from a route that hands
out only HTML display outputs of the cell named.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from strata.notebook import serializer as ser
from strata.notebook.executor import CellExecutor
from strata.notebook.export import ExportFormat, ExportOptions, export_notebook
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

PAGE = "<div id='chart'>bars</div><script>document.title = 'drawn'</script>"


class _Html:
    def _repr_html_(self):
        return PAGE


class _Bundle:
    def _repr_mimebundle_(self, include=None, exclude=None):
        return {"text/html": "<p>from the bundle</p>", "text/plain": "Bundle"}

    def _repr_html_(self):
        return "<p>not asked</p>"


class _BundleWithMetadata:
    def _repr_mimebundle_(self, include=None, exclude=None):
        return {"text/html": "<p>with metadata</p>"}, {"text/html": {"isolated": True}}


class _EmptyBundle:
    """plotly outside Jupyter: an empty bundle, and the page from ``_repr_html_``."""

    def _repr_mimebundle_(self, include=None, exclude=None):
        return {}

    def _repr_html_(self):
        return "<p>plotly</p>"


class _NoHtml:
    def _repr_mimebundle_(self, include=None, exclude=None):
        return {"application/json": {"a": 1}}


class _MarkdownAndHtml:
    def _repr_markdown_(self):
        return "**md**"

    def _repr_html_(self):
        return "<b>html</b>"


def _display(value, tmp_path: Path) -> dict:
    return ser.serialize_display_value(value, tmp_path, 0)


def test_a_value_with_an_html_repr_is_an_html_display(tmp_path):
    payload = _display(_Html(), tmp_path)

    assert payload["content_type"] == "text/html"
    assert (tmp_path / payload["file"]).read_text(encoding="utf-8") == PAGE
    # What an agent or the TUI reads, as for any object.
    assert payload["preview"] == "<_Html object>"


@pytest.mark.parametrize(
    ("value", "html"),
    [
        (_Bundle(), "<p>from the bundle</p>"),
        (_BundleWithMetadata(), "<p>with metadata</p>"),
        (_EmptyBundle(), "<p>plotly</p>"),
    ],
)
def test_the_bundle_comes_first_then_the_html_repr(tmp_path, value, html):
    payload = _display(value, tmp_path)

    assert payload["content_type"] == "text/html"
    assert (tmp_path / payload["file"]).read_text(encoding="utf-8") == html


def test_a_bundle_with_no_html_is_shown_as_before(tmp_path):
    payload = _display(_NoHtml(), tmp_path)

    assert payload["content_type"] == "pickle/object"
    assert payload["preview"] == "<_NoHtml object>"


def test_markdown_still_wins_over_html(tmp_path):
    assert _display(_MarkdownAndHtml(), tmp_path)["content_type"] == "text/markdown"


def test_a_variable_with_an_html_repr_is_stored_as_the_object(tmp_path):
    payload = ser.serialize_value(_Html(), tmp_path, "chart")

    assert payload["content_type"] == "pickle/object"


def test_a_dataframe_stays_a_table(tmp_path):
    pd = pytest.importorskip("pandas")

    assert _display(pd.DataFrame({"a": [1]}), tmp_path)["content_type"] == "arrow/ipc"


# --- In a notebook ------------------------------------------------------------------------

CHART_CELL = f"""
class Chart:
    def _repr_html_(self):
        return {PAGE!r}

Chart()
"""


def _notebook(tmp_path: Path, cells: dict[str, str]) -> tuple[Path, NotebookSession]:
    notebook_dir = create_notebook(tmp_path, "Charts", initialize_environment=False)
    for cell_id, source in cells.items():
        add_cell_to_notebook(notebook_dir, cell_id)
        write_cell(notebook_dir, cell_id, source)
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)
    session.refresh_environment_runtime()
    for cell_id, source in cells.items():
        result = asyncio.run(CellExecutor(session).execute_cell(cell_id, source))
        assert result.success is True, result.error
    return notebook_dir, session


@pytest.fixture
def served(tmp_path: Path, monkeypatch):
    """A notebook with a chart cell, a markdown cell and a variable, open over the router."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from strata.notebook.routes import get_session_manager, router

    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
    notebook_dir, _ = _notebook(
        tmp_path,
        {"c": CHART_CELL, "m": "Markdown('**hi**')\n", "v": "x = 1\n", "r": "print(x)\n"},
    )
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    sid = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)}).json()["session_id"]
    session = get_session_manager().get_session(sid)
    return client, sid, session, notebook_dir


def _uri(session: NotebookSession, cell_id: str) -> str:
    uri = session.notebook_state.get_cell(cell_id).display_outputs[0].artifact_uri
    assert uri is not None
    return uri


def test_the_cell_shows_the_page_as_a_stored_html_output(served):
    _, _, session, _ = served

    output = session.notebook_state.get_cell("c").display_outputs[0]

    assert output.content_type == "text/html"
    assert session.read_display_blob(output).decode() == PAGE


def test_the_route_serves_the_page_sandboxed(served):
    client, sid, session, _ = served

    resp = client.get(
        f"/v1/notebooks/{sid}/cells/c/html", params={"artifact_uri": _uri(session, "c")}
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    # Script runs, but in an opaque origin: none of the notebook's page, cookies or storage.
    assert resp.headers["content-security-policy"] == (
        "sandbox allow-scripts allow-popups allow-downloads"
    )
    assert resp.headers["referrer-policy"] == "no-referrer"
    # Standards mode, as Jupyter shows it: plotly takes its own height, not the frame's.
    assert resp.text.startswith(f"<!doctype html>\n{PAGE}")
    assert "strataOutputHeight" in resp.text


def test_the_route_serves_only_the_named_cells_html_outputs(served):
    client, sid, session, _ = served
    chart = _uri(session, "c")
    markdown = _uri(session, "m")
    manager = session.get_artifact_manager()
    variable = manager.artifact_store.get_latest_version(manager.cell_artifact_id("v", "x"))
    assert variable is not None

    for cell_id, uri in [
        ("m", chart),  # another cell's output
        ("m", markdown),  # a display, but not a page
        ("v", f"strata://artifact/{variable.id}@v={variable.version}"),  # a variable
        ("c", "not-a-uri"),
    ]:
        resp = client.get(f"/v1/notebooks/{sid}/cells/{cell_id}/html", params={"artifact_uri": uri})
        assert resp.status_code == 404, (cell_id, uri)


def test_the_html_export_frames_the_page_and_markdown_says_where_it_is(served):
    _, _, _, notebook_dir = served

    html = export_notebook(notebook_dir, ExportOptions(output_format=ExportFormat.HTML))
    markdown = export_notebook(notebook_dir, ExportOptions(output_format=ExportFormat.MARKDOWN))

    assert '<iframe class="html-output" sandbox="allow-scripts allow-popups"' in html
    # Escaped into the attribute: the page is never part of the export's own document.
    assert 'srcdoc="&lt;div id=&#x27;chart&#x27;&gt;bars&lt;/div&gt;' in html
    assert "<div id='chart'>" not in html
    assert "HTML output: export as HTML, or open the notebook, to see it." in markdown


def test_a_whole_document_keeps_its_own_doctype(tmp_path: Path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from strata.notebook.routes import get_session_manager, router

    document = "<!DOCTYPE html><html><body>whole</body></html>"
    source = f"class Page:\n    def _repr_html_(self):\n        return {document!r}\n\nPage()\n"
    monkeypatch.setattr("strata.notebook.session._uv_sync", lambda path, **kw: True)
    notebook_dir, _ = _notebook(tmp_path, {"d": source})
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    sid = client.post("/v1/notebooks/open", json={"path": str(notebook_dir)}).json()["session_id"]
    session = get_session_manager().get_session(sid)

    resp = client.get(
        f"/v1/notebooks/{sid}/cells/d/html", params={"artifact_uri": _uri(session, "d")}
    )

    assert resp.text.startswith(document)
