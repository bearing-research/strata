"""The TUI must show cell content verbatim, not as Rich console markup.

Every panel in the TUI displays arbitrary user content: stdout, tracebacks,
cell source, artifact values. Handing those to a ``Static`` or a table cell as
a bare ``str`` makes Rich parse ``[...]`` as markup, which fails two ways:

* silently — ``print(data[key])`` in a traceback renders as ``print(data)``,
  because ``key`` is read as a style name. The console panel exists to debug
  failures, and it was dropping the subscript that caused them.
* loudly — ``counts[/tmp/x]`` looks like a closing tag with no opening tag and
  raises ``MarkupError`` out of the render, taking the panel with it.

Both are reachable from ordinary notebook output, so the panels are checked
here with content that triggers each.
"""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.coordinate import Coordinate
from textual.widgets import DataTable, Static

from strata.notebook.tui.app import NotebookTUI, _TableView
from strata.notebook.tui.client import TuiClient
from strata.notebook.tui.viewmodel import CellView

# A traceback line whose subscript is silently eaten, and a path that looks
# like a stray closing tag and crashes the render.
SUBSCRIPT = '  File "load.py", line 3\n    print(data[key])'
BRACKET_PATH = "wrote counts[/tmp/out]"


def _plain(widget: Static) -> str:
    """The text a Static actually renders (markup already applied)."""
    visual = widget.visual
    return getattr(visual, "plain", str(visual))


async def _app(monkeypatch):
    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)
    return NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")


def _select(app: NotebookTUI, cell: CellView) -> None:
    app.vm.cells[cell.id] = cell
    app.vm.cell_order = [cell.id]
    app._selected = cell.id
    app._show_detail(cell.id)


@pytest.mark.asyncio
async def test_console_keeps_the_subscript_in_a_traceback(monkeypatch):
    app = await _app(monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        _select(app, CellView(id="c1", console=SUBSCRIPT))
        await pilot.pause()
        assert "data[key]" in _plain(app.query_one("#console-body", Static))


@pytest.mark.asyncio
async def test_console_survives_a_bracketed_path(monkeypatch):
    app = await _app(monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        _select(app, CellView(id="c1", console=BRACKET_PATH))
        await pilot.pause()
        assert "counts[/tmp/out]" in _plain(app.query_one("#console-body", Static))


@pytest.mark.asyncio
async def test_output_pane_keeps_a_bracketed_error(monkeypatch):
    app = await _app(monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        _select(app, CellView(id="c1", error="KeyError: rows[/idx]"))
        await pilot.pause()
        assert "rows[/idx]" in _plain(app.query_one("#output", Static))


@pytest.mark.asyncio
async def test_agent_feed_keeps_bracketed_text(monkeypatch):
    app = await _app(monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        app.vm.agent_feed = ["reading [/etc/hosts]"]
        app._render_agent()
        await pilot.pause()
        assert "[/etc/hosts]" in _plain(app.query_one("#agent", Static))


# The two table tests below assert a renderable rather than the rendered text.
# ``DataTable`` formats a cell only when its row scrolls into view, and the
# headless pilot lays the table out two lines tall, so the rows never render and
# there is nothing to read back. What the app controls is what it hands the
# widget: a bare ``str`` cell is put through ``Text.from_markup``, anything
# already renderable is not. Asserting the value carries its own text is
# therefore the boundary this code is responsible for.


@pytest.mark.asyncio
async def test_cell_list_hands_the_source_preview_over_as_text(monkeypatch):
    app = await _app(monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        app.vm.cells["c1"] = CellView(id="c1", source="total = df[col].sum()", status="ready")
        app.vm.cell_order = ["c1"]
        app._rebuild_cells()
        await pilot.pause()
        label = app.query_one("#cells", DataTable).get_cell_at(Coordinate(0, 1))
        assert isinstance(label, Text)
        assert "df[col]" in label.plain


@pytest.mark.asyncio
async def test_data_viewer_hands_values_over_as_text(monkeypatch):
    app = await _app(monkeypatch)
    async with app.run_test(size=(100, 40)) as pilot:
        _select(app, CellView(id="c1"))
        app._table_view = _TableView(cell_id="c1", artifact_uri="strata://artifact/a@v=1")
        app._render_table_page(
            {
                "pageable": True,
                "columns": ["path"],
                "rows": [["/data[raw]/part-0"]],
                "total": 1,
                "offset": 0,
            }
        )
        await pilot.pause()
        value = app.query_one("#output-table", DataTable).get_cell_at(Coordinate(0, 0))
        assert isinstance(value, Text)
        assert "/data[raw]/part-0" in value.plain
