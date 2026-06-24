"""Unit tests for the TUI's pure render helpers + client URL/error handling.

These don't launch Textual — they exercise the standalone functions the app
renders through, so they stay fast and headless.
"""

from __future__ import annotations

import httpx
import pytest

from strata.notebook.tui.app import (
    _first_line,
    _glyph,
    _render_outputs,
    _render_table,
    _single_markdown,
    _single_table,
)
from strata.notebook.tui.client import TuiClient, TuiClientError, _json_or_error
from strata.notebook.tui.viewmodel import CellView


def test_glyph_known_and_unknown():
    assert _glyph("running") == "▶"
    assert _glyph("ready") == "✓"
    assert _glyph("mystery") == "?"


def test_first_line_skips_blank_lines():
    assert _first_line("\n\n  x = 1\ny = 2") == "x = 1"
    assert _first_line("   \n  ") == "(empty)"


def test_render_outputs_error_takes_precedence():
    cell = CellView(id="a", error="Traceback…", display_outputs=[{"preview": "ignored"}])
    assert _render_outputs(cell).startswith("[error]")
    assert "Traceback" in _render_outputs(cell)


def test_render_outputs_markdown_image_and_preview():
    cell = CellView(
        id="a",
        display_outputs=[
            {"content_type": "text/markdown", "markdown_text": "# Title"},
            {"content_type": "image/png", "inline_data_url": "data:…"},
            {"content_type": "json/object", "preview": {"k": 1}},
        ],
    )
    rendered = _render_outputs(cell)
    assert "# Title" in rendered
    assert "open in the web UI" in rendered  # image degraded
    assert "{'k': 1}" in rendered


def test_render_outputs_live_outputs_and_stream():
    cell = CellView(
        id="a",
        stream_text="streamed reply",
        outputs=[{"name": "df", "content_type": "arrow/ipc", "preview": "<table>"}],
    )
    rendered = _render_outputs(cell)
    assert "streamed reply" in rendered
    assert "df: arrow/ipc = <table>" in rendered


def test_render_outputs_empty():
    assert _render_outputs(CellView(id="a")) == "(no output)"


def test_single_markdown_detects_pure_markdown_output():
    cell = CellView(
        id="a",
        display_outputs=[{"content_type": "text/markdown", "markdown_text": "# Title\n- a\n- b"}],
    )
    assert _single_markdown(cell) == "# Title\n- a\n- b"


def _table_output(rows_total=100):
    return {
        "content_type": "arrow/ipc",
        "columns": ["id", "name"],
        "rows": rows_total,
        "preview": [[1, "alice"], [2, "bob"]],
    }


def test_single_table_detects_one_tabular_output():
    cell = CellView(id="a", outputs=[_table_output()])
    result = _single_table(cell)
    assert result is not None
    columns, preview, total = result
    assert columns == ["id", "name"]
    assert preview == [[1, "alice"], [2, "bob"]]
    assert total == 100


def test_single_table_none_when_not_a_single_table():
    # Two outputs → not a single table.
    assert _single_table(CellView(id="a", outputs=[_table_output(), _table_output()])) is None
    # Non-tabular output.
    assert _single_table(CellView(id="a", outputs=[{"content_type": "json/object"}])) is None
    # Error/stream present → text path.
    assert _single_table(CellView(id="a", outputs=[_table_output()], error="x")) is None


def test_render_table_includes_columns_values_and_truncation_caption():
    table = _render_table(["id", "name"], [[1, "alice"], [2, "bob"]], total=100)
    text = " ".join(_render_to_text(table).split())  # normalize wrapped whitespace
    assert "id" in text and "name" in text
    assert "alice" in text and "bob" in text
    assert "showing 2 of 100 rows" in text


def _render_to_text(renderable) -> str:
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    Console(file=buf, width=120).print(renderable)
    return buf.getvalue()


def test_single_markdown_none_when_mixed_or_nonmarkdown():
    # Two outputs → not a single markdown block.
    two = CellView(
        id="a",
        display_outputs=[
            {"content_type": "text/markdown", "markdown_text": "# A"},
            {"content_type": "image/png"},
        ],
    )
    assert _single_markdown(two) is None
    # Non-markdown single output.
    json_cell = CellView(id="a", display_outputs=[{"content_type": "json/object"}])
    assert _single_markdown(json_cell) is None
    # Live outputs / error present → use the text path.
    md = {"content_type": "text/markdown", "markdown_text": "# A"}
    assert _single_markdown(CellView(id="a", display_outputs=[md], error="boom")) is None
    assert _single_markdown(CellView(id="a", display_outputs=[md], outputs=[{"name": "x"}])) is None


def test_ws_url_swaps_scheme():
    assert (
        TuiClient("http://localhost:8765").ws_url("S1") == "ws://localhost:8765/v1/notebooks/ws/S1"
    )
    assert TuiClient("https://x.io").ws_url("S2") == "wss://x.io/v1/notebooks/ws/S2"


def test_json_or_error_surfaces_detail():
    resp = httpx.Response(404, json={"detail": "Notebook not found"})
    with pytest.raises(TuiClientError) as exc:
        _json_or_error(resp)
    assert "404" in str(exc.value)
    assert "Notebook not found" in str(exc.value)


def test_json_or_error_returns_body_on_success():
    resp = httpx.Response(200, json={"session_id": "abc", "name": "NB"})
    assert _json_or_error(resp)["session_id"] == "abc"
