"""Unit tests for the TUI's pure render helpers + client URL/error handling.

These don't launch Textual — they exercise the standalone functions the app
renders through, so they stay fast and headless.
"""

from __future__ import annotations

import httpx
import pytest

from strata.notebook.tui.app import _first_line, _glyph, _render_outputs
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
