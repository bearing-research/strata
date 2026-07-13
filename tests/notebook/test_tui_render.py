"""Unit tests for the TUI's pure render helpers + client URL/error handling.

These don't launch Textual — they exercise the standalone functions the app
renders through, so they stay fast and headless.
"""

from __future__ import annotations

import base64
import io

import httpx
import pytest
from PIL import Image as PILImage
from rich.syntax import Syntax
from textual_image.renderable import Image as TerminalImage

from strata.notebook.tui.app import (
    _decode_data_url,
    _glyph,
    _image_renderable,
    _render_outputs,
    _render_table,
    _single_image,
    _single_markdown,
    _single_table,
    _single_table_uri,
    _source_preview,
    _source_renderable,
    _time_str,
)
from strata.notebook.tui.client import TuiClient, TuiClientError, _json_or_error
from strata.notebook.tui.viewmodel import CellView


def _png_data_url(color: str = "red", size: tuple[int, int] = (4, 4)) -> str:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def test_glyph_known_and_unknown():
    assert _glyph("running") == "▶"
    assert _glyph("ready") == "✓"
    assert _glyph("mystery") == "?"


def test_source_preview_skips_annotations_and_blanks():
    # Leading #-comment/annotation block is skipped → shows the first code line,
    # not a redundant "# @name load".
    assert _source_preview("# @name load\nimport pyarrow as pa\nx = 1") == "import pyarrow as pa"
    assert _source_preview("\n\n  x = 1\ny = 2") == "x = 1"
    assert _source_preview("   \n  ") == "(empty)"
    # Comment-only cell falls back to the comment rather than "(empty)".
    assert _source_preview("# just a note") == "# just a note"


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


def test_single_markdown_renders_markdown_language_cell_source():
    # A markdown-language cell produces no outputs — its source IS the rendered
    # Output (the Source tab shows the raw text).
    cell = CellView(id="a", language="markdown", source="# Notes\n\nsome **bold** text")
    assert _single_markdown(cell) == "# Notes\n\nsome **bold** text"
    # Empty markdown cell → None (falls through to the "(no output)" text path).
    assert _single_markdown(CellView(id="a", language="markdown", source="")) is None
    # A markdown cell that somehow carries a display output isn't treated as source.
    with_output = CellView(
        id="a",
        language="markdown",
        source="# raw",
        display_outputs=[
            {"content_type": "image/png", "inline_data_url": "data:image/png;base64,x"}
        ],
    )
    assert _single_markdown(with_output) is None


def _table_output(rows_total=100):
    return {
        "content_type": "arrow/ipc",
        "columns": ["id", "name"],
        "rows": rows_total,
        "preview": [[1, "alice"], [2, "bob"]],
    }


def test_source_renderable_highlights_by_language():
    py = _source_renderable(CellView(id="a", source="x = 1", language="python"))
    assert isinstance(py, Syntax)
    assert py.lexer is not None and py.lexer.name == "Python"

    sql = _source_renderable(CellView(id="a", source="select 1", language="sql"))
    assert isinstance(sql, Syntax) and sql.lexer.name == "SQL"

    # Unknown language defaults to Python; empty source is a plain placeholder.
    unknown = _source_renderable(CellView(id="a", source="y", language="mystery"))
    assert isinstance(unknown, Syntax) and unknown.lexer.name == "Python"
    assert _source_renderable(CellView(id="a", source="")) == "(empty)"


def test_decode_data_url():
    raw = _decode_data_url(_png_data_url())
    assert raw is not None and raw[:8] == b"\x89PNG\r\n\x1a\n"  # PNG magic
    assert _decode_data_url("https://example.com/x.png") is None  # not a data URL
    assert _decode_data_url("data:image/png;base64,abc") is None  # invalid base64 length


def _image_cell(url: str, **kw) -> CellView:
    out = {"content_type": "image/png", "inline_data_url": url}
    return CellView(id="a", display_outputs=[out], **kw)


def test_single_image_detects_one_image_output():
    url = _png_data_url()
    assert _single_image(_image_cell(url)) == url
    # Mixed outputs / non-image / error → None.
    mixed = CellView(
        id="a",
        display_outputs=[
            {"content_type": "image/png", "inline_data_url": url},
            {"content_type": "json/object"},
        ],
    )
    assert _single_image(mixed) is None
    nonimg = CellView(id="a", display_outputs=[{"content_type": "text/markdown"}])
    assert _single_image(nonimg) is None
    assert _single_image(_image_cell(url, error="x")) is None


def test_image_renderable_builds_and_degrades():
    assert isinstance(_image_renderable(_image_cell(_png_data_url())), TerminalImage)
    # Non-image bytes behind a valid data URL → None (not a decodable image).
    bad_url = "data:image/png;base64," + base64.b64encode(b"not a png").decode()
    assert _image_renderable(_image_cell(bad_url)) is None


def test_time_str_formats_duration_and_cache():
    assert _time_str(CellView(id="a")) == ""  # no run yet
    assert _time_str(CellView(id="a", cache_hit=True)) == "cached"
    assert _time_str(CellView(id="a", duration_ms=120)) == "120ms"
    assert _time_str(CellView(id="a", duration_ms=1500)) == "1.5s"
    # cache hit wins over a recorded duration.
    assert _time_str(CellView(id="a", duration_ms=999, cache_hit=True)) == "cached"


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


def test_single_table_uri_returns_backing_artifact():
    out = _table_output()
    out["artifact_uri"] = "strata://artifact/nb_x_cell_a_var___display__0@v=1"
    cell = CellView(id="a", display_outputs=[out])
    assert _single_table_uri(cell) == "strata://artifact/nb_x_cell_a_var___display__0@v=1"


def test_single_table_uri_none_without_uri_or_single_table():
    # Table output but no artifact_uri → nothing to page.
    assert _single_table_uri(CellView(id="a", outputs=[_table_output()])) is None
    # Two tables → not a single table.
    out = _table_output()
    out["artifact_uri"] = "strata://artifact/x@v=1"
    assert _single_table_uri(CellView(id="a", outputs=[out, out])) is None


def test_render_table_includes_columns_values_and_truncation_caption():
    table = _render_table(["id", "name"], [[1, "alice"], [2, "bob"]], total=100)
    text = " ".join(_render_to_text(table).split())  # normalize wrapped whitespace
    assert "id" in text and "name" in text
    assert "alice" in text and "bob" in text
    assert "showing 2 of 100 rows" in text


def test_render_table_caps_wide_tables():
    cols = [f"c{i}" for i in range(20)]
    rows = [[f"v{i}" for i in range(20)]]
    text = " ".join(_render_to_text(_render_table(cols, rows, total=5)).split())
    # Only the first 8 columns render; the rest are signalled in the caption.
    assert "c0" in text and "c7" in text
    assert "c8" not in text and "c19" not in text
    assert "+12 more cols" in text


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
