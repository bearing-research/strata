"""Unit tests for the TUI's pure rendering helpers.

The interactive app lives behind a Textual harness; these tests pin the
small projection functions that turn snapshot payloads into the strings
the table / detail panels render. Keeping them cheap means the
``[tui]`` extra doesn't need to be installed for the CI suite to run —
the helpers don't import Textual.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def app_helpers():
    """Import the helpers lazily so the test doesn't pull in Textual.

    The module's top-level imports include Textual; we import the
    helpers via ``importlib`` after asserting the module imports cleanly
    so a missing ``[tui]`` extra surfaces as a skip rather than a
    collection error.
    """
    pytest.importorskip("textual")
    from strata.notebook.tui import app as module

    return module


class TestRenderStatus:
    def test_known_statuses_get_glyphs(self, app_helpers):
        assert app_helpers._render_status("ready").startswith("✓")
        assert app_helpers._render_status("running").startswith("▶")
        assert app_helpers._render_status("error").startswith("✗")

    def test_none_status_is_blank(self, app_helpers):
        assert app_helpers._render_status(None) == " "

    def test_unknown_status_falls_back_to_question_mark(self, app_helpers):
        assert app_helpers._render_status("zorbified").startswith("?")


class TestSummaryLine:
    def test_picks_first_non_blank(self, app_helpers):
        assert app_helpers._summary_line("\n\n# header\nx = 1") == "# header"

    def test_truncates_long_lines(self, app_helpers):
        line = "a" * 200
        out = app_helpers._summary_line(line)
        assert len(out) <= 80
        assert out.endswith("…")

    def test_empty_source(self, app_helpers):
        assert app_helpers._summary_line("") == "(empty)"
        assert app_helpers._summary_line("\n\n") == "(empty)"


class TestFormatOutputsDict:
    def test_dict_preview_extracts_text(self, app_helpers):
        outputs = {"x": {"text": "1", "preview": "1"}, "y": {"preview": "[1,2,3]"}}
        rendered = app_helpers._format_outputs_dict(outputs)
        assert "x = 1" in rendered
        assert "y = [1,2,3]" in rendered

    def test_scalar_outputs_pass_through(self, app_helpers):
        rendered = app_helpers._format_outputs_dict({"answer": 42})
        assert "answer = 42" in rendered


class TestFormatDisplay:
    def test_markdown_returns_text(self, app_helpers):
        display = {"content_type": "text/markdown", "markdown_text": "# hi"}
        assert app_helpers._format_display(display) == "# hi"

    def test_image_returns_placeholder(self, app_helpers):
        display = {"content_type": "image/png", "artifact_uri": "strata://..."}
        out = app_helpers._format_display(display)
        assert "image/png" in out
        assert "Vue" in out  # nudges users to open in Vue for the actual image

    def test_text_display(self, app_helpers):
        assert app_helpers._format_display({"content_type": "text/plain", "text": "hi"}) == "hi"

    def test_unknown_falls_back_to_content_type(self, app_helpers):
        out = app_helpers._format_display({"content_type": "application/x-arrow"})
        assert "application/x-arrow" in out
