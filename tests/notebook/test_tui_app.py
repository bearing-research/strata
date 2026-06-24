"""Headless layout test for the TUI app (Textual pilot, no terminal, no network).

Guards the regression where the detail panels rendered ~3 lines and didn't
scroll: the inner content was ``height: 1fr`` (filling the scroll region exactly,
so nothing ever overflowed). The fix makes the scroll region ``1fr`` and the
inner content ``height: auto`` so long content overflows → the scrollbar engages.
"""

from __future__ import annotations

import json

import pytest
from textual.widgets import Static

from strata.notebook.tui.app import NotebookTUI
from strata.notebook.tui.client import TuiClient


@pytest.mark.asyncio
async def test_detail_panels_are_tall_and_content_overflows(monkeypatch):
    # Don't hit the network on mount — we only exercise layout/render.
    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(100, 40)) as pilot:
        long_source = "\n".join(f"line {i}" for i in range(50))
        app._dispatch(
            json.dumps(
                {
                    "type": "notebook_state",
                    "seq": 0,
                    "ts": "t",
                    "payload": {
                        "name": "NB",
                        "cells": [{"id": "a", "source": long_source, "status": "ready"}],
                    },
                }
            )
        )
        await pilot.pause()

        panels = list(app.query(".scroll-panel"))
        assert len(panels) == 3
        # On a 40-row terminal the three panels split the detail pane — each is
        # well taller than the old ~3-line squish.
        for panel in panels:
            assert panel.size.height > 5, f"panel squished to {panel.size.height} rows"

        # The source Static is height:auto, so it grows to its ~50-line content
        # and overflows its scroll region (the first .scroll-panel) → scrollbar.
        source = app.query_one("#source", Static)
        source_panel = panels[0]
        assert source.size.height >= 45, f"content didn't grow (got {source.size.height})"
        assert source.size.height > source_panel.size.height


@pytest.mark.asyncio
async def test_number_keys_focus_panels_for_arrow_scrolling(monkeypatch):
    """1/2/3/4 select a panel so the arrow keys navigate/scroll it."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press("2")  # Source
        await pilot.pause()
        assert app.focused is app.query_one("#source-scroll")

        await pilot.press("4")  # Console
        await pilot.pause()
        assert app.focused is app.query_one("#console-scroll")

        await pilot.press("1")  # Cells
        await pilot.pause()
        assert app.focused is app.query_one("#cells")
