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

        from textual.containers import VerticalScroll

        # The three detail panels split the detail pane — each well taller than
        # the old ~3-line squish.
        for pid in ("#source-scroll", "#output-scroll", "#console-scroll"):
            panel = app.query_one(pid, VerticalScroll)
            assert panel.size.height >= 4, f"{pid} squished to {panel.size.height} rows"

        # The source Static is height:auto, so it grows to its ~50-line content
        # and overflows its scroll region → scrollbar.
        source = app.query_one("#source", Static)
        source_panel = app.query_one("#source-scroll", VerticalScroll)
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


@pytest.mark.asyncio
async def test_agent_frames_render_in_agent_panel(monkeypatch):
    """agent_* frames stream into the Agent panel + drive its title/header."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(100, 40)) as pilot:
        app._dispatch(
            json.dumps(
                {
                    "type": "notebook_state",
                    "seq": 0,
                    "ts": "t",
                    "payload": {"name": "NB", "cells": [{"id": "a", "status": "idle"}]},
                }
            )
        )

        def _agent(msg_type, payload):
            app._dispatch(json.dumps({"type": msg_type, "seq": 0, "ts": "t", "payload": payload}))

        _agent("agent_text_delta", {"text": "Analyzing "})
        _agent("agent_text_delta", {"text": "the data."})
        _agent("agent_progress", {"event": "tool_call", "detail": "edit cell a"})
        await pilot.pause()

        agent_text = str(app.query_one("#agent", Static).render())
        assert "Analyzing the data." in agent_text
        assert "tool_call: edit cell a" in agent_text
        # Header banner reflects agent activity.
        assert "agent" in app.sub_title


@pytest.mark.asyncio
async def test_d_opens_dag_screen(monkeypatch):
    """`d` opens the layered DAG screen; Esc closes it."""
    from strata.notebook.tui.app import DagScreen

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(100, 40)) as pilot:
        app._dispatch(
            json.dumps(
                {
                    "type": "notebook_state",
                    "seq": 0,
                    "ts": "t",
                    "payload": {
                        "name": "NB",
                        "cells": [{"id": "a", "status": "ready"}, {"id": "b", "status": "idle"}],
                        "dag": {"edges": [{"from_cell_id": "a", "to_cell_id": "b"}]},
                    },
                }
            )
        )
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()
        assert isinstance(app.screen, DagScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, DagScreen)
