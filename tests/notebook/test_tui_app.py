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

        # Detail panes are tabs: the ACTIVE one gets (nearly) the full height —
        # not the ~1-row squish that stacking them produced on short terminals.
        # Each tab is full-height when activated.
        tabs = ((None, "#source-scroll"), ("3", "#output-scroll"), ("4", "#console-scroll"))
        for key, pid in tabs:
            if key is not None:
                await pilot.press(key)
                await pilot.pause()
            panel = app.query_one(pid, VerticalScroll)
            assert panel.size.height >= 20, f"{pid} only {panel.size.height} rows (terminal is 40)"

        # Back on Source: the Static is height:auto, so it grows to its ~50-line
        # content and overflows the (now tall) scroll region → scrollbar.
        await pilot.press("2")
        await pilot.pause()
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
async def test_resync_preserves_selection(monkeypatch):
    """A periodic/manual resync keeps the current selection (doesn't jump to top)."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(100, 30)) as pilot:
        state = {
            "type": "notebook_state",
            "seq": 0,
            "ts": "t",
            "payload": {
                "name": "NB",
                "cells": [
                    {"id": "a", "source": "x=1"},
                    {"id": "b", "source": "y=2"},
                    {"id": "c", "source": "z=3"},
                ],
            },
        }
        app._dispatch(json.dumps(state))
        await pilot.pause()

        app._select_cell("c")  # move selection to cell c
        await pilot.pause()
        assert app._selected == "c"

        # A no-op resync (identical state) must not disturb the selection.
        app._dispatch(json.dumps(state))
        await pilot.pause()
        assert app._selected == "c"

        # A resync that changes cell a's source rebuilds but keeps selection on c.
        changed = json.loads(json.dumps(state))
        changed["payload"]["cells"][0]["source"] = "x = 999"
        app._dispatch(json.dumps(changed))
        await pilot.pause()
        assert app._selected == "c"


@pytest.mark.asyncio
async def test_follow_mode_tracks_the_running_cell(monkeypatch):
    """With follow on, a cell going `running` is auto-selected; `f` toggles it off."""

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
                        "cells": [{"id": "a"}, {"id": "b"}, {"id": "c"}],
                    },
                }
            )
        )
        await pilot.pause()
        assert app._selected == "a"  # first cell selected on load

        def _status(cid, status):
            app._dispatch(
                json.dumps(
                    {
                        "type": "cell_status",
                        "seq": 0,
                        "ts": "t",
                        "payload": {"cell_id": cid, "status": status},
                    }
                )
            )

        _status("c", "running")
        await pilot.pause()
        assert app._selected == "c"  # followed to the running cell

        await pilot.press("f")  # turn follow off
        _status("b", "running")
        await pilot.pause()
        assert app._selected == "c"  # stays put when follow is off


@pytest.mark.asyncio
async def test_image_output_renders_without_crashing(monkeypatch):
    """A single image/png output is rendered inline (terminal-image renderable)."""
    import base64
    import io

    from PIL import Image as PILImage

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    buf = io.BytesIO()
    PILImage.new("RGB", (8, 8), "green").save(buf, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(80, 24)) as pilot:
        app._dispatch(
            json.dumps(
                {
                    "type": "notebook_state",
                    "seq": 0,
                    "ts": "t",
                    "payload": {
                        "name": "NB",
                        "cells": [
                            {
                                "id": "a",
                                "status": "ready",
                                "display_outputs": [
                                    {"content_type": "image/png", "inline_data_url": data_url}
                                ],
                            }
                        ],
                    },
                }
            )
        )
        await pilot.pause()
        # The image render path was taken (Output isn't the text placeholder).
        out = str(app.query_one("#output", Static).render())
        assert "open in the web UI" not in out

        # `i` enlarges it to the full-screen image view.
        from strata.notebook.tui.app import ImageScreen

        await pilot.press("i")
        await pilot.pause()
        assert isinstance(app.screen, ImageScreen)


@pytest.mark.asyncio
async def test_markdown_language_cell_renders_in_output(monkeypatch):
    """A markdown-language cell renders its source as markdown in the Output tab."""

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
                        "cells": [
                            {
                                "id": "a",
                                "language": "markdown",
                                "source": "# Heading\n\nbody text",
                                "status": "ready",
                            }
                        ],
                    },
                }
            )
        )
        await pilot.pause()
        out = str(app.query_one("#output", Static).render())
        # The markdown render path was taken: the Output is a rich Markdown
        # renderable, not the plain-text "(no output)" placeholder.
        assert "(no output)" not in out
        assert "Markdown" in out


@pytest.mark.asyncio
async def test_cell_test_results_show_badge_in_cell_label(monkeypatch):
    """A cell_test_results frame surfaces a pass/fail badge in the cell-list label."""

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
                    "payload": {"name": "NB", "cells": [{"id": "a", "source": "x=1"}]},
                }
            )
        )
        await pilot.pause()
        app._dispatch(
            json.dumps(
                {
                    "type": "cell_test_results",
                    "seq": 0,
                    "ts": "t",
                    "payload": {"cell_id": "a", "passed": 3, "failed": 0, "errored": 0},
                }
            )
        )
        await pilot.pause()
        assert "✓ 3/3" in app._cell_label(app.vm.cells["a"])


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
async def test_question_mark_opens_help_screen(monkeypatch):
    """`?` opens the keybinding help; Esc closes it."""
    from strata.notebook.tui.app import HelpScreen

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
        await pilot.pause()
        await pilot.press("question_mark")
        await pilot.pause()
        assert isinstance(app.screen, HelpScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, HelpScreen)


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
