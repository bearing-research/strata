"""Headless layout test for the TUI app (Textual pilot, no terminal, no network).

Guards the regression where the detail panels rendered ~3 lines and didn't
scroll: the inner content was ``height: 1fr`` (filling the scroll region exactly,
so nothing ever overflowed). The fix makes the scroll region ``1fr`` and the
inner content ``height: auto`` so long content overflows → the scrollbar engages.
"""

from __future__ import annotations

import json

import pytest
from textual.widgets import DataTable, Static

from strata.notebook.tui.app import NotebookTUI
from strata.notebook.tui.client import TuiClient
from strata.notebook.tui.viewmodel import CellView


class _FakeDataClient:
    """Stand-in for TuiClient that serves canned data-viewer pages."""

    def __init__(self) -> None:
        self.exported: tuple[str, str | None, str] | None = None

    async def get_cell_data_page(
        self, notebook_id, cell_id, artifact_uri, *, offset, limit, sort_by, sort_dir
    ):
        rows = [[i, i * 2] for i in range(offset, min(offset + limit, 2000))]
        return {
            "pageable": True,
            "columns": ["id", "v"],
            "rows": rows,
            "total": 2000,
            "offset": offset,
            "limit": limit,
            "sort_by": sort_by,
            "sort_dir": sort_dir,
        }

    async def export_cell_data(
        self, notebook_id, cell_id, artifact_uri, *, fmt, sort_by, sort_dir
    ) -> bytes:
        self.exported = (fmt, sort_by, sort_dir)
        return b"id,v\n1,2\n"


async def _wait(pilot, cond, tries=40) -> None:
    for _ in range(tries):
        await pilot.pause(0.05)
        if cond():
            return


def _table_cell() -> CellView:
    return CellView(
        id="c1",
        display_outputs=[
            {
                "content_type": "arrow/ipc",
                "columns": ["id", "v"],
                "preview": [[0, 0]],
                "rows": 2000,
                "artifact_uri": "strata://artifact/a@v=1",
            }
        ],
    )


def _select(app: NotebookTUI, cell: CellView) -> None:
    app.vm.cells[cell.id] = cell
    app.vm.cell_order = [cell.id]
    app._selected = cell.id
    app._show_detail(cell.id)


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

        # Two stacked tab-groups (top: code, bottom: runtime). The active pane in
        # each group gets that group's height (~half the column) — not the ~1-row
        # squish that fully-stacking every panel produced on short terminals.
        tabs = (
            (None, "#source-scroll"),  # top group, default
            ("3", "#testsrc-scroll"),  # top group
            ("4", "#output-scroll"),  # bottom group, default
            ("5", "#console-scroll"),  # bottom group
        )
        for key, pid in tabs:
            if key is not None:
                await pilot.press(key)
                await pilot.pause()
            panel = app.query_one(pid, VerticalScroll)
            assert panel.size.height >= 10, f"{pid} only {panel.size.height} rows (terminal is 40)"

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
    """Number keys select a panel (across both groups) so arrows scroll it."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press("3")  # Tests source (top group)
        await pilot.pause()
        assert app.focused is app.query_one("#testsrc-scroll")

        await pilot.press("5")  # Console (bottom group)
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
async def test_live_frame_updates_table_row_without_resync(monkeypatch):
    """A live cell frame updates the cell-list row in place (not only on resync).

    Regression: the columns are keyed by auto-generated ColumnKeys, so update_cell
    by label silently raised and _refresh_cell bailed before re-rendering.
    """
    from textual.widgets import DataTable

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
        table = app.query_one("#cells", DataTable)
        status_col = app._col_keys[0]
        assert str(table.get_cell("a", status_col)) == "○"  # idle glyph
        app._dispatch(
            json.dumps(
                {
                    "type": "cell_status",
                    "seq": 0,
                    "ts": "t",
                    "payload": {"cell_id": "a", "status": "ready"},
                }
            )
        )
        await pilot.pause()
        assert str(table.get_cell("a", status_col)) == "✓"  # updated live, no resync


@pytest.mark.asyncio
async def test_tests_source_tab_shows_test_code(monkeypatch):
    """Key 3 (top group) shows the cell's test source from the snapshot."""
    from textual.containers import VerticalScroll

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
                                "source": "x = 1",
                                "test_source": "def test_x(cell):\n    assert cell.x == 1",
                            }
                        ],
                    },
                }
            )
        )
        await pilot.pause()
        assert app.vm.cells["a"].test_source.startswith("def test_x")
        await pilot.press("3")  # Tests source tab (top group)
        await pilot.pause()
        assert app.focused is app.query_one("#testsrc-scroll", VerticalScroll)
        # A cell with no test source shows the placeholder, not a crash.
        body = str(app.query_one("#testsrc-body", Static).render())
        assert "no tests for this cell" not in body  # this cell HAS test source


@pytest.mark.asyncio
async def test_tests_tab_shows_per_test_outcomes(monkeypatch):
    """Key 6 opens the Tests tab; it renders per-test outcomes + failure messages."""

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
                    "payload": {
                        "cell_id": "a",
                        "passed": 1,
                        "failed": 1,
                        "tests": [
                            {"name": "test_ok", "outcome": "passed", "message": ""},
                            {"name": "test_bad", "outcome": "failed", "message": "assert 1 == 2"},
                        ],
                    },
                }
            )
        )
        await pilot.pause()
        await pilot.press("7")  # focus the Results tab (bottom group)
        await pilot.pause()
        from textual.containers import VerticalScroll

        assert app.focused is app.query_one("#results-scroll", VerticalScroll)
        body = str(app.query_one("#results-body", Static).render())
        assert "test_ok" in body and "test_bad" in body
        assert "assert 1 == 2" in body  # failure message shown


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


def test_nudge_clamps_to_range():
    from strata.notebook.tui.app import _MAX_PCT, _MIN_PCT, _nudge

    assert _nudge(38, 4) == 42
    assert _nudge(38, -4) == 34
    assert _nudge(_MIN_PCT, -10) == _MIN_PCT  # clamped low
    assert _nudge(_MAX_PCT, 10) == _MAX_PCT  # clamped high


@pytest.mark.asyncio
async def test_panel_resize_keys_move_and_reset_boundaries(monkeypatch):
    from strata.notebook.tui.app import _MAX_PCT

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    async with app.run_test(size=(100, 40)) as pilot:
        assert (app._cells_pct, app._top_pct) == (38, 50)

        await pilot.press("ctrl+right")  # widen the cell list
        await pilot.pause()
        assert app._cells_pct == 42
        assert "42" in str(app.query_one("#cells").styles.width)

        for _ in range(20):  # holds at the max — never runs away
            await pilot.press("ctrl+right")
        await pilot.pause()
        assert app._cells_pct == _MAX_PCT

        await pilot.press("ctrl+down")  # grow the top detail region
        await pilot.pause()
        assert app._top_pct == 55

        await pilot.press("ctrl+x")  # reset to defaults
        await pilot.pause()
        assert (app._cells_pct, app._top_pct) == (38, 50)


@pytest.mark.asyncio
async def test_ws_connect_disables_frame_size_cap(monkeypatch):
    """The WS client must pass ``max_size=None`` to ``websockets.connect``.

    notebook_state / cell_output frames carry display outputs (base64 PNG plots,
    large tables) that routinely exceed the websockets client default of 1 MiB.
    Without ``max_size=None`` the client rejects the first oversized frame, closes
    the connection, and the reconnect loop wedges into a storm (a real regression
    against notebooks with image outputs).
    """
    import asyncio

    captured: dict[str, object] = {}

    def fake_connect(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        raise asyncio.CancelledError  # break out of the connect loop immediately

    monkeypatch.setattr("strata.notebook.tui.app.websockets.connect", fake_connect)

    app = NotebookTUI(client=TuiClient("http://localhost:8765"), session_id="x")
    with pytest.raises(asyncio.CancelledError):
        await app._ws_loop("sid-1")

    assert "max_size" in captured, "websockets.connect was not called with max_size"
    assert captured["max_size"] is None


@pytest.mark.asyncio
async def test_data_viewer_pages_sorts_and_exports(monkeypatch, tmp_path):
    """A tabular output with a backing artifact drives the interactive viewer."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)
    monkeypatch.chdir(tmp_path)  # export writes {cell_id}.csv to cwd

    fake = _FakeDataClient()
    app = NotebookTUI(client=fake, session_id="s1")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        _select(app, _table_cell())
        await _wait(pilot, lambda: app._table_view is not None and app._table_view.total > 0)

        dt = app.query_one("#output-table", DataTable)
        assert dt.display is True
        assert app._table_view.total == 2000
        assert dt.row_count == 50
        assert "of 2,000 rows" in str(dt.border_title)

        # Paging advances / rewinds the window.
        app.action_table_next()
        await _wait(pilot, lambda: app._table_view.offset == 50)
        assert app._table_view.offset == 50
        app.action_table_prev()
        await _wait(pilot, lambda: app._table_view.offset == 0)
        assert app._table_view.offset == 0

        # Sorting the focused column cycles asc → desc → cleared.
        dt.move_cursor(column=1)
        app.action_table_sort()
        await _wait(pilot, lambda: app._table_view.sort_by == "v")
        assert app._table_view.sort_dir == "asc"
        app.action_table_sort()
        await _wait(pilot, lambda: app._table_view.sort_dir == "desc")
        app.action_table_sort()
        await _wait(pilot, lambda: app._table_view.sort_by is None)

        # Export writes a CSV to the cwd, carrying the active sort.
        dt.move_cursor(column=1)
        app.action_table_sort()
        await _wait(pilot, lambda: app._table_view.sort_by == "v")
        app.action_table_export()
        await _wait(pilot, lambda: fake.exported is not None)
        assert fake.exported == ("csv", "v", "asc")
        assert (tmp_path / "c1.csv").exists()


@pytest.mark.asyncio
async def test_data_viewer_hidden_for_non_table_output(monkeypatch):
    """Markdown / non-table outputs keep the static area; the table stays hidden."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)
    app = NotebookTUI(client=_FakeDataClient(), session_id="s1")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        cell = CellView(
            id="c1",
            display_outputs=[{"content_type": "text/markdown", "markdown_text": "# Hi"}],
        )
        _select(app, cell)
        await pilot.pause()

        assert app.query_one("#output-table", DataTable).display is False
        assert app._table_view is None


@pytest.mark.asyncio
async def test_data_viewer_ignores_keys_when_hidden(monkeypatch):
    """The n/p/s/e actions are no-ops unless a table is the visible output."""

    async def _noop(self) -> None:
        return None

    monkeypatch.setattr(NotebookTUI, "_bootstrap", _noop)
    app = NotebookTUI(client=_FakeDataClient(), session_id="s1")
    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.pause()
        # No cell selected → no table view; actions must not raise.
        app.action_table_next()
        app.action_table_sort()
        app.action_table_export()
        await pilot.pause()
        assert app._table_view is None
