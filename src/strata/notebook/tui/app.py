"""Read-only Textual spectator app for a live notebook session (TUI Phase 1, M1).

Resolves a session (flag, path, or interactive picker), opens the WS, sends
``notebook_sync``, and renders the resulting ``notebook_state`` plus the live
``cell_status`` / ``cell_console`` / ``cell_output`` stream. No editing, no run
keybindings — a spectator. The cascade/dag/env frames (M2) and the agent panel
(M3) build on this same dispatch loop.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import io
import json
from datetime import UTC, datetime
from typing import Any

import websockets
from PIL import Image as PILImage
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    OptionList,
    Static,
    TabbedContent,
    TabPane,
)
from textual.widgets.option_list import Option
from textual_image.renderable import Image as TerminalImage

from strata.notebook.tui.client import TuiClient, TuiClientError
from strata.notebook.tui.dag_render import render_dag
from strata.notebook.tui.viewmodel import CellView, NotebookViewModel

# Status glyphs for ``CellStatus`` values. ``?`` is the placeholder before the
# first status is known (or for cells a snapshot carries no status for).
_STATUS_GLYPHS: dict[str, str] = {
    "idle": "○",
    "running": "▶",
    "ready": "✓",
    "error": "✗",
    "stale": "⊘",
    "queued": "…",
}


def _utc_iso_z() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def _glyph(status: str) -> str:
    return _STATUS_GLYPHS.get(status, "?")


def _source_preview(source: str) -> str:
    """First line of actual code for the cell-list label.

    Skips the leading ``#``-comment/annotation block (e.g. ``# @name load``) so a
    named cell shows its code, not a redundant repeat of its name. Falls back to
    the first non-blank line (a comment-only cell) or ``(empty)``.
    """
    lines = source.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    for line in lines:
        if line.strip():
            return line.strip()
    return "(empty)"


class SessionPickerScreen(ModalScreen[str]):
    """Modal list of running sessions; dismisses with the chosen ``session_id``."""

    BINDINGS = [Binding("escape", "dismiss_none", "Cancel")]

    def __init__(self, sessions: list[dict[str, Any]]) -> None:
        super().__init__()
        self._sessions = sessions

    def compose(self) -> ComposeResult:
        yield Static("Select a session to watch", classes="panel-title")
        options = [
            Option(
                f"{s.get('name') or '(unnamed)'}   {s.get('path') or ''}",
                id=str(s.get("session_id")),
            )
            for s in self._sessions
        ]
        yield OptionList(*options, id="session-picker")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(str(event.option.id))

    def action_dismiss_none(self) -> None:
        self.app.exit(message="No session selected.")


class DagScreen(ModalScreen[None]):
    """Full-screen layered ASCII view of the notebook DAG (read-only)."""

    BINDINGS = [Binding("escape,d,q", "dismiss", "Close")]

    CSS = """
    DagScreen { align: center middle; }
    #dag-box { width: 90%; height: 90%; border: solid $accent; }
    #dag-art { width: auto; height: auto; padding: 1 2; }
    """

    def __init__(self, dag_text: str) -> None:
        super().__init__()
        self._dag_text = dag_text

    def compose(self) -> ComposeResult:
        yield Static("DAG  (Esc/d/q to close · arrows to scroll)", classes="panel-title")
        with VerticalScroll(id="dag-box"):
            yield Static(self._dag_text, id="dag-art")


class ImageScreen(ModalScreen[None]):
    """Full-screen view of a cell's image output — renders larger than the panel."""

    BINDINGS = [Binding("escape,i,q", "dismiss", "Close")]

    CSS = """
    ImageScreen { align: center middle; }
    #image-box { width: 95%; height: 95%; border: solid $accent; }
    #image-art { width: auto; height: auto; padding: 1 2; }
    """

    def __init__(self, renderable: Any) -> None:
        super().__init__()
        self._renderable = renderable

    def compose(self) -> ComposeResult:
        yield Static("Image  (Esc/i/q to close · arrows to scroll)", classes="panel-title")
        with VerticalScroll(id="image-box"):
            yield Static(self._renderable, id="image-art")


class HelpScreen(ModalScreen[None]):
    """Full-screen keybinding reference (read-only)."""

    BINDINGS = [Binding("escape,question_mark,q", "dismiss", "Close")]

    CSS = """
    HelpScreen { align: center middle; }
    #help-box { width: 60%; height: auto; max-height: 90%; border: solid $accent; }
    #help-art { width: auto; height: auto; padding: 1 2; }
    """

    # (key, action) rows, mirroring docs/notebook/tui.md.
    _ROWS = [
        ("1", "Focus the cell list (↑/↓ move the selection)"),
        ("2 / 3 / 4 / 5", "Source / Output / Console / Agent tab"),
        ("↑ ↓ PgUp PgDn Home End", "Scroll the focused pane"),
        ("f", "Toggle follow mode (auto-select the running cell)"),
        ("d", "Show the notebook DAG"),
        ("i", "Enlarge the selected cell's image output"),
        ("r", "Force an immediate resync (also auto-resyncs in the background)"),
        ("?", "Show this help"),
        ("q", "Quit"),
    ]

    def compose(self) -> ComposeResult:
        yield Static("Keybindings  (Esc/?/q to close)", classes="panel-title")
        table = Table(show_header=True, header_style="bold", expand=False, box=None)
        table.add_column("Key", style="bold cyan", no_wrap=True)
        table.add_column("Action")
        for key, action in self._ROWS:
            table.add_row(key, action)
        with VerticalScroll(id="help-box"):
            yield Static(table, id="help-art")


class NotebookTUI(App[None]):
    """Top-level spectator app."""

    # The detail panes are TABS (one at a time) rather than stacked, so each gets
    # the full height — stacking Source/Output/Console/Agent collapsed them to ~1
    # row on a short (24-line) terminal.
    CSS = """
    #cells { width: 38%; border: solid $primary; }
    #detail { width: 62%; }
    .scroll-panel { height: 1fr; }
    #source, #output, #console-body, #agent { height: auto; width: 1fr; padding: 0 1; }
    #cells:focus, .scroll-panel:focus { background: $boost; }
    .panel-title { background: $primary; color: $text; padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "show_help", "Help"),
        Binding("r", "refresh", "Resync"),
        Binding("d", "show_dag", "DAG"),
        Binding("i", "view_image", "Image"),
        Binding("f", "toggle_follow", "Follow"),
        Binding("1", "focus_cells", "Cells"),
        Binding("2", "show_tab('tab-source')", "Source"),
        Binding("3", "show_tab('tab-output')", "Output"),
        Binding("4", "show_tab('tab-console')", "Console"),
        Binding("5", "show_tab('tab-agent')", "Agent"),
    ]

    # Tab id → the scroll region to focus when that tab is selected.
    _TAB_SCROLL = {
        "tab-source": "#source-scroll",
        "tab-output": "#output-scroll",
        "tab-console": "#console-scroll",
        "tab-agent": "#agent-scroll",
    }

    def __init__(
        self,
        *,
        client: TuiClient,
        session_id: str | None = None,
        notebook_path: str | None = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._session_id = session_id
        self._notebook_path = notebook_path
        self._ws: Any = None
        self.vm = NotebookViewModel()
        self._selected: str | None = None
        self._conn_state = "connecting…"
        # The selected cell's image renderable (if any) — enlarged by `i`.
        self._current_image: Any = None
        # Signature of the last-rendered cell list, so a periodic resync that
        # changed nothing is a no-op (no flicker, no selection jump).
        self._render_sig: tuple[Any, ...] = ()
        # Follow mode: auto-select the cell that goes running so the detail
        # panels track the action (an agent / run-all moving through the notebook).
        self._follow = True

    # -- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield DataTable(id="cells", cursor_type="row", zebra_stripes=True)
            with TabbedContent(id="detail", initial="tab-source"):
                with TabPane("Source", id="tab-source"):
                    with VerticalScroll(id="source-scroll", classes="scroll-panel"):
                        yield Static("", id="source")
                with TabPane("Output", id="tab-output"):
                    with VerticalScroll(id="output-scroll", classes="scroll-panel"):
                        yield Static("", id="output")
                with TabPane("Console", id="tab-console"):
                    with VerticalScroll(id="console-scroll", classes="scroll-panel"):
                        yield Static("", id="console-body")
                with TabPane("Agent", id="tab-agent"):
                    with VerticalScroll(id="agent-scroll", classes="scroll-panel"):
                        yield Static("(no agent activity)", id="agent")
        yield Footer()

    # -- lifecycle -----------------------------------------------------------

    async def on_mount(self) -> None:
        table = self.query_one("#cells", DataTable)
        table.add_columns(" ", "cell", "time")
        self._set_connection("connecting…")
        self.run_worker(self._bootstrap(), name="bootstrap", exclusive=True)
        # Live frames stream status/output/console instantly, but source edits and
        # cell add/remove/reorder only arrive in a full snapshot — so poll one
        # periodically. The rebuild is a no-op when nothing changed (see
        # _rebuild_cells), so this stays cheap and never disturbs the selection.
        self.set_interval(2.5, self._send_sync)

    async def _bootstrap(self) -> None:
        try:
            session_id = await self._resolve_session()
        except TuiClientError as exc:
            self.exit(message=str(exc))
            return
        if session_id is None:
            return  # picker cancelled → app already exiting
        self._session_id = session_id
        await self._ws_loop(session_id)

    async def _resolve_session(self) -> str | None:
        if self._session_id:
            return self._session_id
        if self._notebook_path:
            data = await self._client.open_notebook(self._notebook_path)
            sid = data.get("session_id")
            return str(sid) if sid else None

        sessions = await self._client.list_sessions()
        if not sessions:
            raise TuiClientError(
                "No running notebook sessions. Open one (in the web UI or via "
                "`POST /v1/notebooks/open`), or pass --notebook <path>."
            )
        if len(sessions) == 1:
            return str(sessions[0].get("session_id"))
        return await self.push_screen_wait(SessionPickerScreen(sessions))

    async def action_refresh(self) -> None:
        await self._send_sync()

    def action_show_dag(self) -> None:
        """Open the layered DAG view of the current cells/edges."""
        if not self.vm.cell_order:
            return
        labels = {cid: (self.vm.cells[cid].name or cid) for cid in self.vm.cell_order}
        statuses = {cid: self.vm.cells[cid].status for cid in self.vm.cell_order}
        dag_text = render_dag(
            self.vm.cell_order, labels, statuses, self.vm.edges, selected=self._selected
        )
        self.push_screen(DagScreen(dag_text))

    def action_view_image(self) -> None:
        """Enlarge the selected cell's image output to (almost) full screen."""
        if self._current_image is not None:
            self.push_screen(ImageScreen(self._current_image))

    def action_show_help(self) -> None:
        """Show the keybinding reference."""
        self.push_screen(HelpScreen())

    def action_focus_cells(self) -> None:
        """Focus the cell list so up/down move the selection."""
        try:
            self.query_one("#cells", DataTable).focus()
        except Exception:  # noqa: BLE001 — not mounted yet
            return

    def action_show_tab(self, tab_id: str) -> None:
        """Switch the detail tab and focus its scroll region (up/down scroll it)."""
        try:
            self.query_one("#detail", TabbedContent).active = tab_id
            self.query_one(self._TAB_SCROLL[tab_id], VerticalScroll).focus()
        except Exception:  # noqa: BLE001 — not mounted yet
            return

    # -- WS loop -------------------------------------------------------------

    async def _ws_loop(self, session_id: str) -> None:
        url = self._client.ws_url(session_id)
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(
                    url, additional_headers=self._client.auth_headers or None
                ) as ws:
                    self._ws = ws
                    backoff = 1.0
                    self._set_connection("connected")
                    await self._send_sync()
                    async for raw in ws:
                        self._dispatch(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — any drop → reconnect with backoff
                self._ws = None
                self._set_connection(f"reconnecting… ({type(exc).__name__})")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 10.0)

    async def _send_sync(self) -> None:
        if self._ws is None:
            return
        await self._ws.send(
            json.dumps({"type": "notebook_sync", "seq": 0, "ts": _utc_iso_z(), "payload": {}})
        )

    def _dispatch(self, raw: str | bytes) -> None:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(message, dict):
            return
        msg_type = message.get("type")
        payload = message.get("payload")
        if not isinstance(msg_type, str) or not isinstance(payload, dict):
            return

        if msg_type == "notebook_state":
            self.vm.apply_notebook_state(payload)
            self._rebuild_cells()
            self._render_status()
            return
        changed = self.vm.apply_frame(msg_type, payload)
        for cid in changed:
            self._refresh_cell(cid)
        # Follow mode: jump to a cell as it starts running so the detail panels
        # track the action.
        if self._follow and msg_type == "cell_status":
            cid = payload.get("cell_id")
            if isinstance(cid, str):
                cell = self.vm.cells.get(cid)
                if cell is not None and cell.status == "running":
                    self._select_cell(cid)
        # Notebook-level activity (cascade / env job / agent) updates the header
        # banner + agent panel even when no specific cell changed.
        self._render_status()
        if msg_type.startswith("agent_"):
            self._render_agent()

    # -- rendering -----------------------------------------------------------

    def _set_connection(self, state: str) -> None:
        self._conn_state = state
        self._render_status()

    def action_toggle_follow(self) -> None:
        self._follow = not self._follow
        self._render_status()

    def _render_status(self) -> None:
        self.title = self.vm.notebook_name or "Strata Notebook"
        bits = [self._conn_state]
        if self._follow:
            bits.append("⏵ follow")
        if self.vm.banner:
            bits.append(self.vm.banner)
        self.sub_title = "  ·  ".join(bits)

    def _select_cell(self, cid: str) -> None:
        """Move the cell-list cursor to *cid* (which refreshes the detail panels)."""
        if cid not in self.vm.cell_order:
            return
        table = self.query_one("#cells", DataTable)
        try:
            table.move_cursor(row=self.vm.cell_order.index(cid), animate=False)
        except Exception:  # noqa: BLE001 — row not materialized yet
            return

    def _render_agent(self) -> None:
        body = "\n".join(self.vm.agent_feed) if self.vm.agent_feed else "(no agent activity)"
        self.query_one("#agent", Static).update(body)
        # Reflect agent status on the tab label; the header banner has it too.
        label = f"Agent · {self.vm.agent_status}" if self.vm.agent_status else "Agent"
        try:
            self.query_one("#detail", TabbedContent).get_tab("tab-agent").label = label
        except Exception:  # noqa: BLE001 — tab not mounted yet
            pass
        # Follow the stream: keep the latest reasoning/events in view.
        self.query_one("#agent-scroll", VerticalScroll).scroll_end(animate=False)

    def _rebuild_cells(self) -> None:
        self._set_connection("connected")
        # Skip the rebuild when nothing the list shows has changed — so the
        # periodic resync doesn't flicker the table or move the cursor.
        sig = tuple(
            (cid, c.status, c.name, c.iteration, c.duration_ms, c.cache_hit, c.source)
            for cid in self.vm.cell_order
            for c in (self.vm.cells[cid],)
        )
        if sig == self._render_sig:
            return
        self._render_sig = sig

        table = self.query_one("#cells", DataTable)
        table.clear()
        for cid in self.vm.cell_order:
            cell = self.vm.cells[cid]
            table.add_row(_glyph(cell.status), self._cell_label(cell), _time_str(cell), key=cid)
        if self.vm.cell_order:
            if self._selected not in self.vm.cells:
                self._selected = self.vm.cell_order[0]
            # Restore the cursor to the selected cell (clear() reset it to row 0).
            try:
                table.move_cursor(row=self.vm.cell_order.index(self._selected), animate=False)
            except Exception:  # noqa: BLE001 — row not materialized yet
                pass
            self._show_detail(self._selected)

    def _cell_label(self, cell: CellView) -> str:
        name = cell.name or cell.id[:8]
        suffix = f"  [{cell.iteration}]" if cell.iteration else ""
        return f"{name}  {_source_preview(cell.source)[:40]}{suffix}"

    def _refresh_cell(self, cid: str) -> None:
        cell = self.vm.cells.get(cid)
        if cell is None:
            return
        table = self.query_one("#cells", DataTable)
        try:
            table.update_cell(cid, " ", _glyph(cell.status))
            table.update_cell(cid, "cell", self._cell_label(cell))
            table.update_cell(cid, "time", _time_str(cell))
        except Exception:  # noqa: BLE001 — row may not exist yet (pre-snapshot frame)
            return
        if cid == self._selected:
            self._show_detail(cid)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        cid = event.row_key.value if event.row_key else None
        if isinstance(cid, str):
            self._selected = cid
            self._show_detail(cid)

    def _show_detail(self, cid: str) -> None:
        cell = self.vm.cells.get(cid)
        if cell is None:
            return
        self.query_one("#source", Static).update(_source_renderable(cell))
        # Render a pure-markdown output with Rich, a single tabular output as a
        # real table, or a single image inline; otherwise the plain-text summary.
        output = self.query_one("#output", Static)
        markdown = _single_markdown(cell)
        table = None if markdown is not None else _single_table(cell)
        image = None if (markdown is not None or table is not None) else _image_renderable(cell)
        self._current_image = image  # enable `i` to enlarge when there's an image
        if markdown is not None:
            output.update(Markdown(markdown))
        elif table is not None:
            output.update(_render_table(*table))
        elif image is not None:
            output.update(image)
        else:
            output.update(_render_outputs(cell))
        self.query_one("#console-body", Static).update(cell.console or "(no console output)")


def _single_markdown(cell: CellView) -> str | None:
    """Return the cell's markdown text when its output is exactly one rendered
    markdown block (so it can render with Rich), else None for the text path.
    """
    if cell.error or cell.stream_text or cell.outputs:
        return None
    if len(cell.display_outputs) != 1:
        return None
    output = cell.display_outputs[0]
    if output.get("content_type") == "text/markdown" and isinstance(
        output.get("markdown_text"), str
    ):
        return output["markdown_text"]
    return None


# Cell language → Pygments lexer for source highlighting.
_SOURCE_LEXERS = {
    "python": "python",
    "sql": "sql",
    "markdown": "markdown",
    "r": "r",
    "prompt": "markdown",  # prompt cells are templated text
}


def _source_renderable(cell: CellView):
    """Syntax-highlighted source for the cell's language (plain text on failure)."""
    if not cell.source:
        return "(empty)"
    lexer = _SOURCE_LEXERS.get(cell.language, "python")
    try:
        # ``one-dark`` is the same theme the web UI uses (@codemirror/theme-one-dark),
        # so the terminal source view matches the browser. It's truecolor; Textual
        # downsamples for terminals without 24-bit support.
        return Syntax(cell.source, lexer, theme="one-dark", word_wrap=True)
    except Exception:  # noqa: BLE001 — unknown lexer / pygments hiccup → raw source
        return cell.source


def _time_str(cell: CellView) -> str:
    """Compact last-run timing for the cell list: 'cached', '0.4s', or '120ms'."""
    if cell.cache_hit:
        return "cached"
    ms = cell.duration_ms
    if ms is None:
        return ""
    return f"{ms / 1000:.1f}s" if ms >= 1000 else f"{ms}ms"


def _is_table(output: dict[str, Any]) -> bool:
    """True for a tabular output (arrow/ipc with named columns + a row preview)."""
    return (
        str(output.get("content_type") or "").startswith("arrow")
        and isinstance(output.get("columns"), list)
        and bool(output.get("columns"))
        and isinstance(output.get("preview"), list)
    )


def _single_table(cell: CellView) -> tuple[list[str], list[Any], int | None] | None:
    """Return (columns, preview-rows, total-rows) when the cell has exactly one
    tabular output (so it renders as a real table), else None for the text path.
    """
    if cell.error or cell.stream_text:
        return None
    candidates = [o for o in (*cell.display_outputs, *cell.outputs) if isinstance(o, dict)]
    if len(candidates) != 1 or not _is_table(candidates[0]):
        return None
    output = candidates[0]
    rows = output.get("rows")
    return (
        [str(c) for c in output["columns"]],
        output["preview"],
        rows if isinstance(rows, int) else None,
    )


# A terminal can't show a wide DataFrame's every column legibly — cap the count
# and signal the rest in the caption (the web UI is where you see them all).
_MAX_TABLE_COLS = 8


def _render_table(columns: list[str], preview: list[Any], total: int | None) -> Table:
    """Build a Rich table from a serialized preview (≤20 rows × ≤8 columns).

    Values truncate with an ellipsis (one line each) rather than folding into tall
    rows; extra rows and columns are noted in the caption.
    """
    table = Table(show_header=True, header_style="bold", expand=False)
    shown = columns[:_MAX_TABLE_COLS]
    extra_cols = len(columns) - len(shown)
    for name in shown:
        table.add_column(name, overflow="ellipsis", max_width=24, no_wrap=True)
    if extra_cols:
        table.add_column("…")
    for row in preview[:20]:
        if isinstance(row, list):
            cells = [_cell_str(v) for v in row[: len(shown)]]
        elif isinstance(row, dict):  # defensive: map by column name
            cells = [_cell_str(row.get(name)) for name in shown]
        else:
            continue
        if extra_cols:
            cells.append("…")
        table.add_row(*cells)
    shown_rows = len(preview[:20])
    notes = []
    if total is not None and total > shown_rows:
        notes.append(f"{shown_rows} of {total} rows")
    if extra_cols:
        notes.append(f"+{extra_cols} more cols")
    if notes:
        table.caption = "showing " + ", ".join(notes)
    return table


def _cell_str(value: Any) -> str:
    text = str(value)
    return text if len(text) <= 40 else text[:37] + "…"


def _decode_data_url(url: str) -> bytes | None:
    """Decode a ``data:image/...;base64,<...>`` URL to raw bytes (None if malformed)."""
    marker = "base64,"
    idx = url.find(marker)
    if not url.startswith("data:image/") or idx == -1:
        return None
    try:
        return base64.b64decode(url[idx + len(marker) :])
    except (ValueError, binascii.Error):
        return None


def _single_image(cell: CellView) -> str | None:
    """Return the data URL when the cell's output is exactly one image, else None."""
    if cell.error or cell.stream_text:
        return None
    candidates = [o for o in (*cell.display_outputs, *cell.outputs) if isinstance(o, dict)]
    if len(candidates) != 1:
        return None
    output = candidates[0]
    if output.get("content_type") == "image/png" and isinstance(output.get("inline_data_url"), str):
        return output["inline_data_url"]
    return None


def _image_renderable(cell: CellView) -> Any | None:
    """A terminal-image renderable for a single-image cell, or None to fall back.

    ``TerminalImage`` picks the terminal's best graphics protocol (kitty / iTerm /
    Sixel) and degrades to Unicode half-blocks when none is available.
    """
    url = _single_image(cell)
    if url is None:
        return None
    raw = _decode_data_url(url)
    if raw is None:
        return None
    try:
        # auto/auto preserves aspect ratio and uses as much of the container as
        # possible — so the same renderable fits the panel inline and scales up
        # in the full-screen image view.
        return TerminalImage(PILImage.open(io.BytesIO(raw)), width="auto", height="auto")
    except (OSError, ValueError):  # not a decodable image
        return None


def _render_outputs(cell: CellView) -> str:
    """Best-effort text rendering of a cell's outputs (rich types → 'open in Vue')."""
    if cell.error:
        return f"[error]\n{cell.error}"

    parts: list[str] = []
    for output in cell.display_outputs:
        content_type = str(output.get("content_type") or "")
        if content_type == "text/markdown" and isinstance(output.get("markdown_text"), str):
            parts.append(output["markdown_text"])
        elif content_type == "image/png":
            parts.append("[image/png — open in the web UI to view]")
        elif output.get("preview") is not None:
            parts.append(str(output["preview"]))
        elif content_type:
            parts.append(f"[{content_type}]")

    if cell.stream_text:
        parts.append(cell.stream_text)

    for output in cell.outputs:
        if not isinstance(output, dict):
            continue
        name = output.get("name") or output.get("variable") or "?"
        content_type = output.get("content_type") or ""
        preview = output.get("preview")
        summary = f"{name}: {content_type}"
        if preview is not None:
            summary += f" = {preview}"
        parts.append(summary)

    return "\n\n".join(parts) if parts else "(no output)"
