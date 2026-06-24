"""Read-only Textual spectator app for a live notebook session (TUI Phase 1, M1).

Resolves a session (flag, path, or interactive picker), opens the WS, sends
``notebook_sync``, and renders the resulting ``notebook_state`` plus the live
``cell_status`` / ``cell_console`` / ``cell_output`` stream. No editing, no run
keybindings — a spectator. The cascade/dag/env frames (M2) and the agent panel
(M3) build on this same dispatch loop.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import websockets
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Footer, Header, OptionList, Static
from textual.widgets.option_list import Option

from strata.notebook.tui.client import TuiClient, TuiClientError
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


def _first_line(source: str) -> str:
    for line in source.splitlines():
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


class NotebookTUI(App[None]):
    """Top-level spectator app."""

    CSS = """
    #cells { width: 38%; border: solid $primary; }
    #detail { width: 62%; }
    /* Each panel is a scroll region that fills 1/3 of the detail pane; the inner
       content is height:auto so it grows past the region and the scrollbar
       activates (a 1fr child would fill the region exactly and never scroll). */
    .scroll-panel { height: 1fr; border: solid $primary; }
    #source, #output, #console-body { height: auto; width: 1fr; padding: 0 1; }
    .panel-title { background: $primary; color: $text; padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Resync"),
    ]

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

    # -- layout --------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield DataTable(id="cells", cursor_type="row", zebra_stripes=True)
            with Vertical(id="detail"):
                yield Static("Source", classes="panel-title")
                with VerticalScroll(classes="scroll-panel"):
                    yield Static("", id="source")
                yield Static("Output", classes="panel-title")
                with VerticalScroll(classes="scroll-panel"):
                    yield Static("", id="output")
                yield Static("Console", classes="panel-title")
                with VerticalScroll(classes="scroll-panel"):
                    yield Static("", id="console-body")
        yield Footer()

    # -- lifecycle -----------------------------------------------------------

    async def on_mount(self) -> None:
        table = self.query_one("#cells", DataTable)
        table.add_columns(" ", "cell")
        self._set_connection("connecting…")
        self.run_worker(self._bootstrap(), name="bootstrap", exclusive=True)

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
            return
        changed = self.vm.apply_frame(msg_type, payload)
        for cid in changed:
            self._refresh_cell(cid)

    # -- rendering -----------------------------------------------------------

    def _set_connection(self, state: str) -> None:
        self.title = self.vm.notebook_name or "Strata Notebook"
        self.sub_title = state

    def _rebuild_cells(self) -> None:
        table = self.query_one("#cells", DataTable)
        table.clear()
        for cid in self.vm.cell_order:
            cell = self.vm.cells[cid]
            table.add_row(_glyph(cell.status), self._cell_label(cell), key=cid)
        self._set_connection("connected")
        if self.vm.cell_order:
            if self._selected not in self.vm.cells:
                self._selected = self.vm.cell_order[0]
            self._show_detail(self._selected)

    def _cell_label(self, cell: CellView) -> str:
        name = cell.name or cell.id[:8]
        return f"{name}  {_first_line(cell.source)[:40]}"

    def _refresh_cell(self, cid: str) -> None:
        cell = self.vm.cells.get(cid)
        if cell is None:
            return
        table = self.query_one("#cells", DataTable)
        try:
            table.update_cell(cid, " ", _glyph(cell.status))
            table.update_cell(cid, "cell", self._cell_label(cell))
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
        self.query_one("#source", Static).update(cell.source or "(empty)")
        self.query_one("#output", Static).update(_render_outputs(cell))
        self.query_one("#console-body", Static).update(cell.console or "(no console output)")


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
