"""Read-only Textual app — Phase 0 spike for the notebook TUI.

The app:

1. ``POST /v1/notebooks/open`` to resolve the notebook path → ``session_id``
   and the initial state snapshot.
2. Connects ``ws://.../v1/notebooks/ws/{session_id}``, sends
   ``notebook_sync``, and renders the resulting ``notebook_state``.
3. Listens for ``cell_status`` / ``cell_console`` / ``cell_output`` /
   ``cell_error`` frames and updates the relevant cell's row + detail
   panel.

No editing, no run keybindings, no cascade handling — those land in
Phase 1. This spike validates that the protocol surface documented in
``docs/reference/notebook-protocol.md`` works for a non-Vue client.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import websockets
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, RichLog, Static

# Status glyphs map onto ``CellStatus`` values from the backend. ``?`` is
# the placeholder we show before the first ``cell_status`` frame arrives
# (or for cells the snapshot doesn't carry a status for — e.g. markdown).
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


class NotebookTUI(App[None]):
    """Top-level Textual app.

    Layout:

    - Header (notebook name + connection state)
    - Left: cell list (id, status glyph, first line of source)
    - Right top: source of the highlighted cell
    - Right middle: latest output payload (cell_output or cell_error)
    - Right bottom: console log buffer for the highlighted cell
    - Footer (key bindings)
    """

    CSS = """
    #cells {
        width: 40%;
        border: solid $primary;
    }
    #detail {
        width: 60%;
    }
    #source, #output {
        border: solid $primary;
        padding: 0 1;
        height: 1fr;
    }
    #console {
        border: solid $primary;
        height: 1fr;
    }
    .panel-title {
        background: $primary;
        color: $text;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Resync"),
    ]

    def __init__(
        self,
        *,
        notebook_path: Path,
        server_url: str,
        auth_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.notebook_path = notebook_path
        self.server_url = server_url
        self.auth_headers = auth_headers or {}
        self.session_id: str | None = None
        # cell_id → serialized cell dict (matches ``serialize_cell`` shape).
        self.cells_by_id: dict[str, dict[str, Any]] = {}
        # Order of cells, mirrored from the snapshot.
        self.cell_order: list[str] = []
        # cell_id → console buffer (stdout + stderr interleaved, like Vue).
        self.console_buffers: dict[str, str] = {}
        # cell_id → latest cell_output or cell_error payload.
        self.latest_output: dict[str, dict[str, Any]] = {}
        # cell_id of the currently highlighted row in the cell list.
        self.selected_cell: str | None = None
        # Connection state shown in the header subtitle.
        self.connection_state: str = "connecting…"

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield DataTable(id="cells", cursor_type="row", zebra_stripes=True)
            with Vertical(id="detail"):
                yield Static("Source", classes="panel-title")
                yield Static("", id="source", expand=True)
                yield Static("Output", classes="panel-title")
                yield Static("", id="output", expand=True)
                yield Static("Console", classes="panel-title")
                yield RichLog(id="console", highlight=False, markup=False)
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def on_mount(self) -> None:
        table = self.query_one("#cells", DataTable)
        table.add_columns("id", "status", "source")
        self._update_header()

        try:
            await self._open_notebook()
        except Exception as exc:  # noqa: BLE001 — exit cleanly on any open failure
            self.exit(message=f"Failed to open notebook: {exc}")
            return

        # WS loop runs as a Textual worker so its cancellation hooks into
        # the app shutdown lifecycle automatically.
        self.run_worker(self._ws_loop(), name="ws-loop", exclusive=True)

    async def action_refresh(self) -> None:
        """Resend ``notebook_sync`` to force a fresh ``notebook_state``."""
        ws = getattr(self, "_ws", None)
        if ws is None:
            return
        await ws.send(
            json.dumps({"type": "notebook_sync", "seq": 0, "ts": _utc_iso_z(), "payload": {}})
        )

    # ------------------------------------------------------------------
    # REST + WS plumbing
    # ------------------------------------------------------------------

    async def _open_notebook(self) -> None:
        """POST /v1/notebooks/open and seed the in-memory state."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.server_url}/v1/notebooks/open",
                json={"path": str(self.notebook_path)},
                headers=self.auth_headers,
            )
            response.raise_for_status()
            data = response.json()
            self.session_id = data["session_id"]
            self.sub_title = data.get("name") or self.notebook_path.name
            self._apply_state(data)

    async def _ws_loop(self) -> None:
        """Open the WS, send notebook_sync, dispatch frames forever."""
        assert self.session_id is not None
        ws_url = (
            self.server_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
            + f"/v1/notebooks/ws/{self.session_id}"
        )
        try:
            async with websockets.connect(
                ws_url,
                additional_headers=self.auth_headers or None,
            ) as ws:
                self._ws = ws
                self.connection_state = "connected"
                self._update_header()

                await ws.send(
                    json.dumps(
                        {"type": "notebook_sync", "seq": 1, "ts": _utc_iso_z(), "payload": {}}
                    )
                )
                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    self._handle_message(msg)
        except Exception as exc:  # noqa: BLE001 — log + surface; don't crash the app
            self.connection_state = f"disconnected ({exc.__class__.__name__})"
            self._update_header()

    def _handle_message(self, msg: dict[str, Any]) -> None:
        """Route a single WS frame into a state update."""
        msg_type = msg.get("type")
        payload = msg.get("payload") or {}

        if msg_type == "notebook_state":
            self._apply_state(payload)
            return

        cell_id = payload.get("cell_id")
        if not isinstance(cell_id, str):
            return

        if msg_type == "cell_status":
            cell = self.cells_by_id.get(cell_id)
            if cell is not None:
                cell["status"] = payload.get("status", cell.get("status"))
                self._refresh_cell_row(cell_id)
        elif msg_type == "cell_console":
            stream = payload.get("stream", "stdout")
            text = payload.get("text", "")
            prefix = "" if stream == "stdout" else "[stderr] "
            buffer = self.console_buffers.get(cell_id, "")
            self.console_buffers[cell_id] = buffer + prefix + text
            if cell_id == self.selected_cell:
                console = self.query_one("#console", RichLog)
                console.write(prefix + text.rstrip("\n"))
        elif msg_type in ("cell_output", "cell_error"):
            self.latest_output[cell_id] = {"type": msg_type, "payload": payload}
            if cell_id == self.selected_cell:
                self._refresh_output_panel()

    # ------------------------------------------------------------------
    # State → widget projection
    # ------------------------------------------------------------------

    def _apply_state(self, state: dict[str, Any]) -> None:
        """Replace the in-memory snapshot from a fresh notebook_state."""
        cells = state.get("cells") or []
        self.cells_by_id = {cell["id"]: cell for cell in cells if "id" in cell}
        self.cell_order = [cell["id"] for cell in cells if "id" in cell]

        table = self.query_one("#cells", DataTable)
        table.clear()
        for cell_id in self.cell_order:
            cell = self.cells_by_id[cell_id]
            table.add_row(
                _shorten_cell_id(cell_id),
                _render_status(cell.get("status")),
                _summary_line(cell.get("source") or ""),
                key=cell_id,
            )

        # Preselect the first cell so the detail panels aren't blank.
        if self.cell_order and self.selected_cell not in self.cell_order:
            self.selected_cell = self.cell_order[0]
        self._refresh_detail_panels()

    def _refresh_cell_row(self, cell_id: str) -> None:
        if cell_id not in self.cells_by_id:
            return
        table = self.query_one("#cells", DataTable)
        try:
            row_index = self.cell_order.index(cell_id)
        except ValueError:
            return
        cell = self.cells_by_id[cell_id]
        table.update_cell_at((row_index, 1), _render_status(cell.get("status")))

    def _refresh_detail_panels(self) -> None:
        self.query_one("#source", Static).update(self._source_view())
        self._refresh_output_panel()
        self._refresh_console_panel()

    def _refresh_output_panel(self) -> None:
        self.query_one("#output", Static).update(self._output_view())

    def _refresh_console_panel(self) -> None:
        console = self.query_one("#console", RichLog)
        console.clear()
        if self.selected_cell is None:
            return
        buffer = self.console_buffers.get(self.selected_cell, "")
        for line in buffer.splitlines():
            console.write(line)

    def _source_view(self) -> str:
        if self.selected_cell is None:
            return "(no cell selected)"
        cell = self.cells_by_id.get(self.selected_cell)
        if cell is None:
            return "(cell removed)"
        source = cell.get("source") or "(empty cell)"
        return source

    def _output_view(self) -> str:
        if self.selected_cell is None:
            return ""
        record = self.latest_output.get(self.selected_cell)
        if record is None:
            cell = self.cells_by_id.get(self.selected_cell, {})
            outputs = cell.get("outputs") or {}
            if outputs:
                return _format_outputs_dict(outputs)
            return "(no output yet)"
        payload = record["payload"]
        if record["type"] == "cell_error":
            return f"[error]\n{payload.get('error', '(no detail)')}"
        outputs = payload.get("outputs") or {}
        if outputs:
            return _format_outputs_dict(outputs)
        display = payload.get("display")
        if display:
            return _format_display(display)
        return "(no output)"

    def _update_header(self) -> None:
        # Header reads ``title`` and ``sub_title``; we encode the connection
        # state into the subtitle so users can tell at a glance whether the
        # WS is live.
        if self.sub_title:
            self.sub_title = self.sub_title.split(" — ", 1)[0]
            self.sub_title = f"{self.sub_title} — {self.connection_state}"

    # Wire DataTable row highlight → detail panel refresh.
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        cell_id = event.row_key.value
        if cell_id and cell_id != self.selected_cell:
            self.selected_cell = cell_id
            self._refresh_detail_panels()


# ---------------------------------------------------------------------------
# Pure helpers (kept module-level so they're cheap to test in isolation)
# ---------------------------------------------------------------------------


def _shorten_cell_id(cell_id: str) -> str:
    """Cells use 8-char UUID prefixes already; pass through verbatim."""
    return cell_id


def _render_status(status: str | None) -> str:
    """Map a CellStatus string to a single-character glyph for the table."""
    if not status:
        return " "
    return f"{_STATUS_GLYPHS.get(status, '?')} {status}"


def _summary_line(source: str) -> str:
    """First non-blank line of a cell's source, truncated for the table."""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped if len(stripped) <= 80 else stripped[:77] + "…"
    return "(empty)"


def _format_outputs_dict(outputs: dict[str, Any]) -> str:
    """Render the ``cell_output.outputs`` (variable → preview) mapping."""
    lines: list[str] = []
    for var, preview in outputs.items():
        if isinstance(preview, dict):
            text = preview.get("text") or preview.get("preview") or str(preview)
        else:
            text = str(preview)
        lines.append(f"{var} = {text}")
    return "\n".join(lines)


def _format_display(display: dict[str, Any]) -> str:
    """Render a display payload (image/text/markdown) as text-only summary."""
    content_type = display.get("content_type", "?")
    if content_type == "text/markdown":
        return display.get("markdown_text") or "(markdown — no inline body)"
    if content_type == "image/png":
        return "(image/png — open in Vue to view)"
    text = display.get("text") or display.get("preview")
    if text:
        return str(text)
    return f"(display: {content_type})"
