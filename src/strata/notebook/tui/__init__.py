"""Textual TUI client for the Strata Notebook backend.

Phase 0 spike: read-only viewer. Opens a notebook, connects via WebSocket,
renders cell list + status + outputs. No editing, no run keybindings —
those land in Phase 1 (see issue #37).

Distributed via the ``[tui]`` extra; entrypoint ``strata-notebook-tui``.
The wire protocol it consumes is documented in
``docs/reference/notebook-protocol.md``.
"""

from __future__ import annotations

from strata.notebook.tui.cli import main

__all__ = ["main"]
