"""Command-line entry point for the Strata Notebook TUI spectator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Top-level modules the [tui] extra installs.
_TUI_EXTRA_MODULES = frozenset({"textual", "textual_image", "PIL", "grandalf"})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strata-notebook-tui",
        description="Read-only terminal spectator for a live Strata notebook session.",
    )
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--session",
        metavar="ID",
        help="Attach to a specific running session id (skips the picker).",
    )
    target.add_argument(
        "--notebook",
        type=Path,
        metavar="PATH",
        help="Open/reuse a notebook by directory path (the same path POST /open accepts).",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("STRATA_TUI_SERVER", "http://localhost:8765"),
        help="Base URL of the strata-notebook server (default: $STRATA_TUI_SERVER or :8765).",
    )
    return parser


def run_spectator(
    *,
    server: str,
    session: str | None = None,
    notebook: str | Path | None = None,
) -> None:
    """Launch the read-only Textual spectator against a running server.

    Shared with ``strata watch``. With neither *session* nor *notebook*, attaches
    the caller's only running session or shows a picker.
    """
    # Lazy import so ``--help`` doesn't pay the Textual import cost.
    try:
        from strata.notebook.tui.app import NotebookTUI
    except ModuleNotFoundError as exc:
        if (exc.name or "").partition(".")[0] not in _TUI_EXTRA_MODULES:
            raise
        raise SystemExit(
            f"error: the terminal viewer needs the [tui] extra (no module {exc.name!r}); "
            "in a checkout, `uv sync --extra tui`; for an installed tool, "
            "`uv tool install 'strata-notebook[tui]'`"
        ) from exc
    from strata.notebook.tui.client import TuiClient

    client = TuiClient(server_url=server)
    notebook_path = str(Path(notebook).expanduser().resolve()) if notebook else None

    NotebookTUI(client=client, session_id=session, notebook_path=notebook_path).run()


def main(argv: list[str] | None = None) -> None:
    """Parse args and launch the Textual spectator."""
    args = _build_parser().parse_args(argv)
    run_spectator(
        server=args.server,
        session=args.session,
        notebook=args.notebook,
    )
