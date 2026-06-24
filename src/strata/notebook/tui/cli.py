"""Command-line entry point for the Strata Notebook TUI spectator."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


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
    parser.add_argument(
        "--user-header",
        default=os.environ.get("STRATA_TUI_USER_HEADER_NAME"),
        help="Identity header name (matches the server's personal_mode_user_header).",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("STRATA_TUI_USER"),
        help="Identity header value (needed when the notebook is owned).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Parse args and launch the Textual spectator.

    With no ``--session`` / ``--notebook``, the app lists the caller's running
    sessions and (auto-attaches the only one, else) shows an interactive picker.
    """
    args = _build_parser().parse_args(argv)

    # Lazy import so ``--help`` doesn't pay the Textual import cost.
    from strata.notebook.tui.app import NotebookTUI
    from strata.notebook.tui.client import TuiClient

    headers: dict[str, str] = {}
    if args.user_header and args.user:
        headers[args.user_header] = args.user

    client = TuiClient(server_url=args.server, auth_headers=headers)
    notebook_path = str(args.notebook.expanduser().resolve()) if args.notebook else None

    NotebookTUI(
        client=client,
        session_id=args.session,
        notebook_path=notebook_path,
    ).run()
