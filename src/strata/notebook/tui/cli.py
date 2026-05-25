"""Command-line entrypoint for the Strata Notebook TUI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """Parse CLI args and launch the Textual app.

    The single positional argument is the notebook directory path; the
    backend resolves it via ``POST /v1/notebooks/open``. ``--server`` is
    the HTTP(S) base URL of the strata-notebook server (the WebSocket
    URL is derived by swapping the scheme).
    """
    parser = argparse.ArgumentParser(
        prog="strata-notebook-tui",
        description="Read-only Textual viewer for a Strata notebook.",
    )
    parser.add_argument(
        "notebook",
        type=Path,
        help="Path to a notebook directory (the same path POST /open accepts).",
    )
    parser.add_argument(
        "--server",
        default=os.environ.get("STRATA_TUI_SERVER", "http://localhost:8765"),
        help=(
            "Base URL of the strata-notebook server. "
            "Default: $STRATA_TUI_SERVER or http://localhost:8765."
        ),
    )
    parser.add_argument(
        "--user-header",
        default=os.environ.get("STRATA_TUI_USER_HEADER_NAME"),
        help=(
            "Name of the identity header to send (matches the server's "
            "``personal_mode_user_header`` config). Optional; only needed "
            "when the server runs behind an auth proxy."
        ),
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("STRATA_TUI_USER"),
        help=(
            "Value for the identity header. Must be set whenever the server "
            "configures a user header and the notebook is owned."
        ),
    )
    args = parser.parse_args(argv)

    # Import the app lazily so ``--help`` doesn't pay the Textual import cost.
    from strata.notebook.tui.app import NotebookTUI

    headers: dict[str, str] = {}
    if args.user_header and args.user:
        headers[args.user_header] = args.user

    app = NotebookTUI(
        notebook_path=args.notebook.expanduser().resolve(),
        server_url=args.server.rstrip("/"),
        auth_headers=headers,
    )
    app.run()
