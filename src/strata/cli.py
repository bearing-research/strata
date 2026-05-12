"""Top-level ``strata`` command dispatcher.

Subcommands:
    run     Execute a notebook headlessly (see :mod:`strata.notebook.cli`)
    export  Render a notebook to markdown or HTML for sharing

The existing ``strata-server`` script and ``python -m strata`` entry
points still start the server; they predate this CLI and stay as-is
for back-compat.
"""

from __future__ import annotations

import argparse
import sys

from strata.notebook.cli import add_export_arguments, add_run_arguments, export_main
from strata.notebook.cli import run_main as _run_main_direct


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="strata",
        description="Strata command-line tools.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    run_parser = subparsers.add_parser(
        "run",
        help="Execute a notebook directory headlessly",
        description="Execute every cell in a Strata notebook directory.",
    )
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=_dispatch_run)

    export_parser = subparsers.add_parser(
        "export",
        help="Render a notebook to markdown or HTML",
        description=(
            "Render a Strata notebook directory to a single shareable file. "
            "Source cells, cached display outputs, and console snapshots are "
            "included; prompt-cell responses are intentionally excluded."
        ),
    )
    add_export_arguments(export_parser)
    export_parser.set_defaults(func=_dispatch_export)

    return parser


def _dispatch_run(args: argparse.Namespace) -> int:
    # Re-enter the run command's async runner without re-parsing — we
    # already have a populated namespace from the top-level parser.
    import asyncio

    from strata.notebook.cli import _run_async

    return asyncio.run(_run_async(args))


def _dispatch_export(args: argparse.Namespace) -> int:
    return export_main(args)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    return args.func(args)


def run_main(argv: list[str] | None = None) -> int:
    """Shim for a direct ``strata-run`` entry point, if we ever add one."""
    return _run_main_direct(argv)


if __name__ == "__main__":
    sys.exit(main())
