"""Top-level ``strata`` command dispatcher.

Subcommands:
    run       Execute a notebook headlessly (see :mod:`strata.notebook.cli`)
    validate  Static checks (schema, annotations, DAG) without executing
    new       Scaffold a notebook directory
    export    Render a notebook to markdown or HTML for sharing
    import    Convert a Jupyter .ipynb file into a Strata notebook directory

The existing ``strata-notebook`` script and ``python -m strata`` entry
points still start the server; they predate this CLI and stay as-is
for back-compat.
"""

from __future__ import annotations

import argparse
import sys

from strata.notebook.cli import (
    add_export_arguments,
    add_import_arguments,
    add_new_arguments,
    add_run_arguments,
    add_validate_arguments,
    export_main,
    import_main,
    new_main,
    validate_main,
)
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

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a notebook directory without executing it",
        description=(
            "Static checks for a Strata notebook: notebook.toml parses, the "
            "DAG builds without cycles, and per-cell annotations pass the "
            "same validation the server runs on open. Nothing executes and "
            "no environment is synced. Exit 0 valid (warnings allowed), "
            "1 invalid, 2 invocation error."
        ),
    )
    add_validate_arguments(validate_parser)
    validate_parser.set_defaults(func=_dispatch_validate)

    new_parser = subparsers.add_parser(
        "new",
        help="Scaffold a new notebook directory",
        description=(
            "Create a notebook directory (notebook.toml + pyproject.toml + "
            "cells/) ready for `strata validate` and `strata run`. Idempotent "
            "on an existing notebook directory: the notebook ID and existing "
            "cells are preserved."
        ),
    )
    add_new_arguments(new_parser)
    new_parser.set_defaults(func=_dispatch_new)

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

    import_parser = subparsers.add_parser(
        "import",
        help="Convert a Jupyter .ipynb file into a Strata notebook directory",
        description=(
            "Parse a Jupyter notebook and produce an equivalent Strata "
            "notebook directory. Cells are converted in source order; "
            "Jupyter's trailing-';' display-suppression convention is "
            "preserved. Magics, shell commands, and dependency capture "
            "are not yet implemented."
        ),
    )
    add_import_arguments(import_parser)
    import_parser.set_defaults(func=_dispatch_import)

    return parser


def _dispatch_run(args: argparse.Namespace) -> int:
    # Re-enter the run command's async runner without re-parsing — we
    # already have a populated namespace from the top-level parser.
    import asyncio

    from strata.notebook.cli import _run_async

    return asyncio.run(_run_async(args))


def _dispatch_validate(args: argparse.Namespace) -> int:
    return validate_main(args)


def _dispatch_new(args: argparse.Namespace) -> int:
    return new_main(args)


def _dispatch_export(args: argparse.Namespace) -> int:
    return export_main(args)


def _dispatch_import(args: argparse.Namespace) -> int:
    return import_main(args)


def main(argv: list[str] | None = None) -> int:
    from strata._uv_runtime import assert_uv_managed_runtime

    assert_uv_managed_runtime()

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
