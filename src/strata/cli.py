"""Top-level ``strata`` command dispatcher.

Subcommands:
    run       Execute a notebook headlessly (see :mod:`strata.notebook.cli`)
    validate  Static checks (schema, annotations, DAG) without executing
    new       Scaffold a notebook directory
    export    Render a notebook to markdown or HTML for sharing
    import    Convert a Jupyter .ipynb file into a Strata notebook directory
    artifact  Artifact store maintenance (verify)

The existing ``strata-notebook`` script and ``python -m strata`` entry
points still start the server; they predate this CLI and stay as-is
for back-compat.
"""

from __future__ import annotations

import argparse
import os
import sys

from strata.notebook.agent_launch import add_agent_arguments, agent_main
from strata.notebook.cli import (
    add_cell_arguments,
    add_dag_arguments,
    add_dep_arguments,
    add_export_arguments,
    add_import_arguments,
    add_new_arguments,
    add_run_arguments,
    add_status_arguments,
    add_validate_arguments,
    add_worker_arguments,
    dag_main,
    export_main,
    import_main,
    new_main,
    status_main,
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

    migrate_parser = subparsers.add_parser(
        "migrate",
        help="Copy an existing SQLite artifact store onto Postgres",
        description=(
            "Copy artifact-store metadata from SQLite to Postgres. Setting "
            "STRATA_ARTIFACT_METADATA_DSN alone starts an empty store; this "
            "carries an existing one across. Metadata only -- blobs are "
            "configured separately and are not copied."
        ),
    )
    migrate_parser.add_argument(
        "--artifact-dir",
        dest="artifact_dir",
        default=None,
        help="Source artifact store directory (default: ~/.strata/artifacts)",
    )
    migrate_parser.add_argument(
        "--to-dsn",
        dest="to_dsn",
        default=None,
        help="Target postgresql:// DSN (default: STRATA_ARTIFACT_METADATA_DSN)",
    )
    migrate_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be copied and exit without writing",
    )
    migrate_parser.add_argument(
        "--allow-nonempty-target",
        action="store_true",
        help=(
            "Write into a target that already holds rows. Safe for resuming an "
            "interrupted migration (the copy skips rows already present); unsafe "
            "for merging two different stores."
        ),
    )
    migrate_parser.set_defaults(func=_dispatch_migrate)

    key_parser = subparsers.add_parser(
        "apikey",
        help="Mint, list, and revoke API keys",
        description=(
            "Manage API keys for auth_mode='api_key'. Keys are stored in the "
            "artifact store's database, so they follow whichever metadata "
            "backend it uses and are shared across nodes."
        ),
    )
    key_sub = key_parser.add_subparsers(dest="apikey_command", metavar="<command>")

    def _add_key_store_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--artifact-dir",
            dest="artifact_dir",
            default=None,
            help="Artifact store directory (default: ~/.strata/artifacts)",
        )
        sub.add_argument(
            "--dsn",
            default=None,
            help=(
                "Postgres DSN for the metadata backend. Defaults to "
                "STRATA_ARTIFACT_METADATA_DSN, then SQLite under --artifact-dir."
            ),
        )

    key_create = key_sub.add_parser(
        "create",
        help="Mint a new API key",
        description="Mint a key. The secret is printed once and is not recoverable.",
    )
    key_create.add_argument("principal", help="Principal id the key authenticates as")
    key_create.add_argument("--tenant", default=None, help="Tenant to scope the key to")
    key_create.add_argument(
        "--scope",
        action="append",
        dest="scopes",
        default=None,
        help="Grant a scope (repeatable), e.g. --scope artifacts:write",
    )
    key_create.add_argument("--description", default=None, help="What this key is for")
    key_create.add_argument(
        "--expires-in-days",
        type=float,
        default=None,
        help="Expire the key after this many days (default: no expiry)",
    )
    _add_key_store_args(key_create)
    key_create.set_defaults(func=_dispatch_apikey("cmd_create"))

    key_list = key_sub.add_parser("list", help="List API keys (never shows secrets)")
    key_list.add_argument("--principal", default=None, help="Filter to one principal")
    key_list.add_argument(
        "--format", choices=["human", "json"], default="human", help="Output format"
    )
    _add_key_store_args(key_list)
    key_list.set_defaults(func=_dispatch_apikey("cmd_list"))

    key_revoke = key_sub.add_parser("revoke", help="Revoke an API key by id")
    key_revoke.add_argument("key_id", help="Key id (the middle segment of the key)")
    _add_key_store_args(key_revoke)
    key_revoke.set_defaults(func=_dispatch_apikey("cmd_revoke"))

    artifact_parser = subparsers.add_parser(
        "artifact",
        help="Artifact store inspection and maintenance",
        description="Inspect and maintain a Strata artifact store (no server needed).",
    )
    artifact_sub = artifact_parser.add_subparsers(dest="artifact_command", metavar="<command>")

    def _add_artifact_dir_arg(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--artifact-dir",
            dest="artifact_dir",
            default=None,
            help="Artifact store directory (default: ~/.strata/artifacts)",
        )

    def _add_store_args(sub: argparse.ArgumentParser) -> None:
        _add_artifact_dir_arg(sub)
        sub.add_argument(
            "--format",
            choices=["human", "json"],
            default="human",
            help="Output format (default: human)",
        )

    def _add_tenant_arg(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--tenant",
            default=None,
            help=(
                "Scope name/alias resolution to this tenant. Without it, a name "
                "shared by multiple tenants is reported as ambiguous (service-mode "
                "stores)."
            ),
        )

    list_parser = artifact_sub.add_parser(
        "list",
        help="List artifacts in the store",
        description="List artifacts: id, version, state, rows, size, names.",
    )
    _add_store_args(list_parser)
    list_parser.add_argument("--state", default=None, help="Filter by state (ready/failed/…)")
    list_parser.add_argument("--limit", type=int, default=50, help="Max rows (default 50)")
    list_parser.set_defaults(func=_dispatch_artifact("cmd_list"))

    show_parser = artifact_sub.add_parser(
        "show",
        help="Show one artifact's metadata, names, and inputs",
        description=(
            "Show an artifact. <ref> is a name pointer, an id@v=N, or a "
            "bare artifact id (latest version)."
        ),
    )
    show_parser.add_argument("ref", help="Name, id@v=N, or artifact id")
    _add_store_args(show_parser)
    _add_tenant_arg(show_parser)
    show_parser.set_defaults(func=_dispatch_artifact("cmd_show"))

    lineage_parser = artifact_sub.add_parser(
        "lineage",
        help="Walk an artifact's provenance upstream to tables/snapshots",
        description=(
            "Render the provenance chain, e.g. model <- features <- scan "
            "<- table @ snapshot. <ref> as in `show`."
        ),
    )
    lineage_parser.add_argument("ref", help="Name, id@v=N, or artifact id")
    _add_store_args(lineage_parser)
    lineage_parser.add_argument(
        "--max-depth", type=int, default=10, help="Recursion limit (default 10)"
    )
    _add_tenant_arg(lineage_parser)
    lineage_parser.set_defaults(func=_dispatch_artifact("cmd_lineage"))

    pull_parser = artifact_sub.add_parser(
        "pull",
        help="Write an artifact's blob to a local Arrow IPC file",
        description="Pull an artifact's data. <ref> as in `show`.",
    )
    pull_parser.add_argument("ref", help="Name, id@v=N, or artifact id")
    _add_artifact_dir_arg(pull_parser)
    pull_parser.add_argument("--to", default=None, help="Output path (default <ref>.arrow)")
    _add_tenant_arg(pull_parser)
    pull_parser.set_defaults(func=_dispatch_artifact("cmd_pull"))

    audit_parser = artifact_sub.add_parser(
        "audit",
        help="Show the registry audit: every name/alias/tag mutation",
        description=(
            "Append-only history of name, alias, and tag mutations — "
            'answers "what did this name point to before?". Newest first.'
        ),
    )
    audit_parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Filter to one registry name (optional)",
    )
    _add_artifact_dir_arg(audit_parser)
    audit_parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    audit_parser.add_argument("--limit", type=int, default=50, help="Max entries (default 50)")
    audit_parser.set_defaults(func=_dispatch_artifact("cmd_audit"))

    pending_parser = artifact_sub.add_parser(
        "pending",
        help="List protected-alias changes awaiting approval",
        description=(
            "Protected aliases (registry_protected_aliases config) queue "
            "their changes for approval; this lists the queue."
        ),
    )
    _add_artifact_dir_arg(pending_parser)
    pending_parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    pending_parser.set_defaults(func=_dispatch_artifact("cmd_pending"))

    verify_parser = artifact_sub.add_parser(
        "verify",
        help="Check every artifact blob against its metadata",
        description=(
            "For each ready/superseded artifact: the blob must exist, parse "
            "as exactly one Arrow IPC stream, and match the recorded row "
            "count. Reports inconsistencies; exit 0 clean, 1 problems found, "
            "2 invocation error."
        ),
    )
    _add_store_args(verify_parser)
    verify_parser.set_defaults(func=_dispatch_artifact("cmd_verify"))

    artifact_parser.set_defaults(func=lambda args: (artifact_parser.print_help(), 0)[1])

    # Agent inspect commands (CLI-hardening P0, NotebookOps local backend).
    cell_parser = subparsers.add_parser(
        "cell",
        help="Per-cell operations: list, show, run, test, add, edit, rm, mv, annotate",
        description="Inspect and (later) drive individual notebook cells.",
    )
    add_cell_arguments(cell_parser)

    dag_parser = subparsers.add_parser(
        "dag",
        help="Print a notebook's dependency graph (edges + topological order)",
    )
    add_dag_arguments(dag_parser)
    dag_parser.set_defaults(func=dag_main)

    status_parser = subparsers.add_parser(
        "status",
        help="Print per-cell status + staleness for a notebook",
    )
    add_status_arguments(status_parser)
    status_parser.set_defaults(func=status_main)

    dep_parser = subparsers.add_parser(
        "dep",
        help="Manage notebook dependencies (add, rm)",
        description="Add or remove a Python dependency in a notebook directory.",
    )
    add_dep_arguments(dep_parser)

    worker_parser = subparsers.add_parser(
        "worker",
        help="Manage notebook workers: ls, add, rm, default",
        description=(
            "Register notebook-scoped workers (e.g. a remote executor endpoint) "
            "and set the default. Writes notebook.toml; a running server picks "
            "the change up on its next session reload."
        ),
    )
    add_worker_arguments(worker_parser)

    agent_parser = subparsers.add_parser(
        "agent",
        help="Open a notebook for a coding agent to drive, with a live TUI",
        description=(
            "One-command on-ramp: create-or-open a notebook, start (or reuse) a "
            "server with the MCP endpoint enabled, open a session, write the "
            ".mcp.json + CLAUDE.md a coding agent needs, and attach the TUI. "
            "Then run `claude` in the notebook directory to drive it."
        ),
    )
    add_agent_arguments(agent_parser)
    agent_parser.set_defaults(func=_dispatch_agent)

    watch_parser = subparsers.add_parser(
        "watch",
        help="Attach the read-only TUI to a live notebook session (watch only)",
        description=(
            "Open the terminal spectator on a running notebook session, without "
            "writing any config. Unlike `agent`, it does not create `.mcp.json` / "
            "`CLAUDE.md` or take over the server lifecycle - it just shows a "
            "notebook live. Requires a running server (start one with `agent` or "
            "`python -m strata`)."
        ),
    )
    watch_target = watch_parser.add_mutually_exclusive_group()
    watch_target.add_argument(
        "notebook_dir",
        nargs="?",
        help="Notebook directory to open/reuse a session for (local path).",
    )
    watch_target.add_argument(
        "--session", metavar="ID", help="Attach to a specific running session id."
    )
    watch_parser.add_argument(
        "--server",
        default=os.environ.get("STRATA_TUI_SERVER", "http://localhost:8765"),
        help="Server base URL (default: $STRATA_TUI_SERVER or :8765).",
    )
    watch_parser.add_argument(
        "--user-header",
        default=os.environ.get("STRATA_TUI_USER_HEADER_NAME"),
        help="Identity header name (matches the server's personal_mode_user_header).",
    )
    watch_parser.add_argument(
        "--user",
        default=os.environ.get("STRATA_TUI_USER"),
        help="Identity header value (needed when the notebook is owned).",
    )
    watch_parser.set_defaults(func=_dispatch_watch)

    return parser


def _dispatch_artifact(command: str):
    """Build a dispatcher that lazily imports the artifact CLI module."""

    def _dispatch(args: argparse.Namespace) -> int:
        from strata import artifact_cli

        return getattr(artifact_cli, command)(args)

    return _dispatch


def _dispatch_migrate(args: argparse.Namespace) -> int:
    from strata import migrate_cli

    return migrate_cli.cmd_migrate(args)


def _dispatch_apikey(command: str):
    """Build a dispatcher that lazily imports the API key CLI module."""

    def _dispatch(args: argparse.Namespace) -> int:
        from strata import api_key_cli

        return getattr(api_key_cli, command)(args)

    return _dispatch


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


def _dispatch_agent(args: argparse.Namespace) -> int:
    return agent_main(args)


def _dispatch_watch(args: argparse.Namespace) -> int:
    from strata.notebook.tui.cli import run_spectator

    run_spectator(
        server=args.server,
        session=args.session,
        notebook=args.notebook_dir,
        user_header=args.user_header,
        user=args.user,
    )
    return 0


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
