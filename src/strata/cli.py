"""Top-level ``strata`` command dispatcher.

The ``strata-notebook`` script and ``python -m strata`` still start the server; this CLI
does not.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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
from strata.notebook.tutorial import add_tutorial_arguments, tutorial_main


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
        help="Render a notebook to markdown or HTML, or write a snapshot zip",
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
        help="Convert a Jupyter .ipynb, or unpack a snapshot zip, into a notebook directory",
        description=(
            "Parse a Jupyter notebook and produce an equivalent Strata "
            "notebook directory. Cells are converted in source order; "
            "Jupyter's trailing-';' display-suppression convention is "
            "preserved. Magics and shell commands are converted or dropped "
            "with a note in the cell, and dependencies are read from a "
            "requirements.txt or pyproject.toml beside the .ipynb."
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

    env_parser = subparsers.add_parser(
        "env",
        help="Manage shared notebook environments",
        description=(
            'Shared notebook environments (notebook_env_backend = "shared"): '
            "one environment per lockfile and interpreter, linked from each "
            "notebook's .venv."
        ),
    )
    env_sub = env_parser.add_subparsers(dest="env_command", required=True)
    env_gc = env_sub.add_parser(
        "gc",
        help="Remove shared environments no notebook links to",
        description=(
            "Remove shared environments that no notebook links to and that have "
            "gone unused for the TTL. An environment a notebook links to is kept."
        ),
    )
    env_gc.add_argument(
        "--dir",
        dest="env_dir",
        default=None,
        help="Shared environment directory (default: notebook_shared_env_dir)",
    )
    env_gc.add_argument(
        "--ttl-days",
        dest="ttl_days",
        type=float,
        default=None,
        help="Unused this long before removal (default: notebook_shared_env_ttl_days)",
    )
    env_gc.set_defaults(func=_dispatch_env_gc)

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
            help=(
                "Read this local store (its SQLite file and blobs) and nothing "
                "else. Without it, the server's configured store: [tool.strata] "
                "and STRATA_* settings, metadata DSN and blob backend included"
            ),
        )
        sub.add_argument(
            "--dsn",
            default=None,
            help=(
                "Postgres DSN for the metadata backend. Defaults to the "
                "configured one (STRATA_ARTIFACT_METADATA_DSN) unless "
                "--artifact-dir is given."
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
            help=(
                "Read this local store (its SQLite file and blobs) and nothing "
                "else. Without it, the server's configured store: [tool.strata] "
                "and STRATA_* settings, metadata DSN and blob backend included"
            ),
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
    list_parser.add_argument(
        "--state", default=None, help="Filter by state (ready, superseded, building, failed)"
    )
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

    publish_parser = artifact_sub.add_parser(
        "publish",
        help="Make an artifact readable by anyone with the link",
        description=(
            "Mint a public URL for one artifact version. The page shows the "
            "artifact, the code that produced it, and the code and environment "
            "of every step behind it -- publishing exposes the whole chain, "
            "which is the point and is listed back to you. <ref> as in `show`."
        ),
    )
    publish_parser.add_argument("ref", help="Name, id@v=N, or artifact id")
    _add_store_args(publish_parser)
    publish_parser.add_argument("--title", default=None, help="Human label for the page")
    # --into and --to both name the publish destination, so they exclude each
    # other; where to read from is the separate store args.
    publish_target = publish_parser.add_mutually_exclusive_group()
    publish_target.add_argument(
        "--into",
        default=None,
        metavar="DIR",
        help=(
            "Artifact store to publish into. Defaults to the one your server "
            "serves, so the link resolves; the artifact and its chain are "
            "copied there if they are not already."
        ),
    )
    publish_target.add_argument(
        "--to",
        dest="to_url",
        default=None,
        metavar="URL",
        help=(
            "Strata server to publish to, e.g. https://store.example. The "
            "chain is copied over HTTP and the link is minted there -- for a "
            "hosted deployment, the store that serves the link is never the "
            "machine that ran the cells. Auth from STRATA_STORE_TOKEN, or "
            "--header."
        ),
    )
    publish_parser.add_argument(
        "--header",
        action="append",
        metavar="'Name: value'",
        help=(
            "Extra header for --to, repeatable. STRATA_STORE_TOKEN covers the "
            "usual bearer token; this is for whatever else a proxy wants."
        ),
    )
    publish_target.add_argument(
        "--here",
        action="store_true",
        help=(
            "Publish into the store named by --artifact-dir rather than the "
            "server's. The link then only resolves if your server serves that "
            "same directory."
        ),
    )
    publish_parser.add_argument(
        "--author",
        default=None,
        help=(
            "Who to credit on the page. A local run has no authenticated "
            "identity, so without this the page names no author."
        ),
    )
    publish_parser.add_argument(
        "--max-depth", type=int, default=10, help="Recursion limit when listing the chain"
    )
    _add_tenant_arg(publish_parser)
    publish_parser.set_defaults(func=_dispatch_artifact("cmd_publish"))

    promote_parser = artifact_sub.add_parser(
        "promote",
        help="Copy an artifact and its chain to the team store, and name it there",
        description=(
            "Share a result with your team. Unlike `publish`, this mints no "
            "public link -- it puts the artifact where colleagues' cells "
            "already read from, under a name they can ask for. Everything "
            "behind it travels too: the team cache is keyed by provenance, so "
            "each ancestor that arrives is a cache hit for the next person "
            "whose cell computes the same thing."
        ),
    )
    promote_parser.add_argument("ref", help="Name, id@v=N, or artifact id")
    _add_store_args(promote_parser)
    promote_parser.add_argument(
        "--to",
        dest="to_url",
        required=True,
        metavar="URL",
        help="Team store to promote into, e.g. https://store.example",
    )
    promote_parser.add_argument("--name", required=True, help="Name colleagues will ask for it by")
    promote_parser.add_argument(
        "--alias",
        default=None,
        help=(
            "Also move this alias (e.g. champion) to the promoted version. A "
            "protected alias queues for approval instead of applying"
        ),
    )
    promote_parser.add_argument(
        "--tag",
        action="append",
        metavar="KEY=VALUE",
        help="Tag to set on the promoted version, repeatable",
    )
    promote_parser.add_argument(
        "--table",
        default=None,
        metavar="TABLE",
        help=(
            "Also have the team store write it into this Iceberg table "
            "(<warehouse>#ns.table, or ns.table in the store's catalog)"
        ),
    )
    promote_parser.add_argument(
        "--header",
        action="append",
        metavar="'Name: value'",
        help="Extra header for --to, repeatable. STRATA_STORE_TOKEN covers the bearer token",
    )
    promote_parser.add_argument(
        "--max-depth", type=int, default=10, help="Recursion limit when walking the chain"
    )
    _add_tenant_arg(promote_parser)
    promote_parser.set_defaults(func=_dispatch_artifact("cmd_promote"))

    export_table_parser = artifact_sub.add_parser(
        "export",
        help="Write a tabular artifact into an Iceberg table",
        description=(
            "Write an artifact into an Iceberg table as its current snapshot. "
            "The first write appends, a later one overwrites, and each "
            "snapshot's summary names the artifact version it holds. A new "
            "version may add columns; an incompatible schema is refused. A "
            "table Strata did not write is refused rather than replaced."
        ),
    )
    export_table_parser.add_argument("ref", help="Name, id@v=N, or artifact id")
    _add_store_args(export_table_parser)
    export_table_parser.add_argument(
        "--table",
        required=True,
        metavar="TABLE",
        help="<warehouse>#ns.table, or ns.table in the configured catalog",
    )
    export_table_parser.add_argument(
        "--alias", default=None, help="Tag the new snapshot with this name, e.g. champion"
    )
    export_table_parser.add_argument(
        "--by", default=None, help="Recorded as strata.promoted_by in the snapshot summary"
    )
    _add_tenant_arg(export_table_parser)
    export_table_parser.set_defaults(func=_dispatch_artifact("cmd_export_table"))

    unpublish_parser = artifact_sub.add_parser(
        "unpublish",
        help="Withdraw a published link",
        description=(
            "Revoke public access. The token is never reissued for other "
            "content, so a URL already in print fails closed rather than "
            "starting to resolve to something else."
        ),
    )
    unpublish_parser.add_argument("token", help="The publication token, or its id")
    _add_artifact_dir_arg(unpublish_parser)
    _add_tenant_arg(unpublish_parser)
    unpublish_parser.set_defaults(func=_dispatch_artifact("cmd_unpublish"))

    archive_parser = artifact_sub.add_parser(
        "archive",
        help="Write a self-contained bundle: page, bytes, manifest, README",
        description=(
            "Write a directory that opens with no server -- the page, the "
            "bytes, and a machine-readable manifest. A hosted link resolves "
            "for as long as the server does, and a URL printed in a paper "
            "outlives most servers; this is the copy for a Zenodo or OSF "
            "deposit. <ref> as in `show`, or --token for a publication, whose "
            "bundle then carries its authors and identifiers. A --to ending in "
            ".zip writes the zip GET /p/{token}/archive.zip serves."
        ),
    )
    archive_parser.add_argument("ref", nargs="?", help="Name, id@v=N, or artifact id")
    archive_parser.add_argument(
        "--token",
        default=None,
        help="Archive this publication (its token, or its id), as its record stands",
    )
    archive_parser.add_argument(
        "--to", required=True, help="Directory to write, or a path ending in .zip"
    )
    _add_artifact_dir_arg(archive_parser)
    archive_parser.add_argument("--title", default=None, help="Human label for the page")
    archive_parser.add_argument("--author", default=None, help="Who to credit on the page")
    archive_parser.add_argument(
        "--force",
        action="store_true",
        help="Write into a non-empty directory (may leave a stale payload beside the new one)",
    )
    archive_parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Recursion limit when walking the chain (default 10; with --token, the page's 25)",
    )
    _add_tenant_arg(archive_parser)
    archive_parser.set_defaults(func=_dispatch_artifact("cmd_archive"))

    pull_parser = artifact_sub.add_parser(
        "pull",
        help="Write an artifact's blob to a local Arrow IPC file, or a folder's files",
        description="Pull an artifact's data. <ref> as in `show`.",
    )
    pull_parser.add_argument("ref", help="Name, id@v=N, or artifact id")
    _add_artifact_dir_arg(pull_parser)
    pull_parser.add_argument(
        "--to", default=None, help="Output path (default <ref>.arrow, or <ref>/ for a folder)"
    )
    _add_tenant_arg(pull_parser)
    pull_parser.set_defaults(func=_dispatch_artifact("cmd_pull"))

    audit_parser = artifact_sub.add_parser(
        "audit",
        help="Show the registry audit: every name/alias/tag mutation",
        description=(
            "Append-only history of name, alias, and tag mutations; it "
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

    gc_parser = artifact_sub.add_parser(
        "gc",
        help="Collect artifact versions nothing needs, least recently used first",
        description=(
            "Collect what nothing holds: no name, alias, pin or publication, "
            "not the current value of an id somebody chose (a notebook's cell "
            "outputs), and not used recently. Each limit not given takes the "
            "configured retention (STRATA_ARTIFACT_GC_*), so a bare run does "
            "what the server's scheduled sweep does. Losing a result costs "
            "only a recompute; name or pin one to keep it."
        ),
    )
    _add_store_args(gc_parser)
    gc_parser.add_argument(
        "--max-bytes",
        default=None,
        metavar="SIZE",
        help="Keep the store under this size, e.g. 20G or 500M (K, M, G, T are powers of 1024)",
    )
    gc_parser.add_argument(
        "--max-idle-days",
        type=float,
        default=None,
        help="Collect what has not been used for this many days (0: everything past the floor)",
    )
    gc_parser.add_argument(
        "--min-idle-seconds",
        type=float,
        default=None,
        help="Never collect anything used more recently than this",
    )
    gc_parser.add_argument(
        "--collect-latest",
        action="store_true",
        help="Also collect the current value of ids callers chose. Deletes live state.",
    )
    gc_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be collected, and delete nothing",
    )
    gc_parser.set_defaults(func=_dispatch_artifact("cmd_gc"))

    artifact_parser.set_defaults(func=lambda args: (artifact_parser.print_help(), 0)[1])

    # Agent inspect commands.
    cell_parser = subparsers.add_parser(
        "cell",
        help=(
            "Per-cell operations: list, show, run, cancel, test, add, edit, rm, mv, annotate, "
            "output, pin-fetch"
        ),
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
        help="Manage notebook workers: ls, add, rm, default, add-ssh, rm-ssh",
        description=(
            "Register notebook-scoped workers (e.g. a remote executor endpoint) "
            "and set the default. Writes notebook.toml; a running server picks "
            "the change up on its next session reload. add-ssh / rm-ssh drive a "
            "running server instead."
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

    tutorial_parser = subparsers.add_parser(
        "tutorial",
        help="Open a copy of a bundled sample notebook, or list them",
        description=(
            "Copy a sample that ships with Strata into the notebook storage directory "
            "under a fresh name, open it, and print its URL. With no name, list the "
            "samples. Starts a server if none is running."
        ),
    )
    add_tutorial_arguments(tutorial_parser)
    tutorial_parser.set_defaults(func=tutorial_main)

    scratch_parser = subparsers.add_parser(
        "scratch",
        help="Start, stop or show a project's scratchpad daemon",
        description=(
            "A project's scratchpad lives in <dir>/.strata-scratch/ and is served by one "
            "daemon, which an agent's first call starts and which stops itself after "
            "30 idle minutes. These commands start, stop or show it by hand."
        ),
    )
    scratch_sub = scratch_parser.add_subparsers(dest="scratch_command", required=True)
    for name, help_text in (
        ("start", "Start the daemon (creating the scratchpad) unless one runs; print it"),
        ("stop", "Stop the daemon if one runs"),
        ("status", "Print the running daemon, or say there is none"),
    ):
        sub = scratch_sub.add_parser(name, help=help_text)
        sub.add_argument(
            "--root",
            default=".",
            help="The project directory holding .strata-scratch/ (default: the current one)",
        )
        sub.set_defaults(func=_dispatch_scratch)
    ls_parser = scratch_sub.add_parser(
        "ls", help="List every scratchpad: size, last use, daemon, overage"
    )
    ls_parser.add_argument("--format", choices=["human", "json"], default="human")
    ls_parser.set_defaults(func=_dispatch_scratch_ls)
    gc_parser = scratch_sub.add_parser(
        "gc", help="Sweep a scratchpad now, by idle days then its soft limit"
    )
    gc_target = gc_parser.add_mutually_exclusive_group()
    gc_target.add_argument(
        "--root", default=".", help="The project whose scratchpad to sweep (default: here)"
    )
    gc_target.add_argument("--all", action="store_true", help="Sweep every listed scratchpad")
    gc_parser.set_defaults(func=_dispatch_scratch_gc)
    rm_parser = scratch_sub.add_parser(
        "rm", help="Stop a scratchpad's daemon and delete .strata-scratch/; costs reruns only"
    )
    rm_parser.add_argument(
        "--root", default=".", help="The project whose scratchpad to delete (default: here)"
    )
    rm_parser.set_defaults(func=_dispatch_scratch_rm)
    retire_parser = scratch_sub.add_parser(
        "retire", help="Retire a cell now: its source kept, its values freed by the next sweep"
    )
    retire_parser.add_argument("cell", help="The cell id")
    retire_parser.add_argument(
        "--root", default=".", help="The project whose scratchpad holds it (default: here)"
    )
    retire_parser.set_defaults(func=_dispatch_scratch_retire)

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="Serve a project's scratchpad to a coding agent as a stdio MCP server",
        description=(
            "For an agent's MCP config. Lists the scratchpad's tools at once; the first "
            "tool call starts the project's scratchpad daemon (see `strata scratch`) and "
            "every call runs there. Needs the [mcp] extra."
        ),
    )
    mcp_parser.add_argument(
        "--root",
        default=".",
        help="The project whose scratchpad to serve (default: the current directory)",
    )
    mcp_parser.set_defaults(func=_dispatch_mcp)

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
    watch_parser.set_defaults(func=_dispatch_watch)

    return parser


def _dispatch_artifact(command: str):
    """Build a dispatcher that lazily imports the artifact CLI module."""

    def _dispatch(args: argparse.Namespace) -> int:
        from strata.artifact import cli as artifact_cli

        return getattr(artifact_cli, command)(args)

    return _dispatch


def _dispatch_env_gc(args: argparse.Namespace) -> int:
    from strata.config import StrataConfig
    from strata.notebook.env_backend import shared_env_root
    from strata.notebook.shared_env import collect

    config = StrataConfig.load()
    root = Path(args.env_dir) if args.env_dir else shared_env_root(config)
    ttl_days = args.ttl_days if args.ttl_days is not None else config.notebook_shared_env_ttl_days
    result = collect(root, ttl_days=ttl_days)
    for key in result.removed:
        print(f"removed {key}")
    for key in result.referenced:
        print(f"kept {key} (linked from a notebook)")
    print(f"{len(result.removed)} removed, {len(result.referenced)} still linked")
    return 0


def _dispatch_migrate(args: argparse.Namespace) -> int:
    from strata import migrate_cli

    return migrate_cli.cmd_migrate(args)


def _dispatch_apikey(command: str):
    """Build a dispatcher that lazily imports the API key CLI module."""

    def _dispatch(args: argparse.Namespace) -> int:
        from strata.auth import api_key_cli

        return getattr(api_key_cli, command)(args)

    return _dispatch


def _dispatch_run(args: argparse.Namespace) -> int:
    # The top-level parser already populated the namespace; don't re-parse.
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


def _dispatch_mcp(args: argparse.Namespace) -> int:
    try:
        from strata.mcp_bridge import mcp_main
    except ModuleNotFoundError as exc:
        if exc.name is None or not exc.name.startswith("mcp"):
            raise
        print(
            "strata mcp needs the [mcp] extra: uv tool install 'strata-notebook[mcp]'",
            file=sys.stderr,
        )
        return 2
    return mcp_main(Path(args.root))


def _dispatch_scratch(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict
    from importlib.metadata import version

    from strata import scratch

    root = Path(args.root).expanduser()
    if args.scratch_command == "stop":
        try:
            stopped = scratch.stop_daemon(root)
        except scratch.ScratchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print("stopped" if stopped else "no scratchpad daemon runs here")
        return 0
    try:
        if args.scratch_command == "status":
            running = scratch.find_daemon(root.resolve())
            if running is None:
                print("no scratchpad daemon runs here")
                return 1
        else:
            running = scratch.ensure_daemon(
                root, progress=lambda step: print(f"{step}…", file=sys.stderr, flush=True)
            )
            current = version("strata-notebook")
            if running.version != current:
                print(
                    f"note: the daemon runs Strata {running.version}, not {current}; "
                    "it was busy, so it was not replaced",
                    file=sys.stderr,
                )
    except scratch.ScratchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(asdict(running), indent=2))
    return 0


def _bytes(size: float) -> str:
    unit = "B"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            break
        size /= 1024
    return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"


def _dispatch_scratch_ls(args: argparse.Namespace) -> int:
    import json
    import time

    from strata import scratch
    from strata.notebook.scratch_storage import SAFE_TO_DELETE

    def running(anchor: str) -> bool:
        try:
            return scratch.find_daemon(Path(anchor)) is not None
        except scratch.ScratchError:
            # Alive, but not answering.
            return True

    rows: list[dict[str, Any]] = [
        {**entry, "anchor": anchor, "running": running(anchor)}
        for anchor, entry in sorted(scratch.registered().items())
    ]
    if args.format == "json":
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print("no scratchpads")
        return 0
    print(f"{'SIZE':>10}  {'OVER BY':>10}  {'LAST USED':<16}  DAEMON  ANCHOR")
    for row in rows:
        size = _bytes(row["store_bytes"])
        over = _bytes(row["over_by_bytes"]) if row["over_by_bytes"] else "-"
        used = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["last_used"]))
        daemon = "up" if row["running"] else "-"
        print(f"{size:>10}  {over:>10}  {used:<16}  {daemon:<6}  {row['anchor']}")
    print(f"\n{SAFE_TO_DELETE} `strata scratch rm` deletes one.")
    return 0


def _dispatch_scratch_gc(args: argparse.Namespace) -> int:
    import contextlib
    import json
    from dataclasses import asdict

    from strata import scratch
    from strata.config import StrataConfig
    from strata.notebook.ops import LocalNotebookOps
    from strata.notebook.scratch_storage import retire_idle, upkeep

    anchors = (
        [Path(anchor) for anchor in scratch.registered()]
        if args.all
        else [Path(args.root).expanduser().resolve()]
    )
    swept = {}
    failed = False
    for anchor in anchors:
        directory = scratch.scratch_dir(anchor)
        if not (directory / "notebook.toml").is_file():
            # The rest are still swept.
            print(f"error: {anchor} has no scratchpad", file=sys.stderr)
            failed = True
            continue
        # As its daemon reads it, from inside the scratchpad: the same limits.
        with contextlib.chdir(directory):
            config = StrataConfig.load()
        retired: list[str] | str
        try:
            # Held, so no daemon starts while the files are edited here.
            with scratch.daemon_lock(anchor):
                running = scratch.find_daemon(anchor)
                if running is None:
                    with LocalNotebookOps(directory) as ops:
                        retired = retire_idle(
                            ops, directory, retire_days=config.notebook_scratch_retire_days
                        )
        except scratch.ScratchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            failed = True
            continue
        if running is not None:
            # Its session is the writer, and it retires idle cells itself, hourly, only
            # when nothing runs: removing them from here could catch one mid-run.
            retired = f"left to its daemon (pid {running.pid}), which retires them hourly"
        storage = upkeep(
            anchor,
            soft_limit_bytes=config.notebook_scratch_gc_soft_limit_bytes,
            max_idle_days=config.notebook_scratch_gc_max_idle_days,
            min_idle_seconds=config.artifact_gc_min_idle_seconds,
            # A sweep is not a use: it leaves the scratchpad's last use alone.
            used=False,
        )
        swept[str(anchor)] = {**asdict(storage), "retired": retired}
    print(json.dumps(swept, indent=2))
    return 2 if failed else 0


@contextmanager
def _scratch_ops(anchor: Path, running: Any) -> Generator[Any]:
    """The scratchpad's ops: through its *running* daemon if any, its session being the
    writer."""
    from strata import scratch
    from strata.notebook.ops import LocalNotebookOps, RemoteNotebookOps

    if running is None:
        with LocalNotebookOps(scratch.scratch_dir(anchor)) as local:
            yield local
        return
    ops = RemoteNotebookOps(running.url, running.session_id)
    try:
        yield ops
    finally:
        ops.close()


def _dispatch_scratch_retire(args: argparse.Namespace) -> int:
    import json

    from strata import scratch
    from strata.notebook.ops import NotebookOpsError
    from strata.notebook.scratch_storage import retire_cell, scratch_manager

    anchor = Path(args.root).expanduser().resolve()
    directory = scratch.scratch_dir(anchor)
    if not (directory / "notebook.toml").is_file():
        print(f"error: {anchor} has no scratchpad", file=sys.stderr)
        return 2
    # Held, so no daemon starts while the files are edited here.
    with scratch.daemon_lock(anchor):
        try:
            running = scratch.find_daemon(anchor)
        except scratch.ScratchError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        with _scratch_ops(anchor, running) as ops:
            try:
                cell = ops.get_cell(args.cell)
            except NotebookOpsError as exc:
                print(f"error: {exc}", file=sys.stderr)
                return 2
            if cell.downstream_ids:
                readers = ", ".join(cell.downstream_ids)
                print(
                    f"error: cells {readers} read {args.cell}; retire them first",
                    file=sys.stderr,
                )
                return 2
            if cell.status == "running":
                print(f"error: {args.cell} is running; retire it once it ends", file=sys.stderr)
                return 2
            manager = scratch_manager(directory)
            try:
                record = retire_cell(ops, directory, manager, cell)
            finally:
                manager.close()
    print(json.dumps({"retired": record["cell"], "defines": record["defines"]}, indent=2))
    return 0


def _dispatch_scratch_rm(args: argparse.Namespace) -> int:
    import shutil

    from strata import scratch

    anchor = Path(args.root).expanduser().resolve()
    directory = scratch.scratch_dir(anchor)
    if not (directory / "notebook.toml").is_file():
        print(f"error: {anchor} has no scratchpad", file=sys.stderr)
        return 2
    try:
        scratch.stop_daemon(anchor)
    except scratch.ScratchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    shutil.rmtree(directory)
    scratch.unregister(anchor)
    print(f"removed {directory}")
    return 0


def _dispatch_watch(args: argparse.Namespace) -> int:
    from strata.notebook.tui.cli import run_spectator

    run_spectator(
        server=args.server,
        session=args.session,
        notebook=args.notebook_dir,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    from strata._uv_runtime import assert_uv_managed_runtime

    parser = _build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    # After parsing, as server.main does: `--help` works from any Python, and
    # nothing else runs outside a uv-managed environment.
    assert_uv_managed_runtime()

    return args.func(args)


def run_main(argv: list[str] | None = None) -> int:
    """Shim for a direct ``strata-run`` entry point, if we ever add one."""
    return _run_main_direct(argv)


if __name__ == "__main__":
    sys.exit(main())
