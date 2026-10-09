"""Headless notebook commands, including ``strata run <notebook_dir>``.

Runs through ``NotebookSession`` and ``CellExecutor`` directly, the same path as
the UI, with no HTTP server. Exit codes: 0 all cells succeeded, 1 a cell failed,
2 invocation or setup error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.notebook.ops import CellView, WorkerListView

from strata.notebook.models import CellLanguage

# Disabled when stdout isn't a tty so pipes and CI logs stay clean.
_USE_COLOR = sys.stdout.isatty()


def _color(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(text: str) -> str:
    return _color("32", text)


def _red(text: str) -> str:
    return _color("31", text)


def _dim(text: str) -> str:
    return _color("90", text)


def _yellow(text: str) -> str:
    return _color("33", text)


def _cell_label(source: str, max_len: int = 32) -> str:
    """Short display label for a cell: its first non-blank, non-comment line, or "(empty)"."""
    for raw in source.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        return line[:max_len] + ("…" if len(line) > max_len else "")
    return "(empty)"


def _format_ms(duration_ms: float | int) -> str:
    d = int(duration_ms)
    if d < 1000:
        return f"{d}ms"
    return f"{d / 1000:.1f}s"


# Per-cell console cap so a print-heavy cell can't balloon the JSON result.
_MAX_JSON_CONSOLE_CHARS = 10_000


def _truncate_console(text: str) -> str:
    if len(text) <= _MAX_JSON_CONSOLE_CHARS:
        return text
    omitted = len(text) - _MAX_JSON_CONSOLE_CHARS
    return text[:_MAX_JSON_CONSOLE_CHARS] + f"… [+{omitted} chars truncated]"


def _print_cell_line(entry: dict[str, Any]) -> None:
    """Print a single cell result line in the human format."""
    cell_id_short = entry["id"][:8]
    label = entry["label"]
    status = entry["status"]

    if status == "ok":
        if entry.get("cache_hit"):
            marker = _green("✓")
            tail = _dim("cached")
        else:
            marker = _green("✓")
            tail = _format_ms(entry["duration_ms"])
        print(f"  {cell_id_short} {label:<32} {marker} {tail}")
    elif status == "error":
        marker = _red("✗")
        tail = _format_ms(entry["duration_ms"])
        print(f"  {cell_id_short} {label:<32} {marker} {tail}")
        error = entry.get("error")
        if error:
            for line in str(error).splitlines():
                print(f"      {_red(line)}")
    elif status == "skipped":
        marker = _dim("-")
        reason = entry.get("reason", "skipped")
        print(f"  {cell_id_short} {label:<32} {marker} {_dim(reason)}")


def _print_summary(results: list[dict[str, Any]], total_ms: int) -> None:
    ran = sum(1 for r in results if r["status"] == "ok" and not r.get("cache_hit"))
    cached = sum(1 for r in results if r["status"] == "ok" and r.get("cache_hit"))
    failed = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    parts = []
    if ran:
        parts.append(f"{ran} ran")
    if cached:
        parts.append(f"{cached} cached")
    if failed:
        parts.append(_red(f"{failed} failed"))
    if skipped:
        parts.append(_yellow(f"{skipped} skipped"))
    if not parts:
        parts.append("nothing to run")

    print()
    print(f"{', '.join(parts)} in {_format_ms(total_ms)}")


def _use_existing_environment(session: Any) -> tuple[bool, str | None]:
    """Use the notebook's prepared ``.venv`` as its interpreter (``--no-sync``).

    Returns ``(ok, error_message)``. Checks ``bin/python`` resolves, not just the
    directory, so cells never silently run on whatever ``python`` is on PATH.
    """
    venv_python = session.path / ".venv" / "bin" / "python"
    if not venv_python.exists():
        return False, (
            f"notebook has no usable .venv at {session.path / '.venv'} "
            f"({venv_python} is missing)\n"
            "hint: run without --no-sync, or `uv sync` in the notebook dir first"
        )
    # Sets the interpreter together with every environment field; a session never
    # records an interpreter on its own.
    session.refresh_environment_runtime()
    return True, None


async def _sync_environment(session: Any) -> tuple[bool, str | None]:
    """Run ``uv sync`` through the session's environment jobs; return ``(ok, error_message)``."""
    try:
        job = await session.submit_environment_job(action="sync")
    except Exception as exc:
        return False, f"failed to submit env sync job: {exc}"

    try:
        await session.wait_for_environment_job()
    except Exception as exc:
        return False, f"env sync raised: {exc}"

    # ``_run_environment_job`` mutates this returned snapshot in place.
    # ``session.environment_job`` is reset to None when the job finishes, so it
    # can't be read here.
    if job.status != "completed":
        message = job.error or f"env sync ended with status={job.status}"
        return False, message
    return True, None


async def _drain_warm_pool(session: Any) -> None:
    """Release the warm process pool if one exists; drain errors are swallowed."""
    pool = getattr(session, "warm_pool", None)
    if pool is None:
        return
    try:
        if hasattr(pool, "drain"):
            maybe_awaitable = pool.drain()
            if asyncio.iscoroutine(maybe_awaitable):
                await maybe_awaitable
        elif hasattr(pool, "shutdown_nowait"):
            pool.shutdown_nowait()
    except Exception:
        pass


@contextmanager
def _quiet_notebook_logs():
    """Hold notebook loggers at WARNING; per-cell INFO lines would bury the run's own output.

    ``STRATA_LOG_LEVEL`` opts back in. Restored on exit, so an in-process caller keeps its level.
    """
    notebook_logger = logging.getLogger("strata.notebook")
    previous = notebook_logger.level
    if "STRATA_LOG_LEVEL" not in os.environ:
        notebook_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        notebook_logger.setLevel(previous)


async def _run_async(args: argparse.Namespace) -> int:
    with _quiet_notebook_logs():
        return await _run_notebook(args)


async def _run_notebook(args: argparse.Namespace) -> int:
    notebook_dir = Path(args.path).expanduser().resolve()

    if not notebook_dir.is_dir():
        print(f"error: {notebook_dir} is not a directory", file=sys.stderr)
        return 2
    if not (notebook_dir / "notebook.toml").is_file():
        print(
            f"error: {notebook_dir} is not a Strata notebook (no notebook.toml)",
            file=sys.stderr,
        )
        return 2

    # Late imports so --help / path errors don't pay heavy import cost.
    from strata.notebook.executor import DEFAULT_CELL_TIMEOUT_SECONDS, CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    try:
        state = parse_notebook(notebook_dir)
        session = NotebookSession(state, notebook_dir)
    except Exception as exc:
        print(f"error: failed to open notebook: {exc}", file=sys.stderr)
        return 2

    if session.dag is None:
        detail = getattr(session, "dag_error", None)
        print(
            f"error: notebook DAG could not be built: {detail}"
            if detail
            else "error: notebook DAG could not be built; inspect the notebook "
            "in the UI and resolve the conflicting cells first",
            file=sys.stderr,
        )
        return 2

    # Environment: either sync now, or take the user's prepared venv.
    if args.no_sync:
        ok, err = _use_existing_environment(session)
        if not ok:
            print(f"error: {err}", file=sys.stderr)
            return 2
    else:
        if args.format == "human":
            print(_dim("syncing environment…"))
        ok, err = await _sync_environment(session)
        if not ok:
            print(f"error: {err}", file=sys.stderr)
            await _drain_warm_pool(session)
            return 2

    # Restore R from renv.lock like the server's session open, on the --no-sync path
    # too (cheap when already in sync); otherwise R cells run against an empty
    # project library. Threaded because ``_renv_sync`` shells out synchronously.
    if (notebook_dir / "renv.lock").exists():
        if args.format == "human":
            print(_dim("restoring R environment…"))
        # ``ensure_renv_synced`` swallows expected failures; surface unexpected ones
        # (e.g. a non-executable Rscript) as a clean exit-2 setup error.
        try:
            await asyncio.to_thread(session.ensure_renv_synced)
        except Exception as exc:
            print(f"error: R environment restore failed: {exc}", file=sys.stderr)
            await _drain_warm_pool(session)
            return 2

    if args.format == "human":
        print(f"running: {notebook_dir}")
        print()

    executor = CellExecutor(session)
    # The report gives every cell a digest, so a leaf keeps what it defines.
    executor.store_leaf_outputs = True
    cell_by_id = {c.id: c for c in session.notebook_state.cells}
    results: list[dict[str, Any]] = []
    failed_cells: set[str] = set()
    start = time.monotonic()

    # Each cell executes at most once, so a @nocache producer with several
    # consumers is not re-executed per consumer.
    with executor.one_run():
        for cell_id in session.dag.topological_order:
            cell = cell_by_id.get(cell_id)
            if cell is None:
                # In the DAG but not in notebook_state; shouldn't happen.
                continue

            # Report markdown prose as a no-op success, not "skipped: unsupported language".
            if cell.language == CellLanguage.MARKDOWN:
                entry = {
                    "id": cell_id,
                    "label": f"[markdown] {_cell_label(cell.source)}",
                    "status": "ok",
                    "reason": None,
                    "duration_ms": 0,
                    "cache_hit": True,
                }
                results.append(entry)
                if args.format == "human" and not args.quiet:
                    _print_cell_line(entry)
                continue

            # R runs through the same language dispatch as the session; a missing `Rscript`
            # is a clean cell error.
            if cell.language not in {
                CellLanguage.PYTHON,
                CellLanguage.PROMPT,
                CellLanguage.SQL,
                CellLanguage.R,
                CellLanguage.WIDGET,
            }:
                entry = {
                    "id": cell_id,
                    "label": f"[{cell.language}] {_cell_label(cell.source)}",
                    "status": "skipped",
                    "reason": f"unsupported language: {cell.language}",
                    "duration_ms": 0,
                    "cache_hit": False,
                }
                results.append(entry)
                if args.format == "human" and not args.quiet:
                    _print_cell_line(entry)
                continue

            upstream = session.dag.cell_upstream.get(cell_id, [])
            if any(u in failed_cells for u in upstream):
                entry = {
                    "id": cell_id,
                    "label": _cell_label(cell.source),
                    "status": "skipped",
                    "reason": "upstream failed",
                    "duration_ms": 0,
                    "cache_hit": False,
                }
                results.append(entry)
                failed_cells.add(cell_id)
                if args.format == "human" and not args.quiet:
                    _print_cell_line(entry)
                continue

            try:
                # A per-cell `# @timeout` / notebook.toml `timeout` still wins (see
                # CellExecutor._resolve_effective_timeout).
                cell_timeout = (
                    args.timeout if args.timeout is not None else DEFAULT_CELL_TIMEOUT_SECONDS
                )
                if args.force:
                    result = await executor.execute_cell_force(
                        cell_id, cell.source, timeout_seconds=cell_timeout
                    )
                else:
                    result = await executor.execute_cell(
                        cell_id, cell.source, timeout_seconds=cell_timeout
                    )
            except Exception as exc:
                entry = {
                    "id": cell_id,
                    "label": _cell_label(cell.source),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "duration_ms": 0,
                    "cache_hit": False,
                }
                results.append(entry)
                failed_cells.add(cell_id)
                if args.format == "human" and not args.quiet:
                    _print_cell_line(entry)
                continue

            entry: dict[str, Any] = {
                "id": cell_id,
                "label": _cell_label(cell.source),
                "status": "ok" if result.success else "error",
                "duration_ms": int(result.duration_ms or 0),
                "cache_hit": bool(result.cache_hit),
            }
            # Lets scripts and agents verify values from the JSON instead of reading
            # .strata/. Cache hits don't re-emit console output, so these may be absent.
            if result.stdout:
                entry["stdout"] = _truncate_console(result.stdout)
            if result.stderr:
                entry["stderr"] = _truncate_console(result.stderr)
            # Otherwise silent headless: a cell that mutates an input without exporting it
            # leaves downstream cells reading the stale value.
            if result.mutation_warnings:
                entry["mutation_warnings"] = [dict(w) for w in result.mutation_warnings]
            if not result.success:
                entry["error"] = result.error or "cell failed"
                if result.traceback:
                    entry["traceback"] = result.traceback
                if result.error_code:
                    entry["error_code"] = result.error_code
                failed_cells.add(cell_id)
            else:
                # Makes two run reports comparable: identical "green" JSON can hide different
                # numbers. Read from the store, since a cache hit carries no outputs.
                entry.update(_cell_identity(session, cell, cell_id))
            results.append(entry)
            if args.format == "human" and not args.quiet:
                _print_cell_line(entry)
                for w in result.mutation_warnings:
                    tail = f". {w['suggestion']}" if w.get("suggestion") else ""
                    print(f"      {_yellow('⚠')} {w['message']}{tail}")

    total_ms = int((time.monotonic() - start) * 1000)
    any_failed = any(r["status"] == "error" for r in results)

    if args.format == "json":
        payload = {
            "notebook": str(notebook_dir),
            "success": not any_failed,
            "duration_ms": total_ms,
            "cells": [{k: v for k, v in r.items() if k != "label"} for r in results],
        }
        print(json.dumps(payload, indent=2))
    else:
        _print_summary(results, total_ms)

    await _drain_warm_pool(session)
    return 1 if any_failed else 0


def _cell_identity(session, cell, cell_id: str) -> dict[str, Any]:
    """The cell's provenance hash and its outputs' digests, for the report.

    Best-effort. An absent digest is reported as ``null`` rather than omitted, so
    a diff of two reports shows the missing digest, not a missing output.
    """
    identity: dict[str, Any] = {}
    if cell.last_provenance_hash:
        identity["provenance_hash"] = cell.last_provenance_hash

    outputs = session.get_artifact_manager().cell_output_digests(cell_id)
    if outputs:
        identity["outputs"] = outputs
    return identity


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``run`` subcommand arguments to an existing parser."""
    parser.add_argument(
        "path",
        help="Path to the notebook directory (containing notebook.toml)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache and re-execute every cell",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip `uv sync`; require .venv/ to already exist",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "Per-cell timeout for this run, overriding the 300s default "
            "(a per-cell `# @timeout` or notebook.toml `timeout` still wins). "
            "Use for compute-heavy cells like model training."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-cell output lines (human format only)",
    )


def run_main(argv: list[str] | None = None) -> int:
    """Entry point for ``strata run``; also callable as ``run_main(["./my-notebook"])``."""
    parser = argparse.ArgumentParser(
        prog="strata run",
        description="Execute every cell in a Strata notebook directory.",
    )
    add_run_arguments(parser)
    args = parser.parse_args(argv)
    return asyncio.run(_run_async(args))


def add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``validate`` subcommand arguments to an existing parser."""
    parser.add_argument(
        "path",
        help="Path to the notebook directory (containing notebook.toml)",
    )
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )


def validate_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata validate``.

    Static checks only (nothing executes, nothing syncs): ``notebook.toml``
    parses, cell files load, the DAG has no cycle, and per-cell annotation
    diagnostics. Exit 0 valid (warnings allowed), 1 invalid, 2 invocation error.
    """
    notebook_dir = Path(args.path).expanduser().resolve()

    if not notebook_dir.is_dir():
        print(f"error: {notebook_dir} is not a directory", file=sys.stderr)
        return 2
    if not (notebook_dir / "notebook.toml").is_file():
        print(
            f"error: {notebook_dir} is not a Strata notebook (no notebook.toml)",
            file=sys.stderr,
        )
        return 2

    from strata.notebook.annotation_validation import validate_cell_annotations
    from strata.notebook.models import DiagnosticSeverity
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    notebook_errors: list[dict[str, str]] = []
    cells_payload: list[dict[str, Any]] = []
    error_count = 0
    warning_count = 0

    session = None
    try:
        state = parse_notebook(notebook_dir)
        session = NotebookSession(state, notebook_dir)
    except Exception as exc:
        notebook_errors.append(
            {
                "code": "parse_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )

    if session is not None and session.dag is None:
        # Report the real failure: a duplicate `# @variant` name also fails the build,
        # and ``validate`` is the only command that reports it.
        detail = getattr(session, "dag_error", None)
        notebook_errors.append(
            {
                "code": "dag_build_failed",
                "message": (
                    f"notebook DAG could not be built: {detail}"
                    if detail
                    else "notebook DAG could not be built; inspect the notebook and "
                    "resolve the conflicting cells"
                ),
            }
        )

    if session is not None:
        for cell in session.notebook_state.cells:
            diagnostics = validate_cell_annotations(cell, session.notebook_state)
            for diag in diagnostics:
                if diag.severity == DiagnosticSeverity.ERROR:
                    error_count += 1
                elif diag.severity == DiagnosticSeverity.WARN:
                    warning_count += 1
            cells_payload.append(
                {
                    "id": cell.id,
                    "language": str(cell.language),
                    "defines": list(cell.defines),
                    "references": list(cell.references),
                    "diagnostics": [d.model_dump() for d in diagnostics],
                }
            )

    valid = not notebook_errors and error_count == 0

    if args.format == "json":
        payload = {
            "notebook": str(notebook_dir),
            "valid": valid,
            "errors": notebook_errors,
            "cells": cells_payload,
            "summary": {
                "cells": len(cells_payload),
                "errors": len(notebook_errors) + error_count,
                "warnings": warning_count,
            },
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f"validating: {notebook_dir}")
        for err in notebook_errors:
            print(f"  {_red('✗')} {err['code']}: {err['message']}")
        for cell_entry in cells_payload:
            diags = cell_entry["diagnostics"]
            if not diags:
                continue
            print(f"  cell {cell_entry['id'][:8]} [{cell_entry['language']}]")
            for diag in diags:
                marker = _red("error") if diag["severity"] == "error" else _yellow(diag["severity"])
                line_part = f" (line {diag['line']})" if diag.get("line") else ""
                print(f"    {marker} {diag['code']}: {diag['message']}{line_part}")
        print()
        total_errors = len(notebook_errors) + error_count
        if valid:
            suffix = f", {warning_count} warning(s)" if warning_count else ""
            print(f"{_green('✓')} valid: {len(cells_payload)} cell(s){suffix}")
        else:
            print(f"{_red('✗')} invalid: {total_errors} error(s), {warning_count} warning(s)")

    return 0 if valid else 1


def add_new_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``new`` subcommand arguments to an existing parser."""
    parser.add_argument(
        "name",
        help="Notebook name; the directory is the slugified name under --parent",
    )
    parser.add_argument(
        "--parent",
        default=".",
        help="Parent directory for the notebook (default: current directory)",
    )
    parser.add_argument(
        "--python",
        dest="python_version",
        default=None,
        help="Python major.minor for the notebook venv (default: current interpreter)",
    )
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="Skip creating the uv venv now; `strata run` will sync it later",
    )
    parser.add_argument(
        "--no-git",
        action="store_true",
        help=(
            "Skip writing .gitignore. The default one keeps .strata/, .venv/ "
            "and renv/library/ out of version control; an existing .gitignore "
            "is never replaced either way"
        ),
    )
    parser.add_argument(
        "--project-mount",
        dest="project_mount",
        nargs="?",
        const="project",
        default=None,
        metavar="NAME",
        help=(
            "Mount --parent (the project dir) read-only as a Path variable NAME "
            "(default 'project') in every cell, so cells read project files as "
            "`open(NAME / 'file')` without absolute paths. Pinned, so it never "
            "adds staleness. Handy for a scratchpad living in a project subdir."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )


def new_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata new``.

    Scaffolds notebook.toml, pyproject.toml and cells/. On an existing notebook
    directory, only missing scaffolding is added.
    """
    from strata.notebook.writer import create_notebook

    try:
        notebook_dir = create_notebook(
            Path(args.parent).expanduser().resolve(),
            args.name,
            args.python_version,
            initialize_environment=not args.no_env,
            write_gitignore_file=not getattr(args, "no_git", False),
            project_mount=getattr(args, "project_mount", None),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(
            json.dumps(
                {
                    "notebook_dir": str(notebook_dir),
                    "name": args.name,
                    "environment_initialized": not args.no_env,
                },
                indent=2,
            )
        )
    else:
        print(f"created: {notebook_dir}")
        print(_dim("  add cells under cells/*.py and list them in notebook.toml"))
        print(_dim(f"  validate: strata validate {notebook_dir}"))
        print(_dim(f"  run:      strata run {notebook_dir}"))
    return 0


def add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``export`` subcommand arguments to an existing parser."""
    parser.add_argument(
        "path",
        help="Path to the notebook directory (containing notebook.toml)",
    )
    parser.add_argument(
        "--to",
        dest="output_format",
        choices=["markdown", "html", "snapshot"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="snapshot only: overwrite --out when it already holds something",
    )
    parser.add_argument(
        "--include",
        choices=["all", "selected", "none"],
        default="selected",
        help=(
            "snapshot only: whose artifact bytes to carry. 'all' for moving a "
            "project between machines, 'selected' (with --cells) for a review "
            "snapshot with the rest by reference, 'none' for description only"
        ),
    )
    parser.add_argument(
        "--cells",
        default=None,
        help="snapshot only: comma-separated cell ids whose artifacts to carry",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=None,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--include-inactive-variants",
        action="store_true",
        help="Include inactive variants of every variant group in the output",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Skip the per-cell console (stdout/stderr) snapshots",
    )
    parser.add_argument(
        "--app-view",
        action="store_true",
        help=(
            "App-view snapshot: render only what the app view shows (widgets, "
            "markdown, display outputs) with no cell sources: a portable, "
            "self-contained picture of the dashboard"
        ),
    )
    parser.add_argument(
        "--max-output-bytes",
        type=int,
        default=None,
        help=(
            "Per-output byte cap; truncates console snapshots, JSON previews, "
            "and inline image data URLs. Default 1048576 (1 MB). "
            "Pass 0 to disable."
        ),
    )


def add_import_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach ``import`` subcommand arguments to an existing parser."""
    parser.add_argument(
        "path",
        help=(
            "A Jupyter .ipynb file, or a snapshot .zip exported with `strata export --to snapshot`"
        ),
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=None,
        help=(
            "Target notebook directory. Defaults to a sibling directory "
            "named after the file's stem."
        ),
    )
    parser.add_argument(
        "--check-deps",
        dest="check_deps",
        action="store_true",
        help=(
            "Run `uv lock` after import to verify captured dependencies "
            "resolve. Failures land in the import report. Requires uv on "
            "PATH; seconds-slow on cold caches."
        ),
    )


def _import_snapshot_bundle(path: Path, args: argparse.Namespace) -> int:
    """``strata import <snapshot.zip>``: unpack a bundle into a notebook directory."""
    from strata.config import StrataConfig
    from strata.notebook.routes import _discover_notebooks
    from strata.notebook.snapshot_import import NotASnapshotError, import_snapshot

    stem = path.name.removesuffix(".zip").removesuffix(".snapshot")
    dest = Path(args.output_path) if args.output_path else path.with_name(stem)

    # A notebook id already in use under the storage root is replaced, so two copies
    # never collide when both publish to a shared store. Copies elsewhere aren't visible.
    root = StrataConfig.load().notebook_storage_dir
    taken = {e["notebook_id"] for e in _discover_notebooks(Path(root)) if e.get("notebook_id")}

    import zipfile

    try:
        result = import_snapshot(path, dest, taken_ids=taken)
    except (NotASnapshotError, FileExistsError, zipfile.BadZipFile) as exc:
        # Not a bundle: an input error, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Imported {path} → {result.notebook_dir}")
    print(f"  artifacts: {result.imported_artifacts} imported")
    if result.replaced_id:
        print(
            f"  notebook id {result.replaced_id} was already in use, so this copy is "
            f"{result.notebook_id}"
        )
    if result.by_reference_cells:
        print(
            f"  not carried (they open idle; run them, or pull from a team store): "
            f"{', '.join(result.by_reference_cells)}"
        )
    return 0


def import_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata import``: convert a Jupyter ``.ipynb`` into a notebook directory."""
    from strata.notebook.jupyter_import import import_notebook

    path = Path(args.path)
    if not path.is_file():
        print(f"error: {path} is not a file", file=sys.stderr)
        return 2
    if path.suffix == ".zip":
        # Told apart by extension; a zip that isn't a snapshot is refused, not guessed at.
        return _import_snapshot_bundle(path, args)
    if path.suffix != ".ipynb":
        print(
            f"warning: {path} does not have .ipynb extension; trying to parse anyway",
            file=sys.stderr,
        )

    try:
        result = import_notebook(
            path,
            out_dir=args.output_path,
            check_deps=bool(getattr(args, "check_deps", False)),
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Imported {path} → {result.notebook_dir}")
    print(
        f"  cells: {result.code_cells} code, {result.markdown_cells} markdown"
        f"  ({result.suppressed_outputs} with ; display-suppression)"
    )
    if result.translated_magics:
        print(f"  magics translated: {len(result.translated_magics)}")
    if result.dropped_magics:
        print(f"  magics dropped: {len(result.dropped_magics)} (see report)")
    if result.dropped_shells:
        print(f"  shell commands dropped: {len(result.dropped_shells)}")
    if result.captured_deps:
        print(f"  dependencies captured: {len(result.captured_deps)} → pyproject.toml")
    if result.skipped_cells:
        kinds = ", ".join(sorted(set(result.skipped_cells)))
        print(f"  skipped {len(result.skipped_cells)} cell(s) of unsupported type(s): {kinds}")
    if result.warnings:
        print(f"  warnings: {len(result.warnings)} (see report)")
    if result.report_path is not None:
        print(f"  report: {result.report_path}")
    return 0


def _write_snapshot_bundle(path: Path, args: argparse.Namespace) -> int:
    """``strata export --to snapshot``: the same bundle the route serves.

    Opened offline (no server), but the members come from
    :func:`write_snapshot`, so the CLI and the route cannot disagree.
    """
    import zipfile

    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.snapshot import (
        unknown_selection,
        write_committed_files,
        write_snapshot,
    )

    out_path = args.output_path
    if not out_path:
        print("error: --out is required for a snapshot (it is a zip, not text)", file=sys.stderr)
        return 2

    session = NotebookSession(parse_notebook(path), path)
    selected = [c for c in (args.cells or "").split(",") if c]
    unknown = unknown_selection(session, selected)
    if unknown:
        print(f"error: no such cell(s) in this notebook: {', '.join(unknown)}", file=sys.stderr)
        return 1
    existing = Path(out_path)
    if existing.exists() and existing.stat().st_size and not getattr(args, "force", False):
        # Like ``strata artifact archive``: a path that already holds something is more
        # likely a mistake than an instruction.
        print(f"error: {out_path} already exists; pass --force to overwrite", file=sys.stderr)
        return 2
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
        write_committed_files(session, archive)
        manifest = write_snapshot(session, archive, include=args.include, selected_cells=selected)

    print(f"Wrote {out_path}")
    print(f"  {len(manifest['cells'])} cells, {len(manifest['carried'])} artifact(s) carried")
    if args.include == "selected" and not selected:
        print("  (no --cells given, so every artifact is by reference)")
    return 0


def export_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata export``: write the rendered notebook to stdout or ``--out``."""
    from strata.notebook.export import ExportFormat, ExportOptions, export_notebook

    path = Path(args.path)
    if not (path / "notebook.toml").is_file():
        print(f"error: {path} is not a notebook directory (no notebook.toml)", file=sys.stderr)
        return 2

    if args.output_format == "snapshot":
        return _write_snapshot_bundle(path, args)

    options = ExportOptions(
        output_format=ExportFormat(args.output_format),
        include_inactive_variants=bool(args.include_inactive_variants),
        include_console=not bool(args.no_console),
        app_view=bool(getattr(args, "app_view", False)),
    )
    if args.max_output_bytes is not None:
        options.max_output_bytes = int(args.max_output_bytes)
    rendered = export_notebook(path, options)

    out_path = args.output_path
    if out_path:
        Path(out_path).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


# ---------------------------------------------------------------------------
# Agent inspect commands (NotebookOps, local backend): `strata cell …` etc.
# JSON by default (agent-first); `--format human` gives a compact view.
# ---------------------------------------------------------------------------


def _open_local_ops(notebook_dir_arg: str, author: str | None = None):
    """Open a :class:`LocalNotebookOps` for *notebook_dir_arg*, or None on error.

    The error is printed to stderr; callers return exit 2 on None.
    """
    notebook_dir = Path(notebook_dir_arg).expanduser().resolve()
    if not (notebook_dir / "notebook.toml").is_file():
        print(
            f"error: {notebook_dir} is not a Strata notebook (no notebook.toml)",
            file=sys.stderr,
        )
        return None
    from strata.notebook.ops import LocalNotebookOps

    try:
        return LocalNotebookOps(notebook_dir, author=author)
    except Exception as exc:  # noqa: BLE001 (any open failure is exit 2)
        print(f"error: failed to open notebook: {exc}", file=sys.stderr)
        return None


def _add_target_args(parser: argparse.ArgumentParser) -> None:
    """Register a read command's target: a local dir or ``--server URL --session ID``."""
    parser.add_argument(
        "notebook_dir", nargs="?", help="Path to the notebook directory (local backend)"
    )
    parser.add_argument(
        "--server", help="Server root for a live session, e.g. http://localhost:8765"
    )
    parser.add_argument("--session", help="Session id to drive on --server")
    parser.add_argument(
        "--author",
        default=None,
        help=(
            "Who to credit for cells this command writes, e.g. your agent's "
            "name. Recorded on the cell so a person can tell agent edits from "
            "their own. Ignored by a server that authenticates its callers, "
            "which uses the authenticated identity instead"
        ),
    )


def _open_read_ops(args: argparse.Namespace):
    """Open the ops backend for *args*: remote when ``--server`` is set, else local.

    Returns None on a usage error (already printed to stderr; callers exit 2).
    """
    if args.server:
        if not args.session:
            print("error: --server requires --session <id>", file=sys.stderr)
            return None
        from strata.notebook.ops import RemoteNotebookOps

        return RemoteNotebookOps(args.server, args.session, author=getattr(args, "author", None))
    if not args.notebook_dir:
        print("error: provide a notebook directory or --server/--session", file=sys.stderr)
        return None
    return _open_local_ops(args.notebook_dir, author=getattr(args, "author", None))


def _close_ops(ops: object) -> None:
    """Close a remote ops client if it owns one (local ops hold no client)."""
    close = getattr(ops, "close", None)
    if callable(close):
        close()


@contextmanager
def _read_ops(args: argparse.Namespace):
    """Open the ops backend and close a remote client on exit; yields None on a usage error."""
    ops = _open_read_ops(args)
    try:
        yield ops
    finally:
        if ops is not None:
            _close_ops(ops)


def _emit_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def add_cell_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``strata cell <action>`` group."""
    sub = parser.add_subparsers(dest="cell_command", metavar="<action>")

    list_p = sub.add_parser("list", help="List cells (id, name, status)")
    _add_target_args(list_p)
    list_p.add_argument("--format", choices=["human", "json"], default="json")
    list_p.set_defaults(func=cell_list_main)

    show_p = sub.add_parser(
        "show", help="Show a cell by id, or the cell that defines a variable (--var)"
    )
    _add_target_args(show_p)
    show_p.add_argument("cell_id", nargs="?", help="Cell id to show")
    show_p.add_argument(
        "--var",
        dest="var",
        metavar="NAME",
        help="Show the cell that defines variable NAME (answers 'do I already have NAME?')",
    )
    show_p.add_argument("--format", choices=["human", "json"], default="json")
    show_p.set_defaults(func=cell_show_main)

    output_p = sub.add_parser(
        "output", help="Write a cell's display output (a plot, an image) to a file"
    )
    _add_target_args(output_p)
    output_p.add_argument("cell_id", help="Cell id whose output to save")
    output_p.add_argument(
        "--out",
        dest="out",
        metavar="FILE",
        required=True,
        help="File to write the bytes to (overwritten if it exists)",
    )
    output_p.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Which display output, in emission order; -1 (default) is the last",
    )
    output_p.add_argument("--format", choices=["human", "json"], default="json")
    output_p.set_defaults(func=cell_output_main)

    run_p = sub.add_parser("run", help="Execute one cell")
    _add_target_args(run_p)
    run_p.add_argument("cell_id", help="Cell id to run")
    mode_group = run_p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--rerun",
        action="store_true",
        help="Bypass the target cell's cache (materialize upstreams)",
    )
    mode_group.add_argument(
        "--force", action="store_true", help="Run against existing upstream artifacts only"
    )
    run_p.add_argument(
        "--no-sync", action="store_true", help="Skip `uv sync`; use the existing .venv/bin/python"
    )
    run_p.add_argument("--format", choices=["human", "json"], default="json")
    run_p.set_defaults(func=cell_run_main)

    test_p = sub.add_parser("test", help="Run a cell's unit tests")
    _add_target_args(test_p)
    test_p.add_argument("cell_id", help="Cell id whose tests to run")
    test_p.add_argument(
        "--file",
        help="Set the cell's test source from this file (`-` for stdin) before running",
    )
    test_p.add_argument(
        "--no-sync", action="store_true", help="Skip `uv sync`; use the existing .venv/bin/python"
    )
    test_p.add_argument("--format", choices=["human", "json"], default="json")
    test_p.set_defaults(func=cell_test_main)

    add_p = sub.add_parser("add", help="Add a new cell, optionally running it")
    _add_target_args(add_p)
    add_src = add_p.add_mutually_exclusive_group(required=True)
    add_src.add_argument("--file", help="Source file (`-` for stdin)")
    add_src.add_argument("-c", "--code", help="Inline cell source")
    add_p.add_argument("--after", help="Insert after this cell id (default: at the end)")
    add_p.add_argument(
        "--language",
        choices=["python", "markdown", "sql", "r", "prompt", "widget"],
        default="python",
    )
    add_p.add_argument(
        "--run",
        action="store_true",
        help="Execute the cell after adding it; fold the run result into the output",
    )
    add_p.add_argument(
        "--no-sync",
        action="store_true",
        help="With --run: skip `uv sync`; use the existing .venv/bin/python",
    )
    add_p.add_argument("--format", choices=["human", "json"], default="json")
    add_p.set_defaults(func=cell_add_main)

    edit_p = sub.add_parser("edit", help="Replace a cell's source from a file")
    _add_target_args(edit_p)
    edit_p.add_argument("cell_id", help="Cell id to edit")
    edit_p.add_argument("--file", required=True, help="Source file (`-` for stdin)")
    edit_p.add_argument("--format", choices=["human", "json"], default="json")
    edit_p.set_defaults(func=cell_edit_main)

    rm_p = sub.add_parser("rm", help="Delete a cell")
    _add_target_args(rm_p)
    rm_p.add_argument("cell_id", help="Cell id to delete")
    rm_p.add_argument("--format", choices=["human", "json"], default="json")
    rm_p.set_defaults(func=cell_rm_main)

    mv_p = sub.add_parser("mv", help="Move a cell to a new position")
    _add_target_args(mv_p)
    mv_p.add_argument("cell_id", help="Cell id to move")
    mv_p.add_argument("--to", type=int, required=True, help="Target index (0-based)")
    mv_p.add_argument("--format", choices=["human", "json"], default="json")
    mv_p.set_defaults(func=cell_mv_main)

    annotate_p = sub.add_parser("annotate", help="Set or remove a cell's `# @key` annotations")
    _add_target_args(annotate_p)
    annotate_p.add_argument("cell_id", help="Cell id to annotate")
    annotate_p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="set_",
        metavar="KEY=VALUE",
        help="Set a scalar annotation, e.g. --set worker=gpu-box (repeatable)",
    )
    annotate_p.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="KEY",
        help="Remove an annotation directive, e.g. --unset worker (repeatable)",
    )
    annotate_p.add_argument("--format", choices=["human", "json"], default="json")
    annotate_p.set_defaults(func=cell_annotate_main)

    pin_p = sub.add_parser(
        "pin-fetch",
        help="Pin a cell's `# @fetch` inputs to the bytes last downloaded (writes sha256=)",
    )
    pin_p.add_argument("notebook_dir", help="Path to the notebook directory")
    pin_p.add_argument("cell_id", help="Cell id whose fetches to pin")
    pin_p.add_argument(
        "names", nargs="*", metavar="NAME", help="Fetch names to pin (default: all in the cell)"
    )
    pin_p.add_argument("--author", default=None, help="Who to credit for the edit")
    pin_p.add_argument("--format", choices=["human", "json"], default="json")
    pin_p.set_defaults(func=cell_pin_fetch_main)

    # `strata cell` with no action → help.
    parser.set_defaults(func=lambda args: (parser.print_help(), 0)[1])


def cell_list_main(args: argparse.Namespace) -> int:
    with _read_ops(args) as ops:
        if ops is None:
            return 2
        cells = ops.list_cells()
        if args.format == "json":
            _emit_json([cell.model_dump(mode="json") for cell in cells])
        else:
            for cell in cells:
                print(f"{cell.status:8} {cell.id:18} {cell.name}")
        return 0


def _print_cell_human(cell: CellView) -> None:
    print(f"id:       {cell.id}")
    print(f"name:     {cell.name}")
    print(f"language: {cell.language}")
    print(f"status:   {cell.status}")
    if cell.staleness_reasons:
        print(f"stale:    {', '.join(cell.staleness_reasons)}")
    if cell.error:
        # Console and outputs stay JSON-only, but an error status is useless without
        # its traceback.
        print("--- error ---")
        print(cell.error.rstrip())
    print("--- source ---")
    print(cell.source)


def cell_output_main(args: argparse.Namespace) -> int:
    """Write one display output to a file, so an agent can open the plot."""
    from strata.notebook.ops import NotebookOpsError

    dest = Path(args.out).expanduser()
    if not dest.parent.exists():
        print(f"error: no such directory: {dest.parent}", file=sys.stderr)
        return 2

    with _read_ops(args) as ops:
        if ops is None:
            return 2
        try:
            saved = ops.save_output(args.cell_id, dest, index=args.index)
        except NotebookOpsError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1
        except OSError as e:
            # An existing parent doesn't mean writable: `--out /tmp` is a directory, or the
            # target may be read-only.
            print(f"error: cannot write {dest}: {e.strerror or e}", file=sys.stderr)
            return 2
        if args.format == "json":
            _emit_json(saved.model_dump(mode="json"))
        else:
            print(f"{saved.path}  ({saved.content_type}, {saved.bytes} bytes)")
    return 0


def cell_show_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    var = getattr(args, "var", None)
    # argparse fills notebook_dir first in `cell show --server … <id>`, leaving
    # cell_id None. Remote ignores notebook_dir, so recover the id from it.
    if args.server and args.cell_id is None and args.notebook_dir is not None:
        args.cell_id = args.notebook_dir
    if bool(args.cell_id) == bool(var):
        print("error: provide either a cell_id or --var NAME (exactly one)", file=sys.stderr)
        return 2

    with _read_ops(args) as ops:
        if ops is None:
            return 2
        if var is not None:
            return _cell_show_var(ops, var, args.format)
        try:
            cell = ops.get_cell(args.cell_id)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
        if args.format == "json":
            _emit_json(cell.model_dump(mode="json"))
        else:
            _print_cell_human(cell)
        return 0


def _cell_show_var(ops: Any, var: str, fmt: str) -> int:
    """Show the cell that defines *var*, via ``dag`` plus ``get_cell``."""
    from strata.notebook.ops import NotebookOpsError

    try:
        producers = ops.dag().variable_producer
    except NotebookOpsError as exc:
        return _emit_op_error(exc, fmt)
    producer = producers.get(var)
    if producer is None:
        # Not defined: list what is available, so the miss doubles as discovery.
        if fmt == "json":
            _emit_json({"variable": var, "defined": False, "available": sorted(producers)})
        else:
            print(f"'{var}' is not defined")
            if producers:
                print(f"available: {', '.join(sorted(producers))}")
        return 0
    if producer.startswith(("sweep:", "fanout:")):
        # A variant/sweep group has no single producing cell; report the pointer only.
        if fmt == "json":
            _emit_json({"variable": var, "defined": True, "defined_in": producer})
        else:
            print(f"variable: {var}  (defined in {producer})")
        return 0
    try:
        cell = ops.get_cell(producer)
    except NotebookOpsError as exc:
        # A real error (stale DAG pointer, backend failure); don't mask it as "defined".
        return _emit_op_error(exc, fmt)
    if fmt == "json":
        _emit_json(
            {
                "variable": var,
                "defined": True,
                "defined_in": cell.id,
                "cell": cell.model_dump(mode="json"),
            }
        )
    else:
        print(f"variable: {var}  (defined in {cell.id})")
        _print_cell_human(cell)
    return 0


def _read_source_arg(path: str) -> str:
    """Read cell source from *path*, or stdin when ``path == "-"``."""
    if path == "-":
        return sys.stdin.read()
    return Path(path).expanduser().read_text(encoding="utf-8")


def _emit_op_error(exc: Exception, fmt: str) -> int:
    if fmt == "json":
        _emit_json({"error": str(exc)})
    else:
        print(f"error: {exc}", file=sys.stderr)
    return 1


def cell_add_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    if args.code is not None:
        source = args.code
    else:
        try:
            source = _read_source_arg(args.file)
        except OSError as exc:
            print(f"error: cannot read --file: {exc}", file=sys.stderr)
            return 2

    if args.run:
        import asyncio

        with _quiet_notebook_logs():
            return asyncio.run(_cell_add_run_async(args, source))

    with _read_ops(args) as ops:
        if ops is None:
            return 2
        try:
            cell = ops.add_cell(source, after=args.after, language=args.language)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
        if args.format == "json":
            _emit_json(cell.model_dump(mode="json"))
        else:
            print(f"added {cell.id}  {cell.name}")
        return 0


async def _cell_add_run_async(args: argparse.Namespace, source: str) -> int:
    """Add a cell and run it in one call; the run result is nested under ``run``.

    As in ``cell run``, the local backend syncs its venv first; a remote server
    owns its own.
    """
    from strata.notebook.ops import NotebookOpsError

    is_remote = bool(args.server)
    ops = _open_read_ops(args)
    if ops is None:
        return 2
    try:
        try:
            cell = ops.add_cell(source, after=args.after, language=args.language)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
        if not is_remote:
            rc = await _prepare_env_for_ops(ops, args)
            if rc != 0:
                return rc
        try:
            result = await ops.run_cell(cell.id, mode="normal")
            # Re-fetch so the payload carries rendered outputs; add_cell's pre-run view has none.
            cell = ops.get_cell(cell.id)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
    finally:
        if is_remote:
            _close_ops(ops)
        else:
            await ops.aclose()

    if args.format == "json":
        payload = cell.model_dump(mode="json")
        payload["run"] = result.model_dump(mode="json")
        _emit_json(payload)
    else:
        print(f"added {cell.id}  {cell.name}")
        timing = f"{result.execution_method}, {_format_ms(result.duration_ms)}"
        print(f"{result.status}  {result.cell_id}  ({timing})")
        if result.stdout:
            print("--- stdout ---")
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.error:
            print("--- error ---")
            print(result.error)
    return 0 if result.status == "ok" else 1


def cell_edit_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    try:
        source = _read_source_arg(args.file)
    except OSError as exc:
        print(f"error: cannot read --file: {exc}", file=sys.stderr)
        return 2
    with _read_ops(args) as ops:
        if ops is None:
            return 2
        try:
            cell = ops.edit_cell(args.cell_id, source)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
        if args.format == "json":
            _emit_json(cell.model_dump(mode="json"))
        else:
            print(f"edited {cell.id}  {cell.name}")
        return 0


def cell_rm_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    with _read_ops(args) as ops:
        if ops is None:
            return 2
        try:
            ops.remove_cell(args.cell_id)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
        if args.format == "json":
            _emit_json({"removed": args.cell_id})
        else:
            print(f"removed {args.cell_id}")
        return 0


def cell_mv_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    with _read_ops(args) as ops:
        if ops is None:
            return 2
        try:
            cells = ops.move_cell(args.cell_id, args.to)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
        if args.format == "json":
            _emit_json([cell.model_dump(mode="json") for cell in cells])
        else:
            print("  ".join(cell.id for cell in cells))
        return 0


def _valid_annotation_key(key: str) -> bool:
    return bool(key) and all(c.isalnum() or c == "_" for c in key)


def cell_pin_fetch_main(args: argparse.Namespace) -> int:
    """Write ``sha256=`` into a cell's ``# @fetch`` lines from the notebook's fetch cache.

    Local only: the digests are those of the bytes this notebook last downloaded
    (``.strata/fetch/``), the ones a run would read. Re-pinning after a
    ``fetch_pin_mismatch`` accepts the bytes the URL serves now.
    """
    from strata.notebook.annotations import parse_annotations, pin_fetch_directives
    from strata.notebook.fetch import FetchCache
    from strata.notebook.ops import NotebookOpsError

    ops = _open_local_ops(args.notebook_dir, author=args.author)
    if ops is None:
        return 2
    try:
        cell = ops.get_cell(args.cell_id)
    except NotebookOpsError as exc:
        return _emit_op_error(exc, args.format)

    specs = parse_annotations(cell.source).fetches
    unknown = sorted(set(args.names) - {spec.name for spec in specs})
    if unknown:
        print(
            f"error: cell {args.cell_id} has no @fetch named {', '.join(unknown)}", file=sys.stderr
        )
        return 2
    wanted = [spec for spec in specs if not args.names or spec.name in args.names]
    if not wanted:
        print(f"error: cell {args.cell_id} has no @fetch to pin", file=sys.stderr)
        return 2

    cache = FetchCache(Path(args.notebook_dir).expanduser().resolve())
    pinned: list[dict[str, str]] = []
    for spec in wanted:
        recorded = cache.recorded(spec.url)
        if recorded is None:
            print(
                f"error: nothing downloaded yet for @fetch {spec.name} ({spec.url}); "
                "run the cell first",
                file=sys.stderr,
            )
            return 2
        pinned.append({"name": spec.name, "url": spec.url, "sha256": recorded.sha256})

    source = pin_fetch_directives(cell.source, {p["name"]: p["sha256"] for p in pinned})
    if source != cell.source:
        try:
            cell = ops.edit_cell(args.cell_id, source)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
    if args.format == "json":
        _emit_json({"cell": cell.model_dump(mode="json"), "pinned": pinned})
    else:
        for p in pinned:
            print(f"pinned {p['name']}  sha256={p['sha256']}")
    return 0


def cell_annotate_main(args: argparse.Namespace) -> int:
    """Splice ``# @key`` directives into a cell's source, preserving the body.

    Built on ``get_cell`` + ``edit_cell``, so it works locally or against
    ``--server/--session``.
    """
    if not args.set_ and not args.unset:
        print("error: provide at least one --set KEY=VALUE or --unset KEY", file=sys.stderr)
        return 2

    sets: list[tuple[str, str]] = []
    for item in args.set_:
        key, sep, value = item.partition("=")
        key = key.strip()
        if not sep or not _valid_annotation_key(key):
            print(f"error: --set expects a valid KEY=VALUE, got {item!r}", file=sys.stderr)
            return 2
        sets.append((key, value))
    for key in args.unset:
        if not _valid_annotation_key(key.strip()):
            print(f"error: invalid annotation key {key!r}", file=sys.stderr)
            return 2

    from strata.notebook.annotations import (
        remove_annotation_directive,
        set_annotation_directive,
    )
    from strata.notebook.ops import NotebookOpsError

    with _read_ops(args) as ops:
        if ops is None:
            return 2
        try:
            source = ops.get_cell(args.cell_id).source
            for key, value in sets:
                source = set_annotation_directive(source, key, value)
            for key in args.unset:
                source = remove_annotation_directive(source, key.strip())
            cell = ops.edit_cell(args.cell_id, source)
        except ValueError as exc:  # e.g. a repeatable directive passed to --set
            print(f"error: {exc}", file=sys.stderr)
            return 2
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
        if args.format == "json":
            _emit_json(cell.model_dump(mode="json"))
        else:
            print(f"annotated {cell.id}  {cell.name}")
        return 0


def add_dep_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``strata dep <action>`` group (add, rm)."""
    sub = parser.add_subparsers(dest="dep_command", metavar="<action>")
    for action, verb in (("add", "Add"), ("rm", "Remove")):
        dep_p = sub.add_parser(action, help=f"{verb} a Python dependency")
        _add_target_args(dep_p)
        dep_p.add_argument("package", help="Package spec, e.g. 'pandas' or 'pandas>=2'")
        dep_p.add_argument("--format", choices=["human", "json"], default="json")
        dep_p.set_defaults(func=dep_add_main if action == "add" else dep_rm_main)
    parser.set_defaults(func=lambda args: (parser.print_help(), 0)[1])


# ---------------------------------------------------------------------------
# Worker registration commands (NotebookOps local backend): `strata worker …`
# Writes `[[workers]]` to notebook.toml offline; a running server picks them up
# on reload. A live session should use the MCP worker tools (reload + broadcast).
# ---------------------------------------------------------------------------


def _emit_worker_list(view: WorkerListView, fmt: str) -> None:
    """Print a WorkerListView as JSON, or a compact default-marked table."""
    if fmt == "json":
        _emit_json(view.model_dump(mode="json"))
        return
    for worker in view.workers:
        mark = "*" if worker.is_default else " "
        print(f"{mark} {worker.name:<16} {worker.transport:<9} {worker.url or ''}")


def worker_ls_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    ops = _open_local_ops(args.notebook_dir)
    if ops is None:
        return 2
    try:
        view = ops.list_workers()
    except NotebookOpsError as exc:
        return _emit_op_error(exc, args.format)
    _emit_worker_list(view, args.format)
    return 0


def worker_add_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    ops = _open_local_ops(args.notebook_dir)
    if ops is None:
        return 2
    try:
        view = ops.add_worker(
            args.name,
            url=args.url,
            transport=args.transport,
            runtime_id=args.runtime_id,
            token_env=args.token_env,
            set_default=args.default,
        )
    except NotebookOpsError as exc:
        return _emit_op_error(exc, args.format)
    _emit_worker_list(view, args.format)
    return 0


def worker_rm_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    ops = _open_local_ops(args.notebook_dir)
    if ops is None:
        return 2
    try:
        view = ops.remove_worker(args.name)
    except NotebookOpsError as exc:
        return _emit_op_error(exc, args.format)
    _emit_worker_list(view, args.format)
    return 0


def worker_default_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError

    ops = _open_local_ops(args.notebook_dir)
    if ops is None:
        return 2
    try:
        view = ops.set_default_worker(args.name)
    except NotebookOpsError as exc:
        return _emit_op_error(exc, args.format)
    _emit_worker_list(view, args.format)
    return 0


def worker_add_ssh_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError, RemoteNotebookOps

    ops = RemoteNotebookOps(args.server, args.session)
    try:
        data = ops.add_ssh_worker(
            args.ssh_target,
            name=args.name,
            set_default=not args.no_default,
            install=not args.no_install,
        )
    except NotebookOpsError as exc:
        return _emit_op_error(exc, args.format)
    finally:
        ops.close()
    if args.format == "json":
        _emit_json(data)
    else:
        worker = data.get("worker", {})
        print(f"connected {worker.get('name')} → {worker.get('ssh_target')}")
    return 0


def worker_rm_ssh_main(args: argparse.Namespace) -> int:
    from strata.notebook.ops import NotebookOpsError, RemoteNotebookOps

    ops = RemoteNotebookOps(args.server, args.session)
    try:
        data = ops.remove_ssh_worker(args.name, stop_remote=args.stop_remote)
    except NotebookOpsError as exc:
        return _emit_op_error(exc, args.format)
    finally:
        ops.close()
    if args.format == "json":
        _emit_json(data)
    else:
        print(f"disconnected {args.name} (tunnel torn down: {data.get('torn_down')})")
    return 0


def add_worker_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``strata worker <action>`` group (ls, add, rm, default, add-ssh)."""
    sub = parser.add_subparsers(dest="worker_command", metavar="<action>")

    ls_p = sub.add_parser("ls", help="List registered workers and the default")
    ls_p.add_argument("notebook_dir", help="Path to the notebook directory")
    ls_p.add_argument("--format", choices=["human", "json"], default="json")
    ls_p.set_defaults(func=worker_ls_main)

    add_p = sub.add_parser("add", help="Register an executor worker")
    add_p.add_argument("notebook_dir", help="Path to the notebook directory")
    add_p.add_argument("name", help="Worker name (referenced by # @worker <name>)")
    add_p.add_argument(
        "--url",
        required=True,
        help="Executor endpoint, e.g. http://127.0.0.1:9000/v1/execute",
    )
    add_p.add_argument(
        "--transport", default="direct", help="Transport: direct or signed (default: direct)"
    )
    add_p.add_argument(
        "--runtime-id", dest="runtime_id", help="Stable env fingerprint for provenance"
    )
    add_p.add_argument(
        "--token-env", dest="token_env", help="Env var holding the worker's bearer token"
    )
    add_p.add_argument("--default", action="store_true", help="Also set as the notebook default")
    add_p.add_argument("--format", choices=["human", "json"], default="json")
    add_p.set_defaults(func=worker_add_main)

    rm_p = sub.add_parser("rm", help="Remove a worker")
    rm_p.add_argument("notebook_dir", help="Path to the notebook directory")
    rm_p.add_argument("name", help="Worker name to remove")
    rm_p.add_argument("--format", choices=["human", "json"], default="json")
    rm_p.set_defaults(func=worker_rm_main)

    default_p = sub.add_parser("default", help="Set (or clear) the default worker")
    default_p.add_argument("notebook_dir", help="Path to the notebook directory")
    default_p.add_argument("name", nargs="?", help="Worker name; omit or 'local' to clear")
    default_p.add_argument("--format", choices=["human", "json"], default="json")
    default_p.set_defaults(func=worker_default_main)

    # These drive a running server (the tunnel is server-owned), so they take
    # --server/--session instead of a notebook directory.
    add_ssh_p = sub.add_parser(
        "add-ssh", help="Provision + tunnel a worker over SSH (needs a running server)"
    )
    add_ssh_p.add_argument("ssh_target", help="SSH target, e.g. user@gpu-box")
    add_ssh_p.add_argument(
        "--server", required=True, help="Base URL of the running notebook server"
    )
    add_ssh_p.add_argument(
        "--session", required=True, help="Open session id to attach the worker to"
    )
    add_ssh_p.add_argument("--name", default=None, help="Worker name (default: derived from host)")
    add_ssh_p.add_argument(
        "--no-default", action="store_true", help="Don't make it the notebook default"
    )
    add_ssh_p.add_argument(
        "--no-install", action="store_true", help="Fail if strata-worker isn't already on the box"
    )
    add_ssh_p.add_argument("--format", choices=["human", "json"], default="json")
    add_ssh_p.set_defaults(func=worker_add_ssh_main)

    rm_ssh_p = sub.add_parser("rm-ssh", help="Tear down an SSH worker's tunnel + registration")
    rm_ssh_p.add_argument("name", help="Worker name to disconnect")
    rm_ssh_p.add_argument("--server", required=True, help="Base URL of the running notebook server")
    rm_ssh_p.add_argument("--session", required=True, help="Open session id the worker is on")
    rm_ssh_p.add_argument(
        "--stop-remote", action="store_true", help="Also stop the remote strata-worker process"
    )
    rm_ssh_p.add_argument("--format", choices=["human", "json"], default="json")
    rm_ssh_p.set_defaults(func=worker_rm_ssh_main)

    parser.set_defaults(func=lambda args: (parser.print_help(), 0)[1])


def dep_add_main(args: argparse.Namespace) -> int:
    import asyncio

    return asyncio.run(_dep_async(args, "add"))


def dep_rm_main(args: argparse.Namespace) -> int:
    import asyncio

    return asyncio.run(_dep_async(args, "remove"))


async def _dep_async(args: argparse.Namespace, action: str) -> int:
    is_remote = bool(args.server)
    ops = _open_read_ops(args)
    if ops is None:
        return 2
    from strata.notebook.ops import NotebookOpsError

    try:
        try:
            if action == "add":
                result = await ops.add_dependency(args.package)
            else:
                result = await ops.remove_dependency(args.package)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
    finally:
        if is_remote:
            ops.close()
        else:
            await ops.aclose()
    if args.format == "json":
        _emit_json(result.model_dump(mode="json"))
    else:
        tail = "ok" if result.success else f"failed: {result.error or ''}"
        print(f"{action} {result.package}: {tail}")
    return 0 if result.success else 1


async def _prepare_env_for_ops(ops: object, args: argparse.Namespace) -> int:
    """Sync or (``--no-sync``) verify the notebook venv; return 0 ok, 2 setup failure.

    Setup failures print to stderr.
    """
    from strata.notebook.ops import LocalNotebookOps, NotebookOpsError

    # Every caller skips this for a remote server, which syncs its own venv.
    assert isinstance(ops, LocalNotebookOps)

    try:
        if args.no_sync:
            ops.use_existing_environment()
            return 0
        if args.format == "human":
            print(_dim("syncing environment…"))
        await ops.sync_environment()
    except NotebookOpsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def cell_run_main(args: argparse.Namespace) -> int:
    import asyncio

    with _quiet_notebook_logs():
        return asyncio.run(_cell_run_async(args))


async def _cell_run_async(args: argparse.Namespace) -> int:
    is_remote = bool(args.server)
    ops = _open_read_ops(args)
    if ops is None:
        return 2
    from strata.notebook.ops import NotebookOpsError

    try:
        # A remote server owns its own venv.
        if not is_remote:
            rc = await _prepare_env_for_ops(ops, args)
            if rc != 0:
                return rc
        mode = "force" if args.force else "rerun" if args.rerun else "normal"
        try:
            result = await ops.run_cell(args.cell_id, mode=mode)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
    finally:
        if is_remote:
            ops.close()
        else:
            await ops.aclose()

    if args.format == "json":
        _emit_json(result.model_dump(mode="json"))
    else:
        timing = f"{result.execution_method}, {_format_ms(result.duration_ms)}"
        print(f"{result.status}  {result.cell_id}  ({timing})")
        if result.stdout:
            print("--- stdout ---")
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.error:
            print("--- error ---")
            print(result.error)
    return 0 if result.status == "ok" else 1


def cell_test_main(args: argparse.Namespace) -> int:
    import asyncio

    return asyncio.run(_cell_test_async(args))


async def _cell_test_async(args: argparse.Namespace) -> int:
    is_remote = bool(args.server)
    ops = _open_read_ops(args)
    if ops is None:
        return 2
    from strata.notebook.ops import NotebookOpsError

    try:
        # --file writes the cell's test source before running it.
        if args.file is not None:
            try:
                test_source = _read_source_arg(args.file)
            except OSError as exc:
                print(f"error: cannot read --file: {exc}", file=sys.stderr)
                return 2
            try:
                ops.set_cell_tests(args.cell_id, test_source)
            except NotebookOpsError as exc:
                return _emit_op_error(exc, args.format)
        if not is_remote:
            rc = await _prepare_env_for_ops(ops, args)
            if rc != 0:
                return rc
        try:
            result = await ops.run_tests(args.cell_id)
        except NotebookOpsError as exc:
            return _emit_op_error(exc, args.format)
    finally:
        if is_remote:
            ops.close()
        else:
            await ops.aclose()

    if args.format == "json":
        _emit_json(result.model_dump(mode="json"))
    else:
        glyphs = {"passed": "✓", "failed": "✗", "error": "⚠", "skipped": "○"}
        for case in result.cases:
            print(f"{glyphs.get(case.outcome, '?')} {case.name}")
            if case.message and case.outcome in ("failed", "error"):
                for line in case.message.splitlines():
                    print(f"    {line}")
        print(
            f"{result.passed} passed, {result.failed} failed, "
            f"{result.errored} errored, {result.skipped} skipped"
        )
    if result.pytest_unavailable:
        return 2
    return 1 if (result.failed or result.errored) else 0


def add_dag_arguments(parser: argparse.ArgumentParser) -> None:
    _add_target_args(parser)
    parser.add_argument("--format", choices=["human", "json"], default="json")


def dag_main(args: argparse.Namespace) -> int:
    ops = _open_read_ops(args)
    if ops is None:
        return 2
    dag = ops.dag()
    if args.format == "json":
        _emit_json(dag.model_dump(mode="json"))
    else:
        for edge in dag.edges:
            print(f"{edge.from_cell_id} → {edge.to_cell_id}  ({edge.variable})")
        print(f"topo: {' → '.join(dag.topological_order)}")
    return 0


def add_status_arguments(parser: argparse.ArgumentParser) -> None:
    _add_target_args(parser)
    parser.add_argument("--format", choices=["human", "json"], default="json")


def status_main(args: argparse.Namespace) -> int:
    ops = _open_read_ops(args)
    if ops is None:
        return 2
    status = ops.status()
    if args.format == "json":
        _emit_json(status.model_dump(mode="json"))
    else:
        print(f"notebook: {status.name}  ({status.notebook_id})")
        for cell in status.cells:
            stale = " ·stale" if cell.staleness_reasons else ""
            print(f"  {cell.status:8} {cell.id:18} {cell.name}{stale}")
    return 0
