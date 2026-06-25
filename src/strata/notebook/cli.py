"""Headless notebook runner.

Implements ``strata run <notebook_dir>`` — parse a notebook directory,
optionally sync its uv-managed venv, execute every cell in topological
order, and report success/failure. Reuses ``NotebookSession`` and
``CellExecutor`` directly so the CLI takes the same code path the UI
does, without an intervening HTTP server.

Exit codes:
    0  all cells succeeded
    1  one or more cells failed
    2  invocation / setup error (bad path, env sync failed, etc.)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from strata.notebook.models import CellLanguage

# ANSI colors for human output. Disabled when stdout isn't a tty so that
# pipes and CI logs stay clean.
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
    """Human-readable short label for a cell.

    Uses the first non-blank, non-comment line of source, truncated.
    Falls back to "(empty)" for blank cells. This is a cosmetic field;
    cells are always uniquely identified by their ID.
    """
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


# Per-cell stdout/stderr cap in the JSON payload. Generous enough for
# verification output, bounded so a print-heavy cell can't balloon the
# result document.
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


async def _sync_environment(session: Any) -> tuple[bool, str | None]:
    """Run `uv sync` via the session's environment job machinery.

    Returns ``(ok, error_message)``.
    """
    try:
        job = await session.submit_environment_job(action="sync")
    except Exception as exc:
        return False, f"failed to submit env sync job: {exc}"

    try:
        await session.wait_for_environment_job()
    except Exception as exc:
        return False, f"env sync raised: {exc}"

    # ``submit_environment_job`` returns the job snapshot, and
    # ``_run_environment_job`` mutates *that same object* in place to its
    # terminal status. Read it directly. ``session.environment_job`` is
    # the "currently-running" slot and is reset to None the moment the
    # job finishes — so reading it here always saw None and tripped a
    # false "env sync finished without a status snapshot" error, which
    # made ``strata run`` (without --no-sync) fail on every notebook.
    if job.status != "completed":
        message = job.error or f"env sync ended with status={job.status}"
        return False, message
    return True, None


async def _drain_warm_pool(session: Any) -> None:
    """Release the warm process pool if one was initialized.

    Safe to call regardless of whether a pool exists; silently swallows
    any drain errors since we're on the shutdown path anyway.
    """
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


async def _run_async(args: argparse.Namespace) -> int:
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
    from strata.notebook.executor import CellExecutor
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession

    try:
        state = parse_notebook(notebook_dir)
        session = NotebookSession(state, notebook_dir)
    except Exception as exc:
        print(f"error: failed to open notebook: {exc}", file=sys.stderr)
        return 2

    if session.dag is None:
        print(
            "error: notebook DAG has a cycle or failed to build — "
            "inspect the notebook in the UI and resolve the cycle first",
            file=sys.stderr,
        )
        return 2

    # Environment: either sync now, or verify the user's prepared venv exists.
    if args.no_sync:
        venv_dir = notebook_dir / ".venv"
        if not venv_dir.exists():
            print(
                f"error: notebook has no .venv at {venv_dir}\n"
                f"hint: run without --no-sync, or run `uv sync` in the notebook "
                f"directory first",
                file=sys.stderr,
            )
            return 2
    else:
        if args.format == "human":
            print(_dim("syncing environment…"))
        ok, err = await _sync_environment(session)
        if not ok:
            print(f"error: {err}", file=sys.stderr)
            await _drain_warm_pool(session)
            return 2

    # Restore the R environment from renv.lock, mirroring the server's
    # session-open behaviour. Idempotent and cheap when the project
    # library already matches the lockfile (it skips the Rscript spawn),
    # and a no-op for Python-only notebooks — so it runs on the --no-sync
    # path too. Without it, an R notebook that ships an renv.lock would
    # execute its cells against an empty project library and fail with
    # "there is no package called …". Runs in a thread because
    # ``_renv_sync`` shells out to Rscript synchronously.
    if (notebook_dir / "renv.lock").exists():
        if args.format == "human":
            print(_dim("restoring R environment…"))
        # ``ensure_renv_synced`` swallows the expected failures (Rscript
        # missing, timeout, non-zero restore) and records them as runtime
        # state. Guard the unexpected ones (e.g. a non-executable Rscript
        # raising) so they surface as a clean exit-2 setup error rather
        # than an uncaught traceback.
        try:
            await asyncio.to_thread(session.ensure_renv_synced)
        except Exception as exc:
            print(f"error: R environment restore failed: {exc}", file=sys.stderr)
            await _drain_warm_pool(session)
            return 2

    # Header
    if args.format == "human":
        print(f"running: {notebook_dir}")
        print()

    executor = CellExecutor(session)
    cell_by_id = {c.id: c for c in session.notebook_state.cells}
    results: list[dict[str, Any]] = []
    failed_cells: set[str] = set()
    start = time.monotonic()

    for cell_id in session.dag.topological_order:
        cell = cell_by_id.get(cell_id)
        if cell is None:
            # Cell in the DAG but not in notebook_state — shouldn't happen,
            # but don't crash.
            continue

        # Markdown cells are non-executable prose; surface them as
        # success-with-no-op so ``strata run`` doesn't print a misleading
        # "skipped: unsupported language" line for documentation cells.
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

        # Skip languages we can't execute headlessly. R cells run through
        # the same language-executor dispatch the session uses (Rscript +
        # harness.R); a missing `Rscript` surfaces as a clean cell error,
        # not a crash, so R belongs in the executable set rather than the
        # skip list.
        if cell.language not in {
            CellLanguage.PYTHON,
            CellLanguage.PROMPT,
            CellLanguage.SQL,
            CellLanguage.R,
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

        # Skip if any upstream failed.
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
            if args.force:
                result = await executor.execute_cell_force(cell_id, cell.source)
            else:
                result = await executor.execute_cell(cell_id, cell.source)
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

        entry = {
            "id": cell_id,
            "label": _cell_label(cell.source),
            "status": "ok" if result.success else "error",
            "duration_ms": int(result.duration_ms or 0),
            "cache_hit": bool(result.cache_hit),
        }
        # Carry console output so external authors (scripts, coding
        # agents) can verify computed values from the JSON payload
        # instead of reaching into .strata/ — which is documented as
        # hands-off (issue #114 litmus finding). Cache hits replay the
        # stored result without re-emitting console output, so these
        # keys can be absent on warm runs.
        if result.stdout:
            entry["stdout"] = _truncate_console(result.stdout)
        if result.stderr:
            entry["stderr"] = _truncate_console(result.stderr)
        if not result.success:
            entry["error"] = result.error or "cell failed"
            failed_cells.add(cell_id)
        results.append(entry)
        if args.format == "human" and not args.quiet:
            _print_cell_line(entry)

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
    """Entry point for ``strata run``.

    Can be called directly (``run_main(["./my-notebook"])``) or as a
    subcommand dispatched from :mod:`strata.cli`.
    """
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

    Static checks only — nothing executes, no environment is synced:

    * ``notebook.toml`` parses and the cell files load
    * the DAG builds without cycles
    * per-cell annotation diagnostics (the same validation the server
      runs on open / reload)

    Exit codes mirror ``strata run``: 0 valid (warnings allowed),
    1 invalid (parse failure, DAG cycle, or any error-severity
    diagnostic), 2 invocation error (bad path). Built for the
    agent feedback loop (issue #114): write files → validate → fix →
    run.
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
        notebook_errors.append(
            {
                "code": "dag_cycle",
                "message": (
                    "notebook DAG has a cycle — two or more cells consume "
                    "each other's variables; break the cycle by renaming or "
                    "removing one of the circular references"
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
            print(f"{_green('✓')} valid — {len(cells_payload)} cell(s){suffix}")
        else:
            print(f"{_red('✗')} invalid — {total_errors} error(s), {warning_count} warning(s)")

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
        "--format",
        choices=["human", "json"],
        default="human",
        help="Output format (default: human)",
    )


def new_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata new``.

    Scaffolds a notebook directory (notebook.toml + pyproject.toml +
    cells/) so external tools and coding agents don't hand-roll the
    TOML (issue #114). Idempotent on an existing notebook directory:
    the notebook ID and any existing cells are preserved.
    """
    from strata.notebook.writer import create_notebook

    try:
        notebook_dir = create_notebook(
            Path(args.parent).expanduser().resolve(),
            args.name,
            args.python_version,
            initialize_environment=not args.no_env,
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
        choices=["markdown", "html"],
        default="markdown",
        help="Output format (default: markdown)",
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
        help="Path to a Jupyter .ipynb file",
    )
    parser.add_argument(
        "--out",
        dest="output_path",
        default=None,
        help=(
            "Target notebook directory. Defaults to a sibling directory "
            "named after the .ipynb file stem."
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


def import_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata import``.

    Loads a Jupyter ``.ipynb`` file, converts cells, and writes a
    Strata notebook directory ready to be opened with the server or
    executed with ``strata run``.
    """
    from strata.notebook.jupyter_import import import_notebook

    path = Path(args.path)
    if not path.is_file():
        print(f"error: {path} is not a file", file=sys.stderr)
        return 2
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


def export_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata export``.

    Loads the notebook directory, renders it via
    :func:`strata.notebook.export.export_notebook`, and writes the
    result to stdout (default) or to the ``--out`` path.
    """
    from strata.notebook.export import ExportFormat, ExportOptions, export_notebook

    path = Path(args.path)
    if not (path / "notebook.toml").is_file():
        print(f"error: {path} is not a notebook directory (no notebook.toml)", file=sys.stderr)
        return 2

    options = ExportOptions(
        output_format=ExportFormat(args.output_format),
        include_inactive_variants=bool(args.include_inactive_variants),
        include_console=not bool(args.no_console),
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
# Agent inspect commands (NotebookOps, local backend) — `strata cell …` etc.
#
# Read-only P0 of the CLI-hardening phase: a full-feature agent tool over the
# same NotebookOps core the MCP server will reuse. JSON by default (agent-first);
# `--format human` gives a compact view. See docs/internal/design-cli-hardening.md.
# ---------------------------------------------------------------------------


def _open_local_ops(notebook_dir_arg: str):
    """Open a :class:`LocalNotebookOps` for *notebook_dir_arg*, or None on error.

    Prints the error to stderr; callers return exit 2 on None.
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
        return LocalNotebookOps(notebook_dir)
    except Exception as exc:  # noqa: BLE001 — surface any open failure as exit 2
        print(f"error: failed to open notebook: {exc}", file=sys.stderr)
        return None


def _emit_json(data: object) -> None:
    print(json.dumps(data, indent=2, default=str))


def add_cell_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``strata cell <action>`` group (P0: list, show)."""
    sub = parser.add_subparsers(dest="cell_command", metavar="<action>")

    list_p = sub.add_parser("list", help="List cells (id, name, status)")
    list_p.add_argument("notebook_dir", help="Path to the notebook directory")
    list_p.add_argument("--format", choices=["human", "json"], default="json")
    list_p.set_defaults(func=cell_list_main)

    show_p = sub.add_parser("show", help="Show one cell: source, status, outputs, console")
    show_p.add_argument("notebook_dir", help="Path to the notebook directory")
    show_p.add_argument("cell_id", help="Cell id to show")
    show_p.add_argument("--format", choices=["human", "json"], default="json")
    show_p.set_defaults(func=cell_show_main)

    # `strata cell` with no action → help.
    parser.set_defaults(func=lambda args: (parser.print_help(), 0)[1])


def cell_list_main(args: argparse.Namespace) -> int:
    ops = _open_local_ops(args.notebook_dir)
    if ops is None:
        return 2
    cells = ops.list_cells()
    if args.format == "json":
        _emit_json([cell.model_dump(mode="json") for cell in cells])
    else:
        for cell in cells:
            print(f"{cell.status:8} {cell.id:18} {cell.name}")
    return 0


def cell_show_main(args: argparse.Namespace) -> int:
    ops = _open_local_ops(args.notebook_dir)
    if ops is None:
        return 2
    from strata.notebook.ops import NotebookOpsError

    try:
        cell = ops.get_cell(args.cell_id)
    except NotebookOpsError as exc:
        if args.format == "json":
            _emit_json({"error": str(exc)})
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.format == "json":
        _emit_json(cell.model_dump(mode="json"))
    else:
        print(f"id:       {cell.id}")
        print(f"name:     {cell.name}")
        print(f"language: {cell.language}")
        print(f"status:   {cell.status}")
        if cell.staleness_reasons:
            print(f"stale:    {', '.join(cell.staleness_reasons)}")
        print("--- source ---")
        print(cell.source)
    return 0


def add_dag_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("notebook_dir", help="Path to the notebook directory")
    parser.add_argument("--format", choices=["human", "json"], default="json")


def dag_main(args: argparse.Namespace) -> int:
    ops = _open_local_ops(args.notebook_dir)
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
    parser.add_argument("notebook_dir", help="Path to the notebook directory")
    parser.add_argument("--format", choices=["human", "json"], default="json")


def status_main(args: argparse.Namespace) -> int:
    ops = _open_local_ops(args.notebook_dir)
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
