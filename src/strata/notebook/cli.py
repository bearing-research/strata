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
        await session.submit_environment_job(action="sync")
    except Exception as exc:
        return False, f"failed to submit env sync job: {exc}"

    try:
        await session.wait_for_environment_job()
    except Exception as exc:
        return False, f"env sync raised: {exc}"

    job = session.environment_job
    if job is None:
        return False, "env sync finished without a status snapshot"
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
        if cell.language == "markdown":
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

        # Skip languages we can't execute headlessly.
        if cell.language not in {"python", "prompt", "sql"}:
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


def export_main(args: argparse.Namespace) -> int:
    """Entry point for ``strata export``.

    Loads the notebook directory, renders it via
    :func:`strata.notebook.export.export_notebook`, and writes the
    result to stdout (default) or to the ``--out`` path.
    """
    from strata.notebook.export import ExportOptions, export_notebook

    path = Path(args.path)
    if not (path / "notebook.toml").is_file():
        print(f"error: {path} is not a notebook directory (no notebook.toml)", file=sys.stderr)
        return 2

    options = ExportOptions(
        output_format=args.output_format,
        include_inactive_variants=bool(args.include_inactive_variants),
        include_console=not bool(args.no_console),
    )
    rendered = export_notebook(path, options)

    out_path = args.output_path
    if out_path:
        Path(out_path).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0
