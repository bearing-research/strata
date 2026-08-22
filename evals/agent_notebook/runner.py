"""Run the agent-notebook eval suite and report results.

Per task the runner drives the **real on-ramp** — it reuses the same
``agent_launch`` helpers ``strata agent`` uses (create-or-open, spawn a server
with MCP, open a session, write ``.mcp.json`` + ``CLAUDE.md``) — then hands the
prepared notebook to a driver, scores the run, and tears the server down.

Local (real Claude Code):

    uv run python -m evals.agent_notebook.runner --driver claude_code

CI (deterministic, no LLM):

    uv run python -m evals.agent_notebook.runner \\
        --driver replay --transcripts evals/agent_notebook/transcripts
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
from pathlib import Path
from statistics import mean

from strata.notebook import agent_launch

from .drivers import ClaudeCodeDriver, Driver, ReplayDriver
from .graders import RunResult, score_run
from .tasks import TASKS, TASKS_BY_ID, Task


def _free_port() -> int:
    """Grab an ephemeral port so eval servers never collide with a user's :8765."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _provision_deps(notebook_dir: Path, deps: list[str]) -> None:
    """`uv add` the task's dependencies so the notebook's env is ready.

    Keeps completion about notebook-driving behavior rather than whether the
    agent guessed the right package name (the tasks that *test* dependency
    management say so in the prompt regardless).
    """
    if not deps:
        return
    subprocess.run(
        ["uv", "add", *deps],
        cwd=str(notebook_dir),
        capture_output=True,
        text=True,
        check=True,
        timeout=300,
    )


def _strata_run_ok(notebook_dir: Path) -> bool:
    """Run the notebook headlessly and report whether every cell succeeded."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from strata.cli import main; sys.exit(main(sys.argv[1:]))",
            "run",
            str(notebook_dir),
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    try:
        return bool(json.loads(proc.stdout)["success"])
    except (json.JSONDecodeError, KeyError):
        return False


def run_task(
    task: Task, driver: Driver, workdir: Path, *, live: bool = True, label: str | None = None
) -> RunResult:
    """Prepare the notebook, drive the agent, and score it.

    ``live=True`` (Claude Code) runs the real on-ramp: provision deps, spawn a
    server with MCP, open a session, write the agent config, drive, then
    ``strata run`` for the completion check. ``live=False`` (replay) skips all
    of that — no server, no venv, no LLM — so recorded runs re-score
    deterministically in CI; completion reflects only the seeded notebook.

    ``label`` names the notebook subdir (defaults to the task id); repeats pass a
    distinct label per run so each gets its own fresh notebook.
    """
    notebook_dir = agent_launch._resolve_notebook_dir(
        str(workdir / (label or task.id)), None, initialize_environment=live
    )
    if task.seed is not None:
        task.seed(notebook_dir)

    if not live:
        try:
            traj = driver.run(task, notebook_dir)
        except Exception as exc:  # driver boundary: isolate a bad run, don't abort the suite
            return _errored(task, driver, notebook_dir, f"driver error: {exc}")
        return score_run(
            task.id, driver.name, traj, notebook_dir, task.expect_variables, run_ok=None
        )

    _provision_deps(notebook_dir, task.deps)
    server_url = f"http://127.0.0.1:{_free_port()}"
    host, port = agent_launch._server_endpoints(server_url)
    proc = agent_launch._spawn_server(host, port, notebook_dir)
    try:
        if not agent_launch._await_health(server_url, proc):
            return _errored(task, driver, notebook_dir, "server failed to start")
        if not agent_launch._mcp_mounted(server_url):
            return _errored(task, driver, notebook_dir, "MCP endpoint not mounted")
        session_id = agent_launch._open_session(server_url, notebook_dir)
        agent_launch._write_agent_config(notebook_dir, server_url, session_id)

        traj = driver.run(task, notebook_dir)
        run_ok = _strata_run_ok(notebook_dir) if task.expect_run_clean else None
        return score_run(task.id, driver.name, traj, notebook_dir, task.expect_variables, run_ok)
    finally:
        agent_launch._terminate(proc)


def _errored(task: Task, driver: Driver, notebook_dir: Path, msg: str) -> RunResult:
    from .trajectory import Trajectory

    return score_run(
        task.id,
        driver.name,
        Trajectory(ok=False),
        notebook_dir,
        task.expect_variables,
        run_ok=None,
        error=msg,
    )


def run_suite(
    tasks: list[Task], driver: Driver, workdir: Path, *, live: bool = True, repeats: int = 1
) -> list[RunResult]:
    """Run every task ``repeats`` times; each repeat gets its own fresh notebook."""
    results = []
    for task in tasks:
        for i in range(repeats):
            label = task.id if repeats == 1 else f"{task.id}-{i + 1}"
            results.append(run_task(task, driver, workdir, live=live, label=label))
    return results


def summarize(results: list[RunResult]) -> dict:
    active = [r for r in results if not r.in_tool.no_activity and r.error is None]
    rates = [r.in_tool.in_tool_rate for r in active]
    return {
        "tasks": len(results),
        "errored": sum(1 for r in results if r.error is not None),
        "no_activity": sum(1 for r in results if r.in_tool.no_activity),
        "mean_in_tool_rate": round(mean(rates), 4) if rates else None,
        "completion_passed": sum(1 for r in results if r.completion.passed),
        "ran_clean": sum(1 for r in results if r.completion.runs_clean is True),
    }


def format_table(results: list[RunResult]) -> str:
    """One row per task, aggregated across its repeats."""
    groups: dict[str, list[RunResult]] = {}
    for r in results:
        groups.setdefault(r.task_id, []).append(r)

    header = f"{'task':<16} {'runs':>4} {'in-tool':>8} {'min':>5} {'esc':>4} {'pass':>7}"
    lines = [header, "-" * len(header)]
    for task_id, runs in groups.items():
        active = [r for r in runs if not r.in_tool.no_activity and r.error is None]
        rates = [r.in_tool.in_tool_rate for r in active]
        mean_rate = f"{mean(rates):.0%}" if rates else "n/a"
        min_rate = f"{min(rates):.0%}" if rates else "n/a"
        escapes = sum(r.in_tool.escapes for r in runs)
        passed = sum(1 for r in runs if r.completion.passed)
        lines.append(
            f"{task_id:<16} {len(runs):>4} {mean_rate:>8} {min_rate:>5} "
            f"{escapes:>4} {f'{passed}/{len(runs)}':>7}"
        )
    return "\n".join(lines)


def _build_driver(args: argparse.Namespace) -> Driver:
    if args.driver == "replay":
        if not args.transcripts:
            raise SystemExit("--driver replay requires --transcripts DIR")
        return ReplayDriver(Path(args.transcripts))
    return ClaudeCodeDriver(
        timeout=args.timeout,
        max_budget_usd=args.max_budget_usd,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evals.agent_notebook.runner")
    parser.add_argument("--driver", choices=["replay", "claude_code"], default="claude_code")
    parser.add_argument("--transcripts", help="Transcript dir for --driver replay")
    parser.add_argument("--tasks", help="Comma-separated task ids (default: all)")
    parser.add_argument(
        "--select",
        choices=["all", "core", "hard", "scratchpad"],
        default="all",
        help=(
            "Task group: core (transcript-backed), hard (escape-tempting stress), "
            "scratchpad (un-primed trigger-rate probes), or all"
        ),
    )
    parser.add_argument("--workdir", default=None, help="Where eval notebooks are built")
    parser.add_argument("--out", default=None, help="Write the full JSON report here")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-run agent timeout (s)")
    parser.add_argument("--max-budget-usd", type=float, default=None, help="Claude Code spend cap")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per task (default 1)")
    args = parser.parse_args(argv)

    if args.tasks:
        if args.select != "all":
            print("note: --select is ignored because --tasks was given", file=sys.stderr)
        try:
            tasks = [TASKS_BY_ID[tid.strip()] for tid in args.tasks.split(",")]
        except KeyError as exc:
            raise SystemExit(f"unknown task id: {exc}")
    elif args.select == "core":
        tasks = [t for t in TASKS if not t.hard and not t.scratchpad]
    elif args.select == "hard":
        tasks = [t for t in TASKS if t.hard]
    elif args.select == "scratchpad":
        tasks = [t for t in TASKS if t.scratchpad]
    else:
        tasks = TASKS

    driver = _build_driver(args)
    live = args.driver != "replay"

    # Scratchpad tasks measure the *un-primed* trigger rate, but live `run_task`
    # still runs the priming on-ramp (writes the `strata agent` CLAUDE.md). Warn
    # loudly so the reported in-tool rate isn't misread as un-primed — until the
    # automated un-primed driver exists, run those tasks by the manual procedure
    # in the README.
    if live and any(t.scratchpad for t in tasks):
        print(
            "WARNING: scratchpad tasks run PRIMED here — live run_task writes the "
            "on-ramp CLAUDE.md, so the in-tool rate reflects compliance, NOT the "
            "un-primed trigger rate. See README 'Un-primed trigger rate'.",
            file=sys.stderr,
        )

    # Replay can only score tasks it has a transcript for (hard tasks are
    # live-only). Skip the rest with a note instead of erroring, so the
    # documented `--driver replay` command works whatever the selection.
    if not live:
        transcript_dir = Path(args.transcripts)
        have = {p.stem for p in transcript_dir.glob("*.json")}
        have |= {p.stem for p in transcript_dir.glob("*.jsonl")}
        missing = [t.id for t in tasks if t.id not in have]
        if missing:
            print(
                f"replay: skipping {len(missing)} task(s) without a transcript: "
                f"{', '.join(missing)}",
                file=sys.stderr,
            )
        tasks = [t for t in tasks if t.id in have]

    import tempfile

    workdir_ctx = tempfile.TemporaryDirectory() if args.workdir is None else None
    workdir = Path(args.workdir) if args.workdir else Path(workdir_ctx.name)
    workdir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_suite(tasks, driver, workdir, live=live, repeats=args.repeats)
    finally:
        if workdir_ctx is not None:
            workdir_ctx.cleanup()

    summary = summarize(results)
    print(format_table(results))
    print()
    print(json.dumps(summary, indent=2))

    if args.out:
        report = {"summary": summary, "runs": [r.to_dict() for r in results]}
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Non-zero exit if any run errored, so a broken harness fails CI.
    return 1 if summary["errored"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
