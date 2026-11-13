"""Run uptake trials, or report on them.

python -m evals.agent_notebook.uptake run --arm A --out runs/ --split train --runs 3
python -m evals.agent_notebook.uptake report runs/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import report, run
from .tasks import TASKS_BY_ID, split


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.agent_notebook.uptake")
    commands = parser.add_subparsers(dest="command", required=True)
    go = commands.add_parser("run", help="run trials into --out, skipping any already there")
    go.add_argument("--arm", choices=run.ARMS, required=True)
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--task", action="append", default=[], help="a task id; repeatable")
    go.add_argument("--split", choices=("train", "holdout", "all"), default="train")
    go.add_argument("--runs", type=int, default=1)
    go.add_argument("--model", default=run.MODEL)
    go.add_argument("--claude", default="claude")
    go.add_argument("--load-seconds", type=float, default=20.0)
    show = commands.add_parser("report", help="summarize the trials under a folder")
    show.add_argument("root", type=Path)
    show.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "report":
        summary = report.summarize(report.load(args.root))
        print(json.dumps(summary, indent=2) if args.json else report.render(summary))
        return 0

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set: each trial's fresh config dir has no other login.",
            file=sys.stderr,
        )
        return 2
    if args.task:
        tasks = [TASKS_BY_ID[task_id] for task_id in args.task]
    elif args.split == "all":
        tasks = list(TASKS_BY_ID.values())
    else:
        tasks = list(split(holdout=args.split == "holdout"))
    pinned = run.pins(args.model, args.claude)
    for task in tasks:
        for index in range(args.runs):
            trial_dir = args.out / args.arm / task.id / str(index)
            if (trial_dir / "trial.json").exists():
                continue
            if trial_dir.exists():
                print(f"skipping {trial_dir}: an unfinished trial is there", file=sys.stderr)
                continue
            trial = run.run_trial(
                task,
                args.arm,
                index,
                trial_dir,
                claude=args.claude,
                model=args.model,
                load_seconds=args.load_seconds,
                pinned=pinned,
            )
            state = (
                f"discarded ({trial.discarded})"
                if trial.discarded
                else ("completed" if trial.completed else "failed")
            )
            print(
                f"{args.arm} {task.id} #{index}: {state},"
                f" {trial.wall_s:.0f}s, strata {trial.strata_calls}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
