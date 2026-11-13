"""What a set of uptake trials says, per arm and kind, and each arm against control."""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from . import stats
from .tasks import MUST_NOT, TASKS_BY_ID

# Above this share of discarded runs, a gate is not scored (§5 validity).
DISCARD_LIMIT = 0.10


def load(root: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(root.rglob("trial.json"))]


def _rate(hits: int, n: int) -> dict[str, Any]:
    low, high = stats.wilson(hits, n)
    return {"k": hits, "n": n, "rate": hits / n if n else None, "wilson": [low, high]}


def _group(trials: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [t for t in trials if t["discarded"] is None]
    loads = [t for t in scored if TASKS_BY_ID[t["task"]].loads]
    return {
        "trials": len(trials),
        "discarded": len(trials) - len(scored),
        "completed": _rate(sum(t["completed"] for t in scored), len(scored)),
        "used_strata": _rate(sum(t["strata_calls"] > 0 for t in scored), len(scored)),
        # One load across the whole trial: the slow step ran once.
        "payoff": _rate(sum(len(t["loads"]) == 1 for t in loads), len(loads)),
        "median_wall_s": statistics.median(t["wall_s"] for t in scored) if scored else None,
        "median_agent_turns": (
            statistics.median(sum(u["agent_turns"] for u in t["turns"]) for t in scored)
            if scored
            else None
        ),
        "cost_usd": sum(u["cost_usd"] for t in trials for u in t["turns"]),
    }


def summarize(trials: Iterable[dict[str, Any]]) -> dict[str, Any]:
    trials = list(trials)
    arms = sorted({t["arm"] for t in trials})
    by = {
        arm: {
            "all": _group([t for t in trials if t["arm"] == arm]),
            **{
                kind: _group([t for t in trials if t["arm"] == arm and t["kind"] == kind])
                for kind in sorted({t["kind"] for t in trials if t["arm"] == arm})
            },
        }
        for arm in arms
    }
    warnings = []
    for arm in arms:
        group = by[arm]["all"]
        if group["trials"] and group["discarded"] / group["trials"] > DISCARD_LIMIT:
            warnings.append(f"arm {arm}: {group['discarded']} of {group['trials']} runs discarded")
    if "A" in by and by["A"]["all"]["used_strata"]["k"]:
        warnings.append(f"arm A used strata in {by['A']['all']['used_strata']['k']} runs")
    must_not = {arm: by[arm][MUST_NOT]["used_strata"] for arm in arms if MUST_NOT in by[arm]}
    return {
        "arms": by,
        "versus_control": {arm: _versus(trials, arm) for arm in arms if arm != "A" and "A" in by},
        "false_positive": must_not,
        "warnings": warnings,
        "pins": sorted({json.dumps(t["pins"], sort_keys=True) for t in trials}),
    }


def _versus(trials: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """*arm* against arm A, kind by kind, on the tasks both ran."""
    out: dict[str, Any] = {}
    for kind in sorted({t["kind"] for t in trials}):

        def scored(a: str) -> list[dict[str, Any]]:
            return [
                t for t in trials if t["arm"] == a and t["kind"] == kind and t["discarded"] is None
            ]

        mine, control = scored(arm), scored("A")
        both = {t["task"] for t in mine} & {t["task"] for t in control}
        mine = [t for t in mine if t["task"] in both]
        control = [t for t in control if t["task"] in both]
        if not both:
            continue
        row: dict[str, Any] = {
            "completion_diff": stats.newcombe(
                sum(t["completed"] for t in mine),
                len(mine),
                sum(t["completed"] for t in control),
                len(control),
            ),
            "median_wall_ratio": stats.median_ratio(
                [t["wall_s"] for t in mine], [t["wall_s"] for t in control]
            ),
        }
        mine_l = [t for t in mine if TASKS_BY_ID[t["task"]].loads]
        control_l = [t for t in control if TASKS_BY_ID[t["task"]].loads]
        if mine_l and control_l:
            row["payoff_diff"] = stats.newcombe(
                sum(len(t["loads"]) == 1 for t in mine_l),
                len(mine_l),
                sum(len(t["loads"]) == 1 for t in control_l),
                len(control_l),
            )
        out[kind] = row
    return out


def render(summary: dict[str, Any]) -> str:
    def rate(r: dict[str, Any]) -> str:
        if not r["n"]:
            return "-"
        low, high = r["wilson"]
        return f"{r['k']}/{r['n']} [{low:.0%}, {high:.0%}]"

    lines = []
    for arm, groups in summary["arms"].items():
        lines.append(f"arm {arm}")
        for kind, g in groups.items():
            wall = "-" if g["median_wall_s"] is None else f"{g['median_wall_s']:.0f}s"
            lines.append(
                f"  {kind:<10} trials {g['trials']:>3} (discarded {g['discarded']})"
                f"  completed {rate(g['completed'])}  strata {rate(g['used_strata'])}"
                f"  payoff {rate(g['payoff'])}  wall {wall}  ${g['cost_usd']:.2f}"
            )
    for arm, kinds in summary["versus_control"].items():
        lines.append(f"arm {arm} vs A")
        for kind, row in kinds.items():
            low, high = row["completion_diff"]
            point, wlow, whigh = row["median_wall_ratio"]
            text = (
                f"  {kind:<10} completion {arm}-A [{low:+.0%}, {high:+.0%}]"
                f"  wall ratio {point:.2f} [{wlow:.2f}, {whigh:.2f}]"
            )
            if "payoff_diff" in row:
                plow, phigh = row["payoff_diff"]
                text += f"  payoff {arm}-A [{plow:+.0%}, {phigh:+.0%}]"
            lines.append(text)
    lines += [f"warning: {w}" for w in summary["warnings"]]
    lines += [f"pins: {p}" for p in summary["pins"]]
    return "\n".join(lines)
