"""The field metric: what coding agents did on this machine, read from Claude Code's
transcripts (``~/.claude/projects``).

Run before and after each phase of the agent-uptake work. It reads only the local
user's transcripts and prints aggregates: how often the scratchpad was used (the
``strata-scratchpad`` skill, the Strata MCP tools, the ``strata`` CLI's cell work),
how much ad-hoc Python ran instead, and how much of that ran in the project's own
environment importing its code. Sessions whose working directory is a temporary
folder are eval or scratch runs and are left out.

What a Bash call counts as:

- **compute**: Python from inline source (``python -c``, ``python -``, ``python <<EOF``)
  that is neither of the next two; **in the project's env** when run by ``uv run`` or a
  ``.venv``'s interpreter, **importing the project's code** when it also imports a package
  of the project it runs in (after any ``cd``);
- **JSON pipe**: inline Python reading ``sys.stdin`` after a ``|``, or ``python -m
  json.tool``;
- **Python file edit**: inline Python that writes a file, though it may print as it goes;
- **cat heredoc**: ``cat > file <<EOF``;
- **CLI cell work**: ``strata cell add/edit/run/...`` (the eval classifier's set).

    uv run python -m evals.agent_notebook.field --since 2026-09-01
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import sys
import tomllib
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path

from .trajectory import _CLI_WORK, _bash_segments

DEFAULT_ROOT = Path.home() / ".claude" / "projects"
_TEMP_ROOTS = ("/tmp/", "/private/tmp/", "/var/folders/", "/private/var/folders/")

# A Python run from inline source: ``python -c``, or a script on stdin (``python -``,
# ``python <<EOF``). The runner group says whose environment it ran in.
_INLINE_PY = re.compile(
    r"(?:^|[\s;&|(`])"
    r"(?P<runner>uv\s+run\s+(?:-{1,2}[\w-]+(?:[ =][^\s-]\S*)?\s+)*|\S*\.venv/bin/|\S*/)?"
    r"python3?(?:\.\d+)?(?:\s+-[A-Za-z]+)*\s+(?:-c\b|-(?=\s|$)|<<)"
)
_JSON_TOOL = re.compile(r"\|\s*(?:\S*/)?python3?(?:\.\d+)?\s+-m\s+json\.tool\b")
_WRITES = re.compile(r"\.write_(?:text|bytes)\(|\.writelines?\(|open\([^)]*['\"][wax]b?\+?['\"]")
# A `cd` at a command's start, after a separator or on its own line; its target ends at
# the next separator, as in `cd /repo; ...`.
_CD = re.compile(r"(?:^|[;&|(\n])\s*cd\s+([^\s;&|)]+)")
_CAT_HEREDOC = re.compile(r"(?:^|[\s;&|(`])cat\s+>>?\s*\S+\s*<<")


@dataclass
class Session:
    """What one transcript did."""

    path: str
    subagent: bool
    cwd: str = ""
    started: str = ""  # ISO timestamp of its first record
    skill_triggers: int = 0
    skill_trigger_days: list[str] = field(default_factory=list)
    mcp_calls: int = 0
    cli_work: int = 0
    compute: int = 0
    compute_project_env: int = 0
    compute_imports_project: int = 0
    json_pipes: int = 0
    python_file_edits: int = 0
    cat_heredocs: int = 0
    codegraph: int = 0
    examples: list[str] = field(default_factory=list)

    @property
    def eval_run(self) -> bool:
        return self.cwd.startswith(_TEMP_ROOTS)

    @property
    def used_scratchpad(self) -> bool:
        return bool(self.skill_triggers or self.mcp_calls or self.cli_work)


@functools.cache
def project_modules(cwd: str) -> frozenset[str]:
    """The importable names of the project at *cwd*: its ``[project] name`` and the
    packages in it or in its ``src/``, as they are on disk now."""
    root = Path(cwd)
    names: set[str] = set()
    try:
        meta = tomllib.loads((root / "pyproject.toml").read_text())
    except (OSError, tomllib.TOMLDecodeError):
        meta = {}
    name = meta.get("project", {}).get("name")
    if isinstance(name, str):
        names.add(re.sub(r"[-.]+", "_", name).lower())
    for base in (root, root / "src"):
        try:
            names.update(p.parent.name for p in base.glob("*/__init__.py"))
        except OSError:
            continue
    return frozenset(names)


def command_dir(command: str, before: int, cwd: str) -> str:
    """Where *command* is at offset *before*: *cwd* after its last ``cd`` up to there."""
    where = cwd
    for found in _CD.finditer(command, 0, before):
        target = os.path.expanduser(found.group(1).strip("'\""))
        where = os.path.normpath(os.path.join(where, target))
    return where


def classify_bash(command: str, cwd: str) -> Counter[str]:
    """The kinds of work one Bash call, run from *cwd*, did, counted (see :class:`Session`)."""
    kinds: Counter[str] = Counter()
    if any(_CLI_WORK.search(seg) for seg in _bash_segments(command)):
        kinds["cli_work"] += 1
    if "codegraph " in command:
        kinds["codegraph"] += 1
    if _CAT_HEREDOC.search(command):
        kinds["cat_heredocs"] += 1
    if _JSON_TOOL.search(command):
        kinds["json_pipes"] += 1
    found = _INLINE_PY.search(command)
    if found is None:
        return kinds
    payload = command[found.end() :]
    piped = "|" in command[: found.start() + 1] and "sys.stdin" in payload
    if piped:
        kinds["json_pipes"] += 1
    elif _WRITES.search(payload):  # an edit, though it may print or assert as it goes
        kinds["python_file_edits"] += 1
    else:
        kinds["compute"] += 1
        runner = found.group("runner") or ""
        # `uv run`, or a venv's python; not a uv-managed interpreter's own path.
        if runner.startswith("uv") or ".venv/" in runner:
            kinds["compute_project_env"] += 1
            modules = project_modules(command_dir(command, found.start(), cwd)) if cwd else ()
            if modules and re.search(
                r"(?:^|[\n;])\s*[\"']?\s*(?:import|from)\s+(?:"
                + "|".join(map(re.escape, modules))
                + r")\b",
                payload,
            ):
                kinds["compute_imports_project"] += 1
    return kinds


def _records(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8", errors="replace") as lines:
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def read_session(path: Path) -> Session:
    """Score one transcript file."""
    session = Session(path=str(path), subagent="subagents" in path.parts)
    for record in _records(path):
        if not session.cwd and isinstance(record.get("cwd"), str):
            session.cwd = record["cwd"]
        if not session.started and isinstance(record.get("timestamp"), str):
            session.started = record["timestamp"]
        if record.get("type") != "assistant":
            continue
        content = (record.get("message") or {}).get("content")
        for block in content if isinstance(content, list) else ():
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = str(block.get("name", ""))
            args = block.get("input") or {}
            parts = name.split("__")
            if len(parts) >= 3 and parts[0] == "mcp":
                if "strata" in parts[1]:
                    session.mcp_calls += 1
                elif "codegraph" in parts[1]:
                    session.codegraph += 1
            elif name == "Skill" and "strata-scratchpad" in str(args.get("skill", "")):
                # Dated by its own record: a resumed session outlives its start.
                session.skill_triggers += 1
                session.skill_trigger_days.append(str(record.get("timestamp", ""))[:10])
            elif name == "Bash":
                command = str(args.get("command", ""))
                # Each record has the cwd the call ran from; a `cd` earlier persists.
                cwd = record.get("cwd")
                kinds = classify_bash(command, cwd if isinstance(cwd, str) else session.cwd)
                for kind, count in kinds.items():
                    setattr(session, kind, getattr(session, kind) + count)
                if kinds["compute_project_env"]:
                    session.examples.append(command)
    return session


def sessions(root: Path) -> Iterator[Session]:
    for path in sorted(root.rglob("*.jsonl")):
        yield read_session(path)


@dataclass
class Report:
    window: tuple[str | None, str | None]
    sessions: int
    subagent_sessions: int
    eval_sessions_left_out: int
    skill_triggers: int
    skill_trigger_dates: list[str]
    mcp_calls: int
    cli_work: int
    sessions_using_scratchpad: int
    compute: int
    compute_project_env: int
    compute_imports_project: int
    json_pipes: int
    python_file_edits: int
    cat_heredocs: int
    codegraph_main_sessions: int
    codegraph_subagent_sessions: int
    # ISO week -> [sessions with compute work, of those, sessions that used the scratchpad]
    weekly: dict[str, list[int]]
    examples: list[str]


def _day(timestamp: str) -> date | None:
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def build_report(
    found: Iterable[Session],
    *,
    since: date | None = None,
    until: date | None = None,
    examples: int = 0,
    seed: int = 0,
) -> Report:
    """Aggregate *found* sessions started in [since, until]."""
    kept: list[Session] = []
    left_out = 0
    weekly: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for session in found:
        day = _day(session.started)
        if day is None or (since and day < since) or (until and day > until):
            continue
        if session.eval_run:
            left_out += 1
            continue
        kept.append(session)
        if session.compute:
            year, week, _ = day.isocalendar()
            weekly[f"{year}-W{week:02d}"][0] += 1
            weekly[f"{year}-W{week:02d}"][1] += session.used_scratchpad
    pool = [command for s in kept for command in s.examples]
    sample = random.Random(seed).sample(pool, min(examples, len(pool)))

    def total(attribute: str) -> int:
        return sum(getattr(s, attribute) for s in kept)

    return Report(
        window=(since.isoformat() if since else None, until.isoformat() if until else None),
        sessions=sum(not s.subagent for s in kept),
        subagent_sessions=sum(s.subagent for s in kept),
        eval_sessions_left_out=left_out,
        skill_triggers=total("skill_triggers"),
        skill_trigger_dates=sorted(day for s in kept for day in s.skill_trigger_days),
        mcp_calls=total("mcp_calls"),
        cli_work=total("cli_work"),
        sessions_using_scratchpad=sum(s.used_scratchpad for s in kept),
        compute=total("compute"),
        compute_project_env=total("compute_project_env"),
        compute_imports_project=total("compute_imports_project"),
        json_pipes=total("json_pipes"),
        python_file_edits=total("python_file_edits"),
        cat_heredocs=total("cat_heredocs"),
        codegraph_main_sessions=sum(bool(s.codegraph) for s in kept if not s.subagent),
        codegraph_subagent_sessions=sum(bool(s.codegraph) for s in kept if s.subagent),
        weekly=dict(sorted(weekly.items())),
        examples=[command[:300] for command in sample],
    )


def _share(part: int, whole: int) -> str:
    return f"{part} ({part / whole:.0%})" if whole else str(part)


def render(report: Report) -> str:
    lines = [
        f"window: {report.window[0] or 'start'} to {report.window[1] or 'now'}",
        f"sessions: {report.sessions} main, {report.subagent_sessions} subagent"
        f" ({report.eval_sessions_left_out} eval or scratch runs left out)",
        f"strata-scratchpad skill triggered: {report.skill_triggers}"
        + (f" ({', '.join(report.skill_trigger_dates)})" if report.skill_trigger_dates else ""),
        f"Strata MCP tool calls: {report.mcp_calls}",
        f"strata CLI cell work: {report.cli_work}",
        f"sessions that used the scratchpad: {report.sessions_using_scratchpad}",
        f"compute-shaped ad-hoc Python: {report.compute}",
        f"  in the project's env: {_share(report.compute_project_env, report.compute)}",
        f"  importing the project's code: {_share(report.compute_imports_project, report.compute)}",
        f"not scratchpad work: {report.json_pipes} JSON pipes,"
        f" {report.python_file_edits} Python file edits, {report.cat_heredocs} cat heredocs",
        f"codegraph: {report.codegraph_main_sessions} of {report.sessions} main sessions,"
        f" {report.codegraph_subagent_sessions} of {report.subagent_sessions} subagent sessions",
        "week: sessions with compute work, of those used the scratchpad",
    ]
    lines += [f"  {week}: {n}, {_share(used, n)}" for week, (n, used) in report.weekly.items()]
    if report.examples:
        lines.append("sampled ad-hoc Python in the project's env:")
        lines += [f"  - {example!r}" for example in report.examples]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--since", type=date.fromisoformat)
    parser.add_argument("--until", type=date.fromisoformat)
    parser.add_argument("--examples", type=int, default=0, help="sample this many bypasses")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_report(
        sessions(args.root), since=args.since, until=args.until, examples=args.examples
    )
    print(json.dumps(asdict(report), indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
