"""One uptake trial: a task's turns sent to headless Claude Code in a fresh copy of the
project, under an arm's setup, then scored (design-agent-uptake §5).

Each trial gets its own ``CLAUDE_CONFIG_DIR``, so nothing of the author's own config
reaches it, and the follow-up turns go through ``--resume`` one at a time. A trial
whose stream does not show the arm's setup, on any turn, is discarded and counted,
never scored.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import workspace as ws
from .tasks import Task

ARMS = ("A", "B")
MODEL = "claude-opus-5-5"
REPO = Path(__file__).resolve().parents[3]
SKILL = REPO / "src" / "strata" / ".agents" / "skills" / "strata-scratchpad"

# A Bash command that runs strata or one of its commands (`strata-notebook`): the word at
# a command's start, after a separator or whitespace (`uv run strata`), or at the end of a
# path (`~/.local/bin/strata`); or Python that imports it.
_STRATA_COMMAND = r"(?:^|[\s;&|(`/])strata(?:-\w[\w-]*)?(?:\s|$)|\b(?:import|from)\s+strata\b"
_RUNS_STRATA = re.compile(_STRATA_COMMAND)
# The variables of the process running the eval that must not reach the agent's.
_DROP = re.compile(r"^(CLAUDE|ANTHROPIC_(?!API_KEY$)|VIRTUAL_ENV$|PYTHON(PATH|HOME)$|UV_PROJECT)")

_DENY_HOOK = f"""\
import json, re, sys

command = json.load(sys.stdin).get("tool_input", {{}}).get("command", "")
if re.search({_STRATA_COMMAND!r}, command):
    print("strata is not available in this environment.", file=sys.stderr)
    sys.exit(2)
"""
_MISSING = '#!/bin/sh\necho "{name}: command not found" >&2\nexit 127\n'
# What in a folder above a trial would reach its agent: Claude Code loads memory and
# settings from every parent folder, and git, uv and pytest look up for their roots.
_LEAKS = (
    "CLAUDE.md",
    "CLAUDE.local.md",
    ".claude/CLAUDE.md",
    ".claude/settings.json",
    ".claude/settings.local.json",
    ".mcp.json",
    ".git",
    "pyproject.toml",
)


@dataclass
class Turn:
    prompt: str
    reply: str = ""
    passed: bool = False
    wall_s: float = 0.0
    agent_turns: int = 0
    cost_usd: float = 0.0
    is_error: bool = False
    loads: int = 0
    tools: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Trial:
    task: str
    kind: str
    arm: str
    run: int
    pins: dict[str, str]
    session_id: str | None = None
    # Why the trial is not scored; None when its setup held on every turn.
    discarded: str | None = None
    turns: list[Turn] = field(default_factory=list)
    loads: list[dict[str, Any]] = field(default_factory=list)

    @property
    def completed(self) -> bool:
        return bool(self.turns) and all(t.passed for t in self.turns)

    @property
    def wall_s(self) -> float:
        return sum(t.wall_s for t in self.turns)

    @property
    def strata_calls(self) -> int:
        return sum(1 for t in self.turns for tool in t.tools if uses_strata(tool))

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "completed": self.completed,
            "wall_s": self.wall_s,
            "strata_calls": self.strata_calls,
        }


def uses_strata(tool: dict[str, Any]) -> bool:
    name, args = tool.get("name", ""), tool.get("input") or {}
    if name.startswith("mcp__strata") or name.startswith("mcp__plugin_strata"):
        return True
    if name == "Skill":
        return "strata" in str(args.get("skill", ""))
    return name == "Bash" and bool(_RUNS_STRATA.search(str(args.get("command", ""))))


def pins(model: str, claude: str) -> dict[str, str]:
    def out(*command: str) -> str:
        try:
            done = subprocess.run(command, capture_output=True, text=True, timeout=30)
        except OSError as exc:
            return f"unavailable: {exc}"
        return done.stdout.strip() or done.stderr.strip()

    return {
        "model": model,
        "claude_code": out(claude, "--version"),
        "uv": out("uv", "--version"),
        "strata_commit": out("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "python": sys.version.split()[0],
    }


def isolation_problem(trial_dir: Path) -> str | None:
    """What in a folder above *trial_dir* would reach a trial's agent, or None."""
    for parent in trial_dir.resolve().parents:
        for name in _LEAKS:
            if (parent / name).exists():
                return f"{parent / name} would reach the agent; put trials under /tmp"
    return None


def agent_path(path: str) -> str:
    """*path* without the eval's own environment: its python can import strata."""
    own = {Path(sys.prefix) / "bin", Path(sys.prefix) / "Scripts"}
    if os.environ.get("VIRTUAL_ENV"):
        own.add(Path(os.environ["VIRTUAL_ENV"]) / "bin")
    return os.pathsep.join(
        entry for entry in path.split(os.pathsep) if entry and Path(entry) not in own
    )


def prepare_arm(arm: str, config: Path, bin_dir: Path, path: str) -> None:
    """Write *arm*'s setup into a fresh config dir and the bin dir put before *path*."""
    config.mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    if arm == "A":
        # Every strata command anywhere on the path, not just `strata`, is not found.
        names = {"strata"} | {
            entry.name
            for folder in path.split(os.pathsep)
            if Path(folder).is_dir()
            for entry in Path(folder).iterdir()
            if entry.name.startswith("strata")
        }
        for name in names:
            shim = bin_dir / name
            shim.write_text(_MISSING.format(name=name))
            shim.chmod(0o755)
        hook = config / "deny_strata.py"
        hook.write_text(_DENY_HOOK)
        command = f"{shlex.quote(sys.executable)} {shlex.quote(str(hook))}"
        settings = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": command}]}
                ]
            }
        }
        (config / "settings.json").write_text(json.dumps(settings, indent=2))
    elif arm == "B":
        shutil.copytree(SKILL, config / "skills" / "strata-scratchpad")
        (bin_dir / "strata").symlink_to(Path(sys.executable).parent / "strata")
    else:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def preflight(arm: str, init: dict[str, Any] | None) -> str | None:
    """Why the stream's ``system/init`` event does not show *arm*'s setup, or None."""
    if init is None:
        return "no init event"
    servers = [s.get("name") for s in init.get("mcp_servers") or []]
    if any("strata" in str(name) for name in servers):
        return f"a strata MCP server is connected: {servers}"
    skill = "strata-scratchpad" in (init.get("skills") or [])
    if arm == "A" and skill:
        return "the strata-scratchpad skill is loaded"
    if arm == "B" and not skill:
        return "the strata-scratchpad skill is missing"
    return None


def agent_env(
    config: Path, bin_dir: Path, path: str, load_log: Path, load_seconds: float
) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _DROP.match(k)}
    env.update(
        CLAUDE_CONFIG_DIR=str(config),
        PATH=f"{bin_dir}{os.pathsep}{path}",
        HARBOR_LOAD_LOG=str(load_log),
        HARBOR_LOAD_SECONDS=str(load_seconds),
    )
    return env


def prepare_workspace(dest: Path, *, rows: int, sync: bool) -> Path:
    """The project as a user would have it: built, synced, committed."""
    work = ws.build(dest, rows=rows)
    env = {k: v for k, v in os.environ.items() if not _DROP.match(k)}
    if sync:
        subprocess.run(["uv", "sync", "--quiet"], cwd=work, env=env, check=True, timeout=600)
    git = ["git", "-c", "user.name=eval", "-c", "user.email=eval@localhost"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "harbor"]):
        subprocess.run([*git, *args], cwd=work, env=env, check=True, timeout=60)
    return work


def read_stream(
    text: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]]]:
    """A turn's ``system/init`` event, its ``result`` event and its tool calls."""
    init = result = None
    tools: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("subtype") == "init":
            init = event
        elif event.get("type") == "result":
            result = event
        elif event.get("type") == "assistant":
            for block in (event.get("message") or {}).get("content") or []:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tools.append({"name": block.get("name", ""), "input": block.get("input") or {}})
    return init, result, tools


def run_trial(
    task: Task,
    arm: str,
    run: int,
    trial_dir: Path,
    *,
    claude: str = "claude",
    model: str = MODEL,
    rows: int = 1_000_000,
    load_seconds: float = 20.0,
    sync: bool = True,
    timeout: float = 1800.0,
    pinned: dict[str, str] | None = None,
) -> Trial:
    """Run *task* once under *arm* in *trial_dir*, and write ``trial.json`` there."""
    problem = isolation_problem(trial_dir)
    if problem:
        raise ValueError(problem)
    trial = Trial(task.id, task.kind, arm, run, pinned or pins(model, claude))
    config, bin_dir, load_log = trial_dir / "config", trial_dir / "bin", trial_dir / "loads.jsonl"
    path = agent_path(os.environ.get("PATH", ""))
    prepare_arm(arm, config, bin_dir, path)
    work = prepare_workspace(trial_dir / "harbor", rows=rows, sync=sync)
    env = agent_env(config, bin_dir, path, load_log, load_seconds)
    # Before the agent can touch the workspace's data; a files check reads it after.
    expected = task.expected(work)
    for index, question in enumerate(task.turns):
        command = [claude, "-p", question.prompt, "--model", model]
        if trial.session_id:
            command += ["--resume", trial.session_id]
        command += [
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        started = time.time()
        try:
            done = subprocess.run(
                command,
                cwd=work,
                env=env,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stream = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout or ""
            (trial_dir / f"turn{index}.jsonl").write_text(stream)
            trial.turns.append(Turn(question.prompt, wall_s=time.time() - started))
            trial.discarded = f"turn {index}: timed out after {timeout:.0f} s"
            break
        (trial_dir / f"turn{index}.jsonl").write_text(done.stdout)
        init, result, tools = read_stream(done.stdout)
        turn = Turn(question.prompt, wall_s=time.time() - started, tools=tools)
        trial.turns.append(turn)
        problem = preflight(arm, init) or _broken_turn(init, result, done, trial.session_id)
        if problem:
            trial.discarded = f"turn {index}: {problem}"
            break
        assert init is not None and result is not None
        trial.session_id = init.get("session_id")
        turn.reply = str(result.get("result") or "")
        turn.agent_turns = int(result.get("num_turns") or 0)
        turn.cost_usd = float(result.get("total_cost_usd") or 0.0)
        turn.is_error = bool(result.get("is_error"))
        turn.loads = sum(
            started <= line["at"] <= started + turn.wall_s for line in _loads(load_log)
        )
    trial.loads = _loads(load_log)
    if trial.discarded is None:
        for turn, expect in zip(trial.turns, expected, strict=True):
            turn.passed = not turn.is_error and expect.passes(turn.reply, work)
    (trial_dir / "trial.json").write_text(json.dumps(trial.to_json(), indent=2))
    return trial


def _broken_turn(
    init: dict[str, Any] | None,
    result: dict[str, Any] | None,
    done: subprocess.CompletedProcess[str],
    session_id: str | None,
) -> str | None:
    """Why a turn says nothing about the agent: it never ended, never reached the model
    (a bad key, an outage), or lost the session it resumed."""
    if result is None:
        return f"no result event (exit {done.returncode}): {done.stderr.strip()[-300:]}"
    if result.get("is_error") and not result.get("duration_api_ms"):
        return f"the agent never reached the model: {str(result.get('result'))[:300]}"
    if session_id and init and init.get("session_id") != session_id:
        return "the resumed turn started a new session"
    return None


def _loads(log: Path) -> list[dict[str, Any]]:
    if not log.exists():
        return []
    return [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
