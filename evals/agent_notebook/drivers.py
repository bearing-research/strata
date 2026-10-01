"""Agent drivers: what plays the coding agent for a task.

:class:`ReplayDriver` reads recorded transcripts (deterministic; what CI runs).
:class:`ClaudeCodeDriver` runs ``claude -p`` headless against the live ``/mcp``
server. Both parse stream-json with :func:`parse_stream_json`, so a real run can
be captured once and replayed.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Protocol

from .tasks import Task
from .trajectory import ToolEvent, Trajectory


class Driver(Protocol):
    name: str

    def run(self, task: Task, notebook_dir: Path) -> Trajectory: ...


def _tool_use_from_block(block: dict) -> ToolEvent | None:
    if block.get("type") == "tool_use" and "name" in block:
        return ToolEvent(name=block["name"], arguments=block.get("input") or {})
    return None


def parse_stream_json(text: str) -> Trajectory:
    """Parse Claude Code ``--output-format stream-json`` output into a Trajectory.

    Accepts tool calls as a ``stream_event`` whose ``event`` is a ``tool_use``
    or as ``tool_use`` blocks in an ``assistant`` message. The final ``result``
    object supplies the closing text and error flag.
    """
    events: list[ToolEvent] = []
    final_text = ""
    ok = True
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = obj.get("type")
        if kind == "stream_event":
            evt = _tool_use_from_block(obj.get("event") or {})
            if evt is not None:
                events.append(evt)
        elif kind == "assistant":
            for block in (obj.get("message") or {}).get("content") or []:
                if isinstance(block, dict):
                    evt = _tool_use_from_block(block)
                    if evt is not None:
                        events.append(evt)
        elif kind == "result":
            final_text = str(obj.get("result") or "")
            ok = not obj.get("is_error", False) and obj.get("subtype", "success") == "success"
    return Trajectory(events=events, final_text=final_text, ok=ok, raw=text)


def parse_normalized(obj: dict) -> Trajectory:
    """Parse the hand-authored transcript format used by CI fixtures.

    ``{"events": [{"name": ..., "arguments": {...}}, ...], "final_text": ...,
    "ok": true}``.
    """
    events = [
        ToolEvent(name=e["name"], arguments=e.get("arguments") or {}) for e in obj.get("events", [])
    ]
    return Trajectory(
        events=events,
        final_text=str(obj.get("final_text", "")),
        ok=bool(obj.get("ok", True)),
        raw=obj,
    )


class ReplayDriver:
    """Return a recorded trajectory for each task, keyed by task id.

    Looks for ``<task_id>.jsonl`` (stream-json) then ``<task_id>.json``
    (normalized fixture) in ``transcript_dir``.
    """

    name = "replay"

    def __init__(self, transcript_dir: Path) -> None:
        self.transcript_dir = Path(transcript_dir)

    def run(self, task: Task, notebook_dir: Path) -> Trajectory:
        jsonl = self.transcript_dir / f"{task.id}.jsonl"
        if jsonl.is_file():
            return parse_stream_json(jsonl.read_text(encoding="utf-8"))
        normalized = self.transcript_dir / f"{task.id}.json"
        if normalized.is_file():
            return parse_normalized(json.loads(normalized.read_text(encoding="utf-8")))
        raise FileNotFoundError(f"no transcript for task {task.id!r} in {self.transcript_dir}")


class ClaudeCodeDriver:
    """Drive a real Claude Code headless session against the live MCP server.

    Runs in the notebook directory so it picks up ``CLAUDE.md`` and
    ``.mcp.json``. Permissions are bypassed **on purpose**: the in-tool rate
    measures whether the agent reaches for Bash/Python, so allowlisting only
    the notebook tools would fake a perfect score.
    """

    name = "claude_code"

    def __init__(
        self,
        *,
        binary: str | None = None,
        timeout: float = 600.0,
        max_budget_usd: float | None = None,
    ) -> None:
        self.binary = binary or os.environ.get("STRATA_EVAL_CLAUDE_BIN", "claude")
        self.timeout = timeout
        self.max_budget_usd = max_budget_usd

    def _command(self, prompt: str, *, use_mcp: bool) -> list[str]:
        cmd = [self.binary, "-p", prompt]
        if use_mcp:
            # The un-primed scratchpad flow has no .mcp.json; the agent uses
            # the `strata` CLI and the installed skill instead.
            cmd += ["--mcp-config", ".mcp.json", "--strict-mcp-config"]
        cmd += [
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--verbose",
        ]
        if self.max_budget_usd is not None:
            cmd += ["--max-budget-usd", str(self.max_budget_usd)]
        return cmd

    def run(self, task: Task, notebook_dir: Path) -> Trajectory:
        use_mcp = (notebook_dir / ".mcp.json").is_file()
        proc = subprocess.run(
            self._command(task.prompt, use_mcp=use_mcp),
            cwd=str(notebook_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        traj = parse_stream_json(proc.stdout)
        if proc.returncode != 0 and not traj.events:
            # The agent never started (auth, binary, MCP handshake). Surface
            # stderr rather than score an empty run as a perfect in-tool rate.
            traj.ok = False
            traj.final_text = traj.final_text or proc.stderr.strip()
        return traj
