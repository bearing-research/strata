"""Agent drivers — what plays the role of the coding agent for a task.

Two backends produce the same normalized :class:`Trajectory`:

* :class:`ReplayDriver` — reads recorded transcripts. Deterministic, no LLM;
  this is what CI runs to exercise the harness + graders.
* :class:`ClaudeCodeDriver` — invokes ``claude -p`` headless in the notebook
  directory against the live ``/mcp`` server. This is the real, on-demand
  local eval.

Both share :func:`parse_stream_json`, the parser for Claude Code's
``--output-format stream-json`` — so a real run can be captured once and
replayed forever after.
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

    Tolerant of the two shapes tool calls arrive in — a top-level
    ``stream_event`` whose ``event`` is a ``tool_use``, and an ``assistant``
    message whose ``content`` array holds ``tool_use`` blocks — and reads the
    final ``result`` object for the closing text and error flag.
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
    "ok": true}`` — the minimal shape needed to drive the graders.
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

    Looks in ``transcript_dir`` for ``<task_id>.jsonl`` (real Claude Code
    stream-json) or ``<task_id>.json`` (normalized fixture), in that order.
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

    Runs ``claude -p`` in the notebook directory so it auto-discovers the
    ``CLAUDE.md`` working agreement and the ``.mcp.json`` the on-ramp wrote.
    Permissions are bypassed **on purpose**: the agent must be free to reach for
    Bash/Python, because whether it does is exactly what the in-tool rate
    measures — allowlisting only the notebook tools would fake a perfect score.
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

    def _command(self, prompt: str) -> list[str]:
        cmd = [
            self.binary,
            "-p",
            prompt,
            "--mcp-config",
            ".mcp.json",
            "--strict-mcp-config",
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
        proc = subprocess.run(
            self._command(task.prompt),
            cwd=str(notebook_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        traj = parse_stream_json(proc.stdout)
        if proc.returncode != 0 and not traj.events:
            # No tool calls and a non-zero exit means the agent never got going
            # (auth, missing binary, MCP handshake) — surface stderr, don't
            # silently score an empty run as a perfect in-tool rate.
            traj.ok = False
            traj.final_text = traj.final_text or proc.stderr.strip()
        return traj
