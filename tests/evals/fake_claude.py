"""Stands in for ``claude -p ... --output-format stream-json`` in the uptake runner's
tests: an init event listing the config dir's skills, at most one Bash call that it
really runs, then the reply.

``FAKE_SCRIPT`` (JSON, one ``[command or null, reply]`` per turn) says what each turn
does; ``FAKE_SERVERS`` names MCP servers to report connected; ``FAKE_NEW_SESSION`` is
a turn that starts a new session instead of resuming; ``FAKE_API_ERROR``, a turn that
fails before reaching the model; ``FAKE_HANG``, a turn that never ends. Each call's
argv, and whether its stdin is the null device, go to ``calls.jsonl``.
"""

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path


def emit(event: dict) -> None:
    print(json.dumps(event), flush=True)


def main(args: list[str]) -> int:
    if args == ["--version"]:
        print("0.0.0 (fake)")
        return 0
    config = Path(os.environ["CLAUDE_CONFIG_DIR"])
    calls = config / "calls.jsonl"
    null = os.stat(os.devnull)
    stdin = os.fstat(0)
    with calls.open("a") as out:
        out.write(json.dumps([*args, f"stdin-null={stdin.st_rdev == null.st_rdev}"]) + "\n")
    turn = len(calls.read_text().splitlines()) - 1
    command, reply = json.loads(os.environ["FAKE_SCRIPT"])[turn]
    session = args[args.index("--resume") + 1] if "--resume" in args else str(uuid.uuid4())
    if os.environ.get("FAKE_HANG") == str(turn):
        time.sleep(600)
    api_error = os.environ.get("FAKE_API_ERROR") == str(turn)
    if os.environ.get("FAKE_NEW_SESSION") == str(turn):
        session = str(uuid.uuid4())
    skills = sorted(p.name for p in (config / "skills").glob("*"))
    servers = [
        {"name": n, "status": "connected"}
        for n in os.environ.get("FAKE_SERVERS", "").split(",")
        if n
    ]
    emit(
        {
            "type": "system",
            "subtype": "init",
            "session_id": session,
            "skills": skills,
            "mcp_servers": servers,
        }
    )
    if command:
        block = {"type": "tool_use", "name": "Bash", "input": {"command": command}}
        emit({"type": "assistant", "message": {"content": [block]}})
        subprocess.run(command, shell=True, check=True)
    emit(
        {
            "type": "result",
            "subtype": "success",
            "is_error": api_error,
            "result": "Invalid API key" if api_error else reply,
            "duration_api_ms": 0 if api_error else 1000,
            "session_id": session,
            "num_turns": 2,
            "total_cost_usd": 0.01,
        }
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
