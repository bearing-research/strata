"""``strata agent install`` / ``uninstall``: the scratchpad in a coding agent's own config.

Per target, every piece is keyed or marker-fenced, so ``uninstall`` removes what
``install`` wrote and nothing else:

- Claude Code: the user-scope MCP entry, through the ``claude`` CLI (which owns
  ``~/.claude.json`` and may be writing it); the ``mcp__strata__*`` permission in
  ``settings.json``; a block in ``CLAUDE.md``, for subagents and harnesses that do not
  see MCP instructions.
- Codex: an ``[mcp_servers.strata]`` block in ``config.toml``; a block in ``AGENTS.md``.

A file Strata writes is backed up first. ``uninstall`` puts its original bytes back
when it is as Strata left it, and removes only Strata's part when it was edited since.
``strata mcp`` rewrites the file pieces on start once another Strata version wrote them.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import filelock

from strata.agent_guidance import section

TARGETS = ("claude", "codex")
SERVER_NAME = "strata"
PERMISSION = "mcp__strata__*"
_SKILL = "strata-scratchpad"
_STATE_VERSION = 1


class InstallError(RuntimeError):
    """What the installer cannot do, said so the user can act on it."""


@dataclass(frozen=True)
class Change:
    """One piece written or removed, as the command reports it."""

    what: str
    where: Path


def _home() -> Path:
    return Path.home() / ".strata" / "agent-install"


def _state_path() -> Path:
    return _home() / "state.json"


def _lock() -> filelock.FileLock:
    _home().mkdir(parents=True, exist_ok=True)
    return filelock.FileLock(str(_home() / "state.lock"))


def claude_dir() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude")


def codex_dir() -> Path:
    return Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")


def _claude_json() -> Path:
    # Where the claude CLI keeps user-scope MCP servers: beside the config dir it was
    # pointed at, else in the home directory.
    configured = os.environ.get("CLAUDE_CONFIG_DIR")
    return Path(configured) / ".claude.json" if configured else Path.home() / ".claude.json"


def strata_command() -> str:
    """The running ``strata``, absolute, so a ``uv tool`` install and a venv both work."""
    argv0 = Path(sys.argv[0])
    if argv0.name == "strata" and argv0.is_file():
        return str(argv0.absolute())
    found = shutil.which("strata")
    if found is None:
        raise InstallError("cannot find the `strata` command this runs as; is it on PATH?")
    return str(Path(found).absolute())


# --- the pieces --------------------------------------------------------------------


def _block_begin(fence: str) -> str:
    stamp = version("strata-notebook")
    return f"{fence} strata:begin v{stamp}; `strata agent uninstall` removes this block"


def _fenced(fence: str, end_fence: str, body: str) -> str:
    return f"{_block_begin(fence)}{end_fence}\n{body.rstrip()}\n{fence} strata:end{end_fence}\n"


def _fence_pattern(fence: str, end_fence: str) -> re.Pattern[str]:
    f, e = re.escape(fence), re.escape(end_fence)
    # With the blank line Strata put before it, so removing it leaves the file as it was.
    return re.compile(rf"\n?^{f} strata:begin[^\n]*{e}\n.*?^{f} strata:end{e}\n?", re.M | re.S)


def _add_block(text: str, block: str, pattern: re.Pattern[str]) -> str:
    text = pattern.sub("", text)
    if text and not text.endswith("\n"):
        text += "\n"
    return f"{text}\n{block}" if text else block


def _markdown_block(run_tool: str, inventory_tool: str) -> str:
    body = section("agent_block").format(run_tool=run_tool, inventory_tool=inventory_tool)
    return _fenced("<!--", " -->", body)


_MARKDOWN = _fence_pattern("<!--", " -->")
_TOML = _fence_pattern("#", "")


@dataclass(frozen=True)
class _FilePiece:
    """A change to one file: *add* puts Strata's part in, *remove* takes it out."""

    what: str
    path: Path
    add: Callable[[str], str]
    remove: Callable[[str], str]


def _settings_add(text: str) -> str:
    settings = _load_json(text, "settings.json")
    allow = settings.setdefault("permissions", {}).setdefault("allow", [])
    if PERMISSION not in allow:
        allow.append(PERMISSION)
    return json.dumps(settings, indent=2) + "\n"


def _settings_remove(text: str) -> str:
    settings = _load_json(text, "settings.json")
    allow = settings.get("permissions", {}).get("allow")
    if isinstance(allow, list) and PERMISSION in allow:
        allow.remove(PERMISSION)
    return json.dumps(settings, indent=2) + "\n"


def _load_json(text: str, name: str) -> dict[str, Any]:
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise InstallError(f"{name} is not valid JSON ({exc}); fix it and run again") from exc
    if not isinstance(value, dict):
        raise InstallError(f"{name} is not a JSON object; fix it and run again")
    return value


def _codex_add(command: str) -> Callable[[str], str]:
    def add(text: str) -> str:
        rest = _TOML.sub("", text)
        try:
            servers = tomllib.loads(rest).get("mcp_servers", {})
        except tomllib.TOMLDecodeError as exc:
            raise InstallError(f"config.toml is not valid TOML ({exc}); fix it and run again")
        if SERVER_NAME in servers:
            raise InstallError(
                "config.toml already has its own [mcp_servers.strata]; remove it, or "
                "install for Claude Code only with --target claude"
            )
        # A JSON string is a TOML basic string.
        body = f'[mcp_servers.{SERVER_NAME}]\ncommand = {json.dumps(command)}\nargs = ["mcp"]'
        return _add_block(rest, _fenced("#", "", body), _TOML)

    return add


def _file_pieces(target: str, command: str) -> list[_FilePiece]:
    if target == "claude":
        base = claude_dir()
        block = _markdown_block("the `mcp__strata__run` tool", "`mcp__strata__scratchpad`")
        return [
            _FilePiece(
                "permission mcp__strata__*", base / "settings.json", _settings_add, _settings_remove
            ),
            _FilePiece(
                "scratchpad block",
                base / "CLAUDE.md",
                lambda text: _add_block(text, block, _MARKDOWN),
                lambda text: _MARKDOWN.sub("", text),
            ),
        ]
    base = codex_dir()
    block = _markdown_block("the `strata` MCP server's `run` tool", "Its `scratchpad` tool")
    return [
        _FilePiece(
            "MCP server strata",
            base / "config.toml",
            _codex_add(command),
            lambda t: _TOML.sub("", t),
        ),
        _FilePiece(
            "scratchpad block",
            base / "AGENTS.md",
            lambda text: _add_block(text, block, _MARKDOWN),
            lambda text: _MARKDOWN.sub("", text),
        ),
    ]


# --- state -------------------------------------------------------------------------


def _load_state() -> dict[str, Any]:
    path = _state_path()
    if not path.exists():
        return {"state": _STATE_VERSION, "targets": {}, "files": {}}
    return json.loads(path.read_text())


def _save_state(state: dict[str, Any]) -> None:
    if not state["targets"] and not state["files"]:
        _state_path().unlink(missing_ok=True)
        return
    _write(_state_path(), json.dumps(state, indent=2) + "\n")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _backup_path(path: Path) -> Path:
    return _home() / "backups" / hashlib.sha256(str(path).encode()).hexdigest()[:16]


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.strata-tmp")
    temporary.write_text(text, encoding="utf-8")
    if path.exists():
        shutil.copymode(path, temporary)
    os.replace(temporary, path)


def _read(path: Path) -> str | None:
    return path.read_text(encoding="utf-8") if path.exists() else None


def _apply(piece: _FilePiece, state: dict[str, Any]) -> Change | None:
    """Write *piece*, backing the file up the first time; ``None`` if it was already so."""
    key = str(piece.path)
    current = _read(piece.path)
    record = state["files"].get(key)
    if record is None:
        record: dict[str, Any] = {"existed": current is not None, "backup": current is not None}
        if current is not None:
            _write(_backup_path(piece.path), current)
        new = piece.add(current or "")
    elif current is not None and _sha(current) == record["written"]:
        # As Strata left it: build again from the original, so a refresh stays restorable.
        original = _read(_backup_path(piece.path)) if record["backup"] else ""
        new = piece.add(original or "")
    else:
        # Edited (or deleted) since: the original bytes would lose that edit.
        record["backup"] = False
        record["existed"] = True
        new = piece.add(piece.remove(current or ""))
    record["written"] = _sha(new)
    state["files"][key] = record
    if new == current:
        return None
    _write(piece.path, new)
    return Change(piece.what, piece.path)


def _revert(piece: _FilePiece, state: dict[str, Any]) -> tuple[Change, str] | None:
    key = str(piece.path)
    record = state["files"].pop(key, None)
    current = _read(piece.path)
    backup = _backup_path(piece.path)
    try:
        if record is None or current is None:
            return None
        if _sha(current) == record["written"]:
            if not record["existed"]:
                piece.path.unlink()
                return Change(piece.what, piece.path), "removed the file Strata created"
            if record["backup"]:
                _write(piece.path, backup.read_text(encoding="utf-8"))
                return Change(piece.what, piece.path), "restored as it was"
        _write(piece.path, piece.remove(current))
        return Change(piece.what, piece.path), "removed Strata's part (edited since install)"
    finally:
        backup.unlink(missing_ok=True)


# --- Claude Code's MCP entry -------------------------------------------------------


def _claude(*args: str) -> None:
    claude = shutil.which("claude")
    if claude is None:
        raise InstallError(
            "the `claude` CLI is not on PATH; install Claude Code or pass --target codex"
        )
    done = subprocess.run([claude, "mcp", *args], capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        raise InstallError(f"`claude mcp {args[0]}` failed: {(done.stderr or done.stdout).strip()}")


def _claude_entry() -> dict[str, Any] | None:
    path = _claude_json()
    if not path.exists():
        return None
    try:
        servers = json.loads(path.read_text()).get("mcpServers") or {}
    except json.JSONDecodeError:
        return None
    entry = servers.get(SERVER_NAME)
    return entry if isinstance(entry, dict) else None


def _mcp_entry(command: str) -> dict[str, Any]:
    return {"type": "stdio", "command": command, "args": ["mcp"], "env": {}, "alwaysLoad": True}


def _install_claude_entry(command: str, record: dict[str, Any]) -> Change | None:
    ours = _mcp_entry(command)
    existing = _claude_entry()
    if existing == ours:
        return None
    if existing is not None:
        # Someone else's (the user's own, or another install's): kept, to put back.
        record.setdefault("previous_entry", None if record.get("entry") else existing)
        _claude("remove", "--scope", "user", SERVER_NAME)
    _claude("add-json", "--scope", "user", SERVER_NAME, json.dumps(ours))
    record["entry"] = ours
    return Change("MCP server strata (user scope)", _claude_json())


def _uninstall_claude_entry(record: dict[str, Any]) -> tuple[Change, str] | None:
    ours = record.get("entry")
    existing = _claude_entry()
    if ours is None or existing != ours:
        return None
    _claude("remove", "--scope", "user", SERVER_NAME)
    previous = record.get("previous_entry")
    if previous is not None:
        _claude("add-json", "--scope", "user", SERVER_NAME, json.dumps(previous))
        return Change(
            "MCP server strata (user scope)", _claude_json()
        ), "put back the one there before"
    return Change("MCP server strata (user scope)", _claude_json()), "removed"


# --- commands ----------------------------------------------------------------------


def available_targets() -> list[str]:
    """The agents whose CLI is on PATH, for ``--target auto``."""
    return [target for target in TARGETS if shutil.which(target)]


def _target_key(target: str) -> str:
    return f"{target}:{claude_dir() if target == 'claude' else codex_dir()}"


def install(targets: list[str], command: str) -> list[Change]:
    changes: list[Change] = []
    with _lock():
        state = _load_state()
        # Every file checked before anything is written: a refusal leaves nothing behind.
        for target in targets:
            for piece in _file_pieces(target, command):
                piece.add(_read(piece.path) or "")
        for target in targets:
            record = state["targets"].setdefault(_target_key(target), {"target": target})
            if target == "claude":
                entry = _install_claude_entry(command, record)
                changes += [entry] if entry else []
            for piece in _file_pieces(target, command):
                change = _apply(piece, state)
                changes += [change] if change else []
            if target == "claude":
                changes += _retire_skill()
        state["version"] = version("strata-notebook")
        state["command"] = command
        _save_state(state)
    return changes


def uninstall() -> list[tuple[Change, str]]:
    done: list[tuple[Change, str]] = []
    with _lock():
        state = _load_state()
        for key, record in list(state["targets"].items()):
            target = record["target"]
            for piece in _file_pieces(target, state.get("command", "")):
                if str(piece.path).startswith(key.split(":", 1)[1]):
                    reverted = _revert(piece, state)
                    done += [reverted] if reverted else []
            if target == "claude":
                entry = _uninstall_claude_entry(record)
                done += [entry] if entry else []
            del state["targets"][key]
        _save_state(state)
    return done


def refresh() -> list[Change]:
    """Rewrite the file pieces another Strata version wrote; nothing if none did.

    For ``strata mcp`` on start. The MCP entry is left as it is: it started this.
    """
    state_path = _state_path()
    if not state_path.exists():
        return []
    with _lock():
        state = _load_state()
        if state.get("version") == version("strata-notebook"):
            return []
        changes: list[Change] = []
        for key, record in state["targets"].items():
            prefix = key.split(":", 1)[1]
            for piece in _file_pieces(record["target"], state["command"]):
                if str(piece.path).startswith(prefix):
                    change = _apply(piece, state)
                    changes += [change] if change else []
        state["version"] = version("strata-notebook")
        _save_state(state)
    return changes


def _retire_skill() -> list[Change]:
    """Move a hand-copied scratchpad skill aside: the MCP server and the block replace it."""
    skill = claude_dir() / "skills" / _SKILL
    if not skill.exists():
        return []
    destination = _home() / "retired-skills" / _SKILL
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = destination.with_name(f"{_SKILL}-{suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(skill), destination)
    return [Change(f"retired skill, moved to {destination}", skill)]
