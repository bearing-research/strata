"""``strata agent install`` / ``uninstall``: the scratchpad in a coding agent's own config.

Per target, every piece is keyed or marker-fenced, so ``uninstall`` removes what
``install`` wrote and nothing else:

- Claude Code: the user-scope MCP entry, through the ``claude`` CLI (which owns
  ``~/.claude.json`` and may be writing it); the ``mcp__strata__*`` permission in
  ``settings.json``; a block in ``CLAUDE.md``, for subagents and harnesses that do not
  see MCP instructions. A hand-copied scratchpad skill is moved aside, and back. A
  ``strata`` entry of the user's own is refused, as Codex's is.
- Codex: an ``[mcp_servers.strata]`` block in ``config.toml``; a block in ``AGENTS.md``.

A file Strata writes is backed up first. ``uninstall`` puts its original bytes back
when it is as Strata left it, and removes only Strata's part when it was edited since.
Each target's record names the directory it was installed into, so ``uninstall`` needs
no ``CLAUDE_CONFIG_DIR`` or ``CODEX_HOME`` of its own. The record is saved even when a
step fails, so a failure halfway leaves what was done undoable. ``strata mcp`` rewrites
the file pieces on start once another Strata version wrote them.
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
_STATE_VERSION = 3


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


def _config_dir(target: str) -> tuple[Path, bool]:
    """Where *target* keeps its config now, and whether a variable said so."""
    variable = "CLAUDE_CONFIG_DIR" if target == "claude" else "CODEX_HOME"
    given = os.environ.get(variable)
    default = Path.home() / (".claude" if target == "claude" else ".codex")
    return (Path(given).absolute(), True) if given else (default, False)


def strata_command() -> tuple[str, str | None]:
    """The running ``strata``, absolute, and a warning when it is not a lasting one.

    Refused from uv's cache (``uvx``), which uv cleans up; a project's environment is
    allowed, with a warning: the config is global, so deleting that one environment
    breaks the server in every project.
    """
    argv0 = Path(sys.argv[0])
    if argv0.name == "strata" and argv0.is_file():
        command = argv0.absolute()
    else:
        found = shutil.which("strata")
        if found is None:
            raise InstallError("cannot find the `strata` command this runs as; is it on PATH?")
        command = Path(found).absolute()
    real = os.path.realpath(command)
    environment = Path(real).parent.parent
    cache = os.environ.get("UV_CACHE_DIR")
    if re.search(r"[/\\](archive|builds|environments)-v\d+[/\\]", real) or (
        cache and Path(real).is_relative_to(os.path.realpath(cache))
    ):
        raise InstallError(
            f"this strata runs from uv's cache ({real}), which uv cleans up; install it "
            "with `uv tool install 'strata-notebook[mcp]'` and run `strata agent install` again"
        )
    # A uv tool's environment holds its receipt, wherever UV_TOOL_DIR put it.
    in_tool = (environment / "uv-receipt.toml").exists()
    if not in_tool and (environment / "pyvenv.cfg").exists():
        return str(command), (
            f"the agent config runs {command}, in the environment {environment}: every "
            "project's agent uses it, so removing that environment breaks them all. "
            "`uv tool install 'strata-notebook[mcp]'` is the lasting choice"
        )
    return str(command), None


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


def _servers(text: str) -> dict[str, Any]:
    try:
        servers = tomllib.loads(text).get("mcp_servers", {})
    except tomllib.TOMLDecodeError as exc:
        raise InstallError(f"config.toml is not valid TOML ({exc}); fix it and run again") from exc
    return servers if isinstance(servers, dict) else {}


def _codex_add(command: str) -> Callable[[str], str]:
    def add(text: str) -> str:
        rest = _TOML.sub("", text)
        if SERVER_NAME in _servers(rest):
            raise InstallError(
                "config.toml already has its own [mcp_servers.strata]; remove it, or "
                "install for Claude Code only with --target claude"
            )
        # A JSON string is a TOML basic string.
        body = f'[mcp_servers.{SERVER_NAME}]\ncommand = {json.dumps(command)}\nargs = ["mcp"]'
        new = _add_block(rest, _fenced("#", "", body), _TOML)
        # A table header after an inline ``mcp_servers = {...}`` is not TOML.
        try:
            added = _servers(new).get(SERVER_NAME)
        except InstallError:
            added = None
        if not isinstance(added, dict) or added.get("command") != command:
            raise InstallError(
                "config.toml's mcp_servers cannot take a [mcp_servers.strata] table (is it "
                "an inline table?); add the server there by hand, or use --target claude"
            )
        return new

    return add


def _file_pieces(target: str, base: Path, command: str) -> list[_FilePiece]:
    if target == "claude":
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
    try:
        state = json.loads(path.read_text())
    except ValueError as exc:
        raise InstallError(f"{path} is not readable JSON ({exc}); move it aside to start over")
    if not isinstance(state, dict) or state.get("state") != _STATE_VERSION:
        raise InstallError(f"{path} was written by another Strata; move it aside to start over")
    return state


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
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise InstallError(f"{path} is not UTF-8 text ({exc}); fix it and run again") from exc


def _apply(piece: _FilePiece, state: dict[str, Any]) -> Change | None:
    """Write *piece*, backing the file up the first time; ``None`` if it was already so."""
    key = str(piece.path)
    current = _read(piece.path)
    record: dict[str, Any] | None = state["files"].get(key)
    if record is None:
        record = {"existed": current is not None, "backup": current is not None}
        new = piece.add(current or "")
        if current is not None:
            _write(_backup_path(piece.path), current)
    elif current is not None and _sha(current) == record["written"]:
        # As Strata left it: build again from the original, so a refresh stays restorable.
        original = _read(_backup_path(piece.path)) if record["backup"] else ""
        new = piece.add(original or "")
    else:
        # Edited (or deleted) since: the original bytes would lose that edit.
        new = piece.add(piece.remove(current or ""))
        record = {"existed": True, "backup": False}
    if new != current:
        _write(piece.path, new)
    record["written"] = _sha(new)
    state["files"][key] = record
    return Change(piece.what, piece.path) if new != current else None


def _revert(piece: _FilePiece, state: dict[str, Any]) -> tuple[Change, str] | None:
    key = str(piece.path)
    record = state["files"].get(key)
    if record is None:
        return None
    current = _read(piece.path)
    backup = _backup_path(piece.path)
    done: tuple[Change, str] | None = None
    if current is not None:
        change = Change(piece.what, piece.path)
        if _sha(current) == record["written"] and not record["existed"]:
            piece.path.unlink()
            done = change, "removed the file Strata created"
        elif _sha(current) == record["written"] and record["backup"]:
            _write(piece.path, backup.read_text(encoding="utf-8"))
            done = change, "restored as it was"
        else:
            _write(piece.path, piece.remove(current))
            done = change, "removed Strata's part (edited since install)"
    del state["files"][key]
    backup.unlink(missing_ok=True)
    return done


# --- Claude Code's MCP entry -------------------------------------------------------


def _claude_json(record: dict[str, Any]) -> Path:
    # Where the claude CLI keeps user-scope MCP servers: beside the config dir it was
    # pointed at, else in the home directory.
    if record["configured"]:
        return Path(record["dir"]) / ".claude.json"
    return Path.home() / ".claude.json"


def _claude(record: dict[str, Any], *args: str) -> None:
    claude = shutil.which("claude")
    if claude is None:
        raise InstallError(_NO_CLAUDE)
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CONFIG_DIR"}
    if record["configured"]:
        env["CLAUDE_CONFIG_DIR"] = record["dir"]
    try:
        done = subprocess.run(
            [claude, "mcp", *args], capture_output=True, text=True, timeout=120, env=env
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise InstallError(f"`claude mcp {args[0]}` did not run: {exc}") from exc
    if done.returncode != 0:
        raise InstallError(f"`claude mcp {args[0]}` failed: {(done.stderr or done.stdout).strip()}")


_NO_CLAUDE = "the `claude` CLI is not on PATH; install Claude Code or pass --target codex"


def _claude_entry(record: dict[str, Any]) -> dict[str, Any] | None:
    path = _claude_json(record)
    if not path.exists():
        return None
    try:
        servers = json.loads(path.read_text()).get("mcpServers") or {}
    except (ValueError, OSError):
        return None
    entry = servers.get(SERVER_NAME)
    return entry if isinstance(entry, dict) else None


def _mcp_entry(command: str) -> dict[str, Any]:
    return {"type": "stdio", "command": command, "args": ["mcp"], "env": {}, "alwaysLoad": True}


def _same_server(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Whether two entries start the same server: the CLI may store the rest its own way."""
    if a is None or b is None:
        return False
    return (a.get("command"), a.get("args")) == (b.get("command"), b.get("args"))


def _check_claude(record: dict[str, Any]) -> None:
    """Refuse before anything is written: no ``claude``, or a ``strata`` entry not Strata's.

    One of the user's own is theirs to remove, as Codex's is: replacing it and putting it
    back would leave it lost whenever the CLI fails between the two.
    """
    if shutil.which("claude") is None:
        raise InstallError(_NO_CLAUDE)
    existing = _claude_entry(record)
    if existing is not None and not _same_server(existing, record.get("entry")):
        raise InstallError(
            f"Claude Code already has its own `strata` MCP server in {_claude_json(record)}; "
            "remove it with `claude mcp remove --scope user strata` and run this again, or "
            "install for Codex only with --target codex"
        )


def _install_claude_entry(command: str, record: dict[str, Any]) -> Change | None:
    ours = _mcp_entry(command)
    existing = _claude_entry(record)
    if _same_server(existing, ours):
        record["entry"] = ours
        return None
    if existing is not None:
        # Strata's own, from an earlier install (another command): replaced.
        _claude(record, "remove", "--scope", "user", SERVER_NAME)
    try:
        _claude(record, "add-json", "--scope", "user", SERVER_NAME, json.dumps(ours))
    except InstallError as exc:
        if _same_server(_claude_entry(record), ours):
            # Written, then failed (a timeout): Strata's, so recorded as such.
            record["entry"] = ours
            raise
        if existing is not None:
            try:
                _claude(record, "add-json", "--scope", "user", SERVER_NAME, json.dumps(existing))
            except InstallError as again:
                record.pop("entry", None)
                raise InstallError(
                    f"{exc}; putting the earlier strata entry back failed too ({again}), so "
                    "Claude Code has none now: run `strata agent install` again"
                ) from exc
        raise
    record["entry"] = ours
    return Change("MCP server strata (user scope)", _claude_json(record))


def _uninstall_claude_entry(record: dict[str, Any]) -> tuple[Change, str] | None:
    if not _same_server(_claude_entry(record), record.get("entry")):
        # Not Strata's (replaced since, or removed): left as it is.
        return None
    if shutil.which("claude") is None:
        # Claude Code is gone; the rest is still removed, so uninstall can finish.
        change = Change("MCP server strata (user scope)", _claude_json(record))
        return change, "left in place: no `claude` CLI to remove it with"
    _claude(record, "remove", "--scope", "user", SERVER_NAME)
    return Change("MCP server strata (user scope)", _claude_json(record)), "removed"


# --- the skill ---------------------------------------------------------------------


def _retire_skill(record: dict[str, Any]) -> list[Change]:
    """Move a hand-copied scratchpad skill aside: the MCP server and the block replace it."""
    skill = Path(record["dir"]) / "skills" / _SKILL
    if not skill.exists():
        return []
    destination = _home() / "retired-skills" / _SKILL
    suffix = 1
    while destination.exists():
        suffix += 1
        destination = destination.with_name(f"{_SKILL}-{suffix}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(skill), destination)
    record.setdefault("skills", []).append([str(skill), str(destination)])
    return [Change(f"retired skill, moved to {destination}", skill)]


def _restore_skill(record: dict[str, Any]) -> list[tuple[Change, str]]:
    """The newest retired copy back in place; any older one is said where it stays."""
    done: list[tuple[Change, str]] = []
    for skill, destination in reversed(record.get("skills") or []):
        skill, destination = Path(skill), Path(destination)
        if not destination.exists():
            continue
        if skill.exists():
            done.append((Change("an older copy of the skill", destination), "left there"))
            continue
        skill.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(destination), skill)
        done.append((Change("skill", skill), "put back"))
    return done


# --- commands ----------------------------------------------------------------------


def available_targets() -> list[str]:
    """The agents whose CLI is on PATH, for ``--target auto``."""
    return [target for target in TARGETS if shutil.which(target)]


def _target_key(target: str, base: Path, configured: bool) -> str:
    # Claude's entry lives beside a configured dir, else in the home directory: two installs.
    where = ":configured" if target == "claude" and configured else ""
    return f"{target}:{base}{where}"


def install(targets: list[str], command: str) -> list[Change]:
    changes: list[Change] = []
    with _lock():
        state = _load_state()
        found = {target: _config_dir(target) for target in targets}
        records = {
            target: state["targets"].get(_target_key(target, base, configured))
            or {"target": target, "dir": str(base), "configured": configured}
            for target, (base, configured) in found.items()
        }
        # Every piece checked before anything is written: a refusal leaves nothing behind.
        for target, (base, _) in found.items():
            if target == "claude":
                _check_claude(records[target])
            for piece in _file_pieces(target, base, command):
                piece.add(_read(piece.path) or "")
        try:
            for target, (base, configured) in found.items():
                record = records[target]
                # Recorded before its first write, so a failure halfway is still undone;
                # complete only once all of it is written, which ``refresh`` waits for.
                record["complete"] = False
                record["command"] = command
                state["targets"][_target_key(target, base, configured)] = record
                if target == "claude":
                    entry = _install_claude_entry(command, record)
                    changes += [entry] if entry else []
                for piece in _file_pieces(target, base, command):
                    change = _apply(piece, state)
                    changes += [change] if change else []
                if target == "claude":
                    changes += _retire_skill(record)
                record["complete"] = True
            state["version"] = version("strata-notebook")
        finally:
            _save_state(state)
    return changes


def uninstall() -> list[tuple[Change, str]]:
    done: list[tuple[Change, str]] = []
    with _lock():
        state = _load_state()
        try:
            for key, record in list(state["targets"].items()):
                target = record["target"]
                # Half undone, should the rest fail: never finished again by a refresh.
                record["complete"] = False
                for piece in _file_pieces(target, Path(record["dir"]), record["command"]):
                    reverted = _revert(piece, state)
                    done += [reverted] if reverted else []
                if target == "claude":
                    entry = _uninstall_claude_entry(record)
                    done += [entry] if entry else []
                    done += _restore_skill(record)
                del state["targets"][key]
        finally:
            _save_state(state)
    return done


def refresh() -> list[Change]:
    """Rewrite the file pieces another Strata version wrote; nothing if none did.

    For ``strata mcp`` on start. The MCP entry is left as it is: it started this. A
    target whose install failed halfway is left for ``install`` or ``uninstall``.
    """
    if not _state_path().exists():
        return []
    with _lock():
        state = _load_state()
        if state.get("version") == version("strata-notebook"):
            return []
        changes: list[Change] = []
        try:
            for record in state["targets"].values():
                if not record.get("complete"):
                    continue
                for piece in _file_pieces(record["target"], Path(record["dir"]), record["command"]):
                    change = _apply(piece, state)
                    changes += [change] if change else []
            state["version"] = version("strata-notebook")
        finally:
            _save_state(state)
    return changes
