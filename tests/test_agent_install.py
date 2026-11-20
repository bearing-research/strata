"""``strata agent install`` / ``uninstall``: what each writes into an agent's config."""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import sys
import tomllib
from pathlib import Path

import pytest

from strata import agent_install
from strata.cli import main

STRATA = "/opt/tools/bin/strata"

# The claude CLI's `mcp add-json` / `mcp remove`, on the file the real one writes.
_FAKE_CLAUDE = """\
import json, os, sys
from pathlib import Path
configured = os.environ.get("CLAUDE_CONFIG_DIR")
path = Path(configured or os.environ["HOME"]) / ".claude.json"
data = json.loads(path.read_text()) if path.exists() else {}
servers = data.setdefault("mcpServers", {})
verb, name = sys.argv[2], sys.argv[5]
assert sys.argv[1] == "mcp" and sys.argv[3:5] == ["--scope", "user"], sys.argv
if verb == "add-json":
    if json.loads(sys.argv[6])["command"] == os.environ.get("FAKE_CLAUDE_REJECTS"):
        sys.exit("unknown key alwaysLoad")
    if name in servers:
        sys.exit(f"MCP server {name} already exists in user config")
    servers[name] = json.loads(sys.argv[6])
else:
    del servers[name]
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data))
"""


@pytest.fixture
def agent_home(tmp_path, monkeypatch):
    """Claude Code's and Codex's config dirs, empty, with a fake ``claude`` on PATH."""
    claude = tmp_path / "claude-config"
    codex = tmp_path / "codex-home"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("CODEX_HOME", str(codex))
    # The claude CLI's default, beside the home directory: the test's own.
    monkeypatch.setenv("HOME", str(Path.home()))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "claude"
    script.write_text(f"#!{sys.executable}\n{_FAKE_CLAUDE}")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return claude, codex


def _servers(claude: Path) -> dict:
    path = claude / ".claude.json"
    return json.loads(path.read_text()).get("mcpServers", {}) if path.exists() else {}


def _snapshot(*dirs: Path) -> dict[str, bytes]:
    """Every file under *dirs* but ``.claude.json``, which the claude CLI owns."""
    return {
        str(path): path.read_bytes()
        for base in dirs
        if base.exists()
        for path in sorted(base.rglob("*"))
        if path.is_file() and path.name != ".claude.json"
    }


@pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a shebang script")
def test_install_writes_each_piece_and_uninstall_leaves_the_files_byte_identical(agent_home):
    claude, codex = agent_home
    claude.mkdir()
    (claude / "settings.json").write_text(
        '{\n    "theme": "dark",\n    "permissions": {"allow": ["Bash(ls)"]}\n}'
    )
    (claude / "CLAUDE.md").write_text("# Mine\n\nKeep this.")
    codex.mkdir()
    (codex / "config.toml").write_text('model = "o3"\n\n[mcp_servers.other]\ncommand = "x"\n')
    before = _snapshot(claude, codex)

    agent_install.install(["claude", "codex"], STRATA)

    settings = json.loads((claude / "settings.json").read_text())
    assert settings["theme"] == "dark"
    assert settings["permissions"]["allow"] == ["Bash(ls)", "mcp__strata__*"]
    claude_md = (claude / "CLAUDE.md").read_text()
    assert claude_md.startswith("# Mine\n\nKeep this.\n\n<!-- strata:begin v")
    assert "`mcp__strata__run`" in claude_md and "strata scratch run" in claude_md
    assert _servers(claude)["strata"] == {
        "type": "stdio",
        "command": STRATA,
        "args": ["mcp"],
        "env": {},
        "alwaysLoad": True,
    }
    config = tomllib.loads((codex / "config.toml").read_text())
    assert config["model"] == "o3"
    assert config["mcp_servers"]["other"] == {"command": "x"}
    assert config["mcp_servers"]["strata"] == {"command": STRATA, "args": ["mcp"]}
    assert "`strata` MCP server's `run` tool" in (codex / "AGENTS.md").read_text()

    agent_install.uninstall()

    assert _snapshot(claude, codex) == before
    assert "strata" not in _servers(claude)
    assert not (Path.home() / ".strata" / "agent-install" / "state.json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a shebang script")
def test_installing_twice_changes_nothing_and_still_uninstalls_clean(agent_home):
    claude, _ = agent_home
    claude.mkdir()
    (claude / "CLAUDE.md").write_text("mine\n")

    first = agent_install.install(["claude"], STRATA)
    written = _snapshot(claude)
    second = agent_install.install(["claude"], STRATA)
    agent_install.uninstall()

    assert {change.what for change in first} >= {"scratchpad block", "permission mcp__strata__*"}
    assert second == []
    assert _snapshot(claude) == {str(claude / "CLAUDE.md"): b"mine\n"}
    assert written != _snapshot(claude)


@pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a shebang script")
def test_a_file_edited_after_install_keeps_the_edit_and_loses_only_strata(agent_home):
    claude, _ = agent_home
    claude.mkdir()
    (claude / "CLAUDE.md").write_text("mine\n")
    agent_install.install(["claude"], STRATA)
    edited = (claude / "CLAUDE.md").read_text() + "\nadded later\n"
    (claude / "CLAUDE.md").write_text(edited)
    settings = json.loads((claude / "settings.json").read_text())
    settings["model"] = "opus"
    (claude / "settings.json").write_text(json.dumps(settings))

    done = dict((change.where.name, how) for change, how in agent_install.uninstall())

    text = (claude / "CLAUDE.md").read_text()
    assert "strata:begin" not in text
    assert text.startswith("mine\n") and text.endswith("added later\n")
    assert json.loads((claude / "settings.json").read_text()) == {
        "permissions": {"allow": []},
        "model": "opus",
    }
    assert done["CLAUDE.md"] == "removed Strata's part (edited since install)"


@pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a shebang script")
def test_the_users_own_strata_entry_is_refused_and_nothing_is_written(agent_home):
    """Theirs to remove, as Codex's is: replacing it and putting it back loses it whenever
    the CLI fails between the two."""
    claude, _ = agent_home
    claude.mkdir()
    theirs = {"type": "stdio", "command": "strata", "args": ["mcp"], "alwaysLoad": True}
    (claude / ".claude.json").write_text(json.dumps({"mcpServers": {"strata": theirs}}))

    with pytest.raises(agent_install.InstallError, match="claude mcp remove --scope user"):
        agent_install.install(["claude"], STRATA)

    assert _servers(claude)["strata"] == theirs
    assert _snapshot(claude) == {}


def test_an_existing_codex_strata_server_is_refused_and_nothing_is_written(agent_home):
    _, codex = agent_home
    codex.mkdir()
    (codex / "config.toml").write_text('[mcp_servers.strata]\ncommand = "mine"\n')

    with pytest.raises(agent_install.InstallError, match="already has its own"):
        agent_install.install(["codex"], STRATA)

    assert (codex / "config.toml").read_text() == '[mcp_servers.strata]\ncommand = "mine"\n'
    assert not (codex / "AGENTS.md").exists()


def test_a_settings_file_that_is_not_json_is_refused(agent_home):
    claude, _ = agent_home
    claude.mkdir()
    (claude / "settings.json").write_text("{not json")

    with pytest.raises(agent_install.InstallError, match="settings.json is not valid JSON"):
        agent_install.install(["claude"], STRATA)

    assert (claude / "settings.json").read_text() == "{not json"
    assert _servers(claude) == {}
    assert not (claude / "CLAUDE.md").exists()


def test_refresh_rewrites_what_another_version_wrote(agent_home, monkeypatch):
    _, codex = agent_home
    agent_install.install(["codex"], STRATA)
    agents_md = codex / "AGENTS.md"
    monkeypatch.setattr(agent_install, "version", lambda name: "99.0.0")

    changes = agent_install.refresh()

    assert {change.where for change in changes} == {agents_md, codex / "config.toml"}
    assert "strata:begin v99.0.0" in agents_md.read_text()
    assert agent_install.refresh() == []
    agent_install.uninstall()
    assert not agents_md.exists()


def test_refresh_with_nothing_installed_does_nothing(agent_home):
    assert agent_install.refresh() == []
    assert not (Path.home() / ".strata" / "agent-install").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a shebang script")
def test_a_hand_copied_skill_is_moved_aside(agent_home):
    claude, _ = agent_home
    skill = claude / "skills" / "strata-scratchpad"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("old")

    agent_install.install(["claude"], STRATA)

    assert not skill.exists()
    moved = Path.home() / ".strata" / "agent-install" / "retired-skills" / "strata-scratchpad"
    assert (moved / "SKILL.md").read_text() == "old"


@pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a shebang script")
def test_the_command_installs_and_uninstalls(agent_home, capsys, monkeypatch):
    claude, _ = agent_home
    monkeypatch.setattr(agent_install, "strata_command", lambda: (STRATA, None))

    assert main(["agent", "install", "--target", "claude", "--yes"]) == 0
    assert "wrote permission mcp__strata__*" in capsys.readouterr().out
    assert main(["agent", "uninstall"]) == 0

    assert "removed the file Strata created" in capsys.readouterr().out
    assert _snapshot(claude) == {}


def test_the_command_asks_first_unless_told_yes(agent_home, monkeypatch, capsys):
    # Not a terminal, so nobody could answer.
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))

    assert main(["agent", "install", "--target", "codex"]) == 2
    assert "pass --yes" in capsys.readouterr().err
    assert not agent_home[1].exists()


def test_an_unknown_target_is_refused(agent_home, capsys):
    assert main(["agent", "install", "--target", "cursor", "--yes"]) == 2
    assert "unknown target cursor" in capsys.readouterr().out


@pytest.mark.skipif(shutil.which("claude") is None, reason="Claude Code is not installed")
def test_the_real_claude_cli_takes_the_entry_and_gives_it_back(tmp_path, monkeypatch):
    """The fake above stands in for this CLI; here it is the real one."""
    claude = tmp_path / "claude-config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))

    agent_install.install(["claude"], STRATA)
    assert _servers(claude)["strata"]["alwaysLoad"] is True
    agent_install.uninstall()

    assert "strata" not in _servers(claude)


def test_strata_mcp_refreshes_what_another_version_wrote_before_it_serves(
    agent_home, monkeypatch, tmp_path
):
    import strata.mcp_bridge

    _, codex = agent_home
    agent_install.install(["codex"], STRATA)
    monkeypatch.setattr(agent_install, "version", lambda name: "99.0.0")
    served: list[Path] = []
    monkeypatch.setattr(strata.mcp_bridge, "mcp_main", lambda root: served.append(root) or 0)

    assert main(["mcp", "--root", str(tmp_path)]) == 0

    assert served
    assert "strata:begin v99.0.0" in (codex / "AGENTS.md").read_text()


def test_a_refresh_that_fails_does_not_keep_strata_mcp_from_serving(
    agent_home, monkeypatch, tmp_path, capsys
):
    import strata.mcp_bridge

    def refuse() -> list:
        raise agent_install.InstallError("config.toml is not valid TOML")

    monkeypatch.setattr(agent_install, "refresh", refuse)
    monkeypatch.setattr(strata.mcp_bridge, "mcp_main", lambda root: 0)

    assert main(["mcp", "--root", str(tmp_path)]) == 0
    assert "could not refresh the installed agent config" in capsys.readouterr().err


def test_every_command_and_tool_the_block_names_exists():
    import re
    import shlex

    from strata.agent_guidance import section
    from strata.cli import _build_parser
    from strata.mcp_bridge import listed_tools

    text = section("agent_block")
    commands = re.findall(r"`(strata [^`]+)`", text)
    assert len(commands) == 2
    for command in commands:
        _build_parser().parse_args(shlex.split(command.replace("<code>", "1 + 1"))[1:])
    rendered = agent_install._markdown_block("`mcp__strata__run`", "`mcp__strata__scratchpad`")
    tools = {tool["name"] for tool in listed_tools()}
    assert {name for name in re.findall(r"`mcp__strata__(\w+)`", rendered)} == tools


def test_an_edit_then_a_reinstall_still_keeps_the_edit_on_uninstall(agent_home):
    """Installing again records the edited file; its original bytes would lose the edit."""
    _, codex = agent_home
    codex.mkdir()
    (codex / "AGENTS.md").write_text("mine\n")
    agent_install.install(["codex"], STRATA)
    (codex / "AGENTS.md").write_text((codex / "AGENTS.md").read_text() + "\nadded later\n")

    agent_install.install(["codex"], STRATA)
    agent_install.uninstall()

    text = (codex / "AGENTS.md").read_text()
    assert "strata:begin" not in text
    assert text.startswith("mine\n") and text.endswith("added later\n")


def test_a_refresh_at_the_same_version_leaves_a_removed_block_removed(agent_home):
    _, codex = agent_home
    agent_install.install(["codex"], STRATA)
    agents_md = codex / "AGENTS.md"
    agents_md.write_text("the user took it out\n")

    assert agent_install.refresh() == []
    assert agents_md.read_text() == "the user took it out\n"


_SHEBANG = pytest.mark.skipif(sys.platform == "win32", reason="the fake claude is a shebang script")


@_SHEBANG
def test_no_claude_cli_is_refused_before_anything_is_written(agent_home, monkeypatch):
    _, codex = agent_home
    empty = agent_home[0].parent / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(agent_install.InstallError, match="`claude` CLI is not on PATH"):
        agent_install.install(["codex", "claude"], STRATA)

    assert _snapshot(codex) == {}


@_SHEBANG
def test_a_failure_halfway_is_undone_and_never_finished_by_a_refresh(agent_home, monkeypatch):
    claude, codex = agent_home
    monkeypatch.setenv("FAKE_CLAUDE_REJECTS", STRATA)

    with pytest.raises(agent_install.InstallError, match="unknown key alwaysLoad"):
        agent_install.install(["codex", "claude"], STRATA)
    assert (codex / "AGENTS.md").exists()
    # A later `strata mcp` (Codex's) must not write Claude's half.
    monkeypatch.setattr(agent_install, "version", lambda name: "9.9.9")
    agent_install.refresh()
    assert _snapshot(claude) == {}
    agent_install.uninstall()

    assert _snapshot(codex) == {}


@_SHEBANG
def test_a_failed_add_puts_strata_s_earlier_entry_back(agent_home, monkeypatch):
    claude, _ = agent_home
    agent_install.install(["claude"], STRATA)
    monkeypatch.setenv("FAKE_CLAUDE_REJECTS", "/new/strata")

    with pytest.raises(agent_install.InstallError, match="unknown key alwaysLoad"):
        agent_install.install(["claude"], "/new/strata")

    assert _servers(claude)["strata"]["command"] == STRATA


def test_a_claude_cli_that_hangs_is_an_install_error(agent_home, monkeypatch):
    import subprocess

    def hangs(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 120)

    monkeypatch.setattr(agent_install.subprocess, "run", hangs)

    with pytest.raises(agent_install.InstallError, match="did not run"):
        agent_install.install(["claude"], STRATA)


@_SHEBANG
def test_the_default_location_and_a_configured_one_are_two_installs(agent_home, monkeypatch):
    """Claude reads ``~/.claude.json`` by default, and ``$CLAUDE_CONFIG_DIR/.claude.json``
    when it is set, though both name ``~/.claude``."""
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")
    default = Path.home() / ".claude"
    agent_install.install(["claude"], STRATA)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(default))
    agent_install.install(["claude"], STRATA)

    assert _servers(Path.home())["strata"]["command"] == STRATA
    assert _servers(default)["strata"]["command"] == STRATA
    agent_install.uninstall()
    assert "strata" not in _servers(Path.home())
    assert "strata" not in _servers(default)


@_SHEBANG
def test_uninstall_finds_each_install_by_the_directory_it_went_into(agent_home, monkeypatch):
    """Not by the environment uninstall runs in; nor a directory another one prefixes."""
    claude, _ = agent_home
    work = claude.parent / f"{claude.name}-work"
    for directory in (claude, work):
        directory.mkdir()
        (directory / "CLAUDE.md").write_text(f"{directory.name}\n")
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(directory))
        agent_install.install(["claude"], STRATA)
    monkeypatch.delenv("CLAUDE_CONFIG_DIR")

    agent_install.uninstall()

    for directory in (claude, work):
        assert _snapshot(directory) == {
            str(directory / "CLAUDE.md"): f"{directory.name}\n".encode()
        }
        assert "strata" not in _servers(directory)


@_SHEBANG
def test_an_entry_the_cli_stores_its_own_way_is_still_strata_s(agent_home):
    claude, _ = agent_home
    agent_install.install(["claude"], STRATA)
    data = json.loads((claude / ".claude.json").read_text())
    entry = data["mcpServers"]["strata"]
    del entry["env"]
    entry["timeout"] = 30
    (claude / ".claude.json").write_text(json.dumps(data))

    assert [c.what for c in agent_install.install(["claude"], STRATA)] == []
    agent_install.uninstall()

    assert "strata" not in _servers(claude)


@_SHEBANG
def test_an_entry_the_user_put_in_place_since_is_refused_and_kept(agent_home):
    claude, _ = agent_home
    agent_install.install(["claude"], STRATA)
    data = json.loads((claude / ".claude.json").read_text())
    theirs = {"type": "stdio", "command": "/their/strata", "args": ["mcp"]}
    data["mcpServers"]["strata"] = theirs
    (claude / ".claude.json").write_text(json.dumps(data))

    with pytest.raises(agent_install.InstallError, match="already has its own"):
        agent_install.install(["claude"], STRATA)
    agent_install.uninstall()

    assert _servers(claude)["strata"] == theirs


def test_an_inline_mcp_servers_table_is_refused_and_nothing_is_written(agent_home):
    _, codex = agent_home
    codex.mkdir()
    text = 'mcp_servers = { other = { command = "x" } }\n'
    (codex / "config.toml").write_text(text)

    with pytest.raises(agent_install.InstallError, match="inline table"):
        agent_install.install(["codex"], STRATA)

    assert (codex / "config.toml").read_text() == text
    assert not (codex / "AGENTS.md").exists()


@_SHEBANG
def test_the_skill_comes_back_on_uninstall(agent_home):
    claude, _ = agent_home
    skill = claude / "skills" / "strata-scratchpad"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("old")

    agent_install.install(["claude"], STRATA)
    # Copied in again and retired again: the newest comes back, the older is said.
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("newer")
    agent_install.install(["claude"], STRATA)
    done = agent_install.uninstall()

    assert (skill / "SKILL.md").read_text() == "newer"
    (older,) = [change.where for change, what in done if what == "left there"]
    assert (older / "SKILL.md").read_text() == "old"


@pytest.mark.parametrize(
    ("where", "outcome"),
    [
        ("cache/uv/archive-v0/abc/bin/strata", "refused"),
        ("share/uv/tools/strata-notebook/bin/strata", "quiet"),
        ("AppData/Roaming/uv/data/tools/strata-notebook/bin/strata", "quiet"),
        ("repo/.venv/bin/strata", "warned"),
        ("opt/bin/strata", "quiet"),
    ],
)
def test_the_command_written_is_a_lasting_one(tmp_path, monkeypatch, where, outcome):
    command = tmp_path / where
    command.parent.mkdir(parents=True)
    command.write_text("")
    # Every environment has one, a uv tool's too; told apart by where it is.
    if "/bin/" in where and not where.startswith("opt"):
        (command.parent.parent / "pyvenv.cfg").write_text("")
    monkeypatch.setattr(sys, "argv", [str(command), "agent", "install"])

    if outcome == "refused":
        with pytest.raises(agent_install.InstallError, match="uv's cache"):
            agent_install.strata_command()
        return
    found, warning = agent_install.strata_command()
    assert found == str(command)
    assert (warning is None) == (outcome == "quiet")


def test_a_torn_state_file_does_not_keep_strata_mcp_from_serving(
    agent_home, monkeypatch, tmp_path, capsys
):
    import strata.mcp_bridge

    state = Path.home() / ".strata" / "agent-install" / "state.json"
    state.parent.mkdir(parents=True)
    state.write_text('{"state": 2, "targ')
    monkeypatch.setattr(strata.mcp_bridge, "mcp_main", lambda root: 0)

    assert main(["mcp", "--root", str(tmp_path)]) == 0
    assert "could not refresh" in capsys.readouterr().err
