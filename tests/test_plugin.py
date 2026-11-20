"""The Claude Code plugin (``plugins/strata-scratchpad``): the same MCP server and hooks
``strata agent install`` writes, for users who install through ``/plugin``."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from strata import agent_hook, mcp_bridge

REPO = Path(__file__).resolve().parents[1]
PLUGIN = REPO / "plugins" / "strata-scratchpad"
MANIFEST = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
SERVERS = json.loads((PLUGIN / ".mcp.json").read_text())["mcpServers"]
HOOKS = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]


def _commands() -> dict[str, str]:
    return {event: groups[0]["hooks"][0]["command"] for event, groups in HOOKS.items()}


def test_the_plugins_hook_is_the_packages():
    """A copy, run by path from the plugin: the package's is under a site-packages the
    plugin cannot name."""
    assert (PLUGIN / "hooks" / "agent_hook.py").read_bytes() == Path(
        agent_hook.__file__
    ).read_bytes()


def test_the_server_is_strata_mcp_loaded_up_front():
    assert SERVERS == {"strata": {"command": "strata", "args": ["mcp"], "alwaysLoad": True}}


def test_the_hooks_name_the_run_tool_as_claude_code_names_a_plugins():
    """``mcp__plugin_<plugin>_<server>__<tool>``, for a tool the server lists."""
    (server,) = SERVERS
    tools = {tool["name"] for tool in mcp_bridge.listed_tools()}
    assert "run" in tools
    tool = f"mcp__plugin_{MANIFEST['name']}_{server}__run"

    assert set(_commands()) == {"SessionStart", "UserPromptSubmit"}
    for command in _commands().values():
        assert command.split(" || ")[0].endswith(f" {tool}")


@pytest.mark.skipif(sys.platform == "win32", reason="the commands are bash's")
def test_each_hook_says_the_scratchpad_as_claude_code_runs_it(tmp_path):
    (tmp_path / "pyproject.toml").write_text("")
    digest = tmp_path / agent_hook.DIGEST
    digest.parent.mkdir(parents=True)
    cell = {"cell": "c1", "defines": ["df"], "status": "ready", "upstream": [], "reads": {}}
    digest.write_text(json.dumps({"version": 1, "environment": {}, "cells": [cell]}))
    env = {**os.environ, "CLAUDE_PLUGIN_ROOT": str(PLUGIN)}
    payload = json.dumps({"cwd": str(tmp_path), "prompt": "plot df"})

    for command in _commands().values():
        done = subprocess.run(
            ["bash", "-c", command], input=payload, capture_output=True, text=True, env=env
        )

        assert (done.returncode, done.stderr) == (0, "")
        assert "- df (cell c1): cached" in done.stdout
        assert "mcp__plugin_strata-scratchpad_strata__run" in done.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="the commands are bash's")
def test_a_hook_whose_script_is_gone_never_blocks_the_prompt(tmp_path):
    env = {**os.environ, "CLAUDE_PLUGIN_ROOT": str(tmp_path)}

    done = subprocess.run(
        ["bash", "-c", _commands()["UserPromptSubmit"]], input=b"{}", capture_output=True, env=env
    )

    assert done.returncode == 0


def test_the_skill_is_retired():
    """The MCP instructions and the hooks replace it; arm B's eval keeps its own copy."""
    assert not (PLUGIN / "skills").exists()
    assert not (Path(agent_hook.__file__).parent / ".agents").exists()


def test_the_marketplace_lists_the_plugin():
    marketplace = json.loads((REPO / ".claude-plugin" / "marketplace.json").read_text())

    assert any(
        p["name"] == MANIFEST["name"] and p["source"] == "./plugins/strata-scratchpad"
        for p in marketplace["plugins"]
    )
