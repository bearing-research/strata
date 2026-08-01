"""Tests for the ``strata agent`` on-ramp launcher (offline pieces).

Covers create-or-open resolution, the agent-config file writers, and URL
parsing. The server-spawn / TUI-attach paths need a live process and are
exercised by hand, not here.
"""

from __future__ import annotations

import json

import pytest

from strata.notebook import agent_launch


def test_server_endpoints_defaults_and_explicit() -> None:
    assert agent_launch._server_endpoints("http://localhost:8765") == ("localhost", 8765)
    assert agent_launch._server_endpoints("http://127.0.0.1:9000") == ("127.0.0.1", 9000)
    # No explicit port → the notebook server default, not HTTP's 80.
    assert agent_launch._server_endpoints("http://localhost") == ("localhost", 8765)


def test_resolve_notebook_dir_creates_then_reuses(tmp_path) -> None:
    target = tmp_path / "demo"
    # First call scaffolds it (no venv build, to keep the test fast/offline).
    created = agent_launch._resolve_notebook_dir(str(target), None, initialize_environment=False)
    assert (created / "notebook.toml").is_file()

    # Second call sees notebook.toml and returns the same dir untouched.
    reused = agent_launch._resolve_notebook_dir(str(created), None, initialize_environment=False)
    assert reused == created


def test_write_agent_config_writes_mcp_json(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-abc")

    cfg = json.loads((nb / ".mcp.json").read_text())
    server = cfg["mcpServers"]["strata-notebook"]
    assert server == {"type": "http", "url": "http://localhost:8765/mcp"}


def test_write_agent_config_claude_md_has_session_and_markers(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-xyz")

    text = (nb / "CLAUDE.md").read_text()
    assert agent_launch._BLOCK_START in text
    assert agent_launch._BLOCK_END in text
    assert "sess-xyz" in text
    # Points the agent at the MCP tools, not scratch scripts.
    assert "list_notebooks" in text
    assert "run_cell" in text


def test_write_agent_config_is_idempotent(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-aaa111")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-bbb222")

    text = (nb / "CLAUDE.md").read_text()
    # The managed block is rewritten in place, not duplicated.
    assert text.count(agent_launch._BLOCK_START) == 1
    assert text.count(agent_launch._BLOCK_END) == 1
    # And it reflects the latest launch's session id.
    assert "sess-bbb222" in text
    assert "sess-aaa111" not in text


def test_write_agent_config_preserves_user_claude_md(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    (nb / "CLAUDE.md").write_text("# My own notes\n\nKeep me.\n")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-1")

    text = (nb / "CLAUDE.md").read_text()
    assert "# My own notes" in text
    assert "Keep me." in text
    assert agent_launch._BLOCK_START in text


def test_write_agent_config_rewrites_block_keeps_user_text(tmp_path) -> None:
    nb = tmp_path / "nb"
    nb.mkdir()
    (nb / "CLAUDE.md").write_text("# Mine\n\nprose\n")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-first99")
    agent_launch._write_agent_config(nb, "http://localhost:8765", "sess-second99")

    text = (nb / "CLAUDE.md").read_text()
    assert "# Mine" in text
    assert "prose" in text
    assert text.count(agent_launch._BLOCK_START) == 1
    assert "sess-second99" in text
    assert "sess-first99" not in text


@pytest.mark.parametrize("status", [400, 405, 406, 200])
def test_mcp_mounted_true_for_non_404(monkeypatch, status) -> None:
    class _Resp:
        status_code = status

    monkeypatch.setattr(agent_launch.httpx, "get", lambda *a, **k: _Resp())
    assert agent_launch._mcp_mounted("http://localhost:8765") is True


def test_mcp_mounted_false_for_404(monkeypatch) -> None:
    class _Resp:
        status_code = 404

    monkeypatch.setattr(agent_launch.httpx, "get", lambda *a, **k: _Resp())
    assert agent_launch._mcp_mounted("http://localhost:8765") is False


def test_mcp_mounted_false_when_unreachable(monkeypatch) -> None:
    def _boom(*a, **k):
        raise agent_launch.httpx.ConnectError("refused")

    monkeypatch.setattr(agent_launch.httpx, "get", _boom)
    assert agent_launch._mcp_mounted("http://localhost:8765") is False
