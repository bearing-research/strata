"""The scratchpad MCP profile: two tools over the server's one session.

Cells run for real (the cold harness on the test interpreter), so each view is
what an agent would read back.
"""

from __future__ import annotations

import asyncio
import json
import re
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from strata.agent_guidance import section
from strata.notebook import mcp_server, ws
from strata.notebook.mcp_server import (
    SCRATCHPAD_TOOLS,
    STALE_IN_AGENT_TERMS,
    _add_cell,
    _edit_cell,
    _scratch_run,
    _scratchpad,
    build_mcp_app,
)
from strata.notebook.models import StalenessReason
from strata.notebook.scopes import CLASSIFIED_TOOLS, required_scopes_for_tool
from strata.notebook.session import SessionManager
from strata.notebook.writer import create_notebook
from strata.types import Principal
from tests.notebook.e2e_fixtures import _reset_ws_globals


def _config(**overrides):
    settings = {
        "host": "0.0.0.0",
        "port": 8765,
        "public_base_url": None,
        "public_base_path": "",
        "principal_auth_enabled": False,
    }
    return SimpleNamespace(**{**settings, **overrides})


@pytest.fixture(autouse=True)
def _server(monkeypatch):
    """What the tools read of the running server's config; WS state reset around each test."""
    monkeypatch.setattr(mcp_server, "_server_config", _config)
    _reset_ws_globals()
    yield
    _reset_ws_globals()


@pytest.fixture
def scratch(tmp_path):
    """A SessionManager with one open notebook, and its session id."""
    sm = SessionManager()
    session = sm.open_notebook(create_notebook(tmp_path, "scratch"))
    return sm, session.id


@pytest.mark.asyncio
async def test_run_adds_a_cell_and_returns_what_it_showed(scratch):
    sm, session_id = scratch
    out = await _scratch_run(sm, "x = 40 + 2\nx")

    assert out["status"] == "ok"
    assert out["cache_hit"] is False
    assert out["value"] == 42
    assert out["defines"] == ["x"]
    assert out["url"] == f"http://127.0.0.1:8765/#/notebook/{session_id}"
    assert "fix" not in out and "reran" not in out and "saved_ms" not in out
    assert [cell.id for cell in sm.get_session(session_id).notebook_state.cells] == [out["cell"]]


@pytest.mark.asyncio
async def test_a_value_json_cannot_hold_shows_its_repr(scratch):
    """A list of tuples or a dict with int keys is pickled; its repr is the answer."""
    sm, _ = scratch
    out = await _scratch_run(sm, "sorted({101: 0.5, 102: 0.9}.items(), key=lambda kv: -kv[1])")

    assert out["value"] == "[(102, 0.9), (101, 0.5)]"


@pytest.mark.asyncio
async def test_an_unchanged_cell_hits_and_fresh_runs_it_again(scratch):
    sm, _ = scratch
    first = await _scratch_run(sm, "x = 1\nprint('ran')\nx")

    again = await _scratch_run(sm, "x = 1\nprint('ran')\nx", cell=first["cell"])
    assert again["cache_hit"] is True
    assert again["saved_ms"] == first["duration_ms"]
    assert again["value"] == 1

    fresh = await _scratch_run(sm, "x = 1\nprint('ran')\nx", cell=first["cell"], fresh=True)
    assert fresh["cache_hit"] is False
    assert fresh["stdout"] == "ran\n"


@pytest.mark.asyncio
async def test_run_names_the_upstream_cells_it_brought_up_to_date(scratch):
    sm, session_id = scratch
    upstream = await _scratch_run(sm, "x = 1")
    await _scratch_run(sm, "y = x + 1\ny")
    await _edit_cell(sm, session_id, upstream["cell"], "x = 10")

    out = await _scratch_run(sm, "z = x * 2\nz")
    assert out["reran"] == [{"cell": upstream["cell"], "reason": "its source changed"}]
    assert out["value"] == 20


@pytest.mark.asyncio
async def test_a_cell_that_raises_returns_its_traceback_and_no_fix(scratch):
    sm, _ = scratch
    out = await _scratch_run(sm, "1 / 0")
    assert out["status"] == "error"
    assert "ZeroDivisionError" in out["traceback"]
    assert "fix" not in out and "value" not in out


@pytest.mark.asyncio
async def test_a_refused_cell_gets_a_fix_naming_it(scratch):
    sm, _ = scratch
    out = await _scratch_run(sm, "# @timeout soon\nx = 1")
    assert out["status"] == "error"
    assert "traceback" not in out
    assert out["fix"] == f"fix what `error` names, then call `run` again with cell={out['cell']!r}"


@pytest.mark.asyncio
async def test_an_image_is_written_to_a_file(scratch):
    sm, _ = scratch
    out = await _scratch_run(
        sm,
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "fig, ax = plt.subplots()\nax.plot([1, 2])\nfig",
    )
    assert out["status"] == "ok", out
    assert "value" not in out
    with open(out["path"], "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


@pytest.mark.asyncio
async def test_without_one_open_session_run_and_scratchpad_say_how_to_get_one(tmp_path):
    sm = SessionManager()
    for refused in (await _scratch_run(sm, "x = 1"), _scratchpad(sm)):
        assert refused["status"] == "error"
        assert refused["error"] == "no notebook is open on this server"
        assert refused["fix"].startswith("open one: `strata agent --no-tui --server")

    sm.open_notebook(create_notebook(tmp_path, "one"))
    sm.open_notebook(create_notebook(tmp_path, "two"))
    out = await _scratch_run(sm, "x = 1")
    assert out["error"].startswith("2 notebooks are open")
    assert out["fix"] == "close all but one of them in the notebook UI"


@pytest.mark.asyncio
async def test_an_unknown_cell_is_refused(scratch):
    sm, _ = scratch
    out = await _scratch_run(sm, "x = 1", cell="ghost")
    assert out["error"] == "no cell 'ghost' in the scratchpad"
    assert "`scratchpad`" in out["fix"]


@pytest.mark.asyncio
async def test_an_unusable_environment_is_refused_before_a_cell_is_added(scratch, monkeypatch):
    sm, session_id = scratch
    session = sm.get_session(session_id)
    monkeypatch.setattr(session, "environment_execution_block_message", lambda: "not ready")
    monkeypatch.setattr(session, "environment_sync_state", "failed")

    out = await _scratch_run(sm, "x = 1")
    assert out == {
        "status": "error",
        "error": "not ready",
        "fix": f"run `uv sync` in {session.path}, then call `run` again",
    }
    assert session.notebook_state.cells == []


@pytest.mark.asyncio
async def test_scratchpad_lists_what_exists_and_finds_a_name(scratch):
    sm, session_id = scratch
    cell = (await _scratch_run(sm, "df = list(range(3))\ndf"))["cell"]
    session = sm.get_session(session_id)

    inventory = _scratchpad(sm)
    assert inventory["notebook"] == str(session.path)
    assert inventory["cwd"] == str(session.path / "work")
    assert inventory["python"] == str(session.venv_python)
    (entry,) = inventory["cells"]
    assert entry["cell"] == cell and entry["defines"] == ["df"] and entry["status"] == "ready"
    assert entry["cold_ms"] >= 0

    found = _scratchpad(sm, "df")
    assert found["cell"] == cell
    assert found["source"] == "df = list(range(3))\ndf"
    assert found["value"] == [0, 1, 2]
    assert _scratchpad(sm, "nope") == {"variable": "nope", "defined": False, "available": ["df"]}


@pytest.fixture
async def read_a_file(tmp_path):
    """A scratchpad whose one cell read a file: the manager, its session, the cell, the file."""
    data = tmp_path / "data.txt"
    data.write_text("2")
    sm = SessionManager()
    session = sm.open_notebook(create_notebook(tmp_path, "scratch", scratchpad=True))
    out = await _scratch_run(sm, f"x = int(open({str(data)!r}).read())")
    assert out["status"] == "ok", out
    return sm, session, out["cell"], data


@pytest.mark.asyncio
async def test_the_inventory_sees_a_file_a_cell_read_change_under_the_session(read_a_file):
    """Nothing watches it, so the tool looks again: stale, not the last pass's ready."""
    sm, session, cell, data = read_a_file
    # Just after a run nothing moved, so no pass is made.
    assert not session.code_keys_moved()
    await mcp_server._observe_code_reads(sm)
    assert _scratchpad(sm)["cells"][0]["status"] == "ready"

    data.write_text("3")
    # A check keeps no observation.
    assert session.code_keys_moved() and session.code_keys_moved()
    await mcp_server._observe_code_reads(sm)

    (entry,) = _scratchpad(sm)["cells"]
    assert (entry["cell"], entry["status"], entry["changed_code"]) == (cell, "stale", str(data))
    assert not session.code_keys_moved()


@pytest.mark.asyncio
async def test_while_a_run_is_active_the_inventory_checks_again_later(read_a_file, monkeypatch):
    sm, _, _, data = read_a_file
    data.write_text("3")
    running = [True]
    monkeypatch.setattr(ws, "notebook_has_active_execution", lambda notebook_id: running[0])

    await mcp_server._observe_code_reads(sm)

    assert _scratchpad(sm)["cells"][0]["status"] == "ready"
    running[0] = False
    await mcp_server._observe_code_reads(sm)
    assert _scratchpad(sm)["cells"][0]["status"] == "stale"


def _listed(app) -> dict[str, dict | None]:
    return {tool.name: tool.meta for tool in asyncio.run(app.state.mcp_server.list_tools())}


def test_the_scratchpad_profile_lists_two_always_loaded_tools_with_instructions():
    app = build_mcp_app(SessionManager(), profile="scratchpad")
    assert _listed(app) == {name: {"anthropic/alwaysLoad": True} for name in SCRATCHPAD_TOOLS}
    assert app.state.mcp_server.instructions == section("instructions")
    assert SCRATCHPAD_TOOLS <= CLASSIFIED_TOOLS
    assert required_scopes_for_tool("scratchpad") == ("notebook:read",)


@pytest.mark.parametrize("tool", ["run", "run_snippet"])
def test_a_tool_that_writes_a_cell_and_runs_it_needs_both_scopes(tool, monkeypatch):
    monkeypatch.setattr(mcp_server, "_server_config", lambda: _config(principal_auth_enabled=True))
    runner = Principal(id="runner", scopes=frozenset({"notebook:execute"}))
    monkeypatch.setattr(mcp_server, "_caller", lambda context: runner)
    app = build_mcp_app(SessionManager(), profile="scratchpad")

    with pytest.raises(ToolError, match=f"'{tool}' requires the notebook:write scope"):
        asyncio.run(app.state.mcp_server.call_tool(tool, {"source": "x = 1"}))
    assert required_scopes_for_tool(tool) == ("notebook:write", "notebook:execute")


def test_every_staleness_reason_has_words_for_the_agent():
    assert STALE_IN_AGENT_TERMS.keys() == set(StalenessReason)


def test_the_link_uses_the_public_url_behind_a_proxy(monkeypatch):
    monkeypatch.setattr(
        mcp_server,
        "_server_config",
        lambda: _config(public_base_url="https://lab.example.com/", public_base_path="/o/acme"),
    )
    assert mcp_server._notebook_url("s1") == "https://lab.example.com/o/acme/#/notebook/s1"


def test_unlisted_tools_stay_callable_and_all_lists_them():
    app = build_mcp_app(SessionManager(), profile="scratchpad")
    content = asyncio.run(app.state.mcp_server.call_tool("list_notebooks", {}))
    assert content is not None
    every = _listed(build_mcp_app(SessionManager(), profile="scratchpad", all_tools=True))
    assert SCRATCHPAD_TOOLS < every.keys() and "run_cell" in every


def test_the_notebook_profile_is_unchanged():
    app = build_mcp_app(SessionManager())
    assert app.state.mcp_server.instructions is None
    assert not SCRATCHPAD_TOOLS & _listed(app).keys()


def test_every_tool_the_guidance_names_exists():
    tools = _listed(build_mcp_app(SessionManager(), profile="scratchpad", all_tools=True))
    called = set(re.findall(r"`(\w+)\(", section("notebook_claude_md")))
    assert called and called <= tools.keys()
    named = set(re.findall(r"`(\w+)`", section("instructions")))
    assert SCRATCHPAD_TOOLS <= named


def test_the_instructions_fit_what_claude_code_keeps():
    # Claude Code cuts server instructions at 4,096 characters.
    assert len(section("instructions")) < 4096


@pytest.mark.asyncio
async def test_an_undeclared_read_comes_back_with_its_fix(scratch, tmp_path):
    sm, _ = scratch
    data = tmp_path / "rows.txt"
    data.write_text("1\n2\n")
    out = await _scratch_run(sm, f"n = len(open({str(data)!r}).read().split())\nn")
    assert out["status"] == "ok" and out["value"] == 2
    (diagnostic,) = out["diagnostics"]
    assert diagnostic["code"] == "undeclared_read"
    assert "rows.txt" in diagnostic["message"]


@pytest.mark.asyncio
async def test_a_cell_that_is_not_python_is_refused(scratch):
    sm, session_id = scratch
    cell = (await _add_cell(sm, session_id, "SELECT 1", language="sql"))["id"]

    out = await _scratch_run(sm, "x = 1", cell=cell)
    assert out["error"] == f"cell {cell!r} is a sql cell, and `run` writes Python"
    assert sm.get_session(session_id).notebook_state.get_cell(cell).source == "SELECT 1"


@pytest.mark.asyncio
async def test_a_busy_notebook_leaves_the_cells_as_they_were(scratch, monkeypatch):
    sm, session_id = scratch
    kept = (await _scratch_run(sm, "x = 1"))["cell"]

    async def busy(execution_state, requested_cell):
        return "other"

    monkeypatch.setattr(ws, "_reserve_execution_request", busy)

    def cells():
        # A reload replaces the state, so read it fresh.
        return [(c.id, c.source) for c in sm.get_session(session_id).notebook_state.cells]

    added = await _scratch_run(sm, "y = 2")
    assert added["fix"] == "call `run` again once that run finishes"
    assert cells() == [(kept, "x = 1")]

    edited = await _scratch_run(sm, "x = 3", cell=kept)
    assert edited["fix"] == f"call `run` again with cell={kept!r} once that run finishes"
    # Not rewritten under the run that holds the notebook.
    assert cells() == [(kept, "x = 1")]


@pytest.mark.asyncio
async def test_a_failed_upstream_is_named_as_failed(scratch):
    sm, _ = scratch
    upstream = await _scratch_run(sm, "x = 1 / 0")
    assert upstream["status"] == "error"

    out = await _scratch_run(sm, "y = x + 1")
    assert out["reran"] == [{"cell": upstream["cell"], "reason": "it had failed"}]


def test_scratchpad_shows_storage_its_sweep_could_not_free(scratch):
    from dataclasses import asdict

    from strata.notebook.scratch_storage import SAFE_TO_DELETE, Storage, storage_path

    sm, session_id = scratch
    path = storage_path(sm.get_session(session_id).path)
    held = [{"cell": "a", "bytes": 300}]
    under = Storage(store_bytes=100, soft_limit_bytes=200, over_by_bytes=0, swept_at=0.0)
    path.write_text(json.dumps(asdict(under)))
    assert "storage" not in _scratchpad(sm)

    over = Storage(
        store_bytes=300, soft_limit_bytes=200, over_by_bytes=100, swept_at=0.0, largest=held
    )
    path.write_text(json.dumps(asdict(over)))
    storage = _scratchpad(sm)["storage"]
    assert (storage["over_by_bytes"], storage["largest"]) == (100, held)
    assert "strata scratch ls" in storage["fix"] and SAFE_TO_DELETE in storage["fix"]


@pytest.mark.asyncio
async def test_a_run_needing_a_retired_cell_gets_its_source_in_the_fix(scratch):
    from strata.notebook.ops import LocalNotebookOps
    from strata.notebook.scratch_storage import retire_cell, scratch_manager

    sm, session_id = scratch
    session = sm.get_session(session_id)
    gone = (await _scratch_run(sm, "x = 41"))["cell"]
    ops = LocalNotebookOps.from_session(session)
    retire_cell(ops, session.path, scratch_manager(session.path), ops.get_cell(gone))

    failed = await _scratch_run(sm, "x + 1")
    assert failed["status"] == "error"
    assert gone in failed["fix"] and "x = 41" in failed["fix"]


@pytest.mark.asyncio
async def test_a_retired_cell_or_its_variable_asked_for_by_name_gets_its_source(scratch):
    from strata.notebook.ops import LocalNotebookOps
    from strata.notebook.scratch_storage import retire_cell, scratch_manager

    sm, session_id = scratch
    session = sm.get_session(session_id)
    gone = (await _scratch_run(sm, "x = 41"))["cell"]
    ops = LocalNotebookOps.from_session(session)
    retire_cell(ops, session.path, scratch_manager(session.path), ops.get_cell(gone))

    refused = await _scratch_run(sm, "x = 42", cell=gone)
    assert refused["status"] == "error"
    assert "x = 41" in refused["fix"]
    looked = mcp_server._scratchpad(sm, name="x")
    assert looked["defined"] is False
    assert gone in looked["fix"] and "x = 41" in looked["fix"]


@pytest.mark.asyncio
async def test_a_failure_on_something_else_keeps_its_own_fix(scratch):
    from strata.notebook.ops import LocalNotebookOps
    from strata.notebook.scratch_storage import retire_cell, scratch_manager

    sm, session_id = scratch
    session = sm.get_session(session_id)
    gone = (await _scratch_run(sm, "x = 41"))["cell"]
    ops = LocalNotebookOps.from_session(session)
    retire_cell(ops, session.path, scratch_manager(session.path), ops.get_cell(gone))

    failed = await _scratch_run(sm, "1 / 0\nx")
    assert failed["status"] == "error"
    assert "x = 41" not in failed.get("fix", "")
