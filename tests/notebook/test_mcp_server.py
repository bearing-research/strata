"""Unit tests for the MCP server tool logic + mount gating (Phase 1, read tools).

The tools' logic lives in module-level ``_*`` functions that take a
``SessionManager``, so these tests exercise them directly — no MCP client, no
live socket (avoids the TestClient-WS portal hang, and keeps them fast). A
notebook session is built in-process and registered without a venv sync, since
the read tools never execute a cell.
"""

from __future__ import annotations

import pytest

from strata.notebook.mcp_server import (
    _add_cell,
    _add_dependency,
    _add_worker,
    _connect_ssh_worker,
    _dag,
    _disconnect_ssh_worker,
    _edit_cell,
    _get_cell,
    _get_notebook,
    _get_variable,
    _list_notebooks,
    _list_workers,
    _move_cell,
    _note,
    _remove_cell,
    _remove_dependency,
    _remove_worker,
    _run_cell,
    _run_snippet,
    _run_tests,
    _set_default_worker,
    _status,
    build_mcp_app,
)
from strata.notebook.ops import LocalNotebookOps, NotebookOpsError
from tests.notebook.test_cli import _build_notebook


@pytest.fixture
def sm_with_session(tmp_path):
    """A SessionManager holding one live session for a two-cell chain a→b."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession, SessionManager

    nb_dir = _build_notebook(tmp_path, cells=[("a", "x = 1", None), ("b", "y = x + 1", "a")])
    sm = SessionManager()
    session = NotebookSession(parse_notebook(nb_dir), nb_dir)
    # Register directly: read tools don't run cells, so skip the venv sync that
    # open_notebook would do.
    sm._sessions[session.id] = session
    return sm, session.id, nb_dir


def test_list_notebooks_reports_open_sessions(sm_with_session):
    sm, session_id, nb_dir = sm_with_session
    notebooks = _list_notebooks(sm)
    assert len(notebooks) == 1
    entry = notebooks[0]
    assert entry["session_id"] == session_id
    assert entry["path"] == str(nb_dir)
    assert entry["name"]


def test_get_notebook_returns_cells_in_order(sm_with_session):
    sm, session_id, _ = sm_with_session
    result = _get_notebook(sm, session_id)
    assert [c["id"] for c in result["cells"]] == ["a", "b"]
    assert result["cells"][0]["source"] == "x = 1"
    # Curated view — internal bookkeeping doesn't leak to the agent.
    assert "last_provenance_hash" not in result["cells"][0]


def test_get_cell_and_unknown_cell(sm_with_session):
    sm, session_id, _ = sm_with_session
    cell = _get_cell(sm, session_id, "a")
    assert cell["id"] == "a"
    assert cell["source"] == "x = 1"
    with pytest.raises(NotebookOpsError):
        _get_cell(sm, session_id, "ghost")


def test_dag_exposes_the_edge(sm_with_session):
    sm, session_id, _ = sm_with_session
    dag = _dag(sm, session_id)
    assert any(
        e["from_cell_id"] == "a" and e["to_cell_id"] == "b" and e["variable"] == "x"
        for e in dag["edges"]
    )
    assert dag["topological_order"].index("a") < dag["topological_order"].index("b")


def test_status_summary(sm_with_session):
    sm, session_id, _ = sm_with_session
    status = _status(sm, session_id)
    assert status["name"]
    assert {row["id"] for row in status["cells"]} == {"a", "b"}


def test_unknown_session_raises_valueerror(sm_with_session):
    sm, _, _ = sm_with_session
    # A missing session is a client error, surfaced to the agent as a tool error.
    with pytest.raises(ValueError, match="no open notebook session"):
        _get_notebook(sm, "nope")


def test_from_session_reuses_the_live_session(sm_with_session):
    sm, session_id, _ = sm_with_session
    live = sm.get_session(session_id)
    ops = LocalNotebookOps.from_session(live)
    # Same underlying session object — the warm state, not an offline reopen.
    assert ops._session is live
    assert ops.notebook_dir == live.path


@pytest.mark.asyncio
async def test_run_cell_broadcasts_and_maps(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session
    seen = {}

    async def fake_broadcast(session, cell_id, execution_state, notebook_id, mode="normal"):
        seen["args"] = (cell_id, notebook_id, mode)

        class _Result:
            def to_dict(self):
                return {
                    "cell_id": cell_id,
                    "status": "ready",
                    "cache_hit": False,
                    "execution_method": "subprocess",
                    "duration_ms": 12.0,
                    "stdout": "hi\n",
                    "stderr": "",
                    "error": None,
                }

        return _Result()

    # Patch the shared broadcast path — _run_cell imports it at call time, so
    # patching the source module is enough. No subprocess, no real WS.
    monkeypatch.setattr("strata.notebook.ws.execute_cell_and_broadcast", fake_broadcast)
    # A directly-built test session has no synced venv, so the env-ready guard
    # would refuse; a UI/CLI-opened session in production is ready. Simulate that.
    monkeypatch.setattr(
        sm.get_session(session_id), "environment_execution_block_message", lambda: None
    )
    # Capture the agent_note the run narrates into the Agent panel (#393).
    notes = []

    async def fake_note_broadcast(notebook_id, message):
        notes.append(message)

    monkeypatch.setattr("strata.notebook.ws._broadcast_message", fake_note_broadcast)

    result = await _run_cell(sm, session_id, "a", mode="rerun")
    assert result["cell_id"] == "a"
    assert result["status"] == "ok"  # "ready" → "ok"
    assert result["stdout"] == "hi\n"
    # The live session id is threaded through as the broadcast notebook_id.
    assert seen["args"] == ("a", session_id, "rerun")
    # The run auto-narrates an mcp-sourced agent_note.
    assert any(
        m["type"] == "agent_note"
        and m["payload"]["source"] == "mcp"
        and "ran cell a" in m["payload"]["text"]
        for m in notes
    )


@pytest.mark.asyncio
async def test_run_cell_caps_console_returned_to_the_agent(sm_with_session, monkeypatch):
    """A cell's stdout is captured whole and travels uncapped to here, so a
    print-heavy cell would return megabytes straight into an agent's context.
    The agent-facing result caps each stream and says how much it dropped."""
    sm, session_id, _ = sm_with_session
    flood = "x" * 50_000

    async def fake_broadcast(session, cell_id, execution_state, notebook_id, mode="normal"):
        class _Result:
            def to_dict(self):
                return {
                    "cell_id": cell_id,
                    "status": "ready",
                    "cache_hit": False,
                    "execution_method": "subprocess",
                    "duration_ms": 1.0,
                    "stdout": flood,
                    "stderr": flood,
                    "error": None,
                }

        return _Result()

    monkeypatch.setattr("strata.notebook.ws.execute_cell_and_broadcast", fake_broadcast)
    monkeypatch.setattr(
        sm.get_session(session_id), "environment_execution_block_message", lambda: None
    )

    async def fake_note_broadcast(notebook_id, message):
        return None

    monkeypatch.setattr("strata.notebook.ws._broadcast_message", fake_note_broadcast)

    result = await _run_cell(sm, session_id, "a")
    for stream in ("stdout", "stderr"):
        assert len(result[stream]) < len(flood)
        assert result[stream].startswith("x" * 100)
        assert "chars truncated" in result[stream]


def test_get_variable_defined_and_undefined(sm_with_session):
    sm, session_id, _ = sm_with_session
    # sm_with_session: cell `a` defines x, cell `b` defines y (= x + 1).
    hit = _get_variable(sm, session_id, "x")
    assert hit["defined"] is True and hit["defined_in"] == "a"
    assert hit["cell"]["source"] == "x = 1"
    miss = _get_variable(sm, session_id, "nope")
    assert miss["defined"] is False and miss["available"] == ["x", "y"]


@pytest.mark.asyncio
async def test_run_snippet_adds_then_runs_in_one_call(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session

    async def fake_broadcast(session, cell_id, execution_state, notebook_id, mode="normal"):
        class _Result:
            def to_dict(self):
                return {
                    "cell_id": cell_id,
                    "status": "ready",
                    "cache_hit": False,
                    "execution_method": "subprocess",
                    "duration_ms": 5.0,
                    "stdout": "snippet ran\n",
                    "stderr": "",
                    "error": None,
                }

        return _Result()

    async def fake_sync(notebook_id, session):
        pass

    monkeypatch.setattr("strata.notebook.ws.execute_cell_and_broadcast", fake_broadcast)
    monkeypatch.setattr("strata.notebook.ws.broadcast_notebook_sync", fake_sync)
    monkeypatch.setattr(
        sm.get_session(session_id), "environment_execution_block_message", lambda: None
    )

    view = await _run_snippet(sm, session_id, "print('snippet ran')")
    # One call returns the new cell view AND its run outcome nested under `run`.
    assert view["source"] == "print('snippet ran')"
    assert view["run"]["status"] == "ok"
    assert view["run"]["stdout"] == "snippet ran\n"
    # The cell really landed in the notebook (add half of add-and-run).
    assert _get_cell(sm, session_id, view["id"])["source"] == "print('snippet ran')"


@pytest.mark.asyncio
async def test_run_cell_rejects_bad_mode_missing_cell_and_session(sm_with_session):
    sm, session_id, _ = sm_with_session
    with pytest.raises(ValueError, match="unknown run mode"):
        await _run_cell(sm, session_id, "a", mode="bogus")
    with pytest.raises(NotebookOpsError):
        await _run_cell(sm, session_id, "ghost")
    with pytest.raises(ValueError, match="no open notebook session"):
        await _run_cell(sm, "nope", "a")


@pytest.mark.asyncio
async def test_run_cell_refuses_when_env_not_ready(sm_with_session):
    sm, session_id, _ = sm_with_session
    # A freshly-built session has no synced venv → run_cell refuses with a clear
    # message rather than running into a broken environment.
    with pytest.raises(ValueError, match="environment"):
        await _run_cell(sm, session_id, "a")


@pytest.mark.asyncio
async def test_run_tests_maps_outcomes(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session
    from strata.notebook.models import CellTestCase, CellTestResult

    async def fake_run_cell_tests(self, cell_id, test_source):
        return CellTestResult(
            passed=1,
            failed=1,
            tests=[
                CellTestCase(name="t_ok", outcome="passed"),
                CellTestCase(name="t_bad", outcome="failed", message="assert 1 == 2"),
            ],
        )

    monkeypatch.setattr("strata.notebook.executor.CellExecutor.run_cell_tests", fake_run_cell_tests)
    # run_tests refuses a cell with no test file.
    with pytest.raises(NotebookOpsError):
        await _run_tests(sm, session_id, "a")
    # Give cell 'a' a test source, then it maps the executor's result.
    sm.get_session(session_id).notebook_state.get_cell("a").test_source = "def test_x(cell): pass"
    result = await _run_tests(sm, session_id, "a")
    assert result["passed"] == 1 and result["failed"] == 1
    assert [c["name"] for c in result["cases"]] == ["t_ok", "t_bad"]


@pytest.mark.asyncio
async def test_authoring_add_edit_move_remove_and_broadcast(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session
    broadcasts = []

    async def fake_sync(notebook_id, session):
        broadcasts.append(notebook_id)

    # Each mutation should push a live state sync to the session's spectators.
    monkeypatch.setattr("strata.notebook.ws.broadcast_notebook_sync", fake_sync)

    added = await _add_cell(sm, session_id, "z = 9", after="a", language="python")
    assert added["source"] == "z = 9"
    new_id = added["id"]
    assert _get_cell(sm, session_id, new_id)["source"] == "z = 9"

    edited = await _edit_cell(sm, session_id, new_id, "z = 10")
    assert edited["source"] == "z = 10"
    assert _get_cell(sm, session_id, new_id)["source"] == "z = 10"

    order = await _move_cell(sm, session_id, new_id, 0)
    assert order["cells"][0]["id"] == new_id

    removed = await _remove_cell(sm, session_id, new_id)
    assert removed == {"removed": new_id}
    with pytest.raises(NotebookOpsError):
        _get_cell(sm, session_id, new_id)

    # add, edit, move, remove → four live syncs, all for this session.
    assert broadcasts == [session_id] * 4


@pytest.mark.asyncio
async def test_authoring_errors(sm_with_session):
    sm, session_id, _ = sm_with_session
    with pytest.raises(ValueError, match="no open notebook session"):
        await _add_cell(sm, "nope", "x = 1")
    with pytest.raises(NotebookOpsError):
        await _edit_cell(sm, session_id, "ghost", "x = 1")
    with pytest.raises(NotebookOpsError):
        await _remove_cell(sm, session_id, "ghost")


@pytest.mark.asyncio
async def test_add_and_remove_dependency_and_broadcast(sm_with_session, monkeypatch):
    from types import SimpleNamespace

    sm, session_id, _ = sm_with_session
    broadcasts = []

    async def fake_sync(notebook_id, session):
        broadcasts.append(notebook_id)

    monkeypatch.setattr("strata.notebook.ws.broadcast_notebook_sync", fake_sync)

    async def fake_mutate(package, *, action):
        result = SimpleNamespace(
            package=package, action=action, success=True, lockfile_changed=True, error=None
        )
        return SimpleNamespace(result=result, staleness_map={})

    # No real `uv add` — stub the session's dependency mutation.
    monkeypatch.setattr(sm.get_session(session_id), "mutate_dependency", fake_mutate)

    added = await _add_dependency(sm, session_id, "polars")
    assert added["package"] == "polars"
    assert added["action"] == "add"
    assert added["success"] is True and added["lockfile_changed"] is True

    removed = await _remove_dependency(sm, session_id, "polars")
    assert removed["action"] == "remove"

    assert broadcasts == [session_id, session_id]


@pytest.mark.asyncio
async def test_dependency_missing_session_raises(sm_with_session):
    sm, _, _ = sm_with_session
    with pytest.raises(ValueError, match="no open notebook session"):
        await _add_dependency(sm, "nope", "polars")


@pytest.mark.asyncio
async def test_note_tool_broadcasts_agent_frame(sm_with_session, monkeypatch):
    sm, session_id, _ = sm_with_session
    frames = []

    async def fake_broadcast(notebook_id, message):
        frames.append((notebook_id, message))

    monkeypatch.setattr("strata.notebook.ws._broadcast_message", fake_broadcast)

    result = await _note(sm, session_id, "about to refactor featurize")
    assert result == {"ok": True}
    assert len(frames) == 1
    notebook_id, message = frames[0]
    assert notebook_id == session_id
    assert message["type"] == "agent_note"
    assert message["payload"] == {"source": "agent", "text": "about to refactor featurize"}


@pytest.mark.asyncio
async def test_note_missing_session_raises(sm_with_session):
    sm, _, _ = sm_with_session
    with pytest.raises(ValueError, match="no open notebook session"):
        await _note(sm, "nope", "hello")


@pytest.mark.asyncio
async def test_worker_tools_register_default_and_remove(sm_with_session, monkeypatch):
    sm, session_id, nb_dir = sm_with_session
    broadcasts = []

    async def fake_sync(notebook_id, session):
        broadcasts.append(notebook_id)

    monkeypatch.setattr("strata.notebook.ws.broadcast_notebook_sync", fake_sync)

    url = "http://127.0.0.1:9000/v1/execute"
    added = await _add_worker(
        sm, session_id, "gpu", url, token_env="STRATA_WORKER_TOKEN_GPU", set_default=True
    )
    assert added["default"] == "gpu"
    gpu = next(w for w in added["workers"] if w["name"] == "gpu")
    assert gpu["url"] == url
    assert gpu["is_default"] is True

    # The live session was reloaded, so a plain read tool sees the new worker.
    listed = _list_workers(sm, session_id)
    assert [w["name"] for w in listed["workers"]] == ["local", "gpu"]

    cleared = await _set_default_worker(sm, session_id, "local")
    assert cleared["default"] is None

    removed = await _remove_worker(sm, session_id, "gpu")
    assert [w["name"] for w in removed["workers"]] == ["local"]

    # add, set-default, remove → three live syncs for this session.
    assert broadcasts == [session_id] * 3


@pytest.mark.asyncio
async def test_worker_tools_missing_session_raises(sm_with_session):
    sm, _, _ = sm_with_session
    with pytest.raises(ValueError, match="no open notebook session"):
        await _add_worker(sm, "nope", "gpu", "http://127.0.0.1:9000/v1/execute")


@pytest.mark.asyncio
async def test_connect_and_disconnect_ssh_worker(sm_with_session, monkeypatch):
    from tests.notebook.test_ssh_worker_service import FakeSupervisor

    sm, session_id, _ = sm_with_session
    fake = FakeSupervisor()
    monkeypatch.setattr("strata.notebook.routes.get_worker_supervisor", lambda: fake)
    broadcasts = []

    async def fake_sync(notebook_id, session):
        broadcasts.append(notebook_id)

    monkeypatch.setattr("strata.notebook.ws.broadcast_notebook_sync", fake_sync)

    result = await _connect_ssh_worker(sm, session_id, "user@gpu-box")
    assert result["worker"]["ssh_target"] == "user@gpu-box"
    assert result["worker"]["executor_url"] == "http://127.0.0.1:6000/v1/execute"
    assert fake.established == [("gpu-box", "user@gpu-box")]
    # Registered + made default; a plain read tool now sees it.
    assert result["default"] == "gpu-box"
    assert "gpu-box" in [w["name"] for w in _list_workers(sm, session_id)["workers"]]

    dis = await _disconnect_ssh_worker(sm, session_id, "gpu-box")
    assert dis["torn_down"] is True
    assert [w["name"] for w in dis["workers"]] == ["local"]
    assert broadcasts == [session_id, session_id]  # connect + disconnect each synced


def test_build_mcp_app_returns_mountable_app(sm_with_session):
    sm, _, _ = sm_with_session
    mcp_app = build_mcp_app(sm)
    # [mcp] extra is installed in the dev/CI env (--all-extras), so we get an app.
    assert mcp_app is not None
    # Mountable ASGI app with a lifespan the host can enter.
    assert hasattr(mcp_app, "router")
    assert hasattr(mcp_app.router, "lifespan_context")


# ---------------------------------------------------------------------------
# Registry / publication tools (item 32)
# ---------------------------------------------------------------------------


@pytest.fixture
def sm_with_a_stored_output(sm_with_session):
    """The session above, with cell ``a``'s ``x`` actually stored.

    The tools resolve a variable to the artifact behind it, so a session whose
    cells have never run has nothing for them to find.
    """
    sm, session_id, nb_dir = sm_with_session
    manager = sm._sessions[session_id].get_artifact_manager()
    manager.store_cell_output(
        cell_id="a",
        variable_name="x",
        blob_data=b"1",
        content_type="json/object",
        provenance_hash="a1" * 32,
        input_versions={},
        source="x = 1",
    )
    return sm, session_id, nb_dir


class TestResolvingAnOutput:
    def test_an_unknown_variable_says_what_is_stored(self, sm_with_a_stored_output):
        """Only variables a downstream cell reads become artifacts, so the
        useful error names the ones that did rather than just saying no."""
        from strata.notebook.mcp_server import _cell_output

        sm, session_id, _ = sm_with_a_stored_output

        with pytest.raises(ValueError, match="Stored: x"):
            _cell_output(sm, session_id, "a", "not_a_variable")


class TestLineage:
    def test_it_reports_the_chain_with_the_code_each_step_ran(self, sm_with_a_stored_output):
        from strata.notebook.mcp_server import _lineage

        sm, session_id, _ = sm_with_a_stored_output

        result = _lineage(sm, session_id, "a", "x")

        assert result["provenance_hash"] == "a1" * 32
        assert result["content_sha256"]
        assert [s["source"] for s in result["steps"]] == ["x = 1"]


class TestPromote:
    def test_without_a_team_store_it_says_which_setting_is_missing(
        self, sm_with_a_stored_output, monkeypatch
    ):
        """An agent can act on "set this", and cannot act on a traceback."""
        from types import SimpleNamespace

        import strata.server as server_module
        from strata.notebook.mcp_server import _promote

        sm, session_id, _ = sm_with_a_stored_output
        monkeypatch.setattr(
            server_module,
            "_state",
            SimpleNamespace(config=SimpleNamespace(notebook_remote_store_url=None)),
        )

        with pytest.raises(ValueError, match="notebook_remote_store_url"):
            _promote(sm, session_id, "a", "x", "team/x")


class TestPublishPreflight:
    def test_it_lists_what_the_link_would_expose(self, sm_with_a_stored_output):
        """The chain travels with a publication, which is the point of one and
        the part worth reading before minting it."""
        from strata.notebook.mcp_server import _publish_preflight

        sm, session_id, _ = sm_with_a_stored_output

        result = _publish_preflight(sm, session_id, "a", "x")

        assert result["step_count"] == 1
        assert result["reads_source_of"] == [result["artifact_id"]]
        assert "no credentials" in result["note"]

    def test_it_mints_nothing(self, sm_with_a_stored_output):
        """A preflight that published would be the opposite of a preflight."""
        from strata.notebook.mcp_server import _publish_preflight

        sm, session_id, _ = sm_with_a_stored_output
        manager = sm._sessions[session_id].get_artifact_manager()

        _publish_preflight(sm, session_id, "a", "x")

        assert manager.artifact_store.list_publications() == []


class TestPromoteReaches:
    """Driven against a real store on the other end."""

    @pytest.fixture
    def team(self, tmp_path):
        from tests.conftest import run_server_with_context

        team_dir = tmp_path / "team"
        with run_server_with_context(tmp_path / "cache", team_dir, "personal") as ctx:
            yield ctx.base_url, team_dir

    def test_the_artifact_arrives_under_its_name(self, sm_with_a_stored_output, team, monkeypatch):
        from types import SimpleNamespace

        import httpx

        import strata.notebook.mcp_server as mcp_module
        from strata.artifact_store import ArtifactStore
        from strata.notebook.mcp_server import _promote

        sm, session_id, _ = sm_with_a_stored_output
        base_url, team_dir = team
        # The team store here is a real server in this process, so its own
        # ``_state`` has to stay intact — patch what this module reads, not
        # the state the server is running on.
        monkeypatch.setattr(
            mcp_module,
            "_server_config",
            lambda: SimpleNamespace(
                notebook_remote_store_url=base_url,
                notebook_remote_store_headers={},
            ),
        )

        result = _promote(sm, session_id, "a", "x", "team/x", tags={"stage": "candidate"})

        assert result["status"] == "applied"
        assert httpx.get(f"{base_url}/v1/names/team/x", timeout=10).status_code == 200
        landed = result["artifact_uri"].removeprefix("strata://artifact/")
        artifact_id, _, version = landed.partition("@v=")
        assert ArtifactStore(team_dir).get_tags(artifact_id, int(version))["stage"] == "candidate"


class TestPublish:
    def test_it_copies_the_chain_into_the_store_the_link_resolves_from(
        self, sm_with_a_stored_output, tmp_path, monkeypatch
    ):
        """A notebook writes to its own .strata/artifacts; the server serves
        whatever artifact_dir it was configured with. Minting into the
        notebook's store gives a link the page route never reads."""
        from types import SimpleNamespace

        import strata.server as server_module
        from strata.artifact_store import ArtifactStore
        from strata.notebook.mcp_server import _publish

        sm, session_id, _ = sm_with_a_stored_output
        served_dir = tmp_path / "served"
        monkeypatch.setattr(
            server_module,
            "_state",
            SimpleNamespace(config=SimpleNamespace(artifact_dir=served_dir)),
        )

        result = _publish(sm, session_id, "a", "x", title="Figure 1")

        served = ArtifactStore(served_dir)
        assert result["token"]
        assert result["copied"] == 1
        assert [p.token for p in served.list_publications()] == [result["token"]]
