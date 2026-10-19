"""Asking before a large read (Amber item 90)."""

from __future__ import annotations

import json
import sys

import pytest

from strata.config import StrataConfig
from strata.notebook import reads_index
from strata.notebook.executor import CellExecutor
from strata.notebook.models import CellStatus, FetchSpec, InputWeight
from strata.notebook.read_confirm import confirmation, estimate, over_threshold
from strata.notebook.ws import (
    _ensure_execution_state,
    _handle_cell_execute,
    _handle_read_confirm,
    _notebook_connections,
    forget_notebook_execution_state,
)
from tests.notebook.test_fetch import _cache, _Origin
from tests.notebook.test_table_index import _session, _warehouse

windows = pytest.mark.skipif(sys.platform == "win32", reason="pyiceberg's local paths fail")


def _weight(uri: str, size: int | None, cached: int | None = None, credential=None):
    return InputWeight(
        kind="mount", name="data", uri=uri, credential=credential, bytes=size, cached_bytes=cached
    )


class TestTheEstimate:
    def test_what_the_machine_holds_moves_nothing_and_an_unknown_size_counts_nothing(self):
        read = estimate(
            "c1",
            "local",
            [_weight("s3://b/a", 10, cached=4), _weight("s3://b/c", None), _weight("s3://b/d", 3)],
            "fp",
            StrataConfig(),
        )

        assert read.bytes_to_move == 9

    def test_each_credential_is_rated_and_a_bare_input_by_its_schemes_default(self):
        config = StrataConfig(
            notebook_credential_read_cost={"lab": 0.09, "org": 0.02},
            notebook_mount_credentials={"s3": "org"},
        )
        weights = [
            _weight("gs://b/a", 2 * 10**9, credential="lab"),
            _weight("s3://b/c", 10**9),
            _weight("az://b/d", 10**9),
        ]

        assert estimate("c1", "local", weights, "fp", config).cost == pytest.approx(0.2)

    @pytest.mark.parametrize(
        ("settings", "asks"),
        [
            ({}, False),
            ({"notebook_confirm_read_bytes": 100}, False),
            ({"notebook_confirm_read_bytes": 99}, True),
            ({"notebook_confirm_read_cost": 1.0}, False),
            ({"notebook_confirm_read_cost": 0.5}, True),
        ],
    )
    def test_either_threshold_asks(self, settings, asks):
        config = StrataConfig(notebook_credential_read_cost={"lab": 10**7}, **settings)
        read = estimate("c1", "local", [_weight("s3://b", 100, credential="lab")], "fp", config)

        assert over_threshold(read, config) is asks

    def test_the_index_says_why_a_run_went_ahead(self):
        config = StrataConfig(notebook_confirm_read_bytes=0)
        big = estimate("c1", "local", [_weight("s3://b", 1)], "fp", config)
        small = estimate("c1", "local", [_weight("s3://b", 0)], "fp", config)

        assert confirmation(big, StrataConfig(), approved=False) is None
        assert confirmation(small, config, approved=False) == "under_threshold"
        assert confirmation(big, config, approved=True) == "approved"
        assert confirmation(big, config, approved=False) == "not_asked"


class TestAFetchIsAskedNotRead:
    @pytest.fixture
    def origin(self):
        server = _Origin()
        yield server
        server.close()

    def test_a_head_says_its_size_and_a_304_that_it_is_here(self, tmp_path, origin):
        cache = _cache(tmp_path)
        spec = FetchSpec(name="zones", url=origin.url())
        body = len(origin.body)

        first = cache.probe(spec)
        cache.resolve(spec, max_age=0)
        again = cache.probe(spec)
        origin.body += b"2,Bronx\n"
        changed = cache.probe(spec)

        assert (first, again, changed) == ((body, 0), (body, body), (len(origin.body), 0))
        assert origin.methods == ["HEAD", "GET", "HEAD", "HEAD"]

    def test_bytes_it_will_not_check_are_not_asked_for(self, tmp_path, origin):
        cache = _cache(tmp_path)
        spec = FetchSpec(name="zones", url=origin.url(), refetch="never")
        cache.resolve(spec)

        assert cache.probe(spec) == (len(origin.body),) * 2
        assert origin.methods == ["GET"]

    def test_an_unreachable_url_is_not_guessed(self, tmp_path):
        spec = FetchSpec(name="zones", url="http://127.0.0.1:9/zones.csv")

        assert _cache(tmp_path).probe(spec) == (None, None)


class _Socket:
    def __init__(self):
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))

    def frames(self, kind: str) -> list[dict]:
        return [frame["payload"] for frame in self.sent if frame["type"] == kind]


@pytest.fixture
def threshold(tmp_path, monkeypatch) -> StrataConfig:
    config = StrataConfig(
        cache_dir=tmp_path / "cache",
        notebook_confirm_read_bytes=0,
        notebook_confirm_read_message="Ask the lab first.",
    )
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
    return config


async def _execute(session, socket, cell_id: str) -> None:
    state = _ensure_execution_state(session.id)
    await _handle_cell_execute(socket, session, {"cell_id": cell_id}, state, session.id)
    if state.execution_task is not None:
        await state.execution_task


@windows
async def test_a_cell_over_the_threshold_asks_once_and_runs_after_run(tmp_path, threshold):
    """Amber item 90's done-when, in the notebook."""
    _, uri = _warehouse(tmp_path)
    session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    try:
        await _execute(session, socket, "c1")
        (prompt,) = socket.frames("read_confirm_prompt")
        held_back = session.notebook_state.get_cell("c1").status.value
        await _handle_read_confirm(
            socket,
            session,
            {"prompt_id": prompt["prompt_id"]},
            _ensure_execution_state(session.id),
            session.id,
        )
        state = _ensure_execution_state(session.id)
        if state.execution_task is not None:
            await state.execution_task
        ran = session.notebook_state.get_cell("c1").status.value
        confirmed = reads_index.load(session.path)["c1"]["confirmation"]

        socket.sent.clear()
        session.notebook_state.get_cell("c1").source += "# again\n"
        session.re_analyze_cell("c1")
        await _execute(session, socket, "c1")
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)

    (cell,) = prompt["cells"]
    assert cell["cell_id"] == "c1"
    assert cell["machine"] == "local"
    assert cell["bytes_to_move"] == cell["inputs"][0]["bytes"] > 0
    assert prompt["message"] == "Ask the lab first."
    assert held_back != "ready"
    assert (ran, confirmed) == ("ready", "approved")
    # The same inputs on the same machine: no question, and the run happened.
    assert socket.frames("read_confirm_prompt") == []
    assert session.notebook_state.get_cell("c1").status.value == "ready"


@windows
async def test_an_answer_to_no_question_runs_nothing(tmp_path, threshold):
    _, uri = _warehouse(tmp_path)
    session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    try:
        state = _ensure_execution_state(session.id)
        await _handle_read_confirm(socket, session, {"prompt_id": "nope"}, state, session.id)
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)

    assert [e["error"] for e in socket.frames("error")] == [
        "Read confirmation not found or expired"
    ]
    assert session.notebook_state.get_cell("c1").status.value != "ready"


@windows
@pytest.mark.parametrize(("settings", "recorded"), [({}, None), (None, "not_asked")])
async def test_a_run_that_cannot_ask_goes_ahead_and_says_so(
    tmp_path, monkeypatch, settings, recorded
):
    """A headless run (the executor alone) is never held back."""
    config = StrataConfig(
        cache_dir=tmp_path / "cache",
        **({"notebook_confirm_read_bytes": 0} if settings is None else settings),
    )
    monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
    _, uri = _warehouse(tmp_path)
    session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})

    result = await CellExecutor(session).execute_cell("c1", session.notebook_state.cells[0].source)

    assert result.success, result.error
    assert reads_index.load(session.path)["c1"].get("confirmation") == recorded


@windows
async def test_an_api_run_asked_to_confirm_answers_409_until_approved(tmp_path, threshold):
    from fastapi import HTTPException

    from strata.notebook.routes import execute_cell

    _, uri = _warehouse(tmp_path)
    session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})
    try:
        with pytest.raises(HTTPException) as refused:
            await execute_cell(session.id, session, "c1", confirm_reads=True)
        approved = await execute_cell(session.id, session, "c1", approve_reads=True)
        again = await execute_cell(session.id, session, "c1", mode="rerun", confirm_reads=True)
    finally:
        forget_notebook_execution_state(session.id)

    detail = refused.value.detail
    assert refused.value.status_code == 409
    assert isinstance(detail, dict)
    assert (detail["code"], detail["message"]) == ("large_read", "Ask the lab first.")
    assert [cell["cell_id"] for cell in detail["cells"]] == ["c1"]
    assert approved["status"] == "ready"
    # The yes is kept: the same read is not asked about again.
    assert again["status"] == "ready"
    assert reads_index.load(session.path)["c1"]["confirmation"] == "approved"


@windows
async def test_an_agent_asked_to_confirm_gets_the_estimate_until_it_approves(
    tmp_path, threshold, monkeypatch
):
    from strata.notebook.mcp_server import _run_cell
    from strata.notebook.session import SessionManager

    async def no_note(*args, **kwargs):
        return None

    monkeypatch.setattr("strata.notebook.mcp_server._agent_note", no_note)
    _, uri = _warehouse(tmp_path)
    session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})
    sm = SessionManager()
    sm._sessions[session.id] = session
    try:
        held = await _run_cell(sm, session.id, "c1", confirm_reads=True)
        ran = await _run_cell(sm, session.id, "c1", approve_reads=True)
    finally:
        forget_notebook_execution_state(session.id)

    assert (held["status"], held["error_code"]) == ("error", "large_read")
    assert held["error"] == "Ask the lab first."
    assert [read["cell_id"] for read in held["reads"]] == ["c1"]
    assert session.notebook_state.get_cell("c1").status.value == "ready"
    assert ran["status"] == "ok"


@windows
async def test_a_ready_cell_reads_nothing_so_it_is_not_asked_about(tmp_path, threshold):
    _, uri = _warehouse(tmp_path)
    session = _session(tmp_path, {"c1": f"# @table events {uri}\nsnap = events_snapshot\n"})
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    try:
        # Run headless, unasked; the notebook's Run is then a cache hit.
        await CellExecutor(session).execute_cell("c1", session.notebook_state.cells[0].source)
        session.notebook_state.get_cell("c1").status = CellStatus.READY
        await _execute(session, socket, "c1")
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)

    assert socket.frames("read_confirm_prompt") == []
    assert reads_index.load(session.path)["c1"]["confirmation"] == "not_asked"


@windows
async def test_run_all_asks_once_for_the_cells_over_the_threshold(tmp_path, threshold):
    from strata.notebook.ws import _handle_notebook_run_all

    _, uri = _warehouse(tmp_path)
    session = _session(
        tmp_path,
        {"c1": "x = 1\n", "c2": f"# @table events {uri}\nsnap = (events_snapshot, x)\n"},
    )
    session._analyze_and_build_dag()
    session.environment_sync_state = "ready"
    socket = _Socket()
    _notebook_connections[session.id] = [socket]
    try:
        state = _ensure_execution_state(session.id)
        await _handle_notebook_run_all(socket, session, state, session.id, {})
        (prompt,) = socket.frames("read_confirm_prompt")
        nothing_ran = [c.status.value for c in session.notebook_state.cells]
        await _handle_read_confirm(
            socket, session, {"prompt_id": prompt["prompt_id"]}, state, session.id
        )
        if state.execution_task is not None:
            await state.execution_task
    finally:
        _notebook_connections.pop(session.id, None)
        forget_notebook_execution_state(session.id)

    assert [cell["cell_id"] for cell in prompt["cells"]] == ["c2"]
    assert "ready" not in nothing_ran
    assert [c.status.value for c in session.notebook_state.cells] == ["ready", "ready"]
