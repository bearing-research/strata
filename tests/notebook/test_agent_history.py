"""Tests for agent conversation-history persistence and small helpers.

The agent loop itself needs a live LLM; these target the pure / disk-I/O
surface around it — history load/save/trim/reset, approval resolution,
variable→cell resolution, and arg formatting.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from strata.notebook.llm import agent
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.fixture(autouse=True)
def _isolated_module_state():
    agent.CONVERSATION_HISTORY.clear()
    agent._APPROVAL_FUTURES.clear()
    yield
    agent.CONVERSATION_HISTORY.clear()
    agent._APPROVAL_FUTURES.clear()


class TestHistoryDisk:
    def test_load_missing_returns_empty(self, tmp_path):
        assert agent._load_history_from_disk(tmp_path) == []

    def test_load_corrupt_returns_empty(self, tmp_path):
        path = agent.history_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("{ not valid json")
        assert agent._load_history_from_disk(tmp_path) == []

    def test_load_payload_without_turns_key(self, tmp_path):
        path = agent.history_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"something_else": 1}))
        assert agent._load_history_from_disk(tmp_path) == []

    def test_save_then_load_roundtrip(self, tmp_path):
        turns = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
        agent._save_history_to_disk(tmp_path, turns)
        assert agent.history_path(tmp_path).exists()
        assert agent._load_history_from_disk(tmp_path) == turns


class TestHistoryApi:
    def test_get_history_empty_for_unknown(self):
        assert agent.get_history("unknown-nb") == []

    def test_append_persists_and_reads_back(self, tmp_path):
        agent.append_history("nb1", [{"role": "user", "content": "a"}], tmp_path)
        assert agent.get_history("nb1") == [{"role": "user", "content": "a"}]
        # Also written through to disk.
        assert agent._load_history_from_disk(tmp_path) == [{"role": "user", "content": "a"}]

    def test_get_history_returns_a_copy(self, tmp_path):
        agent.append_history("nb2", [{"role": "user", "content": "a"}], tmp_path)
        snapshot = agent.get_history("nb2")
        snapshot.append({"role": "user", "content": "mutation"})
        assert len(agent.get_history("nb2")) == 1

    def test_append_trims_to_max_turns(self, tmp_path):
        turns = [{"role": "user", "content": str(i)} for i in range(agent.HISTORY_MAX_TURNS + 5)]
        agent.append_history("nb3", turns, tmp_path)
        history = agent.get_history("nb3")
        assert len(history) == agent.HISTORY_MAX_TURNS
        # The oldest turns are dropped; the newest survives.
        assert history[-1]["content"] == str(agent.HISTORY_MAX_TURNS + 4)
        assert history[0]["content"] == str(5)

    def test_history_seeds_from_disk_on_cache_miss(self, tmp_path):
        agent._save_history_to_disk(tmp_path, [{"role": "user", "content": "seed"}])
        # Fresh id not in CONVERSATION_HISTORY → load from disk.
        assert agent.get_history("nb-seed", tmp_path) == [{"role": "user", "content": "seed"}]

    def test_append_without_dir_stays_in_memory(self):
        agent.append_history("nb4", [{"role": "user", "content": "a"}])
        assert agent.get_history("nb4") == [{"role": "user", "content": "a"}]

    def test_reset_clears_memory_and_disk(self, tmp_path):
        agent.append_history("nb5", [{"role": "user", "content": "a"}], tmp_path)
        assert agent.history_path(tmp_path).exists()
        agent.reset_history("nb5", tmp_path)
        assert agent.get_history("nb5") == []
        assert not agent.history_path(tmp_path).exists()


class TestResolveApproval:
    async def test_resolves_pending_future(self):
        fut: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        agent._APPROVAL_FUTURES["req1"] = fut
        assert agent.resolve_approval("req1", True) is True
        assert fut.result() is True

    def test_unknown_request_returns_false(self):
        assert agent.resolve_approval("missing", True) is False

    async def test_already_resolved_future_returns_false(self):
        fut: asyncio.Future[bool] = asyncio.get_event_loop().create_future()
        fut.set_result(True)
        agent._APPROVAL_FUTURES["req2"] = fut
        assert agent.resolve_approval("req2", False) is False


class TestShortArgs:
    def test_empty(self):
        assert agent._short_args({}) == ""

    def test_joins_and_truncates_long_values(self):
        out = agent._short_args({"a": 1, "b": "x" * 100})
        assert "a=1" in out
        assert "b=" + "x" * 50 in out
        assert "x" * 51 not in out  # each value truncated to 50 chars


class TestResolveVariableToCellId:
    @pytest.fixture
    def session(self, tmp_path):
        notebook_dir = create_notebook(tmp_path, "agent_resolve")
        for cell_id, source in [("c1", "x = 1"), ("c2", "y = x + 1")]:
            add_cell_to_notebook(notebook_dir, cell_id)
            write_cell(notebook_dir, cell_id, source)
        return NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    def test_resolves_via_dag(self, session):
        assert agent.resolve_variable_to_cell_id(session, "x") == "c1"
        assert agent.resolve_variable_to_cell_id(session, "y") == "c2"

    def test_unknown_variable_is_none(self, session):
        assert agent.resolve_variable_to_cell_id(session, "missing") is None

    def test_falls_back_to_defines_scan_without_dag(self, session):
        session.dag = None  # force the linear defines-scan path
        assert agent.resolve_variable_to_cell_id(session, "x") == "c1"
        assert agent.resolve_variable_to_cell_id(session, "missing") is None
