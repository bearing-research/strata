"""Tests for the redesigned agent loop primitives.

Focused unit tests on the building blocks that don't need a live LLM:

* Conversation memory: append/get/reset and the trim window
* Approval resolution: futures clear cleanly and missing IDs are no-ops
* Approval callback: ``auto_approve=True`` skips the gate entirely
* Parallel-safe dispatch: read-only tools fan out, mutators run in order
* Streaming protocol: text deltas and tool-call accumulation

The full ``run_agent_loop`` function makes real HTTP calls to a provider,
so it's covered by integration tests, not here.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import patch

import pytest

from strata.notebook.llm.agent import (
    _APPROVAL_FUTURES,
    CONVERSATION_HISTORY,
    HISTORY_MAX_TURNS,
    READ_ONLY_TOOLS,
    AgentToolCall,
    _agent_chat_completion_stream,
    _run_tool_calls_in_safe_groups,
    append_history,
    get_history,
    history_path,
    make_approval_callback,
    reset_history,
    resolve_approval,
)
from strata.notebook.llm.config import LlmConfig


@pytest.fixture(autouse=True)
def _isolate_globals():
    """Each test starts with empty history and approval stores."""
    CONVERSATION_HISTORY.clear()
    _APPROVAL_FUTURES.clear()
    yield
    CONVERSATION_HISTORY.clear()
    _APPROVAL_FUTURES.clear()


class TestConversationMemory:
    def test_append_and_get(self):
        append_history("nb1", [{"role": "user", "content": "hi"}])
        append_history("nb1", [{"role": "assistant", "content": "hello"}])
        assert get_history("nb1") == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_history_is_per_notebook(self):
        append_history("nb1", [{"role": "user", "content": "a"}])
        append_history("nb2", [{"role": "user", "content": "b"}])
        assert get_history("nb1") == [{"role": "user", "content": "a"}]
        assert get_history("nb2") == [{"role": "user", "content": "b"}]

    def test_reset_clears_only_target(self):
        append_history("nb1", [{"role": "user", "content": "a"}])
        append_history("nb2", [{"role": "user", "content": "b"}])
        reset_history("nb1")
        assert get_history("nb1") == []
        assert get_history("nb2") == [{"role": "user", "content": "b"}]

    def test_get_returns_a_copy(self):
        append_history("nb1", [{"role": "user", "content": "a"}])
        copy = get_history("nb1")
        copy.append({"role": "user", "content": "mutated"})
        assert get_history("nb1") == [{"role": "user", "content": "a"}]

    def test_history_window_caps_old_turns(self):
        # Fill past the cap by twice the limit to exercise the trim.
        turns = [{"role": "user", "content": f"u{i}"} for i in range(HISTORY_MAX_TURNS * 2)]
        for t in turns:
            append_history("nb1", [t])
        history = get_history("nb1")
        assert len(history) == HISTORY_MAX_TURNS
        # The most recent turns should survive; the oldest should be evicted.
        assert history[-1]["content"] == f"u{HISTORY_MAX_TURNS * 2 - 1}"
        assert history[0]["content"] == f"u{HISTORY_MAX_TURNS}"


class TestConversationMemoryPersistence:
    """History writes through to ``.strata/agent_history.json`` and survives
    a simulated server restart (the in-process cache being cleared)."""

    def test_append_persists_to_disk(self, tmp_path):
        append_history("nb1", [{"role": "user", "content": "hi"}], tmp_path)
        path = history_path(tmp_path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["turns"] == [{"role": "user", "content": "hi"}]

    def test_history_survives_restart(self, tmp_path):
        append_history(
            "nb1",
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            tmp_path,
        )
        # Simulate a server restart: the in-process cache is gone (and the
        # session id may differ), only the notebook dir survives.
        CONVERSATION_HISTORY.clear()
        assert get_history("nb1-new-session", tmp_path) == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_append_after_restart_merges_with_disk(self, tmp_path):
        append_history("nb1", [{"role": "user", "content": "old"}], tmp_path)
        CONVERSATION_HISTORY.clear()
        append_history("nb1", [{"role": "user", "content": "new"}], tmp_path)
        assert get_history("nb1", tmp_path) == [
            {"role": "user", "content": "old"},
            {"role": "user", "content": "new"},
        ]

    def test_reset_removes_disk_file(self, tmp_path):
        append_history("nb1", [{"role": "user", "content": "hi"}], tmp_path)
        reset_history("nb1", tmp_path)
        assert not history_path(tmp_path).exists()
        CONVERSATION_HISTORY.clear()
        assert get_history("nb1", tmp_path) == []

    def test_missing_file_is_empty_history(self, tmp_path):
        assert get_history("nb1", tmp_path) == []

    def test_corrupt_file_is_empty_history(self, tmp_path):
        path = history_path(tmp_path)
        path.parent.mkdir(parents=True)
        path.write_text("not json{", encoding="utf-8")
        assert get_history("nb1", tmp_path) == []

    def test_trim_window_applies_on_disk(self, tmp_path):
        turns = [{"role": "user", "content": f"u{i}"} for i in range(HISTORY_MAX_TURNS * 2)]
        for t in turns:
            append_history("nb1", [t], tmp_path)
        data = json.loads(history_path(tmp_path).read_text(encoding="utf-8"))
        assert len(data["turns"]) == HISTORY_MAX_TURNS
        assert data["turns"][-1]["content"] == f"u{HISTORY_MAX_TURNS * 2 - 1}"

    def test_no_dir_means_no_disk_io(self, tmp_path):
        append_history("nb1", [{"role": "user", "content": "hi"}])
        assert not history_path(tmp_path).exists()


class TestApprovalResolution:
    @pytest.mark.asyncio
    async def test_resolve_approval_completes_pending_future(self):
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        _APPROVAL_FUTURES["req1"] = fut

        assert resolve_approval("req1", True) is True
        assert await fut is True
        # The future is removed once resolved so a duplicate response is a no-op.
        assert resolve_approval("req1", False) is False

    def test_resolve_approval_returns_false_for_unknown_id(self):
        assert resolve_approval("missing", True) is False


class TestApprovalCallback:
    @pytest.mark.asyncio
    async def test_auto_approve_returns_none(self):
        async def progress(_event: str, _payload: dict[str, Any]) -> None:
            pass

        cb = make_approval_callback("nb1", progress, auto_approve=True)
        assert cb is None

    @pytest.mark.asyncio
    async def test_callback_round_trip(self):
        events: list[tuple[str, dict[str, Any]]] = []

        async def progress(event: str, payload: dict[str, Any]) -> None:
            events.append((event, payload))
            # Simulate the user clicking Approve immediately.
            if event == "confirm_request":
                resolve_approval(payload["request_id"], True)

        cb = make_approval_callback("nb1", progress, auto_approve=False)
        assert cb is not None
        approved = await cb("delete_cell", {"variable_name": "x"})
        assert approved is True
        assert events[0][0] == "confirm_request"
        assert events[0][1]["tool"] == "delete_cell"

    @pytest.mark.asyncio
    async def test_callback_treats_timeout_as_decline(self):
        async def progress(_event: str, _payload: dict[str, Any]) -> None:
            # Never resolves the future → timeout fires.
            pass

        cb = make_approval_callback("nb1", progress, auto_approve=False, timeout_seconds=0.05)
        assert cb is not None
        approved = await cb("delete_cell", {"variable_name": "x"})
        assert approved is False


class TestParallelSafeDispatch:
    """``_run_tool_calls_in_safe_groups`` runs reads in parallel and writes in order."""

    @pytest.mark.asyncio
    async def test_read_only_tools_run_concurrently(self):
        # Two concurrent reads should overlap, so the total wall time
        # is closer to one delay than two.
        delay = 0.1
        call_count = 0

        async def fake_execute_tool(
            session, tool_name, arguments, notebook_id=None, approval_callback=None
        ):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(delay)
            return f"{tool_name} result"

        tool_calls = [
            {
                "id": "c1",
                "function": {"name": "get_notebook_state", "arguments": "{}"},
            },
            {
                "id": "c2",
                "function": {"name": "get_notebook_state", "arguments": "{}"},
            },
        ]
        record: list[AgentToolCall] = []

        with patch("strata.notebook.llm.agent.execute_tool", fake_execute_tool):
            start = time.monotonic()
            messages = await _run_tool_calls_in_safe_groups(
                session=None,  # type: ignore[arg-type]
                tool_calls=tool_calls,
                notebook_id=None,
                approval_callback=None,
                progress_callback=None,
                record=record,
            )
            elapsed = time.monotonic() - start

        assert call_count == 2
        # If they ran serially, elapsed >= 2*delay. The parallel path
        # should comfortably finish under 1.5*delay.
        assert elapsed < delay * 1.5
        assert {m["tool_call_id"] for m in messages} == {"c1", "c2"}

    @pytest.mark.asyncio
    async def test_mutating_tools_run_serially_in_declared_order(self):
        order: list[str] = []

        async def fake_execute_tool(
            session, tool_name, arguments, notebook_id=None, approval_callback=None
        ):
            order.append(arguments["variable_name"])
            await asyncio.sleep(0.01)
            return f"{tool_name} ok"

        tool_calls = [
            {
                "id": "c1",
                "function": {
                    "name": "edit_cell",
                    "arguments": '{"variable_name": "a", "new_source": "1"}',
                },
            },
            {
                "id": "c2",
                "function": {
                    "name": "edit_cell",
                    "arguments": '{"variable_name": "b", "new_source": "2"}',
                },
            },
            {
                "id": "c3",
                "function": {
                    "name": "run_cell",
                    "arguments": '{"variable_name": "c"}',
                },
            },
        ]
        record: list[AgentToolCall] = []

        with patch("strata.notebook.llm.agent.execute_tool", fake_execute_tool):
            await _run_tool_calls_in_safe_groups(
                session=None,  # type: ignore[arg-type]
                tool_calls=tool_calls,
                notebook_id=None,
                approval_callback=None,
                progress_callback=None,
                record=record,
            )

        # Mutators must observe the order the model emitted them.
        assert order == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_returns_messages_in_original_call_order(self):
        async def fake_execute_tool(
            session, tool_name, arguments, notebook_id=None, approval_callback=None
        ):
            return f"{tool_name} result"

        tool_calls = [
            {
                "id": "alpha",
                "function": {
                    "name": "edit_cell",
                    "arguments": '{"variable_name": "a", "new_source": "1"}',
                },
            },
            {
                "id": "beta",
                "function": {"name": "get_notebook_state", "arguments": "{}"},
            },
            {
                "id": "gamma",
                "function": {
                    "name": "run_cell",
                    "arguments": '{"variable_name": "c"}',
                },
            },
        ]
        record: list[AgentToolCall] = []

        with patch("strata.notebook.llm.agent.execute_tool", fake_execute_tool):
            messages = await _run_tool_calls_in_safe_groups(
                session=None,  # type: ignore[arg-type]
                tool_calls=tool_calls,
                notebook_id=None,
                approval_callback=None,
                progress_callback=None,
                record=record,
            )

        # The tool-result messages must follow the assistant's original
        # tool_calls order so tool_call_id refs line up cleanly.
        assert [m["tool_call_id"] for m in messages] == ["alpha", "beta", "gamma"]


class TestStreamProtocol:
    @pytest.mark.asyncio
    async def test_assembles_text_and_tool_calls_from_deltas(self):
        # SSE chunks the upstream API would send: a couple of text deltas,
        # then two tool_calls assembled across multiple deltas, then usage.
        sse_chunks = [
            'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1",'
                '"function":{"name":"get_notebook_state","arguments":""}}]}}]}'
            ),
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":0,'
                '"function":{"arguments":"{}"}}]}}]}'
            ),
            (
                'data: {"choices":[{"delta":{"tool_calls":[{"index":1,"id":"t2",'
                '"function":{"name":"edit_cell","arguments":"{\\"x\\":1}"}}]}}]}'
            ),
            'data: {"choices":[{"finish_reason":"tool_calls","delta":{}}]}',
            'data: {"usage":{"prompt_tokens":10,"completion_tokens":4}}',
            "data: [DONE]",
        ]

        async def fake_aiter_lines(self):
            for line in sse_chunks:
                yield line

        class _FakeStreamResp:
            is_success = True
            status_code = 200

            async def aread(self) -> bytes:
                return b""

            aiter_lines = fake_aiter_lines

        class _FakeStreamCtx:
            async def __aenter__(self_inner):
                return _FakeStreamResp()

            async def __aexit__(self_inner, *args):
                return None

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self_inner):
                return self_inner

            async def __aexit__(self_inner, *args):
                return None

            def stream(self_inner, *args, **kwargs):
                return _FakeStreamCtx()

        config = LlmConfig(
            base_url="https://api.openai.com/v1",
            api_key="sk-test",
            model="gpt-test",
        )

        events: list[dict[str, Any]] = []
        with patch("strata.notebook.llm.agent.httpx.AsyncClient", _FakeClient):
            async for ev in _agent_chat_completion_stream(
                config, [{"role": "user", "content": "hi"}], None
            ):
                events.append(ev)

        deltas = [e for e in events if e["type"] == "text_delta"]
        completes = [e for e in events if e["type"] == "complete"]
        assert [d["text"] for d in deltas] == ["Hel", "lo"]
        assert len(completes) == 1
        complete = completes[0]
        assert complete["message"]["content"] == "Hello"
        tool_calls = complete["message"]["tool_calls"]
        assert [tc["id"] for tc in tool_calls] == ["t1", "t2"]
        assert tool_calls[0]["function"]["name"] == "get_notebook_state"
        assert tool_calls[0]["function"]["arguments"] == "{}"
        assert tool_calls[1]["function"]["arguments"] == '{"x":1}'
        assert complete["finish_reason"] == "tool_calls"
        assert complete["input_tokens"] == 10
        assert complete["output_tokens"] == 4


class TestReadOnlyToolSet:
    def test_get_notebook_state_is_read_only(self):
        assert "get_notebook_state" in READ_ONLY_TOOLS

    def test_destructive_tools_are_not_read_only(self):
        # A safety-critical invariant: nothing destructive may be parallelized.
        for name in ("delete_cell", "add_package", "create_cell", "edit_cell", "run_cell"):
            assert name not in READ_ONLY_TOOLS
