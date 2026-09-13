"""Which assistant tools ask first, and which Auto-approve cannot skip. Item 33.

The built-in assistant gated exactly ``delete_cell`` and ``add_package``, as a
constant, and the Auto-approve toggle turned every gate off. A shared server had
no way to make the assistant ask before running a cell, and no gate an operator
could rely on staying on.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from strata.notebook.llm.agent import (
    _APPROVAL_FUTURES,
    CONVERSATION_HISTORY,
    execute_tool,
    make_approval_callback,
    resolve_approval,
    run_agent_loop,
)
from strata.notebook.llm.config import DEFAULT_APPROVAL_TOOLS, LlmConfig, resolve_llm_config


@pytest.fixture(autouse=True)
def _isolate_globals():
    CONVERSATION_HISTORY.clear()
    _APPROVAL_FUTURES.clear()
    yield
    CONVERSATION_HISTORY.clear()
    _APPROVAL_FUTURES.clear()


class _Server:
    ai_api_key = "k"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestResolution:
    def test_unset_keeps_todays_gates(self):
        config = resolve_llm_config(server_config=_Server())

        assert config is not None
        assert config.approval_tools == DEFAULT_APPROVAL_TOOLS
        assert config.locked_tools == frozenset()

    def test_the_server_list_replaces_the_default(self):
        config = resolve_llm_config(server_config=_Server(ai_approval_tools=["run_cell"]))

        assert config is not None
        assert config.approval_tools == {"run_cell"}

    def test_a_notebook_adds_but_cannot_remove(self):
        """Anyone who can edit notebook.toml would otherwise turn the operator's
        gate off by listing fewer tools."""
        config = resolve_llm_config(
            notebook_config={"approval_tools": ["edit_cell"]},
            server_config=_Server(ai_approval_tools=["run_cell", "delete_cell"]),
        )

        assert config is not None
        assert config.approval_tools == {"run_cell", "delete_cell", "edit_cell"}

    def test_a_locked_tool_is_gated_even_if_not_listed(self):
        config = resolve_llm_config(server_config=_Server(ai_gates_locked=["run_cell"]))

        assert config is not None
        assert "run_cell" in config.approval_tools
        assert config.locked_tools == {"run_cell"}


class TestServerSettings:
    def test_a_typo_fails_at_startup(self):
        """``run_cells`` would gate nothing, on the server that asked for a gate."""
        from pydantic import ValidationError

        from strata.config import StrataConfig

        with pytest.raises(ValidationError, match="run_cells"):
            StrataConfig(ai_approval_tools="delete_cell,run_cells")

    def test_env_style_lists_parse(self):
        from strata.config import StrataConfig

        config = StrataConfig(ai_approval_tools="run_cell, add_package", ai_gates_locked="run_cell")

        assert config.ai_approval_tools == ["run_cell", "add_package"]
        assert config.ai_gates_locked == ["run_cell"]
        assert StrataConfig().ai_approval_tools is None


class TestTheGate:
    async def test_only_gated_tools_are_asked_about(self):
        asked: list[str] = []

        async def _decline(tool: str, _args: dict[str, Any]) -> bool:
            asked.append(tool)
            return False

        declined = await execute_tool(
            None,  # type: ignore[arg-type] — a declined tool never touches it
            "run_cell",
            {"cell_id": "c1"},
            approval_callback=_decline,
            gated_tools=frozenset({"run_cell"}),
        )

        assert asked == ["run_cell"]
        assert declined.startswith("User declined")

    async def test_auto_approve_still_asks_about_a_locked_gate(self):
        events: list[tuple[str, dict[str, Any]]] = []

        async def progress(event: str, payload: dict[str, Any]) -> None:
            events.append((event, payload))
            if event == "confirm_request":
                resolve_approval(payload["request_id"], False)

        callback = make_approval_callback(
            "nb1", progress, auto_approve=True, locked_tools=frozenset({"run_cell"})
        )

        assert callback is not None
        assert await callback("delete_cell", {}) is True
        assert events == []
        assert await callback("run_cell", {}) is False
        assert [e[1]["tool"] for e in events] == ["run_cell"]

    async def test_with_no_one_to_ask_a_locked_gate_declines(self):
        callback = make_approval_callback(
            "nb1", None, auto_approve=False, locked_tools=frozenset({"run_cell"})
        )

        assert callback is not None
        assert await callback("run_cell", {}) is False
        assert await callback("delete_cell", {}) is True


@pytest.mark.parametrize(
    ("auto_approve", "locked"),
    [(False, frozenset()), (True, frozenset({"run_cell"}))],
    ids=["gated", "locked-with-auto-approve"],
)
async def test_the_loop_prompts_for_a_server_gated_run_cell(tmp_path, auto_approve, locked):
    """Through the loop, so the configured set is what actually reaches the
    gate rather than the module default."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    notebook_dir = create_notebook(tmp_path, "Gated", initialize_environment=False)
    add_cell_to_notebook(notebook_dir, "c1", None)
    write_cell(notebook_dir, "c1", "x = 1")
    session = NotebookSession(parse_notebook(notebook_dir), notebook_dir)

    turns = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "run_cell", "arguments": '{"cell_id": "c1"}'},
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]

    async def _stream(_config, _messages, _tools):
        yield {
            "type": "complete",
            "message": turns.pop(0),
            "input_tokens": 1,
            "output_tokens": 1,
            "model": "fake",
        }

    events: list[tuple[str, dict[str, Any]]] = []

    async def progress(event: str, payload: dict[str, Any]) -> None:
        events.append((event, payload))
        if event == "confirm_request":
            resolve_approval(payload["request_id"], False)

    config = LlmConfig(
        base_url="http://llm.invalid",
        api_key="k",
        model="fake",
        approval_tools=DEFAULT_APPROVAL_TOOLS | {"run_cell"},
        locked_tools=locked,
    )
    with patch("strata.notebook.llm.agent._agent_chat_completion_stream", _stream):
        result = await run_agent_loop(
            config,
            session,
            "run it",
            progress_callback=progress,
            auto_approve=auto_approve,
        )

    assert [p["tool"] for e, p in events if e == "confirm_request"] == ["run_cell"]
    assert result.tool_calls[0].result.startswith("User declined")
