"""Notebook agent loop with tool use, streaming, memory, and approvals.

Design notes (vs. the previous single-shot loop):

* **Streaming.** Each iteration calls ``_agent_chat_completion_stream``
  which yields ``text_delta`` events as the model writes its narrative
  and a final ``complete`` event with any tool calls. The progress
  callback fires for both, so the frontend can render a live assistant
  message instead of just event-name strings.

* **Conversation memory.** ``CONVERSATION_HISTORY`` keeps the last few
  user/assistant text turns per notebook so a follow-up Agent press has
  the model's prior turn as context, and mirrors them to
  ``.strata/agent_history.json`` so the context survives a server
  restart. Tool-call traces are *not* persisted — they bloat context
  fast and the surviving prose summary carries the meaningful state.

* **Refreshed snapshot.** The system prompt is short and tool-focused.
  On the first iteration we inject a synthetic "tool result" containing
  the current notebook state so the model has the snapshot for free
  without burning a round-trip. Subsequent calls rely on the model
  re-fetching with ``get_notebook_state`` if it needs to.

* **Approval gate.** ``delete_cell`` and ``add_package`` raise an
  ``agent_confirm_request`` over the progress callback and block on a
  per-notebook future. The frontend posts ``agent_confirm_response``
  via WebSocket; the loop resumes with a tool result reflecting the
  user's decision.

* **Parallel-safe dispatch.** Read-only tools (``get_notebook_state``)
  run concurrently when the model emits multiple in one turn. Mutating
  tools run serially in declared order so cell creation/edit/run see a
  consistent notebook state.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from strata.notebook.llm.config import (
    LlmConfig,
    max_output_tokens_param,
)
from strata.notebook.llm.context import build_notebook_context

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------

# Tools that don't mutate notebook state — safe to fan out in parallel.
READ_ONLY_TOOLS: frozenset[str] = frozenset({"get_notebook_state"})

# Tools that need explicit user approval before running. The frontend
# can opt out of approval with an "auto-approve" toggle, in which case
# the routes layer suppresses the gate by passing ``approval_callback=None``.
DESTRUCTIVE_TOOLS: frozenset[str] = frozenset({"delete_cell", "add_package"})


AGENT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_notebook_state",
            "description": (
                "Get current notebook state: all cells with source code, "
                "defined variables, execution status, and dependency graph."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_cell",
            "description": (
                "Create a new Python or prompt cell. Returns the cell ID "
                "and the variables it defines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "description": "Source code for the cell",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "prompt", "markdown"],
                        "default": "python",
                    },
                    "after_variable": {
                        "type": "string",
                        "description": (
                            "Insert after the cell that defines this variable. "
                            "Omit to append at end."
                        ),
                    },
                },
                "required": ["source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_cell",
            "description": (
                "Replace the source code of an existing cell, identified "
                "by the variable it defines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variable_name": {
                        "type": "string",
                        "description": "Variable defined by the target cell",
                    },
                    "new_source": {
                        "type": "string",
                        "description": "New source code",
                    },
                },
                "required": ["variable_name", "new_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_cell",
            "description": ("Delete an existing cell, identified by the variable it defines."),
            "parameters": {
                "type": "object",
                "properties": {
                    "variable_name": {
                        "type": "string",
                        "description": "Variable defined by the target cell",
                    },
                },
                "required": ["variable_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cell",
            "description": (
                "Execute a cell and return its output, stdout, stderr, "
                "and any errors. Automatically cascades upstream cells."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "variable_name": {
                        "type": "string",
                        "description": "Variable defined by the target cell",
                    },
                },
                "required": ["variable_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_package",
            "description": "Install a Python package into the notebook environment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "package_spec": {
                        "type": "string",
                        "description": "Package spec (e.g. 'pandas>=2.0')",
                    },
                },
                "required": ["package_spec"],
            },
        },
    },
]


_SYSTEM_AGENT = """\
You are a Python notebook assistant with tools for inspecting, creating, \
editing, deleting, and running cells, and installing packages.

Rules:
- Reference cells by the VARIABLE NAME they define, not by ID.
- Use get_notebook_state when you need a fresh view of the notebook.
- After creating or editing a cell, run it to verify.
- If a cell errors, read the error, fix the code, and retry.
- On ModuleNotFoundError, install the missing package, then retry.
- Keep cells focused — one logical step per cell.
- When done, write a brief summary of what you did."""


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


@dataclass
class AgentToolCall:
    """Record of one tool call in the agent loop."""

    tool_name: str
    arguments: dict[str, Any]
    result: str
    duration_ms: float


@dataclass
class AgentLoopResult:
    """Final result of an agent loop run."""

    content: str
    model: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    iterations: int = 0
    tool_calls: list[AgentToolCall] = field(default_factory=list)
    cancelled: bool = False
    error: str | None = None


# Per-notebook persistent memory: only user / assistant text turns. Tool
# traces are not kept across runs (they bloat context fast). The in-process
# dict is a cache over ``.strata/agent_history.json`` so the LLM panel's
# context survives a server restart; pass ``notebook_dir`` to read through
# to / write through to disk.
CONVERSATION_HISTORY: dict[str, list[dict[str, str]]] = {}
HISTORY_MAX_TURNS = 12  # 6 user/assistant pairs

_HISTORY_FILENAME = "agent_history.json"


def history_path(notebook_dir: Path) -> Path:
    return Path(notebook_dir) / ".strata" / _HISTORY_FILENAME


def _load_history_from_disk(notebook_dir: Path) -> list[dict[str, str]]:
    """Return persisted turns, or empty.

    A missing or unparseable file produces an empty history — like
    ``runtime.json``, this is augmenting state, not authoritative, so a
    corrupt file must not break the LLM panel.
    """
    path = history_path(notebook_dir)
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return []
    return data.get("turns", [])


def _save_history_to_disk(notebook_dir: Path, turns: list[dict[str, str]]) -> None:
    """Atomically persist conversation turns under ``.strata/``."""
    path = history_path(notebook_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"turns": turns}, f, indent=2)
    os.replace(tmp_name, path)


def _cached_history(notebook_id: str, notebook_dir: Path | None) -> list[dict[str, str]]:
    """Return the live in-process history list, seeding from disk on miss."""
    if notebook_id not in CONVERSATION_HISTORY and notebook_dir is not None:
        CONVERSATION_HISTORY[notebook_id] = _load_history_from_disk(notebook_dir)
    return CONVERSATION_HISTORY.setdefault(notebook_id, [])


def get_history(notebook_id: str, notebook_dir: Path | None = None) -> list[dict[str, str]]:
    """Return a copy of the persistent conversation history for a notebook."""
    return list(_cached_history(notebook_id, notebook_dir))


def append_history(
    notebook_id: str,
    turns: list[dict[str, str]],
    notebook_dir: Path | None = None,
) -> None:
    """Append turns, trim to the configured window, and persist to disk."""
    history = _cached_history(notebook_id, notebook_dir)
    history.extend(turns)
    if len(history) > HISTORY_MAX_TURNS:
        del history[: len(history) - HISTORY_MAX_TURNS]
    if notebook_dir is not None:
        _save_history_to_disk(notebook_dir, history)


def reset_history(notebook_id: str, notebook_dir: Path | None = None) -> None:
    """Clear stored conversation history for a notebook, on disk too."""
    CONVERSATION_HISTORY.pop(notebook_id, None)
    if notebook_dir is not None:
        history_path(notebook_dir).unlink(missing_ok=True)


# Per-notebook approval futures, keyed by request id. The routes layer
# resolves these when the frontend sends ``agent_confirm_response``.
_APPROVAL_FUTURES: dict[str, asyncio.Future[bool]] = {}


def resolve_approval(request_id: str, approved: bool) -> bool:
    """Resolve a pending approval future. Returns True if found."""
    fut = _APPROVAL_FUTURES.pop(request_id, None)
    if fut is not None and not fut.done():
        fut.set_result(approved)
        return True
    return False


# ---------------------------------------------------------------------------
# Streaming chat completion with tools
# ---------------------------------------------------------------------------


async def _agent_chat_completion_stream(
    config: LlmConfig,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
):
    """Stream a tool-aware chat completion.

    Yields text-delta events and a single terminal ``complete`` event
    carrying the assembled assistant message (with any tool calls), the
    finish reason, token usage, and the model name. Streaming the
    response means the user sees the model's narrative as it appears
    instead of after the entire turn finishes.
    """
    model = config.model
    input_tokens = 0
    output_tokens = 0
    finish_reason: str | None = None

    content_parts: list[str] = []
    # Tool calls arrive as deltas keyed by their array index. Each entry is
    # ``{"id": str, "function": {"name": str, "arguments": str}}`` accumulated
    # across deltas; the OpenAI stream protocol guarantees deltas for the
    # same index belong to the same tool call.
    tool_call_buffers: dict[int, dict[str, Any]] = {}

    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        max_output_tokens_param(config.base_url): config.max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        async with client.stream(
            "POST",
            f"{config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json=body,
        ) as resp:
            if not resp.is_success:
                raw = (await resp.aread()).decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(
                    f"LLM provider returned HTTP {resp.status_code} "
                    f"for model {config.model!r}: {raw}"
                )
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if chunk.get("model"):
                    model = chunk["model"]
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    input_tokens = usage.get("prompt_tokens", input_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                delta = choice.get("delta") or {}
                text = delta.get("content")
                if text:
                    content_parts.append(text)
                    yield {"type": "text_delta", "text": text}

                for tc_delta in delta.get("tool_calls") or []:
                    idx = tc_delta.get("index", 0)
                    buf = tool_call_buffers.setdefault(
                        idx,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if tc_delta.get("id"):
                        buf["id"] = tc_delta["id"]
                    fn_delta = tc_delta.get("function") or {}
                    if fn_delta.get("name"):
                        buf["function"]["name"] = fn_delta["name"]
                    if fn_delta.get("arguments"):
                        buf["function"]["arguments"] += fn_delta["arguments"]

    assembled_tool_calls = [tool_call_buffers[i] for i in sorted(tool_call_buffers)]
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts) or None,
    }
    if assembled_tool_calls:
        message["tool_calls"] = assembled_tool_calls

    yield {
        "type": "complete",
        "message": message,
        "finish_reason": finish_reason,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "model": model,
    }


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def resolve_variable_to_cell_id(
    session: NotebookSession,
    variable_name: str,
) -> str | None:
    """Resolve a variable name to the cell ID that defines it."""
    from strata.notebook.dag import SweepProducer

    if session.dag and variable_name in session.dag.variable_producer:
        producer = session.dag.variable_producer[variable_name]
        # A sweep group has many producing cells; return the first member as a
        # representative (the agent only needs *a* defining cell to point at).
        if isinstance(producer, SweepProducer):
            return producer.variants[0][1] if producer.variants else None
        return producer
    for cell in session.notebook_state.cells:
        if variable_name in cell.defines:
            return cell.id
    return None


# Approval callback signature: (tool_name, arguments) -> awaitable[bool]
ApprovalCallback = Callable[[str, dict[str, Any]], Awaitable[bool]]


async def execute_tool(
    session: NotebookSession,
    tool_name: str,
    arguments: dict[str, Any],
    notebook_id: str | None = None,
    approval_callback: ApprovalCallback | None = None,
) -> str:
    """Execute a tool call and return a text result for the LLM.

    When ``approval_callback`` is set and the tool is in
    ``DESTRUCTIVE_TOOLS``, the callback is awaited before running. If
    the user declines, the tool result reflects that and no mutation
    occurs.
    """
    from strata.notebook.executor import CellExecutor
    from strata.notebook.writer import (
        add_cell_to_notebook,
        remove_cell_from_notebook,
        write_cell,
    )

    if approval_callback is not None and tool_name in DESTRUCTIVE_TOOLS:
        try:
            approved = await approval_callback(tool_name, arguments)
        except Exception as exc:
            return f"Error: approval check failed for {tool_name}: {exc}"
        if not approved:
            return f"User declined to run {tool_name}({_short_args(arguments)})."

    async def _sync_frontend() -> None:
        """Broadcast updated notebook state to all connected frontends."""
        if notebook_id:
            try:
                from strata.notebook.ws import broadcast_notebook_sync

                await broadcast_notebook_sync(notebook_id, session)
            except Exception:
                pass

    try:
        if tool_name == "get_notebook_state":
            return build_notebook_context(session, max_tokens=4000)

        elif tool_name == "create_cell":
            source = arguments.get("source", "")
            language = arguments.get("language", "python")
            after_var = arguments.get("after_variable")
            after_id = None
            if after_var:
                after_id = resolve_variable_to_cell_id(session, after_var)

            cell_id = str(uuid.uuid4())[:8]
            add_cell_to_notebook(session.path, cell_id, after_id, language=language)
            if source:
                write_cell(session.path, cell_id, source)
            session.reload()
            await _sync_frontend()

            cell = next(
                (c for c in session.notebook_state.cells if c.id == cell_id),
                None,
            )
            defines = cell.defines if cell else []
            return f"Created cell {cell_id} (defines: {', '.join(defines) or 'none'})"

        elif tool_name == "edit_cell":
            var_name = arguments.get("variable_name", "")
            new_source = arguments.get("new_source", "")
            cell_id = resolve_variable_to_cell_id(session, var_name)
            if not cell_id:
                valid = [v for c in session.notebook_state.cells for v in c.defines]
                return f"Error: No cell defines '{var_name}'. Valid variables: {valid}"
            write_cell(session.path, cell_id, new_source)
            session.reload()
            await _sync_frontend()
            return f"Edited cell {cell_id} (was defining: {var_name})"

        elif tool_name == "delete_cell":
            var_name = arguments.get("variable_name", "")
            cell_id = resolve_variable_to_cell_id(session, var_name)
            if not cell_id:
                valid = [v for c in session.notebook_state.cells for v in c.defines]
                return f"Error: No cell defines '{var_name}'. Valid variables: {valid}"
            remove_cell_from_notebook(session.path, cell_id)
            session.reload()
            await _sync_frontend()
            return f"Deleted cell {cell_id} (defined: {var_name})"

        elif tool_name == "run_cell":
            var_name = arguments.get("variable_name", "")
            cell_id = resolve_variable_to_cell_id(session, var_name)
            if not cell_id:
                valid = [v for c in session.notebook_state.cells for v in c.defines]
                return f"Error: No cell defines '{var_name}'. Valid variables: {valid}"
            cell = next(
                (c for c in session.notebook_state.cells if c.id == cell_id),
                None,
            )
            if not cell:
                return f"Error: Cell {cell_id} not found in session state"

            if notebook_id:
                from strata.notebook.ws import execute_cell_for_agent

                try:
                    result = await execute_cell_for_agent(
                        notebook_id, session, cell_id, cell.source
                    )
                except RuntimeError as e:
                    return f"Error: {e}"
            else:
                executor = CellExecutor(session, session.warm_pool)
                result = await executor.execute_cell(cell_id, cell.source)

            parts = []
            if result.success:
                parts.append("Execution succeeded.")
                if result.cache_hit:
                    parts.append("(cache hit)")
            else:
                parts.append("Execution FAILED.")
            if result.error:
                parts.append(f"Error: {result.error}")
            if result.stdout:
                stdout = result.stdout[:2000]
                parts.append(f"Stdout:\n{stdout}")
            if result.stderr:
                stderr = result.stderr[:1000]
                parts.append(f"Stderr:\n{stderr}")
            if result.outputs:
                for name, meta in result.outputs.items():
                    preview = meta.get("preview", "")
                    parts.append(f"Output '{name}': {str(preview)[:500]}")
            await _sync_frontend()
            return "\n".join(parts)

        elif tool_name == "add_package":
            spec = arguments.get("package_spec", "")
            if not spec:
                return "Error: package_spec is required"
            try:
                job = await session.submit_environment_job(action="add", package=spec)
                await session.wait_for_environment_job()
                await _sync_frontend()
                history = session.serialize_environment_job_history()
                completed = next((entry for entry in history if entry.get("id") == job.id), None)
                if completed and completed.get("status") == "completed":
                    return f"Installed {spec} successfully."
                error = (
                    completed.get("error")
                    if completed is not None
                    else "Environment job did not finish cleanly"
                )
                return f"Failed to install {spec}: {error}"
            except Exception as e:
                return f"Error installing {spec}: {e}"

        else:
            return f"Error: Unknown tool '{tool_name}'"

    except Exception as e:
        return f"Error in {tool_name}: {type(e).__name__}: {e}"


def _short_args(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{k}={str(v)[:50]}" for k, v in arguments.items())


# ---------------------------------------------------------------------------
# Approval helper
# ---------------------------------------------------------------------------


def make_approval_callback(
    notebook_id: str,
    progress_callback: Callable[[str, dict[str, Any]], Awaitable[None]] | None,
    auto_approve: bool,
    timeout_seconds: float = 120.0,
) -> ApprovalCallback | None:
    """Build the approval callback used by ``execute_tool``.

    * ``auto_approve=True`` → returns ``None``, skipping the gate.
    * Otherwise → returns a callback that emits an
      ``agent_confirm_request`` event and waits for
      ``resolve_approval(request_id, approved)`` to be called by the
      WebSocket layer when the user clicks Approve / Decline.

    The future times out after ``timeout_seconds`` and is treated as a
    decline so the loop never hangs forever on a closed tab.
    """
    if auto_approve or progress_callback is None:
        return None

    async def _ask(tool_name: str, arguments: dict[str, Any]) -> bool:
        request_id = uuid.uuid4().hex[:12]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        _APPROVAL_FUTURES[request_id] = fut
        await progress_callback(
            "confirm_request",
            {
                "request_id": request_id,
                "tool": tool_name,
                "arguments": arguments,
                "summary": f"{tool_name}({_short_args(arguments)})",
            },
        )
        try:
            return await asyncio.wait_for(fut, timeout=timeout_seconds)
        except TimeoutError:
            return False
        finally:
            _APPROVAL_FUTURES.pop(request_id, None)

    return _ask


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


# Progress callback shape: (event_type, payload) -> awaitable[None].
ProgressCallback = Callable[[str, dict[str, Any]], Awaitable[None]]


async def _run_tool_calls_in_safe_groups(
    session: NotebookSession,
    tool_calls: list[dict[str, Any]],
    *,
    notebook_id: str | None,
    approval_callback: ApprovalCallback | None,
    progress_callback: ProgressCallback | None,
    record: list[AgentToolCall],
) -> list[dict[str, Any]]:
    """Execute tool calls with read-only ones fanned out and mutations serial.

    Returns the list of ``role: tool`` messages to append, in the same
    order as ``tool_calls`` so the assistant's ``tool_calls`` array
    matches the ``tool_call_id`` references in the conversation.
    """
    # Pre-parse arguments so failures in JSON are recorded as tool errors
    # rather than aborting the whole batch.
    parsed: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        parsed.append((tc, args))

    results_by_id: dict[str, str] = {}
    durations_by_id: dict[str, float] = {}

    async def _exec(tc: dict[str, Any], args: dict[str, Any]) -> None:
        fn = tc.get("function") or {}
        tool_name = fn.get("name") or ""
        if progress_callback:
            await progress_callback(
                "tool_call",
                {"tool": tool_name, "arguments": args, "tool_call_id": tc.get("id")},
            )
        start = time.time()
        result = await execute_tool(
            session,
            tool_name,
            args,
            notebook_id=notebook_id,
            approval_callback=approval_callback,
        )
        duration_ms = (time.time() - start) * 1000
        results_by_id[tc.get("id") or ""] = result
        durations_by_id[tc.get("id") or ""] = duration_ms
        record.append(
            AgentToolCall(
                tool_name=tool_name,
                arguments=args,
                result=result[:500],
                duration_ms=duration_ms,
            )
        )
        if progress_callback:
            await progress_callback(
                "tool_result",
                {
                    "tool": tool_name,
                    "tool_call_id": tc.get("id"),
                    "result": result[:500],
                    "duration_ms": int(duration_ms),
                },
            )

    read_only: list[tuple[dict[str, Any], dict[str, Any]]] = []
    mutating: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for tc, args in parsed:
        name = (tc.get("function") or {}).get("name") or ""
        if name in READ_ONLY_TOOLS:
            read_only.append((tc, args))
        else:
            mutating.append((tc, args))

    if read_only:
        await asyncio.gather(*(_exec(tc, args) for tc, args in read_only))
    for tc, args in mutating:
        await _exec(tc, args)

    # Re-emit results in the original ``tool_calls`` order so the message
    # array reads naturally and any client UI grouping still aligns.
    return [
        {
            "role": "tool",
            "tool_call_id": tc.get("id"),
            "content": results_by_id.get(tc.get("id") or "", ""),
        }
        for tc, _ in parsed
    ]


async def run_agent_loop(
    config: LlmConfig,
    session: NotebookSession,
    user_message: str,
    *,
    notebook_id: str | None = None,
    max_iterations: int = 10,
    cancel_event: asyncio.Event | None = None,
    progress_callback: ProgressCallback | None = None,
    auto_approve: bool = False,
) -> AgentLoopResult:
    """Run an agent loop with tool use, streaming, memory, and approvals.

    Conversation memory is keyed on ``notebook_id``: prior user/assistant
    text turns are folded into the message list automatically. Pass
    ``notebook_id=None`` to run a one-off loop without memory (used by
    the legacy in-process test path).
    """
    history = get_history(notebook_id, session.path) if notebook_id else []
    snapshot_text = build_notebook_context(session, max_tokens=4000)
    snapshot_call_id = "snapshot_" + uuid.uuid4().hex[:8]

    # Bootstrap message list:
    #   1. Short tool-aware system prompt
    #   2. Persistent prior turns (text only)
    #   3. A synthetic assistant tool call + the snapshot, so the model
    #      starts iteration 1 with notebook state already in hand.
    #   4. The new user message.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _SYSTEM_AGENT},
        *history,
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": snapshot_call_id,
                    "type": "function",
                    "function": {"name": "get_notebook_state", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": snapshot_call_id, "content": snapshot_text},
        {"role": "user", "content": user_message},
    ]

    result = AgentLoopResult(content="", model=config.model)
    approval_callback = make_approval_callback(
        notebook_id or "_anon",
        progress_callback,
        auto_approve=auto_approve,
        timeout_seconds=config.approval_timeout_seconds,
    )

    if progress_callback:
        await progress_callback(
            "started",
            {"detail": user_message[:200], "notebook_id": notebook_id},
        )

    final_assistant_text: str | None = None

    for iteration in range(max_iterations):
        if cancel_event and cancel_event.is_set():
            result.cancelled = True
            result.content = "Agent cancelled by user."
            if progress_callback:
                await progress_callback("cancelled", {"detail": "Agent cancelled"})
            break

        result.iterations = iteration + 1
        if progress_callback:
            await progress_callback(
                "iteration",
                {"index": iteration + 1, "max": max_iterations},
            )

        # Stream this turn.
        completion: dict[str, Any] | None = None
        try:
            async for event in _agent_chat_completion_stream(config, messages, AGENT_TOOLS):
                if event["type"] == "text_delta":
                    if progress_callback:
                        await progress_callback("text_delta", {"text": event["text"]})
                elif event["type"] == "complete":
                    completion = event
        except Exception as e:
            result.error = f"LLM call failed: {e}"
            if progress_callback:
                await progress_callback("error", {"detail": result.error})
            break

        if completion is None:
            result.error = "LLM stream ended without a complete event"
            if progress_callback:
                await progress_callback("error", {"detail": result.error})
            break

        result.total_input_tokens += completion["input_tokens"]
        result.total_output_tokens += completion["output_tokens"]
        result.model = completion["model"]

        message = completion["message"]
        tool_calls = message.get("tool_calls") or []

        # If the model produced text and made no tool calls, this is the
        # final answer and the loop is done.
        if not tool_calls:
            final_assistant_text = message.get("content") or ""
            result.content = final_assistant_text
            if progress_callback:
                await progress_callback(
                    "done",
                    {"detail": final_assistant_text[:200]},
                )
            break

        # Otherwise append the assistant message and execute the tool calls.
        messages.append(message)
        tool_messages = await _run_tool_calls_in_safe_groups(
            session,
            tool_calls,
            notebook_id=notebook_id,
            approval_callback=approval_callback,
            progress_callback=progress_callback,
            record=result.tool_calls,
        )
        messages.extend(tool_messages)

    # Hit iteration limit without a final text → ask once more without tools.
    if final_assistant_text is None and not result.cancelled and result.error is None:
        messages.append(
            {
                "role": "user",
                "content": (
                    "You have reached the maximum number of tool calls. "
                    "Please respond with a summary of what you accomplished."
                ),
            }
        )
        try:
            final: dict[str, Any] | None = None
            async for event in _agent_chat_completion_stream(config, messages, None):
                if event["type"] == "text_delta":
                    if progress_callback:
                        await progress_callback("text_delta", {"text": event["text"]})
                elif event["type"] == "complete":
                    final = event
            if final is not None:
                result.total_input_tokens += final["input_tokens"]
                result.total_output_tokens += final["output_tokens"]
                final_assistant_text = (final["message"].get("content") or "") if final else ""
                result.content = final_assistant_text or ""
        except Exception as e:
            result.error = f"Final LLM call failed: {e}"

        if progress_callback:
            await progress_callback("done", {"detail": result.content[:200]})

    # Persist the user message and the final assistant text into history.
    if notebook_id and not result.cancelled and result.error is None and final_assistant_text:
        append_history(
            notebook_id,
            [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final_assistant_text},
            ],
            session.path,
        )

    return result


__all__ = [
    "AGENT_TOOLS",
    "AgentLoopResult",
    "AgentToolCall",
    "DESTRUCTIVE_TOOLS",
    "READ_ONLY_TOOLS",
    "append_history",
    "execute_tool",
    "get_history",
    "history_path",
    "make_approval_callback",
    "resolve_approval",
    "reset_history",
    "resolve_variable_to_cell_id",
    "run_agent_loop",
]
