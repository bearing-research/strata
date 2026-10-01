"""Structured-output helpers: schema normalization and Anthropic tool-use.

The OpenAI-compat path enforces schemas with ``response_format``; the Anthropic
native path uses one forced tool call whose ``input_schema`` is the schema.
"""

from __future__ import annotations

import json
from typing import Any

from strata.notebook.llm.config import LlmCompletionResult

_ANTHROPIC_TOOL_NAME = "respond"
_ANTHROPIC_API_VERSION = "2023-06-01"


def _normalize_openai_strict_schema(
    node: Any,
) -> tuple[Any, bool]:
    """Return the schema shaped for OpenAI ``strict: true`` and whether strict is safe.

    Strict mode needs ``additionalProperties: false`` on every object (always
    added: it only tightens) and every property in ``required``. If the user left a
    property out of ``required``, strict is reported unsafe rather than silently
    making that field required.
    """
    strict_ok = True

    def _walk(n: Any) -> Any:
        nonlocal strict_ok
        if isinstance(n, dict):
            node_copy = {k: _walk(v) for k, v in n.items()}
            if node_copy.get("type") == "object":
                node_copy.setdefault("additionalProperties", False)
                properties = node_copy.get("properties")
                if isinstance(properties, dict):
                    declared_required = node_copy.get("required")
                    if not isinstance(declared_required, list) or set(declared_required) != set(
                        properties.keys()
                    ):
                        strict_ok = False
            return node_copy
        if isinstance(n, list):
            return [_walk(item) for item in n]
        return n

    return _walk(node), strict_ok


def response_format_for(
    base_url: str,
    *,
    output_type: str | None,
    output_schema: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Pick the provider-appropriate ``response_format`` payload.

    With a schema, endpoints that accept strict ``json_schema`` get it (strict when the
    schema allows it): OpenAI and OpenAI-compatible ones such as Gemini's, matched by
    "openai" in the base URL. Other endpoints, and ``output_type == "json"`` without a
    schema, get ``json_object``. Returns ``None`` when no structured output is requested.
    """
    if output_schema is not None:
        if "openai" in base_url.lower():
            normalized, strict_ok = _normalize_openai_strict_schema(output_schema)
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "PromptResponse",
                    "schema": normalized,
                    "strict": strict_ok,
                },
            }
        return {"type": "json_object"}
    if output_type == "json":
        return {"type": "json_object"}
    return None


def _split_system_and_messages(
    messages: list[dict[str, str]],
) -> tuple[str | None, list[dict[str, str]]]:
    """Lift ``role: system`` turns into one system string for Anthropic's native API."""
    system_chunks: list[str] = []
    remaining: list[dict[str, str]] = []
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content") or ""
            if content:
                system_chunks.append(content)
        else:
            remaining.append(msg)
    system = "\n\n".join(system_chunks) if system_chunks else None
    return system, remaining


def build_anthropic_tool_use_body(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float | None,
    output_schema: dict[str, Any],
) -> dict[str, Any]:
    """Build the native Anthropic ``/v1/messages`` body for schema-constrained output.

    Defines one tool whose ``input_schema`` is the requested schema and forces the
    model to call it, so ``tool_use.input`` conforms to the schema.
    """
    system, filtered_messages = _split_system_and_messages(messages)
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": filtered_messages,
        "tools": [
            {
                "name": _ANTHROPIC_TOOL_NAME,
                "description": (
                    "Respond with a single object matching the input schema. "
                    "Do not return prose — only the structured data."
                ),
                "input_schema": output_schema,
            }
        ],
        "tool_choice": {"type": "tool", "name": _ANTHROPIC_TOOL_NAME},
    }
    if system is not None:
        body["system"] = system
    if temperature is not None:
        body["temperature"] = temperature
    return body


def parse_anthropic_tool_use_response(
    data: dict[str, Any],
    fallback_model: str,
) -> LlmCompletionResult:
    """Extract the forced tool call's arguments as JSON-encoded content.

    Raises ``RuntimeError`` when the model returned prose or stopped before the
    tool call, instead of failing on a ``KeyError``.
    """
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "tool_use":
            tool_input = block.get("input")
            if tool_input is None:
                raise RuntimeError(
                    "Anthropic tool_use block missing 'input' — cannot extract response"
                )
            usage = data.get("usage") or {}
            return LlmCompletionResult(
                content=json.dumps(tool_input),
                model=data.get("model", fallback_model),
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            )
    stop_reason = data.get("stop_reason", "unknown")
    raise RuntimeError(
        f"Anthropic response did not contain a tool_use block "
        f"(stop_reason={stop_reason!r}); check that the model supports tool-use."
    )


__all__ = [
    "_ANTHROPIC_API_VERSION",
    "build_anthropic_tool_use_body",
    "parse_anthropic_tool_use_response",
    "response_format_for",
]
