"""Chat completion HTTP client (sync + streaming).

Single entry point ``chat_completion`` picks between Anthropic native
``/v1/messages`` (when schema-constrained) and the generic OpenAI-compat
``/v1/chat/completions``. ``chat_completion_stream`` is OpenAI-compat only
and yields text deltas suitable for surfacing intermediate output.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from strata.notebook.llm.config import (
    LlmCompletionResult,
    LlmConfig,
    infer_provider_name,
    max_output_tokens_param,
    raise_for_llm_status,
)
from strata.notebook.llm.structured import (
    _ANTHROPIC_API_VERSION,
    build_anthropic_tool_use_body,
    parse_anthropic_tool_use_response,
    response_format_for,
)


async def _chat_completion_anthropic_native(
    config: LlmConfig,
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    output_schema: dict[str, Any],
) -> LlmCompletionResult:
    """Post the native Anthropic ``/v1/messages`` tool-use request."""
    body = build_anthropic_tool_use_body(
        model=config.model,
        messages=messages,
        max_tokens=config.max_output_tokens,
        temperature=temperature,
        output_schema=output_schema,
    )
    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        resp = await client.post(
            f"{config.base_url.rstrip('/')}/messages",
            headers={
                "x-api-key": config.api_key,
                "anthropic-version": _ANTHROPIC_API_VERSION,
                "Content-Type": "application/json",
            },
            json=body,
        )
        raise_for_llm_status(resp, config.model)
        data = resp.json()
    return parse_anthropic_tool_use_response(data, fallback_model=config.model)


async def _chat_completion_openai_compat(
    config: LlmConfig,
    messages: list[dict[str, str]],
    *,
    temperature: float | None,
    response_format: dict[str, Any] | None,
) -> LlmCompletionResult:
    """Post to the OpenAI-compatible ``/v1/chat/completions`` endpoint."""
    body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        max_output_tokens_param(config.base_url): config.max_output_tokens,
    }
    if temperature is not None:
        body["temperature"] = temperature
    if response_format is not None:
        body["response_format"] = response_format

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        resp = await client.post(
            f"{config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
        )
        raise_for_llm_status(resp, config.model)
        data = resp.json()

    choice = data["choices"][0]
    usage = data.get("usage", {})
    return LlmCompletionResult(
        content=choice["message"]["content"],
        model=data.get("model", config.model),
        input_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
    )


async def chat_completion(
    config: LlmConfig,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    output_type: str | None = None,
    output_schema: dict[str, Any] | None = None,
) -> LlmCompletionResult:
    """Send a chat completion, picking the best provider path.

    * Anthropic + schema → native ``/v1/messages`` tool-use (schema
      enforcement unavailable on their OpenAI-compat endpoint).
    * Everything else → OpenAI-compatible ``/v1/chat/completions`` with
      a provider-appropriate ``response_format``.

    Callers that don't care about structured output can omit both
    ``output_type`` and ``output_schema``.
    """
    if output_schema is not None and infer_provider_name(config.base_url) == "anthropic":
        return await _chat_completion_anthropic_native(
            config,
            messages,
            temperature=temperature,
            output_schema=output_schema,
        )
    response_format = response_format_for(
        config.base_url,
        output_type=output_type,
        output_schema=output_schema,
    )
    return await _chat_completion_openai_compat(
        config,
        messages,
        temperature=temperature,
        response_format=response_format,
    )


async def chat_completion_stream(
    config: LlmConfig,
    messages: list[dict[str, str]],
    *,
    temperature: float | None = None,
    output_type: str | None = None,
    output_schema: dict[str, Any] | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a chat completion as text deltas.

    Yields dicts of the form ``{"type": "delta", "text": str}`` for content
    chunks and a final ``{"type": "done", "model": str, "input_tokens": int,
    "output_tokens": int}`` event when the stream ends. The OpenAI-compatible
    API returns ``data: ...`` SSE lines; we parse them and pull
    ``choices[0].delta.content``.

    ``temperature`` / ``output_type`` / ``output_schema`` shape the request
    body exactly like the unary ``_chat_completion_openai_compat`` path —
    OpenAI supports ``response_format`` (including ``json_schema``) with
    ``stream: true``; the deltas are simply the raw JSON text as it
    generates. This function is OpenAI-compat only: the Anthropic native
    tool-use path (Anthropic + schema) has no streaming equivalent here
    yet, so callers that need it must fall back to ``chat_completion``
    (see ``execute_prompt_cell``).
    """
    model = config.model
    input_tokens = 0
    output_tokens = 0

    request_body: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        max_output_tokens_param(config.base_url): config.max_output_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        request_body["temperature"] = temperature
    response_format = response_format_for(
        config.base_url,
        output_type=output_type,
        output_schema=output_schema,
    )
    if response_format is not None:
        request_body["response_format"] = response_format

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        async with client.stream(
            "POST",
            f"{config.base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
            json=request_body,
        ) as resp:
            if not resp.is_success:
                body = (await resp.aread()).decode("utf-8", errors="replace")[:1000]
                raise RuntimeError(
                    f"LLM provider returned HTTP {resp.status_code} "
                    f"for model {config.model!r}: {body}"
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
                delta = choices[0].get("delta") or {}
                text = delta.get("content")
                if text:
                    yield {"type": "delta", "text": text}

    yield {
        "type": "done",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }
