"""Chat completion HTTP client (sync + streaming).

Single entry point ``chat_completion`` picks between Anthropic native
``/v1/messages`` (when schema-constrained) and the generic OpenAI-compat
``/v1/chat/completions``. ``chat_completion_stream`` is OpenAI-compat only
and yields text deltas suitable for surfacing intermediate output.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from strata.notebook.llm.config import (
    LlmCompletionResult,
    LlmConfig,
    LlmHttpError,
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

logger = logging.getLogger(__name__)


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


def _is_structured_output_rejection(status_code: int, body: str) -> bool:
    """True when a 4xx names the optional request extensions we sent.

    Some OpenAI-compatible servers (older Ollama, llama.cpp wrappers,
    strict proxies) reject ``response_format`` or ``stream_options``
    outright instead of ignoring them. Those calls are recoverable by
    retrying without the extensions.
    """
    if status_code not in (400, 422):
        return False
    lowered = body.lower()
    return any(
        marker in lowered
        for marker in ("response_format", "json_schema", "json_object", "stream_options")
    )


def _json_guidance_message(
    output_schema: dict[str, Any] | None,
    output_type: str | None,
) -> dict[str, str] | None:
    """System turn steering the model toward JSON when enforcement is gone.

    Used on the degraded path: the provider refused ``response_format``,
    so shape guidance must travel in the prompt and conformance rests on
    the client-side validation loop.
    """
    if output_schema is not None:
        return {
            "role": "system",
            "content": (
                "Respond with ONLY a valid JSON object — no prose, no code "
                "fences. The object must conform to this JSON Schema:\n" + json.dumps(output_schema)
            ),
        }
    if output_type == "json":
        return {
            "role": "system",
            "content": "Respond with ONLY a valid JSON object — no prose, no code fences.",
        }
    return None


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
    try:
        return await _chat_completion_openai_compat(
            config,
            messages,
            temperature=temperature,
            response_format=response_format,
        )
    except LlmHttpError as e:
        if response_format is None or not _is_structured_output_rejection(e.status_code, e.body):
            raise
        # Provider refused the structured-output extension — degrade to
        # prompt-guided JSON; the caller's validation loop enforces the
        # schema client-side.
        logger.warning(
            "Provider rejected response_format (HTTP %d); degrading to "
            "prompt-guided JSON for model %s",
            e.status_code,
            config.model,
        )
        guidance = _json_guidance_message(output_schema, output_type)
        degraded_messages = messages + [guidance] if guidance else messages
        result = await _chat_completion_openai_compat(
            config,
            degraded_messages,
            temperature=temperature,
            response_format=None,
        )
        result.degraded = True
        return result


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
    "output_tokens": int}`` event when the stream ends.

    ``temperature`` / ``output_type`` / ``output_schema`` shape the request
    body exactly like the unary ``_chat_completion_openai_compat`` path.
    This function is OpenAI-compat only: the Anthropic native tool-use path
    (Anthropic + schema) has no streaming equivalent here yet, so callers
    that need it must fall back to ``chat_completion``.

    Degradation: providers that reject ``response_format`` /
    ``stream_options`` with a 4xx get one retry without the extensions,
    with a schema-guidance system turn appended. The retry is announced
    with a ``{"type": "notice"}`` event and its ``done`` event carries
    ``degraded: True`` — schema conformance then rests on the caller's
    validation loop.
    """
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

    events = _stream_openai_compat(config, request_body)
    try:
        first = await anext(events)
    except StopAsyncIteration:
        return
    except LlmHttpError as e:
        if not _is_structured_output_rejection(e.status_code, e.body):
            raise
        logger.warning(
            "Provider rejected structured-output extensions (HTTP %d); "
            "degrading to prompt-guided streaming for model %s",
            e.status_code,
            config.model,
        )
        yield {
            "type": "notice",
            "text": (
                "Provider rejected structured-output request extensions; "
                "falling back to prompt-guided JSON."
            ),
        }
        degraded_body: dict[str, Any] = {
            "model": config.model,
            "messages": messages,
            max_output_tokens_param(config.base_url): config.max_output_tokens,
            "stream": True,
        }
        if temperature is not None:
            degraded_body["temperature"] = temperature
        guidance = _json_guidance_message(output_schema, output_type)
        if guidance is not None:
            degraded_body["messages"] = messages + [guidance]
        async for event in _stream_openai_compat(config, degraded_body):
            if event["type"] == "done":
                event["degraded"] = True
            yield event
        return

    yield first
    async for event in events:
        yield event


async def _stream_openai_compat(
    config: LlmConfig,
    request_body: dict[str, Any],
) -> AsyncIterator[dict[str, Any]]:
    """Open one SSE stream and yield delta events plus a final done event.

    Raises ``LlmHttpError`` (status + body) before the first yield when
    the provider refuses the request, so callers can decide whether the
    failure is recoverable.
    """
    model = request_body.get("model", config.model)
    input_tokens = 0
    output_tokens = 0

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
                raise LlmHttpError(resp.status_code, body, str(request_body.get("model")))
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
