"""Live-provider LLM integration tests — opt-in, env-gated.

These hit REAL provider APIs and cost (a few cents of) money. They exist
to catch contract drift the MockTransport tests cannot: request shapes
the provider rejects, SSE framing changes, tool-use response changes.

Run them explicitly:

    STRATA_TEST_LIVE_LLM=1 ANTHROPIC_API_KEY=... OPENAI_API_KEY=... \\
        uv run pytest tests/notebook/test_llm_live.py -v

Gating, mirroring the real-backend mount tests' pattern:
- the whole module skips unless ``STRATA_TEST_LIVE_LLM`` is set;
- each provider class skips unless its API key is present.

Model overrides (default to the cheapest sensible model per provider):
``STRATA_TEST_LIVE_ANTHROPIC_MODEL``, ``STRATA_TEST_LIVE_OPENAI_MODEL``.
"""

from __future__ import annotations

import json
import os

import pytest

from strata.notebook.llm import LlmConfig
from strata.notebook.llm.client import chat_completion, chat_completion_stream

if not os.getenv("STRATA_TEST_LIVE_LLM"):
    pytest.skip(
        "Live LLM tests are opt-in: set STRATA_TEST_LIVE_LLM=1 (plus provider API keys)",
        allow_module_level=True,
    )

_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
        "confident": {"type": "boolean"},
    },
    "required": ["answer", "confident"],
    "additionalProperties": False,
}

_SCHEMA_PROMPT = (
    "What is 2 + 2? Respond with your numeric answer and whether you are confident."
)


def _anthropic_config() -> LlmConfig:
    return LlmConfig(
        base_url="https://api.anthropic.com/v1",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        model=os.getenv("STRATA_TEST_LIVE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_output_tokens=256,
    )


def _openai_config() -> LlmConfig:
    return LlmConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.environ["OPENAI_API_KEY"],
        model=os.getenv("STRATA_TEST_LIVE_OPENAI_MODEL", "gpt-4o-mini"),
        max_output_tokens=256,
    )


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
class TestAnthropicLive:
    """Anthropic: OpenAI-compat unary/streaming + native tool-use for schemas."""

    @pytest.mark.asyncio
    async def test_unary_completion(self):
        result = await chat_completion(
            _anthropic_config(),
            [{"role": "user", "content": "Reply with exactly the word: pong"}],
        )
        assert "pong" in result.content.lower()
        assert result.input_tokens > 0
        assert result.output_tokens > 0

    @pytest.mark.asyncio
    async def test_schema_via_native_tool_use(self):
        """Anthropic + schema routes through /v1/messages forced tool-use."""
        result = await chat_completion(
            _anthropic_config(),
            [{"role": "user", "content": _SCHEMA_PROMPT}],
            output_schema=_SCHEMA,
        )
        parsed = json.loads(result.content)
        assert parsed["answer"] == 4
        assert isinstance(parsed["confident"], bool)

    @pytest.mark.asyncio
    async def test_streaming_deltas_and_usage(self):
        events = [
            e
            async for e in chat_completion_stream(
                _anthropic_config(),
                [{"role": "user", "content": "Reply with exactly the word: pong"}],
            )
        ]
        deltas = [e for e in events if e["type"] == "delta"]
        done = events[-1]
        assert deltas, "no streaming deltas arrived"
        assert done["type"] == "done"
        assert "pong" in "".join(d["text"] for d in deltas).lower()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
class TestOpenAILive:
    """OpenAI: unary, strict json_schema enforcement, streaming with usage."""

    @pytest.mark.asyncio
    async def test_unary_completion(self):
        result = await chat_completion(
            _openai_config(),
            [{"role": "user", "content": "Reply with exactly the word: pong"}],
        )
        assert "pong" in result.content.lower()
        assert result.input_tokens > 0

    @pytest.mark.asyncio
    async def test_schema_via_strict_json_schema(self):
        """The full-required schema goes out as strict: true and validates."""
        result = await chat_completion(
            _openai_config(),
            [{"role": "user", "content": _SCHEMA_PROMPT}],
            output_schema=_SCHEMA,
        )
        parsed = json.loads(result.content)
        assert parsed["answer"] == 4
        assert isinstance(parsed["confident"], bool)
        assert result.degraded is False

    @pytest.mark.asyncio
    async def test_streaming_schema_and_usage(self):
        """Streaming + json_schema: deltas accumulate to a valid object and
        stream_options.include_usage delivers token counts."""
        events = [
            e
            async for e in chat_completion_stream(
                _openai_config(),
                [{"role": "user", "content": _SCHEMA_PROMPT}],
                output_schema=_SCHEMA,
            )
        ]
        done = events[-1]
        content = "".join(e["text"] for e in events if e["type"] == "delta")
        parsed = json.loads(content)
        assert parsed["answer"] == 4
        assert done["type"] == "done"
        assert done["output_tokens"] > 0
