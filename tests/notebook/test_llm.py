"""Tests for the LLM provider integration prompt cells use."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from strata.notebook.llm import (
    build_anthropic_tool_use_body,
    chat_completion,
    estimate_tokens,
    infer_provider_name,
    parse_anthropic_tool_use_response,
    render_prompt_template,
    resolve_llm_config,
    response_format_for,
)


class _FakeServerConfig:
    """Minimal server config stub for layering tests."""

    def __init__(self, **kwargs):
        self.ai_api_key = kwargs.get("ai_api_key")
        self.ai_base_url = kwargs.get("ai_base_url")
        self.ai_model = kwargs.get("ai_model")
        self.ai_max_output_tokens = kwargs.get("ai_max_output_tokens")
        self.ai_timeout_seconds = kwargs.get("ai_timeout_seconds")


class TestResolveLlmConfig:
    """``resolve_llm_config`` must not read process env vars.

    Only server config, notebook env (Runtime panel) and notebook.toml ``[ai]`` count; tests
    run with os.environ cleared.
    """

    def test_returns_none_when_no_key(self):
        with patch.dict(os.environ, {}, clear=True):
            assert resolve_llm_config() is None

    def test_process_env_is_ignored(self):
        """Shell-exported keys must not leak into notebooks."""
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_API_KEY": "sk-shell",
                "OPENAI_API_KEY": "sk-shell-openai",
                "STRATA_AI_API_KEY": "sk-shell-generic",
            },
            clear=True,
        ):
            # No notebook, no server config: process env must not rescue this.
            assert resolve_llm_config() is None

    def test_notebook_env_anthropic(self):
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_llm_config(notebook_env={"ANTHROPIC_API_KEY": "sk-ant-nb"})
            assert config is not None
            assert config.api_key == "sk-ant-nb"
            assert "anthropic" in config.base_url
            assert "claude" in config.model

    def test_notebook_env_openai(self):
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_llm_config(notebook_env={"OPENAI_API_KEY": "sk-test"})
            assert config is not None
            assert config.api_key == "sk-test"
            assert "openai" in config.base_url

    def test_notebook_env_generic_strata_key(self):
        """STRATA_AI_API_KEY in notebook env works as a generic fallback."""
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_llm_config(notebook_env={"STRATA_AI_API_KEY": "sk-generic"})
            assert config is not None
            assert config.api_key == "sk-generic"

    def test_notebook_toml_overrides_server_and_env(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-shell"}, clear=True):
            config = resolve_llm_config(
                notebook_config={
                    "api_key": "sk-from-toml",
                    "base_url": "http://localhost:11434/v1",
                    "model": "llama3",
                },
                notebook_env={"OPENAI_API_KEY": "sk-runtime"},
                server_config=_FakeServerConfig(ai_api_key="sk-server"),
            )
            assert config is not None
            assert config.api_key == "sk-from-toml"
            assert config.base_url == "http://localhost:11434/v1"
            assert config.model == "llama3"

    def test_notebook_env_overrides_server_config(self):
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_llm_config(
                notebook_env={"ANTHROPIC_API_KEY": "sk-notebook"},
                server_config=_FakeServerConfig(ai_api_key="sk-server"),
            )
            assert config is not None
            assert config.api_key == "sk-notebook"
            assert "anthropic" in config.base_url

    def test_server_config_layer(self):
        """Server config is the lowest-priority source of defaults."""
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_llm_config(
                server_config=_FakeServerConfig(
                    ai_api_key="sk-server",
                    ai_base_url="https://custom.api.com/v1",
                    ai_model="custom-model",
                    ai_max_output_tokens=2048,
                    ai_timeout_seconds=30.0,
                )
            )
            assert config is not None
            assert config.api_key == "sk-server"
            assert config.base_url == "https://custom.api.com/v1"
            assert config.model == "custom-model"
            assert config.max_output_tokens == 2048


class TestInferProviderName:
    """Provider name inference."""

    def test_anthropic(self):
        assert infer_provider_name("https://api.anthropic.com/v1") == "anthropic"

    def test_openai(self):
        assert infer_provider_name("https://api.openai.com/v1") == "openai"

    def test_google(self):
        assert (
            infer_provider_name("https://generativelanguage.googleapis.com/v1beta/openai")
            == "google"
        )

    def test_local(self):
        assert infer_provider_name("http://localhost:11434/v1") == "local"

    def test_custom(self):
        assert infer_provider_name("https://my-company.com/llm/v1") == "custom"


class TestResponseFormatFor:
    """Pick the right provider-native structured-output payload."""

    _SCHEMA = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }

    def test_openai_with_schema_uses_json_schema(self):
        rf = response_format_for(
            "https://api.openai.com/v1",
            output_type="json",
            output_schema=self._SCHEMA,
        )
        # OpenAI strict mode demands ``additionalProperties: false`` on every object;
        # it is injected automatically.
        expected_schema = {
            **self._SCHEMA,
            "additionalProperties": False,
        }
        assert rf == {
            "type": "json_schema",
            "json_schema": {
                "name": "PromptResponse",
                "schema": expected_schema,
                "strict": True,
            },
        }

    def test_openai_nested_objects_get_additional_properties_injected(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"label": {"type": "string"}},
                        "required": ["label"],
                    },
                },
            },
            "required": ["items"],
        }
        rf = response_format_for(
            "https://api.openai.com/v1",
            output_type="json",
            output_schema=schema,
        )
        normalized = rf["json_schema"]["schema"]
        assert normalized["additionalProperties"] is False
        assert normalized["properties"]["items"]["items"]["additionalProperties"] is False
        assert rf["json_schema"]["strict"] is True

    def test_openai_incomplete_required_falls_back_to_non_strict(self):
        """Strict mode would reject an optional field, so strict is turned off instead."""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number"},
                "comment": {"type": "string"},
            },
            "required": ["score"],  # "comment" intentionally omitted
        }
        rf = response_format_for(
            "https://api.openai.com/v1",
            output_type="json",
            output_schema=schema,
        )
        assert rf["json_schema"]["strict"] is False
        assert rf["json_schema"]["schema"]["additionalProperties"] is False

    def test_anthropic_with_schema_falls_back_to_json_object(self):
        rf = response_format_for(
            "https://api.anthropic.com/v1",
            output_type="json",
            output_schema=self._SCHEMA,
        )
        assert rf == {"type": "json_object"}

    def test_plain_json_without_schema_uses_json_object(self):
        rf = response_format_for(
            "https://api.openai.com/v1",
            output_type="json",
            output_schema=None,
        )
        assert rf == {"type": "json_object"}

    def test_text_output_returns_none(self):
        assert (
            response_format_for(
                "https://api.openai.com/v1",
                output_type="text",
                output_schema=None,
            )
            is None
        )


class TestEstimateTokens:
    """Token estimation."""

    def test_basic(self):
        assert estimate_tokens("hello world") > 0

    def test_empty(self):
        assert estimate_tokens("") == 1

    def test_proportional(self):
        short = estimate_tokens("hello")
        long = estimate_tokens("hello " * 100)
        assert long > short


class TestRenderPromptTemplate:
    """Safe prompt template rendering."""

    def test_renders_attribute_access_without_eval(self):
        variables = {"obj": SimpleNamespace(value=42)}

        rendered = render_prompt_template("Value: {{ obj.value }}", variables)

        assert rendered == "Value: 42"

    def test_blocks_side_effecting_method_calls(self):
        class _Mutating:
            def __init__(self) -> None:
                self.called = False

            def mutate(self) -> str:
                self.called = True
                return "changed"

        value = _Mutating()

        rendered = render_prompt_template("Unsafe: {{ obj.mutate() }}", {"obj": value})

        assert rendered == "Unsafe: {{ obj.mutate() }}"
        assert value.called is False


_SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}


class TestAnthropicToolUseBody:
    """The native Anthropic request body."""

    def test_body_has_forced_tool_choice(self):
        body = build_anthropic_tool_use_body(
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1024,
            temperature=0.0,
            output_schema=_SIMPLE_SCHEMA,
        )
        assert body["tool_choice"] == {"type": "tool", "name": "respond"}
        assert body["tools"][0]["input_schema"] == _SIMPLE_SCHEMA
        assert body["tools"][0]["name"] == "respond"
        assert body["max_tokens"] == 1024
        assert body["temperature"] == 0.0

    def test_system_message_lifted_to_top_level(self):
        body = build_anthropic_tool_use_body(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "You are an extractor."},
                {"role": "user", "content": "Extract from: foo"},
            ],
            max_tokens=256,
            temperature=None,
            output_schema=_SIMPLE_SCHEMA,
        )
        assert body["system"] == "You are an extractor."
        # System is pulled out of messages: the native API rejects role=system inside
        # the messages array.
        assert all(m["role"] != "system" for m in body["messages"])
        assert len(body["messages"]) == 1
        assert "temperature" not in body

    def test_multiple_system_messages_are_joined(self):
        body = build_anthropic_tool_use_body(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": "Rule one."},
                {"role": "system", "content": "Rule two."},
                {"role": "user", "content": "Go"},
            ],
            max_tokens=128,
            temperature=None,
            output_schema=_SIMPLE_SCHEMA,
        )
        assert body["system"] == "Rule one.\n\nRule two."


class TestAnthropicToolUseParse:
    """Extracting the forced tool call from the response."""

    def test_extracts_tool_use_input_as_json(self):
        data = {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "respond",
                    "input": {"answer": "42"},
                }
            ],
            "model": "claude-sonnet-4-6",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = parse_anthropic_tool_use_response(data, fallback_model="claude")
        import json as _json

        assert _json.loads(result.content) == {"answer": "42"}
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model == "claude-sonnet-4-6"

    def test_raises_when_no_tool_use_block(self):
        data = {
            "content": [{"type": "text", "text": "sorry, no tool"}],
            "stop_reason": "end_turn",
        }
        with pytest.raises(RuntimeError, match="tool_use block"):
            parse_anthropic_tool_use_response(data, fallback_model="claude")

    def test_skips_non_tool_use_blocks_before_tool_use(self):
        """Real responses often start with a text block before the tool call."""
        data = {
            "content": [
                {"type": "text", "text": "Thinking..."},
                {"type": "tool_use", "name": "respond", "input": {"answer": "ok"}},
            ],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        result = parse_anthropic_tool_use_response(data, fallback_model="claude")
        import json as _json

        assert _json.loads(result.content) == {"answer": "ok"}


class TestChatCompletionDispatch:
    """Anthropic + schema hits /v1/messages; everything else hits /v1/chat/completions."""

    @pytest.mark.asyncio
    async def test_anthropic_with_schema_routes_to_messages_endpoint(self, monkeypatch):
        import httpx

        from strata.notebook.llm import LlmConfig

        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "respond",
                            "input": {"answer": "42"},
                        }
                    ],
                    "usage": {"input_tokens": 5, "output_tokens": 3},
                    "model": "claude-sonnet-4-6",
                },
            )

        transport = httpx.MockTransport(handler)
        original_client = httpx.AsyncClient

        def patched_client(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_client)

        config = LlmConfig(
            base_url="https://api.anthropic.com/v1",
            api_key="dummy",
            model="claude-sonnet-4-6",
        )
        result = await chat_completion(
            config,
            [{"role": "user", "content": "hi"}],
            output_type="json",
            output_schema=_SIMPLE_SCHEMA,
        )

        assert captured_urls == ["https://api.anthropic.com/v1/messages"]
        import json as _json

        assert _json.loads(result.content) == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_anthropic_without_schema_uses_openai_compat(self, monkeypatch):
        import httpx

        from strata.notebook.llm import LlmConfig

        captured_urls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured_urls.append(str(request.url))
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    "model": "claude-sonnet-4-6",
                },
            )

        transport = httpx.MockTransport(handler)
        original_client = httpx.AsyncClient

        def patched_client(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_client)

        config = LlmConfig(
            base_url="https://api.anthropic.com/v1",
            api_key="dummy",
            model="claude-sonnet-4-6",
        )
        await chat_completion(config, [{"role": "user", "content": "hi"}])

        assert captured_urls == ["https://api.anthropic.com/v1/chat/completions"]


class TestChatCompletionStreamRequestShape:
    """``chat_completion_stream`` sends ``temperature`` and ``response_format`` like the unary path.

    Both stay off the body when not requested.
    """

    _SSE_BODY = (
        b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"lo"}}],'
        b'"usage":{"prompt_tokens":5,"completion_tokens":2},"model":"m-live"}\n\n'
        b"data: [DONE]\n\n"
    )

    def _patch_transport(self, monkeypatch, captured_bodies: list[dict]):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            import json as _json

            captured_bodies.append(_json.loads(request.content))
            return httpx.Response(
                200,
                content=self._SSE_BODY,
                headers={"Content-Type": "text/event-stream"},
            )

        transport = httpx.MockTransport(handler)
        original_client = httpx.AsyncClient

        def patched_client(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    @pytest.mark.asyncio
    async def test_schema_and_temperature_land_in_body(self, monkeypatch):
        from strata.notebook.llm import LlmConfig, chat_completion_stream

        captured: list[dict] = []
        self._patch_transport(monkeypatch, captured)

        config = LlmConfig(base_url="https://api.openai.com/v1", api_key="k", model="m")
        events = [
            event
            async for event in chat_completion_stream(
                config,
                [{"role": "user", "content": "hi"}],
                temperature=0.2,
                output_type="json",
                output_schema=_SIMPLE_SCHEMA,
            )
        ]

        body = captured[0]
        assert body["stream"] is True
        assert body["temperature"] == 0.2
        # OpenAI base URL + schema → json_schema response_format, same
        # as the unary dispatcher.
        assert body["response_format"]["type"] == "json_schema"

        # Deltas accumulate to the full content; usage comes from the
        # final chunk.
        deltas = [e["text"] for e in events if e["type"] == "delta"]
        assert "".join(deltas) == "hello"
        done = events[-1]
        assert done["type"] == "done"
        assert done["model"] == "m-live"
        assert done["input_tokens"] == 5
        assert done["output_tokens"] == 2

    @pytest.mark.asyncio
    async def test_defaults_leave_body_unchanged(self, monkeypatch):
        """No kwargs means no ``temperature`` / ``response_format`` keys in the body."""
        from strata.notebook.llm import LlmConfig, chat_completion_stream

        captured: list[dict] = []
        self._patch_transport(monkeypatch, captured)

        config = LlmConfig(base_url="https://api.openai.com/v1", api_key="k", model="m")
        async for _event in chat_completion_stream(config, [{"role": "user", "content": "hi"}]):
            pass

        body = captured[0]
        assert "temperature" not in body
        assert "response_format" not in body
        assert body["stream"] is True


class TestStructuredOutputDegradation:
    """Providers that reject response_format degrade to prompt-guided JSON."""

    def _patch_transport(self, monkeypatch, handler):
        import httpx

        transport = httpx.MockTransport(handler)
        original_client = httpx.AsyncClient

        def patched_client(*args, **kwargs):
            kwargs["transport"] = transport
            return original_client(*args, **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", patched_client)

    def _config(self):
        from strata.notebook.llm import LlmConfig

        return LlmConfig(
            base_url="http://localhost:11434/v1",  # OpenAI-compat local server
            api_key="dummy",
            model="local-model",
        )

    @pytest.mark.asyncio
    async def test_unary_degrades_on_response_format_rejection(self, monkeypatch):
        import httpx

        bodies: list[dict] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "response_format" in body:
                return httpx.Response(400, json={"error": "response_format is not supported"})
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": '{"answer": "42"}'}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 2},
                    "model": "local-model",
                },
            )

        self._patch_transport(monkeypatch, handler)

        result = await chat_completion(
            self._config(),
            [{"role": "user", "content": "hi"}],
            output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
        )

        # First call carried response_format; the retry dropped it and
        # appended a schema-guidance system turn.
        assert "response_format" in bodies[0]
        assert "response_format" not in bodies[1]
        guidance = bodies[1]["messages"][-1]
        assert guidance["role"] == "system"
        assert "JSON Schema" in guidance["content"]
        assert result.degraded is True
        assert json.loads(result.content) == {"answer": "42"}

    @pytest.mark.asyncio
    async def test_unary_unrelated_400_still_raises(self, monkeypatch):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": "model not found"})

        self._patch_transport(monkeypatch, handler)

        with pytest.raises(RuntimeError, match="model not found"):
            await chat_completion(
                self._config(),
                [{"role": "user", "content": "hi"}],
                output_schema={"type": "object"},
            )

    @pytest.mark.asyncio
    async def test_stream_degrades_with_notice(self, monkeypatch):
        import httpx

        from strata.notebook.llm.client import chat_completion_stream

        bodies: list[dict] = []
        sse = (
            'data: {"choices": [{"delta": {"content": "{\\"answer\\": \\"42\\"}"}}], '
            '"model": "local-model"}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            bodies.append(body)
            if "response_format" in body or "stream_options" in body:
                return httpx.Response(400, json={"error": "stream_options unsupported"})
            return httpx.Response(
                200,
                content=sse.encode(),
                headers={"content-type": "text/event-stream"},
            )

        self._patch_transport(monkeypatch, handler)

        events = [
            e
            async for e in chat_completion_stream(
                self._config(),
                [{"role": "user", "content": "hi"}],
                output_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
            )
        ]

        kinds = [e["type"] for e in events]
        assert kinds[0] == "notice"
        assert "delta" in kinds and kinds[-1] == "done"
        assert events[-1]["degraded"] is True
        # degraded retry dropped both extensions
        assert "response_format" not in bodies[1]
        assert "stream_options" not in bodies[1]

    @pytest.mark.asyncio
    async def test_stream_happy_path_unchanged(self, monkeypatch):
        import httpx

        from strata.notebook.llm.client import chat_completion_stream

        sse = (
            'data: {"choices": [{"delta": {"content": "hello"}}], "model": "m"}\n\ndata: [DONE]\n\n'
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, content=sse.encode(), headers={"content-type": "text/event-stream"}
            )

        self._patch_transport(monkeypatch, handler)

        events = [
            e
            async for e in chat_completion_stream(
                self._config(), [{"role": "user", "content": "hi"}]
            )
        ]
        assert [e["type"] for e in events] == ["delta", "done"]
        assert events[-1].get("degraded", False) is False


class TestCoerceJsonText:
    """Lenient JSON extraction before schema validation."""

    def test_clean_json_passthrough(self):
        from strata.notebook.prompt_executor import _coerce_json_text

        assert _coerce_json_text('{"a": 1}') == '{"a": 1}'

    def test_fenced_json_extracted(self):
        from strata.notebook.prompt_executor import _coerce_json_text

        content = 'Here you go:\n```json\n{"a": 1}\n```\nHope that helps!'
        assert _coerce_json_text(content) == '{"a": 1}'

    def test_prose_wrapped_object_extracted(self):
        from strata.notebook.prompt_executor import _coerce_json_text

        content = 'Sure! The result is {"a": 1} as requested.'
        assert _coerce_json_text(content) == '{"a": 1}'

    def test_top_level_array_extracted(self):
        from strata.notebook.prompt_executor import _coerce_json_text

        content = "The list: [1, 2, 3]."
        assert _coerce_json_text(content) == "[1, 2, 3]"

    def test_unparseable_returned_verbatim(self):
        from strata.notebook.prompt_executor import _coerce_json_text

        assert _coerce_json_text("no json here {broken") == "no json here {broken"
