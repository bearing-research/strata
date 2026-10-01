"""LLM provider integration for Strata prompt cells.

Re-exports the public surface of ``config``, ``structured``, ``client`` and
``prompts`` so callers can import from ``strata.notebook.llm`` directly.
"""

from strata.notebook.llm.client import (
    chat_completion,
    chat_completion_stream,
)
from strata.notebook.llm.config import (
    LlmCompletionResult,
    LlmConfig,
    infer_provider_name,
    resolve_llm_config,
)
from strata.notebook.llm.prompts import (
    estimate_tokens,
    render_prompt_template,
    variable_to_text,
)
from strata.notebook.llm.structured import (
    build_anthropic_tool_use_body,
    parse_anthropic_tool_use_response,
    response_format_for,
)

__all__ = [
    # config
    "LlmConfig",
    "LlmCompletionResult",
    "infer_provider_name",
    "resolve_llm_config",
    # structured
    "build_anthropic_tool_use_body",
    "parse_anthropic_tool_use_response",
    "response_format_for",
    # client
    "chat_completion",
    "chat_completion_stream",
    # prompts
    "estimate_tokens",
    "render_prompt_template",
    "variable_to_text",
]
