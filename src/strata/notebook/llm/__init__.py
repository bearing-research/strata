"""LLM provider integration for Strata prompt cells.

Submodule layout:
* ``config``      — ``LlmConfig``, ``resolve_llm_config``, provider helpers
* ``structured``  — ``response_format_for`` and Anthropic tool-use builders
* ``client``      — ``chat_completion`` and ``chat_completion_stream``
* ``prompts``     — ``variable_to_text``, ``render_prompt_template``

The package re-exports the public surface so callers can import from
``strata.notebook.llm`` directly (e.g. ``LlmConfig``, ``chat_completion``).
"""

from strata.notebook.llm.client import (
    chat_completion,
    chat_completion_stream,
)
from strata.notebook.llm.config import (
    ActionType,
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
    "ActionType",
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
