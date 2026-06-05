"""LLM provider configuration and resolution.

Resolves a notebook's LLM config by merging server defaults, notebook env
vars, and the optional ``[ai]`` section in ``notebook.toml``. Process-level
env vars are deliberately *not* consulted so a key exported in the shell
that started the server doesn't leak into every notebook.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import httpx

logger = logging.getLogger(__name__)

_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "ANTHROPIC_API_KEY": ("https://api.anthropic.com/v1", "claude-sonnet-4-6"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1", "gpt-5.4"),
    "GEMINI_API_KEY": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3-flash",
    ),
    "MISTRAL_API_KEY": ("https://api.mistral.ai/v1", "mistral-large-latest"),
}

ActionType = Literal["chat"]


@dataclass(frozen=True)
class LlmConfig:
    """Resolved LLM provider configuration."""

    base_url: str
    api_key: str
    model: str
    max_context_tokens: int = 100_000
    max_output_tokens: int = 4096
    timeout_seconds: float = 60.0
    approval_timeout_seconds: float = 120.0


@dataclass
class LlmCompletionResult:
    """Result from a chat completion request."""

    content: str
    model: str
    input_tokens: int
    output_tokens: int


def resolve_llm_config(
    notebook_config: dict[str, Any] | None = None,
    server_config: Any | None = None,
    notebook_env: dict[str, str] | None = None,
) -> LlmConfig | None:
    """Merge notebook [ai] config, server config, and notebook env vars.

    Resolution order (highest priority wins):
    1. notebook.toml ``[ai]`` section
    2. Notebook-level env vars (set via the Runtime panel)
    3. Server config (``STRATA_AI_*`` env vars read at server startup)

    Process-level environment variables are **not** consulted, so a key
    accidentally exported in the shell that started the server does not
    leak into every notebook. An admin deploying a shared server can still
    provide a default via the explicit ``STRATA_AI_*`` server config.

    Returns ``None`` if no API key can be found.
    """
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    max_context_tokens = 100_000
    max_output_tokens = 4096
    timeout_seconds = 60.0
    approval_timeout_seconds = 120.0

    # Layer 1 (lowest): server config (explicit STRATA_AI_* at startup)
    if server_config is not None:
        if getattr(server_config, "ai_api_key", None):
            api_key = server_config.ai_api_key
        if getattr(server_config, "ai_base_url", None):
            base_url = server_config.ai_base_url
        if getattr(server_config, "ai_model", None):
            model = server_config.ai_model
        if getattr(server_config, "ai_max_context_tokens", None):
            max_context_tokens = server_config.ai_max_context_tokens
        if getattr(server_config, "ai_max_output_tokens", None):
            max_output_tokens = server_config.ai_max_output_tokens
        if getattr(server_config, "ai_timeout_seconds", None):
            timeout_seconds = server_config.ai_timeout_seconds
        if getattr(server_config, "ai_approval_timeout_seconds", None):
            approval_timeout_seconds = server_config.ai_approval_timeout_seconds

    # Layer 2: notebook-level env vars (from Runtime panel).
    # Setting a provider-specific key here picks up that provider's
    # default base_url and model unless the notebook.toml overrides them.
    if notebook_env:
        for env_var, (default_url, default_model) in _PROVIDER_DEFAULTS.items():
            key = notebook_env.get(env_var)
            if key:
                api_key = key
                base_url = default_url
                model = default_model
                break
        # Generic key (no implicit provider selection)
        if not api_key and notebook_env.get("STRATA_AI_API_KEY"):
            api_key = notebook_env["STRATA_AI_API_KEY"]

    # Layer 3 (highest): notebook.toml [ai] section
    if notebook_config:
        if notebook_config.get("api_key"):
            api_key = notebook_config["api_key"]
        if notebook_config.get("base_url"):
            base_url = notebook_config["base_url"]
        if notebook_config.get("model"):
            model = notebook_config["model"]
        if notebook_config.get("max_context_tokens"):
            max_context_tokens = int(notebook_config["max_context_tokens"])
        if notebook_config.get("max_output_tokens"):
            max_output_tokens = int(notebook_config["max_output_tokens"])
        if notebook_config.get("timeout_seconds"):
            timeout_seconds = float(notebook_config["timeout_seconds"])
        if notebook_config.get("approval_timeout_seconds"):
            approval_timeout_seconds = float(notebook_config["approval_timeout_seconds"])

    if not api_key:
        return None

    return LlmConfig(
        base_url=base_url or "https://api.openai.com/v1",
        api_key=api_key,
        model=model or "gpt-5.4",
        max_context_tokens=max_context_tokens,
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        approval_timeout_seconds=approval_timeout_seconds,
    )


def max_output_tokens_param(base_url: str) -> str:
    """Return the correct max-output-tokens field name for this provider.

    OpenAI's gpt-5 / o-series / gpt-4o reject ``max_tokens`` with
    "unsupported_parameter" and require ``max_completion_tokens``. Other
    OpenAI-compatible providers (Anthropic, Google, Mistral, local
    servers) still accept ``max_tokens``, so we only switch for openai.
    """
    if "openai" in base_url.lower():
        return "max_completion_tokens"
    return "max_tokens"


def raise_for_llm_status(resp: httpx.Response, model: str) -> None:
    """Like ``resp.raise_for_status()`` but surface the provider's error body.

    The default httpx error only includes URL + status; when a provider
    returns 400 with "model not found" or "invalid parameter", the
    message is in the body — which users need to debug.
    """
    if resp.is_success:
        return
    body = resp.text[:1000] if resp.text else "(empty body)"
    raise RuntimeError(f"LLM provider returned HTTP {resp.status_code} for model {model!r}: {body}")


def infer_provider_name(base_url: str) -> str:
    """Infer a human-readable provider name from the base URL."""
    url = base_url.lower()
    if "anthropic" in url:
        return "anthropic"
    if "googleapis" in url or "generativelanguage" in url:
        return "google"
    if "mistral" in url:
        return "mistral"
    if "openai" in url:
        return "openai"
    if "localhost" in url or "127.0.0.1" in url:
        return "local"
    return "custom"
