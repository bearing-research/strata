"""LLM provider configuration and resolution.

Merges server defaults, notebook env vars and the ``[ai]`` section of
``notebook.toml``. Process env vars are deliberately not consulted, so a key
exported in the server's shell does not leak into every notebook.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from typing import Any

import httpx

_PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    "ANTHROPIC_API_KEY": ("https://api.anthropic.com/v1", "claude-sonnet-4-6"),
    "OPENAI_API_KEY": ("https://api.openai.com/v1", "gpt-5.4"),
    "GEMINI_API_KEY": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3-flash",
    ),
    "MISTRAL_API_KEY": ("https://api.mistral.ai/v1", "mistral-large-latest"),
}


@dataclass(frozen=True)
class LlmConfig:
    """Resolved LLM provider configuration."""

    base_url: str
    api_key: str
    model: str
    max_output_tokens: int = 4096
    timeout_seconds: float = 60.0
    # ``None`` when ``base_url`` is trusted (operator's, provider default, or
    # service mode). Otherwise the notebook chose it on a shared server and
    # requests go through the ``@fetch`` guard; these hosts may be private.
    guard_hosts: tuple[str, ...] | None = None


@dataclass
class LlmCompletionResult:
    """Result from a chat completion request.

    ``degraded`` is True when the provider rejected the structured-output extensions
    and the call fell back to prompt-guided JSON, so schema conformance rests on
    client-side validation.
    """

    content: str
    model: str
    input_tokens: int
    output_tokens: int
    degraded: bool = False


def resolve_llm_config(
    notebook_config: dict[str, Any] | None = None,
    server_config: Any | None = None,
    notebook_env: dict[str, str] | None = None,
) -> LlmConfig | None:
    """Merge notebook ``[ai]`` config, notebook env vars and server config.

    Priority, highest first: notebook.toml ``[ai]``, notebook env vars (Runtime
    panel), server config (``STRATA_AI_*`` read at startup). Process env vars are
    not consulted. Returns ``None`` if no API key can be found.
    """
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    max_output_tokens = 4096
    timeout_seconds = 60.0

    # Layer 1 (lowest): server config
    if server_config is not None:
        if getattr(server_config, "ai_api_key", None):
            api_key = server_config.ai_api_key
        if getattr(server_config, "ai_base_url", None):
            base_url = server_config.ai_base_url
        if getattr(server_config, "ai_model", None):
            model = server_config.ai_model
        if getattr(server_config, "ai_max_output_tokens", None):
            max_output_tokens = server_config.ai_max_output_tokens
        if getattr(server_config, "ai_timeout_seconds", None):
            timeout_seconds = server_config.ai_timeout_seconds

    # Layer 2: notebook env vars. A provider-specific key selects that
    # provider's default base_url and model unless notebook.toml overrides.
    if notebook_env:
        for env_var, (default_url, default_model) in _PROVIDER_DEFAULTS.items():
            key = notebook_env.get(env_var)
            if key:
                api_key = key
                base_url = default_url
                model = default_model
                break
        else:
            # Generic key: overrides the server's key, keeps its base_url and model.
            if notebook_env.get("STRATA_AI_API_KEY"):
                api_key = notebook_env["STRATA_AI_API_KEY"]

    # Layer 3 (highest): notebook.toml [ai] section
    notebook_base_url: str | None = None
    if notebook_config:
        if notebook_config.get("api_key"):
            api_key = notebook_config["api_key"]
        if notebook_config.get("base_url"):
            base_url = notebook_base_url = notebook_config["base_url"]
        if notebook_config.get("model"):
            model = notebook_config["model"]
        if notebook_config.get("max_output_tokens"):
            max_output_tokens = int(notebook_config["max_output_tokens"])
        if notebook_config.get("timeout_seconds"):
            timeout_seconds = float(notebook_config["timeout_seconds"])

    if not api_key:
        return None

    return LlmConfig(
        base_url=base_url or "https://api.openai.com/v1",
        api_key=api_key,
        model=model or "gpt-5.4",
        max_output_tokens=max_output_tokens,
        timeout_seconds=timeout_seconds,
        guard_hosts=_base_url_guard(notebook_base_url, server_config),
    )


def _base_url_guard(notebook_base_url: str | None, server_config: Any) -> tuple[str, ...] | None:
    """The ``guard_hosts`` for a base_url ``notebook.toml`` set, if it needs one.

    A prompt cell posts to its base_url from the server and shows the answer, error
    bodies included, so on a service-mode server a notebook-chosen base_url could
    read the metadata address or an internal service; it gets the ``@fetch`` rule
    and allowlist. The operator's ``ai_base_url`` or a provider default needs no
    guard, which also keeps an environment proxy (a guarded client connects directly).
    """
    if notebook_base_url is None:
        return None
    if getattr(server_config, "deployment_mode", None) != "service":
        return None
    trusted = {url for url, _ in _PROVIDER_DEFAULTS.values()}
    operator_url = getattr(server_config, "ai_base_url", None)
    if operator_url:
        trusted.add(str(operator_url).rstrip("/"))
    if notebook_base_url.rstrip("/") in trusted:
        return None
    return tuple(getattr(server_config, "notebook_fetch_allowed_hosts", None) or ())


def max_output_tokens_param(base_url: str) -> str:
    """Return the max-output-tokens field name for this provider.

    OpenAI's gpt-5 / o-series / gpt-4o reject ``max_tokens`` and need
    ``max_completion_tokens``; other OpenAI-compatible providers accept ``max_tokens``.
    """
    if "openai" in base_url.lower():
        return "max_completion_tokens"
    return "max_tokens"


class LlmHttpError(RuntimeError):
    """Provider HTTP error carrying the status code and response body.

    Subclasses RuntimeError so broad handlers still catch it; the typed fields let
    callers decide (e.g. degrade a structured-output request) without parsing text.
    """

    def __init__(self, status_code: int, body: str, model: str):
        super().__init__(f"LLM provider returned HTTP {status_code} for model {model!r}: {body}")
        self.status_code = status_code
        self.body = body


def raise_for_llm_status(resp: httpx.Response, model: str) -> None:
    """Like ``resp.raise_for_status()`` but include the provider's error body.

    httpx's error has only URL and status; the reason ("model not found") is in the
    body.
    """
    if resp.is_success:
        return
    body = resp.text[:1000] if resp.text else "(empty body)"
    raise LlmHttpError(resp.status_code, body, model)


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


def read_notebook_ai_config(session: Any) -> dict | None:
    """The ``[ai]`` table of the session's notebook.toml, if it has one."""
    notebook_toml = session.path / "notebook.toml"
    if not notebook_toml.exists():
        return None
    try:
        with open(notebook_toml, "rb") as f:
            data = tomllib.load(f)
        ai_section = data.get("ai")
        return ai_section if isinstance(ai_section, dict) else None
    except Exception:
        return None


def llm_config_for_session(session: Any) -> LlmConfig | None:
    """Resolve the LLM config a prompt cell in *session* runs with, if any is set."""
    server_config = None
    try:
        from strata.server import get_state

        server_config = get_state().config
    except RuntimeError:
        pass

    notebook_env = getattr(session.notebook_state, "env", None) or {}

    return resolve_llm_config(read_notebook_ai_config(session), server_config, notebook_env)
