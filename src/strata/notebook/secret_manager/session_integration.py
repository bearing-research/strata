"""Glue between ``SessionManager`` and the secret-provider layer.

Keeps the merge precedence between fetched secrets and user-typed env values in
one place. Session code fetches on open and on refresh only, and re-merges the
last fetch with :func:`merge_secrets_into_state` when it reloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from strata.notebook.secret_manager.provider import (
    SECRET_FETCH_TIMEOUT_SECONDS,
    SecretFetchResult,
)
from strata.notebook.secret_manager.registry import get_provider
from strata.notebook.writer import _is_sensitive_env_key

if TYPE_CHECKING:
    from strata.notebook.models import NotebookState


MANUAL_SOURCE = "manual"

# Sent to clients in place of a secret value; sent back, it means "unchanged".
MASKED_ENV_VALUE = "__strata_masked__"


def mask_env(env: Mapping[str, str], sources: Mapping[str, str]) -> dict[str, str]:
    """Return ``env`` with secret values replaced by ``MASKED_ENV_VALUE``.

    A value is secret when its name looks sensitive or a secret manager supplied
    it. Empty values stay empty, so a client can tell an unset key from a set one.
    """
    return {
        key: (
            MASKED_ENV_VALUE
            if value
            and (_is_sensitive_env_key(key) or sources.get(key, MANUAL_SOURCE) != MANUAL_SOURCE)
            else value
        )
        for key, value in env.items()
    }


def unmask_env(submitted: Mapping[str, str], current: Mapping[str, str]) -> dict[str, str]:
    """Resolve ``MASKED_ENV_VALUE`` entries in a client's env to the current values.

    A masked key with no current value is dropped.
    """
    return {
        key: current[key] if value == MASKED_ENV_VALUE else value
        for key, value in submitted.items()
        if value != MASKED_ENV_VALUE or key in current
    }


def fetch_configured_secrets(
    state: NotebookState, *, timeout: float = SECRET_FETCH_TIMEOUT_SECONDS
) -> SecretFetchResult | None:
    """Return the fetch result for ``state``'s configured provider.

    ``None`` when the notebook has no ``[secret_manager]`` block. Never raises: an
    unknown provider name or a failing constructor yields a ``SecretFetchResult``
    with ``error`` set.
    """
    config = state.secret_manager_config
    if not config:
        return None
    provider_name = str(config.get("provider") or "").strip().lower()
    if not provider_name:
        return SecretFetchResult.failure(
            "",
            "[secret_manager] block is present but 'provider' is not set — "
            'add provider = "infisical".',
        )
    try:
        provider = get_provider(provider_name)
    except Exception as exc:  # SecretProviderError or anything weirder
        return SecretFetchResult.failure(provider_name, str(exc))
    try:
        return provider.fetch(dict(config), timeout=timeout)
    except Exception as exc:
        # The protocol says fetch should not raise, but a buggy provider
        # shouldn't take the session down.
        return SecretFetchResult.failure(provider_name, f"provider raised: {exc}")


def apply_secrets_to_notebook_state(
    state: NotebookState, *, timeout: float = SECRET_FETCH_TIMEOUT_SECONDS
) -> SecretFetchResult | None:
    """Fetch secrets and merge them into ``state.env`` in place."""
    result = fetch_configured_secrets(state, timeout=timeout)
    merge_secrets_into_state(state, result)
    return result


def merge_secrets_into_state(state: NotebookState, result: SecretFetchResult | None) -> None:
    """Merge a fetch *result* (``None``: no provider) into ``state.env`` in place.

    A fetched secret fills a key that is absent, blank (sensitive values are
    blanked on disk), or was filled by the provider last time (so rotation is
    picked up). A non-empty value the user set this session wins. Always stamps
    ``env_sources`` so every key has an origin label for the UI.
    """
    # Provider-filled values are the provider's, so a refresh replaces them;
    # otherwise a rotated secret would keep its old value as if set by hand.
    fetched_before = {
        key for key, source in (state.env_sources or {}).items() if source != MANUAL_SOURCE
    }
    state.env_sources = {key: MANUAL_SOURCE for key in state.env}

    if result is None:
        state.env_fetch_error = None
        state.env_fetched_at = None
        return

    state.env_fetched_at = result.fetched_at
    state.env_fetch_error = result.error

    for key, value in result.secrets.items():
        existing = state.env.get(key)
        if existing is None or existing == "" or key in fetched_before:
            state.env[key] = value
            state.env_sources[key] = result.source
        # else: manual override wins.
