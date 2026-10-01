"""Provider registry: name to cached ``SecretProvider`` instance.

Instances are cached per process; providers hold no state beyond an HTTP
client they build on demand.
"""

from __future__ import annotations

from strata.notebook.secret_manager.provider import SecretProvider, SecretProviderError

_cache: dict[str, SecretProvider] = {}


def get_provider(name: str) -> SecretProvider:
    """Return the provider named ``name``, constructing on first use.

    Raises ``SecretProviderError`` for unknown names, so a ``notebook.toml`` typo
    surfaces at session open rather than as a silent empty fetch.
    """
    if name in _cache:
        return _cache[name]
    provider = _build(name)
    _cache[name] = provider
    return provider


def _build(name: str) -> SecretProvider:
    if name == "infisical":
        from strata.notebook.secret_manager.infisical import InfisicalProvider

        return InfisicalProvider()
    raise SecretProviderError(f"Unknown secret provider: {name!r}")


def _reset_for_tests() -> None:
    """Clear the provider cache (tests that install mocks)."""
    _cache.clear()
