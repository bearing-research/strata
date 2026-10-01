"""External secret-manager integration for notebook env vars.

A ``[secret_manager]`` section in ``notebook.toml`` names a provider (Infisical,
via service token); on session open its secrets are merged into the notebook's
env map. Values already in ``[env]`` win over fetched ones.
"""

from __future__ import annotations

from strata.notebook.secret_manager.provider import (
    SecretFetchResult,
    SecretProvider,
    SecretProviderError,
)
from strata.notebook.secret_manager.registry import get_provider
from strata.notebook.secret_manager.session_integration import (
    apply_secrets_to_notebook_state,
    fetch_configured_secrets,
)

__all__ = [
    "SecretFetchResult",
    "SecretProvider",
    "SecretProviderError",
    "apply_secrets_to_notebook_state",
    "fetch_configured_secrets",
    "get_provider",
]
