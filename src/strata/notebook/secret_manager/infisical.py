"""Infisical secret-manager integration via the official Python SDK.

Authentication precedence (highest wins):

1. **Universal Auth / Machine Identity** —
   ``INFISICAL_CLIENT_ID`` + ``INFISICAL_CLIENT_SECRET`` in the process
   environment. This is the path Infisical recommends; service tokens
   are being deprecated upstream.

2. **Service / access token** — ``INFISICAL_TOKEN`` in the process
   environment. Kept for backward compatibility so users with an
   existing service-token setup don't have to migrate immediately.

If neither is set the provider returns a failure with a clear message
pointing the user at both options.

Project routing (``project_id``, ``environment``, ``path``) comes
from the notebook's ``[secret_manager]`` block so the non-sensitive info
can be committed. Override via env vars (``INFISICAL_PROJECT_ID``
etc.) is supported for quick-start use.

The host is where the server's credentials are sent. On a service-mode server
it is the operator's alone (``INFISICAL_HOST``, else the public default): a
notebook's ``base_url`` naming anywhere else is refused before any login.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from strata.notebook.harness_user import running_server_config
from strata.notebook.secret_manager.provider import SecretFetchResult, _now_iso

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "https://app.infisical.com"
_DEFAULT_ENVIRONMENT = "dev"
_DEFAULT_PATH = "/"


class InfisicalProvider:
    """Pulls secrets from an Infisical project using the official SDK."""

    name = "infisical"

    def fetch(self, config: dict[str, Any]) -> SecretFetchResult:
        project_id = config.get("project_id") or os.environ.get("INFISICAL_PROJECT_ID")
        if not project_id:
            return SecretFetchResult.failure(
                self.name,
                "project_id missing — set it in notebook.toml [secret_manager] "
                "or via INFISICAL_PROJECT_ID.",
            )

        environment = (
            config.get("environment")
            or os.environ.get("INFISICAL_ENVIRONMENT")
            or _DEFAULT_ENVIRONMENT
        )
        secret_path = config.get("path") or os.environ.get("INFISICAL_PATH") or _DEFAULT_PATH
        operator_host = (os.environ.get("INFISICAL_HOST") or _DEFAULT_HOST).rstrip("/")
        notebook_host = str(config.get("base_url") or "").rstrip("/")
        host = notebook_host or operator_host
        if (
            host != operator_host
            and getattr(running_server_config(), "deployment_mode", None) == "service"
        ):
            # The login sends the server's machine identity to *host*; on a shared
            # server the notebook's author is not the operator.
            return SecretFetchResult.failure(
                self.name,
                f"[secret_manager] base_url {notebook_host!r} is refused on this server: "
                "it would receive the server's Infisical credentials. The server logs in "
                f"only at {operator_host} (INFISICAL_HOST); remove base_url from "
                "notebook.toml, or ask the operator to set INFISICAL_HOST.",
            )

        client_id = os.environ.get("INFISICAL_CLIENT_ID")
        client_secret = os.environ.get("INFISICAL_CLIENT_SECRET")
        token = os.environ.get("INFISICAL_TOKEN")
        if not ((client_id and client_secret) or token):
            return SecretFetchResult.failure(
                self.name,
                "No Infisical credentials in the process environment. Set either "
                "INFISICAL_CLIENT_ID + INFISICAL_CLIENT_SECRET (Machine Identity / "
                "Universal Auth — recommended) or INFISICAL_TOKEN (service token, "
                "legacy) in the shell that launched Strata.",
            )

        try:
            from infisical_sdk import InfisicalSDKClient
        except ImportError as exc:
            return SecretFetchResult.failure(
                self.name,
                f"infisicalsdk not installed ({exc}). Run `uv sync` and restart the server.",
            )

        client = InfisicalSDKClient(host=host)
        try:
            if client_id and client_secret:
                client.auth.universal_auth.login(
                    client_id=client_id,
                    client_secret=client_secret,
                )
            else:
                client.auth.token_auth.login(token=token or "")
        except Exception as exc:
            return SecretFetchResult.failure(
                self.name,
                f"Infisical authentication failed: {exc}",
            )

        try:
            response = client.secrets.list_secrets(
                project_id=project_id,
                environment_slug=environment,
                secret_path=secret_path,
            )
        except Exception as exc:
            return SecretFetchResult.failure(
                self.name,
                f"Infisical list_secrets failed: {exc}",
            )

        secrets: dict[str, str] = {}
        for entry in getattr(response, "secrets", []) or []:
            key = getattr(entry, "secretKey", None)
            value = getattr(entry, "secretValue", None)
            if isinstance(key, str) and isinstance(value, str):
                secrets[key] = value

        return SecretFetchResult(
            secrets=secrets,
            source=self.name,
            fetched_at=_now_iso(),
            error=None,
        )
