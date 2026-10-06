"""Infisical secret-manager integration via the official Python SDK.

Auth, highest first: Universal Auth (``INFISICAL_CLIENT_ID`` +
``INFISICAL_CLIENT_SECRET``), then a service token (``INFISICAL_TOKEN``); with
neither, the provider fails naming both. Project routing comes from the
notebook's ``[secret_manager]`` block, overridable by env vars.

The host receives the server's credentials, so it is the operator's alone
(``INFISICAL_HOST``, else the public default); a notebook ``base_url`` naming
anywhere else is refused before any login, in every mode.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from strata.notebook.secret_manager.provider import (
    SECRET_FETCH_TIMEOUT_SECONDS,
    SecretFetchResult,
    _now_iso,
)

logger = logging.getLogger(__name__)

_DEFAULT_HOST = "https://app.infisical.com"
_DEFAULT_ENVIRONMENT = "dev"
_DEFAULT_PATH = "/"


class InfisicalProvider:
    """Pulls secrets from an Infisical project using the official SDK."""

    name = "infisical"

    def fetch(
        self, config: dict[str, Any], *, timeout: float = SECRET_FETCH_TIMEOUT_SECONDS
    ) -> SecretFetchResult:
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
        if host != operator_host:
            # The login sends the server's credentials to *host*, and a cloned or
            # shared notebook's author is not the one who set them.
            return SecretFetchResult.failure(
                self.name,
                f"[secret_manager] base_url {notebook_host!r} is refused: it would "
                "receive the server's Infisical credentials. The server logs in only at "
                f"{operator_host} (INFISICAL_HOST); remove base_url from notebook.toml, "
                "or set INFISICAL_HOST where the server starts.",
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
        # The SDK sets no timeout, so a host that accepts and never answers holds the
        # fetch forever.
        adapter = _timeout_adapter(timeout)
        client.api.session.mount("http://", adapter)
        client.api.session.mount("https://", adapter)
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


def _timeout_adapter(timeout: float) -> Any:
    """A ``requests`` adapter that gives a request sent without a timeout *timeout*.

    Built on call so ``requests`` loads only when a notebook fetches secrets.
    """
    from requests.adapters import HTTPAdapter

    class TimeoutAdapter(HTTPAdapter):
        def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
            if timeout is None:
                timeout = default_timeout
            return super().send(
                request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies
            )

    default_timeout = timeout
    return TimeoutAdapter()
