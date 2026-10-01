"""Trusted-proxy authentication and authorization.

Strata does not authenticate: an upstream proxy (NGINX, Envoy, Kong) does, and
Strata trusts its identity headers. The proxy MUST strip client-supplied
``X-Strata-*`` headers, set ``X-Strata-Principal`` to the authenticated id, and
set ``X-Strata-Proxy-Token`` to the shared secret. For example (NGINX)::

    location /strata/ {
        proxy_set_header X-Strata-Principal $authenticated_user;
        proxy_set_header X-Strata-Proxy-Token "secret-token";
        proxy_pass http://strata:8765/;
    }
"""

from __future__ import annotations

import fnmatch
import hmac
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strata.config import AclConfig, AclRule, StrataConfig
    from strata.types import Principal, TableRef


# Principal context

_principal_ctx: ContextVar[Principal | None] = ContextVar("principal", default=None)


def get_principal() -> Principal | None:
    """Return the current request's principal, or None if auth is disabled or unauthenticated."""
    return _principal_ctx.get()


_FORWARDED_PRINCIPAL = "X-Strata-Principal"


def remote_store_headers(config: Any) -> dict[str, str]:
    """Return the headers a request from this server to the team store carries.

    With a caller in context and ``notebook_remote_store_forward_principal`` on,
    the caller's id replaces the principal in ``notebook_remote_store_headers``, so
    work done through a shared server is attributed to the member. Personal mode
    sends the static headers unchanged.
    """
    headers = dict(getattr(config, "notebook_remote_store_headers", {}) or {})
    principal = get_principal()
    if principal is None or not getattr(config, "notebook_remote_store_forward_principal", True):
        return headers
    # Header names are case-insensitive, and the static set is operator-typed.
    headers = {k: v for k, v in headers.items() if k.lower() != _FORWARDED_PRINCIPAL.lower()}
    headers[_FORWARDED_PRINCIPAL] = principal.id
    return headers


@contextmanager
def principal_context(principal: Principal | None) -> Iterator[None]:
    """Make *principal* the current caller for a block, restoring the previous one after.

    For work outside the request task that authenticated it, such as an MCP tool call.
    """
    token = _principal_ctx.set(principal)
    try:
        yield
    finally:
        _principal_ctx.reset(token)


def set_principal(principal: Principal | None) -> None:
    """Set the current request's principal (called by the auth middleware)."""
    _principal_ctx.set(principal)


class AuthError(Exception):
    """Authentication or authorization error carrying the HTTP status (401 or 403)."""

    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# Proxy verification and principal parsing


def verify_proxy_token(request_token: str | None, expected_token: str | None) -> bool:
    """Check the proxy token in constant time; True when no token is configured."""
    if expected_token is None:
        # No token configured: verification is off.
        return True
    if request_token is None:
        return False

    # Constant-time, on UTF-8 bytes: ASGI decodes headers as latin-1, and a
    # non-ASCII str makes ``compare_digest`` raise TypeError (a 500, not a mismatch).
    return hmac.compare_digest(request_token.encode(), expected_token.encode())


def parse_principal(headers: dict[str, str], config: StrataConfig) -> Principal:
    """Parse the Principal from request headers.

    Raises:
        AuthError: If the required principal header is missing.
    """
    from strata.types import Principal

    def _header(name: str) -> str | None:
        return headers.get(name) or headers.get(name.lower())

    principal_id = _header(config.principal_header)
    if not principal_id:
        raise AuthError("Missing principal header", 401)

    tenant = _header(config.tenant_header)
    scopes_str = _header(config.scopes_header) or ""
    scopes = frozenset(scopes_str.split()) if scopes_str else frozenset()

    return Principal(id=principal_id, tenant=tenant, scopes=scopes)


def parse_api_key_principal(headers: dict[str, str], config: StrataConfig) -> Principal:
    """Resolve an ``Authorization: Bearer`` API key to a principal.

    Returns the same ``Principal`` as :func:`parse_principal`, so nothing downstream
    depends on the auth mode.

    Raises
    ------
    AuthError
        If the header is absent or the key does not resolve; one message for both,
        so a caller cannot tell malformed, unknown, revoked or expired keys apart.
    """
    from strata.api_keys import get_api_key_store

    def _header(name: str) -> str | None:
        return headers.get(name) or headers.get(name.lower())

    authorization = _header("Authorization") or ""
    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != "bearer" or not presented.strip():
        raise AuthError("Missing or malformed Authorization header", 401)

    store = get_api_key_store()
    if store is None:
        # Key auth with no key store: fail closed rather than serve anonymously.
        raise AuthError("Unauthorized", 401)

    principal = store.verify(presented.strip())
    if principal is None:
        raise AuthError("Unauthorized", 401)
    return principal


class AclEvaluator:
    """Evaluates ACL rules against principals and tables, deny-first.

    Deny rules win, then allow rules, then the default. Table patterns are fnmatch
    globs, e.g. ``"file:db.*"``, ``"s3:*.*"``, ``"*:*.*"``.
    """

    def __init__(self, acl_config: AclConfig):
        """Initialize the evaluator with ACL configuration."""
        self.config = acl_config

    def _matches_rule(
        self,
        rule: AclRule,
        principal: Principal,
        table_ref: TableRef,
    ) -> bool:
        """Return True if principal (or ``*``), tenant (if set) and any table pattern match."""
        from strata.config import AclRule  # noqa: F401 - for type checking

        # ``principal`` is a pattern; exact equality would let a deny rule like
        # ``svc-*`` silently fail open.
        if not fnmatch.fnmatch(principal.id, rule.principal):
            return False

        if rule.tenant is not None and rule.tenant != principal.tenant:
            return False

        table_str = str(table_ref)
        for pattern in rule.tables:
            if fnmatch.fnmatch(table_str, pattern):
                return True

        return False

    def authorize(
        self,
        principal: Principal,
        table_ref: TableRef,
        aliases: tuple[TableRef, ...] = (),
    ) -> bool:
        """Return whether ``principal`` may access ``table_ref`` (deny, then allow, then default).

        A deny rule matching any of ``aliases`` refuses the table, so another address
        cannot step around it; allow rules match only ``table_ref``, so an alias never grants.
        """
        # Deny first: an explicit deny beats any allow.
        for rule in self.config.deny_rules:
            if any(self._matches_rule(rule, principal, ref) for ref in (table_ref, *aliases)):
                return False

        for rule in self.config.allow_rules:
            if self._matches_rule(rule, principal, table_ref):
                return True

        return self.config.default == "allow"

    def check_scope(self, principal: Principal, required_scope: str) -> bool:
        """Return whether the principal has ``required_scope``; ``admin:*`` grants all."""
        return principal.has_scope(required_scope)
