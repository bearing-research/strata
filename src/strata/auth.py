"""Trusted proxy authentication and authorization.

This module implements a trusted proxy authentication model where Strata
delegates authentication to an upstream proxy (e.g., NGINX, Envoy, Kong)
and trusts identity headers injected by that proxy.

Security model:
1. Proxy authenticates users (OIDC, API keys, etc.)
2. Proxy strips any client-supplied X-Strata-* headers
3. Proxy injects trusted identity headers
4. Proxy sets X-Strata-Proxy-Token for verification
5. Strata verifies token and extracts principal

The proxy MUST:
- Remove all client-supplied X-Strata-* headers before forwarding
- Set X-Strata-Principal to the authenticated user/service ID
- Set X-Strata-Proxy-Token to the shared secret

Example NGINX configuration:
    location /strata/ {
        # Strip client headers
        proxy_set_header X-Strata-Principal "";
        proxy_set_header X-Strata-Proxy-Token "";

        # Inject trusted values
        proxy_set_header X-Strata-Principal $authenticated_user;
        proxy_set_header X-Strata-Proxy-Token "secret-token";

        proxy_pass http://strata:8765/;
    }
"""

from __future__ import annotations

import fnmatch
import hmac
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.config import AclConfig, AclRule, StrataConfig
    from strata.types import Principal, TableRef


# ---------------------------------------------------------------------------
# Principal Context
# ---------------------------------------------------------------------------

# Request-scoped principal context (set by middleware)
_principal_ctx: ContextVar[Principal | None] = ContextVar("principal", default=None)


def get_principal() -> Principal | None:
    """Get the current request's authenticated principal.

    Returns:
        Principal if authenticated, None if auth is disabled or not authenticated.
    """
    return _principal_ctx.get()


def set_principal(principal: Principal | None) -> None:
    """Set the current request's authenticated principal.

    Called by auth middleware after parsing identity headers.
    """
    _principal_ctx.set(principal)


# ---------------------------------------------------------------------------
# Auth Errors
# ---------------------------------------------------------------------------


class AuthError(Exception):
    """Authentication or authorization error.

    Attributes:
        message: Human-readable error description
        status_code: HTTP status code to return (401 or 403)
    """

    def __init__(self, message: str, status_code: int = 401):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ---------------------------------------------------------------------------
# Proxy Verification
# ---------------------------------------------------------------------------


def verify_proxy_token(request_token: str | None, expected_token: str | None) -> bool:
    """Verify that the request came from a trusted proxy.

    Uses constant-time comparison to prevent timing attacks.

    Args:
        request_token: Token from the request header
        expected_token: Expected token from configuration

    Returns:
        True if token matches or no token is configured (auth disabled)
    """
    if expected_token is None:
        # No token configured = skip verification (auth mode probably "none")
        return True
    if request_token is None:
        return False

    # Constant-time comparison to prevent timing attacks. Compare the UTF-8
    # bytes, not the strings: ASGI decodes header values as latin-1, so any
    # non-ASCII byte in the header makes ``compare_digest`` raise TypeError
    # on str inputs — a 500 where the answer is plainly "token doesn't match".
    return hmac.compare_digest(request_token.encode(), expected_token.encode())


def parse_principal(headers: dict[str, str], config: StrataConfig) -> Principal:
    """Parse Principal from request headers.

    Args:
        headers: Request headers dictionary
        config: Strata configuration

    Returns:
        Parsed Principal object

    Raises:
        AuthError: If required principal header is missing
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


# ---------------------------------------------------------------------------
# ACL Evaluator
# ---------------------------------------------------------------------------


class AclEvaluator:
    """Evaluates access control rules against principals and tables.

    ACL evaluation order:
    1. Deny rules are checked first - if any match, access is DENIED
    2. Allow rules are checked - if any match, access is ALLOWED
    3. Default action is applied (allow or deny)

    Table patterns use glob-style matching (fnmatch):
    - "file:db.*" matches "file:db.events", "file:db.users"
    - "s3:*.*" matches any S3 table
    - "*:*.*" matches all tables
    """

    def __init__(self, acl_config: AclConfig):
        """Initialize the evaluator with ACL configuration.

        Args:
            acl_config: Access control list configuration
        """
        self.config = acl_config

    def _matches_rule(
        self,
        rule: AclRule,
        principal: Principal,
        table_ref: TableRef,
    ) -> bool:
        """Check if a rule matches the principal and table.

        A rule matches if ALL conditions are satisfied:
        - Principal matches (or rule principal is "*")
        - Tenant matches (if specified in rule)
        - At least one table pattern matches

        Args:
            rule: ACL rule to check
            principal: Request principal
            table_ref: Target table reference

        Returns:
            True if rule matches
        """
        from strata.config import AclRule  # noqa: F401 - for type checking

        # Check principal match. ``principal`` is documented as a "Principal ID
        # pattern", and tables are fnmatched two lines below — but this used
        # exact string equality, so a rule like ``principal = "svc-*"`` never
        # fired. For a deny rule that fails OPEN, silently, with no startup
        # warning. fnmatch keeps "*" working exactly as before and makes the
        # documented pattern form behave.
        if not fnmatch.fnmatch(principal.id, rule.principal):
            return False

        # Check tenant match (if specified in rule)
        if rule.tenant is not None and rule.tenant != principal.tenant:
            return False

        # Check table pattern match
        table_str = str(table_ref)
        for pattern in rule.tables:
            if fnmatch.fnmatch(table_str, pattern):
                return True

        return False

    def authorize(self, principal: Principal, table_ref: TableRef) -> bool:
        """Check if principal is authorized to access table.

        Evaluation order:
        1. Check deny rules first - if any match, return False
        2. Check allow rules - if any match, return True
        3. Return default action (True if "allow", False if "deny")

        Args:
            principal: Authenticated principal making the request
            table_ref: Canonical table reference being accessed

        Returns:
            True if access is allowed, False if denied
        """
        # Check deny rules first (deny takes precedence)
        for rule in self.config.deny_rules:
            if self._matches_rule(rule, principal, table_ref):
                return False

        # Check allow rules
        for rule in self.config.allow_rules:
            if self._matches_rule(rule, principal, table_ref):
                return True

        # Default action
        return self.config.default == "allow"

    def check_scope(self, principal: Principal, required_scope: str) -> bool:
        """Check if principal has the required scope.

        The special scope 'admin:*' grants all permissions.

        Args:
            principal: Authenticated principal
            required_scope: Scope required for the operation

        Returns:
            True if principal has the scope
        """
        return principal.has_scope(required_scope)
