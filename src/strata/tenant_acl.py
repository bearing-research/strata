"""Tenant-based access control for service mode.

This module provides authorization helpers for multi-tenant access control
in service mode. It enforces that callers can only access resources belonging
to their own tenant, unless they have admin privileges.

Key concepts:
- Tenant: Team/org identifier from X-Strata-Tenant header
- Principal: User/service identifier from X-Strata-Principal header
- Resource tenant: The tenant that owns a resource (artifact, build, etc.)

In service mode with auth_mode="trusted_proxy":
- Tenant is required for all artifact operations
- Resources are isolated by tenant
- Admin scope bypasses tenant checks

Security: 403 Forbidden vs 404 Not Found
- When hide_forbidden_as_not_found=True (default), return 404 for denied access
- This prevents information leakage about resource existence
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import HTTPException

if TYPE_CHECKING:
    from strata.config import StrataConfig
    from strata.types import Principal

logger = logging.getLogger(__name__)

# Request-scoped principal context
_principal_ctx: ContextVar[Principal | None] = ContextVar("principal", default=None)


def get_principal() -> Principal | None:
    """Return the current request's principal from context.

    Returns
    -------
    Principal or None
        The principal set by the auth middleware, or ``None``.
    """
    return _principal_ctx.get()


def set_principal(principal: Principal | None) -> None:
    """Set the current request's principal in context.

    Called by auth middleware after parsing headers.
    """
    _principal_ctx.set(principal)


class TenantAccessError(Exception):
    """Raised when tenant access is denied.

    Attributes
    ----------
    message : str
        Error description.
    status_code : int
        HTTP status code to surface (e.g. 401 / 400 / 403 / 404).
    """

    def __init__(self, message: str, status_code: int = 403):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def require_tenant(config: StrataConfig) -> str:
    """Require tenant to be set in the current request context.

    In service mode with auth enabled, tenant is required for all
    artifact operations. Personal mode skips tenant checks.

    Parameters
    ----------
    config : StrataConfig
        Server configuration.

    Returns
    -------
    str
        Tenant id (or a personal/anonymous sentinel when checks are skipped).

    Raises
    ------
    TenantAccessError
        If a tenant is required but not authenticated/set.
    """
    # Personal mode doesn't require tenant
    if config.deployment_mode == "personal":
        return "__personal__"

    # Service mode without auth doesn't require tenant
    if config.auth_mode == "none":
        return "__anonymous__"

    principal = get_principal()
    if principal is None:
        raise TenantAccessError("Authentication required", 401)

    if principal.tenant is None:
        raise TenantAccessError("Tenant header required for artifact operations", 400)

    return principal.tenant


def authorize_resource_tenant(
    config: StrataConfig,
    resource_tenant: str | None,
    resource_type: str = "resource",
) -> None:
    """Authorize access to a tenant-scoped resource.

    Checks that the caller's tenant matches the resource tenant.
    Admin principals bypass this check.

    Parameters
    ----------
    config : StrataConfig
        Server configuration.
    resource_tenant : str or None
        Tenant that owns the resource.
    resource_type : str, optional
        Resource type for error messages (e.g. ``"artifact"``).

    Raises
    ------
    TenantAccessError
        If access is denied (403 or 404, per ``hide_forbidden_as_not_found``).
    """
    # Personal mode doesn't enforce tenant isolation
    if config.deployment_mode == "personal":
        return

    # Service mode without auth doesn't enforce tenant isolation
    if config.auth_mode == "none":
        return

    principal = get_principal()
    if principal is None:
        raise TenantAccessError("Authentication required", 401)

    # Admin scope bypasses tenant checks
    if principal.has_scope("admin:*"):
        return

    caller_tenant = principal.tenant

    # Tenant is required in service mode
    if caller_tenant is None:
        raise TenantAccessError("Tenant header required for artifact operations", 400)

    # Resource has no tenant (legacy data or misconfiguration)
    if resource_tenant is None:
        # Allow access but log warning
        logger.warning(f"Resource has no tenant, allowing access from {caller_tenant}")
        return

    # Tenant mismatch
    if caller_tenant != resource_tenant:
        # Return 404 to hide resource existence
        if config.hide_forbidden_as_not_found:
            raise TenantAccessError(f"{resource_type.capitalize()} not found", 404)
        raise TenantAccessError(f"Access denied to {resource_type}", 403)


def raise_not_found_or_forbidden(
    config: StrataConfig,
    resource_type: str = "resource",
) -> None:
    """Raise appropriate error for access denial.

    Used when a resource doesn't exist or access is denied.
    Returns 404 when ``hide_forbidden_as_not_found`` is True.

    Parameters
    ----------
    config : StrataConfig
        Server configuration.
    resource_type : str, optional
        Resource type for error messages.

    Raises
    ------
    HTTPException
        404 or 403, per configuration.
    """
    if config.hide_forbidden_as_not_found:
        raise HTTPException(status_code=404, detail=f"{resource_type.capitalize()} not found")
    raise HTTPException(status_code=403, detail=f"Access denied to {resource_type}")


def tenant_scoped_lookup(
    caller_tenant: str | None,
    resource_tenant: str | None,
    config: StrataConfig,
) -> bool:
    """Check if caller can access a resource based on tenant.

    This is a predicate version of :func:`authorize_resource_tenant`,
    useful for filtering queries.

    Parameters
    ----------
    caller_tenant : str or None
        Caller's tenant id.
    resource_tenant : str or None
        Resource's tenant id.
    config : StrataConfig
        Server configuration.

    Returns
    -------
    bool
        ``True`` if access is allowed.
    """
    # Personal mode allows all access
    if config.deployment_mode == "personal":
        return True

    # Service mode without auth allows all access
    if config.auth_mode == "none":
        return True

    # Check admin scope
    principal = get_principal()
    if principal and principal.has_scope("admin:*"):
        return True

    # No tenant on resource (legacy)
    if resource_tenant is None:
        return True

    # Tenant match
    return caller_tenant == resource_tenant


@dataclass
class TenantContext:
    """Container for tenant context in a request.

    Provides convenient access to tenant-related information
    and authorization methods.
    """

    config: StrataConfig
    principal: Principal | None

    @property
    def tenant_id(self) -> str | None:
        """Get the caller's tenant ID."""
        if self.principal:
            return self.principal.tenant
        return None

    @property
    def principal_id(self) -> str | None:
        """Get the caller's principal ID."""
        if self.principal:
            return self.principal.id
        return None

    @property
    def is_admin(self) -> bool:
        """Check if caller has admin privileges."""
        if self.principal:
            return self.principal.has_scope("admin:*")
        return False

    @property
    def requires_tenant(self) -> bool:
        """Check if tenant is required for this request."""
        return self.config.deployment_mode == "service" and self.config.auth_mode != "none"

    def authorize(self, resource_tenant: str | None, resource_type: str = "resource") -> None:
        """Authorize access to a resource, raising on denial.

        Parameters
        ----------
        resource_tenant : str or None
            Tenant that owns the resource.
        resource_type : str, optional
            Resource type for error messages.

        Raises
        ------
        TenantAccessError
            If access is denied.
        """
        authorize_resource_tenant(self.config, resource_tenant, resource_type)

    def can_access(self, resource_tenant: str | None) -> bool:
        """Return whether the caller may access a resource.

        Parameters
        ----------
        resource_tenant : str or None
            Tenant that owns the resource.

        Returns
        -------
        bool
            ``True`` if access is allowed.
        """
        return tenant_scoped_lookup(self.tenant_id, resource_tenant, self.config)


def get_tenant_context(config: StrataConfig) -> TenantContext:
    """Return the tenant context for the current request.

    Parameters
    ----------
    config : StrataConfig
        Server configuration.

    Returns
    -------
    TenantContext
        Context bound to the current principal and config.
    """
    return TenantContext(config=config, principal=get_principal())
