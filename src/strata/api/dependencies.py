"""Typed FastAPI dependencies for the data/artifact plane's mode/auth/tenant gates.

A handler declares its access in its signature (``ReadStore`` cannot open the write
gate), enforced before the body runs. Gate bodies live in ``strata.server`` and are
imported lazily: server imports this module at load time, so the import stays one-way.
"""

from __future__ import annotations

from typing import Annotated, NamedTuple

from fastapi import Depends, HTTPException

# Imported at runtime, not under TYPE_CHECKING: the aliases below embed these in ``Annotated[...]``,
# and FastAPI resolves hints against the router module's globals, where a string forward ref would
# not resolve.
from strata.artifact_store import ArtifactStore
from strata.transforms.build_store import BuildStore
from strata.types import Principal


def read_store() -> ArtifactStore:
    """Artifact store for a read-only endpoint; opens in both modes.

    The handler still owns tenant scoping and ``_authorize_artifact_read`` on the
    concrete record.
    """
    from strata.server import _get_artifact_store

    return _get_artifact_store(allow_read=True)


ReadStore = Annotated[ArtifactStore, Depends(read_store)]


def personal_mode_store() -> ArtifactStore:
    """Artifact store for personal-mode-only endpoints; service mode 403s.

    For management endpoints (list, delete, GC) that expose or mutate the whole
    store and have no tenant-scoped meaning.
    """
    from strata.server import _get_artifact_store

    return _get_artifact_store()


PersonalModeStore = Annotated[ArtifactStore, Depends(personal_mode_store)]


def store_for_scope(scope: str):
    """Artifact store for a retention endpoint, in either mode.

    Service mode opens it only for a principal holding ``scope`` (``admin:*``
    grants it), so never without principal auth. The handler still scopes the
    work to the caller's tenant.
    """

    def _store() -> ArtifactStore:
        from strata.auth import get_principal
        from strata.server import _get_artifact_store, get_state

        config = get_state().config
        if config.deployment_mode == "personal":
            return _get_artifact_store()
        # Without principal auth there is no principal, so this refuses too.
        principal = get_principal()
        if principal is None or not principal.has_scope(scope):
            raise HTTPException(status_code=403, detail="Insufficient scope")
        return _get_artifact_store(allow_read=True)

    return Depends(_store)


def write_store() -> ArtifactStore:
    """Artifact store for a write endpoint (put / set_name / set_alias / tags).

    Service mode requires ``service_writes_enabled`` and the ``artifacts:write``
    scope; both gates open together. Registry approve/reject uses
    ``registry_decision`` instead (approver scope, not ``artifacts:write``).
    """
    from strata.server import _authorize_artifact_write, _get_artifact_store

    store = _get_artifact_store(allow_write=True)
    _authorize_artifact_write()
    return store


WriteStore = Annotated[ArtifactStore, Depends(write_store)]


def current_tenant() -> str | None:
    """Tenant filter for direct artifact endpoints, or ``None`` for unscoped.

    Set under trusted-proxy auth (``admin:*`` stays unscoped); ``None`` when auth
    is off.
    """
    from strata.server import _get_artifact_request_tenant

    return _get_artifact_request_tenant()


CurrentTenant = Annotated[str | None, Depends(current_tenant)]


def current_principal() -> Principal | None:
    """The request's principal, or ``None`` when auth is disabled.

    Routes that require one raise their own 401.
    """
    from strata.auth import get_principal

    return get_principal()


CurrentPrincipal = Annotated[Principal | None, Depends(current_principal)]


class RegistryDecision(NamedTuple):
    """Resolved context for a protected-alias approve/reject decision."""

    principal: Principal | None
    store: ArtifactStore


def registry_decision() -> RegistryDecision:
    """Authorize a protected-alias decision and open the registry write gate.

    Requires the approver scope (``admin:registry``; ``admin:*`` is break-glass)
    but not ``artifacts:write``: approvers govern, they need not publish.
    """
    from strata.server import _get_artifact_store, _require_registry_approver

    principal = _require_registry_approver()
    store = _get_artifact_store(allow_write=True)
    return RegistryDecision(principal=principal, store=store)


RegistryDecisionContext = Annotated[RegistryDecision, Depends(registry_decision)]


def require_scope(scope: str):
    """Path-operation dependency: require ``scope`` under principal auth.

    Without principal auth the endpoint stays open. Use it in the decorator's
    ``dependencies=[...]``, e.g. ``dependencies=[require_scope("admin:cache")]``.
    """

    def _require() -> None:
        from strata.auth import get_principal
        from strata.server import get_state

        state = get_state()
        if state.config.principal_auth_enabled:
            principal = get_principal()
            if principal is None or not principal.has_scope(scope):
                raise HTTPException(status_code=403, detail="Insufficient scope")

    return Depends(_require)


def require_notebook_worker_admin() -> None:
    """Gate the server-managed notebook worker registry.

    Service mode only (409 otherwise), plus ``admin:notebook-workers`` under
    trusted-proxy auth.
    """
    from strata.server import _require_notebook_worker_admin_access

    _require_notebook_worker_admin_access()


# --- Build-store / signed-transport gate ---
# The ``Depends`` wrappers below bind the mode gate and the store resolution together.


def build_transport_available() -> bool:
    """Whether signed build-transport APIs are available (writes or server transforms on)."""
    from strata.server import get_state

    state = get_state()
    return state.config.writes_enabled or state.config.server_transforms_enabled


def runtime_build_store() -> BuildStore | None:
    """Resolve the runtime build store, or ``None`` when no ``artifact_dir`` is set."""
    from strata.artifact_store import get_artifact_store
    from strata.server import get_state
    from strata.transforms.build_store import get_build_store

    state = get_state()
    artifact_dir = state.config.artifact_dir
    if artifact_dir is None:
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Build rows live in the artifact store's database, so they follow
    # whichever backend it is on.
    artifact_store = get_artifact_store(artifact_dir)
    return get_build_store(
        artifact_dir / "artifacts.sqlite",
        dialect=artifact_store.dialect if artifact_store else None,
    )


def require_build_store() -> BuildStore:
    """Param dependency: the build store, 500 if uninitialized.

    No transport-mode gate: the signature-authed upload route must not 404 on mode.
    Routes that should 404 use :data:`BuildTransportStore`.
    """
    store = runtime_build_store()
    if store is None:
        raise HTTPException(status_code=500, detail="Build store not initialized")
    return store


RequiredBuildStore = Annotated[BuildStore, Depends(require_build_store)]


def require_build_transport_store() -> BuildStore:
    """Param dependency: 404 if transport is unavailable, else the build store (500 if None)."""
    if not build_transport_available():
        raise HTTPException(
            status_code=404,
            detail=(
                "Signed build transport is only available when personal-mode "
                "writes or server-mode transforms are enabled"
            ),
        )
    return require_build_store()


BuildTransportStore = Annotated[BuildStore, Depends(require_build_transport_store)]


# --- Table-input resolution + ACL ---
# Pure resolution lives in ``MaterializeService.resolve_input_version``. The request-scoped table
# ACL and HTTP mapping live here, so materialize, explain and the names router share one enforced
# unit.


def authorize_table_access(table_uri: str, table_identity) -> None:
    """Enforce table-level ACL under principal auth; no-op otherwise.

    Shared by the scan path and every table-as-input path. A ``None``
    ``table_identity`` (URI names no table) is denied.

    Raises:
        HTTPException: 401 if no principal; 403, or 404 under
        ``hide_forbidden_as_not_found``, if the ACL denies the table.
    """
    from strata.auth import AclEvaluator, get_principal
    from strata.iceberg import named_catalog, shared_catalog_stores
    from strata.server import get_state
    from strata.types import TableRef

    state = get_state()
    if not state.config.principal_auth_enabled:
        return

    principal = get_principal()
    if principal is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if table_identity is None:
        # Deny-first: a table the ACL cannot name is not one it allows.
        allowed = False
    else:
        named, _ = named_catalog(table_uri, state.config)
        table_ref = TableRef.from_table_identity(
            table_identity, table_uri=table_uri, named_catalog_name=named
        )
        # The names the same table has under the other address forms, which a
        # deny rule written for any one of them must also refuse.
        aliases = tuple(
            TableRef(catalog=store, namespace=table_ref.namespace, table=table_ref.table)
            for store in shared_catalog_stores(table_uri, state.config)
            if store != table_ref.catalog
        )
        allowed = AclEvaluator(state.config.acl_config).authorize(principal, table_ref, aliases)
    if not allowed:
        if state.config.hide_forbidden_as_not_found:
            raise HTTPException(status_code=404, detail="Table not found")
        raise HTTPException(status_code=403, detail="Access denied")


def resolve_input_version(input_uri: str, tenant: str | None = None) -> str:
    """Resolve an input URI to its current version, enforcing table ACL and artifact access.

    Side effect: records a use of an artifact input for retention.

    Raises:
        HTTPException: 400/404 for an unresolvable URI; 401/403/404 for a denied input.
    """
    from strata.server import (
        _authorize_artifact_read,
        _ensure_artifact_access,
        _get_artifact_store,
        _table_identity_from_uri,
        get_state,
    )
    from strata.services.materialize import InputResolutionError, materialize_service

    store = _get_artifact_store(allow_server_mode=True)
    # Authorize a table input before planning, on the identity its URI names, as the scan path does.
    # Planning first would answer a denied caller with the plan's failure (naming the table and its
    # delete files) or a 400 that materialize builds past. The check after resolution stays, for a
    # catalog that resolves the table to another identity.
    if input_uri.startswith(("file://", "s3://")):
        authorize_table_access(input_uri, _table_identity_from_uri(input_uri))
    try:
        resolved = materialize_service.resolve_input_version(
            input_uri, store=store, planner=get_state().planner, tenant=tenant
        )
    except InputResolutionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail) from e

    # Same ACL as the direct scan path, so a transform input cannot bypass it. Runs after resolution
    # (outside the error mapping) so a 401/403/404 isn't rewritten into a 400.
    if resolved.table_identity is not None:
        authorize_table_access(input_uri, resolved.table_identity)
    # Same tenant + provenance-ACL checks as a direct GET /v1/artifacts/{id}. Otherwise a transform
    # input could read another tenant's blob, or in pull mode get a signed URL for it.
    if resolved.artifact is not None:
        _ensure_artifact_access(resolved.artifact, tenant)
        _authorize_artifact_read(resolved.artifact, store)
        # A result read only as another computation's input is still in use:
        # its downstream's cache hits never read it, and retention would
        # otherwise collect it while every request for the downstream needs it.
        if store is not None:
            store.record_use(resolved.artifact.id, resolved.artifact.version)
    return resolved.version
