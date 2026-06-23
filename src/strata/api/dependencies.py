"""Typed FastAPI dependencies for the data/artifact plane.

Each route used to hand-wire its own mode/auth/tenant gate by calling free
helpers (``_get_artifact_store(allow_read=True)`` and friends) directly. That
made the gate a convention the handler had to *remember* — the root cause of the
service-mode findings in #184 and the registry approve/reject/pending bug fixed
in df50987 (handlers that reached ``_get_artifact_store()`` with the wrong flags
and 403'd in the deployment they were built for).

Exposing the gates as ``Depends(...)`` makes the access a handler *declares* in
its signature, enforced before the body runs, instead of a step it can forget or
get wrong. A read route asks for ``ReadStore`` and structurally cannot have
opened the write gate.

This is phase 1 of ``docs/internal/design-server-decomposition.md``: the gate
bodies still live in ``strata.server`` and are delegated to here via lazy import
(server imports this module at load time, so the import must stay one-way). Later
phases move the bodies inward and split routers.
"""

from __future__ import annotations

from typing import Annotated, NamedTuple

from fastapi import Depends, HTTPException

# Imported at runtime (not under TYPE_CHECKING): the typed-dependency aliases
# below embed these in ``Annotated[...]`` as concrete classes, not string
# forward refs. A router module that uses an alias (e.g. ``store: ReadStore``)
# has FastAPI run ``get_type_hints`` against *that module's* globals, where
# these names are absent — a string forward ref would fail to resolve there
# (it only worked while the handlers lived in ``server.py``, which imports them).
from strata.artifact_store import ArtifactStore
from strata.transforms.build_store import BuildStore
from strata.types import Principal


def read_store() -> ArtifactStore:
    """Artifact store for a read-only endpoint.

    Opens in personal mode and in service mode (``allow_read``) — reads are
    shared-cache results, ACL-gated at retrieval, not blocked by mode. The
    handler is still responsible for tenant scoping + ``_authorize_artifact_read``
    on the concrete record.
    """
    from strata.server import _get_artifact_store

    return _get_artifact_store(allow_read=True)


ReadStore = Annotated[ArtifactStore, Depends(read_store)]


def personal_mode_store() -> ArtifactStore:
    """Artifact store for endpoints available in personal mode only.

    The bare gate (no ``allow_read``/``allow_write``): personal mode (always
    ``writes_enabled``) opens it, service mode 403s. Used by the deliberately
    personal-only management endpoints — stats/usage/list, delete, GC — which
    expose or mutate the whole store and have no tenant-scoped service-mode
    semantics. (Whether any of these *should* gain a service-mode read path is
    a separate policy question, not this refactor.)
    """
    from strata.server import _get_artifact_store

    return _get_artifact_store()


PersonalModeStore = Annotated[ArtifactStore, Depends(personal_mode_store)]


def write_store() -> ArtifactStore:
    """Artifact store for a write endpoint (put / set_name / set_alias / tags).

    Opens in personal mode, or in service mode with ``service_writes_enabled``
    AND the ``artifacts:write`` scope (authenticated write-back) — the write
    stamps the caller's tenant/principal, so it lands in their namespace and
    can't target another tenant. Binds the mode gate and the scope check
    together so a write route can't open one without the other.

    Note: the registry approve/reject routes deliberately do NOT use this — a
    governance decision opens the write *mode* gate but is authorized by the
    approver scope, not ``artifacts:write`` (see ``registry_decision``).
    """
    from strata.server import _authorize_artifact_write, _get_artifact_store

    store = _get_artifact_store(allow_write=True)
    _authorize_artifact_write()
    return store


WriteStore = Annotated[ArtifactStore, Depends(write_store)]


def current_tenant() -> str | None:
    """Tenant filter for direct artifact endpoints, or ``None`` for unscoped.

    Under trusted-proxy auth, scopes reads/writes to the caller's tenant
    (``admin:*`` and tenantless legacy artifacts stay unscoped); ``None`` when
    auth is off. The handler passes this to the store and to
    ``_ensure_artifact_access`` on the concrete record.
    """
    from strata.server import _get_artifact_request_tenant

    return _get_artifact_request_tenant()


CurrentTenant = Annotated[str | None, Depends(current_tenant)]


def current_principal() -> Principal | None:
    """The request's authenticated principal, or ``None`` when auth is disabled.

    Optional by design: registry reads and personal-mode operations have no
    principal. Routes that *require* one raise their own 401 (or use a stricter
    dependency once those land).
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

    Approving/rejecting a protected alias is a *governance* write. Under
    trusted-proxy auth it requires the approver scope (``admin:registry``;
    ``admin:*`` is break-glass) and opens the service-mode write gate
    (``service_writes_enabled``) — but deliberately NOT the ``artifacts:write``
    scope, because approvers govern, they need not be publishers. That
    distinction is exactly what the hand-wired routes got wrong; binding both
    halves here keeps them from drifting apart again.
    """
    from strata.server import _get_artifact_store, _require_registry_approver

    principal = _require_registry_approver()
    store = _get_artifact_store(allow_write=True)
    return RegistryDecision(principal=principal, store=store)


RegistryDecisionContext = Annotated[RegistryDecision, Depends(registry_decision)]


def require_scope(scope: str):
    """Path-operation dependency: require ``scope`` under trusted-proxy auth.

    Under ``trusted_proxy`` the caller must hold ``scope`` (``admin:*`` grants
    it); in personal / no-auth mode there is no principal and the endpoint stays
    open — matching the hand-written admin gates this replaces. Gates by side
    effect, so use it in the route decorator's ``dependencies=[...]`` rather than
    as a signature parameter:

        @app.post("/v1/cache/clear", dependencies=[require_scope("admin:cache")])
    """

    def _require() -> None:
        from strata.auth import get_principal
        from strata.server import get_state

        state = get_state()
        if state.config.auth_mode == "trusted_proxy":
            principal = get_principal()
            if principal is None or not principal.has_scope(scope):
                raise HTTPException(status_code=403, detail="Insufficient scope")

    return Depends(_require)


def require_notebook_worker_admin() -> None:
    """Path-operation dependency for the server-managed notebook worker registry.

    Service-mode only (409 otherwise) plus the ``admin:notebook-workers`` scope
    under trusted-proxy auth. Use via ``dependencies=[Depends(...)]``.
    """
    from strata.server import _require_notebook_worker_admin_access

    _require_notebook_worker_admin_access()


# --- Build-store / signed-transport gate (#295) -----------------------------
# The signed build-transport routes (status / manifest / download / upload /
# finalize) need two things the handler used to hand-wire: a mode check
# (``build_transport_available``) and the resolved runtime build store
# (``runtime_build_store``). The plain helpers were ``server._build_transport_
# available`` / ``server._get_runtime_build_store``, reached by the builds router
# via lazy import; they live here now so a router imports them at module top
# instead of from ``strata.server``. The ``Depends`` wrappers below bind the mode
# gate and the store resolution together for the routes where that ordering is
# behavior-preserving.


def build_transport_available() -> bool:
    """Whether signed build-transport APIs are available in the current mode.

    True in personal mode (``writes_enabled``) or when server-mode transforms are
    enabled — the two modes that can issue and honor signed build URLs.
    """
    from strata.server import get_state

    state = get_state()
    return state.config.writes_enabled or state.config.server_transforms_enabled


def runtime_build_store() -> BuildStore | None:
    """Resolve the runtime build store, or ``None`` when no ``artifact_dir`` is set.

    The signed-transport build store is the SQLite registry under the artifact
    directory; a deployment without one has nowhere to track builds.
    """
    from strata.server import get_state
    from strata.transforms.build_store import get_build_store

    state = get_state()
    artifact_dir = state.config.artifact_dir
    if artifact_dir is None:
        return None
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return get_build_store(artifact_dir / "artifacts.sqlite")


def require_build_store() -> BuildStore:
    """Param dependency: the resolved build store, 500 if uninitialized.

    No transport-mode gate — for the signature-authed upload route, which must
    not 404 on mode (the signature is the authorization). Routes that *should*
    404 when transport is off use :data:`BuildTransportStore` instead.
    """
    store = runtime_build_store()
    if store is None:
        raise HTTPException(status_code=500, detail="Build store not initialized")
    return store


RequiredBuildStore = Annotated[BuildStore, Depends(require_build_store)]


def require_build_transport_store() -> BuildStore:
    """Param dependency: 404 if transport is unavailable, else the build store (500 if None).

    Binds the transport-mode gate and the store resolution so the manifest +
    finalize routes can't open one without the other. The 404 here is the single
    message for what were three per-route variants ("polling"/"manifest"/
    "finalize"); it is executor-facing and unasserted.
    """
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
