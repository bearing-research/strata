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

from typing import TYPE_CHECKING, Annotated, NamedTuple

from fastapi import Depends

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore
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


ReadStore = Annotated["ArtifactStore", Depends(read_store)]


def current_principal() -> Principal | None:
    """The request's authenticated principal, or ``None`` when auth is disabled.

    Optional by design: registry reads and personal-mode operations have no
    principal. Routes that *require* one raise their own 401 (or use a stricter
    dependency once those land).
    """
    from strata.auth import get_principal

    return get_principal()


CurrentPrincipal = Annotated["Principal | None", Depends(current_principal)]


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
