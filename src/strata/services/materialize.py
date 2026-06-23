"""Materialize-plane services extracted from ``server.py`` handlers.

Stateless; methods receive an already-resolved artifact store (+ planner +
tenant) from the route's dependencies. No FastAPI/HTTP coupling: the pure
input-version *resolution* lives here and signals failure with the plain
:class:`InputResolutionError` (a status hint, not an ``HTTPException``); the
thin wrapper in ``strata.api.dependencies`` maps that to HTTP and applies the
table ACL. See ``docs/internal/design-server-decomposition.md`` (phase 2/3).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

from strata.artifact_store import TransformSpec, compute_provenance_hash
from strata.types import (
    ExplainMaterializeRequest,
    ExplainMaterializeResponse,
    InputChangeInfo,
)

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore, ArtifactVersion


class InputResolutionError(Exception):
    """An input URI could not be resolved to a version.

    Carries the HTTP ``status_code`` + ``detail`` the original inline resolver
    raised (400 for a malformed/unknown URI or a failed table plan, 404 for an
    unknown name) so the dependency-layer wrapper can reproduce the exact
    response without the service importing FastAPI.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ResolvedInput(NamedTuple):
    """A resolved input version, plus the table identity when the input is a table.

    ``table_identity`` is ``None`` for artifact/name inputs and set for table
    URIs — the wrapper uses it to run the table ACL (deny-first on every table
    input) as a visible step, which the pure resolver deliberately does not do.
    """

    version: str
    table_identity: object | None = None


class MaterializeService:
    """Pure materialize-plane computations (no HTTP, no auth)."""

    def resolve_input_version(
        self,
        input_uri: str,
        *,
        store: ArtifactStore,
        planner,
        tenant: str | None = None,
    ) -> ResolvedInput:
        """Resolve an input URI to its current version (pure; no ACL, no HTTP).

        - ``strata://artifact/{id}@v={n}`` → ``"{id}@v={n}"``
        - ``strata://name/{name}`` → the named artifact's ``"{id}@v={version}"``
        - ``file://…`` / ``s3://…`` table → the current snapshot id, plus the
          plan's ``table_identity`` so the caller can ACL-gate it.

        Raises:
            InputResolutionError: malformed/unknown URI, unknown name, or a table
                whose plan fails — carrying the status the wrapper re-raises.
        """
        # Artifact URI: strata://artifact/{id}@v={version}
        if input_uri.startswith("strata://artifact/"):
            match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", input_uri)
            if match:
                return ResolvedInput(f"{match.group(1)}@v={int(match.group(2))}")
            raise InputResolutionError(400, f"Invalid artifact URI: {input_uri}")

        # Name URI: strata://name/{name}
        if input_uri.startswith("strata://name/"):
            name = input_uri.replace("strata://name/", "")
            artifact = store.resolve_name(name, tenant=tenant)
            if artifact is None:
                raise InputResolutionError(404, f"Name not found: {name}")
            return ResolvedInput(f"{artifact.id}@v={artifact.version}")

        # Table URI: file:// or s3://
        if input_uri.startswith("file://") or input_uri.startswith("s3://"):
            try:
                plan = planner.plan(
                    table_uri=input_uri,
                    snapshot_id=None,  # Current snapshot
                    columns=None,
                    filters=None,
                )
            except Exception as e:
                raise InputResolutionError(
                    400, f"Could not resolve table {input_uri}: {str(e)}"
                ) from e
            return ResolvedInput(str(plan.snapshot_id), table_identity=plan.table_identity)

        raise InputResolutionError(400, f"Unknown input URI type: {input_uri}")

    def compute_provenance(
        self,
        transform_spec: TransformSpec,
        resolved_versions: dict[str, str],
    ) -> str:
        """Provenance hash for a transform over already-resolved input versions.

        Inputs are sorted before hashing, so the hash is independent of input
        ordering — the invariant that keeps the same computation from hashing to
        two different cache keys.
        """
        input_hashes = [f"{uri}:{version}" for uri, version in sorted(resolved_versions.items())]
        return compute_provenance_hash(input_hashes, transform_spec)

    def rebuild_artifact_id(
        self,
        existing: ArtifactVersion | None,
        *,
        refresh: bool,
        new_id: str,
    ) -> str:
        """Artifact id for a build: reuse the existing id on a refresh rebuild.

        A refresh rebuild becomes a new *version* of the same artifact so
        finalize supersedes the old ready version and provenance lookups resolve
        to the rebuild (#123). Every other miss mints a fresh id.
        """
        if refresh and existing is not None:
            return existing.id
        return new_id

    def explain(
        self,
        store: ArtifactStore,
        *,
        request: ExplainMaterializeRequest,
        tenant: str | None,
        resolved_versions: dict[str, str],
    ) -> ExplainMaterializeResponse:
        """Explain what materialize would do, given already-resolved input versions.

        Computes the provenance hash, checks for a cache hit, and — when a name is
        supplied — reports staleness against the name's recorded input versions.
        Version resolution (which can fail with HTTP errors) is the caller's job;
        *resolved_versions* is passed in verbatim, error markers and all.
        """
        transform = request.transform
        transform_spec = TransformSpec(
            executor=transform.executor,
            params=transform.params,
            inputs=request.inputs,
        )

        provenance_hash = self.compute_provenance(transform_spec, resolved_versions)

        existing = store.find_by_provenance(provenance_hash, tenant=tenant)
        if existing is not None:
            return ExplainMaterializeResponse(
                would_hit=True,
                artifact_uri=f"strata://artifact/{existing.id}@v={existing.version}",
                would_build=False,
                resolved_input_versions=resolved_versions,
            )

        # Cache miss — if a name is given, report whether its inputs have drifted.
        changed_inputs: list[InputChangeInfo] = []
        is_stale = False
        stale_reason: str | None = None
        existing_artifact_uri: str | None = None

        if request.name:
            name_status = store.get_name_status(request.name, tenant=tenant)
            if name_status is not None:
                existing_artifact_uri = name_status.artifact_uri
                for input_uri, old_version in name_status.input_versions.items():
                    current_version = resolved_versions.get(input_uri)
                    if current_version and current_version != old_version:
                        changed_inputs.append(
                            InputChangeInfo(
                                input_uri=input_uri,
                                old_version=old_version,
                                new_version=current_version,
                            )
                        )
                is_stale = len(changed_inputs) > 0
                if is_stale:
                    changes = [
                        f"{c.input_uri}: {c.old_version} → {c.new_version}" for c in changed_inputs
                    ]
                    stale_reason = f"Rebuild needed: {', '.join(changes)}"

        return ExplainMaterializeResponse(
            would_hit=False,
            artifact_uri=existing_artifact_uri,
            would_build=True,
            is_stale=is_stale,
            stale_reason=stale_reason,
            changed_inputs=changed_inputs if changed_inputs else None,
            resolved_input_versions=resolved_versions,
        )


materialize_service = MaterializeService()
