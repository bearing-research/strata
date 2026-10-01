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
from strata.iceberg_schema import UnsupportedTableFormatError
from strata.types import (
    ExplainMaterializeRequest,
    ExplainMaterializeResponse,
    InputChangeInfo,
    ReadPlan,
)

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore, ArtifactVersion


def table_input_version(plan: ReadPlan) -> str:
    """The version a table input records: its snapshot and schema id.

    A schema change (a rename, a dropped column) makes no snapshot, so the
    snapshot id alone would let a transform over the table hit the artifact it
    built before the change, with the old column names.
    """
    return f"{plan.snapshot_id}:{plan.schema_id}"


class InputResolutionError(Exception):
    """An input URI could not be resolved to a version.

    Carries the HTTP ``status_code`` + ``detail`` the original inline resolver
    raised (400 for a malformed/unknown URI or a failed table plan, 404 for an
    unknown name, 422 for a table Strata refuses to read) so the
    dependency-layer wrapper can reproduce the exact response without the
    service importing FastAPI.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ResolvedInput(NamedTuple):
    """A resolved input version, plus what the caller needs to authorize it.

    ``table_identity`` is set for table URIs and ``artifact`` for
    artifact/name inputs — the wrapper uses them to run the table ACL
    (deny-first on every table input) and the artifact tenant/ACL gate as
    visible steps, which the pure resolver deliberately does not do.
    """

    version: str
    table_identity: object | None = None
    artifact: ArtifactVersion | None = None


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
        - ``file://…`` / ``s3://…`` table → the current snapshot and schema id
          (:func:`table_input_version`), plus the plan's ``table_identity`` so
          the caller can ACL-gate it.

        Raises:
            InputResolutionError: malformed/unknown URI, unknown name, or a table
                whose plan fails — carrying the status the wrapper re-raises. A
                table Strata refuses to read is 422 with the planner's message,
                as on the scan path.
        """
        if input_uri.startswith("strata://artifact/"):
            match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", input_uri)
            if match:
                # Look the artifact up rather than trust the URI: the runner reads the blob by
                # version with no further check, so a bare regex parse would let a caller name any
                # artifact, including another tenant's. The record feeds the wrapper's tenant/ACL
                # gate.
                artifact_id, version = match.group(1), int(match.group(2))
                artifact = store.get_artifact(artifact_id, version)
                if artifact is None:
                    raise InputResolutionError(404, f"Artifact not found: {input_uri}")
                return ResolvedInput(f"{artifact_id}@v={version}", artifact=artifact)
            raise InputResolutionError(400, f"Invalid artifact URI: {input_uri}")

        if input_uri.startswith("strata://name/"):
            name = input_uri.replace("strata://name/", "")
            artifact = store.resolve_name(name, tenant=tenant)
            if artifact is None:
                raise InputResolutionError(404, f"Name not found: {name}")
            return ResolvedInput(f"{artifact.id}@v={artifact.version}", artifact=artifact)

        if input_uri.startswith("file://") or input_uri.startswith("s3://"):
            try:
                plan = planner.plan(
                    table_uri=input_uri,
                    snapshot_id=None,  # Current snapshot
                    columns=None,
                    filters=None,
                )
            except UnsupportedTableFormatError as e:
                raise InputResolutionError(422, str(e)) from e
            except Exception as e:
                raise InputResolutionError(
                    400, f"Could not resolve table {input_uri}: {str(e)}"
                ) from e
            return ResolvedInput(table_input_version(plan), table_identity=plan.table_identity)

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

    def compute_identity_provenance(
        self,
        table_identity: str,
        snapshot_id: int,
        columns: list[str] | None,
        filters: list,
        schema_id: int | None,
    ) -> str:
        """Provenance hash for a ``scan@v1`` identity transform.

        Uniquely identifies a table scan by table identity + snapshot, the
        (sorted) column projection, and the normalized row filters — so the same
        query dedups to the same artifact. ``schema_id`` is the schema the scan
        read (``ReadPlan.schema_id``): a schema change makes no snapshot. Pure;
        no HTTP, no store.
        """
        import hashlib

        from strata.types import compute_filter_fingerprint

        hasher = hashlib.sha256()
        hasher.update(f"table:{table_identity}@{snapshot_id}".encode())
        hasher.update(f"schema:{schema_id}".encode())
        hasher.update(b"executor:scan@v1")
        if columns:
            hasher.update(f"columns:{sorted(columns)}".encode())
        else:
            hasher.update(b"columns:*")
        filter_fp = compute_filter_fingerprint(filters)
        hasher.update(f"filters:{filter_fp}".encode())
        return hasher.hexdigest()

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

        # Cache miss: if a name is given, report whether its inputs have drifted.
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
