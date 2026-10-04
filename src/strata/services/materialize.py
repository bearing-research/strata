"""Materialize-plane services: input-version resolution and provenance hashing.

Stateless and HTTP-free: failures raise :class:`InputResolutionError` carrying
a status hint, which the wrapper in ``strata.api.dependencies`` maps to HTTP
after applying the table ACL.
"""

from __future__ import annotations

import re
import uuid
from typing import TYPE_CHECKING, NamedTuple

from pyiceberg.exceptions import NoSuchTableError

from strata.artifact_store import TransformSpec, compute_provenance_hash
from strata.iceberg import WarehouseNotFound
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

    A schema change makes no snapshot, so the snapshot alone would let a transform
    hit an artifact built against the old columns. A table with no snapshots has no
    version, so it gets a fresh one each time and nothing built from it dedups.
    """
    if plan.snapshot_id is None:
        return f"empty:{uuid.uuid4().hex}:{plan.schema_id}"
    return f"{plan.snapshot_id}:{plan.schema_id}"


class InputResolutionError(Exception):
    """An input URI could not be resolved to a version.

    Carries ``status_code`` and ``detail`` (400 malformed/unknown URI or failed
    plan, 404 unknown name, 422 table Strata refuses to read) for the dependency
    layer to raise as HTTP.
    """

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ResolvedInput(NamedTuple):
    """A resolved input version, plus what the caller needs to authorize it.

    ``table_identity`` is set for table URIs and ``artifact`` for artifact/name
    inputs; the wrapper runs the table ACL and artifact tenant gate on them.
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
        """Resolve an input URI to its current version (no ACL, no HTTP).

        - ``strata://artifact/{id}@v={n}`` -> ``"{id}@v={n}"``
        - ``strata://name/{name}`` -> the named artifact's ``"{id}@v={version}"``
        - ``file://`` / ``s3://`` table -> :func:`table_input_version`, plus the
          plan's ``table_identity`` for the caller's ACL check.

        Raises:
            InputResolutionError: malformed/unknown URI, unknown name, or a failed
                table plan; a table its catalog does not have, or a local warehouse that
                does not exist, is 404, one Strata refuses to read 422 with the planner's
                message.
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
            except NoSuchTableError as e:
                raise InputResolutionError(404, f"Table not found: {input_uri}") from e
            except WarehouseNotFound as e:
                raise InputResolutionError(404, str(e)) from e
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

        Input versions are sorted before hashing.
        """
        input_hashes = [f"{uri}:{version}" for uri, version in sorted(resolved_versions.items())]
        return compute_provenance_hash(input_hashes, transform_spec)

    def compute_identity_provenance(
        self,
        table_identity: str,
        snapshot_id: int | None,
        columns: list[str] | None,
        filters: list,
        schema_id: int | None,
    ) -> str:
        """Provenance hash for a ``scan@v1`` identity transform.

        Covers table identity, snapshot, ``schema_id`` (a schema change makes no
        snapshot), the sorted projection and the normalized filters, so the same query
        dedups to the same artifact. A table with no snapshots (``snapshot_id`` None)
        has nothing to key a result on, so its hash is unique per call and never dedups.
        """
        import hashlib

        from strata.types import compute_filter_fingerprint

        hasher = hashlib.sha256()
        snapshot = snapshot_id if snapshot_id is not None else f"empty:{uuid.uuid4().hex}"
        hasher.update(f"table:{table_identity}@{snapshot}".encode())
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

        A refresh becomes a new version of the same artifact, so finalize supersedes
        the old one and provenance lookups resolve to the rebuild. Other misses mint a
        fresh id.
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

        Reports the provenance hash, any cache hit and, when a name is requested,
        staleness against that name's recorded inputs. ``resolved_versions`` is used
        verbatim, error markers included.
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
