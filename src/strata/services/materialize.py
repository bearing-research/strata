"""Materialize-plane services extracted from ``server.py`` handlers.

Stateless; methods receive an already-resolved artifact store + tenant from the
route's dependencies. No FastAPI/HTTP coupling — input-version *resolution*
(which can 400/404) stays in the handler, which passes the resolved versions in,
so the service is pure and unit-testable without a TestClient. See
``docs/internal/design-server-decomposition.md`` (phase 2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from strata.artifact_store import TransformSpec, compute_provenance_hash
from strata.types import (
    ExplainMaterializeRequest,
    ExplainMaterializeResponse,
    InputChangeInfo,
)

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore


class MaterializeService:
    """Pure materialize-plane computations (no HTTP, no version resolution)."""

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

        input_hashes = [f"{uri}:{version}" for uri, version in sorted(resolved_versions.items())]
        provenance_hash = compute_provenance_hash(input_hashes, transform_spec)

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
