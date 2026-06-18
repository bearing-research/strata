"""Read-side artifact introspection services (lineage, dependents, …).

Extracted from ``server.py`` handlers so the graph-walking is unit-testable
without spinning a server. ``ArtifactService`` is stateless; every method takes
an already-resolved artifact store (and tenant filter) from the route's
dependencies. No HTTP coupling — the handler does access/state checks and
response shaping; the service walks the graph and returns plain response models.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from strata.artifact_store import TransformSpec
from strata.types import (
    ArtifactDependentsResponse,
    ArtifactLineageResponse,
    DependentInfo,
    LineageEdge,
    LineageNode,
)

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore, ArtifactVersion


def _input_version_to_artifact_ref(
    input_uri: str,
    input_version: str,
) -> tuple[str, str, int] | None:
    """Resolve stored input version metadata back to a concrete artifact URI."""
    if not (input_uri.startswith("strata://artifact/") or input_uri.startswith("strata://name/")):
        return None
    if "@v=" not in input_version:
        return None

    artifact_id, version_text = input_version.split("@v=", 1)
    try:
        version = int(version_text)
    except ValueError:
        return None

    return (f"strata://artifact/{artifact_id}@v={version}", artifact_id, version)


def _transform_ref(transform_spec: str | None) -> str | None:
    """Executor ref from a stored transform_spec, or ``None`` if absent/malformed.

    ``transform_spec`` is client-opaque (it may lack an executor or not be JSON),
    so a parse failure means "no known transform", not an error.
    """
    if not transform_spec:
        return None
    try:
        return TransformSpec.from_json(transform_spec).executor
    except (json.JSONDecodeError, KeyError):
        return None


def _load_input_versions(input_versions: str | None) -> dict[str, str]:
    """Parse the stored ``input_uri -> version`` map, or ``{}`` if absent/malformed."""
    if not input_versions:
        return {}
    try:
        return json.loads(input_versions)
    except (json.JSONDecodeError, ValueError):
        return {}


class ArtifactService:
    """Stateless read-side artifact introspection."""

    def build_lineage(
        self,
        store: ArtifactStore,
        *,
        artifact: ArtifactVersion,
        artifact_id: str,
        version: int,
        tenant_filter: str | None,
        max_depth: int,
    ) -> ArtifactLineageResponse:
        """Build the input-dependency graph for an already-validated artifact.

        BFS over ``input_versions``, resolving artifact inputs to nodes/edges and
        recording table inputs as leaf nodes, bounded by ``max_depth``. The
        caller has already fetched ``artifact``, checked tenant access, and
        verified it is ready; this is pure graph traversal.
        """
        artifact_uri = f"strata://artifact/{artifact_id}@v={version}"
        nodes: dict[str, LineageNode] = {}
        edges: list[LineageEdge] = []
        visited: set[str] = set()
        queue: list[tuple[str, str, int, int]] = []  # (uri, artifact_id, version, depth)

        # Add root node
        nodes[artifact_uri] = LineageNode(
            uri=artifact_uri,
            artifact_id=artifact_id,
            version=version,
            type="artifact",
            transform_ref=_transform_ref(artifact.transform_spec),
            created_at=artifact.created_at,
        )
        visited.add(artifact_uri)

        # Parse input_versions and add to queue
        direct_inputs: list[str] = []
        for input_uri, input_version in _load_input_versions(artifact.input_versions).items():
            direct_inputs.append(input_uri)
            resolved_input = _input_version_to_artifact_ref(input_uri, input_version)
            edge_from_uri = resolved_input[0] if resolved_input is not None else input_uri
            edges.append(
                LineageEdge(
                    from_uri=edge_from_uri,
                    to_uri=artifact_uri,
                    input_version=input_version,
                )
            )

            if resolved_input is not None:
                resolved_uri, inp_artifact_id, inp_version = resolved_input
                queue.append((resolved_uri, inp_artifact_id, inp_version, 1))
            elif input_uri not in visited:
                # It's a table input
                visited.add(input_uri)
                nodes[input_uri] = LineageNode(uri=input_uri, type="table")

        # BFS to traverse transitive dependencies
        max_depth_reached = 0
        while queue:
            uri, art_id, art_ver, depth = queue.pop(0)

            if depth > max_depth:
                continue
            max_depth_reached = max(max_depth_reached, depth)

            node_uri = f"strata://artifact/{art_id}@v={art_ver}"
            if node_uri in visited:
                continue
            visited.add(node_uri)

            # Get the artifact
            input_artifact = store.get_artifact(art_id, art_ver)
            if (
                input_artifact is None
                or input_artifact.state != "ready"
                or (
                    tenant_filter is not None
                    and input_artifact.tenant is not None
                    and input_artifact.tenant != tenant_filter
                )
            ):
                # Add as unknown node
                nodes[node_uri] = LineageNode(
                    uri=node_uri,
                    artifact_id=art_id,
                    version=art_ver,
                    type="artifact",
                )
                continue

            nodes[node_uri] = LineageNode(
                uri=node_uri,
                artifact_id=art_id,
                version=art_ver,
                type="artifact",
                transform_ref=_transform_ref(input_artifact.transform_spec),
                created_at=input_artifact.created_at,
            )

            # Add this artifact's inputs to queue
            for inp_uri, inp_version in _load_input_versions(input_artifact.input_versions).items():
                resolved_input = _input_version_to_artifact_ref(inp_uri, inp_version)
                edge_from_uri = resolved_input[0] if resolved_input is not None else inp_uri
                edges.append(
                    LineageEdge(
                        from_uri=edge_from_uri,
                        to_uri=node_uri,
                        input_version=inp_version,
                    )
                )

                if resolved_input is not None:
                    resolved_uri, nested_id, nested_ver = resolved_input
                    queue.append((resolved_uri, nested_id, nested_ver, depth + 1))
                elif inp_uri not in visited:
                    # Table input
                    visited.add(inp_uri)
                    nodes[inp_uri] = LineageNode(uri=inp_uri, type="table")

        return ArtifactLineageResponse(
            artifact_uri=artifact_uri,
            artifact_id=artifact_id,
            version=version,
            nodes=list(nodes.values()),
            edges=edges,
            depth=max_depth_reached,
            direct_inputs=direct_inputs,
        )

    def build_dependents(
        self,
        store: ArtifactStore,
        *,
        artifact_id: str,
        version: int,
        tenant_filter: str | None,
        limit: int,
    ) -> ArtifactDependentsResponse:
        """List direct (one-hop) dependents of an artifact, newest store order.

        The caller has already verified the target artifact exists, is ready, and
        is in-tenant. ``total_count`` reflects all dependents; the returned list is
        capped at ``limit``.
        """
        dependent_results = store.find_dependents(artifact_id, version, tenant=tenant_filter)

        dependents = [
            DependentInfo(
                artifact_uri=f"strata://artifact/{dep_artifact.id}@v={dep_artifact.version}",
                artifact_id=dep_artifact.id,
                version=dep_artifact.version,
                name=store.get_name_for_artifact(
                    dep_artifact.id, dep_artifact.version, tenant=tenant_filter
                ),
                transform_ref=_transform_ref(dep_artifact.transform_spec),
                created_at=dep_artifact.created_at,
                input_version=input_version,
            )
            for dep_artifact, input_version in dependent_results[:limit]
        ]

        return ArtifactDependentsResponse(
            artifact_uri=f"strata://artifact/{artifact_id}@v={version}",
            artifact_id=artifact_id,
            version=version,
            dependents=dependents,
            total_count=len(dependent_results),
        )


artifact_service = ArtifactService()
