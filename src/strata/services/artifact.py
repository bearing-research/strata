"""Read-side artifact introspection services (lineage, dependents).

Stateless and HTTP-free: the handler resolves the store and tenant, checks
access and shapes the response; the service walks the graph.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, NamedTuple

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


def _leaf_node(input_uri: str, input_version: str, fetched_at: float | None) -> LineageNode:
    """An input that is not an artifact in this store.

    A notebook records a fetched URL against the digest of the bytes it read
    (``sha256:<hex>``), and when the reader recorded the download, that time;
    anything else is a table, versioned by its snapshot.
    """
    if input_version.startswith("sha256:"):
        return LineageNode(
            uri=input_uri,
            type="fetch",
            content_sha256=input_version.removeprefix("sha256:"),
            created_at=fetched_at,
        )
    return LineageNode(uri=input_uri, type="table")


def _transform_ref(transform_spec: str | None) -> str | None:
    """Executor ref from a stored transform_spec, or ``None`` if absent/malformed.

    ``transform_spec`` is client-opaque, so a parse failure means "no known
    transform", not an error.
    """
    if not transform_spec:
        return None
    try:
        return TransformSpec.from_json(transform_spec).executor
    except (json.JSONDecodeError, KeyError):
        return None


class BuildMetadata(NamedTuple):
    """What the producing run recorded about itself.

    Absent for tables, core transforms and older artifacts; an unparseable spec
    reads as "not recorded".
    """

    build_env: str = ""
    build_duration_ms: int = 0
    env_hash: str = ""
    source: str = ""
    # ``{url: unix time}`` each fetched input was downloaded.
    fetched_at: dict[str, float] = {}


def _build_metadata(transform_spec: str | None) -> BuildMetadata:
    """Read the producing run's self-report out of a stored transform_spec."""
    if not transform_spec:
        return BuildMetadata()
    try:
        params = json.loads(transform_spec).get("params", {})
    except (json.JSONDecodeError, ValueError):
        return BuildMetadata()
    if not isinstance(params, dict):
        return BuildMetadata()
    try:
        duration = int(params.get("build_duration_ms") or 0)
    except (TypeError, ValueError):
        duration = 0
    try:
        fetched_at = json.loads(params.get("fetched_at") or "{}")
    except (json.JSONDecodeError, TypeError):
        fetched_at = {}
    return BuildMetadata(
        build_env=str(params.get("build_env") or ""),
        build_duration_ms=duration,
        env_hash=str(params.get("env_hash") or ""),
        source=str(params.get("source") or ""),
        fetched_at=fetched_at if isinstance(fetched_at, dict) else {},
    )


def _fetch_time(meta: BuildMetadata, url: str) -> float | None:
    """When the reader downloaded *url*, or ``None`` if it recorded no usable time."""
    value = meta.fetched_at.get(url)
    return float(value) if isinstance(value, int | float) else None


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

        BFS over ``input_versions`` bounded by ``max_depth``; table inputs become leaf
        nodes. The caller has already checked tenant access and readiness.
        """
        artifact_uri = f"strata://artifact/{artifact_id}@v={version}"
        nodes: dict[str, LineageNode] = {}
        edges: list[LineageEdge] = []
        visited: set[str] = set()
        queue: list[tuple[str, str, int, int]] = []  # (uri, artifact_id, version, depth)

        root_meta = _build_metadata(artifact.transform_spec)
        nodes[artifact_uri] = LineageNode(
            uri=artifact_uri,
            artifact_id=artifact_id,
            version=version,
            type="artifact",
            transform_ref=_transform_ref(artifact.transform_spec),
            created_at=artifact.created_at,
            principal=artifact.principal,
            build_env=root_meta.build_env,
            build_duration_ms=root_meta.build_duration_ms,
            env_hash=root_meta.env_hash,
            source=root_meta.source,
            content_sha256=artifact.content_sha256,
        )
        visited.add(artifact_uri)

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
                visited.add(input_uri)
                nodes[input_uri] = _leaf_node(
                    input_uri, input_version, _fetch_time(root_meta, input_uri)
                )

        # BFS over transitive dependencies.
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

            input_artifact = store.get_artifact(art_id, art_ver)
            # Superseded versions are still readable by id and version and stay in published
            # chains, so they count as steps rather than unknowns.
            if (
                input_artifact is None
                or input_artifact.state not in ("ready", "superseded")
                or (
                    tenant_filter is not None
                    and input_artifact.tenant is not None
                    and input_artifact.tenant != tenant_filter
                )
            ):
                # Bare node: missing, not ready, or another tenant's.
                nodes[node_uri] = LineageNode(
                    uri=node_uri,
                    artifact_id=art_id,
                    version=art_ver,
                    type="artifact",
                )
                continue

            input_meta = _build_metadata(input_artifact.transform_spec)
            nodes[node_uri] = LineageNode(
                uri=node_uri,
                artifact_id=art_id,
                version=art_ver,
                type="artifact",
                transform_ref=_transform_ref(input_artifact.transform_spec),
                created_at=input_artifact.created_at,
                principal=input_artifact.principal,
                build_env=input_meta.build_env,
                build_duration_ms=input_meta.build_duration_ms,
                env_hash=input_meta.env_hash,
                source=input_meta.source,
                content_sha256=input_artifact.content_sha256,
            )

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
                    visited.add(inp_uri)
                    nodes[inp_uri] = _leaf_node(
                        inp_uri, inp_version, _fetch_time(input_meta, inp_uri)
                    )

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
        """List direct (one-hop) dependents of an artifact, in store order.

        The caller has already checked existence, readiness and tenant.
        ``total_count`` counts all dependents; the list is capped at ``limit``.
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
