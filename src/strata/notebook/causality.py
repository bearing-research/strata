"""Causality inspector — explains WHY a cell is stale.

The staleness computation (session.compute_staleness) tells users *that* a cell
is stale. The causality inspector tells them *why*, down to the specific change
that triggered it.

It works by comparing the current provenance components (source hash, input
hashes, env hash) against those stored with the cached artifact. The diff
between old and new components *is* the causality explanation.

The same data also powers "Why did this run?" — same provenance diff, just
rendered in past tense after execution instead of present tense before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from strata.notebook.annotations import parse_annotations
from strata.notebook.env import compute_execution_env_hash, narrow_env_for_provenance
from strata.notebook.provenance import compute_provenance_hash, compute_source_hash
from strata.notebook.workers import worker_runtime_identity

if TYPE_CHECKING:
    from strata.notebook.session import NotebookSession


@dataclass
class CausalityDetail:
    """A single reason contributing to staleness.

    Attributes:
        type: One of 'source_changed', 'input_changed', 'env_changed'
        cell_id: For source/input changes, which cell changed
        cell_name: Human-readable name of the changed cell
        from_version: Old artifact version string (for input_changed)
        to_version: New artifact version string (for input_changed)
        package: Package name (for env_changed)
        from_package_version: Old package version (for env_changed)
        to_package_version: New package version (for env_changed)
    """

    type: str  # 'source_changed' | 'input_changed' | 'env_changed'
    cell_id: str | None = None
    cell_name: str | None = None
    from_version: str | None = None
    to_version: str | None = None
    package: str | None = None
    from_package_version: str | None = None
    to_package_version: str | None = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict, omitting None values."""
        d: dict = {"type": self.type}
        if self.cell_id is not None:
            d["cell_id"] = self.cell_id
        if self.cell_name is not None:
            d["cell_name"] = self.cell_name
        if self.from_version is not None:
            d["from_version"] = self.from_version
        if self.to_version is not None:
            d["to_version"] = self.to_version
        if self.package is not None:
            d["package"] = self.package
        if self.from_package_version is not None:
            d["from_package_version"] = self.from_package_version
        if self.to_package_version is not None:
            d["to_package_version"] = self.to_package_version
        return d


@dataclass
class CausalityChain:
    """Full causality explanation for a stale cell.

    Attributes:
        reason: Primary staleness reason ('self', 'upstream', 'env')
        details: List of specific changes that caused staleness
    """

    reason: str  # 'self' | 'upstream' | 'env'
    details: list[CausalityDetail] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "reason": self.reason,
            "details": [d.to_dict() for d in self.details],
        }


class CausalityInspector:
    """Inspects provenance to explain why cells are stale.

    Compares current provenance components (source hash, input hashes,
    env hash) against those stored in cached artifacts to produce a
    causality chain.
    """

    def __init__(self, session: NotebookSession):
        """Initialize inspector for a session.

        Args:
            session: NotebookSession instance
        """
        self.session = session

    def inspect(self, cell_id: str) -> CausalityChain | None:
        """Explain why a cell is stale using the canonical staleness path."""
        self.session.compute_staleness()
        return self.session.causality_map.get(cell_id)

    def inspect_all(self) -> dict[str, CausalityChain]:
        """Inspect all cells using the canonical staleness path."""
        self.session.compute_staleness()
        return dict(self.session.causality_map)

    def _check_upstream_changes(self, cell_id: str) -> list[CausalityDetail]:
        """Check if upstream cells have changed artifacts.

        Args:
            cell_id: Cell to check upstream changes for

        Returns:
            List of CausalityDetail for changed upstream cells
        """
        details: list[CausalityDetail] = []
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None:
            return details

        for upstream_id in cell.upstream_ids:
            upstream = self.session.notebook_state.get_cell(upstream_id)
            if upstream is None:
                continue

            # If the upstream cell is itself stale, then our inputs have changed
            if upstream.status in ("stale", "idle", "error"):
                details.append(
                    CausalityDetail(
                        type="input_changed",
                        cell_id=upstream_id,
                        cell_name=self._cell_display_name(upstream_id),
                    )
                )
                continue

            # If upstream has run since our last run and produced new artifacts
            if upstream.artifact_uri:
                stored_input_uri = self._get_stored_input_uri(cell_id, upstream_id)
                if stored_input_uri and stored_input_uri != upstream.artifact_uri:
                    details.append(
                        CausalityDetail(
                            type="input_changed",
                            cell_id=upstream_id,
                            cell_name=self._cell_display_name(upstream_id),
                            from_version=stored_input_uri,
                            to_version=upstream.artifact_uri,
                        )
                    )

        return details

    def _get_stored_source_hash(self, cell_id: str) -> str | None:
        """Get the source hash stored with the last artifact for a cell.

        We compute what the source hash was when the artifact was created
        by looking at the provenance metadata.

        Args:
            cell_id: Cell ID

        Returns:
            Stored source hash or None if no artifact exists
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or not cell.artifact_uri:
            return None

        # The artifact exists — its provenance hash was computed from
        # (input_hashes, source_hash, env_hash). We can't decompose the
        # provenance hash, but we can store component hashes separately.
        # For now, check if the overall provenance still matches.
        # If not, we know *something* changed.
        return self._get_artifact_metadata(cell_id, "source_hash")

    def _get_stored_env_hash(self, cell_id: str) -> str | None:
        """Get the env hash stored with the last artifact for a cell.

        Args:
            cell_id: Cell ID

        Returns:
            Stored env hash or None if no artifact exists
        """
        return self._get_artifact_metadata(cell_id, "env_hash")

    def _get_stored_input_uri(self, cell_id: str, upstream_id: str) -> str | None:
        """Get the artifact URI of an upstream cell as stored in our artifact.

        Args:
            cell_id: The cell whose stored input we're checking
            upstream_id: The upstream cell

        Returns:
            Stored artifact URI or None
        """
        # This would ideally come from artifact metadata.
        # For v1.1, we use a simpler heuristic: check if provenance matches.
        return None

    def _get_artifact_metadata(self, cell_id: str, key: str) -> str | None:
        """Get metadata from a cell's stored artifact.

        Reads component hashes (source_hash, env_hash) from the
        artifact's ``transform_spec.params``.

        Args:
            cell_id: Cell ID
            key: Metadata key (e.g. 'source_hash', 'env_hash')

        Returns:
            Value or None
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell is None or not cell.artifact_uri:
            return None

        try:
            parts = cell.artifact_uri.split("/")
            artifact_id = parts[-1].split("@")[0]
            version = int(parts[-1].split("@v=")[1])
            artifact = self.session.artifact_manager.artifact_store.get_artifact(
                artifact_id, version
            )
            if artifact and artifact.transform_spec:
                import json as _json

                spec = _json.loads(artifact.transform_spec)
                return spec.get("params", {}).get(key)
        except (IndexError, ValueError, KeyError):
            pass
        return None

    def _cell_display_name(self, cell_id: str) -> str:
        """Get a human-readable name for a cell.

        Args:
            cell_id: Cell ID

        Returns:
            Display name (first defined variable, or cell ID)
        """
        cell = self.session.notebook_state.get_cell(cell_id)
        if cell and cell.defines:
            return cell.defines[0]
        return cell_id


def compute_causality_on_staleness(
    session: NotebookSession,
) -> dict[str, CausalityChain]:
    """Compute causality chains for all cells during staleness detection.

    This is called alongside compute_staleness() to provide causality
    explanations for stale cells. It uses the same topological walk and
    provenance comparison, but extracts component-level diffs.

    Args:
        session: NotebookSession instance

    Returns:
        Dict mapping cell_id -> CausalityChain for stale cells
    """
    if session.dag is None:
        return {}

    causality_map: dict[str, CausalityChain] = {}

    for cell_id in session.dag.topological_order:
        cell = session.notebook_state.get_cell(cell_id)
        if cell is None:
            continue

        details: list[CausalityDetail] = []
        annotations = parse_annotations(cell.source)
        source_hash = compute_source_hash(cell.source)
        runtime_env = dict(cell.env)
        runtime_env.update(annotations.env)
        declared_env_keys = set(annotations.env) | set(cell.env_overrides or {})
        provenance_env = narrow_env_for_provenance(cell.source, runtime_env, declared_env_keys)
        effective_worker = annotations.worker or cell.worker or session.notebook_state.worker
        env_hash = compute_execution_env_hash(
            session.path,
            provenance_env,
            runtime_identity=worker_runtime_identity(
                session.notebook_state,
                effective_worker,
            ),
        )
        mount_fingerprints, has_rw_mount = session._collect_mount_fingerprints(cell)

        if has_rw_mount:
            # RW mounts are intentionally non-cacheable side effects.
            # They should remain stale/idle without pretending there is a
            # meaningful cached provenance explanation.
            continue

        # Compute current provenance — use per-variable artifact_uris
        input_hashes: list[str] = []
        for upstream_id in cell.upstream_ids:
            upstream = session.notebook_state.get_cell(upstream_id)
            if upstream is None:
                continue
            uris = list(upstream.artifact_uris.values())
            if not uris and upstream.artifact_uri:
                uris = [upstream.artifact_uri]
            for uri in sorted(uris):
                try:
                    parts = uri.split("/")
                    artifact_id = parts[-1].split("@")[0]
                    version = int(parts[-1].split("@v=")[1])
                    artifact = session.artifact_manager.artifact_store.get_artifact(
                        artifact_id, version
                    )
                    if artifact:
                        input_hashes.append(artifact.provenance_hash)
                except (IndexError, ValueError):
                    pass

        provenance_hash = compute_provenance_hash(
            input_hashes + mount_fingerprints, source_hash, env_hash
        )

        if session._resolve_cached_outputs(cell_id, provenance_hash) is not None:
            # Cell is ready — no causality needed
            continue

        # Cell is stale — figure out why
        # Check upstream cells
        for upstream_id in cell.upstream_ids:
            upstream = session.notebook_state.get_cell(upstream_id)
            if upstream is None:
                continue

            # If upstream is stale or has no artifact, our inputs changed
            if upstream.status in ("stale", "idle", "error"):
                upstream_name = upstream.defines[0] if upstream.defines else upstream_id
                details.append(
                    CausalityDetail(
                        type="input_changed",
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )
            # If upstream ran and produced a new artifact since our last run
            elif upstream_id in causality_map:
                # Upstream itself changed — so our inputs changed transitively
                upstream_name = upstream.defines[0] if upstream.defines else upstream_id
                details.append(
                    CausalityDetail(
                        type="input_changed",
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )

        # If no upstream changes detected, it must be source or env
        if not details:
            # Try to decompose by reading stored component hashes
            stored_source_hash = _get_stored_hash(session, cell_id, "source_hash")
            stored_env_hash = _get_stored_hash(session, cell_id, "env_hash")

            if stored_source_hash is None:
                stored_source_hash = cell.last_source_hash
            if stored_env_hash is None:
                stored_env_hash = cell.last_env_hash

            source_changed = stored_source_hash is not None and stored_source_hash != source_hash
            env_changed_flag = stored_env_hash is not None and stored_env_hash != env_hash

            if env_changed_flag:
                details.append(
                    CausalityDetail(
                        type="env_changed",
                        package="notebook env",
                    )
                )
            if (
                not source_changed
                and not env_changed_flag
                and cell.last_provenance_hash is not None
                and cell.last_provenance_hash != provenance_hash
                and cell.upstream_ids
            ):
                upstream_id = cell.upstream_ids[0]
                upstream = session.notebook_state.get_cell(upstream_id)
                upstream_name = (
                    upstream.defines[0]
                    if upstream is not None and upstream.defines
                    else upstream_id
                )
                details.append(
                    CausalityDetail(
                        type="input_changed",
                        cell_id=upstream_id,
                        cell_name=upstream_name,
                    )
                )
            if source_changed or (not env_changed_flag):
                # If source changed, or if we couldn't determine the cause
                # (no stored hashes), fall back to source_changed
                if not details:
                    cell_name = cell.defines[0] if cell.defines else cell_id
                    details.append(
                        CausalityDetail(
                            type="source_changed",
                            cell_id=cell_id,
                            cell_name=cell_name,
                        )
                    )

        # Determine primary reason — env takes precedence when it's the
        # *only* change, since source_changed may be a fallback guess.
        has_source = any(d.type == "source_changed" for d in details)
        has_input = any(d.type == "input_changed" for d in details)
        has_env = any(d.type == "env_changed" for d in details)

        if has_env and not has_source and not has_input:
            reason = "env"
        elif has_input:
            reason = "upstream"
        else:
            reason = "self"

        causality_map[cell_id] = CausalityChain(reason=reason, details=details)

    return causality_map


def _get_stored_hash(session: NotebookSession, cell_id: str, key: str) -> str | None:
    """Read a component hash from a cell's stored artifact metadata.

    Args:
        session: NotebookSession
        cell_id: Cell ID
        key: 'source_hash' or 'env_hash'

    Returns:
        Stored hash string or None if not available
    """
    cell = session.notebook_state.get_cell(cell_id)
    if cell is None or not cell.artifact_uri:
        return None

    try:
        import json as _json

        parts = cell.artifact_uri.split("/")
        artifact_id = parts[-1].split("@")[0]
        version = int(parts[-1].split("@v=")[1])
        artifact = session.artifact_manager.artifact_store.get_artifact(artifact_id, version)
        if artifact and artifact.transform_spec:
            spec = _json.loads(artifact.transform_spec)
            return spec.get("params", {}).get(key)
    except (IndexError, ValueError, KeyError):
        pass
    return None
