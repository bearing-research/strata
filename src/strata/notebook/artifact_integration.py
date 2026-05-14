"""Bridge between notebook execution and Strata artifact store.

This module manages artifact storage and retrieval for notebook cells,
using the existing ArtifactStore and BlobStore classes.

Artifact ID scheme:
    Regular cell output: nb_{notebook_id}_cell_{cell_id}_var_{variable_name}
    Loop iteration:      nb_{notebook_id}_cell_{cell_id}_var_{variable_name}@iter={k}

The ``@iter={k}`` suffix keeps each iteration of a loop cell addressable
as its own artifact, so downstream cells, the inspector, and forking
(``@loop start_from=<cell>@iter=<k>``) can all refer to a specific
iteration by name.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strata.artifact_store import ArtifactStore, TransformSpec
from strata.notebook.models import ArtifactInfo

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactVersion


class NotebookArtifactManager:
    """Manages artifacts for a notebook session.

    Wraps the existing ArtifactStore + BlobStore to provide
    notebook-specific operations.
    """

    def __init__(
        self,
        notebook_id: str,
        artifact_dir: Path | None = None,
    ):
        """Initialize the artifact manager for a notebook.

        Args:
            notebook_id: ID of the notebook
            artifact_dir: Directory for artifacts. If None, uses ~/.strata/notebook_artifacts
        """
        self.notebook_id = notebook_id

        # Default artifact directory
        if artifact_dir is None:
            artifact_dir = Path.home() / ".strata" / "notebook_artifacts" / notebook_id

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Initialize artifact store with local blob storage
        self.artifact_store = ArtifactStore(artifact_dir)

    def find_cached(self, provenance_hash: str) -> ArtifactVersion | None:
        """Find a cached artifact by provenance hash.

        Args:
            provenance_hash: The provenance hash to search for

        Returns:
            ArtifactVersion if found, None otherwise
        """
        return self.artifact_store.find_by_provenance(provenance_hash)

    def cell_artifact_id(
        self,
        cell_id: str,
        variable_name: str,
        iteration: int | None = None,
    ) -> str:
        """Canonical artifact id for a notebook cell's variable.

        Args:
            cell_id: Cell ID.
            variable_name: Variable name.
            iteration: Optional iteration index. When supplied, the id is
                suffixed with ``@iter={k}`` so loop iterations can be
                stored and retrieved independently.
        """
        base = f"nb_{self.notebook_id}_cell_{cell_id}_var_{variable_name}"
        if iteration is None:
            return base
        return f"{base}@iter={iteration}"

    def load_iteration_blob(
        self,
        cell_id: str,
        variable_name: str,
        iteration: int,
    ) -> bytes | None:
        """Load the latest blob bytes for a specific loop iteration.

        Returns ``None`` if no artifact exists for that iteration.
        """
        artifact_id = self.cell_artifact_id(cell_id, variable_name, iteration)
        latest = self.artifact_store.get_latest_version(artifact_id)
        if latest is None or latest.state != "ready":
            return None
        return self.artifact_store.blob_store.read_blob(artifact_id, latest.version)

    def get_iteration_artifact(
        self,
        cell_id: str,
        variable_name: str,
        iteration: int,
    ) -> ArtifactVersion | None:
        """Return the latest ArtifactVersion for a specific loop iteration."""
        artifact_id = self.cell_artifact_id(cell_id, variable_name, iteration)
        latest = self.artifact_store.get_latest_version(artifact_id)
        if latest is None or latest.state != "ready":
            return None
        return latest

    def list_iterations(
        self,
        cell_id: str,
        variable_name: str,
    ) -> list[tuple[int, ArtifactVersion]]:
        """Return ``(iteration, ArtifactVersion)`` for every stored iteration
        of ``cell_id``'s ``variable_name``, sorted ascending by iteration.

        Powers the iteration picker in the inspect panel: the caller can
        build a dropdown of (k, artifact metadata) without needing to
        probe the store for each possible index.
        """
        prefix = self.cell_artifact_id(cell_id, variable_name) + "@iter="
        artifacts = self.artifact_store.list_latest_by_id_prefix(prefix)
        results: list[tuple[int, ArtifactVersion]] = []
        for artifact in artifacts:
            suffix = artifact.id[len(prefix) :]
            try:
                iteration = int(suffix)
            except ValueError:
                continue
            results.append((iteration, artifact))
        results.sort(key=lambda pair: pair[0])
        return results

    def store_cell_output(
        self,
        cell_id: str,
        variable_name: str,
        blob_data: bytes,
        content_type: str,
        schema_json: str | None = None,
        row_count: int | None = None,
        provenance_hash: str = "",
        input_versions: dict[str, str] | None = None,
        source_hash: str = "",
        env_hash: str = "",
        iteration: int | None = None,
    ) -> ArtifactVersion:
        """Store a cell output as an artifact.

        Args:
            cell_id: ID of the cell
            variable_name: Name of the output variable
            blob_data: Serialized blob (Arrow IPC bytes, JSON bytes, pickle bytes)
            content_type: Content type (arrow/ipc, json/object, pickle/object)
            schema_json: Arrow schema as JSON (for arrow/ipc only)
            row_count: Number of rows (for tables)
            provenance_hash: Provenance hash for deduplication
            input_versions: Mapping of input URI -> version
            source_hash: SHA-256 of cell source code (for causality tracking)
            env_hash: SHA-256 of lockfile (for causality tracking)
            iteration: Optional loop iteration index. When set, the artifact
                id is suffixed with ``@iter={k}`` so each loop iteration is
                stored as its own artifact.

        Returns:
            The created ArtifactVersion
        """
        artifact_id = self.cell_artifact_id(cell_id, variable_name, iteration)

        # Create transform spec (for notebook cells, executor is "notebook/cell@v1")
        params: dict[str, str] = {
            "cell_id": cell_id,
            "variable_name": variable_name,
            "content_type": content_type,
        }
        if iteration is not None:
            params["iteration"] = str(iteration)
        if source_hash:
            params["source_hash"] = source_hash
        if env_hash:
            params["env_hash"] = env_hash

        transform_spec = TransformSpec(
            executor="notebook/cell@v1",
            params=params,
            inputs=[],  # Notebook cells don't have explicit input URIs in this context
        )

        # Create artifact version in "building" state
        version = self.artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            transform_spec=transform_spec,
            input_versions=input_versions,
        )

        # Write blob
        self.artifact_store.blob_store.write_blob(artifact_id, version, blob_data)

        # Finalize artifact
        byte_size = len(blob_data)
        # Provide empty schema JSON if not supplied
        schema_str = schema_json if schema_json is not None else ""
        artifact_version = self.artifact_store.finalize_artifact(
            artifact_id=artifact_id,
            version=version,
            schema_json=schema_str,
            row_count=row_count or 0,
            byte_size=byte_size,
        )

        if artifact_version is None:
            raise ValueError(f"Failed to finalize artifact {artifact_id}@v={version}")

        # Guard against provenance dedup: finalize_artifact may detect
        # another artifact with the same provenance under a *different*
        # id and mark our version as "failed". Notebook cells resolve
        # inputs by canonical id, so we promote the canonical version
        # back to ready via the store's public force_finalize_canonical
        # helper. (Previously this reached into the store's private
        # SQLite connection — keeping the SQL inside ArtifactStore so
        # schema changes can't drift between the two definitions.)
        if artifact_version.id != artifact_id:
            canonical = self.artifact_store.force_finalize_canonical(
                artifact_id=artifact_id,
                version=version,
                schema_json=schema_str,
                row_count=row_count or 0,
                byte_size=byte_size,
            )
            if canonical is not None:
                artifact_version = canonical

        return artifact_version

    def load_artifact_data(self, artifact_id: str, version: int) -> bytes:
        """Load artifact blob data.

        Args:
            artifact_id: Artifact ID
            version: Version number

        Returns:
            Blob data (bytes)

        Raises:
            ValueError: If artifact not found or not ready
        """
        artifact = self.artifact_store.get_artifact(artifact_id, version)
        if artifact is None or artifact.state != "ready":
            raise ValueError(f"Artifact {artifact_id}@v={version} not found or not ready")

        blob_data = self.artifact_store.blob_store.read_blob(artifact_id, version)
        if blob_data is None:
            raise ValueError(f"Blob data not found for {artifact_id}@v={version}")
        return blob_data

    def get_artifact_preview(self, artifact_id: str, version: int) -> dict[str, Any]:
        """Get artifact metadata and a data preview.

        Args:
            artifact_id: Artifact ID
            version: Version number

        Returns:
            Dict with: id, version, content_type, rows, bytes, preview

        Raises:
            ValueError: If artifact not found
        """
        artifact = self.artifact_store.get_artifact(artifact_id, version)
        if artifact is None:
            raise ValueError(f"Artifact {artifact_id}@v={version} not found")

        # Extract content_type from transform_spec params
        content_type = "unknown"
        if artifact.transform_spec:
            try:
                spec = json.loads(artifact.transform_spec)
                content_type = spec.get("params", {}).get("content_type", "unknown")
            except (ValueError, KeyError):
                pass

        return {
            "id": artifact.id,
            "version": artifact.version,
            "content_type": content_type,
            "rows": artifact.row_count,
            "bytes": artifact.byte_size,
            "created_at": artifact.created_at,
        }

    def list_cell_artifacts(self, cell_id: str) -> list[tuple[str, ArtifactVersion]]:
        """List canonical artifacts for a cell — one per variable, latest version.

        Returns ``(variable_name, ArtifactVersion)`` for each canonical
        ``nb_{notebook_id}_cell_{cell_id}_var_<name>`` artifact. Loop-
        iteration suffixed ids (``...@iter=k``) are excluded — use
        ``list_iterations`` when iteration-level granularity is wanted.
        """
        prefix = f"nb_{self.notebook_id}_cell_{cell_id}_var_"
        artifacts = self.artifact_store.list_latest_by_id_prefix(prefix)
        results: list[tuple[str, ArtifactVersion]] = []
        for artifact in artifacts:
            suffix = artifact.id[len(prefix) :]
            if "@iter=" in suffix:
                continue
            results.append((suffix, artifact))
        return results

    def get_artifact_info(self, artifact_id: str, version: int) -> ArtifactInfo | None:
        """Get lightweight artifact info for API responses."""
        artifact = self.artifact_store.get_artifact(artifact_id, version)
        if artifact is None:
            return None

        # Content type lives in transform_spec.params — same source
        # get_artifact_preview reads from.
        content_type = "unknown"
        if artifact.transform_spec:
            try:
                spec = json.loads(artifact.transform_spec)
                content_type = spec.get("params", {}).get("content_type", "unknown")
            except (ValueError, KeyError):
                pass

        return ArtifactInfo(
            id=artifact.id,
            version=artifact.version,
            provenance_hash=artifact.provenance_hash,
            content_type=content_type,
            rows=artifact.row_count,
            bytes=artifact.byte_size or 0,
            created_at=artifact.created_at or 0.0,
        )
