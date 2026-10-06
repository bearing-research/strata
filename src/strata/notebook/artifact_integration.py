"""Bridge between notebook execution and the Strata artifact store.

Artifact ids::

    Regular cell output: nb_{notebook_id}_cell_{cell_id}_var_{variable_name}
    Loop iteration:      nb_{notebook_id}_cell_{cell_id}_var_{variable_name}@iter={k}

The ``@iter={k}`` suffix lets downstream cells, the inspector and
``@loop start_from=<cell>@iter=<k>`` address one iteration by name.
"""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from strata.artifact_store import ArtifactStore, StagedVersion, TransformSpec
from strata.auth import get_principal
from strata.notebook.models import ArtifactInfo
from strata.notebook.quiesce import assert_writable

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactVersion


# A cell run creates and finalizes its outputs in one pass, and no build runner writes
# here, so a ``building`` row this old was left by a crash.
_ABANDONED_BUILD_SECONDS = 3600.0


class _Default(Enum):
    CALLER = "caller"


def caller_principal_id() -> str | None:
    """The id of the member a run is for: set in service mode, ``None`` in personal mode.

    A run is a task the triggering request created, which copies that request's
    context, so this is the member who asked even after their request has returned.
    """
    principal = get_principal()
    return principal.id if principal is not None else None


class NotebookArtifactManager:
    """Notebook-specific operations over ``ArtifactStore`` and ``BlobStore``."""

    def __init__(
        self,
        notebook_id: str,
        artifact_dir: Path | None = None,
    ):
        """Open the notebook's artifact store.

        *artifact_dir* defaults to ``~/.strata/notebook_artifacts``.
        """
        self.notebook_id = notebook_id

        if artifact_dir is None:
            artifact_dir = Path.home() / ".strata" / "notebook_artifacts" / notebook_id

        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir = artifact_dir

        self.artifact_store = ArtifactStore(artifact_dir)

    def prune(self, keep_superseded: int, min_idle_seconds: float) -> dict:
        """Drop each cell output's values older than its newest *keep_superseded*.

        Keeps every output's current value, anything named, pinned or published, and
        anything used in the last *min_idle_seconds*. Values a crash left half-written
        are failed first, so they no longer hold their inputs.

        Raises:
            NotebookQuiesced: the notebook is held still for a copy, which
                must see its store as it was.
        """
        from strata.notebook.quiesce import assert_writable

        assert_writable(self.artifact_dir)
        self.artifact_store.sweep_zombie_builds(_ABANDONED_BUILD_SECONDS)
        return self.artifact_store.garbage_collect(
            keep_superseded=keep_superseded, min_idle_seconds=min_idle_seconds
        )

    def find_cached(self, provenance_hash: str) -> ArtifactVersion | None:
        """Find a cached artifact by provenance hash, or None."""
        return self.artifact_store.find_by_provenance(provenance_hash)

    def cell_artifact_id(
        self,
        cell_id: str,
        variable_name: str,
        iteration: int | None = None,
        variant: str | None = None,
    ) -> str:
        """Canonical artifact id for a notebook cell's variable.

        ``iteration`` appends ``@iter={k}``; ``variant`` (a ``# @per_variant`` instance)
        appends ``@variant={name}``. They do not co-occur in practice; if both are
        given, ``@iter`` comes first.
        """
        base = f"nb_{self.notebook_id}_cell_{cell_id}_var_{variable_name}"
        if iteration is not None:
            base = f"{base}@iter={iteration}"
        if variant is not None:
            base = f"{base}@variant={variant}"
        return base

    def load_iteration_blob(
        self,
        cell_id: str,
        variable_name: str,
        iteration: int,
    ) -> bytes | None:
        """Load the latest blob bytes for one loop iteration, or ``None`` if absent."""
        artifact_id = self.cell_artifact_id(cell_id, variable_name, iteration)
        latest = self.artifact_store.get_latest_version(artifact_id)
        if latest is None or latest.state not in ("ready", "superseded"):
            return None
        return self.artifact_store.read_blob(artifact_id, latest.version)

    def get_iteration_artifact(
        self,
        cell_id: str,
        variable_name: str,
        iteration: int,
    ) -> ArtifactVersion | None:
        """Return the latest ArtifactVersion for a specific loop iteration."""
        artifact_id = self.cell_artifact_id(cell_id, variable_name, iteration)
        latest = self.artifact_store.get_latest_version(artifact_id)
        if latest is None or latest.state not in ("ready", "superseded"):
            return None
        return latest

    def list_iterations(
        self,
        cell_id: str,
        variable_name: str,
    ) -> list[tuple[int, ArtifactVersion]]:
        """Return ``(iteration, ArtifactVersion)`` for every stored iteration, ascending."""
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

    def list_variants(
        self,
        cell_id: str,
        variable_name: str,
    ) -> list[tuple[str, ArtifactVersion]]:
        """Return ``(variant_name, ArtifactVersion)`` for every stored fan-out instance, by name."""
        prefix = self.cell_artifact_id(cell_id, variable_name) + "@variant="
        artifacts = self.artifact_store.list_latest_by_id_prefix(prefix)
        results: list[tuple[str, ArtifactVersion]] = []
        for artifact in artifacts:
            variant = artifact.id[len(prefix) :]
            if variant:
                results.append((variant, artifact))
        results.sort(key=lambda pair: pair[0])
        return results

    def store_cell_output(self, **kwargs: Any) -> ArtifactVersion:
        """Store one cell output and make it current; arguments as for :meth:`stage_cell_output`."""
        return self.finalize_cell_outputs([self.stage_cell_output(**kwargs)])[0]

    def stage_cell_output(
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
        source: str = "",
        env_hash: str = "",
        iteration: int | None = None,
        variant: str | None = None,
        build_env: str = "",
        build_duration_ms: float = 0.0,
        principal: str | None | Literal[_Default.CALLER] = _Default.CALLER,
        hardware: dict[str, Any] | None = None,
        extra_params: dict[str, str] | None = None,
    ) -> StagedVersion:
        """Write a cell output's bytes as a ``building`` version nobody reads yet.

        :meth:`finalize_cell_outputs` makes it current, together with the rest of its run.

        Args:
            input_versions: Mapping of input URI to version.
            source: The cell source that produced these bytes. Recorded, never
                hashed (``source_hash`` already keys it). Captured at execution, not
                read back later, because the cell may have been edited since.
            iteration: Loop iteration; suffixes the id with ``@iter={k}``.
            variant: ``# @per_variant`` instance; suffixes the id with
                ``@variant={name}``.
            build_env: Interpreter and platform that produced the bytes (see
                ``harness.build_env_identity``). Recorded, never hashed, so
                cross-machine cache hits still work. Empty when not reported.
            build_duration_ms: How long the producing run took; lets a shared
                cache say what a hit saved. Zero when unrecorded.
            hardware: The machine a worker reported (``strata.notebook.hardware``).
                Recorded, never hashed, like ``build_env``. Absent for a local run.
            principal: Who computed the bytes. Defaults to the member the run is
                for (``caller_principal_id``); a result pulled from a shared store
                passes its publisher, even when that is ``None``.
        """
        # Artifacts live under the notebook dir: a write during a copy would
        # leave the copy's runtime.json and artifacts out of step.
        assert_writable(self.artifact_dir)
        artifact_id = self.cell_artifact_id(cell_id, variable_name, iteration, variant)
        if principal is _Default.CALLER:
            principal = caller_principal_id()

        params: dict[str, str] = {
            "cell_id": cell_id,
            "variable_name": variable_name,
            "content_type": content_type,
        }
        if iteration is not None:
            params["iteration"] = str(iteration)
        if variant is not None:
            params["variant"] = variant
        if source_hash:
            params["source_hash"] = source_hash
        if source:
            params["source"] = source
        if env_hash:
            params["env_hash"] = env_hash
        if build_env:
            params["build_env"] = build_env
        if build_duration_ms > 0:
            params["build_duration_ms"] = str(int(build_duration_ms))
        if hardware:
            params["hardware"] = json.dumps(hardware, sort_keys=True)
        if extra_params:
            params.update(extra_params)

        transform_spec = TransformSpec(
            executor="notebook/cell@v1",
            params=params,
            inputs=[],  # cell inputs are not URIs
        )

        version = self.artifact_store.create_artifact(
            artifact_id=artifact_id,
            provenance_hash=provenance_hash,
            transform_spec=transform_spec,
            input_versions=input_versions,
            principal=principal,
        )

        self.artifact_store.blob_store.write_blob(artifact_id, version, blob_data)

        return StagedVersion(
            artifact_id=artifact_id,
            version=version,
            schema_json=schema_json if schema_json is not None else "",
            row_count=row_count or 0,
            byte_size=len(blob_data),
            content_sha256=hashlib.sha256(blob_data).hexdigest(),
        )

    def finalize_cell_outputs(self, staged: list[StagedVersion]) -> list[ArtifactVersion]:
        """Make one cell run's staged outputs current together, or none of them.

        Each becomes the value of its own id even when another id already holds the
        same provenance, because cells resolve their inputs by canonical id.
        """
        finalized = self.artifact_store.finalize_canonical_together(staged)
        for item, version in zip(staged, finalized, strict=True):
            if version is None:
                raise ValueError(f"Failed to finalize artifact {item.artifact_id}@v={item.version}")
        return [version for version in finalized if version is not None]

    def discard_cell_outputs(self, staged: list[StagedVersion]) -> None:
        """Mark staged outputs failed, for a run whose outputs will not all be stored."""
        for item in staged:
            self.artifact_store.fail_artifact(item.artifact_id, item.version)

    def load_artifact_data(self, artifact_id: str, version: int) -> bytes:
        """Load an artifact version's blob bytes.

        Raises:
            ValueError: If the artifact is not found or not ready.
        """
        artifact = self.artifact_store.get_artifact(artifact_id, version)
        if artifact is None or artifact.state not in ("ready", "superseded"):
            raise ValueError(f"Artifact {artifact_id}@v={version} not found or not ready")

        blob_data = self.artifact_store.read_blob(artifact_id, version)
        if blob_data is None:
            raise ValueError(f"Blob data not found for {artifact_id}@v={version}")
        return blob_data

    def get_artifact_preview(self, artifact_id: str, version: int) -> dict[str, Any]:
        """Get artifact metadata and a data preview.

        Returns a dict with ``id``, ``version``, ``content_type``, ``rows``,
        ``bytes`` and ``preview``.

        Raises:
            ValueError: If the artifact is not found.
        """
        artifact = self.artifact_store.get_artifact(artifact_id, version)
        if artifact is None:
            raise ValueError(f"Artifact {artifact_id}@v={version} not found")

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
        """List a cell's canonical artifacts: one ``(variable_name, latest version)`` per variable.

        Loop-iteration ids (``...@iter=k``) are excluded; use ``list_iterations``.
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

    def cell_output_digests(self, cell_id: str) -> list[dict[str, Any]]:
        """Each of a cell's outputs with the digest of its bytes, sorted by name.

        A row stored before digests existed is filled from its blob here; ``None``
        only when it has no blob. A leaf's ``__console__`` record is not an output.
        """
        return [
            {
                "name": name,
                "artifact_id": artifact.id,
                "version": artifact.version,
                "content_sha256": self.artifact_store.content_digest(artifact.id, artifact.version),
            }
            for name, artifact in sorted(self.list_cell_artifacts(cell_id))
            if name != "__console__"
        ]

    def get_artifact_info(self, artifact_id: str, version: int) -> ArtifactInfo | None:
        """Get lightweight artifact info for API responses."""
        artifact = self.artifact_store.get_artifact(artifact_id, version)
        if artifact is None:
            return None

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
