"""Build-plane services: pure pieces of pull-model manifest assembly.

Handlers keep the mode/transport checks, build lookup and authorization;
``BuildService`` resolves input URIs and assembles the signed-URL manifest. It
is stateless, and handlers map its ``ValueError`` (unresolvable input) to 400.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strata.artifact_store import ArtifactStore
    from strata.transforms.build_store import BuildState
    from strata.transforms.signed_urls import URLSigner


def _resolve_to_artifact_version(
    input_uri: str,
    store: ArtifactStore,
    tenant: str | None = None,
) -> tuple[str, int] | None:
    """Resolve an input URI to ``(artifact_id, version)``, or ``None``.

    Handles ``strata://artifact/{id}@v={n}`` directly and ``strata://name/{name}``
    via the store; any other shape or an unknown name returns ``None``.
    """
    if input_uri.startswith("strata://artifact/"):
        match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", input_uri)
        if match:
            return (match.group(1), int(match.group(2)))
        return None

    if input_uri.startswith("strata://name/"):
        name = input_uri.replace("strata://name/", "")
        artifact = store.resolve_name(name, tenant=tenant)
        if artifact is None:
            return None
        return (artifact.id, artifact.version)

    return None


class BuildService:
    """Stateless build-plane assembly (pull-model manifest)."""

    def derive_build_state(
        self,
        *,
        error_message: str | None,
        completed: bool,
        started: bool,
        artifact_state: str | None,
    ) -> str:
        """Project an identity stream/background build onto the build lifecycle.

        Precedence: a stream error or failed artifact is ``failed``; a ready artifact
        or completed stream is ``ready``; a started stream is ``building``; else
        ``pending``. ``error_message`` beats a ready artifact.
        """
        if error_message or artifact_state == "failed":
            return "failed"
        if artifact_state == "ready" or completed:
            return "ready"
        if started:
            return "building"
        return "pending"

    def assemble_manifest(
        self,
        store: ArtifactStore,
        *,
        signer: URLSigner,
        build: BuildState,
        base_url: str,
        max_output_bytes: int,
        url_expiry_seconds: float,
        lease_owner: str | None = None,
        lease_expires_at: float | None = None,
        presign: bool = False,
    ) -> dict:
        """Resolve a build's inputs and assemble its signed-URL manifest.

        ``presign`` asks for object-store URLs where the store can sign them.

        Raises:
            ValueError: If an input URI cannot be resolved to an artifact version.
        """
        input_artifacts: list[tuple[str, int]] = []
        for input_uri in build.input_uris or []:
            result = _resolve_to_artifact_version(input_uri, store, tenant=build.tenant_id)
            if result is None:
                raise ValueError(f"Cannot resolve input artifact: {input_uri}")
            input_artifacts.append(result)

        metadata = {
            "build_id": build.build_id,
            "artifact_id": build.artifact_id,
            "version": build.version,
            "executor_ref": build.executor_ref,
            "params": build.params or {},
        }

        manifest = signer.generate_build_manifest(
            base_url=base_url,
            build_id=build.build_id,
            metadata=metadata,
            input_artifacts=input_artifacts,
            max_output_bytes=max_output_bytes,
            url_expiry_seconds=url_expiry_seconds,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
            blob_store=store.blob_store if presign else None,
            blob_key=store._blob_key if presign else None,
        )
        return manifest.to_dict()


build_service = BuildService()
