"""Build-plane services (pull-model manifest assembly).

The build routes keep their orchestration in the handler — mode/transport
checks, build-store lookup, and the post-fetch ``_authorize_build_access`` all
stay there. ``BuildService`` holds the pure pieces: resolving a build's input
URIs to ``(artifact_id, version)`` pairs and assembling the signed-URL manifest.
Stateless; the resolved artifact store and config-derived limits are passed in.
A handler maps the ``ValueError`` raised here (unresolvable input) to its 400.
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
    """Resolve an input URI to an ``(artifact_id, version)`` tuple, or ``None``.

    Handles ``strata://artifact/{id}@v={n}`` directly and ``strata://name/{name}``
    via the store; any other shape (or an unknown name) returns ``None``.
    """
    # Artifact URI: strata://artifact/{id}@v={version}
    if input_uri.startswith("strata://artifact/"):
        match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", input_uri)
        if match:
            return (match.group(1), int(match.group(2)))
        return None

    # Name URI: strata://name/{name}
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

        Precedence (highest first): a stream error or a failed artifact ⇒
        ``failed``; a ready artifact or a completed stream ⇒ ``ready``; a started
        stream ⇒ ``building``; otherwise ``pending``. ``error_message`` wins over
        a ready artifact, so a half-failed build never reports ``ready``.
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
    ) -> dict:
        """Resolve a build's inputs and assemble its signed-URL manifest.

        Raises:
            ValueError: an input URI cannot be resolved to an artifact version.
                The handler maps this to a 400.
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
        )
        return manifest.to_dict()


build_service = BuildService()
