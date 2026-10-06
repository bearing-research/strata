"""HMAC-signed capability URLs for the pull-model executor protocol.

Executors fetch inputs and upload output directly through short-lived URLs, keeping the
data plane off Strata. Each URL signs (HMAC-SHA256) its operation, resource ids, expiry
and, for uploads, a size limit: the signature prevents tampering and the expiry bounds
replay.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode

from strata.artifact_store import attempt_blob_id


def lease_token(lease_owner: str | None, lease_expires_at: float | None) -> str:
    """Render a claim as the string a finalize signature covers; empty without a lease.

    The deadline makes it per-claim, not per-owner: a reclaim or a re-issued manifest moves
    the deadline, so capabilities minted for the previous claim stop verifying. ``repr`` on
    the float so the URL's token and the one rebuilt from the stored row are the same string.
    """
    if not lease_owner or lease_expires_at is None:
        return ""
    return f"{lease_owner}:{lease_expires_at!r}"


def lease_attempt(lease: str) -> str | None:
    """The blob attempt id an executor holding ``lease`` writes under.

    Each manifest renews the lease and so gets its own attempt: a holder of an earlier
    manifest can upload, but only to a key finalize never reads. A digest because
    ``owner:deadline`` is not a safe blob key. ``None`` without a lease (the shared key).
    """
    if not lease:
        return None
    return hashlib.sha256(lease.encode()).hexdigest()[:32]


@dataclass(frozen=True)
class SignedDownloadURL:
    """Signed URL for downloading an input artifact; ``expires_at`` is epoch seconds."""

    url: str
    artifact_id: str
    version: int
    expires_at: float


@dataclass(frozen=True)
class SignedUploadURL:
    """Signed URL for uploading build output up to ``max_bytes``; ``expires_at`` is epoch secs."""

    url: str
    build_id: str
    max_bytes: int
    expires_at: float
    fields: dict[str, str] | None = None
    """Form fields to POST with the body as the ``file`` part, when ``url`` is a
    presigned object-store upload rather than a Strata route. ``None`` means
    send the raw body to ``url``."""
    method: str = "POST"
    """``PUT`` for a presigned upload that takes no form (Azure), sent with ``headers``."""
    headers: dict[str, str] | None = None


@dataclass(frozen=True)
class SignedFinalizeURL:
    """Signed URL for finalizing a build output; ``expires_at`` is epoch seconds."""

    url: str
    build_id: str
    expires_at: float


@dataclass(frozen=True)
class BuildManifest:
    """Signed URLs handed to an executor to pull each input, push the output, and finalize.

    ``log_url`` is optional for the executor; ignoring it changes nothing else.
    """

    build_id: str
    metadata: dict[str, Any]
    input_urls: list[SignedDownloadURL]
    output_url: SignedUploadURL
    finalize_url: str
    log_url: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON wire shape sent to the executor.

        ``{build_id, metadata, inputs, output, finalize_url, log_url}``; ``output`` omits the
        ``build_id`` already carried at the top level.
        """
        output = asdict(self.output_url)
        del output["build_id"]
        for optional in ("fields", "headers"):
            if output[optional] is None:
                del output[optional]
        return {
            "build_id": self.build_id,
            "metadata": self.metadata,
            "inputs": [asdict(url) for url in self.input_urls],
            "output": output,
            "finalize_url": self.finalize_url,
            "log_url": self.log_url,
        }


class URLSigner:
    """Signs and verifies pull-model capability URLs with one HMAC secret.

    Use a stable, high-entropy ``secret`` in production so URLs survive restarts and match
    across replicas. A verifier must coerce query parameters back to their signed types
    (``version`` int, ``expires_at`` float) before calling ``verify_*``, or verification fails.
    """

    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def _sign(self, data: dict[str, Any]) -> str:
        """Return the URL-safe base64 HMAC-SHA256 of ``data`` as key-sorted JSON."""
        message = json.dumps(data, sort_keys=True).encode()
        signature = hmac.new(self._secret, message, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(signature).decode()

    def _verify(self, data: dict[str, Any], signature: str) -> bool:
        """Check a signature against ``data`` in constant time."""
        expected = self._sign(data)
        # Bytes, not str: ``signature`` arrives from a query parameter, and
        # ``compare_digest`` raises TypeError comparing non-ASCII strings.
        return hmac.compare_digest(expected.encode(), signature.encode())

    def generate_download_url(
        self,
        base_url: str,
        artifact_id: str,
        version: int,
        build_id: str,
        expiry_seconds: float = 300.0,
    ) -> SignedDownloadURL:
        """Sign a URL for downloading an artifact (default validity 300 s).

        ``build_id`` is recorded for audit, not checked for access.
        """
        expires_at = time.time() + expiry_seconds
        data = {
            "op": "download",
            "artifact_id": artifact_id,
            "version": version,
            "build_id": build_id,
            "expires_at": expires_at,
        }
        params = {
            "artifact_id": artifact_id,
            "version": str(version),
            "build_id": build_id,
            "expires_at": str(expires_at),
            "signature": self._sign(data),
        }
        url = f"{base_url}/v1/artifacts/download?{urlencode(params)}"
        return SignedDownloadURL(
            url=url,
            artifact_id=artifact_id,
            version=version,
            expires_at=expires_at,
        )

    def verify_download_signature(
        self,
        artifact_id: str,
        version: int,
        build_id: str,
        expires_at: float,
        signature: str,
    ) -> bool:
        """True when a download URL's signature is valid and it has not expired."""
        if time.time() > expires_at:
            return False
        data = {
            "op": "download",
            "artifact_id": artifact_id,
            "version": version,
            "build_id": build_id,
            "expires_at": expires_at,
        }
        return self._verify(data, signature)

    def generate_upload_url(
        self,
        base_url: str,
        build_id: str,
        max_bytes: int,
        expiry_seconds: float = 600.0,
        attempt: str | None = None,
    ) -> SignedUploadURL:
        """Sign a URL for uploading build output (default validity 600 s).

        ``max_bytes`` and ``attempt`` (from ``lease_attempt``) are signed, so the size cap cannot
        be raised and the upload cannot target another attempt's key.
        """
        expires_at = time.time() + expiry_seconds
        data: dict[str, Any] = {
            "op": "upload",
            "build_id": build_id,
            "max_bytes": max_bytes,
            "expires_at": expires_at,
        }
        params = {
            "build_id": build_id,
            "max_bytes": str(max_bytes),
            "expires_at": str(expires_at),
        }
        if attempt:
            data["attempt"] = attempt
            params["attempt"] = attempt
        params["signature"] = self._sign(data)
        url = f"{base_url}/v1/artifacts/upload?{urlencode(params)}"
        return SignedUploadURL(
            url=url,
            build_id=build_id,
            max_bytes=max_bytes,
            expires_at=expires_at,
        )

    def verify_upload_signature(
        self,
        build_id: str,
        max_bytes: int,
        expires_at: float,
        signature: str,
        attempt: str = "",
    ) -> bool:
        """True when an upload URL's signature (with any ``attempt``) is valid and unexpired."""
        if time.time() > expires_at:
            return False
        data: dict[str, Any] = {
            "op": "upload",
            "build_id": build_id,
            "max_bytes": max_bytes,
            "expires_at": expires_at,
        }
        if attempt:
            data["attempt"] = attempt
        return self._verify(data, signature)

    def generate_log_url(
        self,
        base_url: str,
        build_id: str,
        expiry_seconds: float = 600.0,
    ) -> str:
        """Sign a URL an executor appends console output to while it runs.

        Its own ``op``, so a leaked log URL cannot be replayed to upload or finalize. No lease:
        console is advisory, and a chunk from a reclaimed worker is worth showing, not a 409.
        """
        expires_at = time.time() + expiry_seconds
        data = {"op": "log", "build_id": build_id, "expires_at": expires_at}
        params = {"expires_at": str(expires_at), "signature": self._sign(data)}
        return f"{base_url}/v1/builds/{build_id}/log?{urlencode(params)}"

    def generate_finalize_url(
        self,
        base_url: str,
        build_id: str,
        expiry_seconds: float = 600.0,
        lease_owner: str | None = None,
        lease_expires_at: float | None = None,
    ) -> SignedFinalizeURL:
        """Sign a URL for finalizing a build (default validity 600 s).

        ``lease_owner`` and ``lease_expires_at`` bind it to one claim, so the server can tell a
        current holder from a reclaimed one; signed, so a stale holder cannot forge the current
        claim. The in-process notebook path has no lease and omits them.
        """
        expires_at = time.time() + expiry_seconds
        lease = lease_token(lease_owner, lease_expires_at)
        data = {
            "op": "finalize",
            "build_id": build_id,
            "expires_at": expires_at,
            "lease": lease,
        }
        params = {
            "expires_at": str(expires_at),
            "signature": self._sign(data),
        }
        if lease:
            params["lease"] = lease
        url = f"{base_url}/v1/builds/{build_id}/finalize?{urlencode(params)}"
        return SignedFinalizeURL(
            url=url,
            build_id=build_id,
            expires_at=expires_at,
        )

    def verify_log_signature(
        self,
        build_id: str,
        expires_at: float,
        signature: str,
    ) -> bool:
        """Verify a log URL's signature and expiry."""
        if time.time() > expires_at:
            return False
        data = {"op": "log", "build_id": build_id, "expires_at": expires_at}
        return self._verify(data, signature)

    def verify_finalize_signature(
        self,
        build_id: str,
        expires_at: float,
        signature: str,
        lease: str = "",
    ) -> bool:
        """True when a finalize URL's signature (including ``lease``) is valid and unexpired."""
        if time.time() > expires_at:
            return False
        data = {
            "op": "finalize",
            "build_id": build_id,
            "expires_at": expires_at,
            "lease": lease,
        }
        return self._verify(data, signature)

    def generate_build_manifest(
        self,
        base_url: str,
        build_id: str,
        metadata: dict[str, Any],
        input_artifacts: list[tuple[str, int]],
        max_output_bytes: int,
        url_expiry_seconds: float = 600.0,
        lease_owner: str | None = None,
        lease_expires_at: float | None = None,
        blob_store: Any = None,
        blob_key: Callable[[str, int], tuple[str, int]] | None = None,
    ) -> BuildManifest:
        """Assemble the signed-URL manifest for a build: inputs, output, finalize and log.

        ``input_artifacts`` are ``(artifact_id, version)`` pairs. With a ``blob_store`` that can
        presign, inputs and output go straight to the object store and only finalize and log stay
        Strata routes; the output key is ``metadata["artifact_id"]`` / ``["version"]``.
        ``blob_key`` maps an input to the key its bytes are presigned under, which may be an
        attempt's or another version's.
        """
        expires_at = time.time() + url_expiry_seconds
        # The executor writes under this claim's own attempt, so an earlier
        # manifest's upload URL lands where finalize never looks.
        attempt = lease_attempt(lease_token(lease_owner, lease_expires_at))
        input_urls = []
        for artifact_id, version in input_artifacts:
            stored_as = (
                blob_key(artifact_id, version) if blob_key is not None else (artifact_id, version)
            )
            presigned = (
                blob_store.presign_get(*stored_as, int(url_expiry_seconds))
                if blob_store is not None
                else None
            )
            input_urls.append(
                SignedDownloadURL(
                    url=presigned,
                    artifact_id=artifact_id,
                    version=version,
                    expires_at=expires_at,
                )
                if presigned is not None
                else self.generate_download_url(
                    base_url=base_url,
                    artifact_id=artifact_id,
                    version=version,
                    build_id=build_id,
                    expiry_seconds=url_expiry_seconds,
                )
            )
        output_artifact = metadata.get("artifact_id"), metadata.get("version")
        presigned_upload = presigned_put = None
        if blob_store is not None and output_artifact[0] and output_artifact[1] is not None:
            output_blob = (
                attempt_blob_id(str(output_artifact[0]), attempt)
                if attempt
                else str(output_artifact[0])
            )
            presigned_upload = blob_store.presign_post(
                output_blob, int(output_artifact[1]), max_output_bytes, int(url_expiry_seconds)
            )
            if presigned_upload is None:
                presigned_put = blob_store.presign_put(
                    output_blob, int(output_artifact[1]), int(url_expiry_seconds)
                )
        if presigned_upload is not None:
            output_url = SignedUploadURL(
                url=presigned_upload[0],
                build_id=build_id,
                max_bytes=max_output_bytes,
                expires_at=expires_at,
                fields=presigned_upload[1],
            )
        elif presigned_put is not None:
            output_url = SignedUploadURL(
                url=presigned_put[0],
                build_id=build_id,
                max_bytes=max_output_bytes,
                expires_at=expires_at,
                method="PUT",
                headers=presigned_put[1],
            )
        else:
            output_url = self.generate_upload_url(
                base_url=base_url,
                build_id=build_id,
                max_bytes=max_output_bytes,
                expiry_seconds=url_expiry_seconds,
                attempt=attempt,
            )
        finalize_url = self.generate_finalize_url(
            base_url=base_url,
            build_id=build_id,
            expiry_seconds=url_expiry_seconds,
            lease_owner=lease_owner,
            lease_expires_at=lease_expires_at,
        ).url
        return BuildManifest(
            build_id=build_id,
            metadata=metadata,
            input_urls=input_urls,
            output_url=output_url,
            finalize_url=finalize_url,
            log_url=self.generate_log_url(
                base_url=base_url,
                build_id=build_id,
                expiry_seconds=url_expiry_seconds,
            ),
        )
