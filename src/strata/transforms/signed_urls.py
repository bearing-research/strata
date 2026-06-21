"""Signed-URL signing for the pull-model executor protocol.

In the pull model an executor fetches its inputs and pushes its output directly
to Strata's storage through short-lived, HMAC-signed capability URLs. This keeps
the data plane off Strata (no bandwidth bottleneck), lets executors retry
transfers natively, and lowers Strata's memory pressure.

Each URL embeds its operation, the resource identifiers, an expiry, and — for
uploads — a size limit, signed with HMAC-SHA256. The signature prevents
tampering and the expiry prevents replay. The signing secret is held by a
:class:`URLSigner` instance rather than process-global state, so it is explicit,
injectable, and testable.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode


@dataclass(frozen=True)
class SignedDownloadURL:
    """Signed URL for downloading an input artifact.

    Attributes
    ----------
    url : str
        Full URL including the signature query parameters.
    artifact_id : str
        Artifact being downloaded.
    version : int
        Artifact version being downloaded.
    expires_at : float
        Unix timestamp after which the URL is rejected.
    """

    url: str
    artifact_id: str
    version: int
    expires_at: float


@dataclass(frozen=True)
class SignedUploadURL:
    """Signed URL for uploading build output.

    Attributes
    ----------
    url : str
        Full URL including the signature query parameters.
    build_id : str
        Build the upload belongs to.
    max_bytes : int
        Maximum permitted upload size, in bytes.
    expires_at : float
        Unix timestamp after which the URL is rejected.
    """

    url: str
    build_id: str
    max_bytes: int
    expires_at: float


@dataclass(frozen=True)
class SignedFinalizeURL:
    """Signed URL for finalizing a build output.

    Attributes
    ----------
    url : str
        Full URL including the signature query parameters.
    build_id : str
        Build being finalized.
    expires_at : float
        Unix timestamp after which the URL is rejected.
    """

    url: str
    build_id: str
    expires_at: float


@dataclass(frozen=True)
class BuildManifest:
    """Bundle of signed URLs handed to an executor for one build.

    The executor uses it to pull each input, push the output, and finalize.

    Attributes
    ----------
    build_id : str
        Build the manifest is for.
    metadata : dict
        Build metadata (transform spec, params, and so on).
    input_urls : list of SignedDownloadURL
        One signed download URL per input artifact.
    output_url : SignedUploadURL
        Signed URL the executor uploads its output to.
    finalize_url : str
        URL the executor calls once the upload is complete.
    """

    build_id: str
    metadata: dict[str, Any]
    input_urls: list[SignedDownloadURL]
    output_url: SignedUploadURL
    finalize_url: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON wire shape sent to the executor.

        The wire shape regroups the URLs under ``inputs`` / ``output``. Each
        input is the full ``SignedDownloadURL``; the output drops the redundant
        ``build_id`` (already carried at the top level) — the one place the wire
        intentionally diverges from a plain field dump.

        Returns
        -------
        dict
            ``{build_id, metadata, inputs, output, finalize_url}``.
        """
        output = asdict(self.output_url)
        del output["build_id"]
        return {
            "build_id": self.build_id,
            "metadata": self.metadata,
            "inputs": [asdict(url) for url in self.input_urls],
            "output": output,
            "finalize_url": self.finalize_url,
        }


class URLSigner:
    """Signs and verifies pull-model capability URLs with one HMAC secret.

    A single instance is constructed at server startup from the configured
    signing secret and shared by the request handlers and the build service.
    Holding the secret on an instance — rather than module-global state — keeps
    it explicit and injectable, and lets tests run with an isolated secret.

    Parameters
    ----------
    secret : bytes
        HMAC-SHA256 signing secret. Use a stable, high-entropy value in
        production so signed URLs survive restarts and match across replicas.

    Notes
    -----
    A signed payload records ``version`` as an ``int`` and ``expires_at`` as a
    ``float``, while the URL carries every parameter as a string. A verifier
    must therefore coerce the query parameters back to those exact types before
    calling the matching ``verify_*`` method, or verification fails.
    """

    def __init__(self, secret: bytes) -> None:
        self._secret = secret

    def _sign(self, data: dict[str, Any]) -> str:
        """Return the base64 HMAC-SHA256 signature of ``data``.

        Parameters
        ----------
        data : dict
            Payload to sign; serialized as canonical (key-sorted) JSON.

        Returns
        -------
        str
            URL-safe base64-encoded signature.
        """
        message = json.dumps(data, sort_keys=True).encode()
        signature = hmac.new(self._secret, message, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(signature).decode()

    def _verify(self, data: dict[str, Any], signature: str) -> bool:
        """Check a signature against ``data`` in constant time.

        Parameters
        ----------
        data : dict
            Payload that was signed.
        signature : str
            Base64-encoded signature to check.

        Returns
        -------
        bool
            ``True`` if the signature matches.
        """
        expected = self._sign(data)
        return hmac.compare_digest(expected, signature)

    def generate_download_url(
        self,
        base_url: str,
        artifact_id: str,
        version: int,
        build_id: str,
        expiry_seconds: float = 300.0,
    ) -> SignedDownloadURL:
        """Sign a URL for downloading an artifact.

        Parameters
        ----------
        base_url : str
            Base URL of the Strata server, e.g. ``"http://localhost:8765"``.
        artifact_id : str
            Artifact to download.
        version : int
            Artifact version to download.
        build_id : str
            Build the download is for (recorded for audit; not an access check).
        expiry_seconds : float, optional
            URL validity window in seconds (default 300, i.e. 5 minutes).

        Returns
        -------
        SignedDownloadURL
            The signed URL and its metadata.
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
        """Verify a download URL's signature and expiry.

        Parameters
        ----------
        artifact_id : str
            Artifact ID from the URL.
        version : int
            Artifact version from the URL.
        build_id : str
            Build ID from the URL.
        expires_at : float
            Expiry timestamp from the URL.
        signature : str
            Signature from the URL.

        Returns
        -------
        bool
            ``True`` if the signature is valid and the URL has not expired.
        """
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
    ) -> SignedUploadURL:
        """Sign a URL for uploading build output.

        Parameters
        ----------
        base_url : str
            Base URL of the Strata server.
        build_id : str
            Build the upload is for.
        max_bytes : int
            Maximum permitted upload size, in bytes (signed, so it cannot be
            raised by tampering with the URL).
        expiry_seconds : float, optional
            URL validity window in seconds (default 600, i.e. 10 minutes).

        Returns
        -------
        SignedUploadURL
            The signed URL and its metadata.
        """
        expires_at = time.time() + expiry_seconds
        data = {
            "op": "upload",
            "build_id": build_id,
            "max_bytes": max_bytes,
            "expires_at": expires_at,
        }
        params = {
            "build_id": build_id,
            "max_bytes": str(max_bytes),
            "expires_at": str(expires_at),
            "signature": self._sign(data),
        }
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
    ) -> bool:
        """Verify an upload URL's signature and expiry.

        Parameters
        ----------
        build_id : str
            Build ID from the URL.
        max_bytes : int
            Maximum upload size from the URL.
        expires_at : float
            Expiry timestamp from the URL.
        signature : str
            Signature from the URL.

        Returns
        -------
        bool
            ``True`` if the signature is valid and the URL has not expired.
        """
        if time.time() > expires_at:
            return False
        data = {
            "op": "upload",
            "build_id": build_id,
            "max_bytes": max_bytes,
            "expires_at": expires_at,
        }
        return self._verify(data, signature)

    def generate_finalize_url(
        self,
        base_url: str,
        build_id: str,
        expiry_seconds: float = 600.0,
    ) -> SignedFinalizeURL:
        """Sign a URL for finalizing a build.

        Parameters
        ----------
        base_url : str
            Base URL of the Strata server.
        build_id : str
            Build to finalize.
        expiry_seconds : float, optional
            URL validity window in seconds (default 600, i.e. 10 minutes).

        Returns
        -------
        SignedFinalizeURL
            The signed URL and its metadata.
        """
        expires_at = time.time() + expiry_seconds
        data = {
            "op": "finalize",
            "build_id": build_id,
            "expires_at": expires_at,
        }
        params = {
            "expires_at": str(expires_at),
            "signature": self._sign(data),
        }
        url = f"{base_url}/v1/builds/{build_id}/finalize?{urlencode(params)}"
        return SignedFinalizeURL(
            url=url,
            build_id=build_id,
            expires_at=expires_at,
        )

    def verify_finalize_signature(
        self,
        build_id: str,
        expires_at: float,
        signature: str,
    ) -> bool:
        """Verify a finalize URL's signature and expiry.

        Parameters
        ----------
        build_id : str
            Build ID from the URL.
        expires_at : float
            Expiry timestamp from the URL.
        signature : str
            Signature from the URL.

        Returns
        -------
        bool
            ``True`` if the signature is valid and the URL has not expired.
        """
        if time.time() > expires_at:
            return False
        data = {
            "op": "finalize",
            "build_id": build_id,
            "expires_at": expires_at,
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
    ) -> BuildManifest:
        """Assemble the full signed-URL manifest for a build.

        Bundles the download URL for each input, the output upload URL, and the
        finalize URL the executor needs to run the build end to end.

        Parameters
        ----------
        base_url : str
            Base URL of the Strata server.
        build_id : str
            Build the manifest is for.
        metadata : dict
            Build metadata (transform spec, params, and so on).
        input_artifacts : list of tuple of (str, int)
            ``(artifact_id, version)`` for each input.
        max_output_bytes : int
            Maximum permitted output size, in bytes.
        url_expiry_seconds : float, optional
            Validity window applied to every URL in the manifest (default 600).

        Returns
        -------
        BuildManifest
            The assembled manifest.
        """
        input_urls = [
            self.generate_download_url(
                base_url=base_url,
                artifact_id=artifact_id,
                version=version,
                build_id=build_id,
                expiry_seconds=url_expiry_seconds,
            )
            for artifact_id, version in input_artifacts
        ]
        output_url = self.generate_upload_url(
            base_url=base_url,
            build_id=build_id,
            max_bytes=max_output_bytes,
            expiry_seconds=url_expiry_seconds,
        )
        finalize_url = self.generate_finalize_url(
            base_url=base_url,
            build_id=build_id,
            expiry_seconds=url_expiry_seconds,
        ).url
        return BuildManifest(
            build_id=build_id,
            metadata=metadata,
            input_urls=input_urls,
            output_url=output_url,
            finalize_url=finalize_url,
        )
