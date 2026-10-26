"""Presigning with each cloud's own signing, against fixed fake credentials and no network.

A GCS V4 signature is verified against the key's public half, and the IAM signBlob path is
faked at the HTTP boundary; the S3 and Azure URLs are checked for the credential, token and
permissions they carry. No test here or in the suite submits a URL to a store or emulator.
"""

from __future__ import annotations

import base64
import hashlib
import json
from urllib.parse import parse_qs, unquote, urlsplit

import pytest
from conftest import role_credentials
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from vestigo.blob_store import AzureBlobStore, GCSBlobStore, S3BlobStore

SERVICE_ACCOUNT = "signer@proj.iam.gserviceaccount.com"


@pytest.fixture(scope="module")
def rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _verify(rsa_key, signature_hex: str, message: bytes) -> None:
    """Raise unless ``signature_hex`` is ``rsa_key``'s RSA-SHA256 signature of ``message``."""
    rsa_key.public_key().verify(
        bytes.fromhex(signature_hex), message, padding.PKCS1v15(), hashes.SHA256()
    )


def _verify_v4_url(rsa_key, url: str) -> None:
    """Rebuild a GCS V4 query URL's string-to-sign and check its signature."""
    parts = urlsplit(url)
    pairs = [pair for pair in parts.query.split("&") if not pair.startswith("X-Goog-Signature=")]
    params = parse_qs(parts.query)
    canonical = "\n".join(
        ["GET", parts.path, "&".join(sorted(pairs)), f"host:{parts.netloc}", "", "host"]
    )
    canonical += "\nUNSIGNED-PAYLOAD"
    scope = params["X-Goog-Credential"][0].split("/", 1)[1]
    string_to_sign = "\n".join(
        [
            "GOOG4-RSA-SHA256",
            params["X-Goog-Date"][0],
            scope,
            hashlib.sha256(canonical.encode()).hexdigest(),
        ]
    )
    _verify(rsa_key, params["X-Goog-Signature"][0], string_to_sign.encode())


# GCS


def _service_account_key(rsa_key) -> dict[str, str]:
    """A service-account key file's contents, holding ``rsa_key``."""
    pem = rsa_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()
    return {
        "type": "service_account",
        "project_id": "proj",
        "private_key_id": "k1",
        "private_key": pem,
        "client_email": SERVICE_ACCOUNT,
        "client_id": "1",
        "token_uri": "https://oauth2.googleapis.com/token",
    }


@pytest.fixture
def gcs_key_store(rsa_key):
    """A GCS store holding a service-account key, as a consumer's credentials setting gives it."""
    return GCSBlobStore(
        bucket="bkt", prefix="artifacts", credentials_json=json.dumps(_service_account_key(rsa_key))
    )


def _keyless_credentials(**attributes):
    """Credentials with no private key, like the GCE metadata server's or workload identity's."""
    from google.auth.credentials import Credentials

    class Keyless(Credentials):
        refreshes = 0

        def refresh(self, request):
            type(self).refreshes += 1
            self.token = "ya29.fake"

    for name, value in attributes.items():
        setattr(Keyless, name, value)
    return Keyless()


def test_keyless_gcs_credentials_sign_through_iam_as_the_attached_account(rsa_key, monkeypatch):
    """The IAM signBlob API signs for workload identity; it is faked at the HTTP boundary."""
    import google.auth
    from google.auth.transport import requests as google_requests

    credentials = _keyless_credentials(service_account_email=SERVICE_ACCOUNT)
    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (credentials, "proj"))
    calls: list[tuple[str, str]] = []

    class _Response:
        status = 200
        headers: dict[str, str] = {}

        def __init__(self, data: bytes):
            self.data = data

    def sign_blob(self, url, method="GET", body=None, headers=None, **kwargs):
        calls.append((url, headers["Authorization"]))
        payload = base64.b64decode(json.loads(body)["payload"])
        signature = rsa_key.sign(payload, padding.PKCS1v15(), hashes.SHA256())
        return _Response(json.dumps({"signedBlob": base64.b64encode(signature).decode()}).encode())

    monkeypatch.setattr(google_requests.Request, "__call__", sign_blob)
    store = GCSBlobStore(bucket="bkt", prefix="artifacts")

    url = store.presign_get("fig", 1, ttl_seconds=60)
    signed = store.presign_post("out", 1, max_bytes=16, ttl_seconds=60)

    assert url is not None and signed is not None
    _verify_v4_url(rsa_key, url)
    _verify(rsa_key, signed[1]["x-goog-signature"], signed[1]["policy"].encode())
    assert [call[0].split("?")[0] for call in calls] == [
        f"https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/{SERVICE_ACCOUNT}:signBlob"
    ] * 2
    assert {call[1] for call in calls} == {"Bearer ya29.fake"}
    assert type(credentials).refreshes == 1


def test_gcs_credentials_with_no_service_account_do_not_presign(monkeypatch):
    """A person's gcloud login has neither a key nor an account IAM can sign as."""
    import google.auth

    user = _keyless_credentials()
    monkeypatch.setattr(google.auth, "default", lambda scopes=None: (user, "proj"))
    store = GCSBlobStore(bucket="bkt")

    assert store.presign_get("fig", 1, 60) is None
    assert store.presign_post("fig", 1, 16, 60) is None


def test_an_anonymous_gcs_store_does_not_presign():
    store = GCSBlobStore(bucket="bkt", anonymous=True)

    assert store.presign_get("fig", 1, 60) is None
    assert store.presign_post("fig", 1, 16, 60) is None


def test_a_key_file_path_signs_like_the_key_itself(rsa_key, tmp_path):
    """The credentials setting is a path or the key's JSON; both reach the same signer."""
    key_file = tmp_path / "key.json"
    key_file.write_text(json.dumps(_service_account_key(rsa_key)))
    store = GCSBlobStore(bucket="bkt", prefix="artifacts", credentials_json=str(key_file))

    url = store.presign_get("fig", 1, ttl_seconds=60)

    assert url is not None
    _verify_v4_url(rsa_key, url)


# S3 role


def test_s3_role_credentials_sign_with_their_session_token(monkeypatch):
    with role_credentials(monkeypatch, "ASIAROLE", "role-secret", "role-token") as seen:
        store = S3BlobStore(bucket="bkt", prefix="artifacts", region="us-east-1")
        url = store.presign_get("fig", 1, ttl_seconds=60)
        signed = store.presign_post("out", 1, max_bytes=16, ttl_seconds=60)

    assert url is not None and signed is not None
    params = parse_qs(urlsplit(url).query)
    assert params["X-Amz-Credential"][0].startswith("ASIAROLE/")
    assert params["X-Amz-Security-Token"] == ["role-token"]
    assert signed[1]["x-amz-credential"].startswith("ASIAROLE/")
    assert signed[1]["x-amz-security-token"] == "role-token"
    assert len(seen) == 1, "the role is resolved once, by the client, not per URL"


# Azure

ACCOUNT_KEY = base64.b64encode(b"k" * 32).decode()


def test_an_azure_account_key_signs_a_read_sas_and_a_create_write_sas():
    store = AzureBlobStore(account_name="acct", container_name="c", account_key=ACCOUNT_KEY)

    url = store.presign_get("fig", 1, ttl_seconds=60)
    put = store.presign_put("out", 7, ttl_seconds=60)

    assert url is not None and put is not None
    parts = urlsplit(url)
    assert parts.netloc == "acct.blob.core.windows.net"
    assert unquote(parts.path) == "/c/artifacts/fig@v=1.arrow"
    params = parse_qs(parts.query)
    assert params["sp"] == ["r"] and params["sr"] == ["b"] and "sig" in params
    put_url, headers = put
    assert unquote(urlsplit(put_url).path) == "/c/artifacts/out@v=7.arrow"
    assert parse_qs(urlsplit(put_url).query)["sp"] == ["cw"]
    assert headers == {"x-ms-blob-type": "BlockBlob"}
    assert store.presign_post("out", 7, 16, 60) is None


def test_an_azure_endpoint_url_signs_urls_under_the_account_segment():
    """A host-form endpoint (Azurite) signs for the account at its first path segment."""
    store = AzureBlobStore(
        account_name="devstoreaccount1",
        container_name="c",
        account_key=ACCOUNT_KEY,
        endpoint_url="http://127.0.0.1:10000",
    )

    url = store.presign_get("fig", 1, ttl_seconds=60)
    put = store.presign_put("out", 7, ttl_seconds=60)

    assert url is not None and put is not None
    parts = urlsplit(url)
    assert parts.scheme == "http" and parts.netloc == "127.0.0.1:10000"
    assert unquote(parts.path) == "/devstoreaccount1/c/artifacts/fig@v=1.arrow"
    assert "sig" in parse_qs(parts.query)
    assert unquote(urlsplit(put[0]).path) == "/devstoreaccount1/c/artifacts/out@v=7.arrow"


def test_an_azure_sas_token_is_not_handed_to_workers():
    store = AzureBlobStore(account_name="acct", container_name="c", sas_token="sv=2024&sig=abc")

    assert store.presign_get("fig", 1, 60) is None
    assert store.presign_put("fig", 1, 60) is None


def test_an_azure_managed_identity_signs_with_a_reused_user_delegation_key(monkeypatch):
    from azure.storage.blob import BlobServiceClient, UserDelegationKey

    requested: list[tuple[float, float]] = []

    def get_user_delegation_key(self, key_start_time, key_expiry_time, **kwargs):
        requested.append((key_start_time.timestamp(), key_expiry_time.timestamp()))
        key = UserDelegationKey()
        key.signed_oid = "object-id"
        key.signed_tid = "tenant-id"
        key.signed_start = key_start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        key.signed_expiry = key_expiry_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        key.signed_service = "b"
        key.signed_version = "2024-08-04"
        key.value = ACCOUNT_KEY
        return key

    monkeypatch.setattr(BlobServiceClient, "get_user_delegation_key", get_user_delegation_key)
    store = AzureBlobStore(account_name="acct", container_name="c", use_default_credential=True)

    url = store.presign_get("fig", 1, ttl_seconds=60)
    put = store.presign_put("out", 1, ttl_seconds=60)

    assert url is not None and put is not None
    assert parse_qs(urlsplit(url).query)["skoid"] == ["object-id"]
    assert parse_qs(urlsplit(put[0]).query)["sp"] == ["cw"]
    assert len(requested) == 1, "one delegation key serves URLs it outlives"
    start, expiry = requested[0]
    assert expiry - start > 60

    # A URL that would outlive the cached key gets a fresh one.
    store.presign_get("fig", 1, ttl_seconds=4 * 3600)
    assert len(requested) == 2
