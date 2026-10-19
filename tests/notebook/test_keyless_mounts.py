"""Keyless ``gs`` and ``az`` mounts: a federated token file, exchanged at each refresh.

Each test runs the real credential library against a local token service, so the token
that reaches it is the one in the file at that moment.
"""

from __future__ import annotations

import datetime
import http.server
import ipaddress
import json
import ssl
import threading
import urllib.parse
from collections.abc import Iterator
from pathlib import Path

import pytest

from strata.notebook.mounts import _mount_filesystem


class _TokenService:
    """Answers every token request, recording the federated token it was sent."""

    def __init__(self, tls: tuple[Path, Path] | None = None):
        self.seen: list[str] = []
        service = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                form = urllib.parse.parse_qs(
                    self.rfile.read(int(self.headers["Content-Length"])).decode()
                )
                # Google's STS takes subject_token; Entra ID takes client_assertion.
                (federated,) = form.get("subject_token") or form["client_assertion"]
                service.seen.append(federated)
                body = json.dumps(
                    {
                        "access_token": f"access-{len(service.seen)}",
                        "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                        "token_type": "Bearer",
                        # Expired at once, so each ask for a token exchanges anew.
                        "expires_in": 0,
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                return None

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        if tls is not None:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(*tls)
            self.server.socket = context.wrap_socket(self.server.socket, server_side=True)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.port = self.server.server_address[1]


def _self_signed(tmp_path: Path) -> tuple[Path, Path]:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")])
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(1)
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.DNSName("localhost"), x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    cert_path, key_path = tmp_path / "cert.pem", tmp_path / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


@pytest.fixture
def token_service() -> Iterator[_TokenService]:
    service = _TokenService()
    yield service
    service.server.shutdown()


@pytest.fixture
def tls_token_service(tmp_path, monkeypatch) -> Iterator[_TokenService]:
    """Entra ID's client refuses a non-TLS authority, so this one has a trusted certificate."""
    import aiohttp.connector

    cert, key = _self_signed(tmp_path)
    service = _TokenService(tls=(cert, key))
    # aiohttp builds its verifying context once, at import, so SSL_CERT_FILE comes too late.
    monkeypatch.setattr(
        aiohttp.connector, "_SSL_CONTEXT_VERIFIED", ssl.create_default_context(cafile=str(cert))
    )
    monkeypatch.setenv("AZURE_AUTHORITY_HOST", f"https://localhost:{service.port}")
    yield service
    service.server.shutdown()


def external_account_file(tmp_path: Path, token_file: Path, port: int) -> Path:
    """Google's workload-identity configuration, pointing at *token_file*."""
    config = tmp_path / "external-account.json"
    config.write_text(
        json.dumps(
            {
                "type": "external_account",
                "audience": "//iam.googleapis.com/projects/1/locations/global/"
                "workloadIdentityPools/amber/providers/org",
                "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                "token_url": f"http://127.0.0.1:{port}/v1/token",
                "credential_source": {"file": str(token_file)},
            }
        )
    )
    return config


def test_a_gs_external_account_token_is_exchanged_from_the_file_at_each_refresh(
    tmp_path, token_service
):
    import google.auth.transport.requests

    token_file = tmp_path / "token"
    token_file.write_text("org-token-1")
    config = external_account_file(tmp_path, token_file, token_service.port)

    fs = _mount_filesystem(
        "gcs", {"token": str(config), "endpoint_url": "http://127.0.0.1:9", "project": "p"}
    )
    credentials = fs.credentials.credentials
    request = google.auth.transport.requests.Request()
    credentials.refresh(request)
    first = set(token_service.seen)
    token_service.seen.clear()
    token_file.write_text("org-token-2")
    credentials.refresh(request)

    assert (first, set(token_service.seen)) == ({"org-token-1"}, {"org-token-2"})


def test_a_gs_token_that_is_not_an_external_account_reaches_gcsfs_as_given(tmp_path):
    fs = _mount_filesystem("gcs", {"token": "anon", "endpoint_url": "http://127.0.0.1:9"})

    assert fs.credentials.method == "anon"


async def _azure_token(fs) -> str:
    token = await fs.credential.get_token("https://storage.azure.com/.default")
    return token.token


def test_an_az_federated_token_file_is_exchanged_from_the_file_at_each_refresh(
    tmp_path, tls_token_service
):
    from fsspec.asyn import sync

    token_file = tmp_path / "token"
    token_file.write_text("org-token-1")

    fs = _mount_filesystem(
        "abfs",
        {
            "account_name": "lab",
            "tenant_id": "amber-tenant",
            "client_id": "source-app",
            "federated_token_file": str(token_file),
        },
    )
    first = sync(fs.loop, _azure_token, fs)
    token_file.write_text("org-token-2")
    second = sync(fs.loop, _azure_token, fs)

    assert (first, second) == ("access-1", "access-2")
    assert tls_token_service.seen == ["org-token-1", "org-token-2"]
    assert fs.client_id is None, "the fields made the credential, not adlfs's client secret"


def test_an_az_federated_token_file_without_its_app_is_refused(tmp_path):
    with pytest.raises(ValueError, match="needs tenant_id and client_id"):
        _mount_filesystem(
            "abfs",
            {"account_name": "lab", "federated_token_file": str(tmp_path / "token")},
        )
