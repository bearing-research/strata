"""Lightweight ``strata`` client for the notebook venv.

This module is **path-loaded** by the harness / warm pool (like
``serializer.py``) and runs in the notebook's venv, which has only
``pyarrow`` + stdlib — NOT ``strata`` or ``httpx``. So it cannot
``import strata``; it re-implements the slice of ``StrataClient`` a cell
needs directly over ``urllib`` + ``pyarrow``, faithful to the same REST
wire protocol as ``strata.client.StrataClient``.

Kept deliberately small: the data ops a cell reaches for (``materialize``,
``put``) and the registry ops (``set_alias`` / ``set_tag`` / ``resolve_*``).
Surface drift from the real client is the maintenance cost of avoiding a
notebook-venv dependency; keep the two in sync when endpoints change.
"""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import uuid
from typing import Any

import pyarrow as pa
from pyarrow import ipc

_DEFAULT_TIMEOUT = 300.0


# ---------------------------------------------------------------------------
# Arrow IPC (mirrors strata.client._convert_to_arrow_ipc)
# ---------------------------------------------------------------------------


def _table_to_ipc(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _dict_to_ipc(data: dict) -> bytes:
    list_values = [v for v in data.values() if isinstance(v, list)]
    if data and len(list_values) == len(data):
        lengths = {len(v) for v in list_values}
        if len(lengths) == 1:
            try:
                return _table_to_ipc(pa.Table.from_pydict(dict(data)))
            except Exception:
                pass
    table = pa.Table.from_pydict({"data": [json.dumps(dict(data))]})
    return _table_to_ipc(table)


def _convert_to_arrow_ipc(data: Any) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, pa.Table):
        return _table_to_ipc(data)
    if isinstance(data, dict):
        return _dict_to_ipc(data)
    try:
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            return _table_to_ipc(pa.Table.from_pandas(data))
    except ImportError:
        pass
    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return _table_to_ipc(data.to_arrow())
    except ImportError:
        pass
    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. "
        "Expected dict, pa.Table, pd.DataFrame, pl.DataFrame, or bytes."
    )


def _parse_artifact_uri(uri: str) -> tuple[str, int]:
    # strata://artifact/{id}@v={version}
    tail = uri.rsplit("/", 1)[-1]
    art_id, _, ver = tail.partition("@v=")
    return art_id, int(ver)


# ---------------------------------------------------------------------------
# Artifact
# ---------------------------------------------------------------------------


class Artifact:
    """Minimal artifact handle — mirrors the fields/methods cells use."""

    def __init__(
        self,
        client: StrataClient,
        artifact_id: str,
        version: int,
        cache_hit: bool = False,
        stream_data: bytes | None = None,
    ) -> None:
        self._client = client
        self.artifact_id = artifact_id
        self.version = version
        self.cache_hit = cache_hit
        self._stream_data = stream_data

    @property
    def uri(self) -> str:
        return f"strata://artifact/{self.artifact_id}@v={self.version}"

    def to_arrow(self) -> pa.Table:
        data = self._stream_data
        if data is None:
            data = self._client._fetch_artifact_bytes(self.artifact_id, self.version)
        if not data:
            return pa.table({})
        return ipc.open_stream(pa.BufferReader(data)).read_all()

    def to_pandas(self):
        return self.to_arrow().to_pandas()


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class StrataClient:
    """Notebook-venv client over urllib (no httpx / no strata import)."""

    def __init__(self, base_url: str, timeout: float = _DEFAULT_TIMEOUT) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- HTTP primitives ---------------------------------------------------

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self._url(path), data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"strata {method} {path} -> {e.code}: {detail}") from None
        return json.loads(raw) if raw else {}

    def _get_bytes(self, path: str) -> bytes:
        req = urllib.request.Request(self._url(path), method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"strata GET {path} -> {e.code}") from None

    def _put_multipart(self, path: str, parts: list[tuple[str, str, str, bytes]]) -> dict:
        """PUT multipart/form-data. ``parts`` = (field, filename, content_type, body)."""
        boundary = f"----strata{uuid.uuid4().hex}"
        buf = io.BytesIO()
        for field, filename, ctype, payload in parts:
            disposition = f'Content-Disposition: form-data; name="{field}"; filename="{filename}"'
            buf.write(f"--{boundary}\r\n".encode())
            buf.write(f"{disposition}\r\n".encode())
            buf.write(f"Content-Type: {ctype}\r\n\r\n".encode())
            buf.write(payload)
            buf.write(b"\r\n")
        buf.write(f"--{boundary}--\r\n".encode())
        req = urllib.request.Request(
            self._url(path),
            data=buf.getvalue(),
            method="PUT",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            raise RuntimeError(f"strata PUT {path} -> {e.code}: {detail}") from None
        return json.loads(raw) if raw else {}

    def _fetch_artifact_bytes(self, artifact_id: str, version: int) -> bytes:
        return self._get_bytes(f"/v1/artifacts/{artifact_id}/v/{version}/data")

    # -- Data ops ----------------------------------------------------------

    def materialize(
        self,
        inputs: list[str],
        transform: dict,
        name: str | None = None,
        mode: str = "stream",
        refresh: bool = False,
    ) -> Artifact:
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")
        body: dict[str, Any] = {"inputs": inputs, "transform": server_transform, "mode": mode}
        if name:
            body["name"] = name
        if refresh:
            body["refresh"] = True

        data = self._request("POST", "/v1/materialize", body)
        artifact_id, version = _parse_artifact_uri(data["artifact_uri"])
        hit = bool(data.get("hit", False))
        stream_url = data.get("stream_url")
        stream_data = None
        if stream_url and mode == "stream":
            stream_data = self._get_bytes(stream_url)
        return Artifact(self, artifact_id, version, cache_hit=hit, stream_data=stream_data)

    def put(
        self,
        inputs: list[str],
        transform: dict,
        data: Any,
        name: str | None = None,
    ) -> Artifact:
        arrow_bytes = _convert_to_arrow_ipc(data)
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")
        metadata: dict[str, Any] = {"inputs": inputs, "transform": server_transform}
        if name:
            metadata["name"] = name
        result = self._put_multipart(
            "/v1/artifacts",
            [
                ("metadata", "metadata.json", "application/json", json.dumps(metadata).encode()),
                ("data", "data.arrow", "application/vnd.apache.arrow.stream", arrow_bytes),
            ],
        )
        artifact_id, version = _parse_artifact_uri(result["artifact_uri"])
        return Artifact(self, artifact_id, version, cache_hit=bool(result.get("hit", False)))

    # -- Registry ops ------------------------------------------------------

    def set_alias(self, name: str, alias: str, artifact_id: str, version: int) -> dict:
        return self._request(
            "PUT",
            f"/v1/names/{name}/aliases/{alias}",
            {"artifact_id": artifact_id, "version": version},
        )

    def resolve_alias(self, name: str, alias: str) -> dict:
        return self._request("GET", f"/v1/names/{name}/aliases/{alias}")

    def set_tag(self, artifact_id: str, version: int, key: str, value: str) -> dict:
        return self._request(
            "PUT",
            f"/v1/artifacts/{artifact_id}/v/{version}/tags",
            {"key": key, "value": str(value)},
        )

    def get_tags(self, artifact_id: str, version: int) -> dict:
        return self._request("GET", f"/v1/artifacts/{artifact_id}/v/{version}/tags").get("tags", {})

    def resolve_name(self, name: str) -> dict:
        return self._request("GET", f"/v1/names/{name}")

    def set_name(self, name: str, artifact_id: str, version: int) -> dict:
        return self._request(
            "POST", "/v1/names", {"name": name, "artifact_id": artifact_id, "version": version}
        )

    def get_registry_audit(self, name: str | None = None, limit: int = 100) -> list[dict]:
        path = f"/v1/registry/audit?limit={limit}"
        if name:
            path += f"&name={name}"
        return self._request("GET", path).get("entries", [])

    def close(self) -> None:
        # urllib opens a connection per request; nothing persistent to close.
        # Present so cells written against StrataClient (which has .close())
        # keep working unchanged.
        return None
