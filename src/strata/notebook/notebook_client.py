"""Lightweight ``strata`` client for the notebook venv.

Path-loaded by the harness and warm pool into a venv with only ``pyarrow`` and
the stdlib, so it re-implements the slice of ``strata.client.StrataClient`` a
cell needs over ``urllib``. Keep the two in sync when endpoints change.
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
    """Minimal artifact handle with the fields and methods cells use."""

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

    def __init__(
        self,
        base_url: str,
        timeout: float = _DEFAULT_TIMEOUT,
        cell_id: str | None = None,
        headers: dict[str, str] | None = None,
        promote_url: str | None = None,
        inputs: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._cell_id = cell_id
        # Sent on every request: auth for a remote shared store. Empty for the local server.
        self._headers = dict(headers or {})
        # Where ``promote`` posts. Not ``base_url``: with a team store, the client points at
        # that store, but the copy route runs on the notebook server, the only process that
        # can read the notebook's artifacts. None when no team store is configured.
        self._promote_url = (promote_url or "").rstrip("/") or None
        # Variable name -> artifact URI for this cell's inputs, so a cell can promote an
        # upstream result by the name it reads it under.
        self._inputs = dict(inputs or {})

    def _stamp_cell(self, artifact: Artifact, name: str | None) -> None:
        """Tag a named artifact with the originating cell; best effort, errors swallowed."""
        if not name or not self._cell_id:
            return
        try:
            self.set_tag(artifact.artifact_id, artifact.version, "nb_cell", self._cell_id)
        except Exception:
            pass

    # -- HTTP primitives ---------------------------------------------------

    def _url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        data = None
        headers = {**self._headers, "Accept": "application/json"}
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
        req = urllib.request.Request(self._url(path), method="GET", headers=dict(self._headers))
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                # Drain in chunks: one big blocking read on a large stream lets the server's send
                # buffer fill, its ``is_disconnected()`` check trips, and the artifact finalizes as
                # ``failed`` (client sees ``IncompleteRead``).
                chunks: list[bytes] = []
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
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
            headers={
                **self._headers,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
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
        # Only synchronous stream materialization here: mode="artifact" starts an async build
        # that needs build-status polling (in strata-client), so fail fast with guidance.
        if mode != "stream":
            raise ValueError(
                f"strata.materialize supports only mode='stream' here; got {mode!r}. "
                "For async artifact builds use the standalone client "
                "(`pip install strata-client`)."
            )
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
        art = Artifact(self, artifact_id, version, cache_hit=hit, stream_data=stream_data)
        self._stamp_cell(art, name)
        return art

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
        art = Artifact(self, artifact_id, version, cache_hit=bool(result.get("hit", False)))
        self._stamp_cell(art, name)
        return art

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

    def promote(
        self,
        ref: str,
        *,
        name: str,
        alias: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> dict:
        """Send an upstream result, and the chain behind it, to the team store.

        ``ref`` is one of this cell's input variable names, or an explicit
        ``<id>@v=<n>``. Only upstream results can be promoted from a cell: its own
        outputs are stored after it returns.
        """
        if self._promote_url is None:
            raise RuntimeError(
                "No team store is configured, so there is nowhere to promote to "
                "(set notebook_remote_store_url on the notebook server)."
            )
        uri = self._inputs.get(ref, ref)
        if "@v=" not in uri:
            known = ", ".join(sorted(self._inputs)) or "none"
            raise RuntimeError(
                f"{ref!r} is not one of this cell's inputs and is not an "
                f"'<id>@v=<n>' reference (inputs: {known})."
            )
        artifact_id, version = _parse_artifact_uri(uri)
        body: dict[str, Any] = {"name": name, "tags": {k: str(v) for k, v in (tags or {}).items()}}
        if alias:
            body["alias"] = alias
        return self._request(
            "POST",
            f"{self._promote_url}/artifacts/{artifact_id}/v/{version}/promote",
            body,
        )

    def get_registry_audit(self, name: str | None = None, limit: int = 100) -> list[dict]:
        path = f"/v1/registry/audit?limit={limit}"
        if name:
            path += f"&name={name}"
        return self._request("GET", path).get("entries", [])

    def close(self) -> None:
        # urllib opens a connection per request. Present so cells written against
        # StrataClient's .close() keep working.
        return None
