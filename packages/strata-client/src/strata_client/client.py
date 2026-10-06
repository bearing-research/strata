"""Sync and async Python clients for a Strata server.

    from strata_client import StrataClient

    with StrataClient() as client:
        artifact = client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = client.fetch(artifact.uri)

``AsyncStrataClient`` has the same shape with ``await``.

Retry behavior:
    Stream fetches retry 429 responses, honoring ``Retry-After`` when present and
    otherwise waiting ``min(base_delay * 2**attempt + jitter, max_delay)``. With the
    defaults (3 retries, 1s base, 30s cap, up to 1s jitter) the waits are 1-2s,
    2-3s and 4-5s before the 429 is raised. ``max_retries=0`` disables retries.
"""

import asyncio
import json
import random
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc

from strata_client._clientconfig import HasServerUrl, resolve_server_url
from strata_client.filters import Filter, FilterOp, FilterValue

type TransformSpec = Mapping[str, object]
type JsonArtifactInput = Mapping[str, object]
type JsonArtifactData = dict[str, object]
type PutData = JsonArtifactInput | pa.Table | bytes


_ARTIFACT_URI_RE = re.compile(r"^strata://artifact/([^@]+)@v=(\d+)$")


def _parse_artifact_uri(uri: str) -> tuple[str, int]:
    """Parse a canonical artifact URI into ``(artifact_id, version)``."""
    match = _ARTIFACT_URI_RE.match(uri)
    if not match:
        raise ValueError(f"Invalid artifact URI: {uri}")
    return match.group(1), int(match.group(2))


#: Names the artifact a stream served; differs from the materialize response's URI when a
#: duplicate build finished first and this one was superseded.
ARTIFACT_URI_HEADER = "X-Strata-Artifact-Uri"


def _streamed_artifact(response: httpx.Response, artifact_id: str, version: int) -> tuple[str, int]:
    """The ``(artifact_id, version)`` a stream response served, defaulting to the one asked for."""
    uri = response.headers.get(ARTIFACT_URI_HEADER)
    return _parse_artifact_uri(uri) if uri else (artifact_id, version)


#: Set by the by-provenance route on a genuine "nobody has computed this".
PROVENANCE_MISS_HEADER = "X-Strata-Provenance-Miss"


def _provenance_miss(response: httpx.Response) -> bool:
    """Whether this 404 is a genuine provenance miss rather than the store not answering.

    A 404 can also mean the server lacks the route or has no artifact store;
    reading those as misses would recompute forever. Only the route's miss
    header counts, so an older server raises instead of answering "no".
    """
    return response.status_code == 404 and PROVENANCE_MISS_HEADER in response.headers


def _table_to_ipc(table: pa.Table) -> bytes:
    """Serialize an Arrow Table to IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


_JSON_BLOB_KEY = b"strata.json.blob"


def _is_json_blob(table: "pa.Table") -> bool:
    """Whether *table* holds a JSON document rather than columnar data.

    The schema marker decides when present. Unmarked tables count as a blob only
    with one ``data`` column and one row, so a columnar ``{"data": [...]}`` is not misread.
    """
    metadata = table.schema.metadata or {}
    if _JSON_BLOB_KEY in metadata:
        return metadata[_JSON_BLOB_KEY] == b"1"
    return table.num_columns == 1 and table.column_names[0] == "data" and table.num_rows == 1


def _dict_to_ipc(data: JsonArtifactInput) -> bytes:
    """Convert a dict to Arrow IPC: columnar if values are equal-length lists, else JSON."""
    list_values = [value for value in data.values() if isinstance(value, list)]
    if data and len(list_values) == len(data):
        lengths = [len(value) for value in list_values]
        if len(set(lengths)) == 1:
            try:
                table = pa.Table.from_pydict(dict(data))
                # Mark columnar explicitly: otherwise ``{"data": ["only"]}`` (one column, one
                # row) looks like the JSON blob shape and the reader tries to parse it as JSON.
                table = table.replace_schema_metadata({_JSON_BLOB_KEY: b"0"})
                return _table_to_ipc(table)
            except Exception:
                pass  # Fall through to JSON storage

    json_str = json.dumps(dict(data))
    table = pa.Table.from_pydict({"data": [json_str]})
    # Mark the encoding explicitly: a columnar ``{"data": [...]}`` has the same
    # shape, so guessing from the column name reads it back wrong.
    table = table.replace_schema_metadata({_JSON_BLOB_KEY: b"1"})
    return _table_to_ipc(table)


def _convert_to_arrow_ipc(data: PutData) -> bytes:
    """Convert a dict, ``pa.Table``, pandas or polars DataFrame to Arrow IPC bytes.

    ``bytes`` pass through unchanged and are assumed to be Arrow IPC already.
    Raises ``TypeError`` for anything else.
    """
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
        # pandas is optional; without it the value can't be a DataFrame.
        pass

    try:
        import polars as pl

        if isinstance(data, pl.DataFrame):
            return _table_to_ipc(data.to_arrow())
    except ImportError:
        # polars is optional; without it the value can't be a polars DataFrame.
        pass

    raise TypeError(
        f"Unsupported data type: {type(data).__name__}. "
        "Expected dict, pa.Table, pd.DataFrame, pl.DataFrame, or bytes."
    )


class MaterializeRequestBody(TypedDict, total=False):
    """Request payload for /v1/materialize."""

    inputs: list[str]
    transform: dict[str, object]
    mode: str
    name: str
    refresh: bool


class ExplainMaterializeRequestBody(TypedDict, total=False):
    """Request payload for /v1/artifacts/explain-materialize."""

    inputs: list[str]
    transform: dict[str, object]
    name: str


class ArtifactUploadMetadata(TypedDict, total=False):
    """Multipart metadata payload for direct artifact upload."""

    inputs: list[str]
    transform: dict[str, object]
    name: str


@dataclass
class RetryConfig:
    """Retry policy for 429 responses.

    Attributes:
        max_retries: Retry attempts after the first try (0 disables retries).
        base_delay: Seconds before the first retry.
        max_delay: Cap in seconds on any single wait.
        jitter: Maximum random seconds added to each wait.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 1.0

    def calculate_delay(self, attempt: int) -> float:
        """Return the wait in seconds before retry *attempt* (0-indexed), capped at max_delay."""
        exponential = self.base_delay * (2**attempt)
        jitter = random.uniform(0, self.jitter)
        return min(exponential + jitter, self.max_delay)


@dataclass
class Artifact:
    """An immutable, versioned artifact, as returned by ``StrataClient.materialize()``.

    Attributes:
        artifact_id: Unique artifact identifier.
        version: Artifact version number.
        cache_hit: True if the server returned an existing result.
        execution: How it was obtained: "cache", "local", "server" or "stream".
        build_id: Build ID when the artifact was built asynchronously.
        name: Name pointer, if one was assigned.

    Example:
        artifact = client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        df = artifact.to_pandas()
    """

    _client: "StrataClient"
    artifact_id: str
    version: int
    cache_hit: bool = False
    execution: str = "cache"  # "cache" | "local" | "server" | "stream"
    build_id: str | None = None
    name: str | None = None
    _stream_data: bytes | None = None  # Cached stream data from fetch()

    @property
    def uri(self) -> str:
        """Artifact URI (strata://artifact/{id}@v={version})."""
        return f"strata://artifact/{self.artifact_id}@v={self.version}"

    @property
    def name_uri(self) -> str | None:
        """Name URI if a name was assigned (strata://name/{name})."""
        return f"strata://name/{self.name}" if self.name else None

    def info(self) -> dict:
        """Get artifact metadata.

        Returns:
            Dict with artifact_id, version, state, row_count, arrow_schema, and more.
        """
        response = self._client._client.get(f"/v1/artifacts/{self.artifact_id}/v/{self.version}")
        response.raise_for_status()
        return response.json()

    def to_table(self) -> pa.Table:
        """Return the data as an Arrow Table, reusing bytes already streamed by materialize."""
        if self._stream_data is not None:
            if not self._stream_data:
                return pa.table({})
            reader = ipc.open_stream(pa.BufferReader(self._stream_data))
            return reader.read_all()
        return self._client._fetch_artifact_data(self.artifact_id, self.version)

    def to_pandas(self):
        """Download artifact data as pandas DataFrame."""
        return self.to_table().to_pandas()

    def to_polars(self):
        """Download artifact data as Polars DataFrame."""
        import polars as pl

        return pl.from_arrow(self.to_table())

    def lineage(self, max_depth: int = 10) -> dict:
        """Get artifact lineage: what this artifact was computed from (see ``dependents``).

        Args:
            max_depth: Maximum traversal depth.

        Returns:
            Dict with 'nodes' and 'edges'.
        """
        response = self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/lineage",
            params={"max_depth": max_depth},
        )
        response.raise_for_status()
        return response.json()

    def dependents(self, limit: int = 100) -> dict:
        """Get the ready artifacts that read this one as a direct input.

        Args:
            limit: Maximum number of dependents returned.

        Returns:
            Dict with a 'dependents' list.
        """
        response = self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/dependents",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()


class StrataClient:
    """Synchronous client for a Strata server; stream fetches retry 429 with backoff.

    Example:
        with StrataClient() as client:
            artifact = client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )
            table = client.fetch(artifact.uri)
    """

    def __init__(
        self,
        config: "HasServerUrl | None" = None,
        base_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config: Anything with ``server_url``; if omitted, resolved from env and pyproject.toml.
            base_url: Server URL; overrides ``config``.
            retry_config: Retry policy for 429 responses (defaults if None).
        """
        self.config = config
        if base_url is not None:
            self.base_url = base_url
        elif config is not None:
            self.base_url = config.server_url
        else:
            self.base_url = resolve_server_url()
        self.retry_config = retry_config or RetryConfig()
        self._client = httpx.Client(base_url=self.base_url, timeout=300.0)

    @classmethod
    def from_transport(
        cls,
        transport: httpx.BaseTransport,
        *,
        base_url: str = "http://test",
        retry_config: RetryConfig | None = None,
    ) -> "StrataClient":
        """Build a client on a custom ``httpx`` transport, such as ``httpx.MockTransport``.

        Skips server-URL resolution (env vars, pyproject.toml); ``config`` is None.
        """
        client = cls.__new__(cls)
        client.config = None
        client.base_url = base_url
        client.retry_config = retry_config or RetryConfig()
        client._client = httpx.Client(transport=transport, base_url=base_url)
        return client

    def __enter__(self) -> "StrataClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def health(self) -> dict:
        """Check server health."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def metrics(self) -> dict:
        """Get server metrics."""
        response = self._client.get("/metrics")
        response.raise_for_status()
        return response.json()

    # --- Unified Materialize API ---

    def fetch(
        self,
        artifact_uri: str,
        timeout: float = 300.0,
    ) -> pa.Table:
        """Fetch an artifact's data, blocking until it is ready.

        Args:
            artifact_uri: ``strata://artifact/{id}@v={version}``.
            timeout: Seconds to wait for a building artifact.

        Returns:
            Arrow Table with the artifact data.

        Raises:
            RuntimeError: If the build failed.
            TimeoutError: If the artifact is still building after ``timeout``.

        Example:
            table = client.fetch(artifact.uri)
        """
        artifact_id, version = _parse_artifact_uri(artifact_uri)
        return self._fetch_artifact_data_with_wait(artifact_id, version, timeout)

    def _fetch_artifact_data_with_wait(
        self,
        artifact_id: str,
        version: int,
        timeout: float,
    ) -> pa.Table:
        """Fetch artifact data, waiting for it to be ready if necessary."""
        start_time = time.time()

        while True:
            status_resp = self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
            status_resp.raise_for_status()
            status = status_resp.json()

            state = status.get("state", "ready")

            if state == "ready":
                return self._fetch_artifact_data(artifact_id, version)
            elif state == "failed":
                error_msg = status.get("error_message", "Unknown error")
                raise RuntimeError(f"Artifact build failed: {error_msg}")
            elif state == "building":
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Artifact {artifact_id}@v={version} timed out after {timeout}s"
                    )
                time.sleep(0.5)
            else:
                # Unknown state: assume ready.
                return self._fetch_artifact_data(artifact_id, version)

    def _fetch_stream_with_retry(self, stream_url: str) -> httpx.Response:
        """Fetch a stream, retrying on 429 responses."""
        last_response = None
        for attempt in range(self.retry_config.max_retries + 1):
            response = self._client.get(stream_url)

            if response.status_code != 429:
                response.raise_for_status()
                return response

            last_response = response

            if attempt >= self.retry_config.max_retries:
                break

            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after) + random.uniform(0, self.retry_config.jitter)
                except ValueError:
                    delay = self.retry_config.calculate_delay(attempt)
            else:
                delay = self.retry_config.calculate_delay(attempt)

            time.sleep(delay)

        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("Max retries exceeded")

    def clear_cache(self) -> dict:
        """Clear the server's disk cache."""
        response = self._client.post("/v1/cache/clear")
        response.raise_for_status()
        return response.json()

    # --- Artifact API ---

    def materialize(
        self,
        inputs: list[str],
        transform: TransformSpec,
        name: str | None = None,
        mode: str = "stream",
        refresh: bool = False,
        wait: bool = True,
        poll_interval: float = 0.5,
        timeout: float = 300.0,
    ) -> Artifact:
        """Find or compute an artifact via ``POST /v1/materialize``.

        Args:
            inputs: Input URIs (table URIs or artifact URIs).
            transform: ``{"executor": "scan@v1", "params": {...}}`` (``ref`` works too).
            name: Optional name to point at the result.
            mode: "stream" fetches the data now; "artifact" starts an async build.
            refresh: Recompute even if a cached result exists.
            wait: Poll an async build until it finishes.
            poll_interval: Seconds between build status polls.
            timeout: Seconds to wait for the build.

        Returns:
            The resulting Artifact.

        Raises:
            RuntimeError: If the build failed.
            TimeoutError: If the build did not finish within ``timeout``.

        Example:
            artifact = client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={
                    "executor": "scan@v1",
                    "params": {
                        "columns": ["id", "value"],
                        "filters": [{"column": "value", "op": ">", "value": 100}],
                    },
                },
            )
            table = client.fetch(artifact.uri)
        """
        return self._materialize_server(
            inputs=inputs,
            transform=transform,
            name=name,
            mode=mode,
            refresh=refresh,
            wait=wait,
            poll_interval=poll_interval,
            timeout=timeout,
        )

    def _materialize_server(
        self,
        inputs: list[str],
        transform: TransformSpec,
        name: str | None,
        mode: str,
        refresh: bool,
        wait: bool,
        poll_interval: float,
        timeout: float,
    ) -> Artifact:
        """Post to /v1/materialize and turn the response into an Artifact."""
        # The public API documents ``ref``; the server only accepts ``executor``.
        # Keep the rename until ``ref`` goes through a deprecation cycle.
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")

        request_body: MaterializeRequestBody = {
            "inputs": inputs,
            "transform": server_transform,
            "mode": mode,
        }
        if name:
            request_body["name"] = name
        if refresh:
            request_body["refresh"] = True

        response = self._client.post("/v1/materialize", json=request_body)
        response.raise_for_status()
        data = response.json()

        artifact_uri = data["artifact_uri"]
        artifact_id, version = _parse_artifact_uri(artifact_uri)
        hit = data.get("hit", False)
        state = data.get("state", "ready")
        stream_id = data.get("stream_id")
        stream_url = data.get("stream_url")
        build_id = data.get("build_id")

        if hit or state == "ready":
            stream_data = None
            if stream_url and mode == "stream":
                streamed = self._fetch_stream_with_retry(stream_url)
                stream_data = streamed.content
                artifact_id, version = _streamed_artifact(streamed, artifact_id, version)

            return Artifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=hit,
                execution="cache" if hit else "server",
                name=name,
                _stream_data=stream_data,
            )

        if mode == "stream" and stream_url:
            streamed = self._fetch_stream_with_retry(stream_url)
            artifact_id, version = _streamed_artifact(streamed, artifact_id, version)
            return Artifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=False,
                execution="stream",
                name=name,
                _stream_data=streamed.content,
            )

        if not wait:
            return Artifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=False,
                execution="server",
                build_id=build_id or stream_id,
                name=name,
            )

        if build_id:
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Build {build_id} timed out after {timeout}s")

                status_resp = self._client.get(f"/v1/artifacts/builds/{build_id}")
                if status_resp.status_code == 404:
                    status_resp = self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")

                status_resp.raise_for_status()
                build_status = status_resp.json()

                current_state = build_status.get("state", "building")
                if current_state == "ready":
                    return Artifact(
                        _client=self,
                        artifact_id=artifact_id,
                        version=version,
                        cache_hit=False,
                        execution="server",
                        build_id=build_id,
                        name=name,
                    )
                elif current_state == "failed":
                    error_msg = build_status.get("error_message", "Unknown error")
                    raise RuntimeError(f"Build failed: {error_msg}")

                time.sleep(poll_interval)

        # May still be building.
        return Artifact(
            _client=self,
            artifact_id=artifact_id,
            version=version,
            cache_hit=False,
            execution="server",
            build_id=build_id or stream_id,
            name=name,
        )

    def _fetch_artifact_data(self, artifact_id: str, version: int) -> pa.Table:
        """Fetch artifact data by ID and version."""
        response = self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}/data")
        response.raise_for_status()
        reader = ipc.open_stream(pa.BufferReader(response.content))
        return reader.read_all()

    def _fetch_artifact_by_uri(self, artifact_uri: str) -> pa.Table:
        """Fetch artifact data by URI."""
        import re

        match = re.match(r"^strata://artifact/([^@]+)@v=(\d+)$", artifact_uri)
        if not match:
            raise ValueError(f"Invalid artifact URI: {artifact_uri}")

        artifact_id = match.group(1)
        version = int(match.group(2))
        return self._fetch_artifact_data(artifact_id, version)

    def get_artifact(self, artifact_id: str, version: int) -> Artifact:
        """Get an existing artifact by ID and version.

        Args:
            artifact_id: Artifact ID.
            version: Version number.

        Returns:
            The Artifact.

        Raises:
            httpx.HTTPStatusError: If the artifact is not found (404).
        """
        response = self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
        response.raise_for_status()
        return Artifact(_client=self, artifact_id=artifact_id, version=version)

    def find_by_provenance(self, provenance_hash: str) -> dict | None:
        """Ask a shared store whether this computation already has a result.

        A miss returns None. Any other failure (403, timeout, 500, a server
        without the route) raises: "could not answer" is not "no".

        Args:
            provenance_hash: The sha256 provenance hash to look up.

        Returns:
            Dict with artifact_id, version, content_type, state, and more; None on a miss.
        """
        response = self._client.get(f"/v1/artifacts/by-provenance/{provenance_hash}")
        if _provenance_miss(response):
            return None
        response.raise_for_status()
        return response.json()

    def put_by_provenance(
        self,
        provenance_hash: str,
        blob: bytes,
        *,
        content_type: str,
        variable_name: str | None = None,
    ) -> dict:
        """Store an opaque blob under a provenance key the server cannot compute itself.

        First writer wins: a hash the store already holds returns the incumbent
        with ``hit=True`` rather than replacing it.

        Args:
            provenance_hash: The sha256 key to store under.
            blob: The serialized value, exactly as it should come back.
            content_type: How readers decode the blob; nothing else records it.
            variable_name: Optional label, recorded for readability.

        Returns:
            Dict with artifact_uri, hit, byte_size.
        """
        metadata: dict[str, str] = {"content_type": content_type}
        if variable_name:
            metadata["variable_name"] = variable_name
        response = self._client.put(
            f"/v1/artifacts/by-provenance/{provenance_hash}",
            files={
                "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
                "data": ("data.bin", blob, "application/octet-stream"),
            },
        )
        response.raise_for_status()
        return response.json()

    def get_artifact_by_name(self, name: str) -> Artifact:
        """Get the artifact a name points to.

        Args:
            name: Artifact name.

        Returns:
            The Artifact.

        Raises:
            httpx.HTTPStatusError: If the name is not found (404).
        """
        resolved = self.resolve_name(name)
        artifact_id, version = _parse_artifact_uri(resolved["artifact_uri"])
        return Artifact(
            _client=self,
            artifact_id=artifact_id,
            version=version,
            name=name,
        )

    def resolve_name(self, name: str) -> dict:
        """Resolve a name to its artifact.

        Args:
            name: Name to resolve.

        Returns:
            Dict with artifact_uri, version, updated_at.
        """
        response = self._client.get(f"/v1/names/{name}")
        response.raise_for_status()
        return response.json()

    def set_name(self, name: str, artifact_id: str, version: int) -> dict:
        """Set or update a name pointer.

        Args:
            name: Name to set.
            artifact_id: Target artifact ID.
            version: Target version.

        Returns:
            Dict with name_uri and artifact_uri.
        """
        response = self._client.post(
            "/v1/names",
            json={"name": name, "artifact_id": artifact_id, "version": version},
        )
        response.raise_for_status()
        return response.json()

    # --- Registry: aliases, tags, audit ---

    def set_alias(self, name: str, alias: str, artifact_id: str, version: int) -> dict:
        """Point ``name @ alias`` (e.g. champion) at an artifact version.

        A name can hold several aliases; every move is recorded in the registry audit.
        """
        response = self._client.put(
            f"/v1/names/{name}/aliases/{alias}",
            json={"artifact_id": artifact_id, "version": version},
        )
        response.raise_for_status()
        return response.json()

    def resolve_alias(self, name: str, alias: str) -> dict:
        """Resolve ``name @ alias`` to its artifact version info."""
        response = self._client.get(f"/v1/names/{name}/aliases/{alias}")
        response.raise_for_status()
        return response.json()

    def list_aliases(self, name: str) -> list[dict]:
        """List the aliases held by a registry name."""
        response = self._client.get(f"/v1/names/{name}/aliases")
        response.raise_for_status()
        return response.json()["aliases"]

    def delete_alias(self, name: str, alias: str) -> dict:
        """Delete ``name @ alias``."""
        response = self._client.delete(f"/v1/names/{name}/aliases/{alias}")
        response.raise_for_status()
        return response.json()

    def set_tag(self, artifact_id: str, version: int, key: str, value: str) -> dict:
        """Set a key/value tag on an artifact version (e.g. auc=0.91)."""
        response = self._client.put(
            f"/v1/artifacts/{artifact_id}/v/{version}/tags",
            json={"key": key, "value": str(value)},
        )
        response.raise_for_status()
        return response.json()

    def get_tags(self, artifact_id: str, version: int) -> dict[str, str]:
        """Get the tags on an artifact version."""
        response = self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}/tags")
        response.raise_for_status()
        return response.json()["tags"]

    def delete_tag(self, artifact_id: str, version: int, key: str) -> dict:
        """Delete one tag from an artifact version."""
        response = self._client.delete(f"/v1/artifacts/{artifact_id}/v/{version}/tags/{key}")
        response.raise_for_status()
        return response.json()

    def list_pending_changes(self) -> list[dict]:
        """List protected-alias changes awaiting approval."""
        response = self._client.get("/v1/registry/pending")
        response.raise_for_status()
        return response.json()["pending"]

    def approve_alias_change(self, name: str, alias: str) -> dict:
        """Apply a pending protected-alias change."""
        response = self._client.post(
            "/v1/registry/pending/approve", json={"name": name, "alias": alias}
        )
        response.raise_for_status()
        return response.json()

    def reject_alias_change(self, name: str, alias: str) -> dict:
        """Discard a pending protected-alias change."""
        response = self._client.post(
            "/v1/registry/pending/reject", json={"name": name, "alias": alias}
        )
        response.raise_for_status()
        return response.json()

    def get_registry_audit(
        self,
        name: str | None = None,
        artifact_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Read the append-only registry audit, newest first.

        Every name, alias and tag change is recorded with actor, from/to versions and time.
        """
        params: dict = {"limit": limit}
        if name is not None:
            params["name"] = name
        if artifact_id is not None:
            params["artifact_id"] = artifact_id
        response = self._client.get("/v1/registry/audit", params=params)
        response.raise_for_status()
        return response.json()["entries"]

    # --- Direct Artifact Upload (for local execution) ---

    def put(
        self,
        inputs: list[str],
        transform: TransformSpec,
        data: PutData,
        name: str | None = None,
    ) -> Artifact:
        """Upload a locally computed result as an artifact, deduplicated by provenance.

        A dict whose values are equal-length lists is stored as columns; any
        other dict is stored as one JSON document.

        Args:
            inputs: Input URIs (artifact or table URIs), recorded for lineage.
            transform: Executor and params; opaque to Strata, used only for dedup.
            data: dict, ``pa.Table``, pandas or polars DataFrame, or Arrow IPC bytes.
            name: Optional name to assign to the artifact.

        Returns:
            The Artifact; ``cache_hit`` is True if an identical one already existed.

        Example:
            artifact = client.put(
                inputs=[],
                transform={"executor": "my_step@v1", "params": {}},
                data={"result": "value", "scores": [1, 2, 3]},
            )
        """
        arrow_bytes = _convert_to_arrow_ipc(data)

        # Map 'ref' to 'executor' for server compatibility
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")

        import json

        metadata: ArtifactUploadMetadata = {
            "inputs": inputs,
            "transform": server_transform,
        }
        if name:
            metadata["name"] = name

        files = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
            "data": ("data.arrow", arrow_bytes, "application/vnd.apache.arrow.stream"),
        }

        response = self._client.put("/v1/artifacts", files=files)
        response.raise_for_status()
        result = response.json()

        artifact_uri = result["artifact_uri"]
        artifact_id, version = _parse_artifact_uri(artifact_uri)

        return Artifact(
            _client=self,
            artifact_id=artifact_id,
            version=version,
            cache_hit=result.get("hit", False),
            execution="cache" if result.get("hit") else "local",
            name=name,
        )

    def put_json(
        self,
        inputs: list[str],
        transform: TransformSpec,
        data: JsonArtifactInput,
        name: str | None = None,
    ) -> Artifact:
        """Upload a JSON-able dict as an artifact; same as :meth:`put`.

        Args:
            inputs: Input URIs, recorded for lineage.
            transform: Executor and params, used for dedup.
            data: JSON data to persist.
            name: Optional name to assign.

        Returns:
            The Artifact.
        """
        return self.put(inputs=inputs, transform=transform, data=data, name=name)

    def get_json(self, artifact_uri: str) -> JsonArtifactData:
        """Fetch an artifact as a dict: the parsed JSON document, or ``{column: values}``.

        Args:
            artifact_uri: ``strata://artifact/{id}@v={version}``.

        Returns:
            The artifact data as a dict.

        Example:
            data = client.get_json("strata://artifact/abc@v=1")
        """
        table = self.fetch(artifact_uri)

        if _is_json_blob(table):
            import json

            json_str = table.column("data")[0].as_py()
            return cast(JsonArtifactData, json.loads(json_str))

        return cast(JsonArtifactData, table.to_pydict())

    # --- Artifact Lifecycle Management ---

    def list_artifacts(
        self,
        limit: int = 100,
        offset: int = 0,
        state: str | None = None,
        name_prefix: str | None = None,
    ) -> dict:
        """List artifacts with optional filtering.

        Args:
            limit: Maximum number of artifacts to return.
            offset: Number of artifacts to skip, for pagination.
            state: Only this state ("ready", "building", "failed").
            name_prefix: Only artifacts with a name starting with this prefix.

        Returns:
            Dict with an 'artifacts' list and pagination info.
        """
        params: dict[str, str | int | float | None] = {"limit": limit, "offset": offset}
        if state is not None:
            params["state"] = state
        if name_prefix is not None:
            params["name_prefix"] = name_prefix

        response = self._client.get("/v1/artifacts", params=params)
        response.raise_for_status()
        return response.json()

    def delete_artifact(self, artifact_id: str, version: int) -> dict:
        """Delete an artifact version's blob and metadata, and any names pointing at it.

        Args:
            artifact_id: Artifact ID.
            version: Version number.

        Returns:
            Dict with the deletion status.
        """
        response = self._client.delete(f"/v1/artifacts/{artifact_id}/v/{version}")
        response.raise_for_status()
        return response.json()

    def garbage_collect(
        self,
        *,
        max_idle_days: float | None = None,
        max_bytes: int | None = None,
        min_idle_seconds: float | None = None,
        collect_latest: bool = False,
        dry_run: bool = False,
    ) -> dict:
        """Collect the artifact versions nothing needs, least recently used first.

        Nothing named, aliased, pinned or published is collected, nor anything
        those depend on, nor the current value of a caller-chosen id (a
        notebook's cell outputs). An unnamed ``materialize`` result is a cache
        entry: name or pin it to keep it. Omitted arguments take the server's
        configured retention, so a bare call matches its scheduled sweep.

        Args:
            max_idle_days: Collect what has not been used for this long.
            max_bytes: Collect until the store is at 80% of this many bytes.
            min_idle_seconds: Never collect anything used more recently.
            collect_latest: Also collect current values of caller-named ids (deletes live state).
            dry_run: Report what would go, and delete nothing.

        Returns:
            Dict with deleted_count, deleted_bytes, store_bytes, and ``collected`` on a dry run.
        """
        params: dict[str, float | int | bool] = {
            "collect_latest": collect_latest,
            "dry_run": dry_run,
        }
        for name, value in (
            ("max_idle_days", max_idle_days),
            ("max_bytes", max_bytes),
            ("min_idle_seconds", min_idle_seconds),
        ):
            if value is not None:
                params[name] = value
        response = self._client.post("/v1/artifacts/gc", params=params)
        response.raise_for_status()
        return response.json()

    def get_artifact_usage(self) -> dict:
        """Get artifact store usage metrics.

        Returns:
            Dict with total_bytes, total_versions, unreferenced_count, and more.
        """
        response = self._client.get("/v1/artifacts/usage")
        response.raise_for_status()
        return response.json()

    # --- Staleness Detection ---

    def get_name_status(self, name: str) -> dict:
        """Report whether a named artifact's inputs have changed since it was built.

        Args:
            name: Name to check, without the ``strata://name/`` prefix.

        Returns:
            Dict with name, artifact_uri, version, state, input_versions,
            is_stale, stale_reason and changed_inputs.

        Example:
            >>> status = client.get_name_status("daily_revenue")
            >>> if status["is_stale"]:
            ...     print(status["stale_reason"])
            Rebuild needed: file:///warehouse#db.events: 123 → 456
        """
        response = self._client.get(f"/v1/artifacts/names/{name}/status")
        response.raise_for_status()
        return response.json()

    def explain_materialize(
        self,
        inputs: list[str],
        transform: TransformSpec,
        name: str | None = None,
    ) -> dict:
        """Dry-run materialize: report cache hit or miss and, if stale, which inputs changed.

        Args:
            inputs: Input URIs (table URIs or artifact URIs).
            transform: ``{"ref": "duckdb_sql@v1", "params": {...}}``.
            name: Optional name to check staleness against.

        Returns:
            Dict with cache_hit, artifact_uri, provenance_hash, is_stale,
            stale_reason and execution ("cache", "local" or "server").

        Example:
            >>> result = client.explain_materialize(
            ...     inputs=["file:///warehouse#db.events"],
            ...     transform={"ref": "duckdb_sql@v1", "params": {"sql": "SELECT * FROM input0"}},
            ...     name="my_transform",
            ... )
            >>> if result["is_stale"]:
            ...     print(result["stale_reason"])
        """
        # Map 'ref' to 'executor' for server compatibility
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")

        request_body: ExplainMaterializeRequestBody = {
            "inputs": inputs,
            "transform": server_transform,
        }
        if name:
            request_body["name"] = name
        response = self._client.post("/v1/artifacts/explain-materialize", json=request_body)
        response.raise_for_status()
        return response.json()

    def is_artifact_stale(self, name: str) -> bool:
        """Check whether a named artifact is stale.

        Args:
            name: Name to check.

        Returns:
            True if the artifact's inputs have changed since it was built.

        Raises:
            httpx.HTTPStatusError: If the name is not found (404).
        """
        status = self.get_name_status(name)
        return status.get("is_stale", False)


class AsyncStrataClient:
    """Async client for a Strata server; stream fetches retry 429 with backoff.

    Example:
        async with AsyncStrataClient() as client:
            artifact = await client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )
            table = await client.fetch(artifact.uri)
    """

    def __init__(
        self,
        config: "HasServerUrl | None" = None,
        base_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the async client.

        Args:
            config: Anything with ``server_url``; if omitted, resolved from env and pyproject.toml.
            base_url: Server URL; overrides ``config``.
            retry_config: Retry policy for 429 responses (defaults if None).
        """
        self.config = config
        if base_url is not None:
            self.base_url = base_url
        elif config is not None:
            self.base_url = config.server_url
        else:
            self.base_url = resolve_server_url()
        self.retry_config = retry_config or RetryConfig()
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=300.0)

    async def __aenter__(self) -> "AsyncStrataClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()

    async def health(self) -> dict:
        """Check server health."""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def metrics(self) -> dict:
        """Get server metrics."""
        response = await self._client.get("/metrics")
        response.raise_for_status()
        return response.json()

    # --- Unified Materialize API ---

    async def fetch(
        self,
        artifact_uri: str,
        timeout: float = 300.0,
    ) -> pa.Table:
        """Fetch an artifact's data, waiting until it is ready.

        Args:
            artifact_uri: ``strata://artifact/{id}@v={version}``.
            timeout: Seconds to wait for a building artifact.

        Returns:
            Arrow Table with the artifact data.

        Raises:
            RuntimeError: If the build failed.
            TimeoutError: If the artifact is still building after ``timeout``.

        Example:
            table = await client.fetch(artifact.uri)
        """
        artifact_id, version = _parse_artifact_uri(artifact_uri)
        return await self._fetch_artifact_data_with_wait(artifact_id, version, timeout)

    async def _fetch_artifact_data_with_wait(
        self,
        artifact_id: str,
        version: int,
        timeout: float,
    ) -> pa.Table:
        """Fetch artifact data, waiting for it to be ready if necessary."""
        start_time = time.time()

        while True:
            status_resp = await self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
            status_resp.raise_for_status()
            status = status_resp.json()

            state = status.get("state", "ready")

            if state == "ready":
                return await self._fetch_artifact_data(artifact_id, version)
            elif state == "failed":
                error_msg = status.get("error_message", "Unknown error")
                raise RuntimeError(f"Artifact build failed: {error_msg}")
            elif state == "building":
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Artifact {artifact_id}@v={version} timed out after {timeout}s"
                    )
                await asyncio.sleep(0.5)
            else:
                # Unknown state: assume ready.
                return await self._fetch_artifact_data(artifact_id, version)

    async def find_by_provenance(self, provenance_hash: str) -> dict | None:
        """Ask a shared store whether this computation already has a result.

        Returns the match's metadata, or None on a genuine miss; any other
        failure raises (see :meth:`StrataClient.find_by_provenance`).
        """
        response = await self._client.get(f"/v1/artifacts/by-provenance/{provenance_hash}")
        if _provenance_miss(response):
            return None
        response.raise_for_status()
        return response.json()

    async def _fetch_artifact_data(self, artifact_id: str, version: int) -> pa.Table:
        """Fetch artifact data by ID and version."""
        response = await self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}/data")
        response.raise_for_status()
        reader = ipc.open_stream(pa.BufferReader(response.content))
        return reader.read_all()

    async def _fetch_stream_with_retry(self, stream_url: str) -> httpx.Response:
        """Fetch a stream, retrying on 429 responses."""
        last_response = None
        for attempt in range(self.retry_config.max_retries + 1):
            response = await self._client.get(stream_url)

            if response.status_code != 429:
                response.raise_for_status()
                return response

            last_response = response

            if attempt >= self.retry_config.max_retries:
                break

            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after) + random.uniform(0, self.retry_config.jitter)
                except ValueError:
                    delay = self.retry_config.calculate_delay(attempt)
            else:
                delay = self.retry_config.calculate_delay(attempt)

            await asyncio.sleep(delay)

        if last_response is not None:
            last_response.raise_for_status()
        raise RuntimeError("Max retries exceeded")

    async def materialize(
        self,
        inputs: list[str],
        transform: TransformSpec,
        name: str | None = None,
        mode: str = "stream",
        refresh: bool = False,
        wait: bool = True,
        poll_interval: float = 0.5,
        timeout: float = 300.0,
    ) -> "AsyncArtifact":
        """Find or compute an artifact via ``POST /v1/materialize``.

        Args:
            inputs: Input URIs (table URIs or artifact URIs).
            transform: ``{"executor": "scan@v1", "params": {...}}`` (``ref`` works too).
            name: Optional name to point at the result.
            mode: "stream" fetches the data now; "artifact" starts an async build.
            refresh: Recompute even if a cached result exists.
            wait: Poll an async build until it finishes.
            poll_interval: Seconds between build status polls.
            timeout: Seconds to wait for the build.

        Returns:
            The resulting AsyncArtifact.

        Raises:
            RuntimeError: If the build failed.
            TimeoutError: If the build did not finish within ``timeout``.

        Example:
            artifact = await client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )
            table = await client.fetch(artifact.uri)
        """
        # Map 'ref' to 'executor' for server compatibility
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")

        request_body: MaterializeRequestBody = {
            "inputs": inputs,
            "transform": server_transform,
            "mode": mode,
        }
        if name:
            request_body["name"] = name
        if refresh:
            request_body["refresh"] = True

        response = await self._client.post("/v1/materialize", json=request_body)
        response.raise_for_status()
        data = response.json()

        artifact_uri = data["artifact_uri"]
        artifact_id, version = _parse_artifact_uri(artifact_uri)
        hit = data.get("hit", False)
        state = data.get("state", "ready")
        stream_id = data.get("stream_id")
        stream_url = data.get("stream_url")
        build_id = data.get("build_id")

        if hit or state == "ready":
            stream_data = None
            if stream_url and mode == "stream":
                streamed = await self._fetch_stream_with_retry(stream_url)
                stream_data = streamed.content
                artifact_id, version = _streamed_artifact(streamed, artifact_id, version)

            return AsyncArtifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=hit,
                execution="cache" if hit else "server",
                name=name,
                _stream_data=stream_data,
            )

        if mode == "stream" and stream_url:
            streamed = await self._fetch_stream_with_retry(stream_url)
            artifact_id, version = _streamed_artifact(streamed, artifact_id, version)
            return AsyncArtifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=False,
                execution="stream",
                name=name,
                _stream_data=streamed.content,
            )

        if build_id and wait:
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Build {build_id} timed out after {timeout}s")

                status_resp = await self._client.get(f"/v1/artifacts/builds/{build_id}")
                if status_resp.status_code == 404:
                    status_resp = await self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")

                status_resp.raise_for_status()
                build_status = status_resp.json()

                current_state = build_status.get("state", "building")
                if current_state == "ready":
                    return AsyncArtifact(
                        _client=self,
                        artifact_id=artifact_id,
                        version=version,
                        cache_hit=False,
                        execution="server",
                        build_id=build_id,
                        name=name,
                    )
                elif current_state == "failed":
                    error_msg = build_status.get("error_message", "Unknown error")
                    raise RuntimeError(f"Build failed: {error_msg}")

                await asyncio.sleep(poll_interval)

        return AsyncArtifact(
            _client=self,
            artifact_id=artifact_id,
            version=version,
            cache_hit=False,
            execution="server",
            build_id=build_id or stream_id,
            name=name,
        )

    async def clear_cache(self) -> dict:
        """Clear the server's disk cache."""
        response = await self._client.post("/v1/cache/clear")
        response.raise_for_status()
        return response.json()

    async def get_artifact(self, artifact_id: str, version: int) -> "AsyncArtifact":
        """Get an existing artifact by ID and version."""
        response = await self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
        response.raise_for_status()
        return AsyncArtifact(_client=self, artifact_id=artifact_id, version=version)

    async def get_artifact_by_name(self, name: str) -> "AsyncArtifact":
        """Get an artifact by its name."""
        response = await self._client.get(f"/v1/names/{name}")
        response.raise_for_status()
        resolved = response.json()

        artifact_id, version = _parse_artifact_uri(resolved["artifact_uri"])
        return AsyncArtifact(
            _client=self,
            artifact_id=artifact_id,
            version=version,
            name=name,
        )

    # --- Direct Artifact Upload (for local execution) ---

    async def put(
        self,
        inputs: list[str],
        transform: TransformSpec,
        data: PutData,
        name: str | None = None,
    ) -> "AsyncArtifact":
        """Upload a locally computed result as an artifact, deduplicated by provenance.

        A dict whose values are equal-length lists is stored as columns; any
        other dict is stored as one JSON document.

        Args:
            inputs: Input URIs (artifact or table URIs), recorded for lineage.
            transform: Executor and params; opaque to Strata, used only for dedup.
            data: dict, ``pa.Table``, pandas or polars DataFrame, or Arrow IPC bytes.
            name: Optional name to assign to the artifact.

        Returns:
            The AsyncArtifact; ``cache_hit`` is True if an identical one already existed.

        Example:
            artifact = await client.put(
                inputs=[],
                transform={"executor": "my_step@v1", "params": {}},
                data={"result": "value", "scores": [1, 2, 3]},
            )
        """
        arrow_bytes = _convert_to_arrow_ipc(data)

        # Map 'ref' to 'executor' for server compatibility
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")

        import json

        metadata: ArtifactUploadMetadata = {
            "inputs": inputs,
            "transform": server_transform,
        }
        if name:
            metadata["name"] = name

        files = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
            "data": ("data.arrow", arrow_bytes, "application/vnd.apache.arrow.stream"),
        }

        response = await self._client.put("/v1/artifacts", files=files)
        response.raise_for_status()
        result = response.json()

        artifact_uri = result["artifact_uri"]
        artifact_id, version = _parse_artifact_uri(artifact_uri)

        return AsyncArtifact(
            _client=self,
            artifact_id=artifact_id,
            version=version,
            cache_hit=result.get("hit", False),
            execution="cache" if result.get("hit") else "local",
            name=name,
        )

    async def put_json(
        self,
        inputs: list[str],
        transform: TransformSpec,
        data: JsonArtifactInput,
        name: str | None = None,
    ) -> "AsyncArtifact":
        """Upload a JSON-able dict as an artifact; same as :meth:`put`.

        Args:
            inputs: Input URIs, recorded for lineage.
            transform: Executor and params, used for dedup.
            data: JSON data to persist.
            name: Optional name to assign.

        Returns:
            The AsyncArtifact.
        """
        return await self.put(inputs=inputs, transform=transform, data=data, name=name)

    async def get_json(self, artifact_uri: str) -> JsonArtifactData:
        """Fetch an artifact as a dict: the parsed JSON document, or ``{column: values}``.

        Args:
            artifact_uri: ``strata://artifact/{id}@v={version}``.

        Returns:
            The artifact data as a dict.

        Example:
            data = await client.get_json("strata://artifact/abc@v=1")
        """
        table = await self.fetch(artifact_uri)

        if _is_json_blob(table):
            import json

            json_str = table.column("data")[0].as_py()
            return cast(JsonArtifactData, json.loads(json_str))

        return cast(JsonArtifactData, table.to_pydict())


@dataclass
class AsyncArtifact:
    """An immutable, versioned artifact, as returned by ``AsyncStrataClient.materialize()``.

    Example:
        artifact = await client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = await artifact.to_table()
    """

    _client: "AsyncStrataClient"
    artifact_id: str
    version: int
    cache_hit: bool = False
    execution: str = "cache"  # "cache" | "server" | "stream"
    build_id: str | None = None
    name: str | None = None
    _stream_data: bytes | None = None  # Cached stream data from fetch()

    @property
    def uri(self) -> str:
        """Artifact URI (strata://artifact/{id}@v={version})."""
        return f"strata://artifact/{self.artifact_id}@v={self.version}"

    @property
    def name_uri(self) -> str | None:
        """Name URI if a name was assigned."""
        return f"strata://name/{self.name}" if self.name else None

    async def info(self) -> dict:
        """Get artifact metadata."""
        response = await self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}"
        )
        response.raise_for_status()
        return response.json()

    async def to_table(self) -> pa.Table:
        """Return the data as an Arrow Table, reusing bytes already streamed by materialize."""
        if self._stream_data is not None:
            if not self._stream_data:
                return pa.table({})
            reader = ipc.open_stream(pa.BufferReader(self._stream_data))
            return reader.read_all()
        response = await self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/data"
        )
        response.raise_for_status()
        reader = ipc.open_stream(pa.BufferReader(response.content))
        return reader.read_all()

    async def to_pandas(self):
        """Download artifact data as pandas DataFrame."""
        table = await self.to_table()
        return table.to_pandas()

    async def to_polars(self):
        """Download artifact data as Polars DataFrame."""
        import polars as pl

        table = await self.to_table()
        return pl.from_arrow(table)

    async def lineage(self, max_depth: int = 10) -> dict:
        """Get artifact lineage: what this artifact was computed from."""
        response = await self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/lineage",
            params={"max_depth": max_depth},
        )
        response.raise_for_status()
        return response.json()

    async def dependents(self, limit: int = 100) -> dict:
        """Get the ready artifacts that read this one as a direct input."""
        response = await self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/dependents",
            params={"limit": limit},
        )
        response.raise_for_status()
        return response.json()


def eq(column: str, value: FilterValue) -> Filter:
    """Create an equality filter."""
    return Filter(column=column, op=FilterOp.EQ, value=value)


def ne(column: str, value: FilterValue) -> Filter:
    """Create a not-equal filter."""
    return Filter(column=column, op=FilterOp.NE, value=value)


def lt(column: str, value: FilterValue) -> Filter:
    """Create a less-than filter."""
    return Filter(column=column, op=FilterOp.LT, value=value)


def le(column: str, value: FilterValue) -> Filter:
    """Create a less-than-or-equal filter."""
    return Filter(column=column, op=FilterOp.LE, value=value)


def gt(column: str, value: FilterValue) -> Filter:
    """Create a greater-than filter."""
    return Filter(column=column, op=FilterOp.GT, value=value)


def ge(column: str, value: FilterValue) -> Filter:
    """Create a greater-than-or-equal filter."""
    return Filter(column=column, op=FilterOp.GE, value=value)
