"""Python client for Strata server.

Provides both sync and async clients for interacting with Strata:

    # Sync client (for scripts, notebooks, CLI tools)
    from strata.client import StrataClient

    with StrataClient() as client:
        # Materialize creates/finds an artifact
        artifact = client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        # Fetch downloads the data
        table = client.fetch(artifact.uri)

    # Async client (for FastAPI, asyncio applications)
    from strata.client import AsyncStrataClient

    async with AsyncStrataClient() as client:
        artifact = await client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        table = await client.fetch(artifact.uri)

Retry Behavior:
    Both clients implement automatic retry with exponential backoff and jitter
    for 429 (Too Many Requests) responses. This turns server overload from a
    hard failure into self-throttling behavior.

    Default settings:
    - max_retries: 3 (total attempts = 4)
    - base_delay: 1.0 seconds
    - max_delay: 30.0 seconds
    - jitter: 0.0-1.0 seconds random addition

    Backoff formula: min(base_delay * 2^attempt + jitter, max_delay)

    Example with defaults:
    - Attempt 1 fails: wait 1.0-2.0s
    - Attempt 2 fails: wait 2.0-3.0s
    - Attempt 3 fails: wait 4.0-5.0s
    - Attempt 4 fails: raise exception

    To disable retries, set max_retries=0.
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

from strata.config import StrataConfig
from strata.types import Filter, FilterOp, FilterValue

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


def _table_to_ipc(table: pa.Table) -> bytes:
    """Serialize an Arrow Table to IPC stream bytes."""
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def _dict_to_ipc(data: JsonArtifactInput) -> bytes:
    """Convert a dict to Arrow IPC stream bytes.

    If all values are equal-length lists the dict is treated as columnar
    and converted directly. Otherwise it's stored as a single JSON column.
    """
    list_values = [value for value in data.values() if isinstance(value, list)]
    if data and len(list_values) == len(data):
        lengths = [len(value) for value in list_values]
        if len(set(lengths)) == 1:
            try:
                table = pa.Table.from_pydict(dict(data))
                return _table_to_ipc(table)
            except Exception:
                pass  # Fall through to JSON storage

    json_str = json.dumps(dict(data))
    table = pa.Table.from_pydict({"data": [json_str]})
    return _table_to_ipc(table)


def _convert_to_arrow_ipc(data: PutData) -> bytes:
    """Convert various supported data types to Arrow IPC bytes.

    Supports: ``dict`` (treated columnar or stored as JSON), ``pa.Table``,
    pandas DataFrame, polars DataFrame, and pre-encoded ``bytes`` (assumed
    to already be Arrow IPC).
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
    """Configuration for retry behavior on 429 responses.

    Attributes:
        max_retries: Maximum number of retry attempts (0 to disable)
        base_delay: Initial delay in seconds before first retry
        max_delay: Maximum delay between retries (caps exponential growth)
        jitter: Maximum random jitter to add (spreads out retries)
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 1.0

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given retry attempt (0-indexed).

        Uses exponential backoff: base_delay * 2^attempt + random_jitter
        Capped at max_delay.
        """
        exponential = self.base_delay * (2**attempt)
        jitter = random.uniform(0, self.jitter)
        return min(exponential + jitter, self.max_delay)


@dataclass
class Artifact:
    """An immutable, versioned artifact from Strata.

    Returned by `client.materialize()`. Provides access to artifact data and metadata.

    Attributes:
        artifact_id: Unique artifact identifier
        version: Artifact version number
        cache_hit: True if artifact was returned from cache
        execution: How the artifact was obtained ("cache", "local", "server", "stream")
        build_id: Build ID if artifact was built asynchronously
        name: Name pointer if one was assigned

    Example:
        artifact = client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        print(f"Artifact: {artifact.uri}")
        print(f"Cache hit: {artifact.cache_hit}")

        # Download the data
        table = client.fetch(artifact.uri)
        df = table.to_pandas()
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
            Dict with artifact_id, version, state, size_bytes, row_count,
            created_at, arrow_schema, etc.
        """
        response = self._client._client.get(f"/v1/artifacts/{self.artifact_id}/v/{self.version}")
        response.raise_for_status()
        return response.json()

    def to_table(self) -> pa.Table:
        """Download artifact data as Arrow Table."""
        # Use cached stream data if available (from fetch with stream mode)
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

    def lineage(self, direction: str = "upstream", max_depth: int = 10) -> dict:
        """Get artifact lineage (dependency graph).

        Args:
            direction: "upstream" (inputs) or "downstream" (dependents)
            max_depth: Maximum traversal depth

        Returns:
            Dict with 'nodes' and 'edges' describing the lineage graph
        """
        response = self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/lineage",
            params={"direction": direction, "max_depth": max_depth},
        )
        response.raise_for_status()
        return response.json()

    def dependents(self, max_depth: int = 10) -> dict:
        """Get artifacts that depend on this artifact.

        Args:
            max_depth: Maximum traversal depth

        Returns:
            Dict with 'dependents' list
        """
        response = self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/dependents",
            params={"max_depth": max_depth},
        )
        response.raise_for_status()
        return response.json()


class StrataClient:
    """Client for interacting with a Strata server.

    Implements automatic retry with exponential backoff for 429 responses,
    turning server overload into self-throttling behavior.

    Example:
        client = StrataClient()

        # Materialize creates/finds an artifact
        artifact = client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )

        # Fetch downloads the data (blocks until artifact is ready)
        table = client.fetch(artifact.uri)

        # Disable retries
        client = StrataClient(retry_config=RetryConfig(max_retries=0))
    """

    def __init__(
        self,
        config: StrataConfig | None = None,
        base_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            config: Strata configuration (uses StrataConfig.load() if None)
            base_url: Override the server URL from config
            retry_config: Retry configuration for 429 responses (uses defaults if None)
        """
        self.config = config or StrataConfig.load()
        self.base_url = base_url or self.config.server_url
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
        """Build a client backed by a custom ``httpx`` transport.

        Intended for tests that want to inject ``httpx.MockTransport``
        without firing ``StrataConfig.load()`` (which reads pyproject /
        env vars). The returned client is fully constructed; close it
        via ``client.close()`` or use it as a context manager.

        ``config`` is left as ``None`` since transport-injected clients
        bypass the normal config-load path.
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

    # -----------------------------------------------------------------------
    # Unified Materialize API
    # -----------------------------------------------------------------------

    def fetch(
        self,
        artifact_uri: str,
        timeout: float = 300.0,
    ) -> pa.Table:
        """Fetch data from an artifact URI.

        Blocks until the artifact is ready, then downloads and returns the data.
        This is the second step after materialize() - it retrieves data from
        an already-materialized artifact.

        Args:
            artifact_uri: Artifact URI (e.g., "strata://artifact/{id}@v={version}")
            timeout: Maximum seconds to wait for artifact to be ready (default 300)

        Returns:
            Arrow Table containing the artifact data

        Example:
            # Step 1: Materialize creates/finds an artifact
            artifact = client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )

            # Step 2: Fetch downloads the data
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
            # Check artifact status
            status_resp = self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
            status_resp.raise_for_status()
            status = status_resp.json()

            state = status.get("state", "ready")

            if state == "ready":
                # Artifact is ready, download data
                return self._fetch_artifact_data(artifact_id, version)
            elif state == "failed":
                error_msg = status.get("error_message", "Unknown error")
                raise RuntimeError(f"Artifact build failed: {error_msg}")
            elif state == "building":
                # Still building, check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Artifact {artifact_id}@v={version} timed out after {timeout}s"
                    )
                time.sleep(0.5)
            else:
                # Unknown state, assume ready and try to fetch
                return self._fetch_artifact_data(artifact_id, version)

    def _fetch_stream_with_retry(self, stream_url: str) -> bytes:
        """Fetch stream data with retry on 429 responses."""
        last_response = None
        for attempt in range(self.retry_config.max_retries + 1):
            response = self._client.get(stream_url)

            if response.status_code != 429:
                response.raise_for_status()
                return response.content

            last_response = response

            if attempt >= self.retry_config.max_retries:
                break

            # Calculate delay
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

    # -----------------------------------------------------------------------
    # Artifact API
    # -----------------------------------------------------------------------

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
        """Materialize a computed artifact.

        This is the main entry point for creating artifacts. Sends a request
        to the unified /v1/materialize endpoint.

        Args:
            inputs: List of input URIs (table URIs or artifact URIs)
            transform: Transform specification with "executor" and "params"
                Example: {"executor": "scan@v1", "params": {}}
            name: Optional name to assign to the result
            mode: "stream" (default) for immediate data, "artifact" for async build
            refresh: Force recompute even if cached
            wait: Wait for async builds to complete
            poll_interval: Seconds between build status polls
            timeout: Maximum seconds to wait for build

        Returns:
            Artifact object with access to metadata and data

        Example:
            # Identity transform (read from table)
            artifact = client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )
            table = client.fetch(artifact.uri)

            # With column projection and filters
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
        """Request server-side execution via unified /v1/materialize endpoint."""
        # Public API documents transform={"ref": ..., "params": ...}; the
        # server only accepts ``executor``. This shim renames so callers can
        # use either spelling. Deleting it would silently break every
        # ``ref``-style call site (examples/10_artifacts.py plus the
        # integration test suite); add an explicit deprecation cycle if it
        # ever needs to go away.
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

        # Parse artifact_uri to get artifact_id and version
        artifact_uri = data["artifact_uri"]
        artifact_id, version = _parse_artifact_uri(artifact_uri)
        hit = data.get("hit", False)
        state = data.get("state", "ready")
        stream_id = data.get("stream_id")
        stream_url = data.get("stream_url")
        build_id = data.get("build_id")

        # Cache hit - artifact is ready
        if hit or state == "ready":
            # On cache hit with stream mode, fetch the data via stream_url
            stream_data = None
            if stream_url and mode == "stream":
                stream_data = self._fetch_stream_with_retry(stream_url)

            return Artifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=hit,
                execution="cache" if hit else "server",
                name=name,
                _stream_data=stream_data,
            )

        # Stream mode - fetch data via stream URL
        if mode == "stream" and stream_url:
            content = self._fetch_stream_with_retry(stream_url)
            return Artifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=False,
                execution="stream",
                name=name,
                _stream_data=content,
            )

        # Artifact mode - poll for completion if wait=True
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

        # Wait for build to complete
        if build_id:
            start_time = time.time()
            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Build {build_id} timed out after {timeout}s")

                # Check build status
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

        # Return artifact (may still be building)
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
            artifact_id: Artifact ID
            version: Version number

        Returns:
            Artifact object

        Raises:
            httpx.HTTPStatusError: If artifact not found (404)
        """
        # Verify artifact exists
        response = self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
        response.raise_for_status()
        return Artifact(_client=self, artifact_id=artifact_id, version=version)

    def get_artifact_by_name(self, name: str) -> Artifact:
        """Get an artifact by its name.

        Args:
            name: Artifact name

        Returns:
            Artifact object

        Raises:
            httpx.HTTPStatusError: If name not found (404)
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
            name: Name to resolve

        Returns:
            Dict with artifact_uri, version, updated_at
        """
        response = self._client.get(f"/v1/names/{name}")
        response.raise_for_status()
        return response.json()

    def set_name(self, name: str, artifact_id: str, version: int) -> dict:
        """Set or update a name pointer.

        Args:
            name: Name to set
            artifact_id: Target artifact ID
            version: Target version

        Returns:
            Dict with name_uri and artifact_uri
        """
        response = self._client.post(
            "/v1/names",
            json={"name": name, "artifact_id": artifact_id, "version": version},
        )
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Direct Artifact Upload (for local execution)
    # -------------------------------------------------------------------------

    def put(
        self,
        inputs: list[str],
        transform: TransformSpec,
        data: PutData,
        name: str | None = None,
    ) -> Artifact:
        """Directly upload and persist an artifact with provenance tracking.

        This is for clients that execute transforms locally and want to persist
        the result with full provenance tracking and deduplication.

        Supports multiple data types with automatic detection:
        - dict: JSON data (nested dicts, lists, primitives)
        - pa.Table: Arrow Table (efficient for columnar data)
        - pd.DataFrame: Pandas DataFrame (converted to Arrow)
        - pl.DataFrame: Polars DataFrame (converted to Arrow)
        - bytes: Raw Arrow IPC bytes (for pre-serialized data)

        Args:
            inputs: List of input URIs (artifact URIs or table URIs) for lineage
            transform: Transform specification (executor + params) for provenance
                The params are opaque to Strata and used only for deduplication.
            data: Data to persist (dict, Arrow Table, DataFrame, or bytes)
            name: Optional name to assign to the artifact

        Returns:
            Artifact object with access to metadata and data

        Example:
            # Persist JSON data
            artifact = client.put(
                inputs=[],
                transform={"executor": "my_step@v1", "params": {}},
                data={"result": "value", "scores": [1, 2, 3]},
            )

            # Persist an Arrow Table
            import pyarrow as pa
            table = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
            artifact = client.put(
                inputs=[],
                transform={"executor": "compute@v1", "params": {}},
                data=table,
            )

            # Persist a Pandas DataFrame
            import pandas as pd
            df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
            artifact = client.put(
                inputs=[],
                transform={"executor": "analyze@v1", "params": {}},
                data=df,
            )
        """
        # Convert data to Arrow IPC bytes
        arrow_bytes = _convert_to_arrow_ipc(data)

        # Map 'ref' to 'executor' for server compatibility
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")

        # Send as multipart form data
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

        # Parse artifact URI
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
        """Directly upload and persist a JSON artifact with provenance tracking.

        This is a convenience wrapper around put() for JSON data.
        See put() for full documentation.

        Args:
            inputs: List of input URIs for lineage
            transform: Transform specification for provenance
            data: JSON data to persist
            name: Optional name to assign

        Returns:
            Artifact object
        """
        return self.put(inputs=inputs, transform=transform, data=data, name=name)

    def get_json(self, artifact_uri: str) -> JsonArtifactData:
        """Get JSON data from an artifact.

        Fetches the artifact data and returns it as a Python dict.
        If the artifact was stored as columnar data, returns the columns.
        If it was stored as a single JSON column, returns the parsed JSON.

        Args:
            artifact_uri: Artifact URI (e.g., "strata://artifact/{id}@v={version}")

        Returns:
            The artifact data as a Python dict

        Example:
            data = client.get_json("strata://artifact/abc@v=1")
            print(data["proposal"])
        """
        table = self.fetch(artifact_uri)

        # Check if this is a single-column JSON artifact
        if table.num_columns == 1 and table.column_names[0] == "data":
            import json

            json_str = table.column("data")[0].as_py()
            return cast(JsonArtifactData, json.loads(json_str))

        # Otherwise return as columnar dict
        return cast(JsonArtifactData, table.to_pydict())

    # -------------------------------------------------------------------------
    # Artifact Lifecycle Management
    # -------------------------------------------------------------------------

    def list_artifacts(
        self,
        limit: int = 100,
        offset: int = 0,
        state: str | None = None,
        name_prefix: str | None = None,
    ) -> dict:
        """List artifacts with optional filtering.

        Args:
            limit: Maximum number of artifacts to return (default 100)
            offset: Number of artifacts to skip for pagination
            state: Filter by state ("ready", "building", "failed")
            name_prefix: Filter by artifacts with names starting with prefix

        Returns:
            Dict with 'artifacts' list and pagination info
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
        """Delete an artifact version.

        Deletes the artifact blob and metadata. Also removes any name pointers
        that reference this specific version.

        Args:
            artifact_id: Artifact ID
            version: Version number

        Returns:
            Dict with deletion status
        """
        response = self._client.delete(f"/v1/artifacts/{artifact_id}/v/{version}")
        response.raise_for_status()
        return response.json()

    def garbage_collect(self, max_age_days: float = 7.0) -> dict:
        """Garbage collect unreferenced artifacts.

        Deletes artifacts that:
        1. Have no name pointer referencing them
        2. Are older than max_age_days
        3. Are in "ready" or "failed" state

        Args:
            max_age_days: Maximum age in days for unreferenced artifacts (default 7)

        Returns:
            Dict with GC statistics (deleted_count, deleted_bytes, cutoff_timestamp)
        """
        response = self._client.post(
            "/v1/artifacts/gc",
            params={"max_age_days": max_age_days},
        )
        response.raise_for_status()
        return response.json()

    def get_artifact_usage(self) -> dict:
        """Get artifact store usage metrics.

        Returns:
            Dict with usage statistics including total_bytes, total_versions,
            unreferenced_count, etc.
        """
        response = self._client.get("/v1/artifacts/usage")
        response.raise_for_status()
        return response.json()

    # -------------------------------------------------------------------------
    # Staleness Detection
    # -------------------------------------------------------------------------

    def get_name_status(self, name: str) -> dict:
        """Get status of a named artifact including staleness info.

        Checks whether the named artifact's inputs have changed since
        it was last built. Use this to determine if a rebuild is needed.

        Args:
            name: Name to check (without strata://name/ prefix)

        Returns:
            Dict with fields:
            - name: The artifact name
            - artifact_uri: URI of the pinned artifact
            - version: Pinned version number
            - state: Artifact state
            - input_versions: Dict mapping input URI -> version when built
            - is_stale: True if any input has changed
            - stale_reason: Human-readable explanation if stale
            - changed_inputs: List of inputs that changed

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
        """Explain what materialize would do without doing it (dry run).

        Use this to check whether a computation would be a cache hit or miss,
        and if stale, which specific inputs have changed.

        Args:
            inputs: List of input URIs (table URIs or artifact URIs)
            transform: Transform specification with "ref" and "params"
                Example: {"ref": "duckdb_sql@v1", "params": {"sql": "..."}}
            name: Optional name to check staleness against

        Returns:
            Dict with fields:
            - cache_hit: True if result would be cached
            - artifact_uri: URI of existing artifact if hit
            - provenance_hash: Hash of the transform specification
            - is_stale: True if named artifact exists but needs rebuild
            - stale_reason: Explanation of why rebuild is needed
            - execution: "cache", "local", or "server"

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
        """Check if a named artifact is stale (convenience method).

        Args:
            name: Name to check

        Returns:
            True if the artifact's inputs have changed since it was built

        Raises:
            httpx.HTTPStatusError: If name not found (404)
        """
        status = self.get_name_status(name)
        return status.get("is_stale", False)


class AsyncStrataClient:
    """Async client for interacting with a Strata server.

    Use this client in async contexts like FastAPI, asyncio applications,
    or when you need to run multiple operations concurrently.

    Implements automatic retry with exponential backoff for 429 responses,
    turning server overload into self-throttling behavior.

    Example:
        async with AsyncStrataClient() as client:
            # Materialize creates/finds an artifact
            artifact = await client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )

            # Fetch downloads the data
            table = await client.fetch(artifact.uri)

        # Concurrent fetches:
        async with AsyncStrataClient() as client:
            artifact1 = await client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )
            artifact2 = await client.materialize(
                inputs=["file:///warehouse#db.users"],
                transform={"executor": "scan@v1", "params": {}},
            )
            tables = await asyncio.gather(
                client.fetch(artifact1.uri),
                client.fetch(artifact2.uri),
            )

        # Disable retries
        async with AsyncStrataClient(retry_config=RetryConfig(max_retries=0)) as client:
            ...
    """

    def __init__(
        self,
        config: StrataConfig | None = None,
        base_url: str | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the async client.

        Args:
            config: Strata configuration (uses StrataConfig.load() if None)
            base_url: Override the server URL from config
            retry_config: Retry configuration for 429 responses (uses defaults if None)
        """
        self.config = config or StrataConfig.load()
        self.base_url = base_url or self.config.server_url
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

    # -----------------------------------------------------------------------
    # Unified Materialize API
    # -----------------------------------------------------------------------

    async def fetch(
        self,
        artifact_uri: str,
        timeout: float = 300.0,
    ) -> pa.Table:
        """Fetch data from an artifact URI.

        Blocks until the artifact is ready, then downloads and returns the data.
        This is the second step after materialize() - it retrieves data from
        an already-materialized artifact.

        Args:
            artifact_uri: Artifact URI (e.g., "strata://artifact/{id}@v={version}")
            timeout: Maximum seconds to wait for artifact to be ready (default 300)

        Returns:
            Arrow Table containing the artifact data

        Example:
            # Step 1: Materialize creates/finds an artifact
            artifact = await client.materialize(
                inputs=["file:///warehouse#db.events"],
                transform={"executor": "scan@v1", "params": {}},
            )

            # Step 2: Fetch downloads the data
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
            # Check artifact status
            status_resp = await self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
            status_resp.raise_for_status()
            status = status_resp.json()

            state = status.get("state", "ready")

            if state == "ready":
                # Artifact is ready, download data
                return await self._fetch_artifact_data(artifact_id, version)
            elif state == "failed":
                error_msg = status.get("error_message", "Unknown error")
                raise RuntimeError(f"Artifact build failed: {error_msg}")
            elif state == "building":
                # Still building, check timeout
                if time.time() - start_time > timeout:
                    raise TimeoutError(
                        f"Artifact {artifact_id}@v={version} timed out after {timeout}s"
                    )
                await asyncio.sleep(0.5)
            else:
                # Unknown state, assume ready and try to fetch
                return await self._fetch_artifact_data(artifact_id, version)

    async def _fetch_artifact_data(self, artifact_id: str, version: int) -> pa.Table:
        """Fetch artifact data by ID and version."""
        response = await self._client.get(f"/v1/artifacts/{artifact_id}/v/{version}/data")
        response.raise_for_status()
        reader = ipc.open_stream(pa.BufferReader(response.content))
        return reader.read_all()

    async def _fetch_stream_with_retry(self, stream_url: str) -> bytes:
        """Fetch stream data with retry on 429 responses."""
        last_response = None
        for attempt in range(self.retry_config.max_retries + 1):
            response = await self._client.get(stream_url)

            if response.status_code != 429:
                response.raise_for_status()
                return response.content

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
        """Materialize a computed artifact.

        This is the main entry point for creating artifacts. Sends a request
        to the unified /v1/materialize endpoint.

        Args:
            inputs: List of input URIs (table URIs or artifact URIs)
            transform: Transform specification with "executor" and "params"
                Example: {"executor": "scan@v1", "params": {}}
            name: Optional name to assign to the result
            mode: "stream" (default) for immediate data, "artifact" for async build
            refresh: Force recompute even if cached
            wait: Wait for async builds to complete
            poll_interval: Seconds between build status polls
            timeout: Maximum seconds to wait for build

        Returns:
            AsyncArtifact object with access to metadata and data

        Example:
            # Identity transform (read from table)
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

        # Cache hit - artifact is ready
        if hit or state == "ready":
            stream_data = None
            if stream_url and mode == "stream":
                stream_data = await self._fetch_stream_with_retry(stream_url)

            return AsyncArtifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=hit,
                execution="cache" if hit else "server",
                name=name,
                _stream_data=stream_data,
            )

        # Stream mode - fetch data via stream URL
        if mode == "stream" and stream_url:
            content = await self._fetch_stream_with_retry(stream_url)
            return AsyncArtifact(
                _client=self,
                artifact_id=artifact_id,
                version=version,
                cache_hit=False,
                execution="stream",
                name=name,
                _stream_data=content,
            )

        # Artifact mode - poll for completion if wait=True
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

                await asyncio.sleep(0.5)

        return AsyncArtifact(
            _client=self,
            artifact_id=artifact_id,
            version=version,
            cache_hit=False,
            execution="server",
            build_id=build_id or stream_id,
            name=name,
        )

    async def _fetch_stream_with_retry(self, stream_url: str) -> bytes:
        """Fetch stream data with retry on 429 responses."""
        last_response = None
        for attempt in range(self.retry_config.max_retries + 1):
            response = await self._client.get(stream_url)

            if response.status_code != 429:
                response.raise_for_status()
                return response.content

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

    # -------------------------------------------------------------------------
    # Direct Artifact Upload (for local execution)
    # -------------------------------------------------------------------------

    async def put(
        self,
        inputs: list[str],
        transform: TransformSpec,
        data: PutData,
        name: str | None = None,
    ) -> "AsyncArtifact":
        """Directly upload and persist an artifact with provenance tracking.

        This is for clients that execute transforms locally and want to persist
        the result with full provenance tracking and deduplication.

        Supports multiple data types with automatic detection:
        - dict: JSON data (nested dicts, lists, primitives)
        - pa.Table: Arrow Table (efficient for columnar data)
        - pd.DataFrame: Pandas DataFrame (converted to Arrow)
        - pl.DataFrame: Polars DataFrame (converted to Arrow)
        - bytes: Raw Arrow IPC bytes (for pre-serialized data)

        Args:
            inputs: List of input URIs (artifact URIs or table URIs) for lineage
            transform: Transform specification (executor + params) for provenance
            data: Data to persist (dict, Arrow Table, DataFrame, or bytes)
            name: Optional name to assign to the artifact

        Returns:
            AsyncArtifact object with access to metadata and data

        Example:
            # Persist JSON data
            artifact = await client.put(
                inputs=[],
                transform={"executor": "my_step@v1", "params": {}},
                data={"result": "value", "scores": [1, 2, 3]},
            )

            # Persist an Arrow Table
            import pyarrow as pa
            table = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
            artifact = await client.put(
                inputs=[],
                transform={"executor": "compute@v1", "params": {}},
                data=table,
            )
        """
        # Convert data to Arrow IPC bytes
        arrow_bytes = _convert_to_arrow_ipc(data)

        # Map 'ref' to 'executor' for server compatibility
        server_transform = dict(transform)
        if "ref" in server_transform:
            server_transform["executor"] = server_transform.pop("ref")

        # Send as multipart form data
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

        # Parse artifact URI
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
        """Directly upload and persist a JSON artifact with provenance tracking.

        This is a convenience wrapper around put() for JSON data.
        See put() for full documentation.

        Args:
            inputs: List of input URIs for lineage
            transform: Transform specification for provenance
            data: JSON data to persist
            name: Optional name to assign

        Returns:
            AsyncArtifact object
        """
        return await self.put(inputs=inputs, transform=transform, data=data, name=name)

    async def get_json(self, artifact_uri: str) -> JsonArtifactData:
        """Get JSON data from an artifact.

        Fetches the artifact data and returns it as a Python dict.
        If the artifact was stored as columnar data, returns the columns.
        If it was stored as a single JSON column, returns the parsed JSON.

        Args:
            artifact_uri: Artifact URI (e.g., "strata://artifact/{id}@v={version}")

        Returns:
            The artifact data as a Python dict

        Example:
            data = await client.get_json("strata://artifact/abc@v=1")
            print(data["proposal"])
        """
        table = await self.fetch(artifact_uri)

        # Check if this is a single-column JSON artifact
        if table.num_columns == 1 and table.column_names[0] == "data":
            import json

            json_str = table.column("data")[0].as_py()
            return cast(JsonArtifactData, json.loads(json_str))

        # Otherwise return as columnar dict
        return cast(JsonArtifactData, table.to_pydict())


@dataclass
class AsyncArtifact:
    """An immutable, versioned artifact from Strata (async version).

    Returned by `AsyncStrataClient.materialize()`.
    Provides async access to artifact metadata and data.

    Example:
        artifact = await client.materialize(
            inputs=["file:///warehouse#db.events"],
            transform={"executor": "scan@v1", "params": {}},
        )
        print(f"Artifact: {artifact.uri}")

        # Download the data
        table = await client.fetch(artifact.uri)
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
        """Download artifact data as Arrow Table."""
        # Use cached stream data if available (from fetch with stream mode)
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

    async def lineage(self, direction: str = "upstream", max_depth: int = 10) -> dict:
        """Get artifact lineage (dependency graph)."""
        response = await self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/lineage",
            params={"direction": direction, "max_depth": max_depth},
        )
        response.raise_for_status()
        return response.json()

    async def dependents(self, max_depth: int = 10) -> dict:
        """Get artifacts that depend on this artifact."""
        response = await self._client._client.get(
            f"/v1/artifacts/{self.artifact_id}/v/{self.version}/dependents",
            params={"max_depth": max_depth},
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
