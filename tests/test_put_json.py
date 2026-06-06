"""Tests for the PUT /v1/artifacts (put_json) endpoint.

These tests verify:
1. Direct JSON upload with provenance tracking
2. Cache hit on duplicate request
3. Lineage tracking via input_versions
4. Name assignment
5. JSON retrieval via get_json
"""

import pytest
import requests

from strata.client import StrataClient


@pytest.fixture
def server_with_artifacts(tmp_path):
    """Start a server in personal mode for artifact testing.

    Uses the shared run_server_with_context helper: health-polled startup
    (a bare ``time.sleep(1)`` raced slow CI runners into connection-refused
    flakes) and graceful shutdown (orphaned daemon servers each kept a live
    build-runner poll loop running for the rest of the suite).
    """
    from tests.conftest import run_server_with_context

    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {"base_url": ctx.base_url, "config": ctx.config}


class TestPutJson:
    """Tests for PUT /v1/artifacts endpoint."""

    def test_put_json_basic(self, server_with_artifacts):
        """Test basic JSON upload."""
        base_url = server_with_artifacts["base_url"]

        response = requests.put(
            f"{base_url}/v1/artifacts",
            json={
                "inputs": [],
                "transform": {
                    "executor": "test@v1",
                    "params": {"step": "PROPOSE"},
                },
                "data": {"proposal": "test proposal", "claims": ["claim1", "claim2"]},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["artifact_uri"].startswith("strata://artifact/")
        assert data["hit"] is False
        assert data["byte_size"] > 0

    def test_put_json_cache_hit(self, server_with_artifacts):
        """Test that duplicate request returns cache hit."""
        base_url = server_with_artifacts["base_url"]

        request_body = {
            "inputs": [],
            "transform": {
                "executor": "delibera@v1",
                "params": {
                    "operation": "WORK",
                    "step_id": "PROPOSE",
                    "code_hash": "sha256:abc123",
                },
            },
            "data": {"result": "computation output"},
        }

        # First request - should not be a hit
        response1 = requests.put(f"{base_url}/v1/artifacts", json=request_body)
        assert response1.status_code == 200
        data1 = response1.json()
        assert data1["hit"] is False

        # Second request - should be a cache hit
        response2 = requests.put(f"{base_url}/v1/artifacts", json=request_body)
        assert response2.status_code == 200
        data2 = response2.json()
        assert data2["hit"] is True
        assert data2["artifact_uri"] == data1["artifact_uri"]

    def test_put_json_with_name(self, server_with_artifacts):
        """Test JSON upload with name assignment."""
        base_url = server_with_artifacts["base_url"]

        response = requests.put(
            f"{base_url}/v1/artifacts",
            json={
                "inputs": [],
                "transform": {"executor": "test@v1", "params": {}},
                "data": {"value": 42},
                "name": "my_artifact",
            },
        )

        assert response.status_code == 200
        data = response.json()
        # Name should be set (even if resolve fails, the response should include it)
        assert data["name_uri"] == "strata://name/my_artifact"
        assert data["artifact_uri"].startswith("strata://artifact/")

    def test_put_json_with_inputs(self, server_with_artifacts):
        """Test JSON upload with input references for lineage."""
        base_url = server_with_artifacts["base_url"]

        # First create a parent artifact
        parent_response = requests.put(
            f"{base_url}/v1/artifacts",
            json={
                "inputs": [],
                "transform": {"executor": "parent@v1", "params": {}},
                "data": {"parent_data": "value"},
            },
        )
        assert parent_response.status_code == 200
        parent_uri = parent_response.json()["artifact_uri"]

        # Create child artifact with parent as input
        child_response = requests.put(
            f"{base_url}/v1/artifacts",
            json={
                "inputs": [parent_uri],
                "transform": {"executor": "child@v1", "params": {}},
                "data": {"child_data": "derived"},
            },
        )
        assert child_response.status_code == 200
        child_uri = child_response.json()["artifact_uri"]

        # Verify both artifacts are accessible
        assert parent_uri.startswith("strata://artifact/")
        assert child_uri.startswith("strata://artifact/")
        assert parent_uri != child_uri  # Different artifacts


class TestPutJsonClient:
    """Tests for client.put_json() method."""

    def test_client_put_json(self, server_with_artifacts):
        """Test client.put_json() method."""
        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            artifact = client.put_json(
                inputs=[],
                transform={
                    "executor": "delibera@v1",
                    "params": {
                        "operation": "WORK",
                        "step_id": "PLAN",
                        "role": "Planner",
                        "code_hash": "sha256:xyz789",
                    },
                },
                data={"branches": ["branch1", "branch2"]},
            )

            assert artifact.uri.startswith("strata://artifact/")
            assert artifact.cache_hit is False
        finally:
            client.close()

    def test_client_put_json_cache_hit(self, server_with_artifacts):
        """Test client.put_json() returns cache_hit=True on duplicate."""
        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            transform = {
                "executor": "delibera@v1",
                "params": {"step_id": "VALIDATE", "code_hash": "sha256:val123"},
            }
            data = {"validation_result": "passed"}

            # First call
            artifact1 = client.put_json(inputs=[], transform=transform, data=data)
            assert artifact1.cache_hit is False

            # Second call - should be cache hit
            artifact2 = client.put_json(inputs=[], transform=transform, data=data)
            assert artifact2.cache_hit is True
            assert artifact2.uri == artifact1.uri
        finally:
            client.close()

    def test_client_get_json(self, server_with_artifacts):
        """Test client.get_json() retrieves data correctly."""
        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            original_data = {
                "proposal": "Test proposal",
                "claims": [{"id": 1, "text": "claim1"}, {"id": 2, "text": "claim2"}],
                "metadata": {"version": 1},
            }

            artifact = client.put_json(
                inputs=[],
                transform={"executor": "test@v1", "params": {}},
                data=original_data,
            )

            # Retrieve the data
            retrieved_data = client.get_json(artifact.uri)
            assert retrieved_data == original_data
        finally:
            client.close()

    def test_client_put_json_with_name(self, server_with_artifacts):
        """Test client.put_json() with name assignment."""
        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            artifact = client.put_json(
                inputs=[],
                transform={"executor": "test@v1", "params": {}},
                data={"value": 123},
                name="test_named_artifact",
            )

            assert artifact.name == "test_named_artifact"
        finally:
            client.close()


class TestDeliberaIntegration:
    """Tests simulating Delibera-like usage patterns."""

    def test_delibera_step_persistence(self, server_with_artifacts):
        """Test Delibera-style step artifact persistence."""
        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            # 1. Persist protocol spec (one-time)
            protocol = client.put_json(
                inputs=[],
                transform={"executor": "protocol@v1", "params": {}},
                data={"version": "tree_v1", "steps": ["PLAN", "PROPOSE", "VALIDATE"]},
                name="protocol_tree_v1",
            )

            # 2. Persist constraints (per-run)
            constraints = client.put_json(
                inputs=[],
                transform={"executor": "constraints@v1", "params": {}},
                data={"risk_tolerance": 0.1, "scope": "narrow"},
                name="constraints_run_42",
            )

            # 3. Planner step (depends on protocol + constraints)
            planner_result = client.put_json(
                inputs=[protocol.uri, constraints.uri],
                transform={
                    "executor": "delibera@v1",
                    "params": {
                        "operation": "WORK",
                        "step_id": "PLAN",
                        "role": "Planner",
                        "code_hash": "sha256:planner_v1",
                    },
                },
                data={"branches": ["option_a", "option_b", "option_c"]},
            )

            # 4. Proposer step (depends on planner output)
            proposer_result = client.put_json(
                inputs=[planner_result.uri, constraints.uri],
                transform={
                    "executor": "delibera@v1",
                    "params": {
                        "operation": "WORK",
                        "step_id": "PROPOSE",
                        "role": "Proposer",
                        "code_hash": "sha256:proposer_v2",
                        "temperature": 0.1,
                    },
                },
                data={
                    "proposal": "Recommended approach...",
                    "claims": [
                        {"id": 1, "text": "Claim A", "evidence": []},
                        {"id": 2, "text": "Claim B", "evidence": []},
                    ],
                },
            )

            # Verify all artifacts exist and are readable
            assert client.get_json(protocol.uri)["version"] == "tree_v1"
            assert client.get_json(constraints.uri)["risk_tolerance"] == 0.1
            assert len(client.get_json(planner_result.uri)["branches"]) == 3
            assert "proposal" in client.get_json(proposer_result.uri)

            # Verify cache hits on replay
            replay_planner = client.put_json(
                inputs=[protocol.uri, constraints.uri],
                transform={
                    "executor": "delibera@v1",
                    "params": {
                        "operation": "WORK",
                        "step_id": "PLAN",
                        "role": "Planner",
                        "code_hash": "sha256:planner_v1",
                    },
                },
                data={"branches": ["option_a", "option_b", "option_c"]},
            )
            assert replay_planner.cache_hit is True
            assert replay_planner.uri == planner_result.uri

        finally:
            client.close()


class TestPutMultipleTypes:
    """Tests for client.put() with different data types."""

    def test_put_arrow_table(self, server_with_artifacts):
        """Test put() with Arrow Table."""
        import pyarrow as pa

        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            # Create Arrow table
            table = pa.table(
                {
                    "id": [1, 2, 3],
                    "value": [10.0, 20.0, 30.0],
                    "name": ["a", "b", "c"],
                }
            )

            artifact = client.put(
                inputs=[],
                transform={"executor": "compute@v1", "params": {}},
                data=table,
            )

            assert artifact.uri.startswith("strata://artifact/")
            assert artifact.cache_hit is False

            # Retrieve and verify
            retrieved = client.fetch(artifact.uri)
            assert retrieved.num_rows == 3
            assert retrieved.column_names == ["id", "value", "name"]
        finally:
            client.close()

    def test_put_pandas_dataframe(self, server_with_artifacts):
        """Test put() with Pandas DataFrame."""
        import pandas as pd

        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            # Create Pandas DataFrame
            df = pd.DataFrame(
                {
                    "x": [1, 2, 3, 4],
                    "y": [10, 20, 30, 40],
                    "label": ["cat", "dog", "bird", "fish"],
                }
            )

            artifact = client.put(
                inputs=[],
                transform={"executor": "ml@v1", "params": {"model": "v1"}},
                data=df,
            )

            assert artifact.uri.startswith("strata://artifact/")
            assert artifact.cache_hit is False

            # Retrieve and verify
            retrieved = client.fetch(artifact.uri)
            assert retrieved.num_rows == 4
            # Pandas may add __index_level_0__ column
            assert "x" in retrieved.column_names
            assert "y" in retrieved.column_names
        finally:
            client.close()

    def test_put_dict_columnar(self, server_with_artifacts):
        """Test put() with dict that has columnar data."""
        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            # Columnar dict (all values are lists of same length)
            data = {
                "id": [1, 2, 3],
                "score": [0.9, 0.8, 0.7],
            }

            artifact = client.put(
                inputs=[],
                transform={"executor": "score@v1", "params": {}},
                data=data,
            )

            # Retrieve - should be columnar
            retrieved = client.fetch(artifact.uri)
            assert retrieved.num_rows == 3
            assert set(retrieved.column_names) == {"id", "score"}
        finally:
            client.close()

    def test_put_dict_nested_json(self, server_with_artifacts):
        """Test put() with nested dict (stored as JSON)."""
        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            # Nested dict (not columnar)
            data = {
                "config": {"learning_rate": 0.01, "epochs": 100},
                "results": {"accuracy": 0.95, "loss": 0.05},
            }

            artifact = client.put(
                inputs=[],
                transform={"executor": "train@v1", "params": {}},
                data=data,
            )

            # Retrieve as JSON
            retrieved = client.get_json(artifact.uri)
            assert retrieved["config"]["learning_rate"] == 0.01
            assert retrieved["results"]["accuracy"] == 0.95
        finally:
            client.close()

    def test_put_cache_hit_with_arrow(self, server_with_artifacts):
        """Test cache hit detection with Arrow Table."""
        import pyarrow as pa

        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            table = pa.table({"x": [1, 2], "y": [3, 4]})
            transform = {"executor": "dedup_test@v1", "params": {"version": 1}}

            # First call
            artifact1 = client.put(inputs=[], transform=transform, data=table)
            assert artifact1.cache_hit is False

            # Second call with same data - should be cache hit
            artifact2 = client.put(inputs=[], transform=transform, data=table)
            assert artifact2.cache_hit is True
            assert artifact2.uri == artifact1.uri
        finally:
            client.close()

    def test_put_with_lineage(self, server_with_artifacts):
        """Test put() with lineage tracking across data types."""
        import pyarrow as pa

        base_url = server_with_artifacts["base_url"]
        client = StrataClient(base_url=base_url)

        try:
            # First artifact: JSON config
            config = client.put(
                inputs=[],
                transform={"executor": "config@v1", "params": {}},
                data={"setting": "value"},
            )

            # Second artifact: Arrow table that depends on config
            table = pa.table({"result": [1, 2, 3]})
            result = client.put(
                inputs=[config.uri],  # Lineage!
                transform={"executor": "process@v1", "params": {}},
                data=table,
            )

            assert result.uri.startswith("strata://artifact/")
            # Different artifacts
            assert result.uri != config.uri
        finally:
            client.close()


class TestAsyncPut:
    """Tests for AsyncStrataClient.put() and related methods."""

    @pytest.fixture
    async def async_client(self, server_with_artifacts):
        """Create an async client for testing."""
        from strata.client import AsyncStrataClient

        base_url = server_with_artifacts["base_url"]
        client = AsyncStrataClient(base_url=base_url)
        yield client
        await client.close()

    @pytest.mark.asyncio
    async def test_async_put_json(self, async_client):
        """Test async client put_json() method."""
        artifact = await async_client.put_json(
            inputs=[],
            transform={
                "executor": "async_test@v1",
                "params": {"step": "test"},
            },
            data={"result": "async value", "items": [1, 2, 3]},
        )

        assert artifact.uri.startswith("strata://artifact/")
        assert artifact.cache_hit is False

    @pytest.mark.asyncio
    async def test_async_put_json_cache_hit(self, async_client):
        """Test async client put_json() returns cache hit on duplicate."""
        transform = {
            "executor": "async_cache@v1",
            "params": {"version": 1},
        }
        data = {"value": 42}

        # First call
        artifact1 = await async_client.put_json(inputs=[], transform=transform, data=data)
        assert artifact1.cache_hit is False

        # Second call - should be cache hit
        artifact2 = await async_client.put_json(inputs=[], transform=transform, data=data)
        assert artifact2.cache_hit is True
        assert artifact2.uri == artifact1.uri

    @pytest.mark.asyncio
    async def test_async_get_json(self, async_client):
        """Test async client get_json() retrieves data correctly."""
        original_data = {
            "nested": {"key": "value"},
            "list": [1, 2, 3],
        }

        artifact = await async_client.put_json(
            inputs=[],
            transform={"executor": "async_json@v1", "params": {}},
            data=original_data,
        )

        # Retrieve the data
        retrieved = await async_client.get_json(artifact.uri)
        assert retrieved == original_data

    @pytest.mark.asyncio
    async def test_async_put_arrow_table(self, async_client):
        """Test async client put() with Arrow Table."""
        import pyarrow as pa

        table = pa.table(
            {
                "id": [1, 2, 3],
                "value": [10.0, 20.0, 30.0],
            }
        )

        artifact = await async_client.put(
            inputs=[],
            transform={"executor": "async_arrow@v1", "params": {}},
            data=table,
        )

        assert artifact.uri.startswith("strata://artifact/")
        assert artifact.cache_hit is False

        # Retrieve and verify
        retrieved = await async_client.fetch(artifact.uri)
        assert retrieved.num_rows == 3
        assert set(retrieved.column_names) == {"id", "value"}

    @pytest.mark.asyncio
    async def test_async_put_pandas_dataframe(self, async_client):
        """Test async client put() with Pandas DataFrame."""
        import pandas as pd

        df = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [3, 4],
            }
        )

        artifact = await async_client.put(
            inputs=[],
            transform={"executor": "async_pandas@v1", "params": {}},
            data=df,
        )

        assert artifact.uri.startswith("strata://artifact/")

        # Retrieve and verify
        retrieved = await async_client.fetch(artifact.uri)
        assert retrieved.num_rows == 2
        assert "x" in retrieved.column_names

    @pytest.mark.asyncio
    async def test_async_put_with_name(self, async_client):
        """Test async client put() with name assignment."""
        artifact = await async_client.put(
            inputs=[],
            transform={"executor": "async_named@v1", "params": {}},
            data={"named": True},
            name="async_test_artifact",
        )

        assert artifact.name == "async_test_artifact"

    @pytest.mark.asyncio
    async def test_async_put_with_lineage(self, async_client):
        """Test async client put() with lineage tracking."""
        import pyarrow as pa

        # Create parent artifact
        parent = await async_client.put(
            inputs=[],
            transform={"executor": "async_parent@v1", "params": {}},
            data={"parent": True},
        )

        # Create child with lineage
        child = await async_client.put(
            inputs=[parent.uri],
            transform={"executor": "async_child@v1", "params": {}},
            data=pa.table({"derived": [1, 2, 3]}),
        )

        assert child.uri != parent.uri
