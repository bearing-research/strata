"""Tests for artifact lineage and dependents endpoints.

These tests verify:
1. GET /v1/artifacts/{id}/v/{version}/lineage - Input dependency traversal
2. GET /v1/artifacts/{id}/v/{version}/dependents - Reverse dependency lookup
"""

import httpx
import pyarrow as pa
import pytest

from tests.conftest import run_server_with_context, table_to_ipc_bytes


@pytest.fixture
def lineage_server(tmp_path):
    """Start a server in personal mode for lineage testing."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {"port": ctx.port, "base_url": ctx.base_url}


def create_artifact(base_url: str, inputs: list[str], executor: str = "test") -> dict:
    """Persist an artifact via PUT /v1/artifacts, returning artifact info.

    The personal-mode build-spec → upload → finalize protocol was replaced
    by the embedded build runner; lineage fixtures persist their tables
    through the direct-put path (same provenance/lineage substrate). Each
    call salts params with the input list so distinct fixtures don't dedup.
    """
    import json as json_module
    import re

    table = pa.table({"x": [1, 2, 3]})
    metadata = {
        "inputs": inputs,
        "transform": {"executor": executor, "params": {"sql": "SELECT 1", "salt": inputs}},
    }
    files = {
        "metadata": ("metadata.json", json_module.dumps(metadata), "application/json"),
        "data": (
            "data.arrow",
            table_to_ipc_bytes(table),
            "application/vnd.apache.arrow.stream",
        ),
    }
    resp = httpx.put(f"{base_url}/v1/artifacts", files=files, timeout=30.0)
    assert resp.status_code == 200
    data = resp.json()
    match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", data["artifact_uri"])
    assert match is not None

    return {
        "artifact_id": match.group(1),
        "version": int(match.group(2)),
        "artifact_uri": data["artifact_uri"],
    }


class TestArtifactLineage:
    """Tests for the artifact lineage endpoint."""

    def test_lineage_single_artifact_no_inputs(self, lineage_server):
        """Lineage of artifact with no inputs returns just the root node."""
        base_url = lineage_server["base_url"]

        # Create artifact with no inputs
        artifact = create_artifact(base_url, inputs=[])

        # Get lineage
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{artifact['artifact_id']}/v/{artifact['version']}/lineage"
        )
        assert resp.status_code == 200
        data = resp.json()

        # Should have just the root node
        assert data["artifact_id"] == artifact["artifact_id"]
        assert data["version"] == artifact["version"]
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 0
        assert data["direct_inputs"] == []
        assert data["depth"] == 0

        # Root node should be the artifact
        root_node = data["nodes"][0]
        assert root_node["type"] == "artifact"
        assert root_node["artifact_id"] == artifact["artifact_id"]

    def test_lineage_with_table_input(self, lineage_server):
        """Lineage of artifact with table input shows table as leaf node."""
        base_url = lineage_server["base_url"]

        # Create artifact with table input
        table_uri = "file:///warehouse#db.events"
        artifact = create_artifact(base_url, inputs=[table_uri])

        # Get lineage
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{artifact['artifact_id']}/v/{artifact['version']}/lineage"
        )
        assert resp.status_code == 200
        data = resp.json()

        # Should have artifact node + table node
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert table_uri in data["direct_inputs"]

        # Check nodes
        node_types = {n["type"] for n in data["nodes"]}
        assert "artifact" in node_types
        assert "table" in node_types

        # Check edge
        edge = data["edges"][0]
        assert edge["from_uri"] == table_uri
        assert artifact["artifact_id"] in edge["to_uri"]

    def test_lineage_with_artifact_input(self, lineage_server):
        """Lineage of artifact with artifact input shows both artifacts."""
        base_url = lineage_server["base_url"]

        # Create base artifact
        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.base"])

        # Create dependent artifact that uses base_artifact as input
        dependent_artifact = create_artifact(
            base_url,
            inputs=[base_artifact["artifact_uri"]],
            executor="dependent_transform",
        )

        # Get lineage of dependent artifact
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{dependent_artifact['artifact_id']}/v/{dependent_artifact['version']}/lineage"
        )
        assert resp.status_code == 200
        data = resp.json()

        # Should have 3 nodes: dependent artifact, base artifact, and table
        assert len(data["nodes"]) == 3

        # Check that we have edges from table->base and base->dependent
        assert len(data["edges"]) == 2

        # Direct inputs should only include the base artifact
        assert len(data["direct_inputs"]) == 1
        assert base_artifact["artifact_uri"] in data["direct_inputs"][0]

    def test_lineage_with_named_artifact_input(self, lineage_server):
        """Named artifact inputs should traverse to the resolved artifact lineage."""
        base_url = lineage_server["base_url"]

        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.base"])
        set_name_resp = httpx.post(
            f"{base_url}/v1/names",
            json={
                "name": "shared-base",
                "artifact_id": base_artifact["artifact_id"],
                "version": base_artifact["version"],
            },
        )
        assert set_name_resp.status_code == 200

        dependent_artifact = create_artifact(
            base_url,
            inputs=["strata://name/shared-base"],
            executor="dependent_transform",
        )

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{dependent_artifact['artifact_id']}/v/{dependent_artifact['version']}/lineage"
        )
        assert resp.status_code == 200
        data = resp.json()

        node_uris = {node["uri"] for node in data["nodes"]}
        assert base_artifact["artifact_uri"] in node_uris
        assert "file:///warehouse#db.base" in node_uris
        assert data["direct_inputs"] == ["strata://name/shared-base"]

    def test_lineage_max_depth(self, lineage_server):
        """Lineage respects max_depth parameter."""
        base_url = lineage_server["base_url"]

        # Create chain: table -> artifact1 -> artifact2 -> artifact3
        a1 = create_artifact(base_url, inputs=["file:///warehouse#db.source"])
        a2 = create_artifact(base_url, inputs=[a1["artifact_uri"]])
        a3 = create_artifact(base_url, inputs=[a2["artifact_uri"]])

        # Get lineage with max_depth=1
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{a3['artifact_id']}/v/{a3['version']}/lineage",
            params={"max_depth": 1},
        )
        assert resp.status_code == 200
        data = resp.json()

        # Should only traverse 1 level, so should have a3 and a2 (not a1 or table)
        # max_depth=1 means we go 1 level deep from the root
        assert data["depth"] <= 1

    def test_lineage_not_found(self, lineage_server):
        """Lineage returns 404 for non-existent artifact."""
        base_url = lineage_server["base_url"]

        resp = httpx.get(f"{base_url}/v1/artifacts/nonexistent-id/v/1/lineage")
        assert resp.status_code == 404


class TestArtifactDependents:
    """Tests for the artifact dependents endpoint."""

    def test_dependents_no_dependents(self, lineage_server):
        """Artifact with no dependents returns empty list."""
        base_url = lineage_server["base_url"]

        # Create standalone artifact
        artifact = create_artifact(base_url, inputs=[])

        # Get dependents
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{artifact['artifact_id']}/v/{artifact['version']}/dependents"
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["artifact_id"] == artifact["artifact_id"]
        assert data["version"] == artifact["version"]
        assert data["dependents"] == []
        assert data["total_count"] == 0

    def test_dependents_single_dependent(self, lineage_server):
        """Find artifact that uses another artifact as input."""
        base_url = lineage_server["base_url"]

        # Create base artifact
        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.source"])

        # Create dependent artifact
        dependent = create_artifact(
            base_url,
            inputs=[base_artifact["artifact_uri"]],
            executor="dependent_transform",
        )

        # Get dependents of base artifact
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{base_artifact['artifact_id']}/v/{base_artifact['version']}/dependents"
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_count"] == 1
        assert len(data["dependents"]) == 1

        dep_info = data["dependents"][0]
        assert dep_info["artifact_id"] == dependent["artifact_id"]
        assert dep_info["version"] == dependent["version"]
        assert dep_info["transform_ref"] == "dependent_transform"

    def test_dependents_multiple_dependents(self, lineage_server):
        """Find multiple artifacts that use the same artifact as input."""
        base_url = lineage_server["base_url"]

        # Create base artifact
        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.source"])

        # Create multiple dependent artifacts
        dep1 = create_artifact(
            base_url, inputs=[base_artifact["artifact_uri"]], executor="transform1"
        )
        dep2 = create_artifact(
            base_url,
            inputs=[base_artifact["artifact_uri"], "file:///other#table"],
            executor="transform2",
        )

        # Get dependents of base artifact
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{base_artifact['artifact_id']}/v/{base_artifact['version']}/dependents"
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_count"] == 2
        assert len(data["dependents"]) == 2

        dep_ids = {d["artifact_id"] for d in data["dependents"]}
        assert dep1["artifact_id"] in dep_ids
        assert dep2["artifact_id"] in dep_ids

    def test_dependents_limit(self, lineage_server):
        """Dependents respects limit parameter."""
        base_url = lineage_server["base_url"]

        # Create base artifact
        base_artifact = create_artifact(base_url, inputs=[])

        # Create multiple dependent artifacts
        for i in range(5):
            create_artifact(
                base_url,
                inputs=[base_artifact["artifact_uri"]],
                executor=f"transform_{i}",
            )

        # Get dependents with limit=2
        resp = httpx.get(
            f"{base_url}/v1/artifacts/{base_artifact['artifact_id']}/v/{base_artifact['version']}/dependents",
            params={"limit": 2},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_count"] == 5  # Total is still 5
        assert len(data["dependents"]) == 2  # But only 2 returned

    def test_dependents_not_found(self, lineage_server):
        """Dependents returns 404 for non-existent artifact."""
        base_url = lineage_server["base_url"]

        resp = httpx.get(f"{base_url}/v1/artifacts/nonexistent-id/v/1/dependents")
        assert resp.status_code == 404


class TestArtifactStoreLineageMethods:
    """Unit tests for ArtifactStore lineage methods."""

    def test_find_dependents_method(self, tmp_path):
        """Test ArtifactStore.find_dependents directly."""
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path)

        # Create base artifact
        base_spec = TransformSpec(
            executor="base_executor",
            params={},
            inputs=["file:///data#table"],
        )
        base_version = store.create_artifact(
            artifact_id="base-123",
            provenance_hash="hash1",
            transform_spec=base_spec,
            input_versions={"file:///data#table": "snapshot-1"},
        )

        # Write blob and finalize
        table = pa.table({"x": [1]})
        store.write_blob("base-123", base_version, table_to_ipc_bytes(table))
        store.finalize_artifact("base-123", base_version, str(table.schema), 1, 100)

        # Create dependent artifact
        dep_spec = TransformSpec(
            executor="dep_executor",
            params={"sql": "SELECT * FROM input"},
            inputs=["strata://artifact/base-123@v=1"],
        )
        dep_version = store.create_artifact(
            artifact_id="dep-456",
            provenance_hash="hash2",
            transform_spec=dep_spec,
            input_versions={"strata://artifact/base-123@v=1": "base-123@v=1"},
        )
        store.write_blob("dep-456", dep_version, table_to_ipc_bytes(table))
        store.finalize_artifact("dep-456", dep_version, str(table.schema), 1, 100)

        # Find dependents
        dependents = store.find_dependents("base-123", 1)

        assert len(dependents) == 1
        dep_artifact, input_ver = dependents[0]
        assert dep_artifact.id == "dep-456"
        assert "base-123@v=1" in input_ver

    def test_find_dependents_uses_exact_artifact_match(self, tmp_path):
        """Prefix-sharing artifact IDs should not confuse reverse dependency lookups."""
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path)
        table = pa.table({"x": [1]})
        blob = table_to_ipc_bytes(table)

        for artifact_id in ("base-1", "base-10"):
            version = store.create_artifact(
                artifact_id=artifact_id,
                provenance_hash=f"hash-{artifact_id}",
                transform_spec=TransformSpec(executor="base_executor", params={}, inputs=[]),
                input_versions={},
            )
            store.write_blob(artifact_id, version, blob)
            store.finalize_artifact(artifact_id, version, str(table.schema), 1, len(blob))

        dep_version = store.create_artifact(
            artifact_id="dep-exact",
            provenance_hash="dep-hash",
            transform_spec=TransformSpec(
                executor="dep_executor",
                params={},
                inputs=[
                    "strata://artifact/base-10@v=1",
                    "strata://artifact/base-1@v=1",
                ],
            ),
            input_versions={
                "strata://artifact/base-10@v=1": "base-10@v=1",
                "strata://artifact/base-1@v=1": "base-1@v=1",
            },
        )
        store.write_blob("dep-exact", dep_version, blob)
        store.finalize_artifact("dep-exact", dep_version, str(table.schema), 1, len(blob))

        dependents = store.find_dependents("base-1", 1)

        assert len(dependents) == 1
        dep_artifact, input_ver = dependents[0]
        assert dep_artifact.id == "dep-exact"
        assert input_ver == "base-1@v=1"

    def test_get_name_for_artifact_method(self, tmp_path):
        """Test ArtifactStore.get_name_for_artifact directly."""
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path)

        # Create artifact
        spec = TransformSpec(executor="test", params={}, inputs=[])
        version = store.create_artifact(
            artifact_id="named-artifact",
            provenance_hash="hash123",
            transform_spec=spec,
        )

        # Write blob and finalize
        table = pa.table({"x": [1]})
        store.write_blob("named-artifact", version, table_to_ipc_bytes(table))
        store.finalize_artifact("named-artifact", version, str(table.schema), 1, 100)

        # No name set yet
        assert store.get_name_for_artifact("named-artifact", version) is None

        # Set name
        store.set_name("my_artifact", "named-artifact", version)

        # Now should find name
        name = store.get_name_for_artifact("named-artifact", version)
        assert name == "my_artifact"
