"""The artifact lineage and dependents endpoints."""

import httpx
import pyarrow as pa
import pytest

from tests.conftest import LIVE_SERVER_TIMEOUT, run_server_with_context, table_to_ipc_bytes


@pytest.fixture
def lineage_server(tmp_path):
    """A personal-mode server for lineage tests."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {"port": ctx.port, "base_url": ctx.base_url}


def create_artifact(base_url: str, inputs: list[str], executor: str = "test") -> dict:
    """Persist an artifact via PUT /v1/artifacts and return its info.

    Params are salted with the input list so distinct fixtures do not dedup.
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
    def test_lineage_single_artifact_no_inputs(self, lineage_server):
        base_url = lineage_server["base_url"]

        artifact = create_artifact(base_url, inputs=[])

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{artifact['artifact_id']}/v/{artifact['version']}/lineage",
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["artifact_id"] == artifact["artifact_id"]
        assert data["version"] == artifact["version"]
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 0
        assert data["direct_inputs"] == []
        assert data["depth"] == 0

        root_node = data["nodes"][0]
        assert root_node["type"] == "artifact"
        assert root_node["artifact_id"] == artifact["artifact_id"]

    def test_lineage_with_table_input(self, lineage_server):
        """A table input appears as a leaf node."""
        base_url = lineage_server["base_url"]

        table_uri = "file:///warehouse#db.events"
        artifact = create_artifact(base_url, inputs=[table_uri])

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{artifact['artifact_id']}/v/{artifact['version']}/lineage",
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert table_uri in data["direct_inputs"]

        node_types = {n["type"] for n in data["nodes"]}
        assert "artifact" in node_types
        assert "table" in node_types

        edge = data["edges"][0]
        assert edge["from_uri"] == table_uri
        assert artifact["artifact_id"] in edge["to_uri"]

    def test_lineage_with_artifact_input(self, lineage_server):
        base_url = lineage_server["base_url"]

        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.base"])

        dependent_artifact = create_artifact(
            base_url,
            inputs=[base_artifact["artifact_uri"]],
            executor="dependent_transform",
        )

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{dependent_artifact['artifact_id']}/v/{dependent_artifact['version']}/lineage",
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        # dependent artifact, base artifact, and table
        assert len(data["nodes"]) == 3

        # table->base and base->dependent
        assert len(data["edges"]) == 2

        assert len(data["direct_inputs"]) == 1
        assert base_artifact["artifact_uri"] in data["direct_inputs"][0]

    def test_lineage_with_named_artifact_input(self, lineage_server):
        """A named artifact input traverses to the resolved artifact's lineage."""
        base_url = lineage_server["base_url"]

        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.base"])
        set_name_resp = httpx.post(
            f"{base_url}/v1/names",
            json={
                "name": "shared-base",
                "artifact_id": base_artifact["artifact_id"],
                "version": base_artifact["version"],
            },
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert set_name_resp.status_code == 200

        dependent_artifact = create_artifact(
            base_url,
            inputs=["strata://name/shared-base"],
            executor="dependent_transform",
        )

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{dependent_artifact['artifact_id']}/v/{dependent_artifact['version']}/lineage",
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        node_uris = {node["uri"] for node in data["nodes"]}
        assert base_artifact["artifact_uri"] in node_uris
        assert "file:///warehouse#db.base" in node_uris
        assert data["direct_inputs"] == ["strata://name/shared-base"]

    def test_lineage_max_depth(self, lineage_server):
        base_url = lineage_server["base_url"]

        # Chain: table -> artifact1 -> artifact2 -> artifact3
        a1 = create_artifact(base_url, inputs=["file:///warehouse#db.source"])
        a2 = create_artifact(base_url, inputs=[a1["artifact_uri"]])
        a3 = create_artifact(base_url, inputs=[a2["artifact_uri"]])

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{a3['artifact_id']}/v/{a3['version']}/lineage",
            params={"max_depth": 1},
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        # max_depth=1 traverses one level from the root: a3 and a2, not a1 or the table.
        assert data["depth"] <= 1

    def test_lineage_not_found(self, lineage_server):
        base_url = lineage_server["base_url"]

        resp = httpx.get(
            f"{base_url}/v1/artifacts/nonexistent-id/v/1/lineage", timeout=LIVE_SERVER_TIMEOUT
        )
        assert resp.status_code == 404


class TestArtifactDependents:
    def test_dependents_no_dependents(self, lineage_server):
        base_url = lineage_server["base_url"]

        artifact = create_artifact(base_url, inputs=[])

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{artifact['artifact_id']}/v/{artifact['version']}/dependents",
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["artifact_id"] == artifact["artifact_id"]
        assert data["version"] == artifact["version"]
        assert data["dependents"] == []
        assert data["total_count"] == 0

    def test_dependents_single_dependent(self, lineage_server):
        base_url = lineage_server["base_url"]

        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.source"])

        dependent = create_artifact(
            base_url,
            inputs=[base_artifact["artifact_uri"]],
            executor="dependent_transform",
        )

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{base_artifact['artifact_id']}/v/{base_artifact['version']}/dependents",
            timeout=LIVE_SERVER_TIMEOUT,
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
        base_url = lineage_server["base_url"]

        base_artifact = create_artifact(base_url, inputs=["file:///warehouse#db.source"])

        dep1 = create_artifact(
            base_url, inputs=[base_artifact["artifact_uri"]], executor="transform1"
        )
        dep2 = create_artifact(
            base_url,
            inputs=[base_artifact["artifact_uri"], "file:///other#table"],
            executor="transform2",
        )

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{base_artifact['artifact_id']}/v/{base_artifact['version']}/dependents",
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_count"] == 2
        assert len(data["dependents"]) == 2

        dep_ids = {d["artifact_id"] for d in data["dependents"]}
        assert dep1["artifact_id"] in dep_ids
        assert dep2["artifact_id"] in dep_ids

    def test_dependents_limit(self, lineage_server):
        base_url = lineage_server["base_url"]

        base_artifact = create_artifact(base_url, inputs=[])

        for i in range(5):
            create_artifact(
                base_url,
                inputs=[base_artifact["artifact_uri"]],
                executor=f"transform_{i}",
            )

        resp = httpx.get(
            f"{base_url}/v1/artifacts/{base_artifact['artifact_id']}/v/{base_artifact['version']}/dependents",
            params={"limit": 2},
            timeout=LIVE_SERVER_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["total_count"] == 5
        assert len(data["dependents"]) == 2

    def test_dependents_not_found(self, lineage_server):
        base_url = lineage_server["base_url"]

        resp = httpx.get(
            f"{base_url}/v1/artifacts/nonexistent-id/v/1/dependents", timeout=LIVE_SERVER_TIMEOUT
        )
        assert resp.status_code == 404


class TestArtifactStoreLineageMethods:
    def test_find_dependents_method(self, tmp_path):
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path)

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

        table = pa.table({"x": [1]})
        store.write_blob("base-123", base_version, table_to_ipc_bytes(table))
        store.finalize_artifact("base-123", base_version, str(table.schema), 1, 100)

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

        dependents = store.find_dependents("base-123", 1)

        assert len(dependents) == 1
        dep_artifact, input_ver = dependents[0]
        assert dep_artifact.id == "dep-456"
        assert "base-123@v=1" in input_ver

    def test_find_dependents_uses_exact_artifact_match(self, tmp_path):
        """Artifact IDs sharing a prefix must not confuse reverse lookups."""
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

    def test_find_dependents_treats_an_underscore_in_the_id_literally(self, tmp_path):
        """``_`` is a LIKE wildcard; ``nb_a`` must not find what read ``nbXa``."""
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path)
        spec = TransformSpec(executor="dep_executor", params={}, inputs=[])
        version = store.create_artifact(
            artifact_id="dep",
            provenance_hash="dep-hash",
            transform_spec=spec,
            input_versions={"strata://artifact/nbXa@v=1": "nbXa@v=1"},
        )
        store.finalize_artifact("dep", version, "", 1, 1)

        assert store.find_dependents("nb_a", 1) == []
        assert [a.id for a, _ in store.find_dependents("nbXa", 1)] == ["dep"]

    def test_list_name_reads_lists_ready_reads_in_the_tenant(self, tmp_path):
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path)
        blob = table_to_ipc_bytes(pa.table({"x": [1]}))

        def reader(artifact_id, tenant, *, finalize=True):
            version = store.create_artifact(
                artifact_id=artifact_id,
                provenance_hash=f"hash-{artifact_id}",
                transform_spec=TransformSpec(executor="e", params={}, inputs=[]),
                input_versions={
                    "strata://name/taxi/model@champion": "m@v=1",
                    "strata://artifact/m@v=1": "m@v=1",
                },
                tenant=tenant,
            )
            if finalize:
                store.write_blob(artifact_id, version, blob)
                store.finalize_artifact(artifact_id, version, "{}", 1, len(blob))

        reader("ours", "team-a")
        reader("theirs", "team-b")
        reader("building", "team-a", finalize=False)

        assert store.list_name_reads(tenant="team-a") == [("team-a", "ours", "taxi/model@champion")]
        assert store.list_name_reads() == [
            ("team-a", "ours", "taxi/model@champion"),
            ("team-b", "theirs", "taxi/model@champion"),
        ]

    def test_get_name_for_artifact_method(self, tmp_path):
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path)

        spec = TransformSpec(executor="test", params={}, inputs=[])
        version = store.create_artifact(
            artifact_id="named-artifact",
            provenance_hash="hash123",
            transform_spec=spec,
        )

        table = pa.table({"x": [1]})
        store.write_blob("named-artifact", version, table_to_ipc_bytes(table))
        store.finalize_artifact("named-artifact", version, str(table.schema), 1, 100)

        assert store.get_name_for_artifact("named-artifact", version) is None

        store.set_name("my_artifact", "named-artifact", version)

        name = store.get_name_for_artifact("named-artifact", version)
        assert name == "my_artifact"
