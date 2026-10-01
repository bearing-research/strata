"""End-to-end artifact workflows against a real server and SQLite store.

Covers chains, lineage, dependents, staleness, name pointers and explain.
"""

import httpx
import pyarrow as pa
import pytest

from tests.conftest import (
    ipc_bytes_to_table,
    run_server_with_context,
    table_to_ipc_bytes,
)


@pytest.fixture
def e2e_server(tmp_path):
    """A running server for E2E tests."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    warehouse_path = tmp_path / "warehouse"

    cache_dir.mkdir()
    artifact_dir.mkdir()
    warehouse_path.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {
            "config": ctx.config,
            "port": ctx.port,
            "base_url": ctx.base_url,
            "warehouse_path": str(warehouse_path),
        }


class ArtifactClient:
    """Helper client for artifact operations."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=30.0)

    def close(self):
        self.client.close()

    def materialize(
        self,
        inputs: list[str],
        executor: str = "local://duckdb_sql@v1",
        params: dict | None = None,
        name: str | None = None,
    ) -> dict:
        body = {
            "inputs": inputs,
            "transform": {
                "executor": executor,
                "params": params or {"sql": "SELECT 1"},
            },
        }
        if name:
            body["name"] = name

        resp = self.client.post("/v1/artifacts/materialize", json=body)
        resp.raise_for_status()
        return resp.json()

    def upload_and_finalize(
        self,
        artifact_id: str,
        version: int,
        table: pa.Table,
        name: str | None = None,
    ) -> dict:
        self.client.post(
            f"/v1/artifacts/upload/{artifact_id}/v/{version}",
            content=table_to_ipc_bytes(table),
            headers={"Content-Type": "application/vnd.apache.arrow.stream"},
        )

        body = {
            "artifact_id": artifact_id,
            "version": version,
            "arrow_schema": str(table.schema),
            "row_count": table.num_rows,
        }
        if name:
            body["name"] = name

        resp = self.client.post("/v1/artifacts/finalize", json=body)
        resp.raise_for_status()
        return resp.json()

    def create_artifact(
        self,
        inputs: list[str],
        result_table: pa.Table,
        executor: str = "local://duckdb_sql@v1",
        params: dict | None = None,
        name: str | None = None,
    ) -> str:
        """Persist a specific result table via PUT /v1/artifacts.

        Shares the provenance, dedup and lineage substrate with the build runner.
        """
        import json as json_module
        import time as time_module

        if name:
            # Named artifacts go through the embedded runner so the name is set by the
            # flow users hit.
            mat = self.materialize(inputs, executor, params, name)
            if mat["hit"]:
                return mat["artifact_uri"]
            artifact_uri = mat["artifact_uri"]
            artifact_id, version = artifact_uri.split("artifact/")[1].split("@v=")
            deadline = time_module.time() + 30.0
            while time_module.time() < deadline:
                resp = self.client.get(f"/v1/artifacts/{artifact_id}/v/{version}")
                state = resp.json().get("state")
                if state == "ready":
                    return artifact_uri
                if state == "failed":
                    raise AssertionError(f"build failed for {artifact_uri}")
                time_module.sleep(0.2)
            raise AssertionError(f"build did not finish for {artifact_uri}")

        metadata: dict = {
            "inputs": inputs,
            "transform": {"executor": executor, "params": params or {"sql": "SELECT 1"}},
        }
        files = {
            "metadata": ("metadata.json", json_module.dumps(metadata), "application/json"),
            "data": (
                "data.arrow",
                table_to_ipc_bytes(result_table),
                "application/vnd.apache.arrow.stream",
            ),
        }
        resp = self.client.put("/v1/artifacts", files=files)
        resp.raise_for_status()
        return resp.json()["artifact_uri"]

    def get_lineage(self, artifact_id: str, version: int, max_depth: int = 10) -> dict:
        resp = self.client.get(
            f"/v1/artifacts/{artifact_id}/v/{version}/lineage",
            params={"max_depth": max_depth},
        )
        resp.raise_for_status()
        return resp.json()

    def get_dependents(self, artifact_id: str, version: int, limit: int = 100) -> dict:
        resp = self.client.get(
            f"/v1/artifacts/{artifact_id}/v/{version}/dependents",
            params={"limit": limit},
        )
        resp.raise_for_status()
        return resp.json()

    def get_name_status(self, name: str) -> dict:
        """Name status, including staleness."""
        resp = self.client.get(f"/v1/artifacts/names/{name}/status")
        resp.raise_for_status()
        return resp.json()

    def fetch_artifact(self, artifact_uri: str) -> pa.Table:
        """Fetch artifact data as an Arrow table."""
        # strata://artifact/{id}@v={version}
        import re

        match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", artifact_uri)
        if not match:
            raise ValueError(f"Invalid artifact URI: {artifact_uri}")

        artifact_id = match.group(1)
        version = int(match.group(2))

        resp = self.client.get(f"/v1/artifacts/{artifact_id}/v/{version}/data")
        resp.raise_for_status()
        return ipc_bytes_to_table(resp.content)


class TestArtifactPipeline:
    def test_create_single_artifact(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            result_table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
            artifact_uri = client.create_artifact(
                inputs=[],
                result_table=result_table,
                params={
                    "sql": "SELECT 1 as x, 'a' as y UNION ALL SELECT 2, 'b' UNION ALL SELECT 3, 'c'"
                },
            )

            assert artifact_uri.startswith("strata://artifact/")
            assert "@v=" in artifact_uri
        finally:
            client.close()

    def test_create_artifact_with_name(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            result_table = pa.table({"value": [42]})
            artifact_uri = client.create_artifact(
                inputs=[],
                result_table=result_table,
                params={"sql": "SELECT 42 as value"},
                name="my_artifact",
            )

            status = client.get_name_status("my_artifact")
            assert status["name"] == "my_artifact"
            assert status["artifact_uri"] == artifact_uri
            assert status["is_stale"] is False
        finally:
            client.close()

    def test_cache_hit_on_duplicate(self, e2e_server):
        """Same inputs + transform return a cache hit."""
        client = ArtifactClient(e2e_server["base_url"])

        try:
            result_table = pa.table({"x": [1]})

            uri1 = client.create_artifact(
                inputs=["table://source"],
                result_table=result_table,
                params={"sql": "SELECT 1 as x"},
            )

            resp = client.materialize(
                inputs=["table://source"],
                params={"sql": "SELECT 1 as x"},
            )

            assert resp["hit"] is True
            assert resp["artifact_uri"] == uri1
        finally:
            client.close()


class TestChainedArtifacts:
    """Artifacts used as inputs to other artifacts."""

    def test_two_level_chain(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            base_table = pa.table({"id": [1, 2, 3], "value": [10, 20, 30]})
            # Base persists via PUT (its table input is synthetic); the derived level
            # below really executes against it.
            base_uri = client.create_artifact(
                inputs=["file:///warehouse#db.source"],
                result_table=base_table,
                params={"sql": "SELECT * FROM input0"},
            )

            derived_table = pa.table({"id": [1, 2, 3], "doubled": [20, 40, 60]})
            derived_uri = client.create_artifact(
                inputs=[base_uri],
                result_table=derived_table,
                params={"sql": "SELECT id, value * 2 as doubled FROM input0"},
                name="derived_artifact",
            )

            assert derived_uri != base_uri
            assert derived_uri.startswith("strata://artifact/")

            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", derived_uri)
            assert match is not None
            derived_id, derived_ver = match.group(1), int(match.group(2))

            lineage = client.get_lineage(derived_id, derived_ver)

            # derived, base, table
            assert len(lineage["nodes"]) == 3

            assert len(lineage["direct_inputs"]) == 1
            assert base_uri in lineage["direct_inputs"][0]
        finally:
            client.close()

    def test_three_level_chain(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            raw_table = pa.table({"x": [1, 2, 3]})
            raw_uri = client.create_artifact(
                inputs=["file:///data#raw"],
                result_table=raw_table,
                params={"sql": "SELECT * FROM input0"},
            )

            clean_table = pa.table({"x": [1, 2, 3], "is_valid": [True, True, True]})
            clean_uri = client.create_artifact(
                inputs=[raw_uri],
                result_table=clean_table,
                params={"sql": "SELECT *, true as is_valid FROM input0"},
            )

            agg_table = pa.table({"total": [6]})
            agg_uri = client.create_artifact(
                inputs=[clean_uri],
                result_table=agg_table,
                params={"sql": "SELECT SUM(x) as total FROM input0"},
            )

            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", agg_uri)
            assert match is not None
            agg_id, agg_ver = match.group(1), int(match.group(2))

            lineage = client.get_lineage(agg_id, agg_ver)

            # agg, clean, raw, table
            assert len(lineage["nodes"]) == 4
            # Depth is the max BFS level: agg=0, clean=1, raw=2, table=2 (sibling).
            assert lineage["depth"] == 2
        finally:
            client.close()


class TestLineageTraversal:
    def test_lineage_with_multiple_inputs(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            users_table = pa.table({"user_id": [1, 2], "name": ["Alice", "Bob"]})
            users_uri = client.create_artifact(
                inputs=["file:///db#users"],
                result_table=users_table,
                params={"sql": "SELECT * FROM input0"},
            )

            orders_table = pa.table({"order_id": [1, 2], "user_id": [1, 2], "amount": [100, 200]})
            orders_uri = client.create_artifact(
                inputs=["file:///db#orders"],
                result_table=orders_table,
                params={"sql": "SELECT * FROM input0"},
            )

            joined_table = pa.table(
                {
                    "name": ["Alice", "Bob"],
                    "amount": [100, 200],
                }
            )
            joined_uri = client.create_artifact(
                inputs=[users_uri, orders_uri],
                result_table=joined_table,
                params={
                    "sql": "SELECT u.name, o.amount FROM input0 u "
                    "JOIN input1 o ON u.user_id = o.user_id"
                },
            )

            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", joined_uri)
            assert match is not None
            joined_id, joined_ver = match.group(1), int(match.group(2))

            lineage = client.get_lineage(joined_id, joined_ver)

            # joined, users, orders, users_table, orders_table
            assert len(lineage["nodes"]) == 5

            assert len(lineage["direct_inputs"]) == 2

            assert len(lineage["edges"]) >= 4
        finally:
            client.close()

    def test_lineage_max_depth_limiting(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            prev_uri = "file:///source#table"
            result_table = pa.table({"x": [1]})

            for i in range(4):
                prev_uri = client.create_artifact(
                    inputs=[prev_uri],
                    result_table=result_table,
                    params={"sql": f"SELECT x FROM input0 -- level {i}"},
                )

            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", prev_uri)
            assert match is not None
            art_id, art_ver = match.group(1), int(match.group(2))

            full_lineage = client.get_lineage(art_id, art_ver, max_depth=10)
            assert len(full_lineage["nodes"]) == 5  # 4 artifacts + 1 table

            limited_lineage = client.get_lineage(art_id, art_ver, max_depth=2)
            assert len(limited_lineage["nodes"]) <= len(full_lineage["nodes"])
        finally:
            client.close()


class TestDependentsTracking:
    """Reverse dependency tracking."""

    def test_find_single_dependent(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            base_table = pa.table({"x": [1, 2, 3]})
            base_uri = client.create_artifact(
                inputs=[],
                result_table=base_table,
                params={"sql": "SELECT 1 as x UNION ALL SELECT 2 UNION ALL SELECT 3"},
            )

            dep_table = pa.table({"x_doubled": [2, 4, 6]})
            dep_uri = client.create_artifact(
                inputs=[base_uri],
                result_table=dep_table,
                params={"sql": "SELECT x * 2 as x_doubled FROM input0"},
            )

            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", base_uri)
            assert match is not None
            base_id, base_ver = match.group(1), int(match.group(2))

            dependents = client.get_dependents(base_id, base_ver)

            assert dependents["total_count"] == 1
            assert len(dependents["dependents"]) == 1
            assert dep_uri in dependents["dependents"][0]["artifact_uri"]
        finally:
            client.close()

    def test_find_multiple_dependents(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            base_table = pa.table({"value": [100]})
            base_uri = client.create_artifact(
                inputs=[],
                result_table=base_table,
                params={"sql": "SELECT 100 as value"},
            )

            dep_uris = []
            for i in range(3):
                dep_table = pa.table({"result": [100 * (i + 1)]})
                dep_uri = client.create_artifact(
                    inputs=[base_uri],
                    result_table=dep_table,
                    params={"sql": f"SELECT value * {i + 1} as result FROM input0"},
                )
                dep_uris.append(dep_uri)

            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", base_uri)
            assert match is not None
            base_id, base_ver = match.group(1), int(match.group(2))

            dependents = client.get_dependents(base_id, base_ver)

            assert dependents["total_count"] == 3
            assert len(dependents["dependents"]) == 3

            dependent_uris = {d["artifact_uri"] for d in dependents["dependents"]}
            for dep_uri in dep_uris:
                assert dep_uri in dependent_uris
        finally:
            client.close()

    def test_no_dependents(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            table = pa.table({"x": [1]})
            artifact_uri = client.create_artifact(
                inputs=[],
                result_table=table,
                params={"sql": "SELECT 1 as x"},
            )

            import re

            match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", artifact_uri)
            assert match is not None
            art_id, art_ver = match.group(1), int(match.group(2))

            dependents = client.get_dependents(art_id, art_ver)

            assert dependents["total_count"] == 0
            assert dependents["dependents"] == []
        finally:
            client.close()


class TestStalenessDetection:
    def test_fresh_artifact_not_stale(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            table = pa.table({"x": [1]})
            client.create_artifact(
                inputs=[],
                result_table=table,
                params={"sql": "SELECT 1 as x"},
                name="fresh_artifact",
            )

            status = client.get_name_status("fresh_artifact")
            assert status["is_stale"] is False
            assert status["stale_reason"] is None
            assert status["changed_inputs"] is None
        finally:
            client.close()


class TestExplainMaterialize:
    """The explain-materialize dry-run endpoint."""

    def test_explain_cache_miss(self, e2e_server):
        client = ArtifactClient(e2e_server["base_url"])

        try:
            # Never run before
            resp = client.client.post(
                "/v1/artifacts/explain-materialize",
                json={
                    "inputs": ["file:///new#table"],
                    "transform": {
                        "executor": "local://duckdb_sql@v1",
                        "params": {"sql": "SELECT * FROM input0"},
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            assert data["would_hit"] is False
            assert data["would_build"] is True
            assert data["artifact_uri"] is None
        finally:
            client.close()

    def test_explain_cache_hit(self, e2e_server):
        """An artifact URI as input gives a cache hit."""
        client = ArtifactClient(e2e_server["base_url"])

        try:
            # No inputs, so the hash is stable.
            base_table = pa.table({"y": [10, 20]})
            base_uri = client.create_artifact(
                inputs=[],
                result_table=base_table,
                params={"sql": "SELECT 10 as y UNION SELECT 20 as y"},
            )

            derived_table = pa.table({"x": [1]})
            derived_uri = client.create_artifact(
                inputs=[base_uri],
                result_table=derived_table,
                params={"sql": "SELECT 1 as x FROM input0 LIMIT 1"},
            )

            # Same computation, so it should hit.
            resp = client.client.post(
                "/v1/artifacts/explain-materialize",
                json={
                    "inputs": [base_uri],
                    "transform": {
                        "executor": "local://duckdb_sql@v1",
                        "params": {"sql": "SELECT 1 as x FROM input0 LIMIT 1"},
                    },
                },
            )
            resp.raise_for_status()
            data = resp.json()

            assert data["would_hit"] is True
            assert data["would_build"] is False
            assert data["artifact_uri"] == derived_uri
        finally:
            client.close()
