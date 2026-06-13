"""Integration tests for personal mode artifacts.

These tests verify end-to-end artifact workflows:
1. Materialize with cache miss (build locally)
2. Materialize with cache hit (return cached)
3. Name pointer CRUD via client
4. Artifact data streaming
5. Service mode blocks artifact endpoints
"""

import time

import httpx
import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from strata_client.client import StrataClient

from tests.conftest import run_server_with_context, table_to_ipc_bytes


def materialize_and_upload(
    client: StrataClient,
    sql: str,
    inputs: list[str] | None = None,
    name: str | None = None,
) -> tuple[str, pa.Table]:
    """Helper to materialize using the unified API.

    Returns (artifact_uri, result_table).
    """
    inputs = inputs or []

    artifact = client.materialize(
        inputs=inputs,
        transform={"ref": "duckdb_sql@v1", "params": {"sql": sql}},
        name=name,
    )

    return artifact.uri, artifact.to_table()


def put_artifact(
    base_url: str,
    table: pa.Table,
    *,
    executor: str = "test_step@v1",
    params: dict | None = None,
    inputs: list[str] | None = None,
    name: str | None = None,
) -> dict:
    """Persist a locally-computed table via PUT /v1/artifacts (multipart)."""
    import json as json_module

    metadata: dict = {
        "inputs": inputs or [],
        "transform": {"executor": executor, "params": params or {}},
    }
    if name:
        metadata["name"] = name
    files = {
        "metadata": ("metadata.json", json_module.dumps(metadata), "application/json"),
        "data": (
            "data.arrow",
            table_to_ipc_bytes(table),
            "application/vnd.apache.arrow.stream",
        ),
    }
    resp = httpx.put(f"{base_url}/v1/artifacts", files=files, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def wait_for_build(base_url: str, artifact_uri: str, timeout: float = 30.0) -> None:
    """Poll an artifact created by the embedded build runner until ready."""
    import re

    match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", artifact_uri)
    assert match is not None
    artifact_id, version = match.group(1), match.group(2)
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = httpx.get(f"{base_url}/v1/artifacts/{artifact_id}/v/{version}")
        state = resp.json().get("state")
        if state == "ready":
            return
        if state == "failed":
            raise AssertionError(f"build failed for {artifact_uri}")
        time.sleep(0.2)
    raise AssertionError(f"build did not finish for {artifact_uri}")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def personal_mode_server(tmp_path):
    """Start a server in personal mode for artifact testing."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {"config": ctx.config, "port": ctx.port, "base_url": ctx.base_url}


@pytest.fixture
def service_mode_server(tmp_path):
    """Start a server in service mode (artifacts disabled)."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with run_server_with_context(cache_dir, deployment_mode="service") as ctx:
        yield {"config": ctx.config, "port": ctx.port, "base_url": ctx.base_url}


# =============================================================================
# HTTP Endpoint Tests
# =============================================================================


class TestArtifactEndpoints:
    """Tests for artifact HTTP endpoints."""

    def test_materialize_cache_miss_queues_embedded_build(self, personal_mode_server):
        """Cache miss queues an embedded build that completes server-side."""
        base_url = personal_mode_server["base_url"]
        response = httpx.post(
            f"{base_url}/v1/artifacts/materialize",
            json={
                "inputs": [],
                "transform": {
                    "executor": "local://duckdb_sql@v1",
                    "params": {"sql": "SELECT 1 as x"},
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["hit"] is False
        assert data["artifact_uri"].startswith("strata://artifact/")
        assert data["build_id"]
        assert data["state"] == "pending"

        # The embedded runner executes the SQL — no client-side build needed.
        wait_for_build(base_url, data["artifact_uri"])

    def test_materialize_unknown_transform_fails_fast(self, personal_mode_server):
        """An executor nothing can run is rejected, not parked in building."""
        response = httpx.post(
            f"{personal_mode_server['base_url']}/v1/artifacts/materialize",
            json={"inputs": [], "transform": {"executor": "nonexistent@v9", "params": {}}},
        )
        assert response.status_code == 400
        detail = response.json()["detail"]
        assert detail["error"] == "transform_unknown"
        assert "duckdb_sql" in str(detail["message"])

    def test_put_persists_local_result(self, personal_mode_server):
        """PUT /v1/artifacts persists a locally-computed result."""
        base_url = personal_mode_server["base_url"]
        table = pa.table({"x": [1, 2, 3]})
        data = put_artifact(base_url, table)
        assert data["artifact_uri"].startswith("strata://artifact/")
        assert data["hit"] is False

    def test_materialize_cache_hit(self, personal_mode_server):
        """Same provenance dedups: the second put is a cache hit."""
        base_url = personal_mode_server["base_url"]
        table = pa.table({"result": [42]})

        first = put_artifact(base_url, table, params={"key": "value"})
        assert first["hit"] is False

        second = put_artifact(base_url, table, params={"key": "value"})
        assert second["hit"] is True
        assert second["artifact_uri"] == first["artifact_uri"]

    def test_artifact_data_streaming(self, personal_mode_server):
        """Fetch artifact data returns Arrow IPC stream."""
        base_url = personal_mode_server["base_url"]

        # Create artifact via PUT
        import re

        table = pa.table({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        data = put_artifact(base_url, table)
        match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", data["artifact_uri"])
        artifact_id, version = match.group(1), int(match.group(2))

        # Fetch and verify
        resp = httpx.get(f"{base_url}/v1/artifacts/{artifact_id}/v/{version}/data")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/vnd.apache.arrow.stream"

        result = ipc.open_stream(pa.BufferReader(resp.content)).read_all()
        assert result.num_rows == 3
        assert set(result.column_names) == {"x", "y"}

    def test_name_crud(self, personal_mode_server):
        """Name pointer CRUD operations."""
        base_url = personal_mode_server["base_url"]

        # Create artifact via the embedded runner.
        import re

        resp = httpx.post(
            f"{base_url}/v1/artifacts/materialize",
            json={
                "inputs": [],
                "transform": {
                    "executor": "local://duckdb_sql@v1",
                    "params": {"sql": "SELECT 1 as x"},
                },
            },
        )
        artifact_uri = resp.json()["artifact_uri"]
        wait_for_build(base_url, artifact_uri)
        match = re.match(r"strata://artifact/([^@]+)@v=(\d+)", artifact_uri)
        artifact_id, version = match.group(1), int(match.group(2))

        # Set name
        resp = httpx.post(
            f"{base_url}/v1/names",
            json={"name": "my-artifact", "artifact_id": artifact_id, "version": version},
        )
        assert resp.json()["name_uri"] == "strata://name/my-artifact"

        # Resolve name
        resp = httpx.get(f"{base_url}/v1/names/my-artifact")
        assert resp.json()["artifact_uri"].startswith("strata://artifact/")

        # List names
        resp = httpx.get(f"{base_url}/v1/names")
        assert resp.json()["names"][0]["name"] == "my-artifact"

        # Delete name
        httpx.delete(f"{base_url}/v1/names/my-artifact")
        resp = httpx.get(f"{base_url}/v1/names/my-artifact")
        assert resp.status_code == 404

    def test_put_then_name_after_the_fact(self, personal_mode_server):
        """A put-created artifact is nameable after the fact in personal mode.

        Friction #7 regression: PUT used to stamp tenant "_default" while the
        names routes resolved no-header requests to tenant None, so naming a
        put-created artifact in a separate call failed with
        "belongs to tenant _default, cannot assign name in tenant None".
        No tenant header anywhere here (personal mode) — it must just work.
        """
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            art = client.put(
                inputs=[],
                transform={"executor": "local-compute", "params": {}},
                data={"x": [1, 2, 3], "y": ["a", "b", "c"]},
            )

            # Name it in a SEPARATE call (not at put time) — the friction case.
            client.set_name("taxi/tip-model", art.artifact_id, art.version)

            resolved = client.resolve_name("taxi/tip-model")
            assert resolved["artifact_uri"] == (
                f"strata://artifact/{art.artifact_id}@v={art.version}"
            )


class TestServiceModeBlocking:
    """Tests that service mode blocks artifact endpoints."""

    def test_materialize_blocked_in_service_mode(self, service_mode_server):
        """Materialize returns 403 in service mode."""
        response = httpx.post(
            f"{service_mode_server['base_url']}/v1/artifacts/materialize",
            json={"inputs": [], "transform": {"executor": "test", "params": {}}},
        )
        assert response.status_code == 403
        assert "writes_disabled" in response.json()["detail"]["error"]

    def test_name_writes_blocked_resolution_reachable_in_service_mode(self, service_mode_server):
        """Name *writes* and the *listing* stay blocked in service mode; name
        *resolution* is a read and is reachable (W2). This gateway has no
        artifact_dir, so resolution returns 404 (no store) rather than a
        mode-gated 403."""
        base_url = service_mode_server["base_url"]

        # Listing all names + writing a name stay blocked (403).
        assert httpx.get(f"{base_url}/v1/names").status_code == 403
        assert (
            httpx.post(
                f"{base_url}/v1/names",
                json={"name": "test", "artifact_id": "x", "version": 1},
            ).status_code
            == 403
        )
        # Resolving a name is a read — reachable, not mode-gated.
        assert httpx.get(f"{base_url}/v1/names/test").status_code == 404


# =============================================================================
# Contract Tests - The Core Workflow
# =============================================================================


class TestArtifactContract:
    """Contract tests that verify the complete artifact workflow.

    These are the critical tests that ensure the full loop works correctly:
    1. materialize() returns cache miss on first call
    2. Local executor (DuckDB) runs the transform
    3. materialize() returns cache hit on second call
    4. Data is accessible via artifact URI and name URI
    5. Persistence survives server restart
    """

    def test_transform_pipeline_with_dependencies(self, personal_mode_server):
        """Full pipeline: source -> transform -> aggregate, with real input dependencies.

        This is the realistic workflow that exercises the entire artifact system:
        - Stage 1: Create source data artifact
        - Stage 2: Filter source data (depends on stage 1)
        - Stage 3: Aggregate filtered data (depends on stage 2)
        - Verify all stages cache correctly and data flows through
        """
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Stage 1: Create source data
            source = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                            SELECT 'click' as event, 1 as user_id, 10 as amount
                            UNION ALL SELECT 'click', 1, 20
                            UNION ALL SELECT 'view', 2, 5
                            UNION ALL SELECT 'click', 2, 30
                            UNION ALL SELECT 'view', 1, 15
                        """
                    },
                },
                name="events-source",
            )
            source_data = source.to_table()
            assert source_data.num_rows == 5

            # Stage 2: Filter to clicks only (depends on source)
            filtered = client.materialize(
                inputs=[source.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input0 WHERE event = 'click'"},
                },
                name="clicks-only",
            )
            assert filtered.cache_hit is False, "Should be cache miss for new transform"

            filtered_data = filtered.to_table()
            assert filtered_data.num_rows == 3
            assert all(e == "click" for e in filtered_data["event"].to_pylist())

            # Stage 3: Aggregate clicks by user (depends on filtered)
            agg_sql = (
                "SELECT user_id, sum(amount) as total FROM input0 GROUP BY user_id ORDER BY user_id"
            )
            aggregated = client.materialize(
                inputs=[filtered.uri],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": agg_sql}},
                name="user-totals",
            )
            assert aggregated.cache_hit is False

            agg_data = aggregated.to_table()
            assert agg_data.to_pydict() == {"user_id": [1, 2], "total": [30, 30]}

            # Verify all stages are now cached
            filtered2 = client.materialize(
                inputs=[source.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input0 WHERE event = 'click'"},
                },
            )
            assert filtered2.cache_hit is True, "Filter stage should be cached"

            aggregated2 = client.materialize(
                inputs=[filtered.uri],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": agg_sql}},
            )
            assert aggregated2.cache_hit is True, "Aggregate stage should be cached"

            # Verify names resolve to correct data
            for name, expected in [
                ("clicks-only", filtered_data.to_pydict()),
                ("user-totals", agg_data.to_pydict()),
            ]:
                artifact = client.get_artifact_by_name(name)
                assert artifact.to_table().to_pydict() == expected

    def test_persistence_across_restart(self, tmp_path):
        """Artifacts persist across server restarts."""
        cache_dir = tmp_path / "cache"
        artifact_dir = tmp_path / "artifacts"
        cache_dir.mkdir()
        artifact_dir.mkdir()

        expected_data = {"x": [1, 2, 3], "y": ["a", "b", "c"]}
        union_sql = "SELECT 1 as x, 'a' as y UNION ALL SELECT 2, 'b' UNION ALL SELECT 3, 'c'"

        # Phase 1: Create artifacts
        with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
            with StrataClient(base_url=ctx.base_url) as client:
                artifact = client.materialize(
                    inputs=[],
                    transform={"ref": "duckdb_sql@v1", "params": {"sql": union_sql}},
                    name="persistent-artifact",
                )
                saved_uri = artifact.uri

        # Phase 2: Restart and verify
        time.sleep(0.2)  # Ensure clean shutdown
        with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
            with StrataClient(base_url=ctx.base_url) as client:
                # Cache should still hit
                artifact = client.materialize(
                    inputs=[],
                    transform={"ref": "duckdb_sql@v1", "params": {"sql": union_sql}},
                )
                assert artifact.cache_hit is True, "Should be cache hit after restart"
                assert artifact.uri == saved_uri

                # Data should be accessible
                data = artifact.to_table()
                assert data.to_pydict() == expected_data

                # Name should still resolve
                resolved = client.get_artifact_by_name("persistent-artifact")
                assert resolved.uri == saved_uri

    def test_provenance_deduplication(self, personal_mode_server):
        """Same inputs (same order) + transform deduplicate via provenance hash.

        Input order is significant: ``duckdb_sql`` binds inputs positionally
        (``input0``/``input1``), so ``[b, a]`` is a different computation
        from ``[a, b]`` and must NOT dedup to the same artifact.
        """
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create real input artifacts
            input_a = client.materialize(
                inputs=[],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": "SELECT 'a' as val"}},
            )
            input_b = client.materialize(
                inputs=[],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": "SELECT 'b' as val"}},
            )
            input_c = client.materialize(
                inputs=[],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": "SELECT 'c' as val"}},
            )

            sql = "SELECT 'dedup' as tag"

            # First call with inputs [a, b]
            artifact1 = client.materialize(
                inputs=[input_a.uri, input_b.uri],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": sql}},
            )
            assert artifact1.cache_hit is False

            # Identical call (same inputs, same order) - dedups to a cache hit.
            artifact_same = client.materialize(
                inputs=[input_a.uri, input_b.uri],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": sql}},
            )
            assert artifact_same.cache_hit is True
            assert artifact_same.uri == artifact1.uri

            # Reordered positional inputs are a DIFFERENT computation - cache
            # miss, distinct artifact. (Inputs feed duckdb as input0/input1;
            # order is part of the provenance via the transform spec.)
            artifact2 = client.materialize(
                inputs=[input_b.uri, input_a.uri],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": sql}},
            )
            assert artifact2.cache_hit is False
            assert artifact2.uri != artifact1.uri

            # Different inputs - should miss
            artifact3 = client.materialize(
                inputs=[input_a.uri, input_c.uri],
                transform={"ref": "duckdb_sql@v1", "params": {"sql": sql}},
            )
            assert artifact3.cache_hit is False
            assert artifact3.uri != artifact1.uri

    def test_artifact_and_name_uri_equivalence(self, personal_mode_server):
        """Artifact URI and name URI return identical data."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            artifact = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 1 as id, 'alice' as name, 95.5 as score"},
                },
                name="equivalence-test",
            )

            via_artifact = artifact.to_table()
            via_name = client.get_artifact_by_name("equivalence-test").to_table()

            assert via_artifact.schema == via_name.schema
            assert via_artifact.to_pydict() == via_name.to_pydict()


# =============================================================================
# Lifecycle Management Tests
# =============================================================================


class TestArtifactLifecycle:
    """Tests for artifact lifecycle management: list, delete, GC, usage."""

    def test_list_artifacts(self, personal_mode_server):
        """List artifacts with pagination and filtering."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create multiple artifacts with names
            for i in range(3):
                materialize_and_upload(
                    client,
                    sql=f"SELECT {i} as idx",
                    name=f"list-test-{i}",
                )

            # List all
            result = client.list_artifacts()
            assert len(result["artifacts"]) >= 3

            # List with limit
            result = client.list_artifacts(limit=2)
            assert len(result["artifacts"]) == 2

            # List only ready artifacts
            result = client.list_artifacts(state="ready")
            assert all(a["state"] == "ready" for a in result["artifacts"])

            # List by name prefix - should return artifacts that have these names
            result = client.list_artifacts(name_prefix="list-test-")
            assert len(result["artifacts"]) == 3
            # Each artifact should have a valid URI
            for a in result["artifacts"]:
                assert a["artifact_uri"].startswith("strata://artifact/")

    def test_delete_artifact(self, personal_mode_server):
        """Delete an artifact version."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create artifact
            artifact = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'to-delete' as status"},
                },
                name="delete-test",
            )

            # Verify it exists
            assert artifact.to_table().to_pydict() == {"status": ["to-delete"]}

            # Delete it
            result = client.delete_artifact(artifact.artifact_id, artifact.version)
            assert result["deleted"] is True

            # Verify it's gone - getting artifact by name should fail
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                client.get_artifact_by_name("delete-test")
            assert exc_info.value.response.status_code == 404

    def test_garbage_collect_unreferenced(self, personal_mode_server):
        """GC removes unreferenced artifacts older than cutoff."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create artifact WITHOUT a name (unreferenced)
            unreferenced = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'gc-candidate' as status"},
                },
                # No name - this artifact is unreferenced
            )
            assert unreferenced.name is None

            # Create artifact WITH a name (referenced)
            _referenced = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'gc-safe' as status"},
                },
                name="gc-protected",
            )

            # Get usage before GC
            usage_before = client.get_artifact_usage()
            assert usage_before["unreferenced_count"] >= 1

            # GC with max_age_days=0 should delete the unreferenced one immediately
            gc_result = client.garbage_collect(max_age_days=0)
            assert gc_result["deleted_count"] >= 1

            # Named artifact should still exist
            fetched = client.get_artifact_by_name("gc-protected")
            assert fetched.to_table().to_pydict() == {"status": ["gc-safe"]}

    def test_usage_metrics(self, personal_mode_server):
        """Usage metrics track artifacts correctly."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Get initial usage
            usage1 = client.get_artifact_usage()
            initial_versions = usage1["total_versions"]

            # Create some artifacts
            for i in range(2):
                materialize_and_upload(
                    client,
                    sql=f"SELECT {i} as idx, 'usage-test' as tag",
                    name=f"usage-{i}",
                )

            # Check usage increased
            usage2 = client.get_artifact_usage()
            assert usage2["total_versions"] >= initial_versions + 2
            assert usage2["total_bytes"] > 0
            assert usage2["name_count"] >= 2

    def test_delete_nonexistent_artifact(self, personal_mode_server):
        """Deleting nonexistent artifact returns 404."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                client.delete_artifact("nonexistent-id", 999)
            assert exc_info.value.response.status_code == 404

    def test_gc_preserves_named_artifacts(self, personal_mode_server):
        """GC never deletes artifacts with name pointers."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create several named artifacts
            artifacts = []
            for i in range(3):
                artifact = client.materialize(
                    inputs=[],
                    transform={
                        "ref": "duckdb_sql@v1",
                        "params": {"sql": f"SELECT {i} as idx"},
                    },
                    name=f"gc-preserve-{i}",
                )
                artifacts.append(artifact)

            # Run aggressive GC
            client.garbage_collect(max_age_days=0)

            # All named artifacts should still exist
            for i, artifact in enumerate(artifacts):
                fetched = client.get_artifact_by_name(f"gc-preserve-{i}")
                assert fetched.to_table().to_pydict() == {"idx": [i]}


# =============================================================================
# Staleness Detection Tests
# =============================================================================


class TestStalenessDetection:
    """Tests for artifact staleness detection endpoints.

    Staleness detection allows users to:
    - Check if a named artifact's inputs have changed
    - Understand why a rebuild is needed
    - Get dry-run materialize explanations
    """

    def test_get_name_status_not_stale(self, personal_mode_server):
        """Name status shows not stale when inputs unchanged."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create artifact with no inputs (will never be stale)
            client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'fresh' as status"},
                },
                name="fresh-artifact",
            )

            # Check status
            status = client.get_name_status("fresh-artifact")
            assert status["name"] == "fresh-artifact"
            assert status["is_stale"] is False
            assert status["stale_reason"] is None
            assert status["changed_inputs"] is None
            assert status["artifact_uri"].startswith("strata://artifact/")
            assert status["state"] == "ready"

    def test_get_name_status_not_found(self, personal_mode_server):
        """Name status returns 404 for unknown name."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                client.get_name_status("nonexistent-name")
            assert exc_info.value.response.status_code == 404

    def test_is_artifact_stale_convenience(self, personal_mode_server):
        """is_artifact_stale convenience method returns boolean."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'test' as value"},
                },
                name="staleness-check",
            )

            # Should not be stale
            assert client.is_artifact_stale("staleness-check") is False

    def test_explain_materialize_hit(self, personal_mode_server):
        """Explain materialize shows would_hit for cached computation."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create and finalize an artifact
            client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'cached' as result"},
                },
                name="explain-hit-test",
            )

            # Explain the same computation
            result = client.explain_materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'cached' as result"},
                },
                name="explain-hit-test",
            )

            assert result["would_hit"] is True
            assert result["would_build"] is False
            assert result["artifact_uri"].startswith("strata://artifact/")
            assert result["is_stale"] is False

    def test_explain_materialize_miss_no_name(self, personal_mode_server):
        """Explain materialize shows would_build for new computation."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            result = client.explain_materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'new-computation' as result"},
                },
            )

            assert result["would_hit"] is False
            assert result["would_build"] is True
            assert result["artifact_uri"] is None
            assert result["is_stale"] is False

    def test_explain_materialize_shows_stale_reason(self, personal_mode_server):
        """Explain materialize shows why rebuild is needed when stale.

        This test creates an artifact that depends on another artifact,
        then modifies the transform (while keeping same inputs) to trigger
        a cache miss. The explain should show that a rebuild is needed.

        Note: Staleness is detected by comparing the named artifact's stored
        input versions against the SAME input URIs' current versions. If the
        input URIs themselves differ, it's a different computation, not staleness.
        """
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create source artifact
            source = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 1 as version"},
                },
                name="stale-source",
            )

            # Create dependent artifact using source
            client.materialize(
                inputs=[source.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT version * 10 as derived FROM input0"},
                },
                name="stale-dependent",
            )

            # Now explain a computation with DIFFERENT transform params (same inputs)
            # This triggers cache miss + staleness check against existing named artifact
            # Different SQL than above
            result = client.explain_materialize(
                inputs=[source.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT version * 100 as derived FROM input0"},
                },
                name="stale-dependent",
            )

            # Should show would_build (cache miss due to different transform)
            # is_stale should be False since input versions haven't changed
            assert result["would_hit"] is False
            assert result["would_build"] is True
            # The artifact exists but inputs haven't changed - it's a different transform
            # So is_stale refers to whether the EXISTING artifact's inputs changed
            # In this case, inputs are the same, so not stale (just a different transform)
            assert result["is_stale"] is False

    def test_name_status_with_artifact_dependency(self, personal_mode_server):
        """Name status correctly reports dependencies for artifact.

        This test verifies that when an artifact depends on another artifact,
        the name status shows the input versions correctly.

        Note: Staleness detection compares stored vs current versions of the
        SAME input URIs. If the input URI itself changes (e.g., pointing to
        a new artifact version), that's a different computation entirely.
        """
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create base artifact
            base = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'v1' as data"},
                },
                name="base-artifact",
            )

            # Create derived artifact that depends on base
            client.materialize(
                inputs=[base.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT data || '-derived' as result FROM input0"},
                },
                name="derived-artifact",
            )

            # Verify derived is NOT stale initially
            status = client.get_name_status("derived-artifact")
            assert status["is_stale"] is False
            # Should show the input dependency
            assert base.uri in status["input_versions"]
            # Input version should be the artifact version string
            assert "@v=" in status["input_versions"][base.uri]

    def test_name_status_reports_input_versions(self, personal_mode_server):
        """Name status reports stored input versions correctly."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create two input artifacts
            input1 = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'input1' as source"},
                },
            )
            input2 = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'input2' as source"},
                },
            )

            # Create artifact that depends on both
            client.materialize(
                inputs=[input1.uri, input2.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'combined' as result"},
                },
                name="multi-input-artifact",
            )

            # Check status shows all input versions
            status = client.get_name_status("multi-input-artifact")
            assert len(status["input_versions"]) == 2
            assert input1.uri in status["input_versions"]
            assert input2.uri in status["input_versions"]
            assert status["is_stale"] is False

    def test_explain_resolved_input_versions(self, personal_mode_server):
        """Explain returns resolved input versions."""
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create an artifact to use as input
            input_artifact = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT 'input-data' as value"},
                },
            )

            # Explain with that input
            result = client.explain_materialize(
                inputs=[input_artifact.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input0"},
                },
            )

            # Should have resolved versions
            assert result["resolved_input_versions"] is not None
            assert input_artifact.uri in result["resolved_input_versions"]
            # Version should be in format "artifact_id@v=N"
            version = result["resolved_input_versions"][input_artifact.uri]
            assert "@v=" in version


# =============================================================================
# Tests for unified materialize() API with real tables
# =============================================================================


@pytest.fixture
def iceberg_warehouse(tmp_path):
    """Create a temporary warehouse with a sample Iceberg table."""
    import sys

    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")

    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField, StringType

    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    # Create a SQL catalog
    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )

    # Create namespace
    catalog.create_namespace("test_db")

    # Define schema
    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
        NestedField(3, "category", StringType(), required=False),
    )

    # Create table
    table = catalog.create_table("test_db.events", schema)

    # Create sample data
    data = pa.table(
        {
            "id": pa.array([1, 2, 3, 4, 5], type=pa.int64()),
            "value": pa.array([10.0, 20.0, 30.0, 40.0, 50.0], type=pa.float64()),
            "category": pa.array(["A", "B", "A", "B", "A"], type=pa.string()),
        }
    )

    # Append data to table
    table.append(data)

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.events",
        "catalog": catalog,
        "table": table,
    }


@pytest.fixture
def artifact_server_with_warehouse(tmp_path, iceberg_warehouse):
    """Start a server with artifact support and an Iceberg warehouse."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {
            **iceberg_warehouse,
            "base_url": ctx.base_url,
            "config": ctx.config,
        }


class TestUnifiedMaterializeAPI:
    """Tests for the unified client.materialize() API with real tables."""

    def test_materialize_from_iceberg_table(self, artifact_server_with_warehouse):
        """Materialize an artifact from a real Iceberg table."""
        table_uri = artifact_server_with_warehouse["table_uri"]
        base_url = artifact_server_with_warehouse["base_url"]

        with StrataClient(base_url=base_url) as client:
            # Materialize with SQL transform
            artifact = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": "SELECT category, SUM(value) as total FROM input0 GROUP BY category"
                    },
                },
                name="category_totals",
            )

            # Check artifact metadata
            assert artifact.artifact_id is not None
            assert artifact.version >= 1
            assert artifact.cache_hit is False  # First time should be cache miss
            assert artifact.execution in ("local", "server")
            assert artifact.name == "category_totals"

            # Check URI format
            assert artifact.uri.startswith("strata://artifact/")
            assert f"@v={artifact.version}" in artifact.uri

            # Fetch the data and verify
            result_table = artifact.to_table()
            assert result_table.num_rows == 2  # Two categories: A and B
            assert set(result_table.column_names) == {"category", "total"}

            # Verify aggregation is correct
            df = artifact.to_pandas()
            totals = dict(zip(df["category"], df["total"]))
            assert totals["A"] == 90.0  # 10 + 30 + 50
            assert totals["B"] == 60.0  # 20 + 40

    def test_materialize_cache_hit(self, artifact_server_with_warehouse):
        """Second materialize with same inputs should hit cache."""
        table_uri = artifact_server_with_warehouse["table_uri"]
        base_url = artifact_server_with_warehouse["base_url"]

        with StrataClient(base_url=base_url) as client:
            # First call - cache miss
            artifact1 = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT COUNT(*) as cnt FROM input0"},
                },
            )
            assert artifact1.cache_hit is False

            # Second call with same transform - should hit cache
            artifact2 = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT COUNT(*) as cnt FROM input0"},
                },
            )
            assert artifact2.cache_hit is True
            assert artifact2.artifact_id == artifact1.artifact_id
            assert artifact2.version == artifact1.version

    def test_materialize_chain_artifacts(self, artifact_server_with_warehouse):
        """Chain artifacts: use output of one as input to another."""
        table_uri = artifact_server_with_warehouse["table_uri"]
        base_url = artifact_server_with_warehouse["base_url"]

        with StrataClient(base_url=base_url) as client:
            # Stage 1: Filter to high-value rows
            filtered = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input0 WHERE value > 25"},
                },
                name="high_value",
            )

            # Verify filtered data
            filtered_df = filtered.to_pandas()
            assert len(filtered_df) == 3  # values 30, 40, 50
            assert all(filtered_df["value"] > 25)

            # Stage 2: Aggregate the filtered data
            aggregated = client.materialize(
                inputs=[filtered.uri],  # Use artifact URI as input
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT AVG(value) as avg_value FROM input0"},
                },
                name="high_value_avg",
            )

            # Verify aggregation
            agg_df = aggregated.to_pandas()
            assert agg_df["avg_value"].iloc[0] == 40.0  # (30 + 40 + 50) / 3

    def test_materialize_multi_input_chain(self, personal_mode_server):
        """Chain artifacts with multiple inputs (fan-in pattern).

        This tests the common pattern where multiple data sources are
        processed independently and then joined together:
        - Create two independent data sources
        - Process each independently
        - Join the processed results together
        - Verify caching works correctly for the entire DAG
        """
        with StrataClient(base_url=personal_mode_server["base_url"]) as client:
            # Create two independent source artifacts
            orders = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                            SELECT 1 as order_id, 100 as customer_id, 50.00 as amount
                            UNION ALL SELECT 2, 100, 75.00
                            UNION ALL SELECT 3, 200, 120.00
                            UNION ALL SELECT 4, 200, 30.00
                        """
                    },
                },
                name="orders",
            )
            assert orders.cache_hit is False

            customers = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                            SELECT 100 as customer_id, 'Alice' as name
                            UNION ALL SELECT 200, 'Bob'
                        """
                    },
                },
                name="customers",
            )
            assert customers.cache_hit is False

            # Process each independently (aggregation)
            order_totals = client.materialize(
                inputs=[orders.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                            SELECT customer_id, SUM(amount) as total_amount
                            FROM input0
                            GROUP BY customer_id
                        """
                    },
                },
                name="order_totals",
            )
            assert order_totals.cache_hit is False

            # Join the two processed artifacts together
            joined = client.materialize(
                inputs=[order_totals.uri, customers.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                            SELECT c.name, o.total_amount
                            FROM input0 o
                            JOIN input1 c ON o.customer_id = c.customer_id
                            ORDER BY c.name
                        """
                    },
                },
                name="customer_order_totals",
            )
            assert joined.cache_hit is False

            # Verify the joined result
            result = joined.to_table().to_pydict()
            assert result == {
                "name": ["Alice", "Bob"],
                "total_amount": [125.0, 150.0],  # Alice: 50+75, Bob: 120+30
            }

            # Verify the entire DAG is cached on re-request
            orders2 = client.materialize(
                inputs=[],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                            SELECT 1 as order_id, 100 as customer_id, 50.00 as amount
                            UNION ALL SELECT 2, 100, 75.00
                            UNION ALL SELECT 3, 200, 120.00
                            UNION ALL SELECT 4, 200, 30.00
                        """
                    },
                },
            )
            assert orders2.cache_hit is True
            assert orders2.uri == orders.uri

            joined2 = client.materialize(
                inputs=[order_totals.uri, customers.uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {
                        "sql": """
                            SELECT c.name, o.total_amount
                            FROM input0 o
                            JOIN input1 c ON o.customer_id = c.customer_id
                            ORDER BY c.name
                        """
                    },
                },
            )
            assert joined2.cache_hit is True
            assert joined2.uri == joined.uri

            # Verify names resolve correctly
            resolved = client.get_artifact_by_name("customer_order_totals")
            assert resolved.uri == joined.uri
            assert resolved.to_table().to_pydict() == result

    def test_materialize_with_refresh(self, artifact_server_with_warehouse):
        """Force refresh recomputes even if cached."""
        table_uri = artifact_server_with_warehouse["table_uri"]
        base_url = artifact_server_with_warehouse["base_url"]

        with StrataClient(base_url=base_url) as client:
            # First call
            artifact1 = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT MAX(value) as max_val FROM input0"},
                },
                name="max_value",
            )

            # Second call with refresh=True should recompute
            artifact2 = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT MAX(value) as max_val FROM input0"},
                },
                name="max_value",
                refresh=True,
            )

            # Both should have valid data
            assert artifact1.to_pandas()["max_val"].iloc[0] == 50.0
            assert artifact2.to_pandas()["max_val"].iloc[0] == 50.0

    def test_artifact_info_and_lineage(self, artifact_server_with_warehouse):
        """Test artifact.info() and artifact.lineage() methods."""
        table_uri = artifact_server_with_warehouse["table_uri"]
        base_url = artifact_server_with_warehouse["base_url"]

        with StrataClient(base_url=base_url) as client:
            artifact = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT * FROM input0"},
                },
            )

            # Get info
            info = artifact.info()
            assert info["artifact_id"] == artifact.artifact_id
            assert info["version"] == artifact.version
            assert info["state"] == "ready"
            assert "row_count" in info
            assert info["row_count"] == 5

    def test_get_artifact_by_name(self, artifact_server_with_warehouse):
        """Test retrieving artifact by name."""
        table_uri = artifact_server_with_warehouse["table_uri"]
        base_url = artifact_server_with_warehouse["base_url"]

        with StrataClient(base_url=base_url) as client:
            # Create named artifact
            original = client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT MIN(value) as min_val FROM input0"},
                },
                name="min_value",
            )

            # Retrieve by name
            retrieved = client.get_artifact_by_name("min_value")
            assert retrieved.artifact_id == original.artifact_id
            assert retrieved.version == original.version
            assert retrieved.name == "min_value"

            # Data should match
            assert retrieved.to_pandas()["min_val"].iloc[0] == 10.0

    def test_explain_materialize_with_real_table(self, artifact_server_with_warehouse):
        """Test explain_materialize() with real table."""
        table_uri = artifact_server_with_warehouse["table_uri"]
        base_url = artifact_server_with_warehouse["base_url"]

        with StrataClient(base_url=base_url) as client:
            # Explain before materializing - should be cache miss
            result = client.explain_materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT DISTINCT category FROM input0"},
                },
            )
            # Note: field names may vary based on server implementation
            assert "cache_hit" in result or "would_hit" in result

            # Actually materialize
            client.materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT DISTINCT category FROM input0"},
                },
            )

            # Explain again - should now be cache hit
            result2 = client.explain_materialize(
                inputs=[table_uri],
                transform={
                    "ref": "duckdb_sql@v1",
                    "params": {"sql": "SELECT DISTINCT category FROM input0"},
                },
            )
            # Check for cache hit indication
            hit_key = "cache_hit" if "cache_hit" in result2 else "would_hit"
            assert result2[hit_key] is True


# =============================================================================
# Regression tests for #121: multi-row-group scan artifact truncation
# =============================================================================


@pytest.fixture
def multi_file_warehouse(tmp_path):
    """Warehouse whose table spans multiple Parquet files (multi-task scans).

    Three separate appends produce three data files, so a scan plan has
    three tasks — the shape that exposed #121 (per-task IPC streams naively
    concatenated; standard readers stopped at the first EOS marker).
    """
    import sys

    if sys.platform == "win32":
        pytest.skip("pyiceberg + pyarrow LocalFileSystem path handling broken on Windows")

    from pyiceberg.catalog.sql import SqlCatalog
    from pyiceberg.schema import Schema
    from pyiceberg.types import DoubleType, LongType, NestedField

    warehouse_path = tmp_path / "warehouse"
    warehouse_path.mkdir()

    catalog = SqlCatalog(
        "strata",
        **{
            "uri": f"sqlite:///{warehouse_path / 'catalog.db'}",
            "warehouse": str(warehouse_path),
        },
    )
    catalog.create_namespace("test_db")

    schema = Schema(
        NestedField(1, "id", LongType(), required=False),
        NestedField(2, "value", DoubleType(), required=False),
    )
    table = catalog.create_table("test_db.chunked", schema)

    rows_per_append = 400
    appends = 3
    for a in range(appends):
        start = a * rows_per_append
        table.append(
            pa.table(
                {
                    "id": pa.array(range(start, start + rows_per_append), type=pa.int64()),
                    "value": pa.array(
                        [float(i) for i in range(start, start + rows_per_append)],
                        type=pa.float64(),
                    ),
                }
            )
        )

    return {
        "warehouse_path": warehouse_path,
        "table_uri": f"file://{warehouse_path}#test_db.chunked",
        "total_rows": rows_per_append * appends,
    }


@pytest.fixture
def multi_file_server(tmp_path, multi_file_warehouse):
    """Personal-mode server over the multi-file warehouse."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(cache_dir, artifact_dir, "personal") as ctx:
        yield {**multi_file_warehouse, "base_url": ctx.base_url}


class TestMultiTaskScanIntegrity:
    """A scan spanning multiple files/row groups must never truncate (#121)."""

    def test_stream_mode_returns_all_rows(self, multi_file_server):
        """Stream-mode materialize delivers every task's rows on the wire."""
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            artifact = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
            )
            table = artifact.to_table()
            assert table.num_rows == multi_file_server["total_rows"]
            ids = sorted(table.column("id").to_pylist())
            assert ids == list(range(multi_file_server["total_rows"]))

    def test_persisted_artifact_returns_all_rows(self, multi_file_server):
        """The persisted blob reads back complete via the data endpoint."""
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            artifact = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
            )
            # Consume the stream so the artifact finalizes.
            artifact.to_table()

            refetched = client.fetch(artifact.uri)
            assert refetched.num_rows == multi_file_server["total_rows"]

    def test_artifact_blob_is_exactly_one_ipc_stream(self, multi_file_server):
        """Raw data endpoint bytes parse as ONE stream matching row_count."""
        import io

        base_url = multi_file_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
            )
            artifact.to_table()
            info = artifact.info()

        url = f"{base_url}/v1/artifacts/{artifact.artifact_id}/v/{artifact.version}/data"
        content = httpx.get(url, timeout=30.0).raise_for_status().content

        f = io.BytesIO(content)
        streams = 0
        readable_rows = 0
        while f.tell() < len(content):
            try:
                reader = ipc.open_stream(f)
            except pa.ArrowInvalid:
                break
            readable_rows += reader.read_all().num_rows
            streams += 1

        assert streams == 1
        assert readable_rows == multi_file_server["total_rows"]
        assert readable_rows == info["row_count"]

    def test_artifact_mode_build_returns_all_rows(self, multi_file_server):
        """mode='artifact' background build persists a complete blob."""
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            artifact = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                mode="artifact",
                wait=True,
            )
            table = client.fetch(artifact.uri)
            assert table.num_rows == multi_file_server["total_rows"]


class TestRefreshSupersede:
    """refresh=True rebuilds the same artifact identity (#123)."""

    def test_refresh_bumps_version_of_same_artifact(self, multi_file_server):
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            first = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
            )
            first.to_table()  # finalize

            rebuilt = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                refresh=True,
            )
            table = rebuilt.to_table()

            # Same identity, new version, complete data
            assert rebuilt.artifact_id == first.artifact_id
            assert rebuilt.version == first.version + 1
            assert table.num_rows == multi_file_server["total_rows"]

    def test_provenance_cache_returns_rebuild(self, multi_file_server):
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            first = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
            )
            first.to_table()
            rebuilt = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                refresh=True,
            )
            rebuilt.to_table()

            # A plain materialize now cache-hits on the rebuild, not the original
            hit = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
            )
            assert hit.cache_hit is True
            assert (hit.artifact_id, hit.version) == (rebuilt.artifact_id, rebuilt.version)

    def test_superseded_version_stays_fetchable(self, multi_file_server):
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            first = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
            )
            first.to_table()
            rebuilt = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                refresh=True,
            )
            rebuilt.to_table()

            # Immutability: the old version's URI still serves its data
            old = client.fetch(first.uri)
            assert old.num_rows == multi_file_server["total_rows"]


class TestNamespacedNames:
    """Names containing '/' must be readable, not write-only (friction #6)."""

    def test_slash_name_round_trip(self, multi_file_server):
        base_url = multi_file_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                name="team/dataset/raw",
            )
            artifact.to_table()

            # Resolve via raw HTTP (slash flows through the :path converter)
            resp = httpx.get(f"{base_url}/v1/names/team/dataset/raw", timeout=30.0)
            assert resp.status_code == 200
            assert resp.json()["artifact_uri"] == artifact.uri

            # Status route with trailing /status segment
            resp = httpx.get(f"{base_url}/v1/artifacts/names/team/dataset/raw/status", timeout=30.0)
            assert resp.status_code == 200
            assert resp.json()["name"] == "team/dataset/raw"

            # SDK paths over the same name
            status = client.get_name_status("team/dataset/raw")
            assert status["is_stale"] is False
            resolved = client.resolve_name("team/dataset/raw")
            assert resolved["artifact_uri"] == artifact.uri

            # Delete
            resp = httpx.delete(f"{base_url}/v1/names/team/dataset/raw", timeout=30.0)
            assert resp.status_code == 200
            resp = httpx.get(f"{base_url}/v1/names/team/dataset/raw", timeout=30.0)
            assert resp.status_code == 404

    def test_list_names_route_still_works(self, multi_file_server):
        """The greedy path converter must not swallow the list route."""
        resp = httpx.get(f"{multi_file_server['base_url']}/v1/names", timeout=30.0)
        assert resp.status_code == 200
        assert "names" in resp.json()


class TestRegistry:
    """Aliases, tags, and audit over HTTP + SDK (#129)."""

    def _make_model(self, client, table_uri, refresh=False):
        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
            name="team/model",
            refresh=refresh,
        )
        artifact.to_table()
        return artifact

    def test_alias_round_trip(self, multi_file_server):
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            artifact = self._make_model(client, multi_file_server["table_uri"])

            client.set_alias("team/model", "champion", artifact.artifact_id, artifact.version)
            resolved = client.resolve_alias("team/model", "champion")
            assert resolved["artifact_id"] == artifact.artifact_id
            assert resolved["version"] == artifact.version

            aliases = client.list_aliases("team/model")
            assert [a["alias"] for a in aliases] == ["champion"]

            client.delete_alias("team/model", "champion")
            with pytest.raises(httpx.HTTPStatusError):
                client.resolve_alias("team/model", "champion")

    def test_promotion_with_history(self, multi_file_server):
        """The friction-#9 flow: promote, promote again, recover the past."""
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            v1 = self._make_model(client, multi_file_server["table_uri"])
            client.set_alias("team/model", "champion", v1.artifact_id, v1.version)

            v2 = self._make_model(client, multi_file_server["table_uri"], refresh=True)
            assert v2.version == v1.version + 1
            client.set_alias("team/model", "champion", v2.artifact_id, v2.version)

            # Current champion is v2; the audit recovers v1
            assert client.resolve_alias("team/model", "champion")["version"] == v2.version
            audit = client.get_registry_audit(name="team/model")
            moves = [e for e in audit if e["action"] == "alias_set"]
            assert moves[0]["from_version"] == v1.version
            assert moves[0]["to_version"] == v2.version

    def test_tags_round_trip(self, multi_file_server):
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            artifact = self._make_model(client, multi_file_server["table_uri"])

            client.set_tag(artifact.artifact_id, artifact.version, "auc", "0.91")
            client.set_tag(artifact.artifact_id, artifact.version, "validated_by", "ci")
            tags = client.get_tags(artifact.artifact_id, artifact.version)
            assert tags == {"auc": "0.91", "validated_by": "ci"}

            client.delete_tag(artifact.artifact_id, artifact.version, "auc")
            assert client.get_tags(artifact.artifact_id, artifact.version) == {"validated_by": "ci"}

    def test_audit_covers_names_and_tags(self, multi_file_server):
        with StrataClient(base_url=multi_file_server["base_url"]) as client:
            artifact = self._make_model(client, multi_file_server["table_uri"])
            client.set_tag(artifact.artifact_id, artifact.version, "k", "v")

            entries = client.get_registry_audit(artifact_id=artifact.artifact_id)
            actions = {e["action"] for e in entries}
            assert "tag_set" in actions
            by_name = client.get_registry_audit(name="team/model")
            assert any(e["action"] == "name_set" for e in by_name)

    def test_alias_on_slash_name_routes_correctly(self, multi_file_server):
        """Greedy {name:path} routes must not swallow /aliases/ URLs."""
        base_url = multi_file_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = self._make_model(client, multi_file_server["table_uri"])
            client.set_alias("team/model", "champion", artifact.artifact_id, artifact.version)

        # Raw HTTP: name contains '/', alias segment must still route
        resp = httpx.get(f"{base_url}/v1/names/team/model/aliases/champion", timeout=30.0)
        assert resp.status_code == 200
        assert resp.json()["alias"] == "champion"

        # And the plain name resolve still works alongside
        resp = httpx.get(f"{base_url}/v1/names/team/model", timeout=30.0)
        assert resp.status_code == 200


@pytest.fixture
def gated_server(tmp_path, multi_file_warehouse):
    """Personal-mode server where the 'champion' alias requires approval."""
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    cache_dir.mkdir()
    artifact_dir.mkdir()

    with run_server_with_context(
        cache_dir,
        artifact_dir,
        "personal",
        registry_protected_aliases=["champion"],
    ) as ctx:
        yield {**multi_file_warehouse, "base_url": ctx.base_url}


class TestApprovalGates:
    """Protected aliases queue for approval instead of applying (#129)."""

    def _make_model(self, client, table_uri):
        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
            name="team/model",
        )
        artifact.to_table()
        return artifact

    def test_protected_alias_goes_pending_then_approves(self, gated_server):
        base_url = gated_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = self._make_model(client, gated_server["table_uri"])

            # Protected: returns 202 pending, alias not applied
            resp = httpx.put(
                f"{base_url}/v1/names/team/model/aliases/champion",
                json={"artifact_id": artifact.artifact_id, "version": artifact.version},
                timeout=30.0,
            )
            assert resp.status_code == 202
            assert resp.json()["status"] == "pending"
            with pytest.raises(httpx.HTTPStatusError):
                client.resolve_alias("team/model", "champion")

            # Visible in the queue
            pending = client.list_pending_changes()
            assert [(p["name"], p["alias"]) for p in pending] == [("team/model", "champion")]

            # Approve applies it
            result = client.approve_alias_change("team/model", "champion")
            assert result["status"] == "approved"
            assert client.resolve_alias("team/model", "champion")["version"] == artifact.version
            assert client.list_pending_changes() == []

    def test_reject_discards_change(self, gated_server):
        base_url = gated_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = self._make_model(client, gated_server["table_uri"])
            httpx.put(
                f"{base_url}/v1/names/team/model/aliases/champion",
                json={"artifact_id": artifact.artifact_id, "version": artifact.version},
                timeout=30.0,
            )
            result = client.reject_alias_change("team/model", "champion")
            assert result["status"] == "rejected"
            with pytest.raises(httpx.HTTPStatusError):
                client.resolve_alias("team/model", "champion")

    def test_unprotected_alias_applies_immediately(self, gated_server):
        with StrataClient(base_url=gated_server["base_url"]) as client:
            artifact = self._make_model(client, gated_server["table_uri"])
            client.set_alias("team/model", "candidate", artifact.artifact_id, artifact.version)
            assert client.resolve_alias("team/model", "candidate")["version"] == artifact.version

    def test_protected_delete_goes_pending(self, gated_server):
        base_url = gated_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = self._make_model(client, gated_server["table_uri"])
            # Set up champion via the approval flow
            httpx.put(
                f"{base_url}/v1/names/team/model/aliases/champion",
                json={"artifact_id": artifact.artifact_id, "version": artifact.version},
                timeout=30.0,
            )
            client.approve_alias_change("team/model", "champion")

            resp = httpx.delete(f"{base_url}/v1/names/team/model/aliases/champion", timeout=30.0)
            assert resp.status_code == 202
            # Still resolvable until approved
            assert client.resolve_alias("team/model", "champion") is not None
            client.approve_alias_change("team/model", "champion")
            with pytest.raises(httpx.HTTPStatusError):
                client.resolve_alias("team/model", "champion")


class TestAliasIdempotenceOverHttp:
    """Re-running an idempotent promote does not refile approvals (D4)."""

    def test_rerequest_of_live_champion_is_unchanged(self, gated_server):
        base_url = gated_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = client.materialize(
                inputs=[gated_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                name="team/model",
            )
            artifact.to_table()

            # First request queues; approve applies it
            httpx.put(
                f"{base_url}/v1/names/team/model/aliases/champion",
                json={"artifact_id": artifact.artifact_id, "version": artifact.version},
                timeout=30.0,
            )
            client.approve_alias_change("team/model", "champion")

            # Identical re-request (the re-run promote cell): unchanged, no queue
            resp = httpx.put(
                f"{base_url}/v1/names/team/model/aliases/champion",
                json={"artifact_id": artifact.artifact_id, "version": artifact.version},
                timeout=30.0,
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "unchanged"
            assert client.list_pending_changes() == []

    def test_unprotected_set_reports_applied_then_unchanged(self, multi_file_server):
        base_url = multi_file_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                name="team/model",
            )
            artifact.to_table()

            first = client.set_alias(
                "team/model", "candidate", artifact.artifact_id, artifact.version
            )
            again = client.set_alias(
                "team/model", "candidate", artifact.artifact_id, artifact.version
            )
            assert first["status"] == "applied"
            assert again["status"] == "unchanged"


class TestRegistryAuthzPersonalMode:
    """The registry surface is personal-mode-gated today; auth hardening
    must not regress the single-user path (Vuln 1/2 fixes are forward-
    looking for service-mode exposure).
    """

    def _make(self, client, table_uri):
        artifact = client.materialize(
            inputs=[table_uri],
            transform={"executor": "scan@v1", "params": {}},
            name="team/model",
        )
        artifact.to_table()
        return artifact

    def test_audit_route_unscoped_without_principal(self, multi_file_server):
        base_url = multi_file_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = self._make(client, multi_file_server["table_uri"])
            client.set_tag(artifact.artifact_id, artifact.version, "k", "v")

        resp = httpx.get(f"{base_url}/v1/registry/audit?name=team/model", timeout=30.0)
        assert resp.status_code == 200
        actions = {e["action"] for e in resp.json()["entries"]}
        assert "name_set" in actions

    def test_approval_open_without_auth(self, gated_server):
        """Personal mode (auth_mode=none) has a single operator — no scope
        gate, self-approval allowed."""
        base_url = gated_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = self._make(client, gated_server["table_uri"])
            httpx.put(
                f"{base_url}/v1/names/team/model/aliases/champion",
                json={"artifact_id": artifact.artifact_id, "version": artifact.version},
                timeout=30.0,
            )
            result = client.approve_alias_change("team/model", "champion")
            assert result["status"] == "approved"
            assert client.resolve_alias("team/model", "champion")["version"] == artifact.version


class TestRegistrySummary:
    """GET /v1/registry/summary assembles the names-table data (each name with
    its aliases + the version's tags) — the dashboard's registry-state source."""

    def test_summary_lists_names_with_aliases_and_tags(self, multi_file_server):
        base_url = multi_file_server["base_url"]
        with StrataClient(base_url=base_url) as client:
            artifact = client.materialize(
                inputs=[multi_file_server["table_uri"]],
                transform={"executor": "scan@v1", "params": {}},
                name="team/model",
            )
            artifact.to_table()
            client.set_tag(artifact.artifact_id, artifact.version, "mae", "1.98")
            client.set_alias("team/model", "candidate", artifact.artifact_id, artifact.version)

        resp = httpx.get(f"{base_url}/v1/registry/summary", timeout=30.0)
        assert resp.status_code == 200
        names = {n["name"]: n for n in resp.json()["names"]}
        assert "team/model" in names
        row = names["team/model"]
        assert row["version"] == artifact.version
        assert row["aliases"] == {"candidate": artifact.version}
        assert row["tags"]["mae"] == "1.98"
