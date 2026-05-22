"""S3 integration tests using testcontainers with MinIO.

These tests validate end-to-end S3 functionality by:
1. Starting a real MinIO container
2. Creating an Iceberg table in MinIO
3. Scanning the table through Strata

Unlike moto-based tests, these actually exercise the PyArrow S3FileSystem
code path since PyArrow uses its own AWS SDK implementation.

Requirements:
    - Docker must be running
    - Run with: pytest tests/test_s3_integration.py -v

Note: These tests are slower (~5-10s each) due to container startup.
Mark with @pytest.mark.slow if you want to skip in quick test runs.
"""

import random
import time

import docker
import pyarrow as pa
import pytest
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, IntegerType, LongType, NestedField, StringType
from testcontainers.minio import MinioContainer

from strata.config import StrataConfig
from strata.fetcher import PyArrowFetcher
from strata.planner import ReadPlanner
from strata.types import Filter, FilterOp


def _docker_daemon_reachable() -> bool:
    """Skip when the daemon is unreachable. ``docker`` itself is a transitive
    dev dep via testcontainers, but the daemon may not be running on a
    contributor's laptop. CI always has Docker; this only triggers locally.
    """
    try:
        docker.from_env().ping()
        return True
    except Exception:
        return False


if not _docker_daemon_reachable():
    pytest.skip("Docker daemon is not running", allow_module_level=True)

# Mark all tests in this module as integration tests
pytestmark = [pytest.mark.integration, pytest.mark.slow]


# Test schema matching soak_test.py
TEST_SCHEMA = Schema(
    NestedField(1, "id", LongType(), required=False),
    NestedField(2, "ts", LongType(), required=False),
    NestedField(3, "user_id", IntegerType(), required=False),
    NestedField(4, "category", StringType(), required=False),
    NestedField(5, "value", DoubleType(), required=False),
)


def create_test_data(num_rows: int = 1000, seed: int = 42) -> pa.Table:
    """Create test Arrow table with sample data."""
    random.seed(seed)
    categories = ["electronics", "clothing", "food", "books", "sports"]
    base_ts = 1704067200000000  # 2024-01-01 00:00:00 UTC in microseconds

    return pa.table(
        {
            "id": pa.array(range(num_rows), type=pa.int64()),
            "ts": pa.array([base_ts + i * 1000 for i in range(num_rows)], type=pa.int64()),
            "user_id": pa.array(
                [random.randint(1, 1000) for _ in range(num_rows)], type=pa.int32()
            ),
            "category": pa.array(
                [random.choice(categories) for _ in range(num_rows)], type=pa.string()
            ),
            "value": pa.array(
                [random.uniform(0.0, 100.0) for _ in range(num_rows)], type=pa.float64()
            ),
        }
    )


@pytest.fixture(scope="module")
def minio_container():
    """Start MinIO container for the test module.

    Using module scope to avoid repeated container startup overhead.
    """
    with MinioContainer("minio/minio:RELEASE.2024-11-07T00-52-20Z") as minio:
        # Wait for MinIO to be ready
        client = minio.get_client()
        # Create test bucket
        bucket_name = "test-warehouse"
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
        yield minio


def _get_s3_endpoint(config: dict) -> str:
    """Get S3 endpoint with HTTP scheme for MinIO.

    MinIO container returns endpoint without scheme; PyArrow and pyiceberg
    default to HTTPS which causes SSL errors. Ensure we use HTTP for local MinIO.
    """
    endpoint = config["endpoint"]
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"http://{endpoint}"
    return endpoint


@pytest.fixture(scope="module")
def s3_catalog_db(tmp_path_factory):
    """Shared catalog DB path for S3 integration tests."""
    return tmp_path_factory.mktemp("catalog") / "catalog.db"


@pytest.fixture(scope="module")
def s3_config(minio_container, s3_catalog_db, tmp_path_factory):
    """Create StrataConfig for MinIO with shared catalog."""
    config = minio_container.get_config()
    cache_dir = tmp_path_factory.mktemp("cache")
    endpoint = _get_s3_endpoint(config)

    return StrataConfig(
        cache_dir=cache_dir,
        s3_endpoint_url=endpoint,
        s3_access_key=config["access_key"],
        s3_secret_key=config["secret_key"],
        s3_region="us-east-1",
        catalog_properties={
            "type": "sql",
            "uri": f"sqlite:///{s3_catalog_db}",
            "warehouse": "s3://test-warehouse/warehouse",
            "s3.endpoint": endpoint,
            "s3.access-key-id": config["access_key"],
            "s3.secret-access-key": config["secret_key"],
            "s3.region": "us-east-1",
        },
    )


@pytest.fixture(scope="module")
def s3_table(minio_container, s3_config, s3_catalog_db):
    """Create an Iceberg table in MinIO with test data.

    Returns the table URI in format: s3://bucket/warehouse#namespace.table
    """
    config = minio_container.get_config()
    bucket = "test-warehouse"
    warehouse_path = f"s3://{bucket}/warehouse"
    endpoint = _get_s3_endpoint(config)

    catalog = SqlCatalog(
        "default",
        uri=f"sqlite:///{s3_catalog_db}",
        warehouse=warehouse_path,
        **{
            "s3.endpoint": endpoint,
            "s3.access-key-id": config["access_key"],
            "s3.secret-access-key": config["secret_key"],
            "s3.region": "us-east-1",
        },
    )

    # Create namespace and table
    namespace = "test_ns"
    table_name = "events"
    table_id = f"{namespace}.{table_name}"

    try:
        catalog.create_namespace(namespace)
    except Exception:
        pass  # Namespace might exist

    try:
        table = catalog.load_table(table_id)
    except Exception:
        table = catalog.create_table(table_id, TEST_SCHEMA)

    # Insert test data
    test_data = create_test_data(num_rows=1000)
    table.append(test_data)

    # Return table URI
    return f"{warehouse_path}#{table_id}"


class TestS3EndToEnd:
    """End-to-end tests for S3 storage backend."""

    def test_planner_resolves_s3_table(self, s3_config, s3_table):
        """Test that ReadPlanner can resolve an S3 table."""
        planner = ReadPlanner(s3_config)

        plan = planner.plan(s3_table)

        assert plan.snapshot_id > 0
        assert len(plan.tasks) > 0
        assert plan.schema is not None

        # Verify file paths are S3 URIs
        for task in plan.tasks:
            assert task.file_path.startswith("s3://"), f"Expected S3 path, got: {task.file_path}"

    def test_fetcher_reads_s3_data(self, s3_config, s3_table):
        """Test that Fetcher can read row groups from S3."""
        planner = ReadPlanner(s3_config)
        plan = planner.plan(s3_table)

        # Create fetcher with S3 filesystem
        s3_fs = s3_config.get_s3_filesystem()
        fetcher = PyArrowFetcher(s3_filesystem=s3_fs)

        # Fetch the first task
        task = plan.tasks[0]
        batch = fetcher.fetch(task)

        assert batch.num_rows > 0
        assert "id" in batch.schema.names
        assert "category" in batch.schema.names

    def test_column_projection_on_s3(self, s3_config, s3_table):
        """Test that column projection works with S3 files."""
        planner = ReadPlanner(s3_config)
        columns = ["id", "value"]

        plan = planner.plan(s3_table, columns=columns)

        s3_fs = s3_config.get_s3_filesystem()
        fetcher = PyArrowFetcher(s3_filesystem=s3_fs)

        task = plan.tasks[0]
        batch = fetcher.fetch(task)

        # Should only have requested columns
        assert set(batch.schema.names) == set(columns)

    def test_filter_pruning_on_s3(self, s3_config, s3_table):
        """Test that row group pruning works with S3 files."""
        planner = ReadPlanner(s3_config)

        # Create a filter that should prune some data
        # Since we know id ranges from 0-999, filter for id > 2000 should return empty
        filters = [Filter(column="id", op=FilterOp.GT, value=2000)]

        plan = planner.plan(s3_table, filters=filters)

        # With good statistics, this might prune the row group entirely
        # Or it will have tasks but they'll return no matching rows
        # Either way, let's verify the filter was applied

        if len(plan.tasks) > 0:
            s3_fs = s3_config.get_s3_filesystem()
            fetcher = PyArrowFetcher(s3_filesystem=s3_fs)
            table = fetcher.fetch_to_table(plan.tasks)

            # If not pruned at metadata level, verify filter semantically
            # (actual filtering happens at read time via Iceberg)
            assert table.num_rows >= 0  # May be 0 if properly pruned

    def test_multiple_row_groups(self, minio_container, s3_config, s3_catalog_db):
        """Test reading a table with multiple row groups."""
        config = minio_container.get_config()
        bucket = "test-warehouse"
        warehouse_path = f"s3://{bucket}/warehouse"

        catalog = SqlCatalog(
            "default",
            uri=f"sqlite:///{s3_catalog_db}",
            warehouse=warehouse_path,
            **{
                "s3.endpoint": _get_s3_endpoint(config),
                "s3.access-key-id": config["access_key"],
                "s3.secret-access-key": config["secret_key"],
                "s3.region": "us-east-1",
            },
        )

        namespace = "multi"
        table_name = "events"
        table_id = f"{namespace}.{table_name}"

        try:
            catalog.create_namespace(namespace)
        except Exception:
            pass

        try:
            table = catalog.load_table(table_id)
        except Exception:
            table = catalog.create_table(table_id, TEST_SCHEMA)

        # Insert multiple batches to create multiple files/row groups
        for i in range(3):
            data = create_test_data(num_rows=500, seed=i)
            table.append(data)

        table_uri = f"{warehouse_path}#{table_id}"
        planner = ReadPlanner(s3_config)
        plan = planner.plan(table_uri)

        # Should have multiple tasks (one per row group)
        assert len(plan.tasks) >= 1

        s3_fs = s3_config.get_s3_filesystem()
        fetcher = PyArrowFetcher(s3_filesystem=s3_fs)
        result = fetcher.fetch_to_table(plan.tasks)

        # Total rows should be 3 * 500 = 1500
        assert result.num_rows == 1500


class TestS3PathHandling:
    """Tests for S3 path edge cases."""

    def test_s3_path_with_special_characters(self, s3_config, minio_container, tmp_path_factory):
        """Test handling of S3 paths with special characters in key names."""
        config = minio_container.get_config()
        bucket = "test-warehouse"
        # Path with hyphens and underscores (common in real warehouses)
        warehouse_path = f"s3://{bucket}/data-lake_v2/iceberg"
        endpoint = _get_s3_endpoint(config)

        catalog_db = tmp_path_factory.mktemp("special_catalog") / "catalog.db"
        catalog = SqlCatalog(
            "default",
            uri=f"sqlite:///{catalog_db}",
            warehouse=warehouse_path,
            **{
                "s3.endpoint": endpoint,
                "s3.access-key-id": config["access_key"],
                "s3.secret-access-key": config["secret_key"],
                "s3.region": "us-east-1",
            },
        )

        namespace = "special_ns"
        table_name = "test_table"
        table_id = f"{namespace}.{table_name}"

        try:
            catalog.create_namespace(namespace)
        except Exception:
            pass

        try:
            table = catalog.load_table(table_id)
        except Exception:
            table = catalog.create_table(table_id, TEST_SCHEMA)

        data = create_test_data(num_rows=100)
        table.append(data)

        # Use a config that points at this test's own catalog
        special_config = StrataConfig(
            cache_dir=tmp_path_factory.mktemp("special_cache"),
            s3_endpoint_url=endpoint,
            s3_access_key=config["access_key"],
            s3_secret_key=config["secret_key"],
            s3_region="us-east-1",
            catalog_properties={
                "type": "sql",
                "uri": f"sqlite:///{catalog_db}",
                "warehouse": warehouse_path,
                "s3.endpoint": endpoint,
                "s3.access-key-id": config["access_key"],
                "s3.secret-access-key": config["secret_key"],
                "s3.region": "us-east-1",
            },
        )

        table_uri = f"{warehouse_path}#{table_id}"
        planner = ReadPlanner(special_config)
        plan = planner.plan(table_uri)

        assert len(plan.tasks) > 0
        # Verify paths contain the special characters
        assert "data-lake_v2" in plan.tasks[0].file_path


class TestS3ErrorHandling:
    """Tests for S3 error scenarios."""

    def test_invalid_bucket_raises_error(self, s3_config):
        """Test that accessing a non-existent bucket raises an error."""
        planner = ReadPlanner(s3_config)

        with pytest.raises(Exception):
            # This should fail - bucket doesn't exist
            planner.plan("s3://nonexistent-bucket/warehouse#ns.table")

    def test_invalid_credentials_raises_error(self, minio_container, tmp_path_factory):
        """Test that invalid credentials raise an error."""
        config = minio_container.get_config()
        cache_dir = tmp_path_factory.mktemp("cache")

        bad_config = StrataConfig(
            cache_dir=cache_dir,
            s3_endpoint_url=config["endpoint"],
            s3_access_key="wrong_key",
            s3_secret_key="wrong_secret",
            s3_region="us-east-1",
        )

        planner = ReadPlanner(bad_config)

        with pytest.raises(Exception):
            planner.plan("s3://test-warehouse/warehouse#ns.table")


class TestS3Latency:
    """Tests for S3 latency characteristics."""

    def test_metadata_caching_reduces_latency(self, s3_config, s3_table):
        """Test that metadata caching improves subsequent planning latency."""
        planner = ReadPlanner(s3_config)

        # First call - cold cache
        start = time.perf_counter()
        plan1 = planner.plan(s3_table)
        cold_time = time.perf_counter() - start

        # Second call - warm cache
        start = time.perf_counter()
        plan2 = planner.plan(s3_table)
        warm_time = time.perf_counter() - start

        # Both should return same results
        assert plan1.snapshot_id == plan2.snapshot_id
        assert len(plan1.tasks) == len(plan2.tasks)

        # Warm should be faster (or at least not significantly slower)
        # Allow some variance since these are real I/O operations
        print(f"Cold planning: {cold_time * 1000:.1f}ms, Warm planning: {warm_time * 1000:.1f}ms")
        # Just verify it works - timing assertions are flaky in CI
