"""Tests for build observability features.

Tests cover:
- BuildMetricsCollector for Prometheus metrics
- BuildContext for structured logging
- Build logs storage in BuildStore
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from strata.logging import BuildContext, get_request_context
from strata.transforms.build_metrics import (
    BuildMetricsCollector,
    get_build_metrics,
    init_build_metrics,
    reset_build_metrics,
)
from strata.transforms.build_store import BuildStore
from tests.conftest import seed_build_targets


class TestBuildMetricsCollector:
    """Tests for BuildMetricsCollector."""

    def test_record_started(self):
        """Test recording build start events."""
        collector = BuildMetricsCollector()

        collector.record_started(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="duckdb_sql@v1",
            queue_wait_ms=42.5,
        )

        stats = collector.get_stats()
        assert stats["builds_started"] == 1
        assert stats["builds_in_flight"] == 1
        assert stats["avg_queue_wait_ms"] == 42.5

    def test_record_succeeded(self):
        """Test recording build success events."""
        collector = BuildMetricsCollector()

        # First record start
        collector.record_started(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="duckdb_sql@v1",
        )

        # Then record success
        collector.record_succeeded(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="duckdb_sql@v1",
            duration_ms=100.0,
            bytes_in=1000,
            bytes_out=500,
        )

        stats = collector.get_stats()
        assert stats["builds_succeeded"] == 1
        assert stats["builds_in_flight"] == 0
        assert stats["total_bytes_in"] == 1000
        assert stats["total_bytes_out"] == 500

    def test_record_failed(self):
        """Test recording build failure events."""
        collector = BuildMetricsCollector()

        collector.record_started(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="duckdb_sql@v1",
        )

        collector.record_failed(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="duckdb_sql@v1",
            duration_ms=50.0,
            error_code="TimeoutError",
        )

        stats = collector.get_stats()
        assert stats["builds_failed"] == 1
        assert stats["builds_in_flight"] == 0
        assert stats["error_codes"]["TimeoutError"] == 1

    def test_record_cancelled(self):
        """Test recording build cancellation events."""
        collector = BuildMetricsCollector()

        collector.record_started(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="duckdb_sql@v1",
        )

        collector.record_cancelled(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="duckdb_sql@v1",
        )

        stats = collector.get_stats()
        assert stats["builds_cancelled"] == 1
        assert stats["builds_in_flight"] == 0

    def test_per_transform_stats(self):
        """Test per-transform metrics tracking."""
        collector = BuildMetricsCollector()

        for i in range(3):
            collector.record_started(
                build_id=f"build-{i}",
                tenant_id="acme",
                transform_ref="duckdb_sql@v1",
            )
            collector.record_succeeded(
                build_id=f"build-{i}",
                tenant_id="acme",
                transform_ref="duckdb_sql@v1",
                duration_ms=100.0,
                bytes_in=1000,
                bytes_out=500,
            )

        transform_stats = collector.get_transform_stats("duckdb_sql@v1")
        assert transform_stats is not None
        assert transform_stats["started"] == 3
        assert transform_stats["succeeded"] == 3
        assert transform_stats["total_bytes_out"] == 1500

    def test_per_tenant_stats(self):
        """Test per-tenant metrics tracking."""
        collector = BuildMetricsCollector()

        # Tenant 1: 2 builds
        for i in range(2):
            collector.record_started(
                build_id=f"t1-{i}",
                tenant_id="tenant1",
                transform_ref="sql@v1",
            )
            collector.record_succeeded(
                build_id=f"t1-{i}",
                tenant_id="tenant1",
                transform_ref="sql@v1",
                duration_ms=100.0,
                bytes_in=1000,
                bytes_out=500,
            )

        # Tenant 2: 1 build
        collector.record_started(
            build_id="t2-0",
            tenant_id="tenant2",
            transform_ref="sql@v1",
        )

        tenant1_stats = collector.get_tenant_stats("tenant1")
        tenant2_stats = collector.get_tenant_stats("tenant2")
        assert tenant1_stats is not None
        assert tenant2_stats is not None

        assert tenant1_stats["started"] == 2
        assert tenant1_stats["succeeded"] == 2
        assert tenant2_stats["started"] == 1
        assert tenant2_stats["succeeded"] == 0

    def test_prometheus_metrics_format(self):
        """Test Prometheus text exposition format."""
        collector = BuildMetricsCollector()

        collector.record_started(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="sql@v1",
        )
        collector.record_succeeded(
            build_id="build-1",
            tenant_id="acme",
            transform_ref="sql@v1",
            duration_ms=100.0,
            bytes_in=1000,
            bytes_out=500,
        )

        prom_metrics = collector.get_prometheus_metrics()

        # Check required metrics are present
        assert "strata_builds_started_total 1" in prom_metrics
        assert "strata_builds_succeeded_total 1" in prom_metrics
        assert "strata_builds_in_flight 0" in prom_metrics
        assert "strata_builds_bytes_in_total 1000" in prom_metrics
        assert "strata_builds_bytes_out_total 500" in prom_metrics

        # Check per-transform metrics
        assert 'strata_build_transform_started_total{transform="sql@v1"} 1' in prom_metrics

    def test_duration_percentiles(self):
        """Test duration percentile calculations."""
        collector = BuildMetricsCollector()

        # Record 10 builds with increasing durations
        for i in range(10):
            duration = (i + 1) * 10.0  # 10, 20, 30, ..., 100
            collector.record_started(
                build_id=f"build-{i}",
                tenant_id="acme",
                transform_ref="sql@v1",
            )
            collector.record_succeeded(
                build_id=f"build-{i}",
                tenant_id="acme",
                transform_ref="sql@v1",
                duration_ms=duration,
                bytes_in=0,
                bytes_out=0,
            )

        pcts = collector.get_duration_percentiles()
        assert pcts["p50_ms"] is not None
        assert pcts["p95_ms"] is not None
        assert pcts["p99_ms"] is not None

    def test_singleton_management(self):
        """Test module-level singleton get/set/reset."""
        reset_build_metrics()
        assert get_build_metrics() is None

        metrics = init_build_metrics()
        assert get_build_metrics() is metrics

        reset_build_metrics()
        assert get_build_metrics() is None


class TestBuildContext:
    """Tests for BuildContext structured logging."""

    def test_build_context_sets_values(self):
        """Test BuildContext sets context values."""
        with BuildContext(
            build_id="build-123",
            tenant_id="acme",
            transform_ref="sql@v1",
            provenance_hash="abc123",
        ):
            ctx = get_request_context()
            assert ctx["build_id"] == "build-123"
            assert ctx["tenant_id"] == "acme"
            assert ctx["transform_ref"] == "sql@v1"
            assert ctx["provenance_hash"] == "abc123"

    def test_build_context_clears_on_exit(self):
        """Test BuildContext clears values after exit."""
        with BuildContext(build_id="build-123"):
            pass

        ctx = get_request_context()
        assert "build_id" not in ctx

    def test_build_context_with_extra_kwargs(self):
        """Test BuildContext with additional keyword arguments."""
        with BuildContext(
            build_id="build-123",
            custom_key="custom_value",
        ):
            ctx = get_request_context()
            assert ctx["build_id"] == "build-123"
            assert ctx["custom_key"] == "custom_value"

    def test_build_context_optional_fields(self):
        """Test BuildContext with only required field."""
        with BuildContext(build_id="build-123"):
            ctx = get_request_context()
            assert ctx["build_id"] == "build-123"
            assert "tenant_id" not in ctx
            assert "transform_ref" not in ctx


class TestBuildStoreLogs:
    """Tests for build logs storage in BuildStore."""

    @pytest.fixture
    def build_store(self, tmp_path: Path) -> BuildStore:
        """Create a temporary build store over a database that also holds
        artifacts, so ``artifact_builds``' foreign key has something to
        resolve against."""
        seed_build_targets(tmp_path)
        return BuildStore(tmp_path / "artifacts.sqlite")

    def test_complete_build_with_logs(self, build_store: BuildStore):
        """Test completing a build with logs."""
        # Create a build
        build = build_store.create_build(
            build_id="build-123",
            artifact_id="artifact-1",
            version=1,
            executor_ref="sql@v1",
        )
        assert build.logs is None

        # Start the build
        build_store.claim_build("build-123", "runner-1")

        # Complete with logs
        build_store.complete_build(
            build_id="build-123",
            output_byte_count=1000,
            logs="[INFO] Transform completed successfully\n[DEBUG] Rows: 42",
        )

        # Verify logs are stored
        build = build_store.get_build("build-123")
        assert build is not None
        assert build.state == "ready"
        assert build.logs == "[INFO] Transform completed successfully\n[DEBUG] Rows: 42"

    def test_fail_build_with_logs(self, build_store: BuildStore):
        """Test failing a build with logs."""
        # Create and claim a build
        build_store.create_build(
            build_id="build-456",
            artifact_id="artifact-2",
            version=1,
            executor_ref="sql@v1",
        )
        build_store.claim_build("build-456", "runner-1")

        # Fail with logs
        build_store.fail_build(
            build_id="build-456",
            error_message="Syntax error",
            error_code="SyntaxError",
            logs="[ERROR] Invalid SQL at line 5\n[ERROR] Unexpected token 'FROM'",
        )

        # Verify logs are stored
        build = build_store.get_build("build-456")
        assert build is not None
        assert build.state == "failed"
        assert build.logs is not None
        assert build.error_message is not None
        assert "Invalid SQL" in build.logs
        assert build.error_message == "Syntax error"

    def test_complete_build_without_logs(self, build_store: BuildStore):
        """Test completing a build without logs."""
        build_store.create_build(
            build_id="build-789",
            artifact_id="artifact-3",
            version=1,
            executor_ref="sql@v1",
        )
        build_store.claim_build("build-789", "runner-1")

        build_store.complete_build(
            build_id="build-789",
            output_byte_count=500,
        )

        build = build_store.get_build("build-789")
        assert build is not None
        assert build.logs is None

    def test_logs_column_migration(self, tmp_path: Path):
        """Test that logs column is added via migration."""
        db_path = tmp_path / "migration_test.sqlite"

        # Create an old-style database without logs column
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE artifact_builds (
                build_id TEXT PRIMARY KEY,
                artifact_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                state TEXT NOT NULL DEFAULT 'pending',
                executor_ref TEXT NOT NULL,
                executor_url TEXT,
                tenant_id TEXT,
                principal_id TEXT,
                created_at REAL NOT NULL,
                started_at REAL,
                completed_at REAL,
                error_message TEXT,
                error_code TEXT,
                input_byte_count INTEGER,
                output_byte_count INTEGER,
                lease_owner TEXT,
                lease_expires_at REAL,
                input_uris TEXT,
                params TEXT,
                name TEXT
            )
            """
        )
        conn.commit()
        conn.close()

        # Initialize BuildStore (should trigger migration)
        store = BuildStore(db_path)

        # Verify we can create a build and store logs
        store.create_build(
            build_id="build-migrated",
            artifact_id="artifact-1",
            version=1,
            executor_ref="sql@v1",
        )
        store.claim_build("build-migrated", "runner-1")
        store.complete_build(
            build_id="build-migrated",
            output_byte_count=100,
            logs="Migration test logs",
        )

        build = store.get_build("build-migrated")
        assert build is not None
        assert build.logs == "Migration test logs"


class TestBuildMetricsIntegration:
    """Integration tests for build metrics with runner-like usage."""

    def test_full_build_lifecycle_metrics(self):
        """Test metrics for a complete build lifecycle."""
        reset_build_metrics()
        collector = init_build_metrics()

        # Simulate build lifecycle
        collector.record_started(
            build_id="build-full",
            tenant_id="integration",
            transform_ref="pandas@v1",
            queue_wait_ms=25.0,
        )

        # Simulate some work
        collector.record_succeeded(
            build_id="build-full",
            tenant_id="integration",
            transform_ref="pandas@v1",
            duration_ms=500.0,
            bytes_in=10000,
            bytes_out=5000,
        )

        # Check all metrics are correct
        stats = collector.get_stats()
        assert stats["builds_started"] == 1
        assert stats["builds_succeeded"] == 1
        assert stats["builds_in_flight"] == 0
        assert stats["total_bytes_in"] == 10000
        assert stats["total_bytes_out"] == 5000

        # Check Prometheus format includes all data
        prom = collector.get_prometheus_metrics()
        assert "strata_builds_started_total 1" in prom
        assert "strata_builds_succeeded_total 1" in prom
        assert "strata_builds_queue_wait_total_ms" in prom

        reset_build_metrics()

    def test_multiple_concurrent_builds_metrics(self):
        """Test metrics tracking for multiple concurrent builds."""
        collector = BuildMetricsCollector()

        # Start 3 builds
        for i in range(3):
            collector.record_started(
                build_id=f"concurrent-{i}",
                tenant_id=f"tenant-{i}",
                transform_ref="sql@v1",
            )

        assert collector.get_stats()["builds_in_flight"] == 3

        # Complete 2, fail 1
        collector.record_succeeded(
            build_id="concurrent-0",
            tenant_id="tenant-0",
            transform_ref="sql@v1",
            duration_ms=100.0,
            bytes_in=1000,
            bytes_out=500,
        )
        collector.record_succeeded(
            build_id="concurrent-1",
            tenant_id="tenant-1",
            transform_ref="sql@v1",
            duration_ms=150.0,
            bytes_in=2000,
            bytes_out=1000,
        )
        collector.record_failed(
            build_id="concurrent-2",
            tenant_id="tenant-2",
            transform_ref="sql@v1",
            duration_ms=50.0,
            error_code="RuntimeError",
        )

        stats = collector.get_stats()
        assert stats["builds_in_flight"] == 0
        assert stats["builds_succeeded"] == 2
        assert stats["builds_failed"] == 1
        assert stats["total_bytes_in"] == 3000
        assert stats["total_bytes_out"] == 1500
