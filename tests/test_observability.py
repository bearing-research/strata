"""Tests for observability modules: gc_tracker, slow_ops, pool_metrics, memory_profiler."""

import gc
import time
from concurrent.futures import ThreadPoolExecutor

import pytest


class TestGCTracker:
    """Tests for GC pause duration tracking."""

    def test_tracker_install_and_uninstall(self):
        """Test that tracker can be installed and uninstalled."""
        from strata.gc_tracker import GCTracker

        tracker = GCTracker()
        assert not tracker._installed

        tracker.install()
        assert tracker._installed
        assert tracker._gc_callback in gc.callbacks

        tracker.uninstall()
        assert not tracker._installed
        assert tracker._gc_callback not in gc.callbacks

    def test_tracker_install_idempotent(self):
        """Test that calling install multiple times is safe."""
        from strata.gc_tracker import GCTracker

        tracker = GCTracker()
        tracker.install()
        tracker.install()  # Should not raise or add duplicate
        assert gc.callbacks.count(tracker._gc_callback) == 1
        tracker.uninstall()

    def test_tracker_records_gc_pauses(self):
        """Test that tracker records GC pause durations."""
        from strata.gc_tracker import GCTracker

        tracker = GCTracker()
        tracker.install()

        try:
            # Force a GC collection
            gc.collect(0)  # Gen 0
            gc.collect(1)  # Gen 1
            gc.collect(2)  # Gen 2

            stats = tracker.get_stats()

            # Should have recorded at least one pause
            assert stats.total_pauses >= 1
            assert stats.total_pause_ms >= 0

            # At least gen0 should have run
            assert stats.gen0_count >= 1 or stats.gen1_count >= 1 or stats.gen2_count >= 1
        finally:
            tracker.uninstall()

    def test_gc_callback_is_nonblocking_when_lock_held(self):
        """Regression: the GC callback must not block on the tracker lock.

        The callback fires *during* a GC, which an allocation under ``_lock``
        (e.g. ``get_stats`` building its result) can trigger on the same thread.
        With a non-reentrant lock, a blocking acquire would self-deadlock — the
        intermittent py3.12/macOS ``/metrics`` hang. So with the lock already
        held, the callback must return promptly (skipping the sample), not hang.
        """
        from strata.gc_tracker import GCTracker

        tracker = GCTracker()
        tracker._gc_callback("start", {"generation": 0})

        # Hold the lock, as get_stats/reset do while allocating their result.
        tracker._lock.acquire()
        try:
            done: list[bool] = []

            def _fire():
                tracker._gc_callback("stop", {"generation": 0})
                done.append(True)

            with ThreadPoolExecutor(max_workers=1) as ex:
                # Run from another thread so a true deadlock would hang the
                # future; a blocking acquire on the held lock would never return.
                ex.submit(_fire).result(timeout=5)

            assert done == [True]
            # The sample was dropped (lock was contended), not recorded.
            assert tracker._total_pauses == 0
        finally:
            tracker._lock.release()

    def test_tracker_get_recent_pauses(self):
        """Test getting recent pause list."""
        from strata.gc_tracker import GCTracker

        tracker = GCTracker(max_recent=10)
        tracker.install()

        try:
            # Force some collections
            for _ in range(5):
                gc.collect(0)

            pauses = tracker.get_recent_pauses(limit=3)
            assert isinstance(pauses, list)
            if len(pauses) > 0:
                pause = pauses[0]
                assert "timestamp" in pause
                assert "generation" in pause
                assert "duration_ms" in pause
        finally:
            tracker.uninstall()

    def test_tracker_reset(self):
        """Test resetting tracker statistics."""
        from strata.gc_tracker import GCTracker

        tracker = GCTracker()
        tracker.install()

        try:
            gc.collect(0)
            stats = tracker.get_stats()
            assert stats.total_pauses >= 1

            tracker.reset()
            stats = tracker.get_stats()
            assert stats.total_pauses == 0
            assert stats.total_pause_ms == 0.0
        finally:
            tracker.uninstall()

    def test_gc_stats_to_dict(self):
        """Test GCStats serialization."""
        from strata.gc_tracker import GCPause, GCStats

        stats = GCStats(
            recent_pauses=[GCPause(time.time(), 0, 1.5) for _ in range(15)],
            gen0_count=10,
            gen0_total_ms=15.0,
            gen0_max_ms=3.0,
            gen1_count=5,
            gen1_total_ms=10.0,
            gen1_max_ms=4.0,
            gen2_count=1,
            gen2_total_ms=5.0,
            gen2_max_ms=5.0,
            total_pauses=16,
            total_pause_ms=30.0,
            max_pause_ms=5.0,
        )

        d = stats.to_dict()
        assert d["total_pauses"] == 16
        assert d["max_pause_ms"] == 5.0
        assert d["gen0"]["count"] == 10
        assert d["gen0"]["avg_ms"] == 1.5
        assert "recent" in d  # Should have percentiles with 15 samples
        assert "p50_ms" in d["recent"]

    def test_global_functions(self):
        """Test module-level convenience functions."""
        from strata.gc_tracker import (
            get_gc_stats,
            get_recent_gc_pauses,
            install_gc_tracker,
            reset_gc_stats,
        )

        tracker = install_gc_tracker()
        assert tracker is not None

        gc.collect(0)

        stats = get_gc_stats()
        assert isinstance(stats, dict)
        assert "total_pauses" in stats

        pauses = get_recent_gc_pauses(limit=5)
        assert isinstance(pauses, list)

        reset_gc_stats()
        stats = get_gc_stats()
        assert stats["total_pauses"] == 0


class TestSlowOps:
    """Tests for slow operation logging and latency histograms."""

    def test_latency_histogram_record(self):
        """Test recording latencies to histogram."""
        from strata.slow_ops import LatencyHistogram

        hist = LatencyHistogram()

        # Record various latencies
        hist.record("plan", 5)  # 0-10ms bucket
        hist.record("plan", 25)  # 10-50ms bucket
        hist.record("plan", 75)  # 50-100ms bucket
        hist.record("plan", 150)  # 100-250ms bucket

        stats = hist.get_histogram("plan")
        assert stats["count"] == 4
        assert stats["sum_ms"] == 255.0
        assert stats["avg_ms"] == 63.75
        assert stats["max_ms"] == 150
        assert stats["buckets"]["0-10ms"] == 1
        assert stats["buckets"]["10-50ms"] == 1
        assert stats["buckets"]["50-100ms"] == 1
        assert stats["buckets"]["100-250ms"] == 1

    def test_latency_histogram_empty_stage(self):
        """Test getting stats for a stage with no data."""
        from strata.slow_ops import LatencyHistogram

        hist = LatencyHistogram()
        stats = hist.get_histogram("nonexistent")
        assert stats["count"] == 0
        assert stats["sum_ms"] == 0.0

    def test_latency_histogram_all_buckets(self):
        """Test that all histogram buckets are populated correctly."""
        from strata.slow_ops import LatencyHistogram

        hist = LatencyHistogram()

        # Test each bucket boundary
        test_values = [
            (5, "0-10ms"),
            (25, "10-50ms"),
            (75, "50-100ms"),
            (175, "100-250ms"),
            (375, "250-500ms"),
            (750, "500-1s"),
            (2500, "1-5s"),
            (10000, "5s+"),
        ]

        for value, expected_bucket in test_values:
            hist.record("test", value)

        stats = hist.get_histogram("test")
        for value, expected_bucket in test_values:
            assert stats["buckets"][expected_bucket] >= 1, f"Bucket {expected_bucket} not populated"

    def test_slow_op_tracker_basic(self):
        """Test SlowOpTracker basic timing."""
        from strata.slow_ops import SlowOpTracker

        tracker = SlowOpTracker()
        tracker.start(scan_id="test-123", table_id="db.schema.table")

        with tracker.time_stage("plan"):
            time.sleep(0.01)  # 10ms

        timings = tracker.finish(bytes_streamed=1000, rows_streamed=100)

        assert timings.scan_id == "test-123"
        assert timings.table_id == "db.schema.table"
        assert timings.plan_ms >= 10  # At least 10ms
        assert timings.total_ms >= 10
        assert timings.bytes_streamed == 1000
        assert timings.rows_streamed == 100

    def test_slow_op_tracker_multiple_stages(self):
        """Test SlowOpTracker with multiple stages."""
        from strata.slow_ops import SlowOpTracker

        tracker = SlowOpTracker()
        tracker.start(scan_id="test", table_id="test.table")

        with tracker.time_stage("plan"):
            time.sleep(0.005)

        with tracker.time_stage("ttfb"):
            time.sleep(0.005)

        with tracker.time_stage("scan_open"):
            time.sleep(0.005)

        timings = tracker.finish()

        assert timings.plan_ms >= 5
        assert timings.ttfb_ms >= 5
        assert timings.scan_open_ms >= 5
        assert timings.total_ms >= 15

    def test_global_latency_functions(self):
        """Test module-level latency functions."""
        from strata.slow_ops import (
            get_latency_stats,
            record_latency,
            reset_latency_stats,
        )

        reset_latency_stats()

        record_latency("plan", 50)
        record_latency("plan", 150)
        record_latency("ttfb", 100)

        stats = get_latency_stats()
        assert "plan" in stats
        assert stats["plan"]["count"] == 2
        assert "ttfb" in stats
        assert stats["ttfb"]["count"] == 1

        reset_latency_stats()
        stats = get_latency_stats()
        assert len(stats) == 0

    def test_get_latency_percentiles(self):
        """Test percentile estimation from histogram."""
        from strata.slow_ops import (
            get_latency_percentiles,
            record_latency,
            reset_latency_stats,
        )

        reset_latency_stats()

        # Record enough samples for percentile estimation
        for i in range(100):
            record_latency("test", i)  # 0-99ms

        percentiles = get_latency_percentiles("test")
        # Keys are p50_ms, p95_ms, p99_ms (with _ms suffix)
        assert "p50_ms" in percentiles
        assert "p95_ms" in percentiles
        assert "p99_ms" in percentiles

        # p50 should be around 50ms (in 50-100ms bucket)
        assert percentiles["p50_ms"] <= 100


class TestPoolMetrics:
    """Tests for thread pool metrics tracking."""

    def test_pool_tracker_register(self):
        """Test registering a thread pool."""
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="test")

        tracker.register_pool("test", executor)

        stats = tracker.get_pool_stats("test")
        assert stats is not None
        assert stats.name == "test"
        assert stats.max_workers == 4

        executor.shutdown(wait=False)

    def test_pool_tracker_utilization(self):
        """Utilization reflects work the tracker is carrying.

        Callers wrap the handoff to the pool in ``tracker.track(...)``; a bare
        ``executor.submit`` is invisible to the tracker by design. Utilization
        used to be read off ``executor._threads`` instead, which counts every
        thread the pool ever created and so never fell back down.
        """
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=4)
        tracker.register_pool("test", executor)

        with tracker.track("test"), tracker.track("test"):
            stats = tracker.get_pool_stats("test")
            assert stats is not None
            assert stats.active_workers == 2
            assert stats.utilization_pct == 50.0

        executor.shutdown(wait=True)

    def test_pool_tracker_queue_depth(self):
        """Test tracking queue depth when pool is saturated."""
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=2)
        tracker.register_pool("test", executor)

        # Submit more work than workers
        futures = [executor.submit(time.sleep, 0.1) for _ in range(6)]

        # Check queue depth while running
        time.sleep(0.01)
        stats = tracker.get_pool_stats("test")
        assert stats is not None
        # Queue depth should be at least some (6 tasks - 2 workers = 4 queued)
        # But timing is tricky, so just check it's non-negative
        assert stats.queue_depth >= 0

        for f in futures:
            f.result()

        executor.shutdown(wait=True)

    def test_pool_tracker_get_all_stats(self):
        """Test getting stats for all registered pools."""
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor1 = ThreadPoolExecutor(max_workers=4)
        executor2 = ThreadPoolExecutor(max_workers=8)

        tracker.register_pool("pool1", executor1)
        tracker.register_pool("pool2", executor2)

        all_stats = tracker.get_all_stats()
        assert "pool1" in all_stats
        assert "pool2" in all_stats
        assert all_stats["pool1"].max_workers == 4
        assert all_stats["pool2"].max_workers == 8

        executor1.shutdown(wait=False)
        executor2.shutdown(wait=False)

    def test_pool_tracker_get_summary(self):
        """Test getting summary for metrics endpoint."""
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=4)
        tracker.register_pool("test", executor)

        summary = tracker.get_summary()
        assert "thread_pools" in summary
        assert "total_pools" in summary
        assert summary["total_pools"] == 1
        assert "test" in summary["thread_pools"]

        executor.shutdown(wait=False)

    def test_pool_stats_to_dict(self):
        """Test ThreadPoolStats serialization."""
        from strata.pool_metrics import ThreadPoolStats

        stats = ThreadPoolStats(
            name="test",
            max_workers=8,
            active_workers=4,
            queue_depth=10,
            utilization_pct=50.0,
            tasks_completed=100,
            tasks_submitted=110,
        )

        d = stats.to_dict()
        assert d["name"] == "test"
        assert d["max_workers"] == 8
        assert d["active_workers"] == 4
        assert d["queue_depth"] == 10
        assert d["utilization_pct"] == 50.0

    def test_connection_metrics_tracking(self):
        """Test HTTP connection metrics."""
        from strata.pool_metrics import ConnectionMetrics

        metrics = ConnectionMetrics()

        # Simulate requests
        metrics.request_started(has_keepalive=True)
        metrics.request_started(has_keepalive=True)
        metrics.request_started(has_keepalive=False)

        stats = metrics.get_stats()
        assert stats["active_requests"] == 3
        assert stats["total_requests"] == 3
        assert stats["max_concurrent_requests"] == 3
        assert stats["keepalive_pct"] == pytest.approx(66.7, rel=0.1)

        metrics.request_completed()
        metrics.request_completed()

        stats = metrics.get_stats()
        assert stats["active_requests"] == 1
        assert stats["total_requests"] == 3  # Total doesn't decrease

    def test_connection_metrics_reset(self):
        """Test resetting connection metrics."""
        from strata.pool_metrics import ConnectionMetrics

        metrics = ConnectionMetrics()
        metrics.request_started()
        metrics.request_started()

        metrics.reset()

        stats = metrics.get_stats()
        assert stats["active_requests"] == 0
        assert stats["total_requests"] == 0

    def test_global_pool_functions(self):
        """Test module-level convenience functions."""
        from strata.pool_metrics import (
            get_connection_metrics,
            get_pool_tracker,
            reset_metrics,
        )

        reset_metrics()

        tracker = get_pool_tracker()
        assert tracker is not None

        conn = get_connection_metrics()
        assert conn is not None

        # Should return same instances
        assert get_pool_tracker() is tracker
        assert get_connection_metrics() is conn


class TestMemoryProfiler:
    """Tests for memory profiling utilities."""

    def test_memory_snapshot(self):
        """Test getting memory snapshot."""
        from strata.memory_profiler import get_memory_snapshot

        snapshot = get_memory_snapshot()

        assert snapshot.arrow_bytes_allocated >= 0
        assert snapshot.arrow_max_memory >= 0
        assert snapshot.arrow_pool_backend in ["default", "system", "jemalloc", "mimalloc"]
        assert snapshot.python_gc_tracked >= 0
        assert len(snapshot.python_gc_objects_by_gen) == 3

    def test_memory_snapshot_to_dict(self):
        """Test MemorySnapshot serialization."""
        from strata.memory_profiler import get_memory_snapshot

        snapshot = get_memory_snapshot()
        d = snapshot.to_dict()

        assert "arrow" in d
        assert "python" in d
        assert "process" in d

        assert "bytes_allocated" in d["arrow"]
        assert "allocated_mb" in d["arrow"]
        assert "pool_backend" in d["arrow"]

        assert "gc_tracked_objects" in d["python"]
        assert "gc_objects_by_gen" in d["python"]

    def test_get_arrow_allocations(self):
        """Test getting Arrow allocation details."""
        from strata.memory_profiler import get_arrow_allocations

        allocs = get_arrow_allocations()

        assert "default_pool" in allocs
        assert "backend" in allocs["default_pool"]
        assert "bytes_allocated" in allocs["default_pool"]
        assert "available_pools" in allocs

    def test_get_python_memory_stats(self):
        """Test getting Python memory statistics."""
        from strata.memory_profiler import get_python_memory_stats

        stats = get_python_memory_stats()

        assert "gc_thresholds" in stats
        assert "gc_stats" in stats
        assert "gc_is_enabled" in stats
        assert "top_object_types" in stats
        assert "total_objects" in stats

        assert len(stats["gc_stats"]) == 3  # 3 generations
        assert len(stats["top_object_types"]) <= 20

    def test_get_detailed_memory_report(self):
        """Test getting detailed memory report."""
        from strata.memory_profiler import get_detailed_memory_report

        report = get_detailed_memory_report()

        assert "snapshot" in report
        assert "arrow_details" in report
        assert "python_details" in report
        assert "recommendations" in report

        assert isinstance(report["recommendations"], list)
        assert len(report["recommendations"]) >= 1

    def test_memory_recommendations_healthy(self):
        """Test that healthy memory state gets appropriate recommendation."""
        from strata.memory_profiler import MemorySnapshot, _get_memory_recommendations

        # Create a healthy snapshot
        snapshot = MemorySnapshot(
            arrow_bytes_allocated=100 * 1024 * 1024,  # 100MB
            arrow_max_memory=150 * 1024 * 1024,  # 150MB
            arrow_pool_backend="mimalloc",
            python_gc_tracked=100,
            python_gc_objects_by_gen=[1000, 500, 200],
            process_rss_bytes=500 * 1024 * 1024,  # 500MB
            process_vms_bytes=1024 * 1024 * 1024,  # 1GB
        )

        python_stats = {"total_objects": 50000}

        recs = _get_memory_recommendations(snapshot, python_stats)
        assert "healthy" in recs[0].lower()

    def test_memory_recommendations_high_arrow(self):
        """Test recommendation for high Arrow memory."""
        from strata.memory_profiler import MemorySnapshot, _get_memory_recommendations

        # Create snapshot with high Arrow allocation
        snapshot = MemorySnapshot(
            arrow_bytes_allocated=2 * 1024**3,  # 2GB
            arrow_max_memory=2 * 1024**3,
            arrow_pool_backend="mimalloc",
            python_gc_tracked=100,
            python_gc_objects_by_gen=[1000, 500, 200],
            process_rss_bytes=3 * 1024**3,
            process_vms_bytes=4 * 1024**3,
        )

        python_stats = {"total_objects": 50000}

        recs = _get_memory_recommendations(snapshot, python_stats)
        assert any("arrow" in r.lower() for r in recs)


class TestDebugEndpoints:
    """Integration tests for debug endpoints in server."""

    @pytest.fixture
    def client(self, tmp_path):
        """Create test client with initialized server state."""

        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.server import ServerState, app
        from strata.slow_ops import reset_latency_stats

        # Reset global state
        reset_metrics()
        reset_latency_stats()

        config = StrataConfig(cache_dir=tmp_path)
        server_module._state = ServerState(config)

        async def run_client():
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                yield client

        # Return a simple wrapper that runs async code
        return server_module, app

    @pytest.mark.asyncio
    async def test_debug_pools_endpoint(self, tmp_path):
        """Test /v1/debug/pools endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.server import ServerState, app

        reset_metrics()
        config = StrataConfig(cache_dir=tmp_path)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/debug/pools")
                assert response.status_code == 200

                data = response.json()
                assert "thread_pools" in data
                assert "planning" in data["thread_pools"]
                assert "fetch" in data["thread_pools"]
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_debug_connections_endpoint(self, tmp_path):
        """Test /v1/debug/connections endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.server import ServerState, app

        reset_metrics()
        config = StrataConfig(cache_dir=tmp_path)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/debug/connections")
                assert response.status_code == 200

                data = response.json()
                assert "active_requests" in data
                assert "total_requests" in data
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_debug_memory_endpoint(self, tmp_path):
        """Test /v1/debug/memory endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.server import ServerState, app

        reset_metrics()
        config = StrataConfig(cache_dir=tmp_path)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Test basic mode
                response = await client.get("/v1/debug/memory")
                assert response.status_code == 200

                data = response.json()
                assert "arrow" in data
                assert "python" in data
                assert "process" in data

                # Test detailed mode
                response = await client.get("/v1/debug/memory?detailed=true")
                assert response.status_code == 200

                data = response.json()
                assert "snapshot" in data
                assert "recommendations" in data
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_debug_latency_endpoint(self, tmp_path):
        """Test /v1/debug/latency endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.server import ServerState, app
        from strata.slow_ops import record_latency, reset_latency_stats

        reset_metrics()
        reset_latency_stats()
        config = StrataConfig(cache_dir=tmp_path)
        server_module._state = ServerState(config)

        # Record some latencies
        record_latency("plan", 50)
        record_latency("ttfb", 100)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/debug/latency")
                assert response.status_code == 200

                data = response.json()
                assert "histograms" in data
                assert "plan" in data["histograms"]
                assert "ttfb" in data["histograms"]
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_debug_gc_pauses_endpoint(self, tmp_path):
        """Test /v1/debug/gc/pauses endpoint."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.gc_tracker import install_gc_tracker
        from strata.pool_metrics import reset_metrics
        from strata.server import ServerState, app

        reset_metrics()
        install_gc_tracker()
        gc.collect()  # Generate some data

        config = StrataConfig(cache_dir=tmp_path)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/v1/debug/gc/pauses")
                assert response.status_code == 200

                data = response.json()
                assert "pauses" in data
                assert "stats" in data
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None

    @pytest.mark.asyncio
    async def test_metrics_includes_pools_and_connections(self, tmp_path):
        """Test that /metrics includes thread pool and connection data."""
        from httpx import ASGITransport, AsyncClient

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.pool_metrics import reset_metrics
        from strata.server import ServerState, app

        reset_metrics()
        config = StrataConfig(cache_dir=tmp_path)
        server_module._state = ServerState(config)

        try:
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/metrics")
                assert response.status_code == 200

                data = response.json()
                assert "thread_pools" in data
                assert "connections" in data
                assert "planning" in data["thread_pools"]
                assert "fetch" in data["thread_pools"]
        finally:
            server_module._state._planning_executor.shutdown(wait=False)
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state = None


class TestPercentilesUseNearestRank:
    """The percentile target is a 1-based sample index, so it must round UP
    and never reach 0.

    ``int(total * pct)`` truncated instead, and a target of 0 is satisfied by
    the very first bucket before any count is added — so a stage with a single
    sample reported the FASTEST bucket at every percentile. One 9-second scan
    came back as ``p99 = 5ms``: exactly backwards for the operator reading it
    to decide whether a stage is slow.
    """

    def test_a_lone_slow_sample_is_not_reported_as_the_fastest_bucket(self):
        from strata.slow_ops import LatencyHistogram

        hist = LatencyHistogram()
        hist.record("scan", 9000.0)

        assert hist.get_percentiles("scan") == {
            "p50_ms": 7500,
            "p95_ms": 7500,
            "p99_ms": 7500,
        }

    def test_a_lone_fast_sample_is_still_fast(self):
        from strata.slow_ops import LatencyHistogram

        hist = LatencyHistogram()
        hist.record("scan", 3.0)

        assert hist.get_percentiles("scan") == {"p50_ms": 5, "p95_ms": 5, "p99_ms": 5}

    def test_a_split_distribution_separates_p50_from_the_tail(self):
        from strata.slow_ops import LatencyHistogram

        hist = LatencyHistogram()
        for _ in range(50):
            hist.record("scan", 1.0)
        for _ in range(50):
            hist.record("scan", 9000.0)

        stats = hist.get_percentiles("scan")
        assert stats["p50_ms"] == 5
        assert stats["p95_ms"] == 7500
        assert stats["p99_ms"] == 7500

    def test_no_samples_still_reports_zeros(self):
        from strata.slow_ops import LatencyHistogram

        hist = LatencyHistogram()
        assert hist.get_percentiles("scan") == {"p50_ms": 0, "p95_ms": 0, "p99_ms": 0}


class TestPoolUtilizationFallsBackToZero:
    """Utilization must describe now, not the busiest moment so far.

    ``active_workers`` was ``len([t for t in executor._threads if
    t.is_alive()])`` -- every thread the pool had ever created. Pool threads
    park on the work queue rather than exiting, so the count only ever rose.
    Once a pool had seen ``max_workers`` concurrent submissions it reported
    100% utilization for the rest of the process's life, and because
    ``check_thread_pools`` returns DEGRADED above 90% and the overall
    dependency status is the worst of all checks, one traffic burst latched
    /health/dependencies to degraded until the server restarted.
    """

    def test_utilization_returns_to_zero_when_work_finishes(self):
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=4)
        tracker.register_pool("test", executor)

        with (
            tracker.track("test"),
            tracker.track("test"),
            tracker.track("test"),
            tracker.track("test"),
        ):
            saturated = tracker.get_pool_stats("test")
            assert saturated.utilization_pct == 100.0

        idle = tracker.get_pool_stats("test")
        assert idle.active_workers == 0
        assert idle.utilization_pct == 0.0

        executor.shutdown(wait=True)

    def test_health_check_recovers_after_a_burst(self):
        from strata.health import HealthStatus, check_thread_pools
        from strata.pool_metrics import get_pool_tracker, reset_metrics

        reset_metrics()
        tracker = get_pool_tracker()
        planning = ThreadPoolExecutor(max_workers=2)
        fetch = ThreadPoolExecutor(max_workers=2)
        tracker.register_pool("planning", planning)
        tracker.register_pool("fetch", fetch)

        try:
            assert check_thread_pools(planning, fetch).status == HealthStatus.HEALTHY

            with (
                tracker.track("planning"),
                tracker.track("planning"),
                tracker.track("fetch"),
                tracker.track("fetch"),
            ):
                burst = check_thread_pools(planning, fetch)
                assert burst.status == HealthStatus.DEGRADED
                assert burst.details["overall_utilization"] == 100.0

            recovered = check_thread_pools(planning, fetch)
            assert recovered.status == HealthStatus.HEALTHY
            assert recovered.details["overall_utilization"] == 0.0
        finally:
            planning.shutdown(wait=False)
            fetch.shutdown(wait=False)
            reset_metrics()

    def test_in_flight_is_capped_so_utilization_stays_a_percentage(self):
        """Queued work counts as in-flight; ``queue_depth`` reports the backlog."""
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=2)
        tracker.register_pool("test", executor)

        with (
            tracker.track("test"),
            tracker.track("test"),
            tracker.track("test"),
            tracker.track("test"),
        ):
            stats = tracker.get_pool_stats("test")
            assert stats.active_workers == 2
            assert stats.utilization_pct == 100.0

        executor.shutdown(wait=True)

    def test_task_counters_are_no_longer_dead(self):
        """``record_task_submitted``/``_completed`` had no callers anywhere.

        Both counters shipped in the /v1/debug/pools payload as a constant 0.
        """
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=2)
        tracker.register_pool("test", executor)

        for _ in range(3):
            with tracker.track("test"):
                pass

        stats = tracker.get_pool_stats("test")
        assert stats.tasks_submitted == 3
        assert stats.tasks_completed == 3

        executor.shutdown(wait=True)

    def test_the_counter_is_released_when_the_body_raises(self):
        from strata.pool_metrics import PoolMetricsTracker

        tracker = PoolMetricsTracker()
        executor = ThreadPoolExecutor(max_workers=2)
        tracker.register_pool("test", executor)

        with pytest.raises(RuntimeError):
            with tracker.track("test"):
                raise RuntimeError("fetch failed")

        stats = tracker.get_pool_stats("test")
        assert stats.active_workers == 0
        assert stats.tasks_completed == 1

        executor.shutdown(wait=True)
