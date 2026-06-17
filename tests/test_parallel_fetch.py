"""Tests for parallel row group fetching."""

import threading

import uvicorn


class TestFetchParallelismConfig:
    """Tests for fetch_parallelism configuration."""

    def test_default_fetch_parallelism(self, tmp_path):
        """Test that default fetch_parallelism is 4."""
        from strata.config import StrataConfig

        config = StrataConfig(cache_dir=tmp_path / "cache")
        assert config.fetch_parallelism == 4

    def test_custom_fetch_parallelism(self, tmp_path):
        """Test custom fetch_parallelism value."""
        from strata.config import StrataConfig

        config = StrataConfig(cache_dir=tmp_path / "cache", fetch_parallelism=8)
        assert config.fetch_parallelism == 8

    def test_fetch_parallelism_env_var(self, monkeypatch, tmp_path):
        """Test STRATA_FETCH_PARALLELISM environment variable."""
        monkeypatch.setenv("STRATA_FETCH_PARALLELISM", "16")

        from strata.config import StrataConfig

        config = StrataConfig.load(cache_dir=tmp_path / "cache")
        assert config.fetch_parallelism == 16


class TestFetchExecutor:
    """Tests for dedicated fetch thread pool."""

    def test_fetch_executor_created(self, tmp_path):
        """Test that ServerState creates dedicated fetch executor."""
        from strata.config import StrataConfig
        from strata.server import ServerState

        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            fetch_parallelism=4,
            max_fetch_workers=48,  # Explicit worker count
        )
        state = ServerState(config)

        # Verify fetch executor exists and uses max_fetch_workers
        assert hasattr(state, "_fetch_executor")
        assert state._fetch_executor._max_workers == 48

        # Cleanup
        state._fetch_executor.shutdown(wait=False)
        state._planning_executor.shutdown(wait=False)

    def test_fetch_executor_sizing_uses_max_fetch_workers(self, tmp_path):
        """Test fetch executor uses max_fetch_workers config."""
        from strata.config import StrataConfig
        from strata.server import ServerState

        config = StrataConfig(
            cache_dir=tmp_path / "cache",
            fetch_parallelism=8,  # Per-scan parallelism
            max_fetch_workers=64,  # Total thread pool size
        )
        state = ServerState(config)

        # Workers = max_fetch_workers (decoupled from interactive/bulk slots)
        assert state._fetch_executor._max_workers == 64

        # Cleanup
        state._fetch_executor.shutdown(wait=False)
        state._planning_executor.shutdown(wait=False)


class TestPrometheusMetrics:
    """Tests for fetch parallelism Prometheus metrics."""

    def test_prometheus_includes_fetch_parallelism(self, tmp_path):
        """Test Prometheus endpoint includes fetch parallelism metrics."""
        import requests

        import strata.server as server_module
        from strata.config import StrataConfig
        from strata.server import ServerState, app
        from tests.conftest import find_free_port, wait_for_server

        port = find_free_port()

        config = StrataConfig(
            host="127.0.0.1",
            port=port,
            cache_dir=tmp_path / "cache",
            fetch_parallelism=4,
        )

        # Initialize state manually
        server_module._state = ServerState(config)

        # Start server in a thread
        server_thread = threading.Thread(
            target=uvicorn.run,
            kwargs={
                "app": app,
                "host": config.host,
                "port": config.port,
                "log_level": "error",
            },
            daemon=True,
        )
        server_thread.start()
        # Poll /health until the server is actually listening — a fixed sleep
        # races a slow/loaded runner (Windows refused the connection here).
        assert wait_for_server(port), "server did not become ready"

        try:
            response = requests.get(f"http://127.0.0.1:{port}/metrics/prometheus")
            assert response.status_code == 200
            content = response.text

            # Check for fetch parallelism metric
            assert "strata_fetch_parallelism" in content
            assert "strata_fetch_parallelism 4" in content

            # Check for fetch executor workers metric
            assert "strata_fetch_executor_workers" in content
            # Workers = max_fetch_workers (default 32)
            assert "strata_fetch_executor_workers 32" in content
        finally:
            server_module._state._fetch_executor.shutdown(wait=False)
            server_module._state._planning_executor.shutdown(wait=False)


class TestReorderingBuffer:
    """Tests for out-of-order fetch completion with reordering."""

    def test_segments_yielded_in_order(self):
        """Test that segments are yielded in correct order despite out-of-order completion."""
        # This is a conceptual test - the actual reordering happens in the streaming
        # endpoint which is harder to unit test. We verify the algorithm works correctly.

        # Simulate out-of-order completions
        completed = {}  # idx -> segment
        next_yield_idx = 0
        yielded = []

        # Segment 2 completes first
        completed[2] = b"segment_2"
        # Segment 0 completes
        completed[0] = b"segment_0"

        # Now we can yield segment 0
        while next_yield_idx in completed:
            yielded.append(completed.pop(next_yield_idx))
            next_yield_idx += 1

        assert yielded == [b"segment_0"]
        assert next_yield_idx == 1

        # Segment 1 completes - now we can yield 1 and 2
        completed[1] = b"segment_1"

        while next_yield_idx in completed:
            yielded.append(completed.pop(next_yield_idx))
            next_yield_idx += 1

        assert yielded == [b"segment_0", b"segment_1", b"segment_2"]
        assert next_yield_idx == 3
        assert completed == {}

    def test_reorder_buffer_handles_gaps(self):
        """Test reorder buffer correctly handles gaps in completion order."""
        completed = {}
        next_yield_idx = 0
        yielded = []

        # Only segment 3 and 5 complete (gaps at 0, 1, 2, 4)
        completed[3] = b"segment_3"
        completed[5] = b"segment_5"

        # Nothing should be yielded yet (waiting for 0)
        while next_yield_idx in completed:
            yielded.append(completed.pop(next_yield_idx))
            next_yield_idx += 1

        assert yielded == []
        assert next_yield_idx == 0

        # Fill in 0, 1, 2
        completed[0] = b"segment_0"
        completed[1] = b"segment_1"
        completed[2] = b"segment_2"

        # Now 0, 1, 2, 3 can be yielded
        while next_yield_idx in completed:
            yielded.append(completed.pop(next_yield_idx))
            next_yield_idx += 1

        assert yielded == [b"segment_0", b"segment_1", b"segment_2", b"segment_3"]
        assert next_yield_idx == 4

        # Add segment 4 to complete the chain
        completed[4] = b"segment_4"

        while next_yield_idx in completed:
            yielded.append(completed.pop(next_yield_idx))
            next_yield_idx += 1

        assert yielded == [
            b"segment_0",
            b"segment_1",
            b"segment_2",
            b"segment_3",
            b"segment_4",
            b"segment_5",
        ]
