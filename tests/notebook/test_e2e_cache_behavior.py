"""E2E tests: artifact caching and provenance deduplication.

Only consumed outputs are stored, so only cells with downstream consumers can cache-hit.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.notebook.e2e_fixtures import (
    NotebookBuilder,
    create_test_app,
    execute_cell_and_wait,
    open_notebook_session,
    ws_connect,
)


@pytest.fixture
def setup():
    app = create_test_app()
    client = TestClient(app)
    with tempfile.TemporaryDirectory() as tmpdir:
        yield client, Path(tmpdir)


class TestCacheHit:
    """Re-executing an unchanged cell whose output is consumed should cache."""

    def test_upstream_cell_cache_hit(self, setup):
        """Run c1→c2 twice; c1 caches on the second run."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 42").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                r1 = execute_cell_and_wait(ws, "c1")
                assert r1["type"] == "cell_output"

                execute_cell_and_wait(ws, "c2")
                ws.clear()

                # Same source, same inputs.
                r2 = execute_cell_and_wait(ws, "c1")
                assert r2["type"] == "cell_output"
                assert r2["payload"].get("cache_hit") is True

    def test_cache_hit_reports_execution_method(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                r2 = execute_cell_and_wait(ws, "c1")
                if r2["payload"].get("cache_hit"):
                    assert r2["payload"].get("execution_method") == "cached"

    def test_force_execution_bypasses_target_cache(self, setup):
        """Force-running a cell should execute, not return a cached artifact."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                ws.execute_force("c2")

                result = None
                while True:
                    msg = ws.receive()
                    if (
                        msg["type"] in ("cell_output", "cell_error")
                        and msg["payload"].get("cell_id") == "c2"
                    ):
                        result = msg
                    if (
                        msg["type"] == "cell_status"
                        and msg["payload"].get("cell_id") == "c2"
                        and msg["payload"].get("status") in ("ready", "error")
                    ):
                        break

                assert result is not None
                if result["type"] == "cell_output":
                    assert result["payload"].get("cache_hit") is not True
                    assert result["payload"].get("execution_method") != "cached"


class TestCacheMiss:
    """Changing cell source should invalidate the cache."""

    def test_source_change_invalidates(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                execute_cell_and_wait(ws, "c1")
                execute_cell_and_wait(ws, "c2")

                ws.update_source("c1", "x = 2")
                ws.receive_until("dag_update")
                ws.clear()

                # New source, so a cache miss.
                r2 = execute_cell_and_wait(ws, "c1")
                assert r2["type"] == "cell_output"
                assert r2["payload"].get("cache_hit") is not True


class TestCascadeCache:
    def test_cascade_then_direct_rerun(self, setup):
        """Run the full cascade, then re-run the upstream directly: a cache hit."""
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 1").add_cell("c2", "y = x + 1", after="c1")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                # Cascade execution (c2 triggers c1).
                execute_cell_and_wait(ws, "c2")
                ws.clear()

                # Re-executing c1 directly is a cache hit.
                r = execute_cell_and_wait(ws, "c1")
                assert r["type"] == "cell_output"
                assert r["payload"].get("cache_hit") is True

                downstream = next(c for c in session.notebook_state.cells if c.id == "c2")
                assert downstream.status == "ready"

    def test_leaf_cell_is_cached_on_its_console_record(self, setup):
        client, tmp = setup
        nb = NotebookBuilder(tmp).add_cell("c1", "x = 42")

        with open_notebook_session(client, nb.path) as (sid, session):
            with ws_connect(client, sid) as ws:
                r1 = execute_cell_and_wait(ws, "c1")
                assert r1["type"] == "cell_output"
                assert r1["payload"].get("cache_hit") is not True

                r2 = execute_cell_and_wait(ws, "c1")
                assert r2["type"] == "cell_output"
                # Nothing reads x, so the cell's (empty) console is its record
                # under this provenance; an unchanged run replays it.
                assert r2["payload"].get("cache_hit") is True
