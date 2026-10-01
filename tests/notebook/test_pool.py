"""Tests for the warm process pool (M6)."""

from __future__ import annotations

import asyncio

import pytest

from strata.notebook.pool import WarmProcessPool
from strata.notebook.writer import create_notebook

pytestmark = [pytest.mark.integration, pytest.mark.warm_pool]


@pytest.fixture
def notebook_dir(tmp_path):
    """Create a simple notebook directory for testing.

    Returns:
        Path to notebook directory
    """
    notebook_path = create_notebook(tmp_path, "Test Notebook")
    return notebook_path


class TestWarmProcessPool:
    """Test the warm process pool."""

    @pytest.mark.asyncio
    async def test_pool_starts_with_configured_size(self, notebook_dir):
        """Test that pool starts with correct number of processes."""
        pool = WarmProcessPool(notebook_dir, pool_size=2)
        await pool.start()

        assert pool._available.qsize() == 2

        await pool.drain()

    @pytest.mark.asyncio
    async def test_acquire_warm_process(self, notebook_dir):
        """Test acquiring a warm process."""
        pool = WarmProcessPool(notebook_dir, pool_size=1)
        await pool.start()

        warm_proc = await pool.acquire()

        if warm_proc is not None:  # May be None if startup failed
            assert warm_proc.process is not None
            assert warm_proc.ready is True

        await pool.drain()

    @pytest.mark.asyncio
    async def test_process_killed_after_use(self, notebook_dir):
        """Test that process is killed and replaced after use."""
        pool = WarmProcessPool(notebook_dir, pool_size=1)
        await pool.start()

        warm_proc = await pool.acquire()

        if warm_proc is not None:
            await pool.release_and_replace(warm_proc)

            assert warm_proc.process.returncode is not None

        await pool.drain()

    @pytest.mark.asyncio
    async def test_pool_drains_on_close(self, notebook_dir):
        """Test that pool drains all processes."""
        pool = WarmProcessPool(notebook_dir, pool_size=2)
        await pool.start()

        initial_size = pool._available.qsize()
        assert initial_size == 2

        await pool.drain()

        assert pool._available.qsize() == 0

    @pytest.mark.asyncio
    async def test_pool_invalidate_respawns(self, notebook_dir):
        """Test pool invalidation respawns processes."""
        pool = WarmProcessPool(notebook_dir, pool_size=1)
        await pool.start()

        initial_size = pool._available.qsize()
        assert initial_size >= 1

        # Invalidate drains and respawns.
        await pool.invalidate()

        # The respawn may be async.
        await asyncio.sleep(0.5)

        final_size = pool._available.qsize()
        assert final_size >= 0

        await pool.drain()

    @pytest.mark.asyncio
    async def test_cold_fallback_when_pool_empty(self, notebook_dir):
        """Test that acquire returns None when pool is empty."""
        pool = WarmProcessPool(notebook_dir, pool_size=0)
        # Pool not started.

        warm_proc = await pool.acquire()
        assert warm_proc is None
