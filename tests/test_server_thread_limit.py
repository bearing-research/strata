"""Tests for the SQL dialect seam."""

from __future__ import annotations


class TestThreadLimitAndPoolAreOneNumber:
    """The pool holds a connection for every thread that may call the store at once."""

    def test_the_server_sets_the_thread_limiter_from_the_shared_limit(self, tmp_path, monkeypatch):
        # A limit other than anyio's default of 40, so a lifespan that left the limiter
        # alone fails here.
        import anyio.to_thread
        from fastapi.testclient import TestClient

        import strata.server as server_module

        monkeypatch.setenv("STRATA_ARTIFACT_DIR", str(tmp_path / "artifacts"))
        monkeypatch.setenv("STRATA_CACHE_DIR", str(tmp_path / "cache"))
        monkeypatch.setattr(server_module, "SERVER_THREAD_LIMIT", 7)

        async def total_tokens() -> float:
            return anyio.to_thread.current_default_thread_limiter().total_tokens

        with TestClient(server_module.app) as client:
            assert client.portal.call(total_tokens) == 7
