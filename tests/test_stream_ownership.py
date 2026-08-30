"""Stream ownership: which node serves a stream, and the redirect it enables.

A stream cannot move between nodes -- its ``ReadPlan`` and its ``asyncio.Task``
are in-process -- so the point of these is not to share stream state but to
stop a routing problem looking identical to an expired stream.
"""

from __future__ import annotations

import time

import pytest

from strata.streaming.ownership import StreamOwnershipStore
from strata.streaming.registry import StreamRegistry


@pytest.fixture
def owners(tmp_path):
    return StreamOwnershipStore(tmp_path / "artifacts.sqlite")


class TestOwnershipStore:
    def test_claim_is_resolvable_by_another_node(self, owners):
        owners.claim("s1", "https://node-a:8765", ttl_seconds=60)
        assert owners.resolve("s1", exclude_node_url="https://node-b:8765") == (
            "https://node-a:8765"
        )

    def test_the_owning_node_resolves_to_nothing(self, owners):
        # Asked for its own claim, a node must 404 rather than redirect to
        # itself: the stream is genuinely gone, not elsewhere.
        owners.claim("s1", "https://node-a:8765", ttl_seconds=60)
        assert owners.resolve("s1", exclude_node_url="https://node-a:8765") is None

    def test_unknown_stream_resolves_to_nothing(self, owners):
        assert owners.resolve("never-existed", exclude_node_url="https://node-b:8765") is None

    def test_expired_claim_resolves_to_nothing(self, owners):
        # A node that dies never releases its claims, so a stale row must not
        # keep advertising it forever.
        owners.claim("s1", "https://node-a:8765", ttl_seconds=-1)
        assert owners.resolve("s1", exclude_node_url="https://node-b:8765") is None

    def test_reclaiming_moves_ownership(self, owners):
        # stream_id is usually the artifact id, so a refresh can legitimately
        # re-stream the same artifact from a different node. Newest wins,
        # because the newest planner is the one holding a live plan.
        owners.claim("s1", "https://node-a:8765", ttl_seconds=60)
        owners.claim("s1", "https://node-b:8765", ttl_seconds=60)
        assert owners.resolve("s1", exclude_node_url="https://node-c:8765") == (
            "https://node-b:8765"
        )

    def test_release_removes_the_claim(self, owners):
        owners.claim("s1", "https://node-a:8765", ttl_seconds=60)
        owners.release("s1")
        assert owners.resolve("s1", exclude_node_url="https://node-b:8765") is None

    def test_sweep_removes_only_expired_claims(self, owners):
        owners.claim("live", "https://node-a:8765", ttl_seconds=60)
        owners.claim("dead", "https://node-a:8765", ttl_seconds=-1)

        assert owners.sweep_expired() == 1
        assert owners.resolve("live", exclude_node_url="https://node-b:8765") is not None
        assert owners.resolve("dead", exclude_node_url="https://node-b:8765") is None


class TestRegistryHooks:
    """The registry stays unaware of storage; ownership is injected."""

    def _state(self, stream_id="s1"):
        from strata.streaming.registry import StreamState

        return StreamState(
            stream_id=stream_id,
            plan=None,
            artifact_id="a1",
            artifact_version=1,
            created_at=time.time(),
        )

    def test_register_claims_and_pop_releases(self):
        claimed: list[tuple[str, float]] = []
        released: list[str] = []
        registry = StreamRegistry(
            ttl_seconds=30.0,
            on_claim=lambda sid, ttl: claimed.append((sid, ttl)),
            on_release=released.append,
        )

        registry.register(self._state())
        assert claimed == [("s1", 30.0)]

        registry.pop("s1")
        assert released == ["s1"]

    def test_popping_an_absent_stream_releases_nothing(self):
        released: list[str] = []
        registry = StreamRegistry(ttl_seconds=30.0, on_release=released.append)

        registry.pop("never-registered")
        assert released == []

    def test_single_node_registries_do_no_ownership_work(self):
        # Without the callbacks -- the default, and every single-node
        # deployment -- register and pop must behave exactly as before.
        registry = StreamRegistry(ttl_seconds=30.0)
        registry.register(self._state())
        assert registry.get("s1") is not None
        assert registry.pop("s1") is not None


class TestRedirectTargetConstruction:
    """The stream id reaches the redirect URL from the request path."""

    def _target(self, owner_url, stream_id):
        from urllib.parse import quote

        # Mirrors the construction in server.get_stream.
        return f"{owner_url.rstrip('/')}/v1/streams/{quote(stream_id, safe='')}"

    def test_ordinary_ids_are_unchanged(self):
        assert self._target("https://node-a:8765", "abc123") == (
            "https://node-a:8765/v1/streams/abc123"
        )

    def test_trailing_slash_on_the_owner_url_is_not_doubled(self):
        assert self._target("https://node-a:8765/", "abc") == ("https://node-a:8765/v1/streams/abc")

    @pytest.mark.parametrize(
        "stream_id",
        ["a?x=1", "a#frag", "a/b", "../../etc", "a b"],
    )
    def test_the_id_stays_one_path_segment(self, stream_id):
        # A raw '?' or '#' would silently turn the rest into a query or
        # fragment, sending the client somewhere other than the stream it
        # asked for; a '/' would change the path shape.
        target = self._target("https://node-a:8765", stream_id)
        assert target.startswith("https://node-a:8765/v1/streams/")
        tail = target[len("https://node-a:8765/v1/streams/") :]
        assert "?" not in tail
        assert "#" not in tail
        assert "/" not in tail

    def test_the_origin_cannot_be_moved_by_the_id(self):
        # The host comes from operator config and the id only ever lands after
        # a fixed prefix, so no id makes this an open redirect.
        for hostile in ["@evil.com", "//evil.com", "https://evil.com"]:
            assert self._target("https://node-a:8765", hostile).startswith(
                "https://node-a:8765/v1/streams/"
            )


class TestOwnerResolution:
    """``_resolve_stream_owner`` decides between a redirect and a 404."""

    class _FakeState:
        def __init__(self, node_url):
            from strata.config import StrataConfig

            self.config = StrataConfig(node_advertised_url=node_url)

    def test_unconfigured_node_never_redirects(self, owners, monkeypatch):
        # Single node: no lookup at all, so the common case pays nothing even
        # if an ownership table happens to exist.
        from strata.server import _resolve_stream_owner

        owners.claim("s1", "https://node-a:8765", ttl_seconds=60)
        monkeypatch.setattr(
            "strata.streaming.ownership.get_stream_ownership_store", lambda *a, **k: owners
        )
        assert _resolve_stream_owner(self._FakeState(None), "s1") is None

    def test_configured_node_resolves_a_sibling(self, owners, monkeypatch):
        from strata.server import _resolve_stream_owner

        owners.claim("s1", "https://node-a:8765", ttl_seconds=60)
        monkeypatch.setattr(
            "strata.streaming.ownership.get_stream_ownership_store", lambda *a, **k: owners
        )
        assert (
            _resolve_stream_owner(self._FakeState("https://node-b:8765"), "s1")
            == "https://node-a:8765"
        )

    def test_lookup_failure_degrades_to_no_redirect(self, monkeypatch):
        # A database hiccup must produce today's 404, not a 500: the caller
        # asked for a stream, not for the ownership table's health.
        from strata.server import _resolve_stream_owner

        class _Broken:
            def resolve(self, *a, **k):
                raise RuntimeError("database is down")

        monkeypatch.setattr(
            "strata.streaming.ownership.get_stream_ownership_store", lambda *a, **k: _Broken()
        )
        assert _resolve_stream_owner(self._FakeState("https://node-b:8765"), "s1") is None
