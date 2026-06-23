"""Unit tests for MaterializeService.explain — pure dry-run logic, no server.

The route resolves input versions (which may 400/404) and hands the resolved map
to the service; these tests drive the service directly with a fake store.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strata.artifact_store import TransformSpec as ArtifactTransformSpec
from strata.services.materialize import InputResolutionError, MaterializeService
from strata.types import ExplainMaterializeRequest, TransformSpec


def _request(inputs, *, name=None):
    return ExplainMaterializeRequest(
        inputs=inputs,
        transform=TransformSpec(executor="scan@v1", params={}),
        name=name,
    )


class _FakeStore:
    """Minimal store stub for the two methods explain() touches."""

    def __init__(self, *, existing=None, name_status=None):
        self._existing = existing
        self._name_status = name_status
        self.find_calls = []
        self.name_calls = []

    def find_by_provenance(self, provenance_hash, *, tenant=None):
        self.find_calls.append((provenance_hash, tenant))
        return self._existing

    def get_name_status(self, name, *, tenant=None):
        self.name_calls.append((name, tenant))
        return self._name_status


@pytest.fixture
def service():
    return MaterializeService()


def test_cache_hit_reports_would_hit(service):
    store = _FakeStore(existing=SimpleNamespace(id="abc", version=3))
    resp = service.explain(
        store,
        request=_request(["file:///wh#db.t"]),
        tenant=None,
        resolved_versions={"file:///wh#db.t": "snap1"},
    )
    assert resp.would_hit is True
    assert resp.would_build is False
    assert resp.artifact_uri == "strata://artifact/abc@v=3"


def test_cache_miss_without_name_would_build(service):
    store = _FakeStore(existing=None)
    resp = service.explain(
        store,
        request=_request(["file:///wh#db.t"]),
        tenant=None,
        resolved_versions={"file:///wh#db.t": "snap1"},
    )
    assert resp.would_hit is False
    assert resp.would_build is True
    assert resp.is_stale is False
    assert resp.changed_inputs is None
    # No name → name lookup is skipped entirely.
    assert store.name_calls == []


def test_named_artifact_with_unchanged_inputs_is_not_stale(service):
    name_status = SimpleNamespace(
        artifact_uri="strata://artifact/old@v=1",
        input_versions={"file:///wh#db.t": "snap1"},
    )
    store = _FakeStore(existing=None, name_status=name_status)
    resp = service.explain(
        store,
        request=_request(["file:///wh#db.t"], name="daily"),
        tenant=None,
        resolved_versions={"file:///wh#db.t": "snap1"},  # same as recorded
    )
    assert resp.would_build is True
    assert resp.is_stale is False
    assert resp.artifact_uri == "strata://artifact/old@v=1"
    assert resp.changed_inputs is None


def test_named_artifact_with_changed_inputs_is_stale(service):
    name_status = SimpleNamespace(
        artifact_uri="strata://artifact/old@v=1",
        input_versions={"file:///wh#db.t": "snap1"},
    )
    store = _FakeStore(existing=None, name_status=name_status)
    resp = service.explain(
        store,
        request=_request(["file:///wh#db.t"], name="daily"),
        tenant=None,
        resolved_versions={"file:///wh#db.t": "snap2"},  # drifted
    )
    assert resp.is_stale is True
    assert resp.changed_inputs is not None
    assert len(resp.changed_inputs) == 1
    change = resp.changed_inputs[0]
    assert (change.old_version, change.new_version) == ("snap1", "snap2")
    assert "snap1 → snap2" in resp.stale_reason


def test_tenant_is_threaded_to_store_calls(service):
    store = _FakeStore(existing=None)
    service.explain(
        store,
        request=_request(["file:///wh#db.t"], name="daily"),
        tenant="acme",
        resolved_versions={"file:///wh#db.t": "snap1"},
    )
    assert store.find_calls[0][1] == "acme"
    assert store.name_calls[0][1] == "acme"


def test_error_markers_pass_through(service):
    store = _FakeStore(existing=None)
    resolved = {"strata://artifact/x@v=1": "<error: Name not found>"}
    resp = service.explain(
        store,
        request=_request(["strata://artifact/x@v=1"]),
        tenant=None,
        resolved_versions=resolved,
    )
    assert resp.resolved_input_versions == resolved


class TestComputeProvenance:
    def _spec(self, inputs):
        return ArtifactTransformSpec(executor="scan@v1", params={}, inputs=inputs)

    def test_is_independent_of_input_ordering(self, service):
        # The cache-integrity invariant: same computation → same hash regardless
        # of the dict iteration order of resolved inputs.
        spec = self._spec(["a", "b"])
        h1 = service.compute_provenance(spec, {"a": "1", "b": "2"})
        h2 = service.compute_provenance(spec, {"b": "2", "a": "1"})
        assert h1 == h2

    def test_changes_with_a_version(self, service):
        spec = self._spec(["a"])
        assert service.compute_provenance(spec, {"a": "1"}) != service.compute_provenance(
            spec, {"a": "2"}
        )


class _FakeResolveStore:
    """Store stub for resolve_input_version: only ``resolve_name`` is touched."""

    def __init__(self, *, named=None):
        self._named = named
        self.calls = []

    def resolve_name(self, name, *, tenant=None):
        self.calls.append((name, tenant))
        return self._named


class _FakePlanner:
    """Planner stub returning a fixed plan, or raising to simulate a bad table."""

    def __init__(self, *, snapshot_id=None, table_identity=None, error=None):
        self._snapshot_id = snapshot_id
        self._table_identity = table_identity
        self._error = error
        self.calls = []

    def plan(self, *, table_uri, snapshot_id, columns, filters):
        self.calls.append(table_uri)
        if self._error is not None:
            raise self._error
        return SimpleNamespace(snapshot_id=self._snapshot_id, table_identity=self._table_identity)


class TestResolveInputVersion:
    """The pure resolver (no ACL, no HTTP) — the unit the dependency wrapper wraps."""

    def test_artifact_uri_returns_id_and_version(self, service):
        resolved = service.resolve_input_version(
            "strata://artifact/abc@v=3", store=_FakeResolveStore(), planner=_FakePlanner()
        )
        assert resolved.version == "abc@v=3"
        assert resolved.table_identity is None

    def test_malformed_artifact_uri_is_400(self, service):
        with pytest.raises(InputResolutionError) as exc:
            service.resolve_input_version(
                "strata://artifact/no-version", store=_FakeResolveStore(), planner=_FakePlanner()
            )
        assert exc.value.status_code == 400

    def test_name_uri_resolves_via_store(self, service):
        store = _FakeResolveStore(named=SimpleNamespace(id="xyz", version=7))
        resolved = service.resolve_input_version(
            "strata://name/daily", store=store, planner=_FakePlanner(), tenant="acme"
        )
        assert resolved.version == "xyz@v=7"
        assert resolved.table_identity is None
        assert store.calls == [("daily", "acme")]  # tenant threaded through

    def test_unknown_name_is_404(self, service):
        with pytest.raises(InputResolutionError) as exc:
            service.resolve_input_version(
                "strata://name/missing", store=_FakeResolveStore(named=None), planner=_FakePlanner()
            )
        assert exc.value.status_code == 404

    def test_table_uri_returns_snapshot_and_identity(self, service):
        planner = _FakePlanner(snapshot_id=4242, table_identity="cat.ns.t")
        resolved = service.resolve_input_version(
            "file:///wh#db.t", store=_FakeResolveStore(), planner=planner
        )
        assert resolved.version == "4242"
        # The identity is surfaced so the wrapper can ACL-gate the table input.
        assert resolved.table_identity == "cat.ns.t"

    def test_table_plan_failure_is_400(self, service):
        planner = _FakePlanner(error=RuntimeError("no such table"))
        with pytest.raises(InputResolutionError) as exc:
            service.resolve_input_version(
                "s3://wh#db.t", store=_FakeResolveStore(), planner=planner
            )
        assert exc.value.status_code == 400
        assert "no such table" in exc.value.detail

    def test_unknown_uri_scheme_is_400(self, service):
        with pytest.raises(InputResolutionError) as exc:
            service.resolve_input_version(
                "ftp://nope", store=_FakeResolveStore(), planner=_FakePlanner()
            )
        assert exc.value.status_code == 400


class TestRebuildArtifactId:
    def test_fresh_miss_uses_new_id(self, service):
        assert service.rebuild_artifact_id(None, refresh=False, new_id="new") == "new"

    def test_refresh_reuses_existing_id(self, service):
        existing = SimpleNamespace(id="old", version=2)
        assert service.rebuild_artifact_id(existing, refresh=True, new_id="new") == "old"

    def test_refresh_without_existing_uses_new_id(self, service):
        assert service.rebuild_artifact_id(None, refresh=True, new_id="new") == "new"

    def test_non_refresh_with_existing_still_uses_new_id(self, service):
        # A non-refresh miss (e.g. a building row) mints a fresh id, not a reuse.
        existing = SimpleNamespace(id="old", version=2)
        assert service.rebuild_artifact_id(existing, refresh=False, new_id="new") == "new"
