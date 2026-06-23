"""Unit tests for MaterializeService.explain — pure dry-run logic, no server.

The route resolves input versions (which may 400/404) and hands the resolved map
to the service; these tests drive the service directly with a fake store.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strata.services.materialize import MaterializeService
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
