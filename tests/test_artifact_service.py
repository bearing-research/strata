"""Unit tests for ``ArtifactService`` — pure graph logic, no server/DB.

The whole point of the service extraction (P2): the lineage BFS is testable with
plain artifact stubs and a fake store, no TestClient and no SQLite. ``build_lineage``
only reads attributes off the artifact records and calls ``store.get_artifact``,
so ``SimpleNamespace`` stubs suffice.
"""

import json
from types import SimpleNamespace

from strata.services.artifact import artifact_service


def _art(
    art_id,
    version,
    *,
    inputs=None,
    state="ready",
    tenant=None,
    created_at=100.0,
    principal=None,
    transform_spec=None,
):
    return SimpleNamespace(
        id=art_id,
        version=version,
        state=state,
        tenant=tenant,
        transform_spec=transform_spec,
        input_versions=json.dumps(inputs) if inputs else None,
        created_at=created_at,
        principal=principal,
    )


def _notebook_spec(*, build_env="", build_duration_ms=None):
    """A stored transform spec as the notebook writes one."""
    params = {"content_type": "json/object"}
    if build_env:
        params["build_env"] = build_env
    if build_duration_ms is not None:
        params["build_duration_ms"] = str(build_duration_ms)
    return json.dumps({"executor": "notebook/cell@v1", "params": params, "inputs": []})


class _FakeStore:
    def __init__(self, *arts):
        self._arts = {(a.id, a.version): a for a in arts}

    def get_artifact(self, art_id, version):
        return self._arts.get((art_id, version))


class _DependentsStore:
    """Fake store for build_dependents: scripted find_dependents + name lookups."""

    def __init__(self, results, names=None):
        self._results = results  # list[(artifact_stub, input_version)]
        self._names = names or {}

    def find_dependents(self, artifact_id, version, *, tenant=None):
        return self._results

    def get_name_for_artifact(self, art_id, version, *, tenant=None):
        return self._names.get((art_id, version))


def _lineage(store, root, *, tenant_filter=None, max_depth=10):
    return artifact_service.build_lineage(
        store,
        artifact=root,
        artifact_id=root.id,
        version=root.version,
        tenant_filter=tenant_filter,
        max_depth=max_depth,
    )


def test_lineage_resolves_artifact_and_table_inputs():
    root = _art(
        "R",
        1,
        inputs={"s3://bucket/t#db.tbl": "snap123", "strata://artifact/A@v=1": "A@v=1"},
    )
    resp = _lineage(_FakeStore(root, _art("A", 1)), root)

    node_uris = {n.uri for n in resp.nodes}
    assert node_uris == {
        "strata://artifact/R@v=1",
        "strata://artifact/A@v=1",
        "s3://bucket/t#db.tbl",
    }
    # The table input is a leaf node typed "table"; the artifact input is "artifact".
    by_uri = {n.uri: n for n in resp.nodes}
    assert by_uri["s3://bucket/t#db.tbl"].type == "table"
    assert by_uri["strata://artifact/A@v=1"].type == "artifact"
    assert set(resp.direct_inputs) == {"s3://bucket/t#db.tbl", "strata://artifact/A@v=1"}
    assert resp.depth == 1
    # Every edge points at the root.
    assert all(e.to_uri == "strata://artifact/R@v=1" for e in resp.edges)


def test_lineage_traverses_transitively_and_respects_max_depth():
    root = _art("R", 1, inputs={"strata://artifact/A@v=1": "A@v=1"})
    a = _art("A", 1, inputs={"strata://artifact/B@v=1": "B@v=1"})
    b = _art("B", 1)
    store = _FakeStore(root, a, b)

    full = _lineage(store, root, max_depth=10)
    assert "strata://artifact/B@v=1" in {n.uri for n in full.nodes}
    assert full.depth == 2

    shallow = _lineage(store, root, max_depth=1)
    assert "strata://artifact/B@v=1" not in {n.uri for n in shallow.nodes}
    assert shallow.depth == 1


def test_lineage_marks_cross_tenant_input_as_unknown_stub():
    root = _art("R", 1, inputs={"strata://artifact/A@v=1": "A@v=1"}, tenant="t1")
    # A belongs to another tenant — it must not leak through lineage.
    other = _art("A", 1, inputs={"strata://artifact/SECRET@v=1": "SECRET@v=1"}, tenant="t2")
    resp = _lineage(_FakeStore(root, other), root, tenant_filter="t1")

    by_uri = {n.uri: n for n in resp.nodes}
    a_node = by_uri["strata://artifact/A@v=1"]
    assert a_node.type == "artifact"
    # Stub node: no transform/created_at, and traversal stopped — its own input
    # (another tenant's SECRET artifact) is never surfaced.
    assert a_node.created_at is None
    assert "strata://artifact/SECRET@v=1" not in by_uri


def test_dependents_maps_results_and_resolves_names():
    results = [
        (_art("D1", 3), "R@v=1"),
        (_art("D2", 1), "R@v=1"),
    ]
    store = _DependentsStore(results, names={("D1", 3): "champion-model"})
    resp = artifact_service.build_dependents(
        store, artifact_id="R", version=1, tenant_filter=None, limit=100
    )

    assert resp.total_count == 2
    assert [d.artifact_uri for d in resp.dependents] == [
        "strata://artifact/D1@v=3",
        "strata://artifact/D2@v=1",
    ]
    assert resp.dependents[0].name == "champion-model"
    assert resp.dependents[1].name is None
    assert all(d.input_version == "R@v=1" for d in resp.dependents)


def test_dependents_caps_list_at_limit_but_total_counts_all():
    results = [(_art(f"D{i}", 1), "R@v=1") for i in range(5)]
    resp = artifact_service.build_dependents(
        _DependentsStore(results), artifact_id="R", version=1, tenant_filter=None, limit=2
    )

    assert resp.total_count == 5  # full count, not the page size
    assert len(resp.dependents) == 2


def test_lineage_reports_who_computed_each_step_and_where():
    """Lineage's reason to exist changed when results became shareable.

    Before a shared store, "who produced this" was always you and "in which
    environment" was always this machine, so neither was worth a column. Once a
    graph can contain a step a teammate ran on another machine, they are the
    questions it is being opened to answer.
    """
    upstream = _art(
        "raw",
        1,
        principal="alice@lab",
        transform_spec=_notebook_spec(
            build_env="cpython-3.14-linux-x86_64", build_duration_ms=38000
        ),
    )
    root = _art(
        "model",
        1,
        inputs={"strata://artifact/raw@v=1": "raw@v=1"},
        principal="bob@lab",
        transform_spec=_notebook_spec(build_env="cpython-3.14-darwin-arm64", build_duration_ms=250),
    )

    graph = artifact_service.build_lineage(
        _FakeStore(upstream, root),
        artifact=root,
        artifact_id="model",
        version=1,
        tenant_filter=None,
        max_depth=5,
    )
    by_uri = {n.uri: n for n in graph.nodes}

    me = by_uri["strata://artifact/model@v=1"]
    assert (me.principal, me.build_env, me.build_duration_ms) == (
        "bob@lab",
        "cpython-3.14-darwin-arm64",
        250,
    )

    # The interesting row: an upstream someone else ran, on a different platform.
    theirs = by_uri["strata://artifact/raw@v=1"]
    assert (theirs.principal, theirs.build_env, theirs.build_duration_ms) == (
        "alice@lab",
        "cpython-3.14-linux-x86_64",
        38000,
    )


def test_lineage_reports_nothing_rather_than_guessing():
    """Tables, core transforms, and anything stored before these fields existed
    record none of it. Empty is the honest answer — a plausible default would be
    a claim about a machine nobody observed."""
    root = _art("plain", 1, inputs={"file:///wh#ns.tbl@snapshot=7": "snapshot=7"})

    graph = artifact_service.build_lineage(
        _FakeStore(root),
        artifact=root,
        artifact_id="plain",
        version=1,
        tenant_filter=None,
        max_depth=5,
    )
    for node in graph.nodes:
        assert node.principal is None
        assert node.build_env == ""
        assert node.build_duration_ms == 0


def test_lineage_survives_a_transform_spec_it_cannot_parse():
    """``transform_spec`` is client-opaque. A malformed one means "not
    recorded", the same reading ``_transform_ref`` already takes — not a 500 on
    a read path over metadata nothing depends on."""
    root = _art("odd", 1, transform_spec="{not json at all")

    graph = artifact_service.build_lineage(
        _FakeStore(root),
        artifact=root,
        artifact_id="odd",
        version=1,
        tenant_filter=None,
        max_depth=5,
    )
    node = graph.nodes[0]
    assert node.build_env == ""
    assert node.build_duration_ms == 0
