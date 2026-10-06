"""Unit tests for ``RegistryService.summary``: pure aggregation with a fake store.

Aliases group per name, and internal ``nb_*`` stamps stay out of the user-facing table.
"""

from types import SimpleNamespace

from strata.services.registry import registry_service


class _FakeStore:
    def __init__(self, *, aliases, names, tags, reads=()):
        self._aliases = aliases  # list of (name, alias, version)
        self._names = names  # list of (name, artifact_id, version)
        self._tags = tags  # {(artifact_id, version): {k: v}}
        self._reads = list(reads)  # list of (artifact_id, reference)

    def list_aliases(self, name, *, tenant=None):
        return [
            SimpleNamespace(name=n, alias=a, version=v, tenant=tenant) for n, a, v in self._aliases
        ]

    def list_names(self, *, tenant=None):
        return [
            SimpleNamespace(name=n, artifact_id=aid, version=v, tenant=tenant)
            for n, aid, v in self._names
        ]

    def get_tags(self, artifact_id, version, *, tenant=None):
        return self._tags.get((artifact_id, version), {})

    def list_name_reads(self, *, tenant=None):
        return self._reads


def test_summary_groups_aliases_and_hides_internal_tags():
    store = _FakeStore(
        aliases=[("model", "champion", 3), ("model", "candidate", 4)],
        names=[("model", "A", 3)],
        tags={("A", 3): {"stage": "prod", "nb_cell": "c1", "nb_notebook": "n1"}},
    )

    rows = registry_service.summary(store, tenant=None)

    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "model"
    assert row["uri"] == "strata://artifact/A@v=3"
    assert row["aliases"] == {"champion": 3, "candidate": 4}
    # nb_* stamps are internal and must not surface in the dashboard table.
    assert row["tags"] == {"stage": "prod"}


def test_summary_name_without_aliases_gets_empty_map():
    store = _FakeStore(
        aliases=[],
        names=[("lonely", "B", 1)],
        tags={},
    )

    rows = registry_service.summary(store, tenant="team-x")

    assert rows[0]["aliases"] == {}
    assert rows[0]["tags"] == {}
    assert rows[0]["readers"] == []


def test_summary_lists_the_notebook_cells_that_read_each_name():
    store = _FakeStore(
        aliases=[],
        names=[("taxi/model", "M", 2), ("taxi/features", "F", 1)],
        tags={},
        reads=[
            # Two outputs of one cell read the same name: one reader.
            ("nb_nb-1_cell_a1b2c3d4_var_score", "taxi/model@champion"),
            ("nb_nb-1_cell_a1b2c3d4_var_error", "taxi/model@champion"),
            ("nb_nb-2_cell_c2_var_x", "taxi/model@v=2"),
            ("nb_nb-2_cell_c3_var_y", "taxi/features"),
            # Not a notebook cell's output.
            ("0f6c-uuid", "taxi/model"),
        ],
    )

    rows = {row["name"]: row for row in registry_service.summary(store, tenant=None)}

    assert rows["taxi/model"]["readers"] == [
        {"notebook_id": "nb-1", "cell_id": "a1b2c3d4", "reference": "taxi/model@champion"},
        {"notebook_id": "nb-2", "cell_id": "c2", "reference": "taxi/model@v=2"},
    ]
    assert rows["taxi/features"]["readers"] == [
        {"notebook_id": "nb-2", "cell_id": "c3", "reference": "taxi/features"}
    ]


class TestTenantScope:
    """A tenant reads its own rows; ``all_tenants`` (personal mode, ``admin:*``) reads every row.

    ``None`` is the default tenant in the store, so passing it to mean "all" found nothing.
    """

    @staticmethod
    def _team_store(tmp_path):
        from strata.artifact_store import ArtifactStore, TransformSpec

        store = ArtifactStore(tmp_path / "artifacts")
        store.create_artifact("m1", "p1", TransformSpec("e", {}, []), tenant="team")
        store.write_blob("m1", 1, b"x")
        store.finalize_artifact("m1", 1, "{}", 1, 1)
        store.set_name("team/model", "m1", 1, tenant="team")
        store.set_alias("team/model", "champion", "m1", 1, tenant="team")
        store.set_tag("m1", 1, "nb_cell", "c1", tenant="team")
        store.set_tag("m1", 1, "stage", "prod", tenant="team")
        return store

    def test_a_tenants_tag_lookup_names_its_artifacts(self, tmp_path):
        rows = registry_service.artifacts_by_tag(
            self._team_store(tmp_path), "nb_cell", tenant="team"
        )

        assert [(r["artifact_id"], r["names"]) for r in rows] == [("m1", ["team/model"])]

    def test_the_whole_store_summary_lists_every_tenants_names(self, tmp_path):
        rows = registry_service.summary(self._team_store(tmp_path), tenant=None, all_tenants=True)

        assert [(r["name"], r["aliases"], r["tags"]) for r in rows] == [
            ("team/model", {"champion": 1}, {"stage": "prod"})
        ]

    def test_the_default_tenant_sees_none_of_another_tenants_rows(self, tmp_path):
        store = self._team_store(tmp_path)

        assert registry_service.summary(store, tenant=None) == []
        assert registry_service.artifacts_by_tag(store, "nb_cell", tenant=None) == []
