"""Unit tests for ``RegistryService.summary`` — pure aggregation, no server/DB.

The summary aggregation groups aliases per name and hides internal ``nb_*``
stamps from the user-facing table. Both are exercised here with a fake store.
"""

from types import SimpleNamespace

from strata.services.registry import registry_service


class _FakeStore:
    def __init__(self, *, aliases, names, tags):
        self._aliases = aliases  # list of (name, alias, version)
        self._names = names  # list of (name, artifact_id, version)
        self._tags = tags  # {(artifact_id, version): {k: v}}

    def list_aliases(self, name, *, tenant=None):
        return [SimpleNamespace(name=n, alias=a, version=v) for n, a, v in self._aliases]

    def list_names(self, *, tenant=None):
        return [SimpleNamespace(name=n, artifact_id=aid, version=v) for n, aid, v in self._names]

    def get_tags(self, artifact_id, version, *, tenant=None):
        return self._tags.get((artifact_id, version), {})


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
