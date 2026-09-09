"""Sharing a result with the team, on purpose.

The team cache offered every downstream-consumed variable of every successful
cell, or nothing. On a shared server that is the point; on a personal one it
means every intermediate a researcher ever computed lands in the team's store
whether or not they meant to share it. Item 21.
"""

from __future__ import annotations

import argparse
import asyncio
import json

import httpx
import pytest

from strata.artifact_store import ArtifactStore
from strata.notebook.artifact_integration import NotebookArtifactManager


@pytest.fixture
def team_dir(tmp_path):
    return tmp_path / "team"


@pytest.fixture
def team_store(tmp_path, team_dir):
    """A real server standing in for the team's shared store."""
    from tests.conftest import run_server_with_context

    with run_server_with_context(tmp_path / "cache", team_dir, "personal") as ctx:
        yield ctx.base_url


@pytest.fixture
def chain(tmp_path):
    """A two-cell notebook: an upstream, and a figure that consumes it."""
    manager = NotebookArtifactManager("nb", artifact_dir=tmp_path / "notebook")
    upstream = manager.store_cell_output(
        cell_id="c1",
        variable_name="rows",
        blob_data=b"[1]",
        content_type="json/object",
        provenance_hash="a1" * 32,
        input_versions={},
        source="rows = [1]",
    )
    ref = f"{upstream.id}@v={upstream.version}"
    figure = manager.store_cell_output(
        cell_id="c2",
        variable_name="model",
        blob_data=b"MODEL",
        content_type="pickle/object",
        provenance_hash="b2" * 32,
        input_versions={f"strata://artifact/{ref}": ref},
        source="model = fit(rows)",
    )
    return {"dir": tmp_path / "notebook", "upstream": upstream, "figure": figure}


def _promote(chain, url, **overrides):
    from strata.artifact_cli import cmd_promote

    args = {
        "ref": chain["figure"].id,
        "artifact_dir": str(chain["dir"]),
        "to_url": url,
        "name": "taxi/model",
        "alias": None,
        "tag": None,
        "header": None,
        "format": "human",
        "tenant": None,
        "max_depth": 10,
    }
    args.update(overrides)
    return cmd_promote(argparse.Namespace(**args))


class TestPromote:
    def test_the_artifact_arrives_under_its_name(self, team_store, team_dir, chain):
        assert _promote(chain, team_store) == 0

        response = httpx.get(f"{team_store}/v1/names/taxi/model", timeout=10)
        assert response.status_code == 200

    def test_the_whole_chain_travels(self, team_store, team_dir, chain):
        """Not just the artifact.

        The cache is keyed by provenance, so each ancestor that arrives is a
        hit for the next person whose cell computes the same thing. Sending
        the result alone would share the answer and none of the work.
        """
        _promote(chain, team_store)

        store = ArtifactStore(team_dir)
        assert store.get_artifact(chain["upstream"].id, chain["upstream"].version) is not None
        assert store.get_artifact(chain["figure"].id, chain["figure"].version) is not None

    def test_an_ancestor_becomes_a_cache_hit(self, team_store, team_dir, chain):
        """The property the chain exists for, stated as the team sees it.

        A colleague whose cell computes the same upstream looks it up by
        provenance hash — and finds it, because someone promoted a result
        built on it.
        """
        _promote(chain, team_store)

        store = ArtifactStore(team_dir)
        found = store.find_by_provenance("a1" * 32)

        assert found is not None
        assert found.state == "ready"

    def test_lineage_resolves_on_the_far_side(self, team_store, team_dir, chain):
        from strata.services.artifact import ArtifactService

        _promote(chain, team_store)

        store = ArtifactStore(team_dir)
        promoted = store.get_artifact(chain["figure"].id, chain["figure"].version)
        lineage = ArtifactService().build_lineage(
            store,
            artifact=promoted,
            artifact_id=promoted.id,
            version=promoted.version,
            tenant_filter=None,
            max_depth=10,
        )

        artifacts = [n for n in lineage.nodes if n.type == "artifact"]
        assert [n.artifact_id for n in artifacts] == [
            chain["figure"].id,
            chain["upstream"].id,
        ]

        # Every node resolves to a row this store actually holds. The builder
        # makes a node out of a recorded *edge*, so the list above is
        # satisfied by a figure whose ancestor never arrived — which is the
        # failure this test exists to catch, and it would not have.
        for node in artifacts:
            assert store.get_artifact(node.artifact_id, node.version) is not None, (
                f"lineage names {node.artifact_id}@v={node.version}, "
                f"which is not in the store it points at"
            )

    def test_it_mints_no_public_link(self, team_store, team_dir, chain):
        """Promoting is not publishing. A team name is not a URL anyone can
        read without credentials."""
        _promote(chain, team_store)

        assert ArtifactStore(team_dir).list_publications() == []

    def test_tags_are_set(self, team_store, team_dir, chain):
        _promote(chain, team_store, tag=["stage=candidate", "owner=fli"])

        store = ArtifactStore(team_dir)
        tags = store.get_tags(chain["figure"].id, chain["figure"].version)

        assert tags.get("stage") == "candidate"
        assert tags.get("owner") == "fli"

    def test_json_reports_what_landed(self, team_store, team_dir, chain, capsys):
        assert _promote(chain, team_store, format="json") == 0

        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "taxi/model"
        assert payload["copied"] == 2
        assert payload["alias_pending"] is False


class TestRefusals:
    def test_an_unreachable_store_is_not_silently_a_success(self, chain):
        """It raises rather than returning 0.

        Promoting is how a result reaches colleagues; a command that printed
        success while the store was unreachable would leave someone believing
        they had shared something.
        """
        with pytest.raises((RuntimeError, httpx.HTTPError)):
            _promote(chain, "http://127.0.0.1:1")

    def test_an_unreadable_artifact_is_refused(self, team_store, tmp_path):
        """A half-written blob has a provenance hash like any other."""
        from strata.artifact_cli import cmd_promote

        store = ArtifactStore(tmp_path / "local")
        store.create_artifact("building", "c3" * 32)

        rc = cmd_promote(
            argparse.Namespace(
                ref="building",
                artifact_dir=str(tmp_path / "local"),
                to_url=team_store,
                name="x",
                alias=None,
                tag=None,
                header=None,
                format="human",
                tenant=None,
                max_depth=10,
            )
        )

        assert rc == 1


class TestPublishPolicy:
    """What the cache offers outward, between "everything" and "nothing"."""

    class _Reached(Exception):
        """Raised in place of building a TeamStore, to say the gate let us by."""

    def _executor(self, policy: str | None, *, enabled: bool = True):
        """A stand-in with just the state the two gates read.

        The gates run before either method touches the session, so the parts
        of a real executor they never reach are not built here — but the
        methods themselves are the real ones, called unbound. Asserting on
        `_team_cache_publish_policy` alone would pass with both gates deleted.
        """
        from types import SimpleNamespace

        config = SimpleNamespace(
            notebook_team_cache_enabled=enabled,
            notebook_remote_store_url="http://store.example",
        )
        if policy is not None:
            config.notebook_team_cache_publish = policy
        return SimpleNamespace(
            _lake_config=lambda: config,
            _ambient_strata_headers=lambda: {},
            session=SimpleNamespace(
                dag=SimpleNamespace(consumed_variables={"c1": {"x"}}),
                environment_attestation_error=lambda: None,
            ),
        )

    def _talks_to_the_store(self, monkeypatch, executor, direction: str) -> bool:
        """Run one gate; report whether it reached the shared store."""
        from strata.notebook import executor as executor_module

        def _sentinel(*_args, **_kwargs):
            raise self._Reached()

        monkeypatch.setattr(executor_module, "TeamStore", _sentinel)
        if direction == "pull":
            call = executor_module.CellExecutor._pull_from_team_store(
                executor,
                cell_id="c1",
                provenance_hash="a" * 64,
                consumed_vars={"x"},
                source_hash="s",
                source="x = 1",
                env_hash="e",
                input_versions={},
            )
        else:
            call = executor_module.CellExecutor._push_to_team_store(executor, cell_id="c1")
        try:
            asyncio.run(call)
        except self._Reached:
            return True
        return False

    @pytest.mark.parametrize(
        "policy,offers,pulls",
        [
            ("all", True, True),
            ("promoted", False, True),
            ("off", False, False),
        ],
    )
    def test_the_policy_decides_each_direction(self, monkeypatch, policy, offers, pulls):
        """`promoted` still pulls: someone who shares only on purpose still
        benefits from work the team already did."""
        assert self._talks_to_the_store(monkeypatch, self._executor(policy), "push") is offers
        assert self._talks_to_the_store(monkeypatch, self._executor(policy), "pull") is pulls

    def test_a_config_without_the_setting_behaves_as_before(self, monkeypatch):
        """An older deployment's config object has no such attribute, and the
        absence must mean the behaviour that existed before the setting did."""
        assert self._talks_to_the_store(monkeypatch, self._executor(None), "push") is True
        assert self._talks_to_the_store(monkeypatch, self._executor(None), "pull") is True

    def test_the_switch_still_wins_over_the_policy(self, monkeypatch):
        """`all` describes what would be offered if the cache were on at all."""
        executor = self._executor("all", enabled=False)

        assert self._talks_to_the_store(monkeypatch, executor, "push") is False
        assert self._talks_to_the_store(monkeypatch, executor, "pull") is False

    def test_the_default_is_todays_behaviour(self):
        from strata.config import StrataConfig

        assert StrataConfig().notebook_team_cache_publish == "all"
