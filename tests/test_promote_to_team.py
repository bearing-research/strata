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


class TestPromoteRoute:
    """``POST /v1/notebooks/{id}/artifacts/{id}/v/{n}/promote`` — the same
    promotion the CLI does, from the button the strip will grow.

    The team store here is the server the fixture runs, and the notebook's own
    store is a directory on disk, which is the real shape: cells write locally
    and promotion is what crosses the gap. The route is called directly rather
    than over HTTP so the running server stays the *target*, not the caller.
    """

    def _session(self, chain):
        from types import SimpleNamespace

        manager = NotebookArtifactManager("nb", artifact_dir=chain["dir"])
        return SimpleNamespace(get_artifact_manager=lambda: manager)

    def _call(self, chain, url, monkeypatch, **body):
        from strata.notebook.routes import PromoteArtifactRequest, promote_notebook_artifact
        from strata.server import get_state

        monkeypatch.setattr(get_state().config, "notebook_remote_store_url", url)
        artifact_id = body.pop("artifact_id", chain["figure"].id)
        version = body.pop("version", chain["figure"].version)
        payload = {"name": "taxi/model"}
        payload.update(body)
        return asyncio.run(
            promote_notebook_artifact(
                "nb",
                self._session(chain),
                artifact_id,
                version,
                PromoteArtifactRequest(**payload),
            )
        )

    def test_it_promotes_the_chain_and_reports_where_it_landed(
        self, team_store, team_dir, chain, monkeypatch
    ):
        result = self._call(chain, team_store, monkeypatch)

        assert result["name"] == "taxi/model"
        assert result["copied"] == 2
        store = ArtifactStore(team_dir)
        assert store.get_artifact(chain["upstream"].id, chain["upstream"].version) is not None
        assert httpx.get(f"{team_store}/v1/names/taxi/model", timeout=10).status_code == 200

    def test_tags_travel(self, team_store, team_dir, chain, monkeypatch):
        self._call(chain, team_store, monkeypatch, tags={"stage": "candidate"})

        tags = ArtifactStore(team_dir).get_tags(chain["figure"].id, chain["figure"].version)
        assert tags.get("stage") == "candidate"

    def test_no_team_store_says_which_setting_is_missing(self, team_store, chain, monkeypatch):
        """A 409 naming the config beats a 500 the UI cannot explain."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as caught:
            self._call(chain, None, monkeypatch)

        assert caught.value.status_code == 409
        assert "notebook_remote_store_url" in caught.value.detail

    def test_an_artifact_this_notebook_does_not_hold_is_a_404(self, team_store, chain, monkeypatch):
        """The route reads this notebook's store, and only that one.

        The id arrives in the URL, so this is what keeps one open notebook
        from pushing an id it does not own to the team.
        """
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as caught:
            self._call(chain, team_store, monkeypatch, artifact_id="someone-elses", version=1)

        assert caught.value.status_code == 404

    def test_an_unreachable_store_is_a_bad_gateway(self, team_store, chain, monkeypatch):
        """Not a 500: the notebook server is fine, the team's store is not."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as caught:
            self._call(chain, "http://127.0.0.1:1", monkeypatch)

        assert caught.value.status_code == 502


class TestAmbientPromoteWiring:
    """What reaches a cell so ``strata.promote`` can work.

    Three pieces have to agree: the manifest carries the callback URL and the
    input URIs, and both execution paths — the cold harness and the warm pool
    worker — hand them to the client. A cell that runs on the pool is the
    default path, so a mismatch there is the failure nobody would see in a
    local test.
    """

    def _executor(self, *, remote_store: str | None, server_url: str = "http://nb.local"):
        from types import SimpleNamespace

        config = SimpleNamespace(
            notebook_remote_store_url=remote_store,
            notebook_remote_store_headers={},
            server_url=server_url,
        )
        return SimpleNamespace(
            _lake_config=lambda: config,
            session=SimpleNamespace(id="sess-1"),
        )

    def test_the_url_points_at_this_server_not_the_team_store(self):
        """The team store cannot read the notebook's artifacts; this server is
        the only process that can, so it is the one that does the copying."""
        from strata.notebook.executor import CellExecutor

        url = CellExecutor._ambient_promote_url(self._executor(remote_store="http://store.example"))

        assert url == "http://nb.local/v1/notebooks/sess-1"

    def test_without_a_team_store_there_is_no_callback_at_all(self):
        from strata.notebook.executor import CellExecutor

        assert CellExecutor._ambient_promote_url(self._executor(remote_store=None)) == ""

    def test_the_manifest_carries_it(self, tmp_path):
        """The link between the two tests above. Both consumers read
        ``strata_promote_url``; this is what puts it there."""
        from strata.notebook.executor import CellExecutor

        executor = self._executor(remote_store="http://store.example")
        executor._ambient_strata_url = lambda: "http://store.example"
        executor._ambient_strata_headers = lambda: {}
        executor._ambient_promote_url = lambda: CellExecutor._ambient_promote_url(executor)

        manifest_path = CellExecutor._write_manifest(
            executor,
            "x = 1",
            {"rows": {"uri": "strata://artifact/nb_rows@v=3", "file": "rows.json"}},
            tmp_path,
            {},
            {},
            cell_id="c2",
        )
        manifest = json.loads(manifest_path.read_text())

        assert manifest["strata_promote_url"] == "http://nb.local/v1/notebooks/sess-1"

    @pytest.mark.parametrize("path", ["harness", "pool_worker"])
    def test_both_execution_paths_hand_it_to_the_client(self, path, monkeypatch):
        import importlib

        module = importlib.import_module(f"strata.notebook.{path}")
        captured: dict = {}

        class _Client:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(module._client_mod, "StrataClient", _Client)
        inject = module.inject_client if path == "harness" else module._inject_client
        inject(
            {
                "strata_url": "http://store.example",
                "strata_cell_id": "c2",
                "strata_promote_url": "http://nb.local/v1/notebooks/sess-1",
                "inputs": {
                    "rows": {"uri": "strata://artifact/nb_rows@v=3", "file": "rows.json"},
                    "unstored": {"file": "x.pickle"},
                },
            },
            {},
        )

        assert captured["promote_url"] == "http://nb.local/v1/notebooks/sess-1"
        # An input with no artifact behind it (a mount, say) is not promotable
        # and must not appear as though it were.
        assert captured["inputs"] == {"rows": "strata://artifact/nb_rows@v=3"}
