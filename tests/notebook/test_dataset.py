"""``@dataset``: a registry name, resolved to one version, as a cell input."""

from __future__ import annotations

import hashlib
import io
import json
import types

import pyarrow as pa
import pytest

from strata.artifact_store import TransformSpec
from strata.notebook import datasets as datasets_module
from strata.notebook.annotations import parse_annotations


def _store_version(
    store,
    blob: bytes,
    *,
    content_type: str | None,
    artifact_id="taxi-model",
    input_versions=None,
    provenance_hash=None,
):
    params = {"content_type": content_type} if content_type else {}
    version = store.create_artifact(
        artifact_id=artifact_id,
        provenance_hash=provenance_hash or hashlib.sha256(blob + artifact_id.encode()).hexdigest(),
        transform_spec=TransformSpec(executor="test@v1", params=params, inputs=[]),
        input_versions=input_versions,
    )
    store.write_blob(artifact_id, version, blob)
    store.finalize_artifact(
        artifact_id,
        version,
        schema_json="{}",
        row_count=0,
        byte_size=len(blob),
        content_sha256=hashlib.sha256(blob).hexdigest(),
    )
    return version


def _json_version(store, value: dict) -> int:
    return _store_version(store, json.dumps(value).encode(), content_type="json/object")


def _model_with_history(store) -> None:
    """``taxi/model@champion``, made from ``taxi/features`` (read by name), made from ``raw``."""
    raw = _store_version(store, b'{"r": 1}', content_type="json/object", artifact_id="raw")
    features = _store_version(
        store,
        b'{"f": 1}',
        content_type="json/object",
        artifact_id="features",
        input_versions={f"strata://artifact/raw@v={raw}": f"raw@v={raw}"},
    )
    store.set_name("taxi/features", "features", features)
    model = _store_version(
        store,
        b'{"t": 3}',
        content_type="json/object",
        input_versions={"strata://name/taxi/features": f"features@v={features}"},
    )
    store.set_alias("taxi/model", "champion", "taxi-model", model)


def _chain(tree: dict) -> list[str]:
    """The artifact refs down a lineage tree that has one input per step."""
    refs = []
    while "artifact_id" in tree:
        refs.append(f"{tree['artifact_id']}@v={tree['version']}")
        if len(tree["inputs"]) != 1:
            break
        (tree,) = tree["inputs"]
    return refs


class TestAnnotation:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("model taxi/model@champion", ("model", "taxi/model", "champion", None)),
            ("model taxi/model@v=3", ("model", "taxi/model", None, 3)),
            ("model taxi/model", ("model", "taxi/model", None, None)),
        ],
    )
    def test_the_declaration_parses(self, value, expected):
        (spec,) = parse_annotations(f"# @dataset {value}\nx = 1").datasets

        assert (spec.name, spec.dataset, spec.alias, spec.version) == expected
        assert spec.reference == value.split()[1]

    @pytest.mark.parametrize(
        "value",
        [
            "model",
            "1m taxi/model",
            "m taxi/model@",
            "m taxi/model@v=0",
            "m taxi/model@v=x",
            "m a b",
        ],
    )
    def test_a_malformed_declaration_is_ignored(self, value):
        assert parse_annotations(f"# @dataset {value}\nx = 1").datasets == []


@pytest.fixture
def notebook(tmp_path, notebook_personal_server):
    """A notebook whose first cell reads a dataset; the server's own store is the registry."""
    from strata.notebook.parser import parse_notebook
    from strata.notebook.session import NotebookSession
    from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

    def _make(source: str):
        nb = create_notebook(tmp_path / "nb", "Reading a dataset")
        add_cell_to_notebook(nb, "c1", None)
        write_cell(nb, "c1", source)
        add_cell_to_notebook(nb, "c2", "c1")
        write_cell(nb, "c2", "doubled = score * 2")
        add_cell_to_notebook(nb, "c3", "c2")
        write_cell(nb, "c3", "shown = doubled")
        session = NotebookSession(parse_notebook(nb), nb)
        session.refresh_environment_runtime()
        return session

    return _make


def _status(session, cell_id: str) -> str:
    return session.compute_staleness()[cell_id].status.value


class TestInACell:
    async def test_the_cell_goes_stale_when_the_alias_moves(
        self, notebook, notebook_personal_server, monkeypatch
    ):
        from strata.notebook.executor import CellExecutor

        registry = notebook_personal_server["artifact_store"]
        registry.set_alias(
            "taxi/model", "champion", "taxi-model", _json_version(registry, {"t": 1})
        )
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)

        first = await CellExecutor(session).execute_cell("c1", source)
        assert first.success, first.error
        assert first.outputs["score"]["preview"] == 1
        assert _status(session, "c1") == "ready"

        registry.set_alias(
            "taxi/model", "champion", "taxi-model", _json_version(registry, {"t": 2})
        )
        # Inside the recheck interval the registry is not asked again.
        assert _status(session, "c1") == "ready"
        monkeypatch.setattr(datasets_module, "STALE_CHECK_SECONDS", 0.0)
        assert _status(session, "c1") != "ready"

        moved = await CellExecutor(session).execute_cell("c1", source)
        assert moved.success, moved.error
        assert moved.cache_hit is False
        assert moved.outputs["score"]["preview"] == 2
        assert _status(session, "c1") == "ready"

    async def test_a_run_refreshes_what_staleness_last_saw(
        self, notebook, notebook_personal_server
    ):
        from strata.notebook.executor import CellExecutor

        registry = notebook_personal_server["artifact_store"]
        registry.set_alias(
            "taxi/model", "champion", "taxi-model", _json_version(registry, {"t": 1})
        )
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)
        assert _status(session, "c1") != "ready"

        registry.set_alias(
            "taxi/model", "champion", "taxi-model", _json_version(registry, {"t": 2})
        )
        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.outputs["score"]["preview"] == 2
        assert _status(session, "c1") == "ready"

    async def test_a_pinned_version_never_goes_stale(
        self, notebook, notebook_personal_server, monkeypatch
    ):
        from strata.notebook.executor import CellExecutor

        registry = notebook_personal_server["artifact_store"]
        registry.set_name("taxi/model", "taxi-model", _json_version(registry, {"t": 1}))
        source = '# @dataset model taxi/model@v=1\nscore = model["t"]'
        session = notebook(source)
        monkeypatch.setattr(datasets_module, "STALE_CHECK_SECONDS", 0.0)

        first = await CellExecutor(session).execute_cell("c1", source)
        assert first.success, first.error

        registry.set_name("taxi/model", "taxi-model", _json_version(registry, {"t": 2}))
        assert _status(session, "c1") == "ready"
        again = await CellExecutor(session).execute_cell("c1", source)
        assert again.cache_hit is True

    async def test_a_name_that_does_not_resolve_fails_the_cell(self, notebook):
        from strata.notebook.executor import CellExecutor

        source = "# @dataset model taxi/model@champion\nscore = 1"
        session = notebook(source)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success is False
        assert "taxi/model@champion is not in the registry" in result.error
        assert _status(session, "c1") != "ready"

    @pytest.mark.parametrize(
        ("content_type", "blob", "expression", "expected"),
        [
            ("pickle/object", None, "type(model).__name__", "Fraction"),
            (None, None, "type(model).__name__ + str(len(model))", "DataFrame2"),
            ("image/png", b"\x89PNG-bytes", "model.read_bytes().decode('latin-1')", None),
        ],
    )
    async def test_the_value_is_bound_by_its_content_type(
        self, notebook, notebook_personal_server, content_type, blob, expression, expected
    ):
        from strata.notebook.executor import CellExecutor

        if content_type == "pickle/object":
            import fractions
            import pickle

            blob = pickle.dumps(fractions.Fraction(1, 3))
        elif content_type is None:
            # A core artifact: Arrow IPC bytes, no content type recorded.
            sink = io.BytesIO()
            table = pa.table({"trip": [1, 2]})
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            blob = sink.getvalue()
        registry = notebook_personal_server["artifact_store"]
        registry.set_name(
            "taxi/model", "taxi-model", _store_version(registry, blob, content_type=content_type)
        )
        source = f"# @dataset model taxi/model\nscore = {expression}"
        session = notebook(source)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success, result.error
        assert result.outputs["score"]["preview"] == (expected or blob.decode("latin-1"))

    async def test_a_loop_cell_refuses_a_dataset(self, notebook):
        from strata.notebook.executor import CellExecutor

        source = "# @loop max_iter=2 carry=score\n# @dataset model taxi/model\nscore = score + 1"
        session = notebook(source)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success is False
        assert "@dataset is not supported on loop cells" in result.error

    async def test_an_r_cell_refuses_a_dataset(self, notebook):
        import time

        from strata.notebook.executor import CellExecutor

        source = "# @dataset model taxi/model\nscore <- 1"
        session = notebook(source)

        result = await CellExecutor(session)._execute_r_cell(
            "c1", source, 30.0, time.time(), materialize_upstreams=False, use_cache=True
        )

        assert result.success is False
        assert "@dataset is not supported on R cells" in result.error

    def test_a_dataset_cell_is_not_batched(self, notebook):
        from strata.notebook.executor import CellExecutor, is_cell_batchable

        session = notebook("# @dataset model taxi/model\nscore = 1")
        cell = session.notebook_state.get_cell("c1")

        assert is_cell_batchable(CellExecutor(session), cell) is False


class TestLineage:
    async def test_downstream_lineage_reaches_the_named_version(
        self, notebook, notebook_personal_server
    ):
        from strata.artifact_cli import _walk_lineage
        from strata.notebook.executor import CellExecutor
        from strata.services.artifact import ArtifactService

        registry = notebook_personal_server["artifact_store"]
        _json_version(registry, {"t": 1})
        registry.set_alias(
            "taxi/model", "champion", "taxi-model", _json_version(registry, {"t": 2})
        )
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)

        upstream = await CellExecutor(session).execute_cell("c1", source)
        downstream = await CellExecutor(session).execute_cell("c2", "doubled = score * 2")
        assert upstream.success, upstream.error
        assert downstream.success, downstream.error

        manager = session.get_artifact_manager()
        ((_, scored),) = manager.list_cell_artifacts("c1")
        graph = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=scored,
            artifact_id=scored.id,
            version=scored.version,
            tenant_filter=None,
            max_depth=5,
        )
        assert graph.direct_inputs == ["strata://name/taxi/model@champion"]
        assert ("taxi-model", 2) in {(n.artifact_id, n.version) for n in graph.nodes}

        ((_, doubled),) = manager.list_cell_artifacts("c2")
        tree = _walk_lineage(manager.artifact_store, doubled, max_depth=5)
        (score_node,) = tree["inputs"]
        (model_node,) = score_node["inputs"]
        assert (model_node["artifact_id"], model_node["version"]) == ("taxi-model", 2)

    async def test_lineage_reaches_the_steps_behind_the_dataset(
        self, notebook, notebook_personal_server
    ):
        from strata.artifact_cli import _walk_lineage
        from strata.notebook.executor import CellExecutor
        from strata.services.artifact import ArtifactService

        _model_with_history(notebook_personal_server["artifact_store"])
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)

        upstream = await CellExecutor(session).execute_cell("c1", source)
        downstream = await CellExecutor(session).execute_cell("c2", "doubled = score * 2")
        assert upstream.success, upstream.error
        assert downstream.success, downstream.error

        manager = session.get_artifact_manager()
        ((_, scored),) = manager.list_cell_artifacts("c1")
        graph = ArtifactService().build_lineage(
            manager.artifact_store,
            artifact=scored,
            artifact_id=scored.id,
            version=scored.version,
            tenant_filter=None,
            max_depth=5,
        )
        steps = {(n.artifact_id, n.version) for n in graph.nodes if n.transform_ref}
        assert {("taxi-model", 1), ("features", 1), ("raw", 1)} <= steps

        ((_, doubled),) = manager.list_cell_artifacts("c2")
        tree = _walk_lineage(manager.artifact_store, doubled, max_depth=5)
        assert _chain(tree)[1:] == [
            f"{scored.id}@v={scored.version}",
            "taxi-model@v=1",
            "features@v=1",
            "raw@v=1",
        ]

    def test_an_ancestor_the_notebook_already_holds_is_reused(self, tmp_path):
        from strata.artifact_cli import _walk_lineage
        from strata.artifact_store import ArtifactStore
        from strata.notebook.models import DatasetSpec

        registry = ArtifactStore(tmp_path / "registry")
        _model_with_history(registry)
        notebook_store = ArtifactStore(tmp_path / "notebook")
        # The same computation as ``features``, already here under another id.
        features = registry.get_artifact("features", 1)
        _store_version(
            notebook_store,
            b'{"f": 1}',
            content_type="json/object",
            artifact_id="features-here",
            provenance_hash=features.provenance_hash,
        )
        local = datasets_module.LocalRegistry(registry)
        resolved = local.resolve(DatasetSpec(name="model", dataset="taxi/model", alias="champion"))

        copied = datasets_module.copy_into(local, resolved, notebook_store)

        model = notebook_store.get_artifact("taxi-model", 1)
        assert copied.local_ref == "taxi-model@v=1"
        assert json.loads(model.input_versions) == {
            "strata://name/taxi/features": "features-here@v=1"
        }
        assert notebook_store.get_artifact("features", 1) is None
        # The row it landed on takes the lineage it lacked.
        assert _chain(_walk_lineage(notebook_store, model, max_depth=5)) == [
            "taxi-model@v=1",
            "features-here@v=1",
            "raw@v=1",
        ]


class TestReaders:
    """The registry dashboard lists, per name, the notebook cells that read it."""

    async def test_the_dashboard_lists_the_cells_that_read_a_name(
        self, notebook, notebook_personal_server
    ):
        import httpx

        from strata.artifact_transfer import copy_chain
        from strata.notebook.executor import CellExecutor
        from strata.notebook.routes import list_notebook_published_artifacts

        registry = notebook_personal_server["artifact_store"]
        version = _json_version(registry, {"t": 1})
        registry.set_name("taxi/model", "taxi-model", version)
        registry.set_alias("taxi/model", "champion", "taxi-model", version)
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)
        result = await CellExecutor(session).execute_cell("c1", source)
        assert result.success, result.error
        notebook_id = session.notebook_state.id
        reader = {"notebook_id": notebook_id, "cell_id": "c1", "reference": "taxi/model@champion"}

        # The notebook's own results, before anything leaves it.
        listed = await list_notebook_published_artifacts(session.id, session)
        assert listed["readers"] == {"taxi/model": [reader]}

        # Once a result reaches the registry's store, every notebook sees the read.
        manager = session.get_artifact_manager()
        ((_, scored),) = manager.list_cell_artifacts("c1")
        copy_chain(manager.artifact_store, registry, scored, max_depth=5)
        response = httpx.get(f"{notebook_personal_server['base_url']}/v1/registry/summary")
        rows = {row["name"]: row for row in response.json()["names"]}
        assert rows["taxi/model"]["readers"] == [reader]


class TestARemoteRegistry:
    """With ``notebook_remote_store_url`` set, the name resolves in that store."""

    @staticmethod
    def _point_at(monkeypatch, url: str):
        from strata.notebook.executor import CellExecutor
        from strata.notebook.session import NotebookSession

        config = types.SimpleNamespace(notebook_remote_store_url=url)
        monkeypatch.setattr(CellExecutor, "_lake_config", lambda self: config)
        monkeypatch.setattr(NotebookSession, "_lake_config", lambda self: config)

    async def test_the_version_is_resolved_and_copied_over_http(
        self, notebook, notebook_personal_server, monkeypatch
    ):
        from strata.notebook.executor import CellExecutor

        registry = notebook_personal_server["artifact_store"]
        registry.set_alias(
            "taxi/model", "champion", "taxi-model", _json_version(registry, {"t": 7})
        )
        self._point_at(monkeypatch, notebook_personal_server["base_url"])
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success, result.error
        assert result.outputs["score"]["preview"] == 7
        copied = session.get_artifact_manager().artifact_store.get_artifact("taxi-model", 1)
        assert copied is not None
        assert copied.provenance_hash == registry.get_artifact("taxi-model", 1).provenance_hash
        assert _status(session, "c1") == "ready"

    async def test_the_chain_behind_the_dataset_is_copied_over_http(
        self, notebook, notebook_personal_server, monkeypatch
    ):
        from strata.artifact_cli import _walk_lineage
        from strata.notebook.executor import CellExecutor

        _model_with_history(notebook_personal_server["artifact_store"])
        self._point_at(monkeypatch, notebook_personal_server["base_url"])
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success, result.error
        store = session.get_artifact_manager().artifact_store
        ((_, scored),) = session.get_artifact_manager().list_cell_artifacts("c1")
        assert _chain(_walk_lineage(store, scored, max_depth=5))[1:] == [
            "taxi-model@v=1",
            "features@v=1",
            "raw@v=1",
        ]

    async def test_an_unreachable_registry_fails_the_cell_rather_than_reading_locally(
        self, notebook, notebook_personal_server, monkeypatch
    ):
        from strata.notebook.executor import CellExecutor

        registry = notebook_personal_server["artifact_store"]
        registry.set_alias(
            "taxi/model", "champion", "taxi-model", _json_version(registry, {"t": 7})
        )
        self._point_at(monkeypatch, "http://127.0.0.1:9")
        source = '# @dataset model taxi/model@champion\nscore = model["t"]'
        session = notebook(source)

        result = await CellExecutor(session).execute_cell("c1", source)

        assert result.success is False
        assert "unreachable" in result.error
