"""A stale cell is explained by the parts its key was built from, against the same parts now."""

from __future__ import annotations

from pathlib import Path

import cellgraph as cg
import pytest

from strata.notebook import nonce
from strata.notebook.env import local_packages
from strata.notebook.executor import CellExecutor
from strata.notebook.inputs import InputFingerprints
from strata.notebook.keys import cell_key
from strata.notebook.runtime_state import load_runtime_state
from strata.notebook.session import NotebookSession, SessionManager
from strata.notebook.staleness_explanations import (
    CausalityChain,
    CausalityDetail,
    CausalityReason,
    CausalityType,
    StaleCell,
    explain_staleness,
    stale_reasons,
)
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


def _notebook(tmp_path: Path, **cells: str) -> Path:
    notebook_dir = create_notebook(tmp_path, "explained")
    after = None
    for cell_id, source in cells.items():
        add_cell_to_notebook(notebook_dir, cell_id, after_cell_id=after)
        write_cell(notebook_dir, cell_id, source)
        after = cell_id
    return notebook_dir


def _open(notebook_dir: Path) -> NotebookSession:
    return SessionManager().open_notebook(notebook_dir)


async def _run(session: NotebookSession, *cell_ids: str) -> None:
    executor = CellExecutor(session)
    for cell_id in cell_ids:
        result = await executor.execute_cell(
            cell_id, session.notebook_state.get_cell(cell_id).source
        )
        assert result.success, result.error


def _edit(session: NotebookSession, cell_id: str, source: str) -> None:
    write_cell(session.path, cell_id, source)
    session.notebook_state.get_cell(cell_id).source = source
    session.re_analyze_cell(cell_id)


def _explained(session: NotebookSession, cell_id: str) -> tuple[str, list[tuple], bool | None]:
    session.compute_staleness()
    chain = session.causality_map[cell_id]
    details = [(d.type.value, d.cell_id or d.input_name) for d in chain.details]
    return chain.reason.value, details, chain.incomplete


def _mounted(tmp_path: Path, name: str = "data") -> tuple[Path, str]:
    data = tmp_path / name
    data.mkdir()
    (data / "one.txt").write_text("one")
    return data, f"# @mount {name} file://{data} ro\n"


@pytest.mark.parametrize("reopened", [False, True])
async def test_a_changed_file_in_a_mount_is_not_called_a_source_change(tmp_path, reopened):
    data, mount = _mounted(tmp_path)
    session = _open(_notebook(tmp_path, c1=mount + "x = 1"))
    await _run(session, "c1")
    if reopened:
        session = _open(session.path)

    (data / "two.txt").write_text("two")

    assert _explained(session, "c1") == ("external", [("external_changed", "data")], None)


async def test_a_mounted_input_is_named_with_its_kind(tmp_path):
    data, mount = _mounted(tmp_path)
    session = _open(_notebook(tmp_path, c1=mount + "x = 1"))
    await _run(session, "c1")

    (data / "two.txt").write_text("two")
    session.compute_staleness()

    [detail] = session.causality_map["c1"].details
    assert (detail.input_kind, detail.input_name) == ("mount", "data")


async def test_a_stale_upstream_is_named_and_the_cell_own_unchanged_parts_are_not(tmp_path):
    data, mount = _mounted(tmp_path)
    other, other_mount = _mounted(tmp_path, "other")
    session = _open(_notebook(tmp_path, c1=mount + "x = 1", c2=other_mount + "y = x + 1"))
    await _run(session, "c1", "c2")

    (data / "two.txt").write_text("two")

    assert _explained(session, "c2") == ("upstream", [("input_changed", "c1")], None)


async def test_a_stale_upstream_does_not_hide_the_cell_own_change(tmp_path):
    data, mount = _mounted(tmp_path)
    other, other_mount = _mounted(tmp_path, "other")
    session = _open(_notebook(tmp_path, c1=mount + "x = 1", c2=other_mount + "y = x + 1"))
    await _run(session, "c1", "c2")

    (data / "two.txt").write_text("two")
    (other / "two.txt").write_text("two")

    assert _explained(session, "c2") == (
        "upstream",
        [("input_changed", "c1"), ("external_changed", "other")],
        None,
    )


async def test_an_upstream_that_ran_again_is_named_from_the_parts(tmp_path):
    session = _open(_notebook(tmp_path, c1="x = 1", c2="y = x + 1"))
    await _run(session, "c1", "c2")

    _edit(session, "c1", "x = 2")
    await _run(session, "c1")

    assert _explained(session, "c2") == ("upstream", [("input_changed", "c1")], None)


async def test_an_edit_that_declares_an_input_is_the_edit(tmp_path):
    _, mount = _mounted(tmp_path)
    session = _open(_notebook(tmp_path, c1="x = 1"))
    await _run(session, "c1")

    _edit(session, "c1", mount + "x = 2")

    assert _explained(session, "c1") == ("self", [("source_changed", "c1")], None)


async def test_an_input_that_cannot_be_resolved_is_unavailable(tmp_path):
    _, mount = _mounted(tmp_path)
    session = _open(_notebook(tmp_path, c1=mount + "x = 1"))
    await _run(session, "c1")
    cell = session.notebook_state.get_cell("c1")
    stored = cell.last_key_parts
    [[kind, name, _]] = stored["inputs"]
    unresolved = f"{name}:unresolved:{nonce.unresolved_nonce()}"

    [detail] = (
        explain_staleness(
            session, {"c1": StaleCell({**stored, "inputs": [[kind, name, unresolved]]})}
        )["c1"]
    ).details

    assert (detail.type, detail.input_kind, detail.input_name) == ("unavailable", "mount", "data")


class TestWhenTheResultIsKnownNotToMatch:
    """Stale is a claim about what the record shows moved; otherwise idle."""

    async def test_a_moved_input_is_stale(self, tmp_path):
        data, mount = _mounted(tmp_path)
        session = _open(_notebook(tmp_path, c1=mount + "x = 1"))
        await _run(session, "c1")
        (data / "one.txt").write_text("moved")

        staleness = session.compute_staleness()["c1"]

        assert staleness.status == "stale"
        assert staleness.reasons == ["external"]

    async def test_with_no_record_of_what_it_read_it_cannot_tell(self, tmp_path):
        """A run from before parts were saved: its key differs, and nothing says why."""
        data, mount = _mounted(tmp_path)
        session = _open(_notebook(tmp_path, c1=mount + "x = 1"))
        await _run(session, "c1")
        session.notebook_state.get_cell("c1").last_key_parts = None
        (data / "one.txt").write_text("moved")

        staleness = session.compute_staleness()["c1"]

        assert staleness.status == "idle"
        assert staleness.reasons == []

    @pytest.mark.parametrize(
        ("schema", "expected"), [(1, ("idle", [])), (None, ("stale", ["self"]))]
    )
    def test_a_record_from_another_key_scheme_was_not_edited(self, tmp_path, schema, expected):
        """After an upgrade the definition key is another formula's: the cell runs again,
        but nothing says its source changed. A record of this scheme still does."""
        session = _open(_notebook(tmp_path, c1="x = 1"))
        cell = session.notebook_state.get_cell("c1")
        old_key = "e" * 64
        cell.last_provenance_hash = "f" * 64
        cell.last_source_hash = old_key
        cell.last_env_hash = session.cell_env_hash(cell, local_packages(session.path))
        cell.last_key_parts = {
            "parts_schema": cg.PARTS_SCHEMA if schema is None else schema,
            "definition_key": old_key,
            "env_hash": cell.last_env_hash,
            "upstream": [],
            "inputs": [],
        }

        staleness = session.compute_staleness()["c1"]

        assert (staleness.status, staleness.reasons) == expected
        if schema is not None:
            chain = session.causality_map.get("c1")
            details = [d.type for d in chain.details] if chain is not None else []
            assert CausalityType.SOURCE_CHANGED not in details

    def test_an_input_that_cannot_be_read_says_nothing(self):
        chain = CausalityChain(
            reason=CausalityReason.EXTERNAL,
            details=[CausalityDetail(type=CausalityType.UNAVAILABLE, input_kind="table")],
        )

        assert stale_reasons(chain) == []

    def test_each_kind_of_change_has_its_reason(self):
        kinds = [
            CausalityType.SOURCE_CHANGED,
            CausalityType.INPUT_CHANGED,
            CausalityType.EXTERNAL_CHANGED,
            CausalityType.IDENTITY_CHANGED,
            CausalityType.ENV_CHANGED,
        ]
        chain = CausalityChain(
            reason=CausalityReason.UNKNOWN, details=[CausalityDetail(type=kind) for kind in kinds]
        )

        assert stale_reasons(chain) == ["self", "upstream", "external", "env"]


def test_every_unresolved_component_is_recognized():
    assert nonce.is_unresolved(f"a:table:unresolved:{nonce.unresolved_nonce()}")
    # A dataset keeps the first 32 digits.
    assert nonce.is_unresolved(f"a:dataset:ref:unresolved:{nonce.unresolved_nonce()[:32]}")
    assert not nonce.is_unresolved("a:file:///data:" + "0" * 64)


async def test_an_upstream_output_that_cannot_be_read_is_unavailable(tmp_path):
    """The key skips an output it cannot read, so its part is simply gone."""
    session = _open(_notebook(tmp_path, c1="x = 1", c2="y = x + 1"))
    await _run(session, "c1", "c2")
    stored = session.notebook_state.get_cell("c2").last_key_parts

    chain = explain_staleness(session, {"c2": StaleCell({**stored, "upstream": []})})["c2"]

    assert chain.reason == "upstream"
    assert [(d.type, d.cell_id) for d in chain.details] == [("unavailable", "c1")]


async def test_the_key_unchanged_says_it_cannot_tell(tmp_path):
    """The result is gone and nothing the key rests on moved. This used to read as an edit."""
    session = _open(_notebook(tmp_path, c1="x = 1", c2="y = x + 1"))
    await _run(session, "c1")
    store = session.artifact_manager.artifact_store
    artifact_id, version = session.parse_artifact_uri(
        session.notebook_state.get_cell("c1").artifact_uri
    )
    assert store.delete_artifact(artifact_id, version)

    assert _explained(session, "c1") == ("unknown", [], True)


async def test_a_record_from_before_parts_is_incomplete_never_a_source_change(tmp_path):
    data, mount = _mounted(tmp_path)
    session = _open(_notebook(tmp_path, c1=mount + "x = 1"))
    await _run(session, "c1")
    session.notebook_state.get_cell("c1").last_key_parts = None

    (data / "two.txt").write_text("two")

    assert _explained(session, "c1") == ("unknown", [], True)


async def test_a_record_from_before_parts_still_reports_what_it_kept(tmp_path):
    session = _open(_notebook(tmp_path, c1="x = 1"))
    await _run(session, "c1")
    session.notebook_state.get_cell("c1").last_key_parts = None

    _edit(session, "c1", "x = 2")

    assert _explained(session, "c1") == ("self", [("source_changed", "c1")], True)


async def test_parts_of_another_schema_are_not_read(tmp_path):
    data, mount = _mounted(tmp_path)
    session = _open(_notebook(tmp_path, c1=mount + "x = 1"))
    await _run(session, "c1")
    cell = session.notebook_state.get_cell("c1")
    cell.last_key_parts = {**cell.last_key_parts, "parts_schema": 999}

    (data / "two.txt").write_text("two")

    assert _explained(session, "c1") == ("unknown", [], True)


async def test_a_cell_with_no_key_now_is_incomplete(tmp_path):
    session = _open(_notebook(tmp_path, c1="x = 1"))
    await _run(session, "c1")

    chain = explain_staleness(session, {"c1": StaleCell(None)})["c1"]

    assert (chain.reason, chain.details, chain.incomplete) == ("unknown", [], True)


async def test_the_parts_are_saved_with_the_run(tmp_path):
    _, mount = _mounted(tmp_path)
    session = _open(_notebook(tmp_path, c1=mount + "x = 1", c2="y = x + 1"))
    await _run(session, "c1", "c2")

    saved = load_runtime_state(session.path).cells
    assert saved["c1"].last_key_parts["inputs"][0][:2] == ["mount", "data"]
    assert [address for address, _ in saved["c2"].last_key_parts["upstream"]] == ["output:c1.x"]
    assert saved["c1"].last_key_parts == session.notebook_state.get_cell("c1").last_key_parts


async def test_a_cache_hit_saves_the_parts_too(tmp_path):
    session = _open(_notebook(tmp_path, c1="x = 1"))
    await _run(session, "c1")
    session.notebook_state.get_cell("c1").last_key_parts = None

    result = await CellExecutor(session).execute_cell("c1", "x = 1")

    assert result.cache_hit
    assert session.notebook_state.get_cell("c1").last_key_parts is not None


async def test_the_wire_leaves_out_incomplete_when_complete(tmp_path):
    session = _open(_notebook(tmp_path, c1="x = 1"))
    await _run(session, "c1")
    _edit(session, "c1", "x = 2")
    session.compute_staleness()

    wire = session.serialize_cell(session.notebook_state.get_cell("c1"))

    assert wire["causality"] == {
        "reason": "self",
        "details": [{"type": "source_changed", "cell_id": "c1", "cell_name": "x"}],
    }
    assert "last_key_parts" not in wire


async def _sql_session(tmp_path: Path) -> tuple[NotebookSession, Path]:
    from tests.notebook.test_sql_cell_executor import (
        _build_notebook_with_sql_cell,
        _make_session,
        _seed_sqlite,
    )

    db_path = tmp_path / "warehouse.db"
    _seed_sqlite(db_path)
    notebook_dir = _build_notebook_with_sql_cell(
        tmp_path,
        db_path=db_path,
        cell_source="-- @sql connection=db\n-- @cache forever\nSELECT * FROM events\n",
    )
    session = _make_session(notebook_dir)
    await _run(session, "c1")
    return session, notebook_dir


async def test_a_sql_cell_is_explained_by_the_identity_it_recorded(tmp_path):
    from tests.notebook.test_sql_cell_executor import _make_session

    session, notebook_dir = await _sql_session(tmp_path)
    assert session.notebook_state.get_cell("c1").last_key_parts["inputs"] == []

    toml = notebook_dir / "notebook.toml"
    moved = tmp_path / "moved.db"
    (tmp_path / "warehouse.db").rename(moved)
    toml.write_text(toml.read_text().replace("warehouse.db", "moved.db"))
    session = _make_session(notebook_dir)

    assert _explained(session, "c1") == ("external", [("identity_changed", None)], None)


async def test_a_sql_cell_nothing_explains_is_incomplete(tmp_path):
    """Its own key rests on more than the parts and the identity."""
    session, _ = await _sql_session(tmp_path)
    stored = session.notebook_state.get_cell("c1").last_key_parts

    chain = explain_staleness(session, {"c1": StaleCell(stored)})["c1"]

    assert (chain.reason, chain.details, chain.incomplete) == ("unknown", [], True)


async def test_a_sql_cell_changed_input_is_named_from_the_parts_strata_supplies(tmp_path):
    """cellgraph leaves a host-keyed cell's inputs to the host, so Strata fills them in."""
    session, _ = await _sql_session(tmp_path)

    def parts(component: str) -> dict:
        key = cell_key(
            session,
            session.analysis,
            "c1",
            inputs=InputFingerprints(mounts=[f"raw:file:///data:{component}"]),
            env_hash="env",
        )
        return key.parts

    resolved = cg.Resolved(identity=None, fingerprint="raw:file:///data:one")
    assert parts("one")["inputs"] == [
        ["mount", "raw", cg.input_component("mount", "raw", resolved)]
    ]

    session.notebook_state.get_cell("c1").last_key_parts = parts("one")
    chain = explain_staleness(session, {"c1": StaleCell(parts("two"))})["c1"]

    assert [(d.type, d.input_kind, d.input_name) for d in chain.details] == [
        ("external_changed", "mount", "raw")
    ]
