"""What an agent can see and do with variant groups.

Round 5 drove a sweep entirely through the REST route and the DAG's producer
string, because the MCP surface had neither: no tool switched a variant or a
group's mode, and ``get_variable`` on a swept variable answered
``defined_in: "fanout:policy"`` and stopped there, naming no instance an agent
could then ask about.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from strata.notebook.mcp_server import _get_variable, _lineage, _set_variant
from strata.notebook.parser import parse_notebook
from strata.notebook.scopes import CLASSIFIED_TOOLS, NOTEBOOK_SCOPE_WRITE, required_scope_for_tool
from strata.notebook.session import NotebookSession, SessionManager
from strata.notebook.writer import set_variant_mode
from tests.notebook.test_cli import _build_notebook

CELLS = [
    ("load", "X = [1.0, 2.0, 3.0]\n", None),
    ("vdouble", "# @variant model double\npreds = [v * 2 for v in X]\n", "load"),
    ("vtriple", "# @variant model triple\npreds = [v * 3 for v in X]\n", "vdouble"),
    ("ev", "# @per_variant\nscore = sum(preds)\n", "vtriple"),
    ("report", "current = dict(score)\ncurrent\n", "ev"),
]


def _registered(nb: Path) -> tuple[SessionManager, str]:
    sm = SessionManager()
    session = NotebookSession(parse_notebook(nb), nb)
    sm._sessions[session.id] = session
    return sm, session.id


@pytest.fixture
def swept(tmp_path):
    nb = _build_notebook(tmp_path, cells=CELLS)
    set_variant_mode(nb, "model", "sweep")
    sm, session_id = _registered(nb)
    return sm, session_id, nb


def test_get_variable_names_the_instances_behind_a_swept_variable(swept):
    sm, session_id, _ = swept

    answer = _get_variable(sm, session_id, "score")

    assert answer["defined"] is True
    assert answer["defined_in"] == "fanout:model"
    # Each instance, and the exact spelling `lineage` takes for it.
    by_variant = {entry["variant"]: entry for entry in answer["variants"]}
    assert sorted(by_variant) == ["double", "triple"]
    assert by_variant["triple"]["cell_id"] == "ev"
    assert by_variant["triple"]["lineage_variable"] == "score@variant=triple"


def test_get_variable_names_the_member_cells_of_a_sweep_group(swept):
    """A sweep group's members are cells of their own; lineage takes the plain
    name on the member cell, not an ``@variant=`` spelling."""
    sm, session_id, _ = swept

    by_variant = {e["variant"]: e for e in _get_variable(sm, session_id, "preds")["variants"]}

    assert by_variant["double"]["cell_id"] == "vdouble"
    assert by_variant["double"]["lineage_variable"] == "preds"


def test_the_lineage_spelling_get_variable_hands_back_works(swept, tmp_path):
    """The point of naming it: an agent can pass it straight to `lineage`."""
    from strata.notebook.executor import CellExecutor

    sm, session_id, _ = swept
    session = sm.get_session(session_id)
    asyncio.run(
        CellExecutor(session).execute_cell(
            "report", session.notebook_state.get_cell("report").source
        )
    )

    spelling = next(
        e["lineage_variable"]
        for e in _get_variable(sm, session_id, "score")["variants"]
        if e["variant"] == "triple"
    )
    chain = _lineage(sm, session_id, "ev", spelling)

    assert chain["artifact_id"].endswith("@variant=triple")
    assert chain["steps"]


def test_set_variant_switches_the_active_one(tmp_path):
    nb = _build_notebook(tmp_path, cells=CELLS)
    sm, session_id = _registered(nb)

    result = asyncio.run(_set_variant(sm, session_id, "model", active="triple"))

    group = next(g for g in result["variant_groups"] if g["group"] == "model")
    assert group["active_name"] == "triple"
    assert group["mode"] == "switch"


def test_set_variant_switches_the_mode(tmp_path):
    nb = _build_notebook(tmp_path, cells=CELLS)
    sm, session_id = _registered(nb)

    result = asyncio.run(_set_variant(sm, session_id, "model", mode="sweep"))

    group = next(g for g in result["variant_groups"] if g["group"] == "model")
    assert group["mode"] == "sweep"


def test_set_variant_needs_something_to_set(tmp_path):
    nb = _build_notebook(tmp_path, cells=CELLS)
    sm, session_id = _registered(nb)

    with pytest.raises(ValueError, match="active"):
        asyncio.run(_set_variant(sm, session_id, "model"))
    with pytest.raises(ValueError, match="switch"):
        asyncio.run(_set_variant(sm, session_id, "model", mode="sideways"))


def test_set_variant_is_classified_as_a_write(tmp_path):
    assert "set_variant" in CLASSIFIED_TOOLS
    assert required_scope_for_tool("set_variant") == NOTEBOOK_SCOPE_WRITE


def test_a_fanout_consumer_records_every_instance_it_read(tmp_path):
    """Lineage walks the inputs an artifact recorded.

    A fan-out cell keeps one URI per variable, whichever variant stored last,
    so a consumer that read every instance recorded one of them and its
    lineage showed a single variant behind a dict built from all of them.
    """
    from strata.notebook.executor import CellExecutor

    nb = _build_notebook(tmp_path, cells=CELLS)
    set_variant_mode(nb, "model", "sweep")
    session = NotebookSession(parse_notebook(nb), nb)
    asyncio.run(
        CellExecutor(session).execute_cell(
            "report", session.notebook_state.get_cell("report").source
        )
    )

    refs = session._collect_input_refs("report")

    variants = sorted(ref.split("@variant=")[1].split("@v=")[0] for ref in refs.values())
    assert variants == ["double", "triple"]


def test_set_variant_refuses_a_group_no_cell_declares(tmp_path):
    """The writer appends an entry for whatever it is given.

    A typo would otherwise add a junk `[[variant_group]]` block to the
    *committed* notebook.toml, report success, and leave the notebook on the
    variant it was already running.
    """
    nb = _build_notebook(tmp_path, cells=CELLS)
    sm, session_id = _registered(nb)
    before = (nb / "notebook.toml").read_text()

    with pytest.raises(ValueError, match="no variant group 'polcy'"):
        asyncio.run(_set_variant(sm, session_id, "polcy", active="triple"))

    assert (nb / "notebook.toml").read_text() == before


def test_set_variant_refuses_a_variant_the_group_does_not_have(tmp_path):
    """An unknown name persists, the DAG falls back to the first variant in
    source order, and the answer would say the unknown one was selected."""
    nb = _build_notebook(tmp_path, cells=CELLS)
    sm, session_id = _registered(nb)

    with pytest.raises(ValueError, match="no variant 'quadruple'"):
        asyncio.run(_set_variant(sm, session_id, "model", active="quadruple"))
    with pytest.raises(ValueError, match="no variant ''"):
        asyncio.run(_set_variant(sm, session_id, "model", active=""))


def test_set_variant_tells_an_attached_viewer(tmp_path, monkeypatch):
    """Every other notebook-mutating tool reloads and broadcasts; a viewer that
    misses this one keeps the old tab strip and pre-switch staleness badges."""
    import strata.notebook.mcp_server as mcp_server

    nb = _build_notebook(tmp_path, cells=CELLS)
    sm, session_id = _registered(nb)
    broadcast: list[str] = []
    notes: list[str] = []

    async def _fake_broadcast(sid, session):
        broadcast.append(sid)

    async def _fake_note(sid, source, text):
        notes.append(text)

    monkeypatch.setattr(mcp_server, "_sync_and_broadcast", _fake_broadcast)
    monkeypatch.setattr(mcp_server, "_agent_note", _fake_note)

    asyncio.run(_set_variant(sm, session_id, "model", mode="sweep"))

    assert broadcast == [session_id]
    assert notes and "model" in notes[0]


def test_a_chained_instance_records_the_variant_it_zipped_to(tmp_path):
    """A chained `# @per_variant` cell binds one upstream instance as a scalar.

    Recording the whole set made lineage on `score2@variant=double` name
    `score@variant=triple` as an ancestor.
    """
    from strata.notebook.executor import CellExecutor

    cells = [
        ("load", "X = [1.0, 2.0, 3.0]\n", None),
        ("vdouble", "# @variant model double\npreds = [v * 2 for v in X]\n", "load"),
        ("vtriple", "# @variant model triple\npreds = [v * 3 for v in X]\n", "vdouble"),
        ("ev", "# @per_variant\nscore = sum(preds)\n", "vtriple"),
        ("ev2", "# @per_variant\nscore2 = score * 2\n", "ev"),
        ("report", "current = dict(score2)\ncurrent\n", "ev2"),
    ]
    nb = _build_notebook(tmp_path, cells=cells)
    set_variant_mode(nb, "model", "sweep")
    session = NotebookSession(parse_notebook(nb), nb)
    asyncio.run(
        CellExecutor(session).execute_cell(
            "report", session.notebook_state.get_cell("report").source
        )
    )

    # Each instance of the chained cell records only its own upstream instance.
    for variant in ("double", "triple"):
        refs = session._collect_input_refs("ev2", variant=variant)
        named = sorted(r.split("@variant=")[1].split("@v=")[0] for r in refs.values())
        assert named == [variant], (variant, named)

    # The collapse consumer below it still records every instance it read.
    collapsed = session._collect_input_refs("report")
    assert sorted(r.split("@variant=")[1].split("@v=")[0] for r in collapsed.values()) == [
        "double",
        "triple",
    ]
