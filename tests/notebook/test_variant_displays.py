"""A ``# @per_variant`` cell shows every variant's displays, in turn, each naming its variant.

Each instance stores its displays under its own variant, so one variant's never replaces
another's, and the fan-out gathers them for the cell.
"""

from __future__ import annotations

from pathlib import Path

from strata.notebook.executor import CellExecutor
from strata.notebook.ops import _cell_view_from_wire
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.tui.app import _render_outputs
from strata.notebook.tui.viewmodel import CellView
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    set_variant_mode,
    write_cell,
)

SHOWS = "# @per_variant\nf'val={val}'\n"


def _session(tmp_path: Path, cell: str = SHOWS) -> NotebookSession:
    """Variants one (val 1) and two (val 2) of group g, swept, and *cell* as ``c``."""
    nb = create_notebook(tmp_path, "Variants")
    after = None
    for cell_id, source in {
        "a": "# @variant g one\nval = 1\n",
        "b": "# @variant g two\nval = 2\n",
        "c": cell,
    }.items():
        add_cell_to_notebook(nb, cell_id, after)
        write_cell(nb, cell_id, source)
        after = cell_id
    set_variant_mode(nb, "g", "sweep")
    session = NotebookSession(parse_notebook(nb), nb)
    session.refresh_environment_runtime()
    return session


def _shown(outputs) -> list[tuple[str | None, object]]:
    return [
        (o["variant"], o["preview"]) if isinstance(o, dict) else (o.variant, o.preview)
        for o in outputs
    ]


async def _edit(session: NotebookSession, cell_id: str, source: str) -> None:
    session.notebook_state.get_cell(cell_id).source = source
    session.re_analyze_cell(cell_id)
    assert (await CellExecutor(session).execute_cell(cell_id, source)).success


class TestAPerVariantCellShowsEveryVariant:
    async def test_a_run_shows_each_variants_display_named(self, tmp_path):
        session = _session(tmp_path)

        result = await CellExecutor(session).execute_cell("c", SHOWS)

        assert result.success, result.error
        expected = [("one", "val=1"), ("two", "val=2")]
        assert _shown(result.display_outputs) == expected
        assert _shown(session.notebook_state.get_cell("c").display_outputs) == expected
        assert result.display_output["variant"] == "two"

    async def test_each_variant_keeps_its_own_artifact(self, tmp_path):
        session = _session(tmp_path)

        result = await CellExecutor(session).execute_cell("c", SHOWS)

        uris = [output["artifact_uri"] for output in result.display_outputs]
        assert "@variant=one@" in uris[0]
        assert "@variant=two@" in uris[1]

    async def test_a_hit_shows_them_again(self, tmp_path):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("c", SHOWS)).success

        again = await CellExecutor(session).execute_cell("c", SHOWS)

        assert again.cache_hit
        assert _shown(again.display_outputs) == [("one", "val=1"), ("two", "val=2")]
        assert _shown(session.notebook_state.get_cell("c").display_outputs) == [
            ("one", "val=1"),
            ("two", "val=2"),
        ]

    async def test_one_variant_run_again_shows_beside_the_other_hit(self, tmp_path):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("c", SHOWS)).success
        await _edit(session, "b", "# @variant g two\nval = 3\n")

        again = await CellExecutor(session).execute_cell("c", SHOWS)

        assert not again.cache_hit
        assert _shown(again.display_outputs) == [("one", "val=1"), ("two", "val=3")]

    async def test_a_reverted_variant_shows_its_earlier_display(self, tmp_path):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("c", SHOWS)).success
        await _edit(session, "b", "# @variant g two\nval = 3\n")
        assert (await CellExecutor(session).execute_cell("c", SHOWS)).success
        await _edit(session, "b", "# @variant g two\nval = 2\n")

        again = await CellExecutor(session).execute_cell("c", SHOWS)

        assert again.cache_hit
        assert _shown(again.display_outputs) == [("one", "val=1"), ("two", "val=2")]

    async def test_a_reverted_variant_with_an_output_shows_its_earlier_display(self, tmp_path):
        cell = "# @per_variant\nscaled = val * 10\nf'val={val}'\n"
        session = _session(tmp_path, cell)
        nb = session.path
        add_cell_to_notebook(nb, "d", "c")
        write_cell(nb, "d", "total = sum(scaled.values())\n")
        session = NotebookSession(parse_notebook(nb), nb)
        session.refresh_environment_runtime()
        assert (await CellExecutor(session).execute_cell("c", cell)).success
        await _edit(session, "b", "# @variant g two\nval = 3\n")
        assert (await CellExecutor(session).execute_cell("c", cell)).success
        await _edit(session, "b", "# @variant g two\nval = 2\n")

        again = await CellExecutor(session).execute_cell("c", cell)

        assert again.cache_hit
        assert _shown(again.display_outputs) == [("one", "val=1"), ("two", "val=2")]

    async def test_a_variant_that_shows_nothing_shows_nothing(self, tmp_path):
        cell = "# @per_variant\nif val == 2:\n    display(f'val={val}')\n"
        session = _session(tmp_path, cell)
        assert (await CellExecutor(session).execute_cell("c", cell)).success

        again = await CellExecutor(session).execute_cell("c", cell)

        assert again.cache_hit
        assert _shown(again.display_outputs) == [("two", "val=2")]

    async def test_they_are_there_after_a_reopen(self, tmp_path):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("c", SHOWS)).success

        reopened = NotebookSession(parse_notebook(session.path), session.path)
        reopened.compute_staleness()

        assert _shown(reopened.notebook_state.get_cell("c").display_outputs) == [
            ("one", "val=1"),
            ("two", "val=2"),
        ]

    async def test_a_failed_variant_leaves_nothing_shown(self, tmp_path):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("c", SHOWS)).success
        failing = "# @per_variant\nassert val == 1\nf'val={val}'\n"
        session.notebook_state.get_cell("c").source = failing
        session.re_analyze_cell("c")

        failed = await CellExecutor(session).execute_cell("c", failing)

        assert not failed.success
        assert failed.display_outputs == []
        assert session.notebook_state.get_cell("c").display_outputs == []
        reopened = NotebookSession(parse_notebook(session.path), session.path)
        assert reopened.notebook_state.get_cell("c").display_outputs == []


class TestEverySurfaceNamesTheVariant:
    async def test_the_agent_view_names_it(self, tmp_path):
        session = _session(tmp_path)
        assert (await CellExecutor(session).execute_cell("c", SHOWS)).success

        view = _cell_view_from_wire(session.notebook_state.get_cell("c").serialize())

        assert [(o.variant, o.preview) for o in view.outputs] == [
            ("one", "val=1"),
            ("two", "val=2"),
        ]

    def test_the_tui_names_each_variant_once(self):
        cell = CellView(
            id="c",
            display_outputs=[
                {"content_type": "json/object", "preview": "a1", "variant": "one"},
                {"content_type": "json/object", "preview": "a2", "variant": "one"},
                {"content_type": "json/object", "preview": "b", "variant": "two"},
            ],
        )

        assert _render_outputs(cell) == "[variant=one]\n\na1\n\na2\n\n[variant=two]\n\nb"

    def test_the_tui_names_nothing_for_a_plain_cell(self):
        cell = CellView(id="c", display_outputs=[{"content_type": "json/object", "preview": "x"}])

        assert _render_outputs(cell) == "x"
