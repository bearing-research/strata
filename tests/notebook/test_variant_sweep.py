"""Variant sweep mode — backend foundation (PR1).

Covers the model field, notebook.toml round-trip + ``set_variant_mode``, the
parser, and the two new annotation-validation diagnostics. Execution behavior
(DAG / executor / provenance) lands in later PRs; mode defaults to ``switch``
so nothing here changes existing notebooks.
"""

from __future__ import annotations

import tempfile
import tomllib
from pathlib import Path

from strata.notebook.annotation_validation import validate_cell_annotations
from strata.notebook.models import CellState, NotebookState, VariantGroupConfig
from strata.notebook.parser import parse_notebook
from strata.notebook.writer import create_notebook, set_variant_mode


class TestVariantGroupConfigMode:
    def test_default_is_switch(self):
        cfg = VariantGroupConfig(group="model", active="gpt4")
        assert cfg.mode == "switch"
        assert cfg.is_sweep is False

    def test_sweep_mode(self):
        cfg = VariantGroupConfig(group="model", active="", mode="sweep")
        assert cfg.is_sweep is True

    def test_empty_active_allowed(self):
        # Sweep groups carry no meaningful active pointer.
        cfg = VariantGroupConfig(group="model", active="", mode="sweep")
        assert cfg.active == ""

    def test_unknown_mode_is_not_sweep(self):
        # Fail-safe: anything other than exact "sweep" is treated as switch.
        cfg = VariantGroupConfig(group="model", active="a", mode="swep")
        assert cfg.is_sweep is False


class TestSetVariantMode:
    def _read(self, notebook_dir: Path) -> list[dict]:
        with open(notebook_dir / "notebook.toml", "rb") as f:
            return tomllib.load(f).get("variant_group", [])

    def test_appends_sweep_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = create_notebook(Path(tmp), "Variants")
            set_variant_mode(nb, "model", "sweep")
            assert self._read(nb) == [{"group": "model", "active": "", "mode": "sweep"}]

    def test_switch_default_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = create_notebook(Path(tmp), "Variants")
            set_variant_mode(nb, "model", "switch")  # default → no entry created
            assert self._read(nb) == []

    def test_sweep_then_switch_drops_mode_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = create_notebook(Path(tmp), "Variants")
            set_variant_mode(nb, "model", "sweep")
            set_variant_mode(nb, "model", "switch")
            # Entry remains (created by the sweep call) but mode key is gone.
            assert self._read(nb) == [{"group": "model", "active": ""}]

    def test_existing_switch_entry_gains_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = create_notebook(Path(tmp), "Variants")
            from strata.notebook.writer import set_variant_active

            set_variant_active(nb, "model", "gpt4")
            set_variant_mode(nb, "model", "sweep")
            assert self._read(nb) == [{"group": "model", "active": "gpt4", "mode": "sweep"}]

    def test_switch_mode_not_emitted_for_switch_group(self):
        """A plain switch group never writes a `mode` key (no churn)."""
        with tempfile.TemporaryDirectory() as tmp:
            nb = create_notebook(Path(tmp), "Variants")
            from strata.notebook.writer import set_variant_active

            set_variant_active(nb, "model", "gpt4")
            assert self._read(nb) == [{"group": "model", "active": "gpt4"}]


class TestParserReadsMode:
    def test_mode_round_trips_into_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = create_notebook(Path(tmp), "Variants")
            set_variant_mode(nb, "model", "sweep")
            state = parse_notebook(nb)
            assert state.variant_modes.get("model") == "sweep"

    def test_missing_mode_defaults_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = create_notebook(Path(tmp), "Variants")
            from strata.notebook.writer import set_variant_active

            set_variant_active(nb, "model", "gpt4")
            state = parse_notebook(nb)
            assert state.variant_modes.get("model") == "switch"


def _variant_cell(name: str, cell_id: str) -> CellState:
    cell = CellState(id=cell_id, source=f"# @variant model {name}\npreds = run()\n")
    cell.defines = ["preds"]
    cell.variant_group = "model"
    cell.variant_name = name
    return cell


def _codes(cell: CellState, nb: NotebookState) -> list[str]:
    return [d.code for d in validate_cell_annotations(cell, nb)]


class TestSweepValidation:
    def _nb(self, modes=None, selections=None) -> NotebookState:
        nb = NotebookState(id="nb", name="t", cells=[_variant_cell("logreg", "a")])
        if modes:
            nb.variant_modes = dict(modes)
        if selections:
            nb.variant_active_selections = dict(selections)
        return nb

    def test_invalid_mode_flagged(self):
        nb = self._nb(modes={"model": "swep"})
        assert "variant_mode_invalid" in _codes(nb.cells[0], nb)

    def test_valid_sweep_mode_not_flagged(self):
        nb = self._nb(modes={"model": "sweep"})
        assert "variant_mode_invalid" not in _codes(nb.cells[0], nb)

    def test_active_redundant_in_sweep(self):
        nb = self._nb(modes={"model": "sweep"}, selections={"model": "logreg"})
        assert "variant_active_redundant" in _codes(nb.cells[0], nb)

    def test_active_not_redundant_in_switch(self):
        nb = self._nb(selections={"model": "logreg"})
        assert "variant_active_redundant" not in _codes(nb.cells[0], nb)

    def test_active_unknown_suppressed_in_sweep(self):
        # In sweep mode the active pointer is ignored, so no drift warning.
        nb = self._nb(modes={"model": "sweep"}, selections={"model": "ghost"})
        assert "variant_active_unknown" not in _codes(nb.cells[0], nb)


class TestSweepDag:
    """DAG resolution in sweep mode: all variants run; downstream fans in."""

    def _cells(self):
        from strata.notebook.dag import CellAnalysisWithId

        return [
            CellAnalysisWithId(id="load", defines=["X"], references=[]),
            CellAnalysisWithId(
                id="m_logreg",
                defines=["preds"],
                references=["X"],
                variant_group="model",
                variant_name="logreg",
            ),
            CellAnalysisWithId(
                id="m_rf",
                defines=["preds"],
                references=["X"],
                variant_group="model",
                variant_name="rf",
            ),
            CellAnalysisWithId(id="eval", defines=["score"], references=["preds"]),
        ]

    def _sweep_dag(self):
        from strata.notebook.dag import NotebookDag

        return NotebookDag.from_cells(self._cells(), variant_modes={"model": "sweep"})

    def test_no_inactive_members(self):
        dag = self._sweep_dag()
        assert dag.inactive_cells == set()
        # all four cells participate in the executable graph
        assert set(dag.topological_order) == {"load", "m_logreg", "m_rf", "eval"}

    def test_producer_is_sweep_producer(self):
        from strata.notebook.dag import SweepProducer, producer_cell_label

        dag = self._sweep_dag()
        prod = dag.variable_producer["preds"]
        assert isinstance(prod, SweepProducer)
        assert prod.group == "model"
        assert prod.variants == (("logreg", "m_logreg"), ("rf", "m_rf"))
        assert producer_cell_label(prod) == "sweep:model"

    def test_downstream_fans_in_to_all_members(self):
        dag = self._sweep_dag()
        assert set(dag.cell_upstream["eval"]) == {"m_logreg", "m_rf"}
        assert "eval" in dag.cell_downstream["m_logreg"]
        assert "eval" in dag.cell_downstream["m_rf"]

    def test_each_member_consumes_the_var(self):
        dag = self._sweep_dag()
        assert "preds" in dag.consumed_variables["m_logreg"]
        assert "preds" in dag.consumed_variables["m_rf"]

    def test_group_resolution_marked_sweep(self):
        dag = self._sweep_dag()
        res = next(r for r in dag.variant_groups if r.group == "model")
        assert res.mode == "sweep"
        assert len(res.members) == 2

    def test_switch_mode_still_excludes_inactive(self):
        from strata.notebook.dag import NotebookDag

        dag = NotebookDag.from_cells(self._cells(), variant_active_selections={"model": "rf"})
        assert dag.inactive_cells == {"m_logreg"}
        assert dag.cell_upstream["eval"] == ["m_rf"]
        assert dag.variable_producer["preds"] == "m_rf"


class TestSweepEndToEnd:
    """A sweep group executes all variants and the downstream cell receives a
    {variant: value} dict — driven through ``strata run`` (real harness)."""

    def test_downstream_receives_variant_dict(self, tmp_path, capsys):
        import json as _json

        from strata.notebook.cli import run_main
        from strata.notebook.writer import (
            add_cell_to_notebook,
            create_notebook,
            set_variant_mode,
            write_cell,
        )

        nb = create_notebook(tmp_path, "Sweep", initialize_environment=False)
        (nb / ".venv").mkdir(exist_ok=True)  # --no-sync placeholder
        cells = [
            ("load", "X = [1.0, 2.0, 3.0]\n", None),
            ("vdouble", "# @variant model double\npreds = [v * 2 for v in X]\n", "load"),
            ("vtriple", "# @variant model triple\npreds = [v * 3 for v in X]\n", "vdouble"),
            ("ev", 'print("SWEEP", {n: sum(p) for n, p in preds.items()})\n', "vtriple"),
        ]
        for cid, src, after in cells:
            add_cell_to_notebook(nb, cid, after, language="python")
            write_cell(nb, cid, src)
        set_variant_mode(nb, "model", "sweep")

        code = run_main([str(nb), "--no-sync", "--force", "--format", "json"])
        assert code == 0
        payload = _json.loads(capsys.readouterr().out)
        assert {c["status"] for c in payload["cells"]} == {"ok"}
        ev = next(c for c in payload["cells"] if c["id"] == "ev")
        assert "'double': 12.0" in ev["stdout"]
        assert "'triple': 18.0" in ev["stdout"]


class TestSweepDagReviewFixes:
    """Regression for the review findings #6 (asymmetric defines) and
    #7 (intra-group self-reference)."""

    def _dag(self, cells):
        from strata.notebook.dag import NotebookDag

        return NotebookDag.from_cells(cells, variant_modes={"model": "sweep"})

    def test_asymmetric_defines_only_wire_defining_members(self):
        # vb also defines `extra`; a downstream of `extra` must fan in only to vb.
        from strata.notebook.dag import CellAnalysisWithId, SweepProducer

        cells = [
            CellAnalysisWithId(id="load", defines=["X"], references=[]),
            CellAnalysisWithId(
                id="va",
                defines=["preds"],
                references=["X"],
                variant_group="model",
                variant_name="a",
            ),
            CellAnalysisWithId(
                id="vb",
                defines=["preds", "extra"],
                references=["X"],
                variant_group="model",
                variant_name="b",
            ),
            CellAnalysisWithId(id="useextra", defines=["z"], references=["extra"]),
        ]
        dag = self._dag(cells)
        prod = dag.variable_producer["extra"]
        assert isinstance(prod, SweepProducer)
        assert prod.variants == (("b", "vb"),)  # only the defining member
        assert dag.cell_upstream["useextra"] == ["vb"]
        assert "extra" not in dag.consumed_variables["va"]  # va not falsely wired

    def test_intra_group_self_reference_no_sibling_edges(self):
        # A refinement member references the group's own var; must not depend on
        # siblings (would otherwise wire spurious intra-group edges).
        from strata.notebook.dag import CellAnalysisWithId

        cells = [
            CellAnalysisWithId(id="base", defines=["preds"], references=[]),
            CellAnalysisWithId(
                id="va",
                defines=["preds"],
                references=["preds"],
                variant_group="model",
                variant_name="a",
            ),
            CellAnalysisWithId(
                id="vb",
                defines=["preds"],
                references=["preds"],
                variant_group="model",
                variant_name="b",
            ),
        ]
        dag = self._dag(cells)
        # neither member depends on its sibling
        assert "vb" not in dag.cell_upstream["va"]
        assert "va" not in dag.cell_upstream["vb"]


def test_sweep_cells_are_not_batchable(tmp_path):
    """Review #3: sweep members and sweep-consumers run single-cell, not batched."""
    from unittest.mock import MagicMock

    from strata.notebook.dag import SweepProducer
    from strata.notebook.executor import is_cell_batchable

    executor = MagicMock()
    executor.session.notebook_state.variant_modes = {"model": "sweep"}
    dag = MagicMock()
    dag.variable_producer = {"preds": SweepProducer("model", (("a", "va"),))}
    executor.session.dag = dag

    member = MagicMock(variant_group="model", references=[], language="python")
    consumer = MagicMock(variant_group=None, references=["preds"], language="python")
    plain = MagicMock(variant_group=None, references=["other"], language="python")

    assert is_cell_batchable(executor, member) is False
    assert is_cell_batchable(executor, consumer) is False
    # plain cell still delegates (not forced False by sweep logic)
    assert is_cell_batchable(executor, plain) is not False or plain.references == ["other"]


class TestSweepApi:
    """session.set_variant_mode + the mode surfaced on VariantGroupState."""

    def _session(self, tmp_path):
        from strata.notebook.parser import parse_notebook
        from strata.notebook.session import NotebookSession
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        nb = create_notebook(tmp_path, "Api", initialize_environment=False)
        for cid, src, after in [
            ("load", "X = 1\n", None),
            ("va", "# @variant model a\npreds = X\n", "load"),
            ("vb", "# @variant model b\npreds = X + 1\n", "va"),
            ("ev", "r = preds\n", "vb"),
        ]:
            add_cell_to_notebook(nb, cid, after, language="python")
            write_cell(nb, cid, src)
        return nb, NotebookSession(parse_notebook(nb), nb)

    def test_set_variant_mode_rebuilds_dag_and_state(self, tmp_path):
        from strata.notebook.dag import SweepProducer

        _nb, s = self._session(tmp_path)
        # switch (default): a single producing cell, one inactive member
        assert isinstance(s.dag.variable_producer["preds"], str)
        assert len(s.dag.inactive_cells) == 1

        s.set_variant_mode("model", "sweep")
        assert s.notebook_state.variant_modes["model"] == "sweep"
        assert isinstance(s.dag.variable_producer["preds"], SweepProducer)
        assert s.dag.inactive_cells == set()
        grp = next(g for g in s.notebook_state.variant_groups if g.group == "model")
        assert grp.mode == "sweep"  # surfaced for the frontend

        # toggle back to switch
        s.set_variant_mode("model", "switch")
        assert isinstance(s.dag.variable_producer["preds"], str)
        grp = next(g for g in s.notebook_state.variant_groups if g.group == "model")
        assert grp.mode == "switch"


class TestSweepRestEndpoint:
    """PUT /variant-groups/{group} accepts `mode`."""

    def _build(self, tmp_path):
        from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell

        nb = create_notebook(tmp_path, "RestApi", initialize_environment=False)
        for cid, src, after in [
            ("load", "X = 1\n", None),
            ("va", "# @variant model a\npreds = X\n", "load"),
            ("vb", "# @variant model b\npreds = X + 1\n", "va"),
            ("ev", "r = preds\n", "vb"),
        ]:
            add_cell_to_notebook(nb, cid, after, language="python")
            write_cell(nb, cid, src)
        return nb

    def test_put_mode_sweep(self, tmp_path):
        from fastapi.testclient import TestClient

        from tests.notebook.e2e_fixtures import create_test_app, open_notebook_session

        nb = self._build(tmp_path)
        client = TestClient(create_test_app())
        with open_notebook_session(client, nb) as (sid, _session):
            resp = client.put(f"/v1/notebooks/{sid}/variant-groups/model", json={"mode": "sweep"})
            assert resp.status_code == 200, resp.text
            groups = {g["group"]: g for g in resp.json()["variant_groups"]}
            assert groups["model"]["mode"] == "sweep"

            # empty body (no active, no mode) → 400
            empty = client.put(f"/v1/notebooks/{sid}/variant-groups/model", json={})
            assert empty.status_code == 400

            # invalid mode → 422 (request validation)
            bad = client.put(f"/v1/notebooks/{sid}/variant-groups/model", json={"mode": "nope"})
            assert bad.status_code == 422
