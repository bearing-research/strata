"""Smoke corpus runner for Jupyter notebook import.

PR 5 of the Jupyter interop work. Each fixture under
``tests/notebook/jupyter_corpus/smoke/`` is a hand-crafted ``.ipynb``
that exercises a different facet of the converter — pandas/numpy/
matplotlib/sklearn idioms, ``%pip install`` capture, ``;``-suppression,
variable rebinding, etc.

Scoring rubric (per the design doc):

    parse:    can we read the .ipynb at all?
    convert:  did jupyter_import produce a valid Strata notebook dir?
    dag:      does the DAG build with no cycles / unbound references?
    run:      does `strata run` complete with no exceptions?
    artifact: do leaf cells produce non-empty artifacts?

Fast mode (default): parse + convert + dag. Network-free, ~1s per
notebook, runs on every PR. Catches converter regressions, magic
table breakage, DAG analysis issues.

Full mode (``STRATA_CORPUS_RUN=1``): also exercises ``strata run``
through the harness. Slow + network-dependent — opt-in only,
intended for nightly CI or manual pre-release verification.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from strata.notebook.analyzer import analyze_cell
from strata.notebook.dag import CellAnalysisWithId, build_dag
from strata.notebook.jupyter_import import import_notebook
from strata.notebook.parser import parse_notebook

_SMOKE_DIR = Path(__file__).parent / "jupyter_corpus" / "smoke"
_SMOKE_NOTEBOOKS = sorted(_SMOKE_DIR.glob("*.ipynb"))


@dataclass
class CorpusScore:
    """Result of scoring one notebook through the rubric.

    Each step is gated by the previous: ``convert`` is only checked
    when ``parse`` passed, etc. The first failing step is recorded in
    ``failed_at`` so test output points the reader at the right place.
    """

    notebook: str
    parse: bool = False
    convert: bool = False
    dag: bool = False
    failed_at: str = ""
    error: str = ""
    # PR 6+: filled in when STRATA_CORPUS_RUN=1.
    run: bool | None = None
    artifact: bool | None = None
    # Bookkeeping surfaced for verifying the converter actually did work
    # (e.g. captured the %pip install line, dropped the %matplotlib).
    magics_translated: int = 0
    magics_dropped: int = 0
    deps_captured: list[str] = field(default_factory=list)


def _score(ipynb_path: Path, out_dir: Path) -> CorpusScore:
    """Score one notebook through parse → convert → dag."""
    score = CorpusScore(notebook=ipynb_path.name)

    # parse
    try:
        with ipynb_path.open("r", encoding="utf-8") as f:
            nb = json.load(f)
        if not isinstance(nb, dict) or "cells" not in nb:
            raise ValueError("Not an nbformat object (missing cells)")
        score.parse = True
    except Exception as exc:
        score.failed_at = "parse"
        score.error = f"{type(exc).__name__}: {exc}"
        return score

    # convert
    try:
        result = import_notebook(ipynb_path, out_dir=out_dir)
        score.convert = True
        score.magics_translated = len(result.translated_magics)
        score.magics_dropped = len(result.dropped_magics)
        score.deps_captured = list(result.captured_deps)
    except Exception as exc:
        score.failed_at = "convert"
        score.error = f"{type(exc).__name__}: {exc}"
        return score

    # dag — analyze each python cell, build the dependency graph. This
    # mirrors what session._analyze_and_build_dag does for python cells,
    # without spinning up a Session (which would drag in the process
    # pool and venv machinery).
    try:
        nb_state = parse_notebook(result.notebook_dir)
        analyses: list[CellAnalysisWithId] = []
        for cell in nb_state.cells:
            if cell.language != "python":
                continue
            cell_analysis = analyze_cell(cell.source)
            if cell_analysis.error:
                raise ValueError(f"analyze_cell failed for {cell.id}: {cell_analysis.error}")
            analyses.append(
                CellAnalysisWithId(
                    id=cell.id,
                    defines=cell_analysis.defines,
                    references=cell_analysis.references,
                )
            )
        build_dag(analyses)  # raises on cycles
        score.dag = True
    except Exception as exc:
        score.failed_at = "dag"
        score.error = f"{type(exc).__name__}: {exc}"

    return score


def _full_mode_enabled() -> bool:
    return os.environ.get("STRATA_CORPUS_RUN", "0") not in ("", "0", "false", "False")


@pytest.mark.skipif(not _SMOKE_NOTEBOOKS, reason="smoke corpus directory is empty")
@pytest.mark.parametrize("notebook_path", _SMOKE_NOTEBOOKS, ids=lambda p: p.name)
def test_corpus_smoke(notebook_path: Path, tmp_path: Path) -> None:
    """Each smoke notebook must clear parse → convert → dag.

    Regression = test failure. The fixtures themselves are committed
    and don't change, so any failure here points at converter or DAG-
    analysis breakage.
    """
    score = _score(notebook_path, out_dir=tmp_path / notebook_path.stem)
    assert score.parse and score.convert and score.dag, (
        f"{notebook_path.name} failed at {score.failed_at}: {score.error}"
    )


def test_corpus_smoke_directory_is_populated() -> None:
    """Standalone guard: the smoke corpus directory must contain at
    least 5 fixtures (per the design doc). Catches accidental deletion
    or refactors that move fixtures out of the discovery path."""
    assert len(_SMOKE_NOTEBOOKS) >= 5, (
        f"Smoke corpus has only {len(_SMOKE_NOTEBOOKS)} notebooks under {_SMOKE_DIR}; "
        "the design doc requires at least 5"
    )


def test_corpus_exercises_converter_translations(tmp_path: Path) -> None:
    """At least one smoke notebook must exercise each branch of the
    converter we care about: magic translation and dep capture.
    Otherwise the corpus stops being a useful regression net for
    those code paths."""
    saw_translated = False
    saw_deps = False
    for nb_path in _SMOKE_NOTEBOOKS:
        score = _score(nb_path, out_dir=tmp_path / nb_path.stem)
        if score.magics_translated:
            saw_translated = True
        if score.deps_captured:
            saw_deps = True
    assert saw_translated, "no smoke notebook exercises magic translation"
    assert saw_deps, "no smoke notebook exercises dep capture"


@pytest.mark.skipif(not _full_mode_enabled(), reason="full mode requires STRATA_CORPUS_RUN=1")
@pytest.mark.parametrize("notebook_path", _SMOKE_NOTEBOOKS, ids=lambda p: p.name)
def test_corpus_smoke_full_run(notebook_path: Path, tmp_path: Path) -> None:
    """Opt-in: actually execute each smoke notebook through ``strata
    run``. Needs ``uv sync`` so requires network; intended for nightly
    CI or pre-release verification. Skipped by default.

    Placeholder for PR 6 — wires in run+artifact scoring once the
    extended-corpus runner lands.
    """
    pytest.skip("full-mode runner lands in PR 6 alongside the extended corpus")
