"""Smoke corpus runner for Jupyter notebook import.

PRs 5–6 of the Jupyter interop work. Each fixture under
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

PR 5 covers parse + convert + dag — network-free, ~80ms total, every PR.

PR 6 adds run + artifact behind the ``STRATA_CORPUS_RUN=1`` env knob.
That path executes the notebook end-to-end: uv sync the notebook
venv, run all cells through Strata's harness, score per-cell results
out of the ``strata run --format json`` payload. Slow and networked
(every smoke notebook installs its primary library), so it stays
opt-in. Intended use: nightly schedule, manual pre-release runs,
and any time a converter change touches code that produces source
the harness actually has to execute.

External / URL-fetched corpus (the "extended" tier in the design)
is intentionally not part of this file — that's release-ops
infrastructure and lands separately.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


_UV_SYNC_TIMEOUT_S = 300
_STRATA_RUN_TIMEOUT_S = 300


def _full_mode_enabled() -> bool:
    """``STRATA_CORPUS_RUN=1`` (or ``true``) opts into run + artifact scoring."""
    return os.environ.get("STRATA_CORPUS_RUN", "0").lower() in ("1", "true", "yes")


def _score(ipynb_path: Path, out_dir: Path, *, full: bool = False) -> CorpusScore:
    """Score one notebook through parse → convert → dag (+ run + artifact when ``full``)."""
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
    dag = None
    analyses: list[CellAnalysisWithId] = []
    try:
        nb_state = parse_notebook(result.notebook_dir)
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
        dag = build_dag(analyses)  # raises on cycles
        score.dag = True
    except Exception as exc:
        score.failed_at = "dag"
        score.error = f"{type(exc).__name__}: {exc}"
        return score

    if not full:
        return score

    # run — actually execute the notebook end-to-end. Sync the notebook
    # venv first; we passed initialize_environment=False through the
    # import path so the venv doesn't exist yet. ``strata run``
    # invocation goes through the same CLI users hit, so any breakage
    # here is breakage the user would see.
    notebook_dir = result.notebook_dir
    try:
        uv_sync = subprocess.run(
            ["uv", "sync"],
            cwd=str(notebook_dir),
            capture_output=True,
            text=True,
            timeout=_UV_SYNC_TIMEOUT_S,
            check=False,
        )
        if uv_sync.returncode != 0:
            score.failed_at = "run"
            score.error = f"uv sync failed: {uv_sync.stderr.strip()[:500]}"
            return score
    except FileNotFoundError:
        pytest.skip("uv not installed — full mode requires the uv CLI on PATH")
    except subprocess.TimeoutExpired:
        score.failed_at = "run"
        score.error = f"uv sync timed out after {_UV_SYNC_TIMEOUT_S}s"
        return score

    try:
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "strata.cli",
                "run",
                str(notebook_dir),
                "--no-sync",
                "--format",
                "json",
            ],
            capture_output=True,
            text=True,
            timeout=_STRATA_RUN_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        score.failed_at = "run"
        score.error = f"strata run timed out after {_STRATA_RUN_TIMEOUT_S}s"
        return score

    # ``strata run --format json`` writes per-cell results to stdout
    # even when it exits non-zero (one or more cells failed). Parse
    # stdout first so the cell-level error makes it into the score;
    # only fall back to stderr if stdout isn't JSON at all.
    payload: dict | None = None
    try:
        payload = json.loads(run.stdout)
    except json.JSONDecodeError:
        payload = None
    if payload is None:
        score.failed_at = "run"
        msg = run.stderr.strip() or run.stdout.strip() or "(no output)"
        score.error = f"strata run failed (rc={run.returncode}): {msg[:500]}"
        return score
    if not payload.get("success", False) or run.returncode != 0:
        bad = [c for c in payload.get("cells") or [] if c.get("status") != "ok"]
        first = bad[0] if bad else {}
        score.failed_at = "run"
        score.error = (
            f"{len(bad)} cell(s) failed; "
            f"first: {first.get('id')} — {first.get('error', '(no error message)')[:300]}"
        )
        return score
    score.run = True

    # artifact — per the design rubric: "do leaf cells produce
    # non-empty artifacts?". Leaf cells are the DAG terminals — the
    # ones nothing else depends on, which is where a missing output
    # would mean nothing downstream covered for it. For each leaf, an
    # observable trace must exist: an Arrow artifact whose filename
    # carries the cell_id, a console snapshot, or a display output in
    # runtime.json.
    artifact_dir = notebook_dir / ".strata" / "artifacts"
    console_dir = notebook_dir / ".strata" / "console"
    runtime_path = notebook_dir / ".strata" / "runtime.json"
    runtime_data: dict = {}
    if runtime_path.is_file():
        try:
            runtime_data = json.loads(runtime_path.read_text())
        except json.JSONDecodeError:
            runtime_data = {}
    runtime_cells = runtime_data.get("cells") or {}

    leaf_ids = (
        [cid for cid, downstream in (dag.cell_downstream or {}).items() if not downstream]
        if dag is not None
        else []
    )
    # Restrict to python leaves — markdown cells never persist anything.
    python_cell_ids = {a.id for a in analyses}
    leaf_python_ids = [cid for cid in leaf_ids if cid in python_cell_ids]

    if not leaf_python_ids:
        # No python leaves (notebook is markdown-only or every python cell
        # has downstream consumers — both are fine). Artifact step is
        # vacuously satisfied.
        score.artifact = True
        return score

    missing: list[str] = []
    for cell_id in leaf_python_ids:
        has_artifact = any(artifact_dir.rglob(f"*cell_{cell_id}*.arrow"))
        has_console = (console_dir / f"{cell_id}.json").exists()
        cell_runtime = runtime_cells.get(cell_id) or {}
        has_display = bool(cell_runtime.get("display_outputs"))
        if not (has_artifact or has_console or has_display):
            missing.append(cell_id)

    if missing:
        score.failed_at = "artifact"
        score.error = (
            f"{len(missing)} leaf cell(s) produced no observable output "
            f"(no artifact / console / display): {missing}"
        )
        return score
    score.artifact = True
    return score


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


@pytest.mark.skipif(
    not _SMOKE_NOTEBOOKS or not _full_mode_enabled(),
    reason="full mode requires STRATA_CORPUS_RUN=1 (slow, networked)",
)
@pytest.mark.parametrize("notebook_path", _SMOKE_NOTEBOOKS, ids=lambda p: p.name)
def test_corpus_smoke_full_run(notebook_path: Path, tmp_path: Path) -> None:
    """Full rubric: parse → convert → dag → run → artifact.

    Each smoke fixture is end-to-end executed through ``strata run``
    after its venv is synced. Slow (one ``uv sync`` per fixture, then
    real cell execution), so this is opt-in: set ``STRATA_CORPUS_RUN=1``
    locally, or wire it into a nightly schedule in CI.

    Regression here means the converter produced source the harness
    can't actually run — a class of bug the fast-mode tier wouldn't
    catch.
    """
    score = _score(notebook_path, out_dir=tmp_path / notebook_path.stem, full=True)
    assert (
        score.parse and score.convert and score.dag and score.run and (score.artifact is not False)
    ), f"{notebook_path.name} failed at {score.failed_at}: {score.error}"
