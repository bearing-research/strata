"""Smoke and extended corpus runner for Jupyter notebook import.

Smoke fixtures in ``jupyter_corpus/smoke/`` are committed ``.ipynb`` files scored on
parse, convert and dag, network-free, on every run. ``STRATA_CORPUS_RUN=1`` adds run and
artifact scoring (a real venv sync and cell execution) and the extended corpus: URLs
pinned to commits in ``jupyter_corpus/extended.yaml``, cached under
``~/.cache/strata-jupyter-corpus/``.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from strata.notebook.analyzer import analyze_cell
from strata.notebook.dag import CellAnalysisWithId, NotebookDag
from strata.notebook.jupyter_import import import_notebook
from strata.notebook.parser import parse_notebook

_SMOKE_DIR = Path(__file__).parent / "jupyter_corpus" / "smoke"
_SMOKE_NOTEBOOKS = sorted(_SMOKE_DIR.glob("*.ipynb"))
_EXTENDED_MANIFEST = Path(__file__).parent / "jupyter_corpus" / "extended.yaml"
_EXTENDED_CACHE = Path.home() / ".cache" / "strata-jupyter-corpus"
_FETCH_TIMEOUT_S = 30


@dataclass
class CorpusScore:
    """Result of scoring one notebook.

    Each step is checked only when the previous one passed; ``failed_at`` names the first
    failure.
    """

    notebook: str
    parse: bool = False
    convert: bool = False
    dag: bool = False
    failed_at: str = ""
    error: str = ""
    # Filled in when STRATA_CORPUS_RUN=1.
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
    """``STRATA_CORPUS_RUN`` set to 1, true or yes opts into run and artifact scoring."""
    return os.environ.get("STRATA_CORPUS_RUN", "0").lower() in ("1", "true", "yes")


def _score(ipynb_path: Path, out_dir: Path, *, full: bool = False) -> CorpusScore:
    """Score one notebook through parse, convert and dag (plus run and artifact when ``full``)."""
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

    # dag: analyze each python cell and build the graph as
    # session._analyze_and_build_dag does, without a Session (which would drag in
    # the process pool and venv machinery).
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
        dag = NotebookDag.from_cells(analyses)  # raises on cycles
        score.dag = True
    except Exception as exc:
        score.failed_at = "dag"
        score.error = f"{type(exc).__name__}: {exc}"
        return score

    if not full:
        return score

    # run: execute end-to-end through the same `strata run` CLI users hit. Sync
    # the venv first, since the import passed initialize_environment=False.
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

    # ``strata run --format json`` writes per-cell results to stdout even on a
    # non-zero exit, so parse stdout first; fall back to stderr only if it is not JSON.
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

    # artifact: every leaf cell (nothing depends on it) must leave a trace: an
    # Arrow artifact whose filename carries the cell_id, a console snapshot, or a
    # display output in runtime.json.
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
    # Python leaves only; markdown cells never persist anything.
    python_cell_ids = {a.id for a in analyses}
    leaf_python_ids = [cid for cid in leaf_ids if cid in python_cell_ids]

    if not leaf_python_ids:
        # No python leaves (markdown-only, or every python cell has consumers), so
        # the artifact step is vacuously satisfied.
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
    """Each smoke notebook must clear parse, convert and dag.

    The fixtures never change, so a failure is converter or DAG-analysis breakage.
    """
    score = _score(notebook_path, out_dir=tmp_path / notebook_path.stem)
    assert score.parse and score.convert and score.dag, (
        f"{notebook_path.name} failed at {score.failed_at}: {score.error}"
    )


def test_corpus_smoke_directory_is_populated() -> None:
    """At least 5 fixtures; catches fixtures deleted or moved out of the discovery path."""
    assert len(_SMOKE_NOTEBOOKS) >= 5, (
        f"Smoke corpus has only {len(_SMOKE_NOTEBOOKS)} notebooks under {_SMOKE_DIR}; "
        "the design doc requires at least 5"
    )


def test_corpus_exercises_converter_translations(tmp_path: Path) -> None:
    """Some smoke notebook must exercise magic translation and dependency capture, or the
    corpus stops guarding those paths.
    """
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


# Extended corpus: fetched from pinned URLs in extended.yaml


@dataclass
class _ExtendedEntry:
    """One row from the extended-corpus manifest."""

    name: str
    url: str
    expected: str = "pass"  # 'pass' or 'parse_only'
    reason: str = ""


def _load_extended_manifest() -> list[_ExtendedEntry]:
    """Read ``extended.yaml``; empty when the file is missing, which parametrizes zero tests."""
    if not _EXTENDED_MANIFEST.is_file():
        return []
    import yaml  # PyYAML; transitive dep already present via pytest plugins

    try:
        data = yaml.safe_load(_EXTENDED_MANIFEST.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return []
    entries = data.get("notebooks") or []
    out: list[_ExtendedEntry] = []
    for item in entries:
        if not isinstance(item, dict) or "name" not in item or "url" not in item:
            continue
        out.append(
            _ExtendedEntry(
                name=str(item["name"]),
                url=str(item["url"]),
                expected=str(item.get("expected") or "pass"),
                reason=str(item.get("reason") or ""),
            )
        )
    return out


def _fetch_notebook(url: str) -> Path:
    """Download a notebook to the local cache and return its path.

    URLs are commit-pinned, so an entry keyed on the URL digest never goes stale. Raises on
    network failure; callers turn that into a skip.
    """
    _EXTENDED_CACHE.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    target = _EXTENDED_CACHE / f"{digest}.ipynb"
    if target.is_file() and target.stat().st_size > 0:
        return target

    req = urllib.request.Request(url, headers={"User-Agent": "strata-jupyter-corpus/1.0"})
    with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as resp:
        body = resp.read()
    target.write_bytes(body)
    return target


_EXTENDED_ENTRIES = _load_extended_manifest()


@pytest.mark.skipif(
    not _EXTENDED_ENTRIES or not _full_mode_enabled(),
    reason="extended corpus requires STRATA_CORPUS_RUN=1 and a non-empty manifest",
)
@pytest.mark.parametrize("entry", _EXTENDED_ENTRIES, ids=lambda e: e.name)
def test_corpus_extended(entry: _ExtendedEntry, tmp_path: Path) -> None:
    """Score one extended-corpus entry end to end.

    Fetch failures are skips; anything else is a real signal. Report-only: the
    ``jupyter-corpus`` workflow does not gate other CI.
    """
    try:
        ipynb_path = _fetch_notebook(entry.url)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        pytest.skip(f"fetch failed: {type(exc).__name__}: {exc}")

    out_dir = tmp_path / entry.name
    full = entry.expected == "pass"
    score = _score(ipynb_path, out_dir=out_dir, full=full)

    if entry.expected == "parse_only":
        assert score.parse and score.convert and score.dag, (
            f"{entry.name} (expected parse_only) failed at {score.failed_at}: {score.error}"
        )
    else:
        assert (
            score.parse
            and score.convert
            and score.dag
            and score.run
            and (score.artifact is not False)
        ), f"{entry.name} failed at {score.failed_at}: {score.error}"


@pytest.mark.skipif(
    not _SMOKE_NOTEBOOKS or not _full_mode_enabled(),
    reason="full mode requires STRATA_CORPUS_RUN=1 (slow, networked)",
)
@pytest.mark.parametrize("notebook_path", _SMOKE_NOTEBOOKS, ids=lambda p: p.name)
def test_corpus_smoke_full_run(notebook_path: Path, tmp_path: Path) -> None:
    """Full rubric through ``strata run`` after a real venv sync, opt-in for speed.

    Catches converter output the harness cannot run, which the fast tier misses.
    """
    score = _score(notebook_path, out_dir=tmp_path / notebook_path.stem, full=True)
    assert (
        score.parse and score.convert and score.dag and score.run and (score.artifact is not False)
    ), f"{notebook_path.name} failed at {score.failed_at}: {score.error}"
