"""``RLanguageAnalyzer``: defines/references for R cells.

Pipes the cell source to ``Rscript`` running ``analyze_cell.R``
(``codetools::findGlobals`` plus a top-level-assign scan) and parses its JSON.
A hard timeout keeps a hung interpreter from wedging the session; results are
cached by source hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from strata.notebook.languages.analyzer import (
    AnalyzedCell,
    register_language_analyzer,
)
from strata.notebook.models import CellLanguage

if TYPE_CHECKING:
    from strata.notebook.models import CellState
    from strata.notebook.session import NotebookSession

logger = logging.getLogger(__name__)

# The helper does no I/O, so anything past this is a hung interpreter.
# Cell analysis must never block, as in the Python analyzer.
_ANALYZE_TIMEOUT_SECONDS = 5.0

_HELPER_PATH = Path(__file__).parent / "analyze_cell.R"

# Keyed on ``sha256(source)``, so edits invalidate by construction.
_CACHE: dict[str, AnalyzedCell] = {}


class RscriptUnavailableError(RuntimeError):
    """Raised when ``Rscript`` is not on ``PATH``, so callers can say R is not installed."""


def _source_hash(source: str) -> str:
    """Stable cache key for ``source``: same hash function as Python's analyzer."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _run_rscript(source: str) -> AnalyzedCell:
    """Invoke ``Rscript`` against the embedded helper and parse the JSON.

    Raises ``RscriptUnavailableError`` when ``Rscript`` is missing. A timeout, a
    non-zero exit or invalid JSON is logged and returns an empty ``AnalyzedCell``,
    isolating the cell in the DAG.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RscriptUnavailableError(
            "Rscript not found on PATH; install R (https://www.r-project.org/) "
            "to enable R cell support."
        )

    try:
        proc = subprocess.run(
            [rscript, "--no-init-file", "--vanilla", str(_HELPER_PATH)],
            input=source,
            capture_output=True,
            text=True,
            timeout=_ANALYZE_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Rscript timed out after %.1fs analyzing R cell; returning empty "
            "AnalyzedCell so the cell stays isolated in the DAG.",
            _ANALYZE_TIMEOUT_SECONDS,
        )
        return AnalyzedCell()

    if proc.returncode != 0:
        logger.warning(
            "Rscript exited %d analyzing R cell; stderr=%r",
            proc.returncode,
            proc.stderr.strip()[:500] if proc.stderr else "",
        )
        return AnalyzedCell()

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Rscript helper produced non-JSON output: %r (raw=%r)",
            exc,
            proc.stdout[:200],
        )
        return AnalyzedCell()

    # A parse error is a normal mid-typing state. Match the Python analyzer:
    # error in ``CellAnalysis.error``, empty cell for the DAG, no logging.
    if payload.get("parse_error"):
        logger.debug("R cell parse error: %s", payload.get("parse_error"))
        return AnalyzedCell()

    defines = list(payload.get("defines") or [])
    references = list(payload.get("references") or [])
    # R has no ``mutation_defines`` equivalent (``a[k] <- v`` tracking) yet.
    return AnalyzedCell(
        defines=defines,
        references=references,
        mutation_defines=[],
    )


class _RAnalyzer:
    """Adapter that satisfies the ``LanguageAnalyzer`` protocol."""

    def analyze(self, cell: CellState, session: NotebookSession) -> AnalyzedCell:
        del session  # not consumed: R has no dialect-style runtime context.
        source = cell.source or ""
        if not source.strip():
            return AnalyzedCell()

        key = _source_hash(source)
        cached = _CACHE.get(key)
        if cached is not None:
            return cached

        try:
            result = _run_rscript(source)
        except RscriptUnavailableError:
            # Without R, return an empty analysis so loading doesn't crash; the cell
            # stays isolated in the DAG and the executor refuses to run it.
            logger.info(
                "R cell %s has no DAG analysis: Rscript not on PATH",
                cell.id,
            )
            result = AnalyzedCell()
            # Don't cache: once R is installed the next call must invoke Rscript.
            return result

        _CACHE[key] = result
        return result


register_language_analyzer(CellLanguage.R, _RAnalyzer())
