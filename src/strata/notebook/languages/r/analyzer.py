"""``RLanguageAnalyzer`` — defines/references for R cells.

Shells out to ``Rscript`` with the helper at ``analyze_cell.R`` to
extract DAG inputs/outputs. The R script uses ``codetools::findGlobals``
+ a manual top-level-assign scan; the parent process here is a thin
wrapper that:

- Spawns ``Rscript`` with a hard timeout (~5s) so a hung interpreter
  can't wedge the session.
- Pipes the cell source on stdin.
- Parses the JSON object the helper writes to stdout.
- Caches by source hash so unchanged cells don't pay the spawn cost
  on every reload.

The analyzer registers itself against the ``LanguageAnalyzer`` protocol
from #54. Cells declared as ``language = "r"`` in ``notebook.toml`` go
through this adapter just like Python / SQL / prompt cells go through
their respective adapters.

Note: R cells can be **declared** before #57 lands; this analyzer
makes them participate in the DAG. Attempting to **execute** an R
cell before #57 ships raises ``UnknownLanguageError`` from the
executor registry.
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
    """Raised when ``Rscript`` is not on ``PATH``.

    Distinct from ``FileNotFoundError`` so callers can surface a useful
    "R isn't installed" message rather than a stack trace pointing at
    ``subprocess.run``. Resolution flows through #55's renv bootstrap
    (which expects R + Rscript already on the user's machine).
    """


def _source_hash(source: str) -> str:
    """Stable cache key for ``source`` — same hash function as Python's analyzer."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _run_rscript(source: str) -> AnalyzedCell:
    """Invoke ``Rscript`` against the embedded helper and parse the JSON.

    Three failure modes, each surfaced as a usable error rather than
    swallowed:

    - ``Rscript`` not on ``PATH`` → ``RscriptUnavailableError``. The
      cell loses DAG analysis until R is installed, but the rest of
      the notebook keeps working.
    - Hard timeout → return empty ``AnalyzedCell``. The R interpreter
      hung; logging captures it. The cell ends up isolated in the DAG
      (no edges in or out) which is the safe fallback.
    - Helper exited non-zero or stdout wasn't valid JSON → return
      empty ``AnalyzedCell`` after logging. Same isolation behaviour.
    """
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RscriptUnavailableError(
            "Rscript not found on PATH — install R (https://www.r-project.org/) "
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
        del session  # not consumed — R has no dialect-style runtime context.
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
