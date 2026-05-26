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

# Hard wall on how long ``Rscript`` is given to parse + report on a
# cell. The helper itself is bounded — no I/O, no network — so anything
# past 5s is a hung interpreter we'd rather kill than wait on. Mirrors
# the Python analyzer's behaviour of never blocking on cell analysis.
_ANALYZE_TIMEOUT_SECONDS = 5.0

# Embedded R helper. Lives next to this module so ``importlib.resources``-
# style packaging picks it up cleanly in installed builds.
_HELPER_PATH = Path(__file__).parent / "analyze_cell.R"

# Cache keyed on ``sha256(source)`` so re-analyzing an unchanged cell
# doesn't re-spawn ``Rscript``. Survives the session — the cell content
# *is* the cache key, so source edits invalidate by construction. New
# entries replace old ones for the same hash (idempotent).
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

    # A parse error from the R interpreter is a normal cell-edit state
    # (user halfway through typing). The Python analyzer drops the
    # error message into ``CellAnalysis.error`` and the DAG sees an
    # empty cell; matching that shape keeps the two languages
    # consistent. Logging here would be noisy.
    if payload.get("parse_error"):
        logger.debug("R cell parse error: %s", payload.get("parse_error"))
        return AnalyzedCell()

    defines = list(payload.get("defines") or [])
    references = list(payload.get("references") or [])
    # R has no equivalent of Python's ``mutation_defines`` (subscript-
    # assign tracking via ``a[k] <- v``). Treating mutations as defines
    # is a possible #57+ enhancement; for now leave the field empty.
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
            # When R isn't installed, we still need to return *something*
            # so notebook loading doesn't crash. Empty analysis leaves
            # R cells isolated in the DAG, which is the right shape —
            # cells without analysis can't connect to upstream / downstream
            # nodes, and the executor will refuse to run them with a
            # clear "R not installed" message once #57 lands.
            logger.info(
                "R cell %s has no DAG analysis: Rscript not on PATH",
                cell.id,
            )
            result = AnalyzedCell()
            # Do NOT cache the empty result — once R is installed, we
            # want the next analyze call to actually invoke Rscript
            # rather than hit a stale cache miss.
            return result

        _CACHE[key] = result
        return result


# Register at module load. Importing ``strata.notebook.languages.r``
# (via the package ``__init__``) wires the adapter into the registry.
register_language_analyzer(CellLanguage.R, _RAnalyzer())
