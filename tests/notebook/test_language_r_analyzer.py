"""Tests for the R DAG analyzer (``strata.notebook.languages.r``).

Two tiers:

- **Unit tests** monkeypatch ``subprocess.run`` / ``shutil.which`` to
  assert wrapper behaviour without needing R installed. These run
  everywhere — CI matrix, dev machines without R, etc.
- **Integration tests** spawn real ``Rscript`` via the embedded helper
  and assert end-to-end behaviour. Gated on the ``rscript_available``
  marker so they skip cleanly when R isn't on ``PATH``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from strata.notebook.languages import (
    AnalyzedCell,
    analyze_cell_by_language,
    get_language_analyzer,
)
from strata.notebook.languages.r import analyzer as r_analyzer
from strata.notebook.languages.r.analyzer import (
    RscriptUnavailableError,
    _RAnalyzer,
    _run_rscript,
    _source_hash,
)
from strata.notebook.models import CellLanguage, CellState


@pytest.fixture(autouse=True)
def _reset_r_cache():
    """Clear the analyzer's source-hash cache before each test.

    The cache survives the session in production but tests need a fresh
    slate so monkeypatching ``_run_rscript`` actually runs the patched
    version instead of returning a cached real-Rscript result.
    """
    r_analyzer._CACHE.clear()
    yield
    r_analyzer._CACHE.clear()


def _make_cell(source: str = "", language: CellLanguage = CellLanguage.R) -> CellState:
    return CellState(id="r-cell-1", source=source, language=language, order=0)


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistry:
    """R analyzer must be discoverable via the same registry as the others."""

    def test_r_is_registered(self):
        analyzer = get_language_analyzer(CellLanguage.R)
        assert isinstance(analyzer, _RAnalyzer)

    def test_dispatch_routes_through_r_analyzer(self, monkeypatch):
        """The registry-level dispatch helper hits the R adapter for ``CellLanguage.R``."""
        called_with: list[str] = []

        def fake_rscript(source: str) -> AnalyzedCell:
            called_with.append(source)
            return AnalyzedCell(defines=["routed"], references=[])

        monkeypatch.setattr(r_analyzer, "_run_rscript", fake_rscript)
        cell = _make_cell("y <- 1")
        result = analyze_cell_by_language(cell, session=SimpleNamespace())
        assert result.defines == ["routed"]
        assert called_with == ["y <- 1"]

    def test_r_enum_value_round_trips(self):
        """``CellLanguage.R`` must serialize to the expected string."""
        assert CellLanguage.R == "r"
        assert CellLanguage("r") is CellLanguage.R


# ---------------------------------------------------------------------------
# Wrapper behaviour — monkeypatched subprocess (no real R needed)
# ---------------------------------------------------------------------------


class TestRscriptUnavailable:
    """Missing ``Rscript`` surfaces as an info log + empty result, not a crash."""

    def test_run_rscript_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(RscriptUnavailableError) as excinfo:
            _run_rscript("x <- 1")
        assert "Rscript not found on PATH" in str(excinfo.value)
        assert "install r" in str(excinfo.value).lower()

    def test_analyze_returns_empty_when_rscript_missing(self, monkeypatch):
        """A notebook without R installed still opens cleanly."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        result = _RAnalyzer().analyze(_make_cell("y <- x + 1"), session=None)
        assert result == AnalyzedCell()

    def test_empty_result_not_cached_when_rscript_missing(self, monkeypatch):
        """Once R is installed, the next analyze must actually try Rscript.

        Caching the no-R fallback would mean users have to restart the
        server after installing R for analysis to start working.
        """
        monkeypatch.setattr(shutil, "which", lambda name: None)
        cell = _make_cell("y <- x + 1")
        _RAnalyzer().analyze(cell, session=None)
        assert _source_hash(cell.source) not in r_analyzer._CACHE


class TestWrapperFailureModes:
    """Wrapper handles subprocess failures gracefully."""

    def _fake_subprocess(
        self, monkeypatch, *, stdout: str = "", returncode: int = 0, stderr: str = ""
    ):
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")

        def fake_run(*args, **kwargs):
            return SimpleNamespace(stdout=stdout, stderr=stderr, returncode=returncode)

        monkeypatch.setattr(subprocess, "run", fake_run)

    def test_timeout_returns_empty(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: "/fake/Rscript")

        def fake_run(*args, **kwargs):
            raise subprocess.TimeoutExpired(cmd="Rscript", timeout=5.0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        result = _run_rscript("very long source")
        assert result == AnalyzedCell()

    def test_nonzero_exit_returns_empty(self, monkeypatch):
        """Non-zero exit logs + returns empty; doesn't crash the analyzer."""
        self._fake_subprocess(monkeypatch, returncode=1, stderr="some R error")
        result = _run_rscript("x <-")
        assert result == AnalyzedCell()

    def test_non_json_stdout_returns_empty(self, monkeypatch):
        self._fake_subprocess(monkeypatch, stdout="not json at all")
        result = _run_rscript("x <- 1")
        assert result == AnalyzedCell()

    def test_parse_error_returns_empty(self, monkeypatch):
        payload = json.dumps(
            {"defines": [], "references": [], "parse_error": "unexpected end of input"}
        )
        self._fake_subprocess(monkeypatch, stdout=payload)
        result = _run_rscript("x <-")
        assert result == AnalyzedCell()

    def test_successful_payload_round_trips(self, monkeypatch):
        payload = json.dumps({"defines": ["y", "z"], "references": ["x"]})
        self._fake_subprocess(monkeypatch, stdout=payload)
        result = _run_rscript("y <- x + 1; z <- y * 2")
        assert result.defines == ["y", "z"]
        assert result.references == ["x"]
        assert result.mutation_defines == []


class TestCaching:
    """Source-hash cache avoids re-spawning Rscript on unchanged cells."""

    def _patch_with_counter(self, monkeypatch):
        """Patch ``_run_rscript`` with a counting fake; return the counter list."""
        calls: list[str] = []

        def fake(source: str) -> AnalyzedCell:
            calls.append(source)
            return AnalyzedCell(defines=["v"], references=[])

        monkeypatch.setattr(r_analyzer, "_run_rscript", fake)
        return calls

    def test_repeat_source_hits_cache(self, monkeypatch):
        calls = self._patch_with_counter(monkeypatch)
        cell = _make_cell("y <- 1")
        _RAnalyzer().analyze(cell, session=None)
        _RAnalyzer().analyze(cell, session=None)
        _RAnalyzer().analyze(cell, session=None)
        assert len(calls) == 1, "second + third analyze should hit the cache"

    def test_changed_source_invalidates(self, monkeypatch):
        calls = self._patch_with_counter(monkeypatch)
        _RAnalyzer().analyze(_make_cell("y <- 1"), session=None)
        _RAnalyzer().analyze(_make_cell("y <- 2"), session=None)
        assert len(calls) == 2, "source edit forces re-analysis"

    def test_empty_source_short_circuits(self, monkeypatch):
        """No source → no Rscript spawn at all."""
        calls = self._patch_with_counter(monkeypatch)
        result = _RAnalyzer().analyze(_make_cell("   \n   "), session=None)
        assert result == AnalyzedCell()
        assert calls == []


# ---------------------------------------------------------------------------
# Integration tests — real Rscript
# ---------------------------------------------------------------------------


_RSCRIPT_AVAILABLE = shutil.which("Rscript") is not None
_skip_no_rscript = pytest.mark.skipif(
    not _RSCRIPT_AVAILABLE,
    reason="Rscript not on PATH; install R (https://www.r-project.org/) to run",
)


@_skip_no_rscript
class TestIntegrationRealRscript:
    """End-to-end against a real R install.

    These tests verify the embedded helper actually produces the
    expected JSON shape on real R behaviour. They run automatically
    when ``Rscript`` is on PATH; otherwise they skip.
    """

    def test_simple_assign(self):
        """Acceptance example: ``y <- x + 1`` references only ``x``."""
        cell = _make_cell("y <- x + 1")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "y" in result.defines
        # ONLY ``x`` should be a reference — not ``+`` or anything else.
        # The walker skips function-call ops so binary operators don't
        # show up.
        assert result.references == ["x"]

    def test_multiple_assigns_locally_defined_not_a_reference(self):
        """``y <- 1; z <- y + 1`` — ``y`` is defined locally before being read.

        Cross-cell DAG only cares about inputs from other cells. A name
        that's both defined and later read in the same cell isn't a
        cross-cell input, so it must NOT appear in references.
        """
        cell = _make_cell("y <- 1\nz <- y + 1")
        result = _RAnalyzer().analyze(cell, session=None)
        assert set(result.defines) >= {"y", "z"}
        assert "y" not in result.references
        assert result.references == []

    def test_read_before_write_self_assign(self):
        """``y <- y + 1`` reads the upstream ``y`` before redefining it.

        Regression for PR #67 review finding: the codetools-based
        approach used to drop self-assign reads because it treated any
        locally-assigned name as a local-only binding. The new walker
        applies the read-before-locally-defined rule per statement, so
        the upstream ``y`` survives as a real DAG dependency.
        """
        cell = _make_cell("y <- y + 1")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "y" in result.defines
        assert "y" in result.references

    def test_read_before_write_subscript_filter(self):
        """``df <- df[complete.cases(df), ]`` keeps ``df`` as a reference.

        Same shape as the self-assign case, but with the
        ``complete.cases`` function call inside. Confirms that the
        walker recurses through function-arg subtrees while still
        skipping the function name itself (no ``complete.cases`` in
        references).
        """
        cell = _make_cell("df <- df[complete.cases(df), ]")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "df" in result.defines
        assert "df" in result.references
        assert "complete.cases" not in result.references
        # ``[`` is also a function call internally — must not leak.
        assert "[" not in result.references

    def test_function_call_names_not_references(self):
        """Acceptance example: ``library(arrow); df <- read_parquet(...)``.

        Neither ``arrow`` (NSE library arg) nor ``read_parquet`` (function
        call name) is a cross-cell reference. Regression for PR #67
        review finding: previously the codetools approach included
        ``refs$functions`` which leaked function names like
        ``read_parquet`` into references.
        """
        cell = _make_cell("library(arrow)\ndf <- read_parquet('a.parquet')")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "df" in result.defines
        assert "arrow" not in result.references
        assert "read_parquet" not in result.references
        # No data references — the only inputs are the file path
        # literal and the package.
        assert result.references == []

    def test_namespace_access_not_a_reference(self):
        """``arrow::read_parquet(path)`` — neither side of ``::`` is a reference."""
        cell = _make_cell("df <- arrow::read_parquet(path)")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "df" in result.defines
        # ``path`` is a real data reference (literal name passed as arg).
        assert "path" in result.references
        assert "arrow" not in result.references
        assert "read_parquet" not in result.references

    def test_member_access_only_lhs_is_a_reference(self):
        """``df$col`` — ``df`` is read; ``col`` is a slot name, not a reference."""
        cell = _make_cell("x <- df$col")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "x" in result.defines
        assert "df" in result.references
        assert "col" not in result.references

    def test_parse_error_returns_empty(self):
        cell = _make_cell("x <-")  # incomplete
        result = _RAnalyzer().analyze(cell, session=None)
        assert result == AnalyzedCell()

    def test_empty_cell_returns_empty(self):
        cell = _make_cell("")
        result = _RAnalyzer().analyze(cell, session=None)
        assert result == AnalyzedCell()


# ---------------------------------------------------------------------------
# Executor side: R cells should fail loudly until #57 lands
# ---------------------------------------------------------------------------
