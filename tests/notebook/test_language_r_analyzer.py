"""Tests for the R DAG analyzer (``strata.notebook.languages.r``).

Unit tests fake ``subprocess.run`` / ``shutil.which`` and run without R;
integration tests spawn real ``Rscript`` and skip when it is not on ``PATH``.
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
from tests.notebook.conftest import skip_if_no_r as _skip_no_rscript


@pytest.fixture(autouse=True)
def _reset_r_cache():
    """Clear the analyzer's source-hash cache so patched ``_run_rscript`` actually runs."""
    r_analyzer._CACHE.clear()
    yield
    r_analyzer._CACHE.clear()


def _make_cell(source: str = "", language: CellLanguage = CellLanguage.R) -> CellState:
    return CellState(id="r-cell-1", source=source, language=language, order=0)


# Registry wiring


class TestRegistry:
    def test_r_is_registered(self):
        analyzer = get_language_analyzer(CellLanguage.R)
        assert isinstance(analyzer, _RAnalyzer)

    def test_dispatch_routes_through_r_analyzer(self, monkeypatch):
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
        assert CellLanguage.R == "r"
        assert CellLanguage("r") is CellLanguage.R


# Wrapper behaviour: monkeypatched subprocess (no real R needed)


class TestRscriptUnavailable:
    """Missing ``Rscript`` surfaces as an info log and an empty result, not a crash."""

    def test_run_rscript_raises_when_missing(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        with pytest.raises(RscriptUnavailableError) as excinfo:
            _run_rscript("x <- 1")
        assert "Rscript not found on PATH" in str(excinfo.value)
        assert "install r" in str(excinfo.value).lower()

    def test_analyze_returns_empty_when_rscript_missing(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda name: None)
        result = _RAnalyzer().analyze(_make_cell("y <- x + 1"), session=None)
        assert result == AnalyzedCell()

    def test_empty_result_not_cached_when_rscript_missing(self, monkeypatch):
        """Caching the no-R fallback would force a server restart after installing R."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        cell = _make_cell("y <- x + 1")
        _RAnalyzer().analyze(cell, session=None)
        assert _source_hash(cell.source) not in r_analyzer._CACHE


class TestWrapperFailureModes:
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
        calls = self._patch_with_counter(monkeypatch)
        result = _RAnalyzer().analyze(_make_cell("   \n   "), session=None)
        assert result == AnalyzedCell()
        assert calls == []


# Integration tests: real Rscript


@_skip_no_rscript
class TestIntegrationRealRscript:
    """End-to-end against a real R install; skipped when ``Rscript`` is not on PATH."""

    def test_simple_assign(self):
        """``y <- x + 1`` references only ``x``."""
        cell = _make_cell("y <- x + 1")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "y" in result.defines
        # Only ``x`` is a reference; the walker skips function-call ops so binary
        # operators don't show up.
        assert result.references == ["x"]

    def test_multiple_assigns_locally_defined_not_a_reference(self):
        """``y <- 1; z <- y + 1``: a name defined before it is read is not a cross-cell input."""
        cell = _make_cell("y <- 1\nz <- y + 1")
        result = _RAnalyzer().analyze(cell, session=None)
        assert set(result.defines) >= {"y", "z"}
        assert "y" not in result.references
        assert result.references == []

    def test_read_before_write_self_assign(self):
        """``y <- y + 1`` reads the upstream ``y`` before redefining it.

        The read-before-locally-defined rule applies per statement, so the self-assign
        read stays a DAG dependency.
        """
        cell = _make_cell("y <- y + 1")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "y" in result.defines
        assert "y" in result.references

    def test_read_before_write_subscript_filter(self):
        """``df <- df[complete.cases(df), ]`` keeps ``df`` but not ``complete.cases``.

        The walker recurses into function arguments while skipping the function name.
        """
        cell = _make_cell("df <- df[complete.cases(df), ]")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "df" in result.defines
        assert "df" in result.references

    @pytest.mark.parametrize(
        "source",
        ["df$b <- 2", "df[i] <- 0", "df[['b']] <- 2", "names(df) <- cols", "2 -> df$b"],
    )
    def test_replacement_assign_reads_and_defines_root(self, source):
        """``df$b <- v`` is ``df <- `$<-`(df, "b", v)``: the root is read and defined."""
        result = _RAnalyzer().analyze(_make_cell(source), session=None)
        assert result.defines == ["df"]
        assert "df" in result.references

    def test_replacement_assign_after_local_define_is_local(self):
        result = _RAnalyzer().analyze(_make_cell("df <- data.frame(a = 1)\ndf$b <- 2"), None)
        assert result.defines == ["df"]
        assert "df" not in result.references
        assert "complete.cases" not in result.references
        # ``[`` is a function call internally and must not leak.
        assert "[" not in result.references

    def test_function_call_names_not_references(self):
        """``library(arrow); df <- read_parquet(...)``: neither name is a reference.

        ``arrow`` is an NSE library argument and ``read_parquet`` a call name.
        """
        cell = _make_cell("library(arrow)\ndf <- read_parquet('a.parquet')")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "df" in result.defines
        assert "arrow" not in result.references
        assert "read_parquet" not in result.references
        # No data references: the only inputs are the file path literal and the package.
        assert result.references == []

    def test_namespace_access_not_a_reference(self):
        """``arrow::read_parquet(path)``: neither side of ``::`` is a reference."""
        cell = _make_cell("df <- arrow::read_parquet(path)")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "df" in result.defines
        # ``path`` is a real data reference (literal name passed as arg).
        assert "path" in result.references
        assert "arrow" not in result.references
        assert "read_parquet" not in result.references

    def test_member_access_only_lhs_is_a_reference(self):
        """``df$col``: ``df`` is read; ``col`` is a slot name, not a reference."""
        cell = _make_cell("x <- df$col")
        result = _RAnalyzer().analyze(cell, session=None)
        assert "x" in result.defines
        assert "df" in result.references
        assert "col" not in result.references

    def test_single_multichar_name_not_split_into_chars(self):
        """``df <- 1`` returns ``defines=['df']``, never ``['d', 'f']``.

        ``auto_unbox`` collapses 1-element vectors to JSON strings, which Python then
        iterates by character. Single-char names hide the bug, so pin a multi-char one.
        """
        cell = _make_cell("df <- 1")
        result = _RAnalyzer().analyze(cell, session=None)
        assert result.defines == ["df"]
        assert "d" not in result.defines
        assert "f" not in result.defines

    def test_parse_error_returns_empty(self):
        cell = _make_cell("x <-")  # incomplete
        result = _RAnalyzer().analyze(cell, session=None)
        assert result == AnalyzedCell()

    def test_empty_cell_returns_empty(self):
        cell = _make_cell("")
        result = _RAnalyzer().analyze(cell, session=None)
        assert result == AnalyzedCell()
