"""Tests for the per-language analyzer registry.

Most behavior is exercised end-to-end by the DAG + cell analysis suites
(``test_analyzer.py``, ``test_prompt_analyzer.py``, ``test_sql_*.py``,
``test_dag.py``). These tests target the registry surface itself —
adding a language, looking one up, the failure shape for an unregistered
language.
"""

from __future__ import annotations

import pytest

from strata.notebook.languages import (
    AnalyzedCell,
    UnknownLanguageError,
    analyze_cell_by_language,
    get_language_analyzer,
    register_language_analyzer,
)
from strata.notebook.models import CellLanguage, CellState


def _make_cell(language: CellLanguage, source: str = "") -> CellState:
    return CellState(id="c1", source=source, language=language, order=0)


class TestBuiltInRegistrations:
    """The four shipped languages must resolve at import time."""

    @pytest.mark.parametrize(
        "language",
        [
            CellLanguage.PYTHON,
            CellLanguage.PROMPT,
            CellLanguage.SQL,
            CellLanguage.MARKDOWN,
        ],
    )
    def test_shipped_language_registered(self, language):
        assert get_language_analyzer(language) is not None


class TestDispatch:
    """The dispatch helper returns the language-specific analysis."""

    def test_python_extracts_defines_and_references(self, monkeypatch):
        # No need for a NotebookSession — the Python adapter ignores it,
        # so any object that satisfies the call signature is fine.
        cell = _make_cell(CellLanguage.PYTHON, "y = x + 1")
        analyzed = analyze_cell_by_language(cell, session=object())
        assert "y" in analyzed.defines
        assert "x" in analyzed.references

    def test_markdown_returns_empty(self):
        cell = _make_cell(CellLanguage.MARKDOWN, "# Heading\n\nProse.")
        analyzed = analyze_cell_by_language(cell, session=object())
        assert analyzed.defines == []
        assert analyzed.references == []
        assert analyzed.mutation_defines == []


class TestErrors:
    """Missing-language path must fail loudly, not silently empty out."""

    def test_unregistered_language_raises_unknownlanguageerror(self):
        # Create a fake language value that the registry doesn't know about.
        # Using a sentinel-ish object that won't equal any CellLanguage member.
        fake_lang = "totally-not-a-real-language"
        with pytest.raises(UnknownLanguageError):
            get_language_analyzer(fake_lang)  # type: ignore[arg-type]


class TestRegisterIsExtensible:
    """A new language can register without touching existing dispatch sites."""

    def test_register_overrides_existing(self):
        """Re-registering a language replaces the prior adapter.

        Matches the SQL ``DriverAdapter`` registry's behaviour. Tests
        rely on this to swap in fakes; production code shouldn't, but
        the contract is the contract.
        """

        captured: list[str] = []

        class FakeAnalyzer:
            def analyze(self, cell, session):  # noqa: ANN001
                captured.append(cell.source)
                return AnalyzedCell(defines=["fake"], references=[])

        original = get_language_analyzer(CellLanguage.MARKDOWN)
        try:
            register_language_analyzer(CellLanguage.MARKDOWN, FakeAnalyzer())
            cell = _make_cell(CellLanguage.MARKDOWN, "x")
            analyzed = analyze_cell_by_language(cell, session=object())
            assert analyzed.defines == ["fake"]
            assert captured == ["x"]
        finally:
            # Restore so unrelated tests don't see the fake.
            register_language_analyzer(CellLanguage.MARKDOWN, original)
