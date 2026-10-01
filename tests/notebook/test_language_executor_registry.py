"""Tests for the per-language executor registry surface.

Registration, lookup, the unregistered-language failure, and each adapter's
behaviour flags; execution itself is covered by the wider notebook suite.
"""

from __future__ import annotations

import pytest

from strata.notebook.languages import (
    get_language_executor,
    register_language_executor,
)
from strata.notebook.languages.executor import UnknownLanguageError
from strata.notebook.models import CellLanguage


class TestBuiltInRegistrations:
    """The five shipped languages resolve at import (R registers on ``languages.r`` import)."""

    @pytest.mark.parametrize(
        "language",
        [
            CellLanguage.PYTHON,
            CellLanguage.PROMPT,
            CellLanguage.SQL,
            CellLanguage.MARKDOWN,
            CellLanguage.R,
        ],
    )
    def test_shipped_language_registered(self, language):
        assert get_language_executor(language) is not None


class TestBehaviourFlags:
    """Behaviour flags drive staleness gates in session.compute_staleness."""

    def test_markdown_skips_execution_provenance(self):
        assert get_language_executor(CellLanguage.MARKDOWN).skips_execution_provenance is True

    def test_others_compute_provenance(self):
        for lang in (CellLanguage.PYTHON, CellLanguage.PROMPT, CellLanguage.SQL, CellLanguage.R):
            assert get_language_executor(lang).skips_execution_provenance is False, lang

    def test_prompt_and_sql_have_alternate_cache_scheme(self):
        """A per-language cache hash means the generic miss check must preserve READY."""
        assert get_language_executor(CellLanguage.PROMPT).has_alternate_cache_scheme is True
        assert get_language_executor(CellLanguage.SQL).has_alternate_cache_scheme is True

    def test_python_markdown_r_use_generic_cache_scheme(self):
        """These store under the standard per-variable hash (or nothing for markdown)."""
        assert get_language_executor(CellLanguage.PYTHON).has_alternate_cache_scheme is False
        assert get_language_executor(CellLanguage.MARKDOWN).has_alternate_cache_scheme is False
        assert get_language_executor(CellLanguage.R).has_alternate_cache_scheme is False


class TestErrors:
    """Missing-language path must fail loudly."""

    def test_unregistered_language_raises_unknownlanguageerror(self):
        fake_lang = "totally-not-a-real-language"
        with pytest.raises(UnknownLanguageError):
            get_language_executor(fake_lang)  # type: ignore[arg-type]


class TestRegisterIsExtensible:
    """A new language can register without touching dispatch sites."""

    def test_register_overrides_existing(self):
        """Re-registering replaces the prior adapter; tests rely on this to swap in fakes."""

        class FakeExecutor:
            skips_execution_provenance = False
            has_alternate_cache_scheme = False

            async def execute(
                self,
                executor,  # noqa: ANN001
                cell_id,  # noqa: ANN001
                source,  # noqa: ANN001
                start_time,  # noqa: ANN001
                *,
                timeout_seconds,  # noqa: ANN001
                materialize_upstreams,  # noqa: ANN001
                use_cache,  # noqa: ANN001
            ):
                raise NotImplementedError

            def is_batchable(self, cell, executor):  # noqa: ANN001
                return True  # markdown is normally non-batchable

        original = get_language_executor(CellLanguage.MARKDOWN)
        try:
            register_language_executor(CellLanguage.MARKDOWN, FakeExecutor())
            replaced = get_language_executor(CellLanguage.MARKDOWN)
            assert replaced is not original
            assert isinstance(replaced, FakeExecutor)
        finally:
            register_language_executor(CellLanguage.MARKDOWN, original)


class TestIsBatchableShortcuts:
    """Non-Python languages return ``False`` without inspecting the cell."""

    @pytest.mark.parametrize(
        "language",
        [CellLanguage.PROMPT, CellLanguage.SQL, CellLanguage.MARKDOWN, CellLanguage.R],
    )
    def test_non_python_languages_return_false(self, language):
        """The adapter ignores the cell for these, so a sentinel suffices.

        Python's ``is_batchable`` needs a real cell; see ``test_executor_batch.py``.
        """
        # Sentinel cell + executor; must never be touched.
        sentinel = object()
        assert get_language_executor(language).is_batchable(sentinel, sentinel) is False
