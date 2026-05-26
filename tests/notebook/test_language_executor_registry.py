"""Tests for the per-language executor registry.

End-to-end behaviour is exercised by the full notebook suite (cell
execution, batching, staleness). These tests target the registry
surface — adding a language, looking one up, the failure shape for an
unregistered language, and the behaviour-flag projections each adapter
exposes.
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
    """The four shipped languages must resolve at import time."""

    def test_python_registered(self):
        assert get_language_executor(CellLanguage.PYTHON) is not None

    def test_prompt_registered(self):
        assert get_language_executor(CellLanguage.PROMPT) is not None

    def test_sql_registered(self):
        assert get_language_executor(CellLanguage.SQL) is not None

    def test_markdown_registered(self):
        assert get_language_executor(CellLanguage.MARKDOWN) is not None


class TestBehaviourFlags:
    """Behaviour flags drive staleness gates in session.compute_staleness."""

    def test_markdown_skips_execution_provenance(self):
        """Markdown short-circuits the entire provenance chain."""
        assert get_language_executor(CellLanguage.MARKDOWN).skips_execution_provenance is True

    def test_others_compute_provenance(self):
        for lang in (CellLanguage.PYTHON, CellLanguage.PROMPT, CellLanguage.SQL):
            assert get_language_executor(lang).skips_execution_provenance is False, lang

    def test_prompt_and_sql_have_alternate_cache_scheme(self):
        """Per-language cache hash → the generic miss check needs to preserve READY."""
        assert get_language_executor(CellLanguage.PROMPT).has_alternate_cache_scheme is True
        assert get_language_executor(CellLanguage.SQL).has_alternate_cache_scheme is True

    def test_python_and_markdown_use_generic_cache_scheme(self):
        """Python stores under the standard per-variable hash; markdown has no cache."""
        assert get_language_executor(CellLanguage.PYTHON).has_alternate_cache_scheme is False
        assert get_language_executor(CellLanguage.MARKDOWN).has_alternate_cache_scheme is False


class TestErrors:
    """Missing-language path must fail loudly."""

    def test_unregistered_language_raises_unknownlanguageerror(self):
        fake_lang = "totally-not-a-real-language"
        with pytest.raises(UnknownLanguageError):
            get_language_executor(fake_lang)  # type: ignore[arg-type]


class TestRegisterIsExtensible:
    """A new language can register without touching dispatch sites."""

    def test_register_overrides_existing(self):
        """Re-registering a language replaces the prior adapter.

        Matches the analyzer registry's behaviour. Tests rely on this
        to swap in fakes; production code shouldn't.
        """

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
                return True  # Fake — markdown is normally non-batchable.

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
        [CellLanguage.PROMPT, CellLanguage.SQL, CellLanguage.MARKDOWN],
    )
    def test_non_python_languages_return_false(self, language):
        """No need to fabricate a cell; the adapter ignores it for these.

        The PYTHON adapter's ``is_batchable`` runs the full
        worker/loop/timeout/mount check and needs a real cell + executor;
        end-to-end behaviour is covered in ``test_executor_batch.py``.
        """
        # Sentinel cell + executor — should never be touched.
        sentinel = object()
        assert get_language_executor(language).is_batchable(sentinel, sentinel) is False
