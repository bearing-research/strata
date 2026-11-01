"""Each language's analyzer is cellgraph's: the one registered there is the one the
session's DAG reads."""

from __future__ import annotations

import cellgraph as cg
import pytest

from strata.notebook.models import CellLanguage
from strata.notebook.parser import parse_notebook
from strata.notebook.session import NotebookSession
from strata.notebook.writer import add_cell_to_notebook, create_notebook, write_cell


@pytest.mark.parametrize("language", list(CellLanguage))
def test_every_language_has_an_analyzer(language):
    assert cg.get_language_analyzer(language.value) is not None


def test_the_session_reads_the_registered_analyzer(tmp_path):
    class FakeAnalyzer:
        owns_key = True

        def analyze(self, cell):
            return cg.Analyzed(defines=["fake"])

        def normalize(self, body):
            return body

    nb = create_notebook(tmp_path, "Registry", initialize_environment=False)
    add_cell_to_notebook(nb, "m", None, language="markdown")
    write_cell(nb, "m", "Prose.")
    add_cell_to_notebook(nb, "c", "m")
    write_cell(nb, "c", "print(fake)\n")
    original = cg.get_language_analyzer("markdown")
    cg.register_language_analyzer("markdown", FakeAnalyzer())
    try:
        session = NotebookSession(parse_notebook(nb), nb)
    finally:
        cg.register_language_analyzer("markdown", original)
    markdown, reader = session.notebook_state.cells
    assert markdown.defines == ["fake"]
    assert reader.upstream_ids == ["m"]
