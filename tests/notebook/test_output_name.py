"""A SQL or prompt cell's ``@name`` is its output variable, so it must be one.

It used to fall back to ``result`` without a word, and a reader of the name it asked for
found nothing.
"""

from __future__ import annotations

import pytest

from strata.notebook.annotation_validation import blocking_diagnostic
from strata.notebook.models import CellState, ConnectionSpec, NotebookState


def _blocking(language: str, source: str):
    cell = CellState(id="c", source=source, language=language)
    state = NotebookState(
        id="nb", cells=[cell], connections=[ConnectionSpec(name="db", driver="sqlite")]
    )
    return blocking_diagnostic(cell, state)


@pytest.mark.parametrize("name", ["top-orders", "1st", "class", "\ufb01le"])
@pytest.mark.parametrize(
    ("language", "head", "line"),
    [("sql", "-- @sql connection=db\n-- @name {}\n", 2), ("prompt", "# @name {}\n", 1)],
)
def test_a_name_that_is_no_variable_blocks_the_cell(language, head, line, name):
    diagnostic = _blocking(language, head.format(name) + "SELECT 1\n")

    assert diagnostic is not None
    assert (diagnostic.code, diagnostic.line) == ("name_unreadable", line)
    assert "output variable" in diagnostic.message


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("sql", "-- @sql connection=db\n-- @name top_orders\nSELECT 1\n"),
        ("prompt", "# @name summary\nSay hi.\n"),
        # A Python cell's name is a label for the DAG, any text.
        ("python", "# @name Load the data\nx = 1\n"),
    ],
)
def test_a_name_that_is_a_variable_or_a_label_is_fine(language, source):
    assert _blocking(language, source) is None
