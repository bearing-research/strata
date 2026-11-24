"""Validation on cellgraph reports what Strata's own validators reported.

``diagnostics_oracle.json`` was recorded by those validators, before cellgraph
replaced them; each case's notebook, opened now, must report the same diagnostics
(code, severity, line, message, in order), the same upstream cells and the same
``dag_error``. It was recorded with ``Rscript`` on PATH, which an R cell's
analysis needs; CI's R job runs the R cases.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from tests.contract.cellgraph import diagnostics_corpus  # noqa: E402

_ORACLE = json.loads((Path(__file__).parent / "diagnostics_oracle.json").read_text())
_CASES = diagnostics_corpus.cases()
# Codes added since the recording; every other report stays as recorded. The corpus's
# prompts name inputs for the wiring, with no cell defining them, so they report
# ``prompt_input_undefined``; two are SQL's ``--`` checks, and a cell of another language
# that fans out reports ``per_variant_not_python``.
_NEWER_CODES = {
    "prompt_input_undefined",
    "sql_annotation_hash",
    "annotation_not_for_sql",
    "per_variant_not_python",
}


def _as_recorded(observed: dict) -> dict:
    cells = {
        cell: [d for d in diagnostics if d[0] not in _NEWER_CODES]
        for cell, diagnostics in observed["cells"].items()
    }
    return {**observed, "cells": cells}


def _in_sql_marker(recorded: dict) -> dict:
    """The recorded messages as a SQL cell now spells its annotations."""
    return json.loads(json.dumps(recorded).replace("`# @sql", "`-- @sql"))


def test_the_oracle_holds_every_case():
    assert sorted(_ORACLE) == sorted(_CASES)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=pytest.mark.r) if name.startswith("r-") else name
        for name in sorted(_CASES)
    ],
)
def test_a_cell_reports_what_strata_reported(name, tmp_path):
    diagnostics_corpus.write(tmp_path / name, _CASES[name])
    observed = _as_recorded(diagnostics_corpus.observe(tmp_path / name))
    expected = _in_sql_marker(_ORACLE[name]) if name.startswith("sql-") else _ORACLE[name]
    assert observed == expected
