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
import shutil
from pathlib import Path

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from tests.contract.cellgraph import diagnostics_corpus  # noqa: E402

_ORACLE = json.loads((Path(__file__).parent / "diagnostics_oracle.json").read_text())
_CASES = diagnostics_corpus.cases()


def test_the_oracle_holds_every_case():
    assert sorted(_ORACLE) == sorted(_CASES)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_cell_reports_what_strata_reported(name, tmp_path):
    if name.startswith("r-") and shutil.which("Rscript") is None:
        pytest.skip("Rscript is not on PATH")
    diagnostics_corpus.write(tmp_path / name, _CASES[name])
    observed = diagnostics_corpus.observe(tmp_path / name)
    if name.startswith("sql-"):
        observed, expected = _as_recorded(observed), _in_sql_marker(_ORACLE[name])
    else:
        expected = _ORACLE[name]
    assert observed == expected


# Codes from after the recording: SQL's ``--`` checks.
_NEWER_SQL_CODES = {"sql_annotation_hash", "annotation_not_for_sql"}


def _as_recorded(observed: dict) -> dict:
    cells = {
        cell: [d for d in diagnostics if d[0] not in _NEWER_SQL_CODES]
        for cell, diagnostics in observed["cells"].items()
    }
    return {**observed, "cells": cells}


def _in_sql_marker(recorded: dict) -> dict:
    """The recorded messages as a SQL cell now spells its annotations."""
    return json.loads(json.dumps(recorded).replace("`# @sql", "`-- @sql"))
