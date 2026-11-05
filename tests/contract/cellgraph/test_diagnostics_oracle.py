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
# Added since the recording. The corpus's prompts name inputs for the wiring, with no cell
# defining them, so they report this; every other report stays as recorded.
_NEWER_THAN_THE_ORACLE = "prompt_input_undefined"


def _without_newer_code(observed: dict) -> dict:
    cells = {
        cell: [d for d in diagnostics if d[0] != _NEWER_THAN_THE_ORACLE]
        for cell, diagnostics in observed["cells"].items()
    }
    return {**observed, "cells": cells}


def test_the_oracle_holds_every_case():
    assert sorted(_ORACLE) == sorted(_CASES)


@pytest.mark.parametrize("name", sorted(_CASES))
def test_a_cell_reports_what_strata_reported(name, tmp_path):
    if name.startswith("r-") and shutil.which("Rscript") is None:
        pytest.skip("Rscript is not on PATH")
    diagnostics_corpus.write(tmp_path / name, _CASES[name])
    assert _without_newer_code(diagnostics_corpus.observe(tmp_path / name)) == _ORACLE[name]
