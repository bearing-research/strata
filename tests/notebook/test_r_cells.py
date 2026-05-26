"""Cross-language R cell integration tests — the #59 capstone suite.

This file pins the end-to-end "Python ↔ R" story: Python cells produce
artifacts that R cells consume via Arrow IPC, R cells produce artifacts
that downstream Python cells read back. All gated on Rscript + the R
``arrow`` package being available; tests skip cleanly otherwise.

First slice (this PR): one smoke test that exercises the full
Py → R → Py loop. Cross-language error shapes, provenance / cache
behaviour, annotations, and real-renv restore land in follow-up PRs.
"""

from __future__ import annotations

import pytest

from strata.notebook.executor import CellExecutor
from tests.notebook.conftest import skip_if_no_r, skip_if_no_r_arrow

pytestmark = [skip_if_no_r, skip_if_no_r_arrow]
# Note: deliberately NOT ``@pytest.mark.integration``. That marker
# opts out of conftest's autouse ``fast_notebook_env`` override, which
# replaces the Python cell's ``uv run python harness.py`` invocation
# with a direct call to the dev interpreter (so the test gets pandas /
# pyarrow without a per-notebook ``uv sync``). The R harness path
# (``_run_r_harness``) is *not* overridden — it always shells out to
# real ``Rscript`` — so leaving the autouse override active gives us
# the natural mix: Python cells run against the dev venv, R cells run
# against the system R install. The renv-managed library tier that
# the issue describes lands in a follow-up PR.


@pytest.mark.asyncio
async def test_py_to_r_to_py_arrow_roundtrip(r_notebook):
    """A Python DataFrame round-trips through an R cell back into Python.

    Three cells:
      c1 (Python) — build a ``pandas.DataFrame`` and bind it as ``df``.
      c2 (R)      — read ``df`` as a ``data.frame``, append a derived
                    column, bind the result as ``df_r``.
      c3 (Python) — read ``df_r`` back (as a pandas DataFrame via the
                    Arrow IPC handoff) and compute a scalar.

    What this pins:
      * R harness ingests an Arrow IPC artifact produced by the Python
        harness (``arrow::read_ipc_stream`` round-trips
        column names + types).
      * R harness emits an Arrow IPC artifact via
        ``arrow::write_ipc_stream``, picked up by the Python harness on
        the next cell with no extra coercion.
      * DAG analysis on a mixed-language notebook wires Python ↔ R
        upstream/downstream edges off bare variable names — no
        language-specific annotation needed.

    Materialisation: each cell runs with ``materialize_upstreams=True``
    (the default), so calling ``execute_cell("c3", ...)`` would also
    walk back through c2 and c1. Spelling each ``execute_cell`` out
    here keeps the failure-localisation obvious — if c2 breaks, the
    test surfaces it on the c2 assertion rather than a confusing
    cascade error on c3.
    """
    py_c1 = "import pandas as pd\ndf = pd.DataFrame({'x': [1, 2, 3], 'y': [10, 20, 30]})\n"
    r_c2 = "df_r <- df\ndf_r$z <- df_r$x + df_r$y\n"
    py_c3 = "total = int(df_r['z'].sum())\n"

    _, session = r_notebook(
        cells=[
            ("c1", None, py_c1, "python"),
            ("c2", "c1", r_c2, "r"),
            ("c3", "c2", py_c3, "python"),
        ]
    )
    executor = CellExecutor(session)

    r1 = await executor.execute_cell("c1", py_c1)
    assert r1.success is True, r1.error
    assert "df" in r1.outputs
    assert r1.outputs["df"]["content_type"] == "arrow/ipc"

    r2 = await executor.execute_cell("c2", r_c2)
    assert r2.success is True, r2.error
    assert "df_r" in r2.outputs
    assert r2.outputs["df_r"]["content_type"] == "arrow/ipc"
    # R round-trip preserves the three-row shape + adds the derived
    # column; ``rows`` / ``columns`` come from harness.R's write_arrow
    # metadata.
    assert r2.outputs["df_r"]["rows"] == 3
    assert r2.outputs["df_r"]["columns"] == 3

    r3 = await executor.execute_cell("c3", py_c3)
    assert r3.success is True, r3.error
    # 11 + 22 + 33 == 66.
    assert r3.outputs["total"]["preview"] == 66
