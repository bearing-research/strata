"""The R scenario's trace and its replay, apart so CI's R job runs them without the others.

See ``test_cellgraph_traces.py``.
"""

from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from tests.contract.cellgraph import record  # noqa: E402
from tests.contract.cellgraph.test_replay import assert_replays  # noqa: E402


def test_the_r_trace_is_reproduced():
    reason = record.missing_requirement("r_directives")
    if reason is not None:
        pytest.skip(reason)

    difference = record.difference_from_trace("r_directives")

    assert difference is None, (
        f"{difference}\nIf this change is intended, re-record with "
        "`uv run python -m tests.contract.cellgraph.record --update r_directives`."
    )


def test_the_r_trace_replays(tmp_path):
    reason = record.missing_requirement("r_directives")
    if reason is not None:
        pytest.skip(reason)

    assert_replays("r_directives", tmp_path)
