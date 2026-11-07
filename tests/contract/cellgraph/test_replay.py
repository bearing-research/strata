"""cellgraph, with Strata's vocabulary, replays every recorded trace exactly.

Each step's notebook is rebuilt from files alone and analyzed under the
``legacy-1`` formats; the DAG, names, diagnostics, plannability, plans, every
definition, execution and instance key, and every run's cache decision must match
what Strata recorded (``replay.py``). A mismatch names the scenario, the step, the
cell and the first part that differs.
"""

from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from strata.notebook import vocabulary  # noqa: E402
from strata.notebook.annotations import ANNOTATION_KEYS  # noqa: E402
from tests.contract.cellgraph import record, replay  # noqa: E402

# What a trace records that is a Strata bug, as (scenario, step, cell, part): the replay
# may differ there. Fixing the bug re-records the trace and removes its entry.
KNOWN_RECORDED_BUGS: set[tuple[str, int, str, str]] = {
    # An R cell with no consumed variable and no display is never cached, so it has nothing
    # to find: rr only assigns, and rc and r2 only print. r2's key repeats after r1's
    # metadata-only edits (@timeout, @name), which are not in r1's key.
    ("r_directives", 3, "r2", "cache decision"),
    ("r_directives", 5, "r2", "cache decision"),
    ("r_directives", 17, "rr", "cache decision"),
    ("r_directives", 23, "rc", "cache decision"),
}


def assert_replays(name, tmp_path):
    report = replay.replay(name, tmp_path)
    found = {(m.scenario, m.step, m.cell, m.part) for m in report.mismatches}
    unexpected = [
        m
        for m in report.mismatches
        if (m.scenario, m.step, m.cell, m.part) not in KNOWN_RECORDED_BUGS
    ]
    assert not unexpected, "\n".join(str(m) for m in unexpected)
    expected_bugs = {bug for bug in KNOWN_RECORDED_BUGS if bug[0] == name}
    assert found == expected_bugs, f"a known recorded bug no longer shows: {expected_bugs - found}"
    assert report.keyed > 0


@pytest.mark.parametrize("name", [n for n in record.scenario_names() if n not in record.REQUIRES])
def test_the_trace_replays(name, tmp_path):
    assert_replays(name, tmp_path)


def test_the_vocabulary_has_every_annotation_key():
    assert vocabulary.strata_vocabulary().keys == ANNOTATION_KEYS


def test_sql_and_prompt_cells_are_host_keyed():
    import cellgraph as cg

    assert not cg.get_language_analyzer("sql").owns_key
    assert not cg.get_language_analyzer("prompt").owns_key
    assert cg.get_language_analyzer("python").owns_key
    assert cg.get_language_analyzer("r").owns_key
