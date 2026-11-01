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

# A Strata bug the prompt trace records: under `strata run` the executor keys a prompt
# cell's downstream without the prompt's output, so after the [ai] model changes, d3 is
# served its old answer as a hit. Staleness and cellgraph key it on the output, under a
# key never published, so they cannot reproduce the hit. Fixing the executor
# re-records the trace and empties this.
KNOWN_RECORDED_BUGS = {
    ("prompt", 3, "d3", "cache decision"),
    ("prompt", 4, "d3", "cache decision"),
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
