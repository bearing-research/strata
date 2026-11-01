"""Strata still does what the cellgraph traces recorded, step by step.

Each scenario is re-recorded and compared with ``traces/<name>.json``, exactly,
except the causality map. A change here is a change in Strata's graph, keys, cache
decisions or staleness: if it is intended, re-record with
``uv run python -m tests.contract.cellgraph.record --update <name>`` and say in the
PR which behavior changed and why.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from tests.contract.cellgraph import record

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)


@pytest.mark.parametrize("name", record.scenario_names())
def test_the_trace_is_reproduced(name):
    reason = record.missing_requirement(name)
    if reason is not None:
        pytest.skip(reason)
    expected = json.loads((record.TRACES / f"{name}.json").read_text())

    difference = record.describe_difference(expected, record.record(name))

    assert difference is None, (
        f"{difference}\nIf this change is intended, re-record with "
        f"`uv run python -m tests.contract.cellgraph.record --update {name}`."
    )


def test_every_scenario_has_a_trace_and_a_readme_entry():
    names = record.scenario_names()
    readme = (record.HERE / "README.md").read_text()
    table = readme.split("## Scenarios", 1)[1].split("\n## ", 1)[0]
    assert sorted(p.stem for p in record.TRACES.glob("*.json")) == names
    assert sorted(re.findall(r"^\| `(\w+)` \|", table, re.MULTILINE)) == names


def test_traces_are_format_1():
    for path in record.TRACES.glob("*.json"):
        assert json.loads(path.read_text())["trace_format"] == record.TRACE_FORMAT == 1


def test_the_difference_names_its_path():
    trace = {"steps": [{"cells": {"c1": {"keys": {"input_hashes": ["a", "b", "c"]}}}}]}
    changed = json.loads(json.dumps(trace))
    changed["steps"][0]["cells"]["c1"]["keys"]["input_hashes"][2] = "d"
    assert record.first_difference(trace, changed) == "steps[0].cells.c1.keys.input_hashes[2]"


def test_the_causality_map_is_outside_the_gate():
    trace = {"steps": [{"cells": {}, record.OUTSIDE_EXACT_GATE: {"c1": "self"}}]}
    moved = {"steps": [{"cells": {}, record.OUTSIDE_EXACT_GATE: {"c1": "upstream"}}]}
    assert record.describe_difference(trace, moved) is None
