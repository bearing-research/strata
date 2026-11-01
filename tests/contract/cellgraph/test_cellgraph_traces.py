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

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

from tests.contract.cellgraph import record  # noqa: E402


# A scenario that needs more than the dev environment has its own test file, so the
# CI job that has the tool runs only that one.
@pytest.mark.parametrize("name", [n for n in record.scenario_names() if n not in record.REQUIRES])
def test_the_trace_is_reproduced(name):
    difference = record.difference_from_trace(name)

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


def test_a_bool_is_not_an_int():
    trace = {"steps": [{"cells": {"c1": {"run": {"cache_hit": True}}}}]}
    changed = {"steps": [{"cells": {"c1": {"run": {"cache_hit": 1}}}}]}
    assert record.first_difference(trace, changed) == "steps[0].cells.c1.run.cache_hit"


def test_values_are_compared_as_written():
    trace = {"steps": [{"cells": {"c1": {"keys": {"input_hashes": ["a", "b"]}}}}]}
    as_tuple = {"steps": [{"cells": {"c1": {"keys": {"input_hashes": ("a", "b")}}}}]}
    assert record.describe_difference(trace, as_tuple) is None


def test_every_scenario_with_a_requirement_has_its_own_test():
    for name in record.REQUIRES:
        assert (
            f'record.difference_from_trace("{name}")'
            in (record.HERE / f"test_cellgraph_{name}_trace.py").read_text()
        )


def test_the_causality_map_is_outside_the_gate():
    trace = {"steps": [{"cells": {}, record.OUTSIDE_EXACT_GATE: {"c1": "self"}}]}
    moved = {"steps": [{"cells": {}, record.OUTSIDE_EXACT_GATE: {"c1": "upstream"}}]}
    assert record.describe_difference(trace, moved) is None
