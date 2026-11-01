"""The vocabulary's parsed values are what ``parse_annotations`` reads, line for line.

Strata's facade builds ``CellAnnotations`` from them, so they must agree on every
source: readable lines, and the unreadable ones the vocabulary reports as errors,
which ``parse_annotations`` reads leniently (``@nocache maybe`` is on).
"""

from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("the notebook subsystem is not run on Windows", allow_module_level=True)

import cellgraph as cg  # noqa: E402

from strata.notebook.vocabulary import (  # noqa: E402
    annotations_from,
    parse_values,
    strata_vocabulary,
)
from tests.contract.cellgraph import record, replay  # noqa: E402
from tests.contract.cellgraph import test_vocabulary_parity as parity  # noqa: E402
from tests.contract.cellgraph.legacy_annotations import (  # noqa: E402
    legacy_parse_annotations as parse_annotations,
)

# Lines each key's parser must read as ``parse_annotations`` does, readable or not.
EDGES = {
    "worker": ["# @worker gpu\n# @worker", "# @worker\n# @worker gpu", "# @worker  gpu  "],
    "timeout": ["# @timeout abc\n# @timeout 30", "# @timeout 30\n# @timeout abc", "# @timeout -5"],
    "env": ["# @env A=1\n# @env B\n# @env A=2", "# @env =x", "# @env  K = v "],
    "mount": ["# @mount a\n# @mount d file:///tmp/x rw", "# @mount d s3://b/k ro credential=c"],
    "table": ["# @table t ns.t\n# @table bad", "# @table t ns.t snapshot=5"],
    "write_table": ["# @write_table y cat:ns.t merge on=a,b", "# @write_table y"],
    "output_dir": ["# @output_dir out\n# @output_dir out\n# @output_dir 1bad"],
    "fetch": ["# @fetch f https://x/y sha256=ab refetch=never", "# @fetch f"],
    "dataset": ["# @dataset d name@v=2", "# @dataset d"],
    "nocache": [
        "# @nocache maybe",
        "# @nocache\n# @nocache off",
        "# @nocache off\n# @stochastic",
        "# @stochastic\n# @nocache off",
        "# @stochastic off\n# @nocache",
    ],
    "stochastic": ["# @stochastic maybe", "# @stochastic\n# @stochastic off"],
    "name": ["# @name a\n# @name", "# @name\n# @name b", "# @name 1bad"],
    "model": ["# @model m\n# @model"],
    "temperature": ["# @temperature -1", "# @temperature 0.5\n# @temperature hot"],
    "output": ["# @output xml", "# @output json\n# @output"],
    "max_tokens": ["# @max_tokens 0", "# @max_tokens 10\n# @max_tokens many"],
    "system": ["# @system be brief\n# @system"],
    "sql": [
        "# @sql connection=a bogus=1",
        "# @sql connection=a\n# @sql write=maybe",
        "# @sql write=true connection=b",
    ],
    "cache": ["# @cache forever", "# @cache ttl=0", "# @cache bogus", "# @cache ttl=60 extra"],
    "loop": [
        "# @loop max_iter=abc carry=x",
        "# @loop_until x > 3",
        "# @loop max_iter=3 carry=x\n# @loop_until done",
        "# @loop max_iter=3 carry=x start_from=c1@iter=2",
    ],
    "variant": ["# @variant g a\n# @variant onlyone", "# @variant onlyone\n# @variant g b"],
    "live": ["# @live maybe", "# @live\n# @live off"],
    "per_variant": ["# @per_variant a b", "# @per_variant g\n# @per_variant", "# @per_variant 1"],
    "after": ["# @after a, b\n# @after\n# @after a c", "# @after"],
}
LANGUAGES = ("python", "r", "sql", "prompt", "widget", "markdown")


def _corpus():
    """``(language, source)`` for every edge, case and scenario cell."""
    for sources in EDGES.values():
        for source in sources:
            for language in LANGUAGES:
                yield language, source + "\nx = 1"
    for cells in parity.CASES.values():
        yield from ((language, source) for _id, language, source, _toml in cells)
    for cells, _toml in parity.CORE_CASES.values():
        yield from ((language, source) for _id, language, source, _toml in cells)
    for source, _expected in parity.SQL_CASES.values():
        yield "sql", "# @name t\n" + source


def _check(language, source):
    values, directives = parse_values(source, language)
    assert annotations_from(values, directives) == parse_annotations(source, language), (
        language,
        source,
    )
    # The parse phase of a full analysis reads the same values.
    analysis = cg.analyze(
        [cg.Cell("c", source, language, meta={"language": language})], strata_vocabulary()
    )
    assert analysis.cells["c"].annotations == values, (language, source)


def test_every_edge_and_case_reads_as_parse_annotations():
    for language, source in _corpus():
        _check(language, source)


@pytest.mark.parametrize("name", [n for n in record.scenario_names() if n not in record.REQUIRES])
def test_every_scenario_cell_reads_as_parse_annotations(name, tmp_path):
    for _index, _step, state in replay.notebooks(name, tmp_path):
        for cell in state.cells:
            _check(cell.language.value, cell.source)


def test_the_edges_cover_every_key():
    assert set(EDGES) | {"app", "validate_retries", "output_schema"} == {
        a.key for a in strata_vocabulary()
    }
