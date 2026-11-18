"""The index from a scratchpad cell's lookup key to what its runs read (``code_keys``)."""

from __future__ import annotations

import json

from strata.notebook import code_keys
from strata.notebook.code_keys import INDEX, CodeIndex, code_key


def test_a_run_that_read_nothing_is_never_under_the_lookup_key_itself():
    # So a result stored under the lookup key, by a run that recorded nothing, is not one.
    assert code_key("L", {}) != "L"
    assert code_key("L", {}) != code_key("L", {"/a": "absent"})


def test_resolving_without_observing_takes_each_set_as_it_was_read(tmp_path):
    data = tmp_path / "data.txt"
    data.write_text("1")
    index = CodeIndex(tmp_path)
    seen = {str(data): "sha256:old"}
    index.record("L", seen, {})

    stored = {code_key("L", seen)}
    assert index.resolve("L", stored.__contains__, observe=False).key == code_key("L", seen)
    # Observed, the file is not as recorded.
    observed = index.resolve("L", stored.__contains__)
    assert observed.changed == str(data)
    assert observed.key == code_key("L", {})


def test_the_index_keeps_the_newest_lookup_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(code_keys, "MAX_LOOKUPS", 2)
    index = CodeIndex(tmp_path)
    for lookup in ("a", "b", "a", "c"):
        index.record(lookup, {}, {})

    assert list(json.loads((tmp_path / INDEX).read_text())) == ["a", "c"]


def test_an_index_that_cannot_be_written_costs_nothing_but_itself(tmp_path):
    (tmp_path / ".strata").write_text("not a directory")
    index = CodeIndex(tmp_path)

    index.record("L", {"/a": "absent"}, {})

    assert not (tmp_path / INDEX).exists()
    assert [p.name for p in tmp_path.iterdir()] == [".strata"]
