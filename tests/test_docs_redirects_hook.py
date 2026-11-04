"""The docs hook that keeps split pages' URLs. docs.yml builds the real stubs."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "redirects", Path(__file__).parents[1] / "docs_hooks" / "redirects.py"
)
assert _spec and _spec.loader
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)


def test_an_anchor_goes_to_the_first_page_that_has_it():
    anchors = {"a.md": {"one", "both"}, "b.md": {"two", "both"}}

    assert hook.anchors_to_pages(["a.md", "b.md"], anchors) == {
        "both": "a.md",
        "one": "a.md",
        "two": "b.md",
    }
    assert hook.anchors_to_pages(["b.md", "a.md"], anchors)["both"] == "b.md"


def test_a_page_that_is_not_in_the_list_gives_no_anchors():
    assert hook.anchors_to_pages(["a.md"], {"b.md": {"two"}}) == {}


def test_a_stub_sends_a_known_anchor_to_its_page_and_keeps_an_unknown_one():
    html = hook.stub_html("../../new/", {"sql": "../../new/sql/#sql"})

    assert 'const moved = {"sql": "../../new/sql/#sql"};' in html
    assert 'moved[anchor] ?? "../../new/" + location.hash' in html


def test_a_stub_works_without_a_script():
    html = hook.stub_html("../../new/", {})

    assert '<meta http-equiv="refresh" content="0; url=../../new/">' in html
    assert '<a href="../../new/">' in html


def test_a_split_page_names_real_docs():
    docs = Path(__file__).parents[1] / "docs"
    for default, pages in hook.SPLIT.values():
        assert default in pages
        assert all((docs / page).is_file() for page in pages)
    for renamed in hook.RENAMED.values():
        assert all((docs / target.partition("#")[0]).is_file() for target in renamed.values())


def test_a_split_page_leaves_no_source_behind_it():
    # A source at the old path would build over its stub.
    docs = Path(__file__).parents[1] / "docs"
    assert not [old for old in hook.SPLIT if (docs / f"{old}.md").exists()]
