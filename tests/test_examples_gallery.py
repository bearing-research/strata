"""Every notebook under examples/ is listed in both galleries and in the docs' Examples tab."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_EXAMPLES = sorted(p.parent.name for p in (_REPO / "examples").glob("*/notebook.toml"))


def test_there_are_examples_to_check() -> None:
    assert len(_EXAMPLES) > 10


@pytest.mark.parametrize(
    ("gallery", "link"),
    [
        ("docs/notebook/examples.md", "../examples/{}.md"),
        ("examples/README.md", "[`{}/`]({}/)"),
        ("mkdocs.yml", "- examples/{}.md"),
    ],
)
def test_the_gallery_links_every_example(gallery: str, link: str) -> None:
    text = (_REPO / gallery).read_text(encoding="utf-8")

    missing = [name for name in _EXAMPLES if link.format(name, name) not in text]

    assert missing == []


@pytest.mark.parametrize("name", _EXAMPLES)
def test_an_examples_readme_is_titled_with_its_notebooks_name(name: str) -> None:
    """The README's title is the docs page's title and its entry in the Examples tab."""
    notebook = tomllib.loads((_REPO / "examples" / name / "notebook.toml").read_text())
    readme = (_REPO / "examples" / name / "README.md").read_text(encoding="utf-8")

    assert readme.split("\n", 1)[0] == f"# {notebook['name']}"
