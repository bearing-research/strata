"""Every notebook under examples/ is listed in both galleries."""

from __future__ import annotations

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
    ],
)
def test_the_gallery_links_every_example(gallery: str, link: str) -> None:
    text = (_REPO / gallery).read_text(encoding="utf-8")

    missing = [name for name in _EXAMPLES if link.format(name, name) not in text]

    assert missing == []
