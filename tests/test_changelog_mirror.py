"""docs/changelog.md mirrors CHANGELOG.md; the newest section is where they drift."""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_PRERELEASE = re.compile(r"## \d[\d.]*(?:a|b|rc|\.dev)\d+\b")


def _newest(text: str) -> str:
    # The first section ("Unreleased" between releases, the dated one at a cut),
    # plus the one below while a release candidate's stub sits on top.
    start = end = text.index("\n## ") + 1
    while True:
        prerelease = _PRERELEASE.match(text, end)
        nxt = text.find("\n## ", end)
        end = len(text) if nxt == -1 else nxt + 1
        if not prerelease or nxt == -1:
            return text[start:end]


def test_docs_changelog_mirrors_the_newest_section() -> None:
    root = _newest((_REPO / "CHANGELOG.md").read_text(encoding="utf-8"))
    docs = _newest((_REPO / "docs" / "changelog.md").read_text(encoding="utf-8"))

    # The docs page sits inside docs/, so its relative links drop that prefix.
    assert docs == root.replace("](docs/", "](")
