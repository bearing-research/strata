"""docs/changelog.md mirrors CHANGELOG.md; the newest section is where they drift."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _newest(text: str) -> str:
    # The first section, "Unreleased" between releases and the dated one at a cut.
    start = text.index("\n## ") + 1
    end = text.find("\n## ", start)
    return text[start : end if end != -1 else len(text)]


def test_docs_changelog_mirrors_the_newest_section() -> None:
    root = _newest((_REPO / "CHANGELOG.md").read_text(encoding="utf-8"))
    docs = _newest((_REPO / "docs" / "changelog.md").read_text(encoding="utf-8"))

    # The docs page sits inside docs/, so its relative links drop that prefix.
    assert docs == root.replace("](docs/", "](")
