"""docs/changelog.md mirrors CHANGELOG.md; the Unreleased section is where they drift."""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent


def _unreleased(text: str) -> str:
    start = text.index("## Unreleased\n")
    end = text.find("\n## ", start + 1)
    return text[start : end if end != -1 else len(text)]


def test_docs_changelog_mirrors_the_unreleased_section() -> None:
    root = _unreleased((_REPO / "CHANGELOG.md").read_text(encoding="utf-8"))
    docs = _unreleased((_REPO / "docs" / "changelog.md").read_text(encoding="utf-8"))

    # The docs page sits inside docs/, so its relative links drop that prefix.
    assert docs == root.replace("](docs/", "](")
