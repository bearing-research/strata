"""The READMEs are PyPI long descriptions, where a relative link or image does not resolve."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
_READMES = [
    _REPO / "README.md",
    _REPO / "packages" / "strata-client" / "README.md",
    _REPO / "packages" / "strata-pool" / "README.md",
]
_TARGETS = re.compile(r"\]\(([^)\s]+)|(?:src|srcset|href)=\"([^\"]+)\"")


@pytest.mark.parametrize("readme", _READMES, ids=lambda p: str(p.relative_to(_REPO)))
def test_readme_links_and_images_are_absolute(readme: Path) -> None:
    targets = [a or b for a, b in _TARGETS.findall(readme.read_text(encoding="utf-8"))]

    relative = [t for t in targets if not re.match(r"(?:https?:|mailto:|#)", t)]

    assert relative == []


def test_the_pattern_sees_the_root_readme_images() -> None:
    text = (_REPO / "README.md").read_text(encoding="utf-8")
    targets = [a or b for a, b in _TARGETS.findall(text)]

    assert any(t.endswith("notebook-anatomy-light.png") for t in targets)
