"""The agent scratchpad skill ships inside the package and is well-formed.

A coding agent discovers it from ``<site-packages>/strata/.agents/skills/``, so
it must travel with the installed package (see the maturin ``include`` entry in
pyproject.toml) and carry the frontmatter that makes it model-invocable.
"""

from __future__ import annotations

from pathlib import Path

import strata

SKILL = Path(strata.__file__).parent / ".agents" / "skills" / "strata-scratchpad" / "SKILL.md"


def test_skill_ships_with_the_package():
    assert SKILL.is_file(), f"scratchpad skill missing at {SKILL}"


def test_skill_has_name_and_description_frontmatter():
    text = SKILL.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    _, frontmatter, _ = text.split("---\n", 2)
    assert "name: strata-scratchpad" in frontmatter
    # The description is the trigger — it must mention the scratchpad intent.
    desc = next(ln for ln in frontmatter.splitlines() if ln.startswith("description:"))
    assert "scratchpad" in desc.lower()


def test_skill_points_at_the_one_call_primitive_not_bash():
    text = SKILL.read_text(encoding="utf-8")
    assert "--run" in text and "run_snippet" in text
    assert "# @nocache" in text
