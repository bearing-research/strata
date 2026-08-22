"""The agent scratchpad skill ships inside the package and is well-formed.

A coding agent discovers it from ``<site-packages>/strata/.agents/skills/``, so
it must travel with the installed package (see the maturin ``include`` entry in
pyproject.toml) and carry the frontmatter that makes it model-invocable.
"""

from __future__ import annotations

import json
from pathlib import Path

import strata

SKILL = Path(strata.__file__).parent / ".agents" / "skills" / "strata-scratchpad" / "SKILL.md"

# The repo also ships the same skill inside a Claude Code plugin (for marketplace
# distribution). These paths are repo-relative (present in a checkout, absent in a
# bare installed package).
REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "strata-scratchpad"
PLUGIN_SKILL = PLUGIN_DIR / "skills" / "strata-scratchpad" / "SKILL.md"
MARKETPLACE = REPO_ROOT / ".claude-plugin" / "marketplace.json"


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


def test_plugin_skill_is_byte_identical_to_the_package_skill():
    # The plugin bundles a copy of the skill for marketplace distribution; keep it
    # byte-identical to the package copy so the two never drift.
    assert PLUGIN_SKILL.is_file(), f"plugin skill copy missing at {PLUGIN_SKILL}"
    assert PLUGIN_SKILL.read_bytes() == SKILL.read_bytes()


def test_plugin_and_marketplace_manifests_wire_up():
    marketplace = json.loads(MARKETPLACE.read_text(encoding="utf-8"))
    assert any(
        p["name"] == "strata-scratchpad" and p["source"] == "./plugins/strata-scratchpad"
        for p in marketplace["plugins"]
    )
    plugin = json.loads((PLUGIN_DIR / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
    assert plugin["name"] == "strata-scratchpad"
