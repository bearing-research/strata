"""Mkdocs hook: render each ``examples/*/notebook.toml`` to ``docs/examples/<name>.md``.

The generated files are gitignored. Registered in ``mkdocs.yml``:

    hooks:
      - docs_hooks/export_examples.py
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def on_pre_build(config):
    """Render every notebook under ``examples/`` to ``docs/examples/<name>.md``."""
    config_file = config.get("config_file_path")
    if not config_file:
        return
    repo_root = Path(config_file).parent
    examples_dir = repo_root / "examples"
    docs_dir = Path(config["docs_dir"])
    out_dir = docs_dir / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not examples_dir.is_dir():
        return

    try:
        from strata.notebook.export import export_notebook
    except Exception as exc:  # pragma: no cover - import errors surface in build logs
        logger.warning("Skipping export hook: strata import failed: %s", exc)
        return

    rendered_count = 0
    for notebook_dir in sorted(examples_dir.iterdir()):
        if not (notebook_dir / "notebook.toml").is_file():
            continue
        # Skip scratch dirs; ``test_notebook`` is a conventional local-only one.
        if notebook_dir.name.startswith(("_", ".")) or notebook_dir.name == "test_notebook":
            continue
        try:
            rendered = export_notebook(notebook_dir)
        except Exception as exc:
            logger.warning(
                "export_examples: %s failed to render: %s",
                notebook_dir.name,
                exc,
            )
            continue

        target = out_dir / f"{notebook_dir.name}.md"
        rendered = with_thumbnail(rendered, notebook_dir.name, docs_dir / "assets" / "examples")
        target.write_text(rendered, encoding="utf-8")
        rendered_count += 1

    if rendered_count:
        logger.info("export_examples: rendered %d notebook(s)", rendered_count)


def with_thumbnail(rendered: str, name: str, thumbs_dir: Path) -> str:
    """Put the example's screenshot (``scripts/capture_docs_shots.py``) under its title.

    An example with no screenshot yet (one whose run needs what the capturing
    machine lacked) renders without one.
    """
    if not (thumbs_dir / f"{name}-light.webp").is_file():
        return rendered
    lines = rendered.split("\n")
    title_at = next((i for i, line in enumerate(lines) if line.startswith("# ")), None)
    if title_at is None:
        return rendered
    title = lines[title_at].removeprefix("# ").strip()
    alt = f"{title}: what a run of the notebook shows"
    images = [
        f"![{alt}](../assets/examples/{name}-{theme}.webp#only-{theme})"
        for theme in ("light", "dark")
    ]
    return "\n".join([*lines[: title_at + 1], "", *images, *lines[title_at + 1 :]])
