"""Mkdocs hook: render every notebook in ``examples/`` to docs.

Runs ``strata.notebook.export.export_notebook`` against each
``examples/*/notebook.toml`` and writes the resulting markdown to
``docs/examples/<name>.md`` so mkdocs picks it up at build time.
The generated files are gitignored — they live in the build tree
only, never in commits.

Registered in ``mkdocs.yml``:

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
        # Skip scratch directories. ``test_notebook`` is the conventional
        # local-only example directory contributors create when poking at
        # the server; not part of the shipped catalog.
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
        target.write_text(rendered, encoding="utf-8")
        rendered_count += 1

    if rendered_count:
        logger.info("export_examples: rendered %d notebook(s)", rendered_count)
