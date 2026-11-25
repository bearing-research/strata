"""Mkdocs hook: render each ``examples/*/notebook.toml`` to ``docs/examples/<name>.md``.

A page is the example as it ran: the README's title and opening, then every cell
with its outputs, then the rest of the README under "About this example". The
outputs come from a run ``scripts/capture_docs_shots.py --only examples`` recorded
in ``docs/assets/examples/runs/<name>.md`` (its pictures beside it); the docs build
runs nothing. A run is used only while the cells are as they were when it ran
(``source_fingerprint``). Without one, the page shows the code alone.

The generated files are gitignored. Registered in ``mkdocs.yml``:

    hooks:
      - docs_hooks/export_examples.py
"""

from __future__ import annotations

import hashlib
import logging
import re
import tomllib
from pathlib import Path

logger = logging.getLogger(__name__)

# Recorded runs, relative to the docs directory; excluded from the build as pages.
RUNS_DIR = Path("assets") / "examples" / "runs"
# A table output shows this many rows on a page, then says how many it has.
TABLE_ROWS = 8
# What an example's cells are exported with, for a page: the page has its own title
# and places the README itself.
PAGE_EXPORT = {"include_readme": False, "max_table_rows": TABLE_ROWS}
_RUN_HEADER = re.compile(r"<!-- strata example run, source ([0-9a-f]{16}) -->\n")
# A code cell's name: its heading, then its chips, its code or the prompt note. Marked
# so the page draws it as the editor does a cell's header, apart from the prose's headings.
_CELL_HEADING = re.compile(r"^(## [^\n]*)(\n\n(?:<sub>|```|\*Prompt cell))", re.M)
# The cell's kind is plain from its code; the other chips (worker, loop, variant) stay.
_KIND_CHIP = re.compile(r"<sub>\*\*kind\*\* [a-z]+(?:  ·  )?")


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
        from strata.notebook.export import ExportOptions, export_notebook
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
        name = notebook_dir.name
        run = recorded_run(docs_dir / RUNS_DIR / f"{name}.md", source_fingerprint(notebook_dir))
        if run is not None:
            cells = run
        else:
            try:
                cells = export_notebook(notebook_dir, ExportOptions(**PAGE_EXPORT))
            except Exception as exc:
                logger.warning("export_examples: %s failed to render: %s", name, exc)
                continue
        readme = (notebook_dir / "README.md").read_text(encoding="utf-8")
        page = render_page(readme, cells, ran=run is not None)
        if run is None:
            # Its card is the one picture of what it makes.
            page = with_thumbnail(page, name, docs_dir / "assets" / "examples")
        (out_dir / f"{name}.md").write_text(page, encoding="utf-8")
        rendered_count += 1

    if rendered_count:
        logger.info("export_examples: rendered %d notebook(s)", rendered_count)


def source_fingerprint(notebook_dir: Path) -> str:
    """The cells' files and sources, in the notebook's order: what a recorded run showed."""
    toml = tomllib.loads((notebook_dir / "notebook.toml").read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for cell in toml.get("cells", []):
        digest.update(cell["file"].encode() + b"\0")
        digest.update((notebook_dir / "cells" / cell["file"]).read_bytes() + b"\0")
    return digest.hexdigest()[:16]


def run_header(fingerprint: str) -> str:
    """The first line of a recorded run, naming the source it ran."""
    return f"<!-- strata example run, source {fingerprint} -->\n"


def recorded_run(path: Path, fingerprint: str) -> str | None:
    """The recorded run at *path*, if it ran the cells as they are now."""
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    match = _RUN_HEADER.match(text)
    if match is None or match.group(1) != fingerprint:
        logger.warning(
            "export_examples: %s ran other cells than the example has now; showing its code "
            "only. Record it again: uv run python scripts/capture_docs_shots.py --only "
            "examples --example %s",
            path.name,
            path.stem,
        )
        return None
    return text


def render_page(readme: str, cells: str, *, ran: bool) -> str:
    """The example page: the README's title and opening, the cells, then the README's notes.

    The first cell's own title would repeat the page's, so it goes.
    """
    title, _, rest = readme.partition("\n")
    rest = "\n" + rest.strip()
    lead, _, notes = rest.partition("\n## ")
    cells = _RUN_HEADER.sub("", cells, count=1)
    cells = "\n".join(_tidy_line(line) for line in cells.strip().split("\n"))
    cells = re.sub(r"^<sub></sub>\n\n", "", cells, flags=re.M)
    cells = re.sub(r"\A# [^\n]*\n+", "", cells)
    cells = _CELL_HEADING.sub(r"\1 { .nb-cell }\2", cells)
    pieces = [title, ""]
    if lead.strip():
        pieces += [lead.strip(), ""]
    if not ran:
        pieces += ["*No run is recorded for this example, so the page shows its code only.*", ""]
    pieces += ["---", "", cells.strip(), ""]
    if notes.strip():
        about = re.sub(r"^(#{2,5}) ", r"#\1 ", "## " + notes.strip(), flags=re.M)
        pieces += ["---", "", "## About this example", "", about, ""]
    return "\n".join(pieces)


def _tidy_line(line: str) -> str:
    return _KIND_CHIP.sub("<sub>", line) if line.startswith("<sub>") else line


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
