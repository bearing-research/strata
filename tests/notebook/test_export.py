"""Tests for ``strata.notebook.export``."""

from __future__ import annotations

from pathlib import Path

import pytest

from strata.notebook.export import ExportOptions, export_notebook
from strata.notebook.writer import (
    add_cell_to_notebook,
    create_notebook,
    set_variant_active,
    update_cell_console_output,
    update_cell_display_outputs,
    write_cell,
)


def _make_notebook(tmp_path: Path, name: str = "export_test") -> Path:
    nb_dir = create_notebook(tmp_path, name, initialize_environment=False)
    return nb_dir


def test_export_renders_source_only_when_no_cached_state(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\nprint(x)\n")

    rendered = export_notebook(nb_dir)

    assert "## c1" in rendered
    assert "```python" in rendered
    assert "x = 1" in rendered
    assert 'title="Output"' not in rendered  # no cached outputs


def test_export_renders_arrow_table_preview_as_markdown_table(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "df = pd.DataFrame(...)\n")

    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "arrow/ipc",
                "rows": 3,
                "columns": ["name", "score"],
                "preview": [
                    {"name": "Alice", "score": 95},
                    {"name": "Bob", "score": 87},
                    {"name": "Carol", "score": 92},
                ],
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir)
    assert "| name | score |" in rendered
    assert "| Alice | 95 |" in rendered
    assert "| Carol | 92 |" in rendered


def test_export_renders_arrow_table_with_positional_rows(tmp_path: Path) -> None:
    """The serializer emits preview rows as positional lists (one entry per
    column), not dicts. This is the *real* on-disk shape — the dict-row
    test above mirrors what a hand-written test fixture might look like.
    """
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "df = pd.DataFrame(...)\n")

    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "arrow/ipc",
                "rows": 3,
                "columns": ["name", "score"],
                "preview": [
                    ["Alice", 95],
                    ["Bob", 87],
                    ["Carol", 92],
                ],
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir)
    assert "| name | score |" in rendered
    assert "| Alice | 95 |" in rendered
    assert "| Carol | 92 |" in rendered


def test_export_renders_empty_table_with_header(tmp_path: Path) -> None:
    """A table with columns but no preview rows still emits the header row,
    so the reader can see what columns the cell produces."""
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "df = pd.DataFrame(columns=['name', 'score'])\n")

    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "arrow/ipc",
                "rows": 0,
                "columns": ["name", "score"],
                "preview": [],
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir)
    assert "| name | score |" in rendered  # header still present
    assert "| --- | --- |" in rendered  # separator still present


def test_export_code_fence_grows_when_body_contains_triple_backticks(
    tmp_path: Path,
) -> None:
    """Cells that embed fenced markdown examples (typical for prompt cells
    or library cells documenting their interface) would corrupt the export
    if we always used three backticks. The fence has to be longer than the
    longest backtick run in the body.
    """
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    source = '"""Example doc:\n\n```python\nx = 1\n```\n"""\nx = 1\n'
    write_cell(nb_dir, "c1", source)

    rendered = export_notebook(nb_dir)
    # Outer fence is four backticks; inner ```python survives intact.
    assert "````python" in rendered
    assert "```python\nx = 1\n```" in rendered  # inner fence untouched
    # And the outer fence properly closes
    assert rendered.rstrip().endswith("````\n") or "\n````" in rendered


def test_image_output_renders_when_inline_data_url_present() -> None:
    """Direct renderer test — bypasses the writer's transient-field strip.

    The TOML writer drops inline_data_url on save (it's large); in the
    real flow hydration via the artifact store re-attaches it. This
    unit test exercises the renderer assuming hydration has already
    happened.
    """
    from strata.notebook.export import _render_display_output
    from strata.notebook.models import CellOutput

    data_url = "data:image/png;base64,iVBORw0KGgoAAAANS"
    output = CellOutput(
        content_type="image/png",
        inline_data_url=data_url,
    )

    blocks = _render_display_output(output, notebook_dir=Path("."), notebook_id="dummy")
    from strata.notebook.export import ImageBlock

    assert len(blocks) == 1
    assert isinstance(blocks[0], ImageBlock)
    assert blocks[0].data_url == data_url


def test_export_renders_json_output_as_fenced_json_block(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "result = {'a': 1}\n")

    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "json/object",
                "preview": {"a": 1, "b": [2, 3]},
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir)
    assert '```json title="Output"' in rendered
    assert '"a": 1' in rendered
    assert '"b": [' in rendered


def test_export_skips_pickled_output_with_placeholder(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "obj = MyThing()\n")

    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "pickle/object",
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir)
    assert "Pickled output" in rendered


def test_export_renders_console_output(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "print('hi')\n")

    update_cell_console_output(nb_dir, "c1", "hi\nworld\n", "")

    rendered = export_notebook(nb_dir)
    assert '```text title="stdout"' in rendered
    assert "hi" in rendered
    assert "world" in rendered


def test_export_omits_console_when_disabled(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "print('hi')\n")
    update_cell_console_output(nb_dir, "c1", "hi\n", "")

    rendered = export_notebook(nb_dir, ExportOptions(include_console=False))
    assert 'title="stdout"' not in rendered


def test_export_prompt_cell_never_includes_response(tmp_path: Path) -> None:
    """Privacy default: prompt cell sources render, cached responses don't."""
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "p1", language="prompt")
    write_cell(nb_dir, "p1", "Summarize {{ df }} in one sentence.\n")

    update_cell_display_outputs(
        nb_dir,
        "p1",
        [
            {
                "content_type": "text/markdown",
                "markdown_text": "SECRET LLM RESPONSE WE MUST NOT LEAK",
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir)
    assert "Summarize {{ df }}" in rendered
    assert "SECRET LLM RESPONSE" not in rendered
    assert "response intentionally excluded" in rendered.lower()


def test_export_filters_inactive_variants_by_default(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "load")
    write_cell(nb_dir, "load", "X = 1\n")
    add_cell_to_notebook(nb_dir, "model_a", after_cell_id="load")
    write_cell(nb_dir, "model_a", "# @variant model gpt4\npreds = 1\n")
    add_cell_to_notebook(nb_dir, "model_b", after_cell_id="model_a")
    write_cell(nb_dir, "model_b", "# @variant model claude\npreds = 2\n")
    set_variant_active(nb_dir, "model", "gpt4")

    rendered = export_notebook(nb_dir)
    assert "## model_a" in rendered
    assert "## model_b" not in rendered  # filtered (inactive)


def test_export_with_include_inactive_variants_shows_all(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "model_a")
    write_cell(nb_dir, "model_a", "# @variant model gpt4\npreds = 1\n")
    add_cell_to_notebook(nb_dir, "model_b", after_cell_id="model_a")
    write_cell(nb_dir, "model_b", "# @variant model claude\npreds = 2\n")
    set_variant_active(nb_dir, "model", "gpt4")

    rendered = export_notebook(nb_dir, ExportOptions(include_inactive_variants=True))
    assert "## model_a" in rendered
    assert "## model_b" in rendered


def test_export_renders_readme_intro_when_present(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    (nb_dir / "README.md").write_text(
        "# My Demo Notebook\n\nWalks through the cool feature.\n",
        encoding="utf-8",
    )
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")

    rendered = export_notebook(nb_dir)
    assert "# My Demo Notebook" in rendered
    assert "Walks through the cool feature." in rendered
    # Don't add a duplicate H1 — README already owns the page title.
    assert "# Notebook:" not in rendered


def test_export_falls_back_to_notebook_heading_when_no_readme(tmp_path: Path) -> None:
    """Without a README, the export needs its own H1 so the page has a title."""
    nb_dir = _make_notebook(tmp_path, name="HeadlessNB")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")

    rendered = export_notebook(nb_dir)
    assert "# Notebook: HeadlessNB" in rendered


def test_export_strips_ansi_escape_codes_from_console(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "print('hi')\n")
    # ANSI-colored "RED" string, the kind rich/colorama emits
    ansi_stdout = "\x1b[31mRED\x1b[0m output\n"
    update_cell_console_output(nb_dir, "c1", ansi_stdout, "")

    rendered = export_notebook(nb_dir)
    assert "RED output" in rendered
    assert "\x1b[" not in rendered
    assert "[31m" not in rendered  # raw escape leftover


def test_export_pickle_placeholder_includes_type_hint(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "obj = MyThing()\n")

    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "pickle/object",
                "preview": "<MyThing object>",
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir)
    assert "<MyThing object>" in rendered
    assert "not rendered" in rendered.lower()


def test_export_html_format_returns_html_envelope(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")

    rendered = export_notebook(nb_dir, ExportOptions(output_format="html"))
    assert rendered.startswith("<!doctype html>")
    assert "<title>" in rendered
    # Pygments highlighting fires for Python source
    assert 'class="codehilite"' in rendered


def test_export_html_renders_table_with_thead_and_tbody(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "df = ...\n")
    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "arrow/ipc",
                "rows": 2,
                "columns": ["name", "score"],
                "preview": [
                    {"name": "Alice", "score": 95},
                    {"name": "Bob", "score": 87},
                ],
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir, ExportOptions(output_format="html"))
    assert "<thead>" in rendered
    assert "<th>name</th>" in rendered
    assert "<td>Alice</td>" in rendered


def test_export_html_escapes_user_content(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", 'print("<script>alert(1)</script>")\n')

    rendered = export_notebook(nb_dir, ExportOptions(output_format="html"))
    # Raw script tag must not survive the escape pass
    assert "<script>alert(1)</script>" not in rendered
    assert "&lt;script&gt;" in rendered or "&lt;span" in rendered  # escaped


def test_export_renders_against_existing_example(tmp_path: Path, monkeypatch) -> None:
    """Smoke-test that every shipped example exports cleanly."""
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        pytest.skip("examples directory not present")

    notebooks = [d for d in examples_dir.iterdir() if (d / "notebook.toml").is_file()]
    assert notebooks, "no example notebooks discovered"

    for nb_dir in notebooks:
        rendered = export_notebook(nb_dir)
        assert rendered.strip(), f"export produced empty output for {nb_dir.name}"
        # Every export carries at least one heading — either from the
        # README's h1 or from the fallback "Notebook: <name>" we emit
        # when there's no README.
        assert "\n# " in rendered or rendered.startswith("# "), (
            f"export for {nb_dir.name} has no top-level heading"
        )


def test_export_never_emits_prompt_response_marker_for_real_examples() -> None:
    """For every example with a prompt cell, the privacy note appears
    and no cached response content can leak. The note is a positive
    signal that the prompt-cell privacy branch was taken; if it's
    absent on a notebook containing a prompt cell, the renderer drifted.
    """
    repo_root = Path(__file__).resolve().parents[2]
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        pytest.skip("examples directory not present")

    expected_marker = "response intentionally excluded"

    for nb_dir in examples_dir.iterdir():
        if not (nb_dir / "notebook.toml").is_file():
            continue
        # Look at the toml to see if this example has any prompt cells
        toml_text = (nb_dir / "notebook.toml").read_text(encoding="utf-8")
        if 'language = "prompt"' not in toml_text:
            continue

        rendered_md = export_notebook(nb_dir)
        rendered_html = export_notebook(nb_dir, ExportOptions(output_format="html"))

        assert expected_marker in rendered_md.lower(), (
            f"prompt-cell privacy marker missing in markdown export of {nb_dir.name}"
        )
        assert expected_marker in rendered_html.lower(), (
            f"prompt-cell privacy marker missing in HTML export of {nb_dir.name}"
        )
