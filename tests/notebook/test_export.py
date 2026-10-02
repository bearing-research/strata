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
    """The serializer emits preview rows as positional lists, not dicts: the real on-disk shape."""
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
    """A body with fenced examples (common in prompt cells) needs a fence longer than its
    longest backtick run, or the export breaks.
    """
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    source = '"""Example doc:\n\n```python\nx = 1\n```\n"""\nx = 1\n'
    write_cell(nb_dir, "c1", source)

    rendered = export_notebook(nb_dir)
    # Outer fence is four backticks; inner ```python survives intact.
    assert "````python" in rendered
    assert "```python\nx = 1\n```" in rendered  # inner fence untouched
    assert rendered.rstrip().endswith("````\n") or "\n````" in rendered


def test_image_output_renders_when_inline_data_url_present() -> None:
    """Renderer-only: assumes hydration re-attached ``inline_data_url``, which the writer strips."""
    from strata.notebook.export import _render_display_output
    from strata.notebook.models import CellOutput

    data_url = "data:image/png;base64,iVBORw0KGgoAAAANS"
    output = CellOutput(
        content_type="image/png",
        inline_data_url=data_url,
    )

    blocks = _render_display_output(
        output,
        notebook_dir=Path("."),
        notebook_id="dummy",
        max_bytes=0,
    )
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


def test_export_markdown_cell_renders_without_cell_banner(tmp_path: Path) -> None:
    """A markdown cell's own heading is its section divider; a '## cell-id' banner above it
    would duplicate titles.
    """
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "intro", language="markdown")
    write_cell(
        nb_dir,
        "intro",
        "# Section heading\n\nProse explaining the upcoming code.\n",
    )

    rendered = export_notebook(nb_dir)
    assert "# Section heading" in rendered
    # No banner ahead of the markdown content
    assert "## intro" not in rendered
    # No metadata chips before markdown content
    assert "**kind** markdown" not in rendered


def test_export_sanitizes_active_html_in_markdown_cells(tmp_path: Path) -> None:
    """The UI renders markdown with raw HTML off; the export must match before python-markdown
    passes raw HTML into the published page.
    """
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "danger", language="markdown")
    write_cell(
        nb_dir,
        "danger",
        "## Sanitization\n\n"
        "<script>alert('xss')</script>\n\n"
        '<img src="x" onerror="alert(\'attr\')">\n\n'
        '<iframe src="//evil"></iframe>\n\n'
        "[click](javascript:alert(1))\n\n"
        "[doc](data:text/html;base64,PHNjcmlwdD4=)\n",
    )

    rendered = export_notebook(nb_dir)

    assert "## Sanitization" in rendered
    assert "<script>" not in rendered
    assert "&lt;script>" in rendered
    assert "<iframe" not in rendered
    assert "<img " not in rendered
    assert "&lt;img" in rendered
    assert "javascript:" not in rendered
    assert "data:text/html" not in rendered
    assert "[click](#)" in rendered
    assert "[doc](#)" in rendered


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("<img/src=x/onerror=alert(1)>", "&lt;img/src=x/onerror=alert(1)>"),
        ('<a href="javascript:alert(1)">x</a>', '&lt;a href="javascript:alert(1)">x&lt;/a>'),
        ("<form action=javascript:alert(1)><button>go</button></form>", None),
        ('<svg><a xlink:href="javascript:alert(1)"><text>x</text></a></svg>', None),
        ("<javascript:alert(1)>", "&lt;javascript:alert(1)>"),
        ("[x]( javascript:alert(1))", "[x]( #)"),
        ("[x](javascript:alert(1))", "[x](#)"),
        ("[x](<javascript:alert(1)>)", "[x](#)"),
        ("[x](jav&#x61;script:alert(1))", "[x](#)"),
        ("[x](JavaScript\\:alert(1))", "[x](#)"),
        ("[x](vbscript:msgbox(1))", "[x](#)"),
        ("[x](data:text/html,<script>alert(1)</script>)", "[x](#)"),
        ("[r]: javascript:alert(1)\n\n[go][r]", "[r]: #\n\n[go][r]"),
    ],
)
def test_sanitizer_neutralizes_bypass_payloads(payload: str, expected: str | None) -> None:
    from strata.notebook.export import _sanitize_markdown_body

    sanitized = _sanitize_markdown_body(payload)

    assert "<" not in sanitized.replace("&lt;", "")
    if expected is not None:
        assert sanitized == expected


def test_sanitizer_keeps_code_safe_links_and_markdown(tmp_path: Path) -> None:
    """Escaping is limited to raw HTML: code keeps its ``<``, safe links and quotes survive."""
    from strata.notebook.export import _sanitize_markdown_body

    body = (
        "> quote with a < b\n\n"
        "`<b>inline</b>` and <b>bold</b>\n\n"
        "[ok](https://example.com/a_(b)) <https://example.com> <mailto:a@b.c> <a@b.co>\n\n"
        "![img](data:image/png;base64,AAAA)\n\n"
        "```html\n<script>shown as code</script>\n```\n"
    )

    assert _sanitize_markdown_body(body) == (
        "> quote with a &lt; b\n\n"
        "`<b>inline</b>` and &lt;b>bold&lt;/b>\n\n"
        "[ok](https://example.com/a_(b)) <https://example.com> <mailto:a@b.c> <a@b.co>\n\n"
        "![img](data:image/png;base64,AAAA)\n\n"
        "```html\n<script>shown as code</script>\n```\n"
    )


def test_export_sanitizes_active_html_in_readme(tmp_path: Path) -> None:
    """The README intro takes the same MarkdownBlock path; unsanitized, an untrusted notebook's
    README is an XSS vector against viewers of the export.
    """
    nb_dir = _make_notebook(tmp_path)
    (nb_dir / "README.md").write_text(
        "# Demo\n\n<script>alert('readme')</script>\n",
        encoding="utf-8",
    )
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")

    rendered = export_notebook(nb_dir)
    assert "<script>" not in rendered
    assert "&lt;script>alert('readme')&lt;/script>" in rendered


def test_export_escapes_benign_inline_html_like_the_ui(tmp_path: Path) -> None:
    """The UI shows raw HTML as text, so ``<sub>`` and ``<details>`` export as text too."""
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "ok", language="markdown")
    write_cell(
        nb_dir,
        "ok",
        "## Notes\n\nWater is H<sub>2</sub>O.\n\n"
        "<details><summary>more</summary>hidden text</details>\n",
    )

    rendered = export_notebook(nb_dir)
    assert "H&lt;sub>2&lt;/sub>O" in rendered
    assert "<details>" not in rendered


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
    # No duplicate H1; the README already owns the page title.
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


def test_export_truncates_console_over_byte_cap(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "print('x')\n")
    long_stdout = "noise\n" * 5000  # ~30 KB, comfortably above 1 KB cap
    update_cell_console_output(nb_dir, "c1", long_stdout, "")

    rendered = export_notebook(nb_dir, ExportOptions(max_output_bytes=1024))
    assert "more bytes truncated" in rendered
    assert "noise" in rendered


def test_export_truncation_disabled_when_max_bytes_zero(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "print('x')\n")
    long_stdout = "noise\n" * 5000
    update_cell_console_output(nb_dir, "c1", long_stdout, "")

    rendered = export_notebook(nb_dir, ExportOptions(max_output_bytes=0))
    assert "more bytes truncated" not in rendered


def test_export_swaps_oversized_image_for_size_note(tmp_path: Path) -> None:
    """An inline image over the cap becomes a note, not a silent drop.

    Renderer-level because the writer strips a 1 MB data URL as transient.
    """
    from strata.notebook.export import _render_display_output
    from strata.notebook.models import CellOutput

    # 200 KB data URL
    huge = "data:image/png;base64," + ("A" * 200_000)
    output = CellOutput(content_type="image/png", inline_data_url=huge)

    blocks = _render_display_output(
        output,
        notebook_dir=Path("."),
        notebook_id="dummy",
        max_bytes=100_000,  # below the 200 KB image
    )

    # No ImageBlock; one NoteBlock explaining the size issue.
    from strata.notebook.export import ImageBlock, NoteBlock

    assert all(not isinstance(b, ImageBlock) for b in blocks)
    note = next(b for b in blocks if isinstance(b, NoteBlock))
    assert "too large to inline" in note.text


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
        # Every export has a heading: the README's h1, or the fallback
        # "Notebook: <name>" when there is no README.
        assert "\n# " in rendered or rendered.startswith("# "), (
            f"export for {nb_dir.name} has no top-level heading"
        )


def test_strata_export_cli_writes_markdown_to_stdout(tmp_path: Path) -> None:
    """End-to-end: shell out to `strata export <dir>` and verify stdout."""
    import subprocess

    nb_dir = _make_notebook(tmp_path, name="cli_smoke")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")

    result = subprocess.run(
        ["strata", "export", str(nb_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "## c1" in result.stdout
    assert "```python" in result.stdout
    assert "x = 1" in result.stdout


def test_strata_export_cli_writes_html_to_out_path(tmp_path: Path) -> None:
    """--to html --out <path>: produces a standalone HTML file."""
    import subprocess

    nb_dir = _make_notebook(tmp_path, name="cli_html")
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "x = 1\n")

    out_path = tmp_path / "out.html"
    result = subprocess.run(
        [
            "strata",
            "export",
            str(nb_dir),
            "--to",
            "html",
            "--out",
            str(out_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""  # nothing on stdout when --out is set
    body = out_path.read_text(encoding="utf-8")
    assert body.startswith("<!doctype html>")
    assert 'class="codehilite"' in body


def test_strata_export_cli_returns_2_on_bad_path(tmp_path: Path) -> None:
    """Non-notebook path produces a clean error + non-zero exit."""
    import subprocess

    result = subprocess.run(
        ["strata", "export", str(tmp_path / "does-not-exist")],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 2
    assert "not a notebook directory" in result.stderr.lower()


def test_strata_export_cli_include_inactive_variants_flag(tmp_path: Path) -> None:
    """--include-inactive-variants stacks all variants in the output."""
    import subprocess

    nb_dir = _make_notebook(tmp_path, name="cli_variants")
    add_cell_to_notebook(nb_dir, "model_a")
    write_cell(nb_dir, "model_a", "# @variant model gpt4\npreds = 1\n")
    add_cell_to_notebook(nb_dir, "model_b", after_cell_id="model_a")
    write_cell(nb_dir, "model_b", "# @variant model claude\npreds = 2\n")
    set_variant_active(nb_dir, "model", "gpt4")

    # Default: only the active variant
    result = subprocess.run(
        ["strata", "export", str(nb_dir)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "## model_a" in result.stdout
    assert "## model_b" not in result.stdout

    # With flag: both
    result = subprocess.run(
        ["strata", "export", str(nb_dir), "--include-inactive-variants"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "## model_a" in result.stdout
    assert "## model_b" in result.stdout


def test_export_never_emits_prompt_response_marker_for_real_examples() -> None:
    """For every example with a prompt cell, the privacy note appears and no cached response
    leaks; a missing note means the renderer drifted.
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


# App-view snapshot profile (ExportOptions.app_view)


def _app_opts(fmt: str = "markdown") -> ExportOptions:
    from strata.notebook.export import ExportFormat

    return ExportOptions(app_view=True, output_format=ExportFormat(fmt))


def test_app_view_snapshot_shows_outputs_not_source(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "c1")
    write_cell(nb_dir, "c1", "# @name Scores\nsecret_source_marker = 1\n")
    update_cell_display_outputs(
        nb_dir,
        "c1",
        [
            {
                "content_type": "arrow/ipc",
                "rows": 1,
                "columns": ["name", "score"],
                "preview": [{"name": "Alice", "score": 95}],
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir, _app_opts())

    assert "| Alice | 95 |" in rendered  # the output is shown
    assert "secret_source_marker" not in rendered  # the source is not
    assert "```python" not in rendered  # no code fences at all


def test_app_view_snapshot_renders_widget_control_values(tmp_path: Path) -> None:
    from strata.notebook.runtime_state import persist_cell_widget_values

    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "controls", language="widget")
    write_cell(
        nb_dir,
        "controls",
        "alpha = slider(0, 1, default=0.5)\ncurve = dropdown(['linear', 'sqrt'])\n",
    )
    persist_cell_widget_values(nb_dir, "controls", {"alpha": 0.7})

    rendered = export_notebook(nb_dir, _app_opts())

    # Persisted value wins; unset control falls back to its default.
    assert "alpha" in rendered and "0.7" in rendered
    assert "curve" in rendered and "linear" in rendered


def test_app_view_snapshot_hides_app_hidden_cell(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "setup")
    write_cell(nb_dir, "setup", "# @app hide\nx = 1\n")
    update_cell_display_outputs(
        nb_dir,
        "setup",
        [
            {
                "content_type": "arrow/ipc",
                "rows": 1,
                "columns": ["hidden"],
                "preview": [{"hidden": "should_not_appear"}],
                "bytes": 0,
            }
        ],
    )

    rendered = export_notebook(nb_dir, _app_opts())

    assert "should_not_appear" not in rendered


def test_app_view_snapshot_excludes_prompt_cells(tmp_path: Path) -> None:
    nb_dir = _make_notebook(tmp_path)
    add_cell_to_notebook(nb_dir, "p1", language="prompt")
    write_cell(nb_dir, "p1", "Summarize {{ df }} in one sentence.\n")
    update_cell_display_outputs(
        nb_dir,
        "p1",
        [{"content_type": "text/markdown", "markdown": "a leaked response", "bytes": 0}],
    )

    rendered = export_notebook(nb_dir, _app_opts())

    assert "leaked response" not in rendered
    assert "Summarize" not in rendered
