"""The docs hook behind ``llms.txt``. docs.yml builds the real site and checks the files exist."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "llms_txt", Path(__file__).parents[1] / "docs_hooks" / "llms_txt.py"
)
assert _spec and _spec.loader
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

_URLS = {
    "notebook/cells.md": "https://x/notebook/cells/",
    "assets/a.png": "https://x/assets/a.png",
}


def _absolute(markdown: str, source: str = "notebook/agent.md") -> str:
    return hook.absolute_links(markdown, source, _URLS.get)


def test_a_relative_link_is_resolved_from_the_page_that_wrote_it():
    assert _absolute("[c](cells.md)") == "[c](https://x/notebook/cells/)"
    assert _absolute("[c](notebook/cells.md)", "index.md") == "[c](https://x/notebook/cells/)"
    assert _absolute("![a](../assets/a.png)") == "![a](https://x/assets/a.png)"


def test_a_fragment_stays_on_the_link():
    assert _absolute("[c](cells.md#kinds)") == "[c](https://x/notebook/cells/#kinds)"


def test_reference_links_and_html_are_resolved_too():
    assert _absolute("[c]: cells.md") == "[c]: https://x/notebook/cells/"
    assert _absolute('<img src="../assets/a.png">') == '<img src="https://x/assets/a.png">'


def test_a_link_that_is_not_a_docs_file_is_left_alone():
    for link in (
        "[a](https://example.com/cells.md)",
        "[a](#kinds)",
        "[a](/cells.md)",
        "[a](mailto:x@example.com)",
        "[a](missing.md)",
    ):
        assert _absolute(link) == link


def test_a_description_is_the_first_sentence_of_prose():
    markdown = "# Title\n\n**Strata** runs [cells](cells.md). It caches them.\n\nMore."
    assert hook.description(markdown) == "Strata runs cells."


def test_a_description_skips_what_is_not_prose():
    markdown = """# Title

!!! note
    An aside.

| a |
| - |

- a list

```bash
one

two
```

    indented

The prose."""
    assert hook.description(markdown) == "The prose."


def test_a_question_mark_does_not_end_a_description():
    assert hook.description("Press ? anywhere to see the shortcuts.") == (
        "Press ? anywhere to see the shortcuts."
    )


def test_a_long_description_is_cut():
    text = hook.description("word " * 100, limit=20)
    assert len(text) <= 20
    assert text.endswith("…")


def test_a_page_with_no_prose_has_no_description():
    assert hook.description("# Title\n\n- a list\n") == ""


def test_llms_txt_format():
    text = hook.render_llms_txt(
        "Strata",
        "A summary.",
        [
            ("Empty", []),
            ("Guides", [("Cells", "https://x/cells/index.md", "All of them."), ("B", "u", "")]),
        ],
    )
    assert text == (
        "# Strata\n\n> A summary.\n\n"
        "## Guides\n\n- [Cells](https://x/cells/index.md): All of them.\n- [B](u)\n"
    )


def test_links_name_the_deployed_version(monkeypatch):
    config = {"site_url": "https://x/strata/"}
    monkeypatch.delenv("MIKE_DOCS_VERSION", raising=False)
    assert hook._base_url(config) == "https://x/strata/"
    monkeypatch.setenv("MIKE_DOCS_VERSION", "0.12.0")
    assert hook._base_url(config) == "https://x/strata/0.12.0/"
