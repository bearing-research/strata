"""Mkdocs hook: publish each page's Markdown, and ``llms.txt`` and ``llms-full.txt`` over them.

A page at ``notebook/cells/`` gets ``notebook/cells/index.md``, its source with relative links
made absolute, since they were written against ``docs/`` and not against the published URL.
``llms.txt`` follows the nav, so a nav change needs nothing here. Registered in ``mkdocs.yml``:

    hooks:
      - docs_hooks/llms_txt.py
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Callable
from pathlib import Path

# A link can be inline, a reference definition, or HTML under md_in_html.
_LINK = re.compile(r'(\]\(|^\[[^\]]+\]:[ \t]*|(?:src|href)=")([^)\s"]+)', re.MULTILINE)
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_INLINE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_SENTENCE_END = re.compile(r"(?<=[.!])\s")
_NOT_PROSE = ("#", "!", "|", "<", "=== ", "- ", "* ", ">", "--8<--", "```", "~~~")

# Long, and useful only to someone looking for a change: listed, but left out of llms-full.txt.
_OPTIONAL = {"changelog.md"}

_pages: dict[str, tuple[str, str]] = {}
_files = None
_nav = None


def absolute_links(markdown: str, source: str, url_of: Callable[[str], str | None]) -> str:
    """Make the links in ``source``'s *markdown* absolute, leaving what ``url_of`` does not know."""

    def replace(match: re.Match[str]) -> str:
        prefix, target = match.groups()
        if target.startswith(("#", "/")) or _SCHEME.match(target):
            return match.group(0)
        path, hash_, fragment = target.partition("#")
        url = url_of(posixpath.normpath(posixpath.join(posixpath.dirname(source), path)))
        return f"{prefix}{url}{hash_}{fragment}" if url else match.group(0)

    return _LINK.sub(replace, markdown)


def description(markdown: str, limit: int = 160) -> str:
    """The first sentence of the page's first paragraph of prose, for its line in ``llms.txt``."""
    in_fence = False
    for block in re.split(r"\n\s*\n", markdown):
        was_fenced = in_fence
        for line in block.splitlines():
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
        text = block.strip()
        if was_fenced or in_fence or not text or text.startswith(_NOT_PROSE) or block[0].isspace():
            continue
        plain = " ".join(_INLINE_LINK.sub(r"\1", text).split()).replace("**", "")
        sentence = _SENTENCE_END.split(plain, maxsplit=1)[0]
        return sentence if len(sentence) <= limit else sentence[: limit - 1].rstrip() + "…"
    return ""


def render_llms_txt(
    name: str,
    summary: str,
    sections: list[tuple[str, list[tuple[str, str, str]]]],
) -> str:
    """The ``llms.txt`` format: a title, a summary, then sections of links."""
    parts = [f"# {name}\n\n> {summary}\n"]
    for heading, links in sections:
        if not links:
            continue
        lines = [
            f"- [{title}]({url})" + (f": {text}" if text else "") for title, url, text in links
        ]
        parts.append(f"## {heading}\n\n" + "\n".join(lines) + "\n")
    return "\n".join(parts)


def _base_url(config) -> str:
    # Under ``mike deploy`` this already names the version being deployed.
    return config["site_url"].rstrip("/") + "/"


def _expand_snippets(markdown: str, config) -> str:
    if "--8<--" not in markdown:
        return markdown
    from markdown import Markdown

    settings = config["mdx_configs"]["pymdownx.snippets"]
    md = Markdown(
        extensions=["pymdownx.snippets"], extension_configs={"pymdownx.snippets": settings}
    )
    return "\n".join(md.preprocessors["snippet"].run(markdown.split("\n")))


def on_config(config):
    _pages.clear()
    return config


def on_files(files, config):
    global _files
    _files = files
    return files


def on_nav(nav, config, files):
    global _nav
    _nav = nav
    return nav


def on_page_markdown(markdown, page, config, files):
    base = _base_url(config)

    def url_of(path: str) -> str | None:
        file = files.get_file_from_path(path)
        return f"{base}{file.url}" if file else None

    expanded = _expand_snippets(markdown, config)
    _pages[page.file.src_uri] = (page.title, absolute_links(expanded, page.file.src_uri, url_of))
    return markdown


def _markdown_url(page, base: str) -> str:
    return posixpath.join(base, posixpath.dirname(page.file.dest_uri), "index.md")


def _flatten(item) -> list:
    return [item] if item.is_page else [page for child in item.children for page in _flatten(child)]


def on_post_build(config):
    site = Path(config["site_dir"])
    base = _base_url(config)
    for file in _files.documentation_pages():
        if file.src_uri in _pages:
            target = site / posixpath.dirname(file.dest_uri) / "index.md"
            target.write_text(_pages[file.src_uri][1], encoding="utf-8")

    def links(pages: list) -> list[tuple[str, str, str]]:
        return [
            (
                _pages[page.file.src_uri][0],
                _markdown_url(page, base),
                page.meta.get("description") or description(_pages[page.file.src_uri][1]),
            )
            for page in pages
        ]

    def core(pages: list) -> list:
        return [page for page in pages if page.file.src_uri not in _OPTIONAL]

    top_pages = core([item for item in _nav.items if item.is_page])
    groups = [(i.title, core(_flatten(i))) for i in _nav.items if not i.is_page]
    in_nav = {page.file.src_uri for item in _nav.items for page in _flatten(item)}
    # A page outside the nav is a rendered example, reached from the examples page.
    optional = [f.page for f in _files.documentation_pages() if f.src_uri in _OPTIONAL]
    optional += [
        f.page
        for f in _files.documentation_pages()
        if f.src_uri in _pages and f.src_uri not in in_nav and f.src_uri not in _OPTIONAL
    ]
    sections = [("Start", links(top_pages))]
    sections += [(title, links(pages)) for title, pages in groups]
    sections.append(("Optional", links(optional)))
    (site / "llms.txt").write_text(
        render_llms_txt(config["site_name"], config["site_description"], sections),
        encoding="utf-8",
    )

    full = [
        f"<!-- {_markdown_url(page, base)} -->\n\n{_pages[page.file.src_uri][1].strip()}\n"
        for page in top_pages + [page for _, pages in groups for page in pages]
    ]
    (site / "llms-full.txt").write_text("\n".join(full), encoding="utf-8")
