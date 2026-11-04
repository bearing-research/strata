"""Mkdocs hook: keep the URLs of pages that were split, with their anchors.

A static page cannot tell a server about a fragment, so each old URL is a small page whose
script sends ``#heading`` to whichever new page has that heading now.
"""

from __future__ import annotations

import json
import posixpath
from pathlib import Path

# Old URL: (where it goes by default, the pages it became, first match wins).
SPLIT = {
    "notebook/cells": (
        "guides/cells/index.md",
        [
            "guides/cells/index.md",
            "guides/cells/python.md",
            "guides/cells/prompt.md",
            "guides/cells/sql.md",
            "guides/cells/widget.md",
            "guides/share/app-view.md",
            "guides/cells/loops-and-variants.md",
            "guides/cells/r.md",
            "reference/annotations/r.md",
            "concepts/artifacts-and-lineage.md",
        ],
    ),
    "notebook/annotations": (
        "reference/annotations/index.md",
        [
            "reference/annotations/index.md",
            "reference/annotations/python.md",
            "reference/annotations/prompt.md",
            "reference/annotations/loops-and-variants.md",
            "reference/annotations/sql.md",
            "reference/annotations/widget.md",
            "guides/data/mounts.md",
            "guides/data/fetch-and-datasets.md",
            "guides/data/write-back.md",
            "guides/cells/loops-and-variants.md",
        ],
    ),
    "notebook/concepts": (
        "concepts/caching.md",
        [
            "concepts/caching.md",
            "concepts/dag-and-staleness.md",
            "concepts/artifacts-and-lineage.md",
            "concepts/file-format-and-git.md",
        ],
    ),
}
# Headings that changed their text when they became a page's title.
RENAMED = {
    "notebook/cells": {
        "r-cells-020": "guides/cells/r.md",
        "cell-types": "guides/cells/index.md",
    },
    "notebook/annotations": {
        "cell-annotations-environment": "reference/annotations/index.md",
        "name_1": "reference/annotations/sql.md#name",
    },
    "notebook/concepts": {"notebook-concepts": "concepts/caching.md"},
}

_urls: dict[str, str] = {}
_anchors: dict[str, set[str]] = {}


def anchors_to_pages(pages: list[str], anchors: dict[str, set[str]]) -> dict[str, str]:
    """Each anchor of *pages* with the first page that has it."""
    found: dict[str, str] = {}
    for page in pages:
        for anchor in sorted(anchors.get(page, ())):
            found.setdefault(anchor, page)
    return found


def stub_html(default: str, moved: dict[str, str]) -> str:
    """A page that sends its visitor on, with its fragment, to where that heading is now."""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Moved</title>
<link rel="canonical" href="{default}">
<noscript><meta http-equiv="refresh" content="0; url={default}"></noscript>
<script>
const moved = {json.dumps(moved, sort_keys=True)};
const anchor = decodeURIComponent(location.hash.slice(1));
location.replace(moved[anchor] ?? "{default}" + location.hash);
</script>
</head>
<body><p>This page has moved: <a href="{default}">{default}</a></p></body>
</html>
"""


def _toc_ids(toc) -> set[str]:
    ids: set[str] = set()
    for item in toc:
        ids.add(item.id)
        ids |= _toc_ids(item.children)
    return ids


def on_config(config):
    _urls.clear()
    _anchors.clear()
    return config


def on_page_content(html, page, config, files):
    _urls[page.file.src_uri] = page.file.url
    _anchors[page.file.src_uri] = _toc_ids(page.toc)
    return html


def on_post_build(config):
    site = Path(config["site_dir"])
    for old, (default, pages) in SPLIT.items():

        def relative(target: str) -> str:
            page, hash_, fragment = target.partition("#")
            return f"{posixpath.relpath(_urls[page], old)}/{hash_}{fragment}"

        moved = {
            anchor: relative(f"{page}#{anchor}")
            for anchor, page in anchors_to_pages(pages, _anchors).items()
        }
        moved |= {anchor: relative(target) for anchor, target in RENAMED.get(old, {}).items()}
        target = site / old / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(stub_html(relative(default), moved), encoding="utf-8")
