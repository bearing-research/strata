"""Text helpers."""

import re


def slugify(title: str) -> str:
    """A URL slug: lower case, words joined by single hyphens."""
    return re.sub(r"[^a-z0-9]", "-", title.lower()).strip("-")
