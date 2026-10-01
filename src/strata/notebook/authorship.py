"""Who changed a cell.

In service mode the authenticated principal is the author; a declared author is
ignored, since a client that could name itself would be claiming an identity
rather than presenting one. In personal mode there is no authentication, so the
client's declared name is recorded (the browser is ``local``; an agent sends its
own name on each tool call). That is a claim, not a fact.
"""

from __future__ import annotations

# A personal server is one person, so what matters is human versus agent, not which human.
LOCAL_AUTHOR = "local"

# Bounded: written into committed config from an unauthenticated claim.
MAX_AUTHOR_LENGTH = 128


def resolve_author(declared: str | None = None) -> str:
    """Who to record for the edit being made now.

    The authenticated principal if any; otherwise the declared author, or ``local``.
    """
    principal = _current_principal()
    if principal is not None:
        return principal
    return clean_author(declared) or LOCAL_AUTHOR


def clean_author(declared: str | None) -> str | None:
    """A declared author, trimmed and bounded, or ``None`` if it was empty.

    Control characters are dropped rather than escaped: the value lands in TOML
    and in a cell view, where a newline in a byline is never intended.
    """
    if not declared:
        return None
    cleaned = "".join(ch for ch in str(declared) if ch.isprintable()).strip()
    return cleaned[:MAX_AUTHOR_LENGTH] or None


def _current_principal() -> str | None:
    """The authenticated caller's id, when the request carried one."""
    try:
        from strata.auth import get_principal
    except ImportError:
        return None
    principal = get_principal()
    return principal.id if principal is not None else None
