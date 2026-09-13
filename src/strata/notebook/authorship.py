"""Who changed a cell.

A notebook has an owner; a cell had nobody. Runs are attributed through the
artifact's ``principal``, but only for remote builds and team-store offers, so
an edit left no record beyond git — and on a server where an agent and a person
both write cells, "which of these did the agent write" had no answer at all.

Two sources, because the two deployments answer the question differently. In
service mode the server authenticates every request and the principal *is* the
answer; a client that could name itself there would be claiming an identity
rather than presenting one, so a declared author is ignored. In personal mode
there is no authentication and nothing to check against, so the client says who
it is: the browser is ``local``, the built-in assistant is ``assistant``, and an
external agent sends its own name on each tool call. That is a claim rather than
a fact, which is the honest amount of trust available on a machine where anyone
who can reach the server is already its owner.

The built-in assistant is the one exception to "the principal wins", and
deliberately: it runs inside an authenticated session, so taking the principal
would record every assistant-written cell as the person's own and erase the
distinction this field exists to draw. It records both — see
:func:`resolve_assistant_author`.
"""

from __future__ import annotations

# What the browser says when it says nothing. A personal server is one person,
# so the useful distinction there is not *which* human but human-versus-agent.
LOCAL_AUTHOR = "local"

# What the built-in assistant writes. Not a name — one server's assistant is
# every user's assistant — but the human/agent distinction is the one a person
# opening the notebook actually wants.
ASSISTANT_AUTHOR = "assistant"

# Bounded because it is written into committed config from an unauthenticated
# claim. Long enough for `agent:<id>/<sub>`, short enough not to be a payload.
MAX_AUTHOR_LENGTH = 128


def resolve_author(declared: str | None = None) -> str:
    """Who to record for the edit being made now.

    The authenticated principal wherever there is one; otherwise what the
    client declared, and ``local`` when it declared nothing.
    """
    principal = _current_principal()
    if principal is not None:
        return principal
    return clean_author(declared) or LOCAL_AUTHOR


def resolve_assistant_author() -> str:
    """Who to record when the built-in assistant writes a cell.

    Neither answer alone is right. Recording the principal loses the human /
    agent distinction, which is the question the field exists to answer — every
    cell in a session would read as the person's own. Recording only
    ``assistant`` loses whose session produced it, which on a shared server is
    the other half of an audit.

    So both, in the scheme-prefixed shape the agent identities already use:
    ``assistant`` alone in personal mode, ``assistant:<principal>`` where there
    is one.
    """
    principal = _current_principal()
    return f"{ASSISTANT_AUTHOR}:{principal}" if principal else ASSISTANT_AUTHOR


def clean_author(declared: str | None) -> str | None:
    """A declared author, trimmed and bounded, or ``None`` if it was empty.

    Control characters are dropped rather than escaped: this ends up in TOML
    and in a cell view, and a newline in the middle of a byline is never
    something a caller meant.
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
