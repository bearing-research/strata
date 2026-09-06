"""The little SVG pill a README can carry.

A shields-style badge has a grammar — ``label | status``, green for good — and
that grammar is an assertion. This feature has been careful not to claim a
result was verified, so the badge has to resist its own shape: it reports the
size of the recorded chain and nothing else, and it is not green, because in
badge convention green means "passing" and borrowing it would smuggle back the
reading everything else here refuses.

Self-rendered rather than delegated to shields.io. A remote badge tells a third
party which artifact each README view is looking at, and it fails entirely on
the internal and air-gapped servers a lab is most likely to run.
"""

from __future__ import annotations

from html import escape

# Approximate advance widths for 11px DejaVu Sans, which is what renders in
# practice. Exactness does not matter because every run is drawn with
# ``textLength``: the estimate sets the pill's width, and the text is then made
# to fit it. A bad estimate looks slightly loose or tight, never overflowing.
_NARROW = set("iljtfrI.,:;'|!()[]{} ")
_WIDE = set("mwMW@")


def _text_width(text: str) -> float:
    width = 0.0
    for char in text:
        if char in _NARROW:
            width += 3.6
        elif char in _WIDE:
            width += 9.6
        elif char.isupper() or char.isdigit():
            width += 7.2
        else:
            width += 6.3
    return width


_LABEL_BG = "#4a5262"
# Slate, deliberately. Green is the badge convention for "passing", and a
# provenance record is not a pass.
_VALUE_BG = "#2f6f9f"
_WITHDRAWN_BG = "#6b6b6b"

_PAD = 9.0


def render_badge(*, label: str, value: str, title: str) -> str:
    """One pill: ``label`` on the left, ``value`` on the right."""
    label_w = _text_width(label) + _PAD * 2
    value_w = _text_width(value) + _PAD * 2
    total = label_w + value_w
    value_bg = _WITHDRAWN_BG if value == "withdrawn" else _VALUE_BG

    def run(text: str, x: float, width: float) -> str:
        # Drawn twice: a dark copy one pixel down for the engraved look every
        # badge has, then the real one. `textLength` pins the run to the width
        # the layout was computed from, so a font this server cannot know about
        # cannot push text past the edge of its own pill.
        content = escape(text)
        length = width - _PAD * 2
        return (
            f'<text x="{x + width / 2:.1f}" y="15" fill="#010101" fill-opacity=".3" '
            f'textLength="{length:.1f}" lengthAdjust="spacingAndGlyphs">{content}</text>'
            f'<text x="{x + width / 2:.1f}" y="14" textLength="{length:.1f}" '
            f'lengthAdjust="spacingAndGlyphs">{content}</text>'
        )

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total:.0f}" height="20" '
        f'role="img" aria-label="{escape(title)}">'
        f"<title>{escape(title)}</title>"
        '<linearGradient id="s" x2="0" y2="100%">'
        '<stop offset="0" stop-color="#bbb" stop-opacity=".1"/>'
        '<stop offset="1" stop-opacity=".1"/></linearGradient>'
        f'<clipPath id="r"><rect width="{total:.0f}" height="20" rx="3" fill="#fff"/></clipPath>'
        '<g clip-path="url(#r)">'
        f'<rect width="{label_w:.0f}" height="20" fill="{_LABEL_BG}"/>'
        f'<rect x="{label_w:.0f}" width="{value_w:.0f}" height="20" fill="{value_bg}"/>'
        f'<rect width="{total:.0f}" height="20" fill="url(#s)"/></g>'
        '<g fill="#fff" text-anchor="middle" font-size="11" '
        'font-family="Verdana,DejaVu Sans,Geneva,sans-serif">'
        f"{run(label, 0, label_w)}{run(value, label_w, value_w)}"
        "</g></svg>"
    )


def badge_for(*, publication, step_count: int) -> str:
    """The badge for one publication.

    A withdrawn publication still renders, and says so. The alternative is a
    broken image in whatever README carries it, which tells a reader nothing
    except that something is wrong with the server.
    """
    if not publication.is_active:
        return render_badge(
            label="provenance",
            value="withdrawn",
            title="This artifact's publication was withdrawn",
        )
    value = f"{step_count} step{'s' if step_count != 1 else ''}"
    return render_badge(
        label="provenance",
        value=value,
        title=f"Provenance recorded: {value} behind this result",
    )
