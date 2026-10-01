"""Self-rendered SVG badge for a publication's README.

It reports only the size of the recorded chain and is deliberately not green:
in badge convention green means "passing", and nothing here claims a result
was verified. Rendered locally rather than via shields.io so README views do
not leak to a third party and air-gapped servers still work.
"""

from __future__ import annotations

from html import escape

# Approximate advance widths for 11px DejaVu Sans. Exactness does not matter: every run is drawn
# with ``textLength``, so a bad estimate looks loose or tight but never overflows.
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
        # Drawn twice: a dark copy one pixel down for the engraved look, then the real one.
        # `textLength` pins the run to the computed width so an unknown font cannot overflow the
        # pill.
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
    """Return the badge for one publication.

    A withdrawn publication still renders and says so, rather than leaving a
    broken image in the README.
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
