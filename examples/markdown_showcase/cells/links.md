## Inline links

Standard markdown link: [Strata homepage](https://bearing-research.github.io/strata/).

Link with a query string:
[search](https://example.com/search?q=strata&page=2).

Link to a fragment: [strata install](https://bearing-research.github.io/strata/getting-started/installation/#prerequisites).

## Autolinks (CommonMark)

Wrapped in angle brackets: <https://example.com>.

Email autolink: <hello@example.com>.

## Plain URLs (linkify)

A bare URL inside a sentence — like https://example.com/page — should
auto-detect into a clickable link without explicit markdown syntax.
The old hand-rolled parser missed this case.

## Reference links

Reference-style: [Strata][nb] uses an artifact store at its core.

[nb]: https://bearing-research.github.io/strata/

## Link safety

All rendered links should open in a new tab and carry
`rel="noreferrer noopener"`. Right-click → Inspect on any link above
to verify.
