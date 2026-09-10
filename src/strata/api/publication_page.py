"""Server-rendered page for a published artifact.

Self-contained HTML with no external requests: a page meant to outlive the
work it documents should not depend on a CDN still being there, and a referee
opening a link from a paper should not be reporting requests to third parties
to read it.

Everything interpolated here is attacker-controlled in the sense that matters —
cell source, titles, variable names and table URIs are all written by whoever
used the notebook, and this page is served unauthenticated. Every value goes
through :func:`html.escape`; there is no path that writes a caller-supplied
string into the document unescaped.
"""

from __future__ import annotations

import datetime
from html import escape

_STYLE = """
:root { color-scheme: light dark; --fg:#12151a; --muted:#5b6472; --bg:#fbfbfd;
        --card:#fff; --line:#e3e6ec; --code:#f5f6f9; --accent:#2a5db0; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e8ee; --muted:#9aa3b2; --bg:#14171c; --card:#1b1f26;
          --line:#2a2f38; --code:#11141a; --accent:#8ab4ff; }
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
       font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width: 52rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
h1 { font-size: 1.5rem; margin: 0 0 .35rem; }
h2 { font-size: 1rem; margin: 2.25rem 0 .75rem; letter-spacing:.02em;
     text-transform: uppercase; color: var(--muted); }
.sub { color: var(--muted); margin: 0 0 0.5rem; }
.sub .affil { color: var(--muted); font-size: 0.9em; }
.cite { color: var(--muted); margin: 0 0 2rem; font-size: 0.95em; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:1rem 1.15rem; margin-bottom:1rem; }
pre { background:var(--code); border:1px solid var(--line); border-radius:8px;
      padding:.85rem 1rem; overflow-x:auto; margin:0;
      font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace; }
table { width:100%; border-collapse:collapse; }
td { padding:.3rem 0; vertical-align:top; }
td.k { color:var(--muted); width:11rem; white-space:nowrap; padding-right:1rem; }
code { font:13px ui-monospace,SFMono-Regular,Menlo,monospace; word-break:break-all; }
figure { margin:0; }
figure img { max-width:100%; border:1px solid var(--line); border-radius:8px; }
.step { border-left:2px solid var(--line); padding-left:1rem; margin-left:.4rem; }
.step h3 { font-size:.95rem; margin:0 0 .4rem; font-weight:600; }
.note { font-size:13px; color:var(--muted); }
.banner { border:1px solid var(--line); border-left:3px solid var(--accent);
          background:var(--card); border-radius:8px; padding:.85rem 1rem;
          margin-bottom:1.75rem; font-size:13.5px; }
a { color:var(--accent); }
.snippet { margin:0 0 .9rem; }
.snippet:last-child { margin-bottom:0; }
.snippet p { margin:0 0 .3rem; font-size:13px; color:var(--muted); }
"""


_ID_LINKS = {
    "doi": "https://doi.org/{}",
    "arxiv": "https://arxiv.org/abs/{}",
    "zenodo": "https://zenodo.org/record/{}",
    "url": "{}",
}


def _byline(publication) -> str:
    """ " by A, B and C", or the grant-maker when no authors were declared.

    Author order is meaningful, so it is preserved rather than sorted, and an
    ORCID becomes a link because that is the only form in which a name on a
    page disambiguates one researcher from another.
    """
    if not publication.authors:
        return f" by {escape(publication.published_by)}" if publication.published_by else ""

    rendered: list[str] = []
    for author in publication.authors:
        name = escape(author["name"])
        orcid = author.get("orcid")
        if orcid:
            url = f"https://orcid.org/{escape(orcid, quote=True)}"
            name = f"<a href='{url}' rel='noopener'>{name}</a>"
        affiliation = author.get("affiliation")
        if affiliation:
            name += f" <span class='affil'>({escape(affiliation)})</span>"
        rendered.append(name)

    if len(rendered) == 1:
        return f" by {rendered[0]}"
    return f" by {', '.join(rendered[:-1])} and {rendered[-1]}"


def _citation_line(publication) -> str:
    """The identifiers a reader cites this by, as links they can follow.

    Empty when there are none: a citation line saying nothing is worse than no
    citation line, because a reader reads it as "there is no DOI for this" when
    the truth is that nobody has recorded one yet.
    """
    if not publication.external_ids:
        return ""
    links = []
    for entry in publication.external_ids:
        value = entry["value"]
        template = _ID_LINKS.get(entry["scheme"], "{}")
        url = value if value.startswith("http") else template.format(value)
        label = value if entry["scheme"] == "url" else f"{entry['scheme'].upper()} {value}"
        links.append(f"<a href='{escape(url, quote=True)}' rel='noopener'>{escape(label)}</a>")
    return f"<p class='cite'>Cite as {' · '.join(links)}</p>"


def _when(ts: float | None) -> str:
    if not ts:
        return "not recorded"
    return datetime.datetime.fromtimestamp(ts, datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")


def _rows(pairs: list[tuple[str, str]]) -> str:
    body = "".join(f"<tr><td class='k'>{escape(k)}</td><td>{v}</td></tr>" for k, v in pairs if v)
    return f"<table>{body}</table>"


def _code(value: str) -> str:
    return f"<code>{escape(value)}</code>"


def _source_block(source: str) -> str:
    if not source:
        return (
            "<p class='note'>No source recorded. Artifacts stored before the "
            "source was captured, tables, and core transforms have none.</p>"
        )
    return f"<pre>{escape(source)}</pre>"


def _artifact_label(node) -> str:
    if node.type != "artifact":
        return escape(node.uri)
    return escape(f"{node.artifact_id}@v={node.version}")


def render_publication(
    *,
    publication,
    artifact,
    lineage,
    content_type: str,
    image_src: str | None,
    bundle_filename: str | None = None,
    oembed_url: str | None = None,
    json_ld: str | None = None,
    share: list[tuple[str, str]] | None = None,
) -> str:
    """Render the page for one published artifact.

    ``bundle_filename`` switches it from *hosted* to *archival*: the same
    document, but pointing at a file sitting next to it rather than at routes
    on a server. That is the only difference between the two, and keeping it to
    one branch is deliberate — a bundle that drifted from the live page would
    make the archived copy a second, less trustworthy account of the same
    result.
    """
    title = publication.title or f"{artifact.id}@v={artifact.version}"

    if publication.revoked_at is not None:
        return _document(
            title="Withdrawn",
            body=(
                "<h1>This artifact has been withdrawn</h1>"
                f"<p class='sub'>Published {escape(_when(publication.published_at))}, "
                f"withdrawn {escape(_when(publication.revoked_at))}.</p>"
                "<div class='banner'>The link is intact and still names the same "
                "artifact — it was never repointed at other content. Whoever "
                "published it has withdrawn public access.</div>"
            ),
        )

    root = next((n for n in lineage.nodes if n.artifact_id == artifact.id), None)
    ancestors = [n for n in lineage.nodes if n is not root]

    parts: list[str] = [
        f"<h1>{escape(title)}</h1>",
        f"<p class='sub'>{'Archived' if bundle_filename else 'Published'} "
        f"{escape(_when(publication.published_at))}" + _byline(publication) + ".</p>",
        _citation_line(publication),
        "<div class='banner'><strong>What this page shows.</strong> The code, "
        "inputs and environment recorded when these bytes were produced, and "
        "the chain of steps behind them. It does <em>not</em> claim the result "
        "was reproduced — that needs a re-run, and randomness, thread counts, "
        "floating-point order and unavailable input data each break it. The "
        "integrity digest below shows the bytes have not changed since "
        + ("this bundle was made" if bundle_filename else "publication")
        + "; it cannot show they were honestly produced.</div>",
    ]

    if image_src:
        parts.append(
            "<h2>Result</h2><figure class='card'>"
            f"<img alt='The result' src='{escape(image_src)}'></figure>"
        )

    parts.append("<h2>This artifact</h2><div class='card'>")
    parts.append(
        _rows(
            [
                ("Artifact", _code(f"{artifact.id}@v={artifact.version}")),
                ("Content type", escape(content_type) if content_type else ""),
                ("Produced", escape(_when(artifact.created_at))),
                (
                    # "Computed by", not "Author": the page header already
                    # names whoever published it, and the two are different
                    # facts. Sitting one under the other as "Published by F.
                    # Li" and "Author: not recorded" read as a contradiction
                    # rather than as the distinction it is.
                    "Computed by",
                    escape(artifact.principal)
                    if artifact.principal
                    else "<span class='note'>not recorded — a local run has "
                    "no authenticated identity</span>",
                ),
                (
                    "Environment",
                    escape(root.build_env) if root and root.build_env else "",
                ),
                ("Provenance hash", _code(artifact.provenance_hash)),
                (
                    "Content digest (SHA-256)",
                    _code(publication.content_sha256) if publication.content_sha256 else "",
                ),
                (
                    "Size",
                    f"{artifact.byte_size:,} bytes" if artifact.byte_size else "",
                ),
                (
                    "Rows",
                    f"{artifact.row_count:,}" if artifact.row_count else "",
                ),
            ]
        )
    )
    parts.append("</div>")

    parts.append("<h2>The code that produced it</h2><div class='card'>")
    parts.append(_source_block(root.source if root else ""))
    parts.append("</div>")

    parts.append("<h2>What it was built from</h2>")
    if not ancestors:
        parts.append(
            "<div class='card'><p class='note'>No recorded inputs. This step "
            "read nothing from another step in the same store.</p></div>"
        )
    # A cell that produces several consumed variables contributes one ancestor
    # per variable, each carrying that cell's source. Printed straight, a cell
    # defining five variables repeats its code five times, which reads as a
    # rendering fault rather than as five artifacts from one step.
    shown_sources: dict[str, str] = {}
    for node in ancestors:
        parts.append("<div class='card step'>")
        parts.append(f"<h3>{_artifact_label(node)}</h3>")
        parts.append(
            _rows(
                [
                    ("Kind", escape(node.type)),
                    ("Produced", escape(_when(node.created_at))),
                    ("Author", escape(node.principal) if node.principal else ""),
                    ("Environment", escape(node.build_env)),
                    ("Environment hash", _code(node.env_hash) if node.env_hash else ""),
                ]
            )
        )
        if node.type == "artifact":
            first = shown_sources.get(node.source) if node.source else None
            if first is not None:
                parts.append(
                    "<p class='note'>Produced by the same cell as "
                    f"<code>{escape(first)}</code>, shown above.</p>"
                )
            else:
                if node.source:
                    shown_sources[node.source] = f"{node.artifact_id}@v={node.version}"
                parts.append(_source_block(node.source))
        parts.append("</div>")

    if share is not None:
        parts.append("<h2>Putting it somewhere</h2><div class='card'>")
        for caption, snippet in share:
            parts.append(
                f"<div class='snippet'><p>{escape(caption)}</p><pre>{escape(snippet)}</pre></div>"
            )
        parts.append("</div>")

    parts.append("<h2>Checking it yourself</h2><div class='card'>")
    if bundle_filename is not None:
        # A bundle has no server behind it, so the check is one the reader runs
        # themselves. Naming the command matters more than it looks: an
        # archived page that says "verified" and offers no way to test the
        # claim is asking to be taken on faith, which is the opposite of why
        # the bundle exists.
        parts.append(
            f"<p>The bytes are the file <code>{escape(bundle_filename)}</code> "
            "beside this page. Check it against the digest recorded when this "
            "bundle was made:</p>"
            f"<pre>sha256sum {escape(bundle_filename)}\n"
            f"# {escape(publication.content_sha256 or 'no digest recorded')}</pre>"
        )
    else:
        parts.append(
            f"<p>The bytes are at <a href='/p/{escape(publication.token)}/data'>"
            f"/p/{escape(publication.token)}/data</a>. "
            f"<a href='/p/{escape(publication.token)}/verify'>Verify</a> re-reads "
            "them and compares against the digest recorded at publication.</p>"
        )
    parts.append(
        "<p class='note'>Re-running the computation is a separate matter, and "
        "one only you can do: the source and environment above are what it "
        "would take.</p></div>"
    )

    return _document(title=title, body="".join(parts), oembed_url=oembed_url, json_ld=json_ld)


def content_type_of(artifact) -> str:
    """The stored ``content_type`` param, or '' when the spec says nothing.

    Shared by the hosted page and the archival bundle. Two copies decided the
    bundle's filename and the page's "Content type" row independently, which is
    the same drift ``build_record`` exists to prevent.
    """
    import json

    if not artifact.transform_spec:
        return ""
    try:
        params = json.loads(artifact.transform_spec).get("params", {})
    except (json.JSONDecodeError, ValueError):
        return ""
    return str(params.get("content_type") or "") if isinstance(params, dict) else ""


CLAIMS = {
    "transparency": "Source, inputs and environment as recorded at execution.",
    "integrity": (
        "content_sha256 is the digest of the bytes at publication. It shows "
        "they have not changed since; it does not show they were honestly "
        "produced."
    ),
    "reproduction": "Not claimed. Re-running is left to the reader.",
}


_EMBED_STYLE = """
:root { color-scheme: light dark; --fg:#12151a; --muted:#5b6472; --card:#fff;
        --line:#e3e6ec; --accent:#2a5db0; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6e8ee; --muted:#9aa3b2; --card:#1b1f26; --line:#2a2f38;
          --accent:#8ab4ff; }
}
* { box-sizing: border-box; }
body { margin:0; background:transparent; color:var(--fg);
       font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
a.card { display:block; text-decoration:none; color:inherit; background:var(--card);
         border:1px solid var(--line); border-radius:10px; overflow:hidden; }
a.card:hover { border-color:var(--accent); }
.fig { display:block; width:100%; max-height:300px; object-fit:contain;
       background:#fff; }
.body { padding:.7rem .85rem; }
.title { font-weight:600; font-size:14px; margin:0 0 .2rem; }
.meta { color:var(--muted); margin:0; }
.more { color:var(--accent); margin:.35rem 0 0; }
"""


def render_embed(*, publication, artifact, lineage, image_src: str | None, page_url: str) -> str:
    """A compact card for an iframe on someone else's page.

    Not a smaller copy of the full page. An embed lives in a post or a wiki
    where the surrounding text is doing the explaining, so it carries the
    result, what it is, and an honest one-line summary of the chain — then
    links out. Reproducing the claim language in a 300px card would either
    crowd out the figure or, worse, abbreviate the caveats into the badge this
    feature deliberately does not have.

    The whole card is the link, and it opens the full page: whatever a reader
    decides on the strength of a figure in someone else's blog, the provenance
    is one click away rather than paraphrased here.

    The figure is height-capped rather than left to its natural size. A square
    plot at full width is taller than the frame an oEmbed consumer reserves,
    which pushed the title and the link to the provenance below the fold — an
    embed that is only an image, which is the one thing it must not be.
    """
    # Every non-root node, matching what the full page lists as an ancestor.
    # Counting only artifacts dropped table inputs, so a figure read straight
    # from a table reported no steps at all while the page showed one.
    root = next((node for node in lineage.nodes if node.artifact_id == artifact.id), None)
    steps = sum(1 for node in lineage.nodes if node is not root)
    title = publication.title or f"{artifact.id}@v={artifact.version}"

    bits = []
    if steps > 0:
        bits.append(f"{steps} step{'s' if steps != 1 else ''} behind it")
    if artifact.created_at:
        bits.append(_when(artifact.created_at))
    summary = " · ".join(bits)

    figure = f"<img class='fig' alt='' src='{escape(image_src)}'>" if image_src else ""
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='robots' content='noindex'>"
        f"<title>{escape(title)}</title><style>{_EMBED_STYLE}</style></head><body>"
        f"<a class='card' href='{escape(page_url)}' target='_blank' rel='noopener'>"
        f"{figure}"
        "<div class='body'>"
        f"<p class='title'>{escape(title)}</p>"
        f"<p class='meta'>{escape(summary)}</p>"
        "<p class='more'>See what produced it →</p>"
        "</div></a></body></html>"
    )


def build_record(
    *, publication, artifact, lineage, content_type: str, archived: bool = False
) -> dict:
    """The machine-readable account behind the page.

    Shared by the hosted JSON route and the archival bundle's manifest so the
    two cannot drift. A bundle that described a result differently from the
    live page would be a second, quieter account of the same thing — exactly
    what a reader checking a citation should never have to reconcile.

    The *event* block is the one part that legitimately differs. Archiving is
    not publishing — no link is minted and nothing is served — so a bundle
    reporting a ``published_at`` would be dating an event that never happened,
    to a machine consumer of a deposit that has no way to know better.
    """
    event = (
        {
            "archived_at": publication.published_at,
            "archived_by": publication.published_by,
            "authors": [dict(a) for a in publication.authors],
            "external_ids": [dict(e) for e in publication.external_ids],
        }
        if archived
        else {
            "token": publication.token or None,
            "published_at": publication.published_at,
            "published_by": publication.published_by,
            "revoked_at": publication.revoked_at,
            "authors": [dict(a) for a in publication.authors],
            "external_ids": [dict(e) for e in publication.external_ids],
        }
    )
    return {
        ("archive" if archived else "publication"): {
            "artifact_id": publication.artifact_id,
            "version": publication.version,
            "title": publication.title,
            **event,
        },
        "artifact": {
            "artifact_id": artifact.id,
            "version": artifact.version,
            "provenance_hash": artifact.provenance_hash,
            "content_type": content_type,
            "created_at": artifact.created_at,
            "byte_size": artifact.byte_size,
            "row_count": artifact.row_count,
            "principal": artifact.principal,
        },
        "content_sha256": publication.content_sha256,
        "lineage": lineage.model_dump(),
        "claims": CLAIMS,
    }


def _document(
    *,
    title: str,
    body: str,
    oembed_url: str | None = None,
    json_ld: str | None = None,
    share: list[tuple[str, str]] | None = None,
) -> str:
    # The discovery link is how a wiki or CMS turns a pasted URL into the card
    # without being told the endpoint exists. Omitted for the archival bundle,
    # which has no server to ask.
    discovery = (
        f"<link rel='alternate' type='application/json+oembed' "
        f"href='{escape(oembed_url)}' title='{escape(title)}'>"
        if oembed_url
        else ""
    )
    # The same graph the crate carries, inline, for anything that reads
    # structured data off a page.
    #
    # Every `<` becomes `\u003c`, not just `</`. HTML-escaping is wrong here —
    # JSON-LD is script content, and entities would corrupt the JSON while
    # leaving the injection — but escaping only the closing form is not enough
    # either: `<!--<script>` puts the tokenizer in script-data-double-escaped
    # state, where this block's own `</script>` no longer closes the element
    # and the rest of the document is swallowed as script text. The page then
    # renders blank, which is a self-inflicted defacement of the one page this
    # feature exists to serve. `\u003c` is a JSON string escape, so the parsed
    # value is unchanged.
    structured = (
        f"<script type='application/ld+json'>{json_ld.replace('<', '\\u003c')}</script>"
        if json_ld
        else ""
    )
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='robots' content='noindex'>"
        f"{discovery}{structured}"
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{body}</main></body></html>"
    )
