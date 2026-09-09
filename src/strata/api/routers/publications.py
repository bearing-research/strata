"""Publication routes: opt-in public read grants for a single artifact version.

The point is a figure in a paper carrying a URL that a referee — with no
account and no install — can open to see what produced it. That requires two
things nothing else here does: an *unauthenticated* read path, and a rendering
a person can read rather than a JSON envelope.

What the page asserts is deliberately narrow. It shows the code, inputs,
environment and chain (transparency), and it states that the bytes match the
hash recorded when they were produced (integrity). It does **not** say
"verified" or "reproduced": reproduction needs a re-run, and RNG seeds, thread
counts, float accumulation order and private input data each break it
independently. In a research context a green check is read as "someone
reproduced this", and a badge that can be wrong is worse than no badge.

Integrity here also proves only that the record was not edited after the fact
— not that it was not fabricated by whoever controls the store. The page says
so in those words rather than implying more.

Publishing exposes the whole ancestry, by design: upstream cell sources,
environments, table names and authors. That is the transparency being asked
for, and it is also why publishing is explicit and per-version, never a
one-click on a notebook. ``GET /v1/publications/{token}/preview`` returns
exactly what would become public, before it does.
"""

from __future__ import annotations

import json
from html import escape
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from strata.api.badge import badge_for
from strata.api.dependencies import (
    CurrentPrincipal,
    CurrentTenant,
    ReadStore,
    require_scope,
)
from strata.api.provenance_ld import build_crate
from strata.api.publication_page import (
    build_record,
    content_type_of,
    render_embed,
    render_publication,
)
from strata.services.artifact import ArtifactService

router = APIRouter(tags=["publications"])


# Schemes a reader can be expected to resolve. Anything else is better refused
# than stored: a record that accepts arbitrary scheme names produces citation
# lines nobody can follow, and the caller finds out from a reader rather than
# from the API.
EXTERNAL_ID_SCHEMES = ("doi", "zenodo", "arxiv", "url")


class Author(BaseModel):
    """Who wrote the work, as distinct from who made the grant."""

    name: str = Field(..., min_length=1, max_length=256)
    orcid: str | None = Field(default=None, max_length=64)
    affiliation: str | None = Field(default=None, max_length=512)


class ExternalId(BaseModel):
    scheme: Literal["doi", "zenodo", "arxiv", "url"]
    value: str = Field(..., min_length=1, max_length=512)


class PublishRequest(BaseModel):
    title: str | None = None
    authors: list[Author] | None = None
    external_ids: list[ExternalId] | None = None


class PublicationCreditsRequest(BaseModel):
    """What a publication can be told after it exists.

    Deliberately no ``artifact_id`` or ``version``: the binding is permanent,
    and a request shape that cannot name them cannot be talked into changing
    them. A field left ``None`` is untouched; an empty list clears it.
    """

    authors: list[Author] | None = None
    external_ids: list[ExternalId] | None = None


class PublicationResponse(BaseModel):
    token: str
    url: str
    artifact_id: str
    version: int
    title: str | None = None
    published_at: float
    published_by: str | None = None
    revoked_at: float | None = None
    authors: list[dict[str, str]] = []
    external_ids: list[dict[str, str]] = []


def _to_response(publication) -> PublicationResponse:
    return PublicationResponse(
        token=publication.token,
        url=f"/p/{publication.token}",
        artifact_id=publication.artifact_id,
        version=publication.version,
        title=publication.title,
        published_at=publication.published_at,
        published_by=publication.published_by,
        revoked_at=publication.revoked_at,
        authors=[dict(a) for a in publication.authors],
        external_ids=[dict(e) for e in publication.external_ids],
    )


@router.post(
    "/v1/artifacts/{artifact_id}/v/{version}/publish",
    response_model=PublicationResponse,
    dependencies=[require_scope("artifacts:publish")],
)
async def publish_artifact(
    artifact_id: str,
    version: int,
    store: ReadStore,
    tenant_filter: CurrentTenant,
    principal: CurrentPrincipal,
    request: PublishRequest | None = None,
):
    """Grant unauthenticated read access to one artifact version.

    Idempotent: publishing an already-published version returns its existing
    token rather than minting a second. Two live URLs for one artifact would
    mean revoking one and believing the artifact had been withdrawn.
    """
    try:
        publication = store.publish_artifact(
            artifact_id,
            version,
            tenant=tenant_filter,
            published_by=principal.id if principal is not None else None,
            title=request.title if request is not None else None,
            authors=[a.model_dump(exclude_none=True) for a in (request.authors or [])]
            if request is not None
            else None,
            external_ids=[e.model_dump() for e in (request.external_ids or [])]
            if request is not None
            else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_response(publication)


@router.get("/v1/publications", response_model=list[PublicationResponse])
async def list_publications(
    store: ReadStore,
    tenant_filter: CurrentTenant,
    include_revoked: bool = False,
):
    """Every grant in this tenant, newest first."""
    return [
        _to_response(p)
        for p in store.list_publications(tenant=tenant_filter, include_revoked=include_revoked)
    ]


@router.patch(
    "/v1/publications/{token}",
    response_model=PublicationResponse,
    dependencies=[require_scope("artifacts:publish")],
)
async def update_publication_credits(
    token: str,
    request: PublicationCreditsRequest,
    store: ReadStore,
    tenant_filter: CurrentTenant,
):
    """Record who wrote a publication and what identifies it.

    Separate from publishing because a DOI is registered against a deposit
    that already has to be reachable, so the identifier almost always arrives
    after the token does.

    It cannot repoint the token. ``artifact_id`` and ``version`` are not fields
    of the request and not columns this write names — a citation whose target
    could change under the reader would be worthless, and the request shape is
    where that is easiest to guarantee.
    """
    publication = store.update_publication_credits(
        token,
        tenant=tenant_filter,
        authors=(
            [a.model_dump(exclude_none=True) for a in request.authors]
            if request.authors is not None
            else None
        ),
        external_ids=(
            [e.model_dump() for e in request.external_ids]
            if request.external_ids is not None
            else None
        ),
    )
    if publication is None:
        raise HTTPException(status_code=404, detail="No publication with that token")
    return _to_response(publication)


@router.delete(
    "/v1/publications/{token}",
    dependencies=[require_scope("artifacts:publish")],
)
async def revoke_publication(
    token: str,
    store: ReadStore,
    tenant_filter: CurrentTenant,
):
    """Withdraw a grant.

    The row survives, so the token is never reissued for other content — a URL
    already printed in a paper has to fail closed rather than start resolving
    to something else.
    """
    if not store.revoke_publication(token, tenant=tenant_filter):
        raise HTTPException(status_code=404, detail="No active publication with that token")
    return {"revoked": True, "token": token}


# ---------------------------------------------------------------------------
# Public routes — no authentication. The token is the credential.
# ---------------------------------------------------------------------------


def _load_published(store, token: str, *, require_active: bool):
    """Resolve a token to (publication, artifact), or raise the right error.

    A revoked grant resolves rather than 404s so the page can say "withdrawn"
    — a reader chasing a footnote deserves that answer rather than one that
    reads as a typo. Byte and verify routes still refuse it.
    """
    publication = store.get_publication(token)
    if publication is None:
        raise HTTPException(status_code=404, detail="No such publication")
    if require_active and not publication.is_active:
        raise HTTPException(status_code=410, detail="This publication was withdrawn")

    artifact = store.get_artifact(publication.artifact_id, publication.version)
    if artifact is None:
        raise HTTPException(status_code=404, detail="The published artifact is gone")
    return publication, artifact


@router.get("/v1/publications/{token}", response_model=None)
async def read_publication_record(token: str, store: ReadStore):
    """The machine-readable record behind the page. Unauthenticated."""
    publication, artifact = _load_published(store, token, require_active=False)
    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=25,
    )
    record = build_record(
        publication=publication,
        artifact=artifact,
        lineage=lineage,
        content_type=content_type_of(artifact),
    )
    record["publication"]["url"] = f"/p/{publication.token}"
    return record


@router.get("/p/{token}", response_class=HTMLResponse)
async def publication_page(token: str, store: ReadStore, http_request: Request):
    """The page a citation points at. Unauthenticated, self-contained HTML."""
    publication, artifact = _load_published(store, token, require_active=False)

    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=25,
    )
    content_type = content_type_of(artifact)

    inline_png = None
    if publication.is_active and content_type == "image/png":
        inline_png = _inline_png(store, publication)

    base = _public_base(http_request)
    page_url = quote(f"{base}/p/{token}", safe="")
    crate = build_crate(
        publication=publication,
        artifact=artifact,
        lineage=lineage,
        content_type=content_type,
        payload_id=f"{base}/p/{token}/data",
        include_descriptor=False,
    )
    return HTMLResponse(
        render_publication(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type=content_type,
            image_src=inline_png,
            oembed_url=f"{base}/oembed?url={page_url}",
            json_ld=json.dumps(crate),
            # Nobody assembles a linked badge by hand, and a snippet that has
            # to be reconstructed from three route names is a snippet nobody
            # uses.
            share=[
                (
                    "A badge for a README, linking here:",
                    f"[![provenance]({base}/p/{token}/badge.svg)]({base}/p/{token})",
                ),
                (
                    "The card, embedded in a page:",
                    f'<iframe src="{base}/p/{token}/embed" width="480" '
                    'height="420" frameborder="0"></iframe>',
                ),
                ("The chain, as RO-Crate JSON-LD:", f"{base}/p/{token}/ro-crate"),
            ],
        )
    )


# A figure is the case this feature exists for, so it is embedded rather than
# linked — a page that renders the plot immediately is the difference between
# a reader seeing the result and a reader downloading a file. Bounded: past
# this, the page links to the bytes instead of carrying them.
_MAX_INLINE_PNG_BYTES = 4 * 1024 * 1024


def _inline_png(store, publication) -> str | None:
    import base64

    if (store.blob_size(publication.artifact_id, publication.version) or 0) > (
        _MAX_INLINE_PNG_BYTES
    ):
        return None
    reader_cm = store.open_blob_reader(publication.artifact_id, publication.version)
    if reader_cm is None:
        return None
    with reader_cm as reader:
        blob = reader.read(_MAX_INLINE_PNG_BYTES + 1)
    if len(blob) > _MAX_INLINE_PNG_BYTES:
        return None
    return f"data:image/png;base64,{base64.b64encode(blob).decode('ascii')}"


@router.get("/p/{token}/data")
async def publication_data(token: str, store: ReadStore):
    """The published bytes. Unauthenticated, and only ever this artifact's own.

    Ancestors are described on the page but never served: showing which steps
    produced a result is transparency, handing over the upstream datasets is
    not the same thing and was not what publishing consented to.
    """
    publication, _ = _load_published(store, token, require_active=True)

    reader_cm = store.open_blob_reader(publication.artifact_id, publication.version)
    if reader_cm is None:
        raise HTTPException(status_code=404, detail="The published bytes are gone")

    def _iter_blob():
        with reader_cm as reader:
            while chunk := reader.read(1024 * 1024):
                yield chunk

    return StreamingResponse(
        _iter_blob(),
        media_type=content_type_of(store.get_artifact(publication.artifact_id, publication.version))
        or "application/octet-stream",
    )


@router.get("/p/{token}/archive.zip")
async def publication_archive(token: str, store: ReadStore):
    """The self-contained bundle, as a zip. Unauthenticated, like the page.

    Same files ``strata artifact archive`` writes, from the same code — a
    service that holds only HTTP access to the store can build the deposit
    copy without the store's credentials, which is the whole reason this is a
    route and not only a command.

    It contains nothing the page does not already show: the artifact's own
    bytes, the record, the RO-Crate, and the chain as rendered. Upstream bytes
    stay where they are, exactly as on ``/p/{token}/data``.

    A withdrawn publication refuses here as it does for bytes and verify. The
    page still resolves and says "withdrawn", because a reader chasing a
    footnote deserves that answer; handing them the archive anyway would
    undo the withdrawal.
    """
    import io
    import tempfile
    import zipfile
    from base64 import b64encode
    from hashlib import sha256
    from pathlib import Path

    from strata.api.publication_bundle import write_bundle

    publication, artifact = _load_published(store, token, require_active=True)

    with tempfile.TemporaryDirectory() as workdir:
        dest = Path(workdir)
        try:
            written = write_bundle(store, artifact, dest, publication=publication)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        buffer = io.BytesIO()
        # Deterministic member order, so two archives of one publication differ
        # only where their contents do. Reading order rather than alphabetical:
        # a person who unzips this should meet index.html first.
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
            for name in written:
                bundle.write(dest / name, arcname=name)

    payload = buffer.getvalue()
    digest = b64encode(sha256(payload).digest()).decode()
    return Response(
        content=payload,
        media_type="application/zip",
        headers={
            "Content-Digest": f"sha-256=:{digest}:",
            "Content-Disposition": f'attachment; filename="{token}.zip"',
        },
    )


@router.get("/p/{token}/verify")
async def verify_publication(token: str, store: ReadStore):
    """Re-read the bytes and compare against the digest recorded at publication.

    The one check this page can actually perform. It catches a blob that has
    been altered or corrupted since publishing; it says nothing about whether
    the result was honestly produced in the first place, which no amount of
    hashing can establish.
    """
    publication, _ = _load_published(store, token, require_active=True)
    if not publication.content_sha256:
        raise HTTPException(
            status_code=409,
            detail="No digest was recorded for this publication; nothing to check against",
        )

    actual = store.blob_digest(publication.artifact_id, publication.version)
    return {
        "matches": actual == publication.content_sha256,
        "recorded_sha256": publication.content_sha256,
        "actual_sha256": actual,
        "checks": "That the bytes are unchanged since publication. Not that they are correct.",
    }


# ---------------------------------------------------------------------------
# Embedding — the card, and the oEmbed endpoint that unfurls a pasted link
# ---------------------------------------------------------------------------

# The card's natural size. oEmbed consumers use these to reserve space before
# the iframe loads; the card itself is fluid and fills whatever it is given.
_EMBED_WIDTH = 480
_EMBED_HEIGHT = 420


def _public_base(request: Request) -> str:
    """The origin a *reader* reaches this server on.

    An embed's URLs are consumed by someone else's page, so they have to be
    the public ones. ``request.base_url`` is right for a directly-reachable
    server and wrong behind a reverse proxy on another host, where it is the
    internal address: the published page would advertise an oEmbed endpoint no
    consumer can resolve, and that endpoint would reject the public URL a wiki
    actually pastes. ``public_base_url`` is how an operator says what the
    outside sees.
    """
    from strata.server import get_state

    try:
        configured = get_state().config.public_base_url
    except RuntimeError:
        configured = None
    return (configured or str(request.base_url)).rstrip("/")


def _same_host(left: str, right: str) -> bool:
    """Compare two origins' hosts, ignoring case and a default port.

    A consumer pastes whatever the reader's address bar held, which may differ
    from the canonical form in case or an explicit ``:443``. Rejecting those
    would 404 the tools this endpoint exists for, over a difference that names
    the same server.
    """
    from urllib.parse import urlparse

    def _key(value: str) -> tuple[str, str]:
        parsed = urlparse(value if "//" in value else f"//{value}")
        host = (parsed.hostname or "").lower()
        default = {"http": 80, "https": 443}.get(parsed.scheme or "")
        port = parsed.port if parsed.port != default else None
        return host, str(port or "")

    return _key(left) == _key(right)


@router.get("/p/{token}/embed", response_class=HTMLResponse)
async def publication_embed(token: str, store: ReadStore, http_request: Request):
    """A compact card, sized for an iframe in a post or a wiki."""
    publication, artifact = _load_published(store, token, require_active=True)
    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=25,
    )
    content_type = content_type_of(artifact)
    image_src = _inline_png(store, publication) if content_type == "image/png" else None

    return HTMLResponse(
        render_embed(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            image_src=image_src,
            page_url=f"{_public_base(http_request)}/p/{token}",
        )
    )


@router.get("/oembed")
async def oembed(url: str, store: ReadStore, http_request: Request, format: str = "json"):
    """oEmbed provider, so pasting a link unfurls into the card.

    The endpoint every wiki, CMS and note-taking tool already knows how to ask.
    Without it an embed means hand-writing an ``<iframe>``, which most of those
    tools will not accept from an author in the first place.

    Only ``json`` is served. XML is in the spec and nothing has asked for it
    this decade; a 501 naming the reason beats a silently empty document.
    """
    if format != "json":
        raise HTTPException(status_code=501, detail="Only format=json is supported")

    token = _token_from_url(url, _public_base(http_request))
    if token is None:
        raise HTTPException(status_code=404, detail="Not a publication URL on this server")

    publication, artifact = _load_published(store, token, require_active=True)
    base = _public_base(http_request)
    return {
        "version": "1.0",
        "type": "rich",
        "provider_name": "Strata",
        "provider_url": base,
        "title": publication.title or f"{artifact.id}@v={artifact.version}",
        "width": _EMBED_WIDTH,
        "height": _EMBED_HEIGHT,
        # Escaped: ``base`` derives from the Host header, and this string is
        # rendered verbatim by whatever page consumes the oEmbed response.
        "html": (
            f'<iframe src="{escape(base, quote=True)}/p/{escape(token, quote=True)}/embed" '
            f'width="{_EMBED_WIDTH}" '
            f'height="{_EMBED_HEIGHT}" frameborder="0" '
            'style="border:0;max-width:100%" '
            'title="Strata published artifact" loading="lazy"></iframe>'
        ),
    }


def _token_from_url(url: str, base: str) -> str | None:
    """Pull the token out of a publication URL, or ``None`` if it is not one.

    Matched against this server's own origin. An oEmbed provider that happily
    described URLs on other hosts would be answering for pages it has never
    seen.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not _same_host(base, url):
        return None
    parts = [segment for segment in parsed.path.split("/") if segment]
    if len(parts) < 2 or parts[0] != "p":
        return None
    return parts[1]


@router.get("/p/{token}/ro-crate")
async def publication_ro_crate(token: str, store: ReadStore, http_request: Request):
    """The chain as RO-Crate JSON-LD, for software rather than readers.

    The same graph the page carries inline and a deposited bundle ships as
    ``ro-crate-metadata.json`` — served on its own so a harvester can fetch it
    without scraping a page for a script tag.
    """
    publication, artifact = _load_published(store, token, require_active=True)
    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=25,
    )
    base = _public_base(http_request)
    return JSONResponse(
        build_crate(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type=content_type_of(artifact),
            payload_id=f"{base}/p/{token}/data",
            # Unlike the inline block, this response *is* the crate document,
            # so it carries the descriptor. Without it there is no
            # ``conformsTo`` and a harvester cannot tell an RO-Crate from any
            # other JSON-LD — which is the whole reason to fetch this endpoint.
            include_descriptor=True,
        ),
        media_type="application/ld+json",
    )


@router.get("/p/{token}/badge.svg")
async def publication_badge(token: str, store: ReadStore):
    """A README-sized pill reporting the size of the recorded chain.

    Served rather than snapshotted so a withdrawn publication stops asserting.
    Note that GitHub proxies badge images through its own cache, so a
    withdrawal can take a while to show — a badge is a pointer, never a
    revocation mechanism, and the docs say so.
    """
    publication, artifact = _load_published(store, token, require_active=False)
    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=25,
    )
    root = next((node for node in lineage.nodes if node.artifact_id == artifact.id), None)
    steps = sum(1 for node in lineage.nodes if node is not root)

    return Response(
        content=badge_for(publication=publication, step_count=steps),
        media_type="image/svg+xml",
        # Short, because the badge is the only surface that can go stale in
        # someone else's page after a withdrawal.
        headers={"Cache-Control": "public, max-age=300"},
    )
