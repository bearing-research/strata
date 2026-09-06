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

from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from strata.api.dependencies import (
    CurrentPrincipal,
    CurrentTenant,
    ReadStore,
    require_scope,
)
from strata.api.publication_page import (
    build_record,
    content_type_of,
    render_embed,
    render_publication,
)
from strata.services.artifact import ArtifactService

router = APIRouter(tags=["publications"])


class PublishRequest(BaseModel):
    title: str | None = None


class PublicationResponse(BaseModel):
    token: str
    url: str
    artifact_id: str
    version: int
    title: str | None = None
    published_at: float
    published_by: str | None = None
    revoked_at: float | None = None


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
    return HTMLResponse(
        render_publication(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type=content_type,
            image_src=inline_png,
            oembed_url=f"{base}/oembed?url={page_url}",
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
    """The origin a reader reached this server on.

    Taken from the request rather than configuration because an embed's URLs
    are consumed by someone else's page: a link built from an internal
    hostname resolves for the server and for nobody the embed was made for.
    """
    return str(request.base_url).rstrip("/")


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
        "html": (
            f'<iframe src="{base}/p/{token}/embed" width="{_EMBED_WIDTH}" '
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
    if urlparse(base).netloc != parsed.netloc:
        return None
    parts = [segment for segment in parsed.path.split("/") if segment]
    if len(parts) < 2 or parts[0] != "p":
        return None
    return parts[1]
