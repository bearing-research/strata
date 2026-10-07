"""Publication routes: opt-in public read grants for a single artifact version.

A published version gets an unauthenticated URL a referee can open. The page
asserts transparency (code, inputs, environment, chain) and integrity (bytes
match the recorded hash), never "verified" or "reproduced": reproduction needs
a re-run, and integrity does not rule out fabrication by whoever controls the
store. Publishing exposes the whole ancestry, which is why it is explicit and
per-version; ``GET /v1/artifacts/{id}/v/{n}/lineage`` shows what would become
public.
"""

from __future__ import annotations

import asyncio
import json
from html import escape
from pathlib import Path
from typing import Literal
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from strata.api.badge import badge_for
from strata.api.dependencies import (
    CurrentPrincipal,
    CurrentTenant,
    ReadStore,
    require_scope,
)
from strata.api.provenance_ld import build_crate
from strata.api.publication_bundle import (
    PUBLICATION_MAX_DEPTH,
    cached_bundle_zip,
    companion_digests,
    drop_cached_bundles,
    payload_filename,
)
from strata.api.publication_page import (
    build_record,
    content_type_of,
    render_embed,
    render_publication,
)
from strata.api.served_bytes import served_media_type
from strata.services.artifact import ArtifactService

router = APIRouter(tags=["publications"])


# Schemes a reader can resolve. Anything else is refused rather than stored as a citation line
# nobody can follow.
EXTERNAL_ID_SCHEMES = ("doi", "zenodo", "arxiv", "url")


class Author(BaseModel):
    """Who wrote the work, as distinct from who made the grant."""

    # So a whitespace-only name fails min_length rather than crediting nobody.
    model_config = ConfigDict(str_strip_whitespace=True)

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

    No ``artifact_id`` or ``version``, so the binding cannot be changed; the route
    refuses either key rather than ignore it. A field left ``None`` is untouched; an
    empty list clears it.
    """

    # Extra keys are kept so the route can see a repoint attempt.
    model_config = ConfigDict(extra="allow")

    authors: list[Author] | None = None
    external_ids: list[ExternalId] | None = None


class PublicationResponse(BaseModel):
    # The store keeps only the token's SHA-256 (``id``), so the token and its link
    # are in the response that mints them and in no other.
    id: str
    token: str | None = None
    url: str | None = None
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
        id=publication.id,
        token=publication.token or None,
        url=f"/p/{publication.token}" if publication.token else None,
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

    Idempotent: an already-published version returns its existing grant, so one
    revocation always withdraws the artifact. Only the call that mints it carries
    the token: the store keeps its hash.
    """
    from strata.server import _authorize_artifact_read, _ensure_artifact_access

    # The strongest form of retrieval there is: what this hands out is readable
    # by anyone with the link. A principal the ACL denies the artifact's inputs
    # cannot read it here and must not be able to publish it either.
    artifact = _ensure_artifact_access(store.get_artifact(artifact_id, version), tenant_filter)
    _authorize_artifact_read(artifact, store)
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


# ``{token}`` on the authenticated routes is the raw token or the publication's id.
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
    principal: CurrentPrincipal,
):
    """Record who wrote a publication and what identifies it (e.g. a DOI).

    Separate from publishing because a DOI usually arrives after the token. It
    cannot repoint the token to another artifact or version, and refuses to try.
    """
    repoint = sorted({"artifact_id", "version"} & set(request.model_extra or {}))
    if repoint:
        raise HTTPException(
            status_code=400,
            detail=(
                f"A publication's {' and '.join(repoint)} cannot change; "
                "publish the other version for a new link"
            ),
        )
    publication = store.update_publication_credits(
        token,
        tenant=tenant_filter,
        actor=principal.id if principal is not None else None,
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
    principal: CurrentPrincipal,
):
    """Withdraw a grant.

    The row survives so the token is never reissued: a printed URL fails closed
    rather than resolving to something else.
    """
    if not store.revoke_publication(
        token, tenant=tenant_filter, actor=principal.id if principal is not None else None
    ):
        raise HTTPException(status_code=404, detail="No active publication with that token")
    drop_cached_bundles(store, token)
    return {"revoked": True, "token": token}


# --- Public routes: no authentication; the token is the credential ---


def _load_published(store, token: str, *, require_active: bool):
    """Resolve a token to (publication, artifact), or raise 404/410.

    With ``require_active=False`` a revoked grant still resolves, so the page can
    say "withdrawn"; routes that serve content pass ``True`` and get a 410.
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
    """The machine-readable record behind the page. Unauthenticated.

    A withdrawn publication answers with only what its page shows: that it was withdrawn, and when.
    """
    publication, artifact = _load_published(store, token, require_active=False)
    if not publication.is_active:
        return {
            "publication": {
                "token": publication.token,
                "url": f"/p/{publication.token}",
                "title": publication.title,
                "published_at": publication.published_at,
                "revoked_at": publication.revoked_at,
            }
        }
    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=PUBLICATION_MAX_DEPTH,
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
        max_depth=PUBLICATION_MAX_DEPTH,
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
            base=base,
            # Nobody assembles a linked badge by hand from three route names.
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


# A figure is embedded rather than linked, so a reader sees the plot instead of downloading a file.
# Bounded: past this, the page links to the bytes.
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

    Ancestors are described on the page but their bytes are never served.
    """
    publication, artifact = _load_published(store, token, require_active=True)

    reader_cm = store.open_blob_reader(publication.artifact_id, publication.version)
    if reader_cm is None:
        raise HTTPException(status_code=404, detail="The published bytes are gone")

    def _iter_blob():
        with reader_cm as reader:
            while chunk := reader.read(1024 * 1024):
                yield chunk

    media_type, headers = served_media_type(content_type_of(artifact))
    return StreamingResponse(_iter_blob(), media_type=media_type, headers=headers)


# One archive build per publication at a time; a second request waits for the first's file.
_archive_locks: dict[str, asyncio.Lock] = {}


async def _built_archive(store, artifact, publication) -> tuple[Path, str]:
    """The publication's archive, built off the loop and once per record: anyone with the link
    can ask."""
    async with _archive_locks.setdefault(publication.id, asyncio.Lock()):
        try:
            return await asyncio.to_thread(
                cached_bundle_zip, store, artifact, publication=publication
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))


@router.get("/p/{token}/archive.zip")
async def publication_archive(token: str, store: ReadStore):
    """The self-contained bundle as a zip, as ``strata artifact archive`` writes it.

    Unauthenticated, so a service with only HTTP access can build the deposit
    copy. Holds the artifact's bytes, record, RO-Crate and rendered chain, never
    upstream bytes. A withdrawn publication gets a 410.
    """
    from base64 import b64encode

    publication, artifact = _load_published(store, token, require_active=True)
    path, digest = await _built_archive(store, artifact, publication)
    return FileResponse(
        path,
        media_type="application/zip",
        headers={
            "Content-Digest": f"sha-256=:{b64encode(bytes.fromhex(digest)).decode()}:",
            "Content-Disposition": f'attachment; filename="{token}.zip"',
        },
    )


@router.get("/p/{token}/verify")
async def verify_publication(
    token: str,
    store: ReadStore,
    sha256: str | None = Query(default=None, pattern="^[0-9a-fA-F]{64}$"),
):
    """Re-read the bytes and compare against the digest recorded at publication.

    Detects alteration or corruption since publishing; says nothing about whether
    the result was honestly produced. With ``sha256``, the digest of a file the
    caller holds, it also says which of the publication's files that is: the
    published bytes or the archive's Parquet copy.
    """
    publication, artifact = _load_published(store, token, require_active=True)
    if not publication.content_sha256:
        raise HTTPException(
            status_code=409,
            detail="No digest was recorded for this publication; nothing to check against",
        )

    actual = await asyncio.to_thread(
        store.blob_digest, publication.artifact_id, publication.version
    )
    unchanged = actual == publication.content_sha256
    result: dict = {
        "matches": unchanged,
        "recorded_sha256": publication.content_sha256,
        "actual_sha256": actual,
        "checks": "That the bytes are unchanged since publication. Not that they are correct.",
    }
    if sha256 is None:
        return result

    claimed = sha256.lower()
    if claimed == publication.content_sha256:
        held: str | None = payload_filename(content_type_of(artifact))
    else:
        path, _ = await _built_archive(store, artifact, publication)
        held = companion_digests(path).get(claimed)
    result.update({"sha256": claimed, "file": held, "matches": unchanged and held is not None})
    return result


# --- Embedding: the card, and the oEmbed endpoint that unfurls a pasted link ---

# The card's natural size. oEmbed consumers use these to reserve space before
# the iframe loads; the card itself is fluid and fills whatever it is given.
_EMBED_WIDTH = 480
_EMBED_HEIGHT = 420


def _public_base(request: Request) -> str:
    """The origin and base path a *reader* reaches this server on.

    ``public_base_url`` when set, since behind a reverse proxy ``request.base_url``
    is the internal address and embeds would advertise URLs no consumer can reach.
    """
    from strata.server import get_state

    try:
        configured = get_state().config.public_base_url
    except RuntimeError:
        configured = None
    if configured:
        return configured.rstrip("/") + request.scope.get("root_path", "")
    return str(request.base_url).rstrip("/")


def _same_host(left: str, right: str) -> bool:
    """Compare two origins' hosts, ignoring case and a default port.

    A pasted URL may differ from the canonical form in case or an explicit ``:443``.
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
        max_depth=PUBLICATION_MAX_DEPTH,
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
        ),
        # Only this route may be framed anywhere; the middleware's default covers the rest.
        headers={"Content-Security-Policy": "frame-ancestors *"},
    )


@router.get("/oembed")
async def oembed(url: str, store: ReadStore, http_request: Request, format: str = "json"):
    """oEmbed provider, so pasting a publication link unfurls into the card.

    Only ``json`` is served; any other format gets a 501.
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

    Only URLs on this server's own origin match.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if not _same_host(base, url):
        return None
    base_path = urlparse(base).path
    if not parsed.path.startswith(f"{base_path}/"):
        return None
    parts = [segment for segment in parsed.path[len(base_path) :].split("/") if segment]
    if len(parts) < 2 or parts[0] != "p":
        return None
    return parts[1]


@router.get("/p/{token}/ro-crate")
async def publication_ro_crate(token: str, store: ReadStore, http_request: Request):
    """The chain as RO-Crate JSON-LD, for software rather than readers.

    The same graph the page embeds and a bundle ships as ``ro-crate-metadata.json``.
    """
    publication, artifact = _load_published(store, token, require_active=True)
    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=PUBLICATION_MAX_DEPTH,
    )
    base = _public_base(http_request)
    return JSONResponse(
        build_crate(
            publication=publication,
            artifact=artifact,
            lineage=lineage,
            content_type=content_type_of(artifact),
            payload_id=f"{base}/p/{token}/data",
            # Unlike the inline block, this response is the crate document, so it carries the
            # descriptor; without ``conformsTo`` a harvester cannot tell an RO-Crate from any other
            # JSON-LD.
            include_descriptor=True,
        ),
        media_type="application/ld+json",
    )


@router.get("/p/{token}/badge.svg")
async def publication_badge(token: str, store: ReadStore):
    """A README-sized pill reporting the size of the recorded chain.

    Served live so a withdrawn publication stops asserting, though image proxies
    such as GitHub's may cache it for a while.
    """
    publication, artifact = _load_published(store, token, require_active=False)
    lineage = ArtifactService().build_lineage(
        store,
        artifact=artifact,
        artifact_id=publication.artifact_id,
        version=publication.version,
        tenant_filter=None,
        max_depth=PUBLICATION_MAX_DEPTH,
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
