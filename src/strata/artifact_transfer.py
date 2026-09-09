"""Moving an artifact and the chain behind it into another store.

Two callers want the same walk. ``strata artifact publish`` copies a chain
into the store that serves the link, which on a hosted deployment is a
different machine from the one that ran the cells. ``promote`` copies a chain
into the team's store so colleagues can find the result by name — and so the
team cache, which is keyed by provenance, hits on every step behind it.

The walk is written once, against a :class:`PublicationTarget`, so a store
reached over HTTP and a store on this disk are two transports rather than two
implementations of the same ancestors-first, rewrite-the-edges logic.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

from strata.artifact_store import (
    ArtifactStore,
    ArtifactVersion,
    ImportedArtifact,
    Publication,
)

# One record and its bytes. Generous, because an artifact can be large and the
# alternative to waiting is a half-copied chain.
REMOTE_TIMEOUT_SECONDS = 300.0


class PublicationTarget(Protocol):
    """Where a chain is copied to and a grant is minted.

    Two implementations: an ``ArtifactStore`` on this machine, and
    ``RemoteStore`` over HTTP. Declared so the copy walk is written once
    against a contract rather than twice against two transports.
    """

    db_path: Path

    def import_artifact(self, record: ArtifactVersion, blob: bytes | None) -> ImportedArtifact: ...

    def publish_artifact(
        self,
        artifact_id: str,
        version: int,
        *,
        tenant: str | None = None,
        published_by: str | None = None,
        title: str | None = None,
    ) -> Publication: ...


class RemoteStore:
    """A store on another machine, reached over HTTP.

    Duck-types the two methods ``copy_chain`` uses, so copying a
    chain to a served store on a different host is the same walk with a
    different transport rather than a second implementation of the same
    ancestors-first, rewrite-the-edges logic.
    """

    def __init__(self, base_url: str, headers: dict[str, str] | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        # Distinct from any local store's, so the caller's "are these the same
        # store" check never accidentally matches.
        self.db_path = Path(f"<remote:{self.base_url}>")

    def import_artifact(self, record: ArtifactVersion, blob: bytes | None):
        """POST one record and its bytes; return where the far side put them."""
        import httpx

        from strata.artifact_store import ImportedArtifact

        metadata = {
            key: getattr(record, key)
            for key in (
                "id",
                "version",
                "state",
                "provenance_hash",
                "schema_json",
                "row_count",
                "byte_size",
                "created_at",
                "transform_spec",
                "input_versions",
                "principal",
            )
        }
        if blob is not None:
            metadata["content_sha256"] = hashlib.sha256(blob).hexdigest()

        files: dict[str, tuple[str, Any, str]] = {
            "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
        }
        if blob is not None:
            files["data"] = ("data.bin", blob, "application/octet-stream")

        response = httpx.post(
            f"{self.base_url}/v1/artifacts/import",
            files=files,
            headers=self._headers,
            timeout=REMOTE_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Import of {record.id}@v={record.version} was refused with "
                f"HTTP {response.status_code}: {detail_of(response)}"
            )
        body = response.json()
        return ImportedArtifact(
            id=str(body["id"]),
            version=int(body["version"]),
            written=bool(body.get("written")),
        )

    def set_name(self, name: str, artifact_id: str, version: int) -> None:
        """Point a team name at what landed here."""
        self._post(
            "/v1/names",
            {"name": name, "artifact_id": artifact_id, "version": version},
            what=f"name {name!r}",
        )

    def set_alias(self, name: str, alias: str, artifact_id: str, version: int) -> bool:
        """Move ``name@alias``. Returns whether it applied rather than queued.

        A protected alias answers 202 and lands in the pending queue for
        someone else to approve, which is the point of protecting it — so that
        is a normal outcome to report, not a failure to raise.
        """
        response = self._post(
            f"/v1/names/{name}/aliases/{alias}",
            {"artifact_id": artifact_id, "version": version},
            what=f"alias {name}@{alias}",
            method="put",
        )
        return response.status_code != 202

    def set_tag(self, artifact_id: str, version: int, key: str, value: str) -> None:
        self._post(
            f"/v1/artifacts/{artifact_id}/v/{version}/tags",
            {"key": key, "value": value},
            what=f"tag {key}",
            method="put",
        )

    def _post(self, path: str, payload: dict, *, what: str, method: str = "post"):
        import httpx

        response = getattr(httpx, method)(
            f"{self.base_url}{path}",
            json=payload,
            headers=self._headers,
            timeout=REMOTE_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"The store refused to set the {what}: "
                f"HTTP {response.status_code}: {detail_of(response)}"
            )
        return response

    def publish_artifact(
        self,
        artifact_id: str,
        version: int,
        *,
        tenant: str | None = None,
        published_by: str | None = None,
        title: str | None = None,
    ) -> Publication:
        """Mint the grant on the far side, where the link will resolve from.

        ``tenant`` and ``published_by`` are deliberately not sent: the far side
        takes both from the authenticated caller, so a client that could name
        them would be claiming an identity rather than presenting one.
        """
        import httpx

        payload = {"title": title}
        response = httpx.post(
            f"{self.base_url}/v1/artifacts/{artifact_id}/v/{version}/publish",
            json=payload,
            headers=self._headers,
            timeout=REMOTE_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise ValueError(
                f"the store refused to publish with HTTP {response.status_code}: "
                f"{detail_of(response)}"
            )
        body = response.json()
        return Publication(
            token=str(body["token"]),
            artifact_id=str(body["artifact_id"]),
            version=int(body["version"]),
            title=body.get("title"),
            published_at=float(body.get("published_at") or time.time()),
            published_by=body.get("published_by"),
            content_sha256=body.get("content_sha256"),
            revoked_at=body.get("revoked_at"),
        )


def detail_of(response) -> str:
    """The server's own explanation, when it gave one."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:200]
    if isinstance(payload, dict):
        return str(payload.get("detail") or payload.get("error") or payload)
    return str(payload)


def remap_input_versions(record: ArtifactVersion, remap: dict[str, str]) -> ArtifactVersion:
    """Point a record's lineage edges at the rows its ancestors landed on.

    Edges are recorded twice over, as the key ``strata://artifact/<id>@v=<n>``
    and again as the value ``<id>@v=<n>``; the walk reads the key
    (``_walk_lineage``) and staleness reads the value, so both have to move or
    the two disagree about the same edge.
    """
    if not record.input_versions:
        return record
    edges = json.loads(record.input_versions)
    prefix = "strata://artifact/"
    moved = {}
    for uri, version in edges.items():
        ref = uri[len(prefix) :] if uri.startswith(prefix) else None
        landed = remap.get(ref) if ref else None
        if landed is None:
            moved[uri] = version
        else:
            moved[f"{prefix}{landed}"] = landed
    return replace(record, input_versions=json.dumps(moved))


def copy_chain(
    source: ArtifactStore, target: PublicationTarget, artifact: ArtifactVersion, max_depth: int
) -> tuple[int, str]:
    """Copy an artifact and everything behind it into the served store.

    Notebook cells write to the notebook's own ``.strata/artifacts``; the server
    serves whatever ``artifact_dir`` it was configured with, which by default is
    ``~/.strata/artifacts``. Publishing a figure therefore minted a token in a
    store the page route never reads, and the link 404'd — the primary case the
    feature exists for, working only when the two happened to be the same
    directory.

    The ancestry goes too, and has to: the page shows the code and environment
    of every upstream step, so copying the artifact alone would publish a
    result whose chain resolves to nothing.

    Ancestors first, so a descendant is never briefly readable with edges
    pointing at rows that have not landed, and so each descendant can be
    rewritten to name where its ancestors actually landed: an ancestor whose
    computation the target already holds under another id resolves to that row,
    and an edge still naming the source's id would resolve to nothing here.

    Returns how many artifacts were newly written, and the ref the published
    artifact itself landed on, which is not the caller's when it deduplicated.
    """
    from strata.services.artifact import ArtifactService

    lineage = ArtifactService().build_lineage(
        source,
        artifact=artifact,
        artifact_id=artifact.id,
        version=artifact.version,
        tenant_filter=None,
        max_depth=max_depth,
    )
    copied = 0
    remap: dict[str, str] = {}
    published_ref = f"{artifact.id}@v={artifact.version}"
    landed_ref = published_ref
    for node in reversed(lineage.nodes):
        # Table nodes are leaves naming an external source, not artifacts this
        # store holds; there is nothing to copy and nothing to serve.
        if node.type != "artifact" or node.artifact_id is None or node.version is None:
            continue

        record = source.get_artifact(node.artifact_id, node.version)
        if record is None:
            continue
        record = remap_input_versions(record, remap)
        reader_cm = source.open_blob_reader(node.artifact_id, node.version)
        blob = None
        if reader_cm is not None:
            with reader_cm as reader:
                blob = reader.read()
        imported = target.import_artifact(record, blob)
        if imported.written:
            copied += 1

        source_ref = f"{node.artifact_id}@v={node.version}"
        if imported.ref != source_ref:
            remap[source_ref] = imported.ref
            if source_ref == published_ref:
                landed_ref = imported.ref
    return copied, landed_ref


@dataclass(frozen=True)
class Promotion:
    """What promoting placed in the team's store, and under what name."""

    name: str
    ref: str
    copied: int
    alias: str | None = None
    # A protected alias lands in the pending queue instead of moving, which is
    # the point of protecting it. The caller has to say so: a promotion that
    # reported plain success would leave someone believing the champion moved.
    alias_pending: bool = False


def promote_artifact(
    source: ArtifactStore,
    target: RemoteStore,
    artifact: ArtifactVersion,
    *,
    name: str,
    alias: str | None = None,
    tags: dict[str, str] | None = None,
    max_depth: int = 10,
) -> Promotion:
    """Copy an artifact and its chain to the team store, and name it there.

    Publishing mints a public link. Promoting does not: it puts a result where
    colleagues can find it by name, inside the store their own cells already
    read from.

    The chain travels for the same reason it does when publishing, and for one
    more: the team cache is keyed by provenance, so an ancestor that arrives is
    a cache hit for the next person whose cell computes the same thing. Sending
    the artifact alone would share the answer and none of the work.

    Raises:
        ValueError: If the artifact's bytes are not readable, so what would
            arrive is a name pointing at nothing.
        RuntimeError: If the far side refused a copy or a registry write. The
            chain already copied stays — it is keyed by provenance, so it is a
            usable cache entry whether or not it ever got a name.
    """
    if artifact.state not in ("ready", "superseded"):
        raise ValueError(
            f"{artifact.id}@v={artifact.version} is not readable (state={artifact.state})"
        )

    copied, landed_ref = copy_chain(source, target, artifact, max_depth)
    landed_id, _, landed_version = landed_ref.partition("@v=")
    version = int(landed_version)

    target.set_name(name, landed_id, version)
    alias_pending = False
    if alias:
        alias_pending = not target.set_alias(name, alias, landed_id, version)
    for key, value in (tags or {}).items():
        target.set_tag(landed_id, version, key, value)

    return Promotion(
        name=name,
        ref=landed_ref,
        copied=copied,
        alias=alias,
        alias_pending=alias_pending,
    )
