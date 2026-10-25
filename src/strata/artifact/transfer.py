"""Copying an artifact and the chain behind it into another store.

``strata artifact publish`` copies a chain into the store that serves the link;
``promote`` copies one into the team store. The ancestors-first,
rewrite-the-edges walk is written once against :class:`PublicationTarget`, so
local and HTTP stores are two transports of the same logic.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from collections.abc import Collection, Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from vestigo.blob_store import BLOB_STREAM_CHUNK_BYTES
from vestigo.store import (
    ArtifactStore,
    ArtifactVersion,
    ImportedArtifact,
    Publication,
)

from strata.artifact.retention import requirements

if TYPE_CHECKING:
    from strata.types import ArtifactLineageResponse, LineageNode

# What another store needs to rebuild an artifact version: everything but the tenant, which the
# destination takes from the authenticated caller. One list shared by the HTTP transfer and the
# snapshot bundle, so neither importer silently loses a field the other carries.
RECORD_FIELDS = (
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


def record_metadata(record: ArtifactVersion) -> dict[str, Any]:
    """An artifact version as the portable dict :data:`RECORD_FIELDS` names."""
    return {key: getattr(record, key) for key in RECORD_FIELDS}


# One record and its bytes. Generous, because an artifact can be large and the
# alternative to waiting is a half-copied chain.
REMOTE_TIMEOUT_SECONDS = 300.0


class PublicationTarget(Protocol):
    """Where a chain is copied to and a grant is minted: ``ArtifactStore`` or ``RemoteStore``."""

    db_path: Path

    def import_artifact(
        self,
        record: ArtifactVersion,
        blob: bytes | Path | None,
        requires: Collection[tuple[str, int]] = (),
    ) -> ImportedArtifact: ...

    def find_by_provenance(
        self, provenance_hash: str, tenant: str | None = None
    ) -> ArtifactVersion | None: ...

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
    """A store on another machine, reached over HTTP; duck-types what ``copy_chain`` uses."""

    def __init__(
        self, base_url: str, headers: dict[str, str] | None = None, *, remap: bool = False
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})
        # Land a version another tenant (or computation) holds under a fresh id
        # rather than failing with 409; copy_chain rewrites edges to the new id.
        self._remap = remap
        # Distinct from any local store's, so the caller's "are these the same
        # store" check never accidentally matches.
        self.db_path = Path(f"<remote:{self.base_url}>")

    def import_artifact(
        self,
        record: ArtifactVersion,
        blob: bytes | Path | None,
        requires: Collection[tuple[str, int]] = (),
    ):
        """Upload the bytes, then POST the record naming them; return where they landed.

        The bytes are streamed to the staging route and checked there against their
        digest, so a file is never read into memory. *requires* is not sent: the
        receiving server reads it off the bytes, as its import route does for everyone.
        """
        del requires
        import httpx
        from vestigo.store import ImportedArtifact

        metadata = record_metadata(record)
        params = {"remap": "true"} if self._remap else None
        if blob is None:
            # Nothing to stage: the far side already holds the bytes, or has none to hold.
            response = httpx.post(
                f"{self.base_url}/v1/artifacts/import",
                files={"metadata": ("metadata.json", json.dumps(metadata), "application/json")},
                params=params,
                headers=self._headers,
                timeout=REMOTE_TIMEOUT_SECONDS,
            )
        else:
            digest = _sha256_of(blob)
            metadata["content_sha256"] = digest
            if isinstance(blob, Path):
                with open(blob, "rb") as body:
                    staged = httpx.put(
                        f"{self.base_url}/v1/artifacts/import/blobs/{digest}",
                        content=_chunks(body),
                        headers=self._headers,
                        timeout=REMOTE_TIMEOUT_SECONDS,
                    )
            else:
                staged = httpx.put(
                    f"{self.base_url}/v1/artifacts/import/blobs/{digest}",
                    content=blob,
                    headers=self._headers,
                    timeout=REMOTE_TIMEOUT_SECONDS,
                )
            if staged.status_code >= 400:
                raise RuntimeError(
                    f"Upload of {record.id}@v={record.version}'s bytes was refused with "
                    f"HTTP {staged.status_code}: {detail_of(staged)}"
                )
            response = httpx.post(
                f"{self.base_url}/v1/artifacts/import",
                json=metadata,
                params=params,
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

    def find_by_provenance(
        self, provenance_hash: str, tenant: str | None = None
    ) -> ArtifactVersion | None:
        """The ready version the far side holds for *provenance_hash*, or ``None``.

        ``tenant`` is not sent: the far side scopes the lookup to the authenticated caller.
        Any refusal reads as a miss, and the caller sends the bytes.
        """
        import httpx

        response = httpx.get(
            f"{self.base_url}/v1/artifacts/by-provenance/{provenance_hash}",
            headers=self._headers,
            timeout=REMOTE_TIMEOUT_SECONDS,
        )
        if response.status_code != 200:
            return None
        body = response.json()
        return ArtifactVersion(
            id=str(body["artifact_id"]),
            version=int(body["version"]),
            state=str(body.get("state") or "ready"),
            provenance_hash=str(body["provenance_hash"]),
            content_sha256=body.get("content_sha256"),
        )

    def set_name(self, name: str, artifact_id: str, version: int) -> None:
        """Point a team name at what landed here."""
        self._post(
            "/v1/names",
            {"name": name, "artifact_id": artifact_id, "version": version},
            what=f"name {name!r}",
        )

    def set_alias(self, name: str, alias: str, artifact_id: str, version: int) -> bool:
        """Move ``name@alias``; return whether it applied rather than queued.

        A protected alias answering 202 (pending approval) is a normal outcome, not an error.
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

    def export_table(
        self, artifact_id: str, version: int, table: str, alias: str | None = None
    ) -> dict:
        """Have the far side write a version it holds into an Iceberg table."""
        from urllib.parse import quote

        import httpx

        response = httpx.post(
            f"{self.base_url}/v1/artifacts/{quote(artifact_id, safe='')}/v/{version}/export",
            json={"table": table, "alias": alias},
            headers=self._headers,
            timeout=REMOTE_TIMEOUT_SECONDS,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"The store refused to write {artifact_id}@v={version} to {table}: "
                f"HTTP {response.status_code}: {detail_of(response)}"
            )
        return response.json()

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

        ``tenant`` and ``published_by`` are not sent: the far side takes both from the
        authenticated caller.
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
            # Absent when the far side already held a grant: only minting shows the token.
            token=body.get("token") or "",
            id=str(body.get("id") or ""),
            artifact_id=str(body["artifact_id"]),
            version=int(body["version"]),
            title=body.get("title"),
            published_at=float(body.get("published_at") or time.time()),
            published_by=body.get("published_by"),
            content_sha256=body.get("content_sha256"),
            revoked_at=body.get("revoked_at"),
        )


def _chunks(handle) -> Iterator[bytes]:
    while chunk := handle.read(BLOB_STREAM_CHUNK_BYTES):
        yield chunk


def _sha256_of(blob: bytes | Path) -> str:
    if isinstance(blob, bytes):
        return hashlib.sha256(blob).hexdigest()
    hasher = hashlib.sha256()
    with open(blob, "rb") as handle:
        for chunk in _chunks(handle):
            hasher.update(chunk)
    return hasher.hexdigest()


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

    Each edge is stored as both key ``strata://artifact/<id>@v=<n>`` (read by the
    lineage walk) and value ``<id>@v=<n>`` (read by staleness); both must move.
    A ``strata://name/`` input keeps its name and moves only the version it resolved to.
    """
    if not record.input_versions:
        return record
    edges = json.loads(record.input_versions)
    prefix = "strata://artifact/"
    moved = {}
    for uri, version in edges.items():
        if uri.startswith("strata://name/") and version in remap:
            moved[uri] = remap[version]
            continue
        ref = uri[len(prefix) :] if uri.startswith(prefix) else None
        landed = remap.get(ref) if ref else None
        if landed is None:
            moved[uri] = version
        else:
            moved[f"{prefix}{landed}"] = landed
    return replace(record, input_versions=json.dumps(moved))


def copy_chain(
    source: ArtifactStore, target: PublicationTarget, artifact: ArtifactVersion, max_depth: int
) -> tuple[list[str], str]:
    """Copy an artifact and its whole ancestry into the served store.

    Notebook cells write to the notebook's ``.strata/artifacts``, which the server
    does not serve, and the published page needs every upstream step. Ancestors go
    first, so a descendant is never readable with dangling edges and can be
    rewritten to where each ancestor landed (possibly an existing row with the same
    computation under another id). A ``# @output_dir`` folder in the chain goes with
    every file it names, before its manifest.

    Returns the refs newly written on the target, in landing order, and the ref the
    artifact itself landed on, which differs from the caller's when it deduplicated.

    Raises:
        ValueError: A folder in the chain names a file the source does not hold.
    """
    from strata.notebook import output_dirs
    from strata.services.artifact import ArtifactService

    lineage = ArtifactService().build_lineage(
        source,
        artifact=artifact,
        artifact_id=artifact.id,
        version=artifact.version,
        tenant_filter=None,
        max_depth=max_depth,
    )
    written: list[str] = []
    remap: dict[str, str] = {}
    published_ref = f"{artifact.id}@v={artifact.version}"
    landed_ref = published_ref
    for node in _ancestors_first(lineage):
        # Table nodes are leaves naming an external source, not artifacts this
        # store holds; there is nothing to copy and nothing to serve.
        if node.type != "artifact" or node.artifact_id is None or node.version is None:
            continue

        record = source.get_artifact(node.artifact_id, node.version)
        if record is None:
            continue
        record = remap_input_versions(record, remap)
        manifest = None
        if record.params.get("content_type") == output_dirs.CONTENT_TYPE:
            manifest = _copy_folder_files(source, target, record, written)
        imported = _copy_record(source, target, record, manifest)
        if imported.written:
            written.append(imported.ref)

        source_ref = f"{node.artifact_id}@v={node.version}"
        if imported.ref != source_ref:
            remap[source_ref] = imported.ref
            if source_ref == published_ref:
                landed_ref = imported.ref
    return written, landed_ref


def _copy_record(
    source: ArtifactStore,
    target: PublicationTarget,
    record: ArtifactVersion,
    blob: bytes | None = None,
) -> ImportedArtifact:
    """Import one version into *target* with *blob*, or else its bytes in *source*."""
    content_type = record.params.get("content_type")
    if blob is not None:
        digest = hashlib.sha256(blob).hexdigest()
        return target.import_artifact(
            replace(record, content_sha256=digest),
            blob,
            requires=requirements(record.id, content_type, blob),
        )
    with tempfile.TemporaryDirectory(prefix="strata_copy_") as workdir:
        # Spooled to disk, so an artifact never sits whole in memory.
        reader_cm = source.open_blob_reader(record.id, record.version)
        path = None
        if reader_cm is not None:
            path = Path(workdir) / "blob"
            hasher = hashlib.sha256()
            with reader_cm as reader, open(path, "wb") as out:
                for chunk in _chunks(reader):
                    hasher.update(chunk)
                    out.write(chunk)
            record = replace(record, content_sha256=hasher.hexdigest())
        return target.import_artifact(
            record, path, requires=requirements(record.id, content_type, path)
        )


def _copy_folder_files(
    source: ArtifactStore,
    target: PublicationTarget,
    manifest_record: ArtifactVersion,
    written: list[str],
) -> bytes:
    """Copy each file a folder manifest names; return the manifest naming where each landed.

    A file is not a lineage edge of its manifest, so the walk never reaches it. One the
    target already holds (same id and digest, so the same bytes) is not sent again, and
    the manifest names the version the target has. Every file is checked before any is
    sent, so a refused folder leaves none of its files behind.
    """
    from strata.notebook import output_dirs

    ref = f"{manifest_record.id}@v={manifest_record.version}"
    manifest = source.read_blob(manifest_record.id, manifest_record.version)
    if manifest is None:
        raise ValueError(f"{ref} is a folder whose manifest has no stored bytes")
    landed: dict[tuple[str, int], tuple[str, int]] = {}
    to_send: list[ArtifactVersion] = []
    for file_id, version, sha256 in output_dirs.file_refs(manifest_record.id, manifest):
        record = output_dirs.held_file(source, manifest_record.tenant, file_id, version)
        if record is None or record.state not in ("ready", "superseded"):
            raise ValueError(f"{ref} is a folder naming {file_id}@v={version}, which is gone")
        held = target.find_by_provenance(record.provenance_hash, tenant=record.tenant)
        if held is not None and held.content_sha256 == sha256:
            landed[(file_id, version)] = (held.id, held.version)
            continue
        if not source.blob_exists(file_id, version):
            raise ValueError(
                f"{ref} is a folder naming {file_id}@v={version}, whose bytes are gone"
            )
        to_send.append(record)
    for record in to_send:
        imported = _copy_record(source, target, record)
        if imported.written:
            written.append(imported.ref)
        landed[(record.id, record.version)] = (imported.id, imported.version)
    # Under the id it is sent with: a remap only suffixes that, which keeps the base its
    # files' ids are read from.
    return output_dirs.relocate(manifest, manifest_record.id, landed)


def _ancestors_first(lineage: ArtifactLineageResponse) -> list[LineageNode]:
    """The lineage's nodes, each after every node its edges name as an input.

    BFS order is not enough: a direct input can also be an ancestor of another input, and a
    descendant copied before it keeps an edge naming where the ancestor was, not where it landed.
    """
    parents: dict[str, list[str]] = {node.uri: [] for node in lineage.nodes}
    for edge in lineage.edges:
        if edge.to_uri in parents and edge.from_uri in parents:
            parents[edge.to_uri].append(edge.from_uri)

    ordered: list[LineageNode] = []
    placed: set[str] = set()
    pending = list(reversed(lineage.nodes))
    while pending:
        ready = [node for node in pending if all(p in placed for p in parents[node.uri])]
        # A cycle only comes from hand-edited rows; copy the rest in the old order.
        for node in ready or pending[:1]:
            ordered.append(node)
            placed.add(node.uri)
        pending = [node for node in pending if node.uri not in placed]
    return ordered


# The ``nb_`` prefix keeps it out of the tag lists the registry shows: it is a
# record of how an artifact arrived, not something anyone set on purpose.
PROMOTION_TAG = "nb_promotion"


@dataclass(frozen=True)
class Promotion:
    """What promoting placed in the team's store, and under what name, if any."""

    name: str | None
    ref: str
    copied: int
    alias: str | None = None
    # A protected alias lands in the pending queue instead of moving. The caller has to say so:
    # plain success would leave someone believing the champion moved.
    alias_pending: bool = False
    # The Iceberg table it was also written to, and the snapshot that holds it.
    table: str | None = None
    table_snapshot: int | None = None


def promote_artifact(
    source: ArtifactStore,
    target: RemoteStore,
    artifact: ArtifactVersion,
    *,
    name: str | None,
    alias: str | None = None,
    tags: dict[str, str] | None = None,
    max_depth: int = 10,
    table: str | None = None,
) -> Promotion:
    """Copy an artifact and its chain to the team store, and name it there if ``name`` is given.

    Unlike publishing, promoting mints no public link. The chain travels so each
    ancestor becomes a provenance cache hit for colleagues.

    Raises:
        ValueError: If the artifact's bytes are not readable.
        RuntimeError: If the far side refused a copy or a registry write. The chain
            already copied stays as usable cache entries.
    """
    if artifact.state not in ("ready", "superseded"):
        raise ValueError(
            f"{artifact.id}@v={artifact.version} is not readable (state={artifact.state})"
        )
    if alias and not name:
        # An alias is an alias of a name; there is nothing for it to hang on.
        raise ValueError("An alias needs a name to be an alias of")

    written, landed_ref = copy_chain(source, target, artifact, max_depth)
    landed_id, _, landed_version = landed_ref.partition("@v=")
    version = int(landed_version)

    # Stamp only what this promotion wrote, so a team-cache hit can name the promotion it came from;
    # a row the store already held arrived some other way. Stamp before naming: a retry after a
    # refused name finds every row present and has nothing left to stamp.
    # The stamp's value is the name, so an unnamed copy stamps nothing.
    if name:
        for ref in written:
            written_id, _, written_version = ref.partition("@v=")
            target.set_tag(written_id, int(written_version), PROMOTION_TAG, name)
        target.set_name(name, landed_id, version)
    alias_pending = False
    if alias and name:  # always both: an alias without a name was refused above
        alias_pending = not target.set_alias(name, alias, landed_id, version)
    for key, value in (tags or {}).items():
        # An ``nb_`` stamp records where a row came from. On a row this promotion deduplicated onto,
        # it describes someone else's notebook; overwriting it would move their result onto this
        # cell's strip.
        if key.startswith("nb_") and landed_ref not in written:
            continue
        target.set_tag(landed_id, version, key, value)

    table_snapshot = None
    if table:
        # The team store writes it, with its own catalog: the store is where
        # the bytes and the warehouse credentials both are. A pending alias is
        # not a tag yet; approving it moves the tag then.
        exported = target.export_table(
            landed_id, version, table, alias=alias if alias and not alias_pending else None
        )
        table_snapshot = exported.get("snapshot_id")

    return Promotion(
        name=name,
        ref=landed_ref,
        copied=len(written),
        alias=alias,
        alias_pending=alias_pending,
        table=table,
        table_snapshot=table_snapshot,
    )
